// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore

/// Unit tests for `LedgerSnapshotCodec` and `LedgerSnapshotFingerprint`.
///
/// v1 (issue #136 / ADR 034): the codec reconstructed the ledger bit-for-bit
/// as fully-materialized `[Float]` rows.
///
/// v2 (issue #137): the ledger's inner value is `Indexer.LedgerToken`
/// (`.materialized([Float])` or `.bucketRef(genID:bucketID:pairOffset:)`),
/// so the codec must round-trip both token kinds — and a mixture of both
/// within one chunk — exactly, plus safely reject a v1-format payload
/// (magic/version mismatch) rather than misparse it.
@Suite("Ledger Snapshot Codec")
struct LedgerSnapshotCodecTests {

    // MARK: - Payload round-trip

    @Test("empty ledger round-trips to an empty ledger")
    func emptyLedgerRoundTrip() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [:]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded.isEmpty)
    }

    @Test("single chunk / single materialized token round-trips exactly")
    func singleChunkSingleToken() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            7: [.materialized([1.5, -2.25, 0.0, 3.125])]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded == ledger)
    }

    @Test("multi-chunk / multi-token preserves values and token order")
    func multiChunkMultiToken() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([0.1, 0.2]), .materialized([0.3, 0.4]), .materialized([0.5, 0.6])],
            42: [.materialized([-1.0, -2.0])],
            9001: [.materialized([7.0, 8.0]), .materialized([9.0, 10.0])],
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
        // Token order within a chunk must be preserved exactly.
        #expect(decoded[1] == [.materialized([0.1, 0.2]), .materialized([0.3, 0.4]), .materialized([0.5, 0.6])])
    }

    @Test("special float values (inf, -inf, negative zero, subnormal) survive")
    func specialFloatValues() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([Float.infinity, -Float.infinity, -0.0, Float.leastNonzeroMagnitude])]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        guard case .materialized(let row) = try #require(decoded[1]?.first) else {
            Issue.record("expected a materialized token")
            return
        }
        #expect(row[0] == Float.infinity)
        #expect(row[1] == -Float.infinity)
        #expect(row[2].sign == .minus)   // negative zero preserved
        #expect(row[2] == 0.0)
        #expect(row[3] == Float.leastNonzeroMagnitude)
    }

    @Test("NaN survives round-trip bit-for-bit")
    func nanRoundTrip() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [1: [.materialized([Float.nan, 1.0])]]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        guard case .materialized(let row) = try #require(decoded[1]?.first) else {
            Issue.record("expected a materialized token")
            return
        }
        #expect(row[0].isNaN)
        #expect(row[1] == 1.0)
    }

    // MARK: - bucketRef round-trip (issue #137)

    @Test("single bucketRef token round-trips exactly")
    func singleBucketRefToken() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            7: [.bucketRef(genID: 3, bucketID: 11, pairOffset: 42)]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded == ledger)
    }

    @Test("a chunk with mixed materialized and bucketRef tokens preserves order and kind")
    func mixedMaterializedAndBucketRefTokens() throws {
        // Mirrors the real-world shape (issue #137 spec): a chunk flushed
        // once (now bucket-refs) that later received additional add() calls
        // (still materialized) before the next flush.
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            5: [
                .bucketRef(genID: 1, bucketID: 2, pairOffset: 0),
                .bucketRef(genID: 1, bucketID: 2, pairOffset: 1),
                .materialized([1.0, 2.0]),
                .materialized([3.0, 4.0]),
            ]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
        #expect(decoded[5] == [
            .bucketRef(genID: 1, bucketID: 2, pairOffset: 0),
            .bucketRef(genID: 1, bucketID: 2, pairOffset: 1),
            .materialized([1.0, 2.0]),
            .materialized([3.0, 4.0]),
        ])
    }

    @Test("bucketRef ids spanning negative/large Int64 values survive")
    func bucketRefExtremeValues() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.bucketRef(genID: Int64.max, bucketID: 0, pairOffset: 0)]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
    }

    // MARK: - Corrupt payload / version detection

    @Test("truncated payload throws corruptPayload")
    func truncatedPayloadThrows() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([1.0, 2.0])], 2: [.materialized([3.0, 4.0])]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        // Chop off the last few bytes so the declared counts overrun the buffer.
        let truncated = data.prefix(data.count - 3)
        #expect(throws: LedgerSnapshotCodec.Error.self) {
            _ = try LedgerSnapshotCodec.decode(Data(truncated), dims: 2)
        }
    }

    @Test("empty payload with nonzero declared count throws")
    func emptyPayloadThrows() throws {
        #expect(throws: LedgerSnapshotCodec.Error.self) {
            // A single stray byte can't even hold the 5-byte magic+version header.
            _ = try LedgerSnapshotCodec.decode(Data([0x01]), dims: 2)
        }
    }

    @Test("trailing garbage bytes are rejected")
    func trailingBytesThrows() throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [1: [.materialized([1.0, 2.0])]]
        var data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        data.append(contentsOf: [0xDE, 0xAD])
        #expect(throws: LedgerSnapshotCodec.Error.self) {
            _ = try LedgerSnapshotCodec.decode(data, dims: 2)
        }
    }

    @Test("a v1 (pre-issue-#137) payload is rejected as unsupportedVersion, not misparsed")
    func v1PayloadRejected() throws {
        // Reproduce the exact v1 layout: UInt32 chunkCount, then per chunk
        // Int64 chunkID + UInt32 tokenCount + tokenCount*dims Float32LE, with
        // no magic, no version byte, no per-token tag. v2's decoder must
        // reject this outright (the v1 chunkCount here, 1, does not match the
        // v2 magic 0xFFFF_FFFF) rather than silently misinterpret the bytes.
        var v1 = Data()
        func appendU32(_ v: UInt32) {
            v1.append(UInt8(v & 0xFF)); v1.append(UInt8((v >> 8) & 0xFF))
            v1.append(UInt8((v >> 16) & 0xFF)); v1.append(UInt8((v >> 24) & 0xFF))
        }
        func appendI64(_ v: Int64) {
            let bits = UInt64(bitPattern: v)
            for shift in stride(from: 0, through: 56, by: 8) {
                v1.append(UInt8((bits >> UInt64(shift)) & 0xFF))
            }
        }
        appendU32(1) // chunkCount
        appendI64(7) // chunkID
        appendU32(1) // tokenCount
        appendU32(Float(1.0).bitPattern)
        appendU32(Float(2.0).bitPattern)

        #expect(throws: LedgerSnapshotCodec.Error.unsupportedVersion(0)) {
            _ = try LedgerSnapshotCodec.decode(v1, dims: 2)
        }
    }

    // MARK: - Fingerprint

    @Test("fingerprint derives max/sum fields from generations")
    func fingerprintDerivation() {
        let gens = [
            GenerationRecord(id: 3, level: 0, numEmbeddings: 100, minChunkID: 1, maxChunkID: 40, created: Date(timeIntervalSince1970: 0)),
            GenerationRecord(id: 7, level: 1, numEmbeddings: 250, minChunkID: 1, maxChunkID: 90, created: Date(timeIntervalSince1970: 1)),
        ]
        let fp = LedgerSnapshotFingerprint.compute(chunkCount: 12, generations: gens)
        #expect(fp.chunkCount == 12)
        #expect(fp.maxChunkID == 90)
        #expect(fp.totalEmbeddings == 350)
        #expect(fp.maxGenerationID == 7)
        #expect(fp.generationCount == 2)
    }

    @Test("empty generations yield a zeroed fingerprint")
    func fingerprintEmpty() {
        let fp = LedgerSnapshotFingerprint.compute(chunkCount: 0, generations: [])
        #expect(fp == LedgerSnapshotFingerprint(chunkCount: 0, maxChunkID: 0, totalEmbeddings: 0, maxGenerationID: 0, generationCount: 0))
    }

    @Test("fingerprint of a record matches its stored fields")
    func fingerprintOfRecord() {
        let record = LedgerSnapshotRecord(
            dims: 8, chunkCount: 5, maxChunkID: 55, totalEmbeddings: 500,
            maxGenerationID: 9, generationCount: 4, payload: Data()
        )
        let fp = LedgerSnapshotFingerprint.of(record)
        #expect(fp == LedgerSnapshotFingerprint(chunkCount: 5, maxChunkID: 55, totalEmbeddings: 500, maxGenerationID: 9, generationCount: 4))
    }

    @Test("any differing field makes fingerprints unequal")
    func fingerprintInequality() {
        let base = LedgerSnapshotFingerprint(chunkCount: 1, maxChunkID: 2, totalEmbeddings: 3, maxGenerationID: 4, generationCount: 5)
        #expect(base != LedgerSnapshotFingerprint(chunkCount: 99, maxChunkID: 2, totalEmbeddings: 3, maxGenerationID: 4, generationCount: 5))
        #expect(base != LedgerSnapshotFingerprint(chunkCount: 1, maxChunkID: 99, totalEmbeddings: 3, maxGenerationID: 4, generationCount: 5))
        #expect(base != LedgerSnapshotFingerprint(chunkCount: 1, maxChunkID: 2, totalEmbeddings: 99, maxGenerationID: 4, generationCount: 5))
        #expect(base != LedgerSnapshotFingerprint(chunkCount: 1, maxChunkID: 2, totalEmbeddings: 3, maxGenerationID: 99, generationCount: 5))
        #expect(base != LedgerSnapshotFingerprint(chunkCount: 1, maxChunkID: 2, totalEmbeddings: 3, maxGenerationID: 4, generationCount: 99))
    }
}
