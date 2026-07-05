// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore

/// Unit tests for `LedgerSnapshotCodec` and `LedgerSnapshotFingerprint`
/// (issue #136 / ADR 034). The codec must reconstruct the ledger bit-for-bit
/// (same `[Float]` values, same per-chunk token ordering) or NFCorpus parity
/// would silently regress on the fast path.
@Suite("Ledger Snapshot Codec")
struct LedgerSnapshotCodecTests {

    // MARK: - Payload round-trip

    @Test("empty ledger round-trips to an empty ledger")
    func emptyLedgerRoundTrip() throws {
        let ledger: [Int64: [[Float]]] = [:]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded.isEmpty)
    }

    @Test("single chunk / single token round-trips exactly")
    func singleChunkSingleToken() throws {
        let ledger: [Int64: [[Float]]] = [
            7: [[1.5, -2.25, 0.0, 3.125]]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded == ledger)
    }

    @Test("multi-chunk / multi-token preserves values and token order")
    func multiChunkMultiToken() throws {
        let ledger: [Int64: [[Float]]] = [
            1: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            42: [[-1.0, -2.0]],
            9001: [[7.0, 8.0], [9.0, 10.0]],
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
        // Token order within a chunk must be preserved exactly.
        #expect(decoded[1] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    }

    @Test("special float values (inf, -inf, negative zero, subnormal) survive")
    func specialFloatValues() throws {
        let ledger: [Int64: [[Float]]] = [
            1: [[Float.infinity, -Float.infinity, -0.0, Float.leastNonzeroMagnitude]]
        ]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        let row = try #require(decoded[1]?.first)
        #expect(row[0] == Float.infinity)
        #expect(row[1] == -Float.infinity)
        #expect(row[2].sign == .minus)   // negative zero preserved
        #expect(row[2] == 0.0)
        #expect(row[3] == Float.leastNonzeroMagnitude)
    }

    @Test("NaN survives round-trip bit-for-bit")
    func nanRoundTrip() throws {
        let ledger: [Int64: [[Float]]] = [1: [[Float.nan, 1.0]]]
        let data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        let row = try #require(decoded[1]?.first)
        #expect(row[0].isNaN)
        #expect(row[1] == 1.0)
    }

    // MARK: - Corrupt payload detection

    @Test("truncated payload throws corruptPayload")
    func truncatedPayloadThrows() throws {
        let ledger: [Int64: [[Float]]] = [1: [[1.0, 2.0]], 2: [[3.0, 4.0]]]
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
            // A single stray byte can't even hold the 4-byte chunk count header.
            _ = try LedgerSnapshotCodec.decode(Data([0x01]), dims: 2)
        }
    }

    @Test("trailing garbage bytes are rejected")
    func trailingBytesThrows() throws {
        let ledger: [Int64: [[Float]]] = [1: [[1.0, 2.0]]]
        var data = LedgerSnapshotCodec.encode(ledger, dims: 2)
        data.append(contentsOf: [0xDE, 0xAD])
        #expect(throws: LedgerSnapshotCodec.Error.self) {
            _ = try LedgerSnapshotCodec.decode(data, dims: 2)
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
