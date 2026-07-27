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
    func emptyLedgerRoundTrip() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [:]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded.isEmpty)
    }

    @Test("single chunk / single materialized token round-trips exactly")
    func singleChunkSingleToken() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            7: [.materialized([1.5, -2.25, 0.0, 3.125])]
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded == ledger)
    }

    @Test("multi-chunk / multi-token preserves values and token order")
    func multiChunkMultiToken() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([0.1, 0.2]), .materialized([0.3, 0.4]), .materialized([0.5, 0.6])],
            42: [.materialized([-1.0, -2.0])],
            9001: [.materialized([7.0, 8.0]), .materialized([9.0, 10.0])],
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
        // Token order within a chunk must be preserved exactly.
        #expect(decoded[1] == [.materialized([0.1, 0.2]), .materialized([0.3, 0.4]), .materialized([0.5, 0.6])])
    }

    @Test("special float values (inf, -inf, negative zero, subnormal) survive")
    func specialFloatValues() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([Float.infinity, -Float.infinity, -0.0, Float.leastNonzeroMagnitude])]
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 4)
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
    func nanRoundTrip() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [1: [.materialized([Float.nan, 1.0])]]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
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
    func singleBucketRefToken() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            7: [.bucketRef(genID: 3, bucketID: 11, pairOffset: 42)]
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 4)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 4)
        #expect(decoded == ledger)
    }

    @Test("a chunk with mixed materialized and bucketRef tokens preserves order and kind")
    func mixedMaterializedAndBucketRefTokens() async throws {
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
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
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
    func bucketRefExtremeValues() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.bucketRef(genID: Int64.max, bucketID: 0, pairOffset: 0)]
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
        let decoded = try LedgerSnapshotCodec.decode(data, dims: 2)
        #expect(decoded == ledger)
    }

    // MARK: - Corrupt payload / version detection

    @Test("truncated payload throws corruptPayload")
    func truncatedPayloadThrows() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            1: [.materialized([1.0, 2.0])], 2: [.materialized([3.0, 4.0])]
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
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
    func trailingBytesThrows() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [1: [.materialized([1.0, 2.0])]]
        var data = await LedgerSnapshotCodec.encode(ledger, dims: 2)
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

    // MARK: - Wire-format regression (issue #144, REQ-3)

    /// Byte-for-byte comparison against a hand-built reference encoding of
    /// the documented v2 wire format (magic/version/chunkCount, then per
    /// chunk: chunkID/tokenCount, then per token: tag + payload). Independent
    /// of `LedgerSnapshotCodec`'s own implementation, so it catches wire-format
    /// drift from the REQ-3 per-byte-`Data.append` → bulk-`[UInt8]`-buffer
    /// rewrite (an indexing bug in the rewrite could silently write the right
    /// bytes in the wrong place within an otherwise well-formed frame, which a
    /// round-trip-only test would not catch since `decode` would still parse
    /// it "successfully").
    @Test("encode() output matches the documented wire format byte-for-byte")
    func encodeMatchesDocumentedWireFormat() async throws {
        let ledger: [Int64: [Indexer.LedgerToken]] = [
            7: [
                .materialized([1.5, -2.25]),
                .bucketRef(genID: 3, bucketID: 11, pairOffset: 42),
            ],
        ]
        let data = await LedgerSnapshotCodec.encode(ledger, dims: 2)

        var expected = Data()
        func u32(_ v: UInt32) {
            expected.append(UInt8(v & 0xFF)); expected.append(UInt8((v >> 8) & 0xFF))
            expected.append(UInt8((v >> 16) & 0xFF)); expected.append(UInt8((v >> 24) & 0xFF))
        }
        func i64(_ v: Int64) {
            let bits = UInt64(bitPattern: v)
            for shift in stride(from: 0, through: 56, by: 8) {
                expected.append(UInt8((bits >> UInt64(shift)) & 0xFF))
            }
        }
        func f32(_ v: Float) { u32(v.bitPattern) }

        u32(0xFFFF_FFFF)           // magic
        expected.append(2)         // version
        u32(1)                     // chunkCount
        i64(7)                     // chunkID
        u32(2)                     // tokenCount
        expected.append(0)         // tag: materialized
        f32(1.5)
        f32(-2.25)
        expected.append(1)         // tag: bucketRef
        i64(3)                     // genID
        i64(11)                    // bucketID
        i64(42)                    // pairOffset

        #expect(data == expected)
    }

    // MARK: - Progress callback + yield points (issue #144, REQ-4/REQ-5)

    @Test("onProgress fires at the yield-interval cadence plus a final checkpoint")
    func onProgressCadence() async throws {
        // Not a multiple of the encoder's internal yield interval (512), so
        // this also exercises the "final checkpoint with a partial last
        // stretch" path.
        let chunkCount = 1_100
        var ledger: [Int64: [Indexer.LedgerToken]] = [:]
        ledger.reserveCapacity(chunkCount)
        for c in 1...chunkCount {
            ledger[Int64(c)] = [.materialized([Float(c)])]
        }

        final actor ProgressRecorder {
            private(set) var updates: [SnapshotProgress] = []
            func record(_ p: SnapshotProgress) { updates.append(p) }
        }
        let recorder = ProgressRecorder()

        _ = await LedgerSnapshotCodec.encode(ledger, dims: 1) { progress in
            await recorder.record(progress)
        }

        let updates = await recorder.updates
        // Checkpoints at 512, 1024, then a final one at 1100 (the partial
        // last stretch) — three total.
        #expect(updates.map(\.chunksEncoded) == [512, 1024, 1100])
        for update in updates {
            #expect(update.totalChunks == chunkCount)
        }
        #expect(updates.last?.bytesEncoded == data(forChunkCount: chunkCount))
    }

    @Test("empty ledger never invokes onProgress")
    func onProgressNotCalledForEmptyLedger() async throws {
        actor Flag {
            private(set) var called = false
            func set() { called = true }
        }
        let flag = Flag()
        _ = await LedgerSnapshotCodec.encode([:], dims: 4) { _ in await flag.set() }
        #expect(await flag.called == false)
    }

    /// A concurrent task only makes progress between suspension points. If
    /// `encode()`'s per-chunk loop never yielded (the REQ-4 regression this
    /// guards against), a large encode would run to completion as one
    /// uninterrupted stretch and the ticker would show no progress until
    /// after encode had already finished. Sampling the ticker from inside
    /// `onProgress` (itself only invoked mid-encode, at yield checkpoints)
    /// gives a deterministic way to observe interleaving without racing on
    /// wall-clock timing.
    @Test("encode() yields periodically, letting a concurrent task advance before it finishes")
    func encodeYieldsForConcurrentWork() async throws {
        let chunkCount = 5_000
        var ledger: [Int64: [Indexer.LedgerToken]] = [:]
        ledger.reserveCapacity(chunkCount)
        for c in 1...chunkCount {
            ledger[Int64(c)] = [.materialized([Float](repeating: 0.5, count: 32))]
        }

        actor Ticker {
            private(set) var count = 0
            func increment() { count += 1 }
        }
        let ticker = Ticker()
        let tickTask = Task {
            while !Task.isCancelled {
                await ticker.increment()
                await Task.yield()
            }
        }
        defer { tickTask.cancel() }

        actor Flag {
            private(set) var observedInterleaving = false
            func set() { observedInterleaving = true }
        }
        let flag = Flag()

        let startCount = await ticker.count
        _ = await LedgerSnapshotCodec.encode(ledger, dims: 32) { progress in
            guard progress.chunksEncoded < progress.totalChunks else { return }
            if await ticker.count > startCount {
                await flag.set()
            }
        }

        #expect(await flag.observedInterleaving, "a concurrent task should be able to advance while a large encode() is still in progress")
    }

    private func data(forChunkCount chunkCount: Int) -> Int {
        // header(9) + per-chunk(12 + tag(1) + 1 float(4)) for a dims=1
        // single-materialized-token-per-chunk ledger, matching onProgressCadence's fixture.
        9 + chunkCount * (12 + 1 + 4)
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
