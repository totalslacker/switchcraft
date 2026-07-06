// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore

/// Tests for the ledger-snapshot startup fast path (issue #136 / ADR 034).
///
/// Covers the four scenarios from the acceptance criteria:
///  - clean-shutdown reproducer (fast path taken, `recoveredConflictCount == 0`)
///  - crash-safety fallback (no/absent snapshot → full rehydration walk)
///  - fingerprint-mismatch fallback (storage mutated out-of-band → detected)
///  - corrupt-payload fallback (fingerprint matches, payload undecodable)
@Suite("Indexer Snapshot Fast Path")
struct IndexerSnapshotFastPathTests {

    static let dims = 4

    /// Deterministic unit-norm row seeded by `(chunk, token)` so two runs
    /// produce identical embeddings.
    static func row(chunk: Int64, token: Int) -> [Float] {
        var v = [Float](repeating: 0, count: dims)
        var s = UInt64(chunk) &* 2_654_435_761 &+ UInt64(token) &* 40_503
        for j in 0..<dims {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            v[j] = Float(s >> 33) / Float(UInt32.max) - 0.5
        }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if norm > 0 { for j in 0..<dims { v[j] /= norm } }
        return v
    }

    /// Add `chunkCount` chunks (2 tokens each) and flush. Returns the storage
    /// and the indexer's post-flush ledger contents for equality checks.
    static func populateAndFlush(
        chunkCount: Int,
        config: IndexerConfig = .testing()
    ) async throws -> (storage: InMemoryStorage, ledger: [Int64: [[Float]]]) {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: config)
        for c in 1...chunkCount {
            let chunkID = Int64(c)
            var flat = [Float]()
            for t in 0..<2 { flat.append(contentsOf: row(chunk: chunkID, token: t)) }
            try await indexer.add(chunkID: chunkID, embeddings: flat, dims: dims)
        }
        _ = try await indexer.flush()
        let ledger = try await indexer.ledgerContents()
        return (storage, ledger)
    }

    // MARK: - 1. Clean-shutdown reproducer

    @Test("clean shutdown: second init takes the snapshot fast path with recoveredConflictCount == 0")
    func cleanShutdownFastPath() async throws {
        let (storage, originalLedger) = try await Self.populateAndFlush(chunkCount: 6)

        // A snapshot must have been written by flush().
        #expect(try await storage.loadLedgerSnapshot() != nil)

        // Re-init over the same storage (simulating a fresh process).
        let reopened = try await Indexer(storage: storage, config: .testing())

        #expect(await reopened.didUseSnapshotFastPath == true)
        #expect(await reopened.recoveredConflictCount == 0)

        // The fast path reconstructs the exact in-memory ledger. Both sides of
        // this comparison already resolve through the same Q4-lossy path
        // (issue #137 / ADR 035): `originalLedger` was captured via
        // `ledgerContents()` right after flush, when this corpus's tokens are
        // already `.bucketRef` (materialized on demand by `ledgerContents()`
        // itself), and `reloaded` resolves the snapshot's own bucket-refs the
        // same way — so this assertion is "same bucket data resolves
        // identically," not "snapshot floats are more precise than a Q4 walk"
        // (ADR 034's original claim, corrected by ADR 035 §6).
        let reloaded = try await reopened.ledgerContents()
        #expect(reloaded == originalLedger)

        // Invalidate-on-load: the on-disk snapshot must be cleared after the load.
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }

    @Test("invalidate-on-load: a second re-init after the fast path falls back (no snapshot left)")
    func invalidateOnLoadForcesFallbackNextTime() async throws {
        let (storage, _) = try await Self.populateAndFlush(chunkCount: 5)

        // First re-init consumes + clears the snapshot via the fast path.
        let first = try await Indexer(storage: storage, config: .testing())
        #expect(await first.didUseSnapshotFastPath == true)

        // Second re-init finds no snapshot (simulating a crash before the next
        // flush/shutdown could rewrite it) and must fall back to the full walk.
        let second = try await Indexer(storage: storage, config: .testing())
        #expect(await second.didUseSnapshotFastPath == false)
        // Full-walk rehydration still reconstructs every chunk.
        let ledger = try await second.ledgerContents()
        #expect(ledger.count == 5)
    }

    // MARK: - 2. Crash-safety fallback (no snapshot)

    @Test("crash safety: data present but no snapshot → full rehydration walk")
    func noSnapshotFallsBackToFullWalk() async throws {
        let (storage, _) = try await Self.populateAndFlush(chunkCount: 6)

        // Simulate a crash: the snapshot was never written / was lost.
        try await storage.clearLedgerSnapshot()

        let reopened = try await Indexer(storage: storage, config: .testing())
        #expect(await reopened.didUseSnapshotFastPath == false)
        let ledger = try await reopened.ledgerContents()
        #expect(ledger.count == 6)
    }

    @Test("crash safety: unclean shutdown with a 2-gen conflict recovers via full walk")
    func conflictWithoutSnapshotRecovers() async throws {
        // Hand-build a 2-generation conflict (same chunkID in two active gens)
        // with NO snapshot — exactly the crash-mid-compaction case ADR 024
        // guards. The fast path must be skipped and the conflict recovered.
        let storage = InMemoryStorage()
        try await storage.open()
        let sharedChunkID: Int64 = 10
        let now = Date()
        let zero = [Float](repeating: 0, count: Self.dims)

        let gen1 = try await storage.insertGeneration(GenerationRecord(
            level: 2, numEmbeddings: 1, minChunkID: sharedChunkID, maxChunkID: sharedChunkID, created: now
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen1.id,
            center: Indexer.encodeFloat32LE([1, 0, 0, 0]),
            indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zero)
        ))
        let gen2 = try await storage.insertGeneration(GenerationRecord(
            level: 0, numEmbeddings: 1, minChunkID: sharedChunkID, maxChunkID: sharedChunkID, created: now
        ))
        _ = try await storage.insertBucket(BucketRecord(
            generationID: gen2.id,
            center: Indexer.encodeFloat32LE([0, 1, 0, 0]),
            indices: IndicesCodec.encode([IndexPair(chunkID: UInt32(sharedChunkID), tokenOffset: 0)]),
            residuals: Q4Codec.encodeResiduals(zero)
        ))

        #expect(try await storage.loadLedgerSnapshot() == nil)

        let indexer = try await Indexer(
            storage: storage,
            config: .testing(rehydrationConflictBehavior: .autoRecover)
        )
        #expect(await indexer.didUseSnapshotFastPath == false)
        #expect(await indexer.recoveredConflictCount == 1)
    }

    // MARK: - 3. Fingerprint-mismatch fallback

    @Test("fingerprint mismatch: storage mutated out-of-band → fast path skipped")
    func fingerprintMismatchFallsBack() async throws {
        let (storage, _) = try await Self.populateAndFlush(chunkCount: 6)
        #expect(try await storage.loadLedgerSnapshot() != nil)

        // Mutate storage via a path that bypasses the ledger: inserting a chunk
        // changes chunkCount, which is part of the fingerprint (mirrors the
        // real out-of-band mutation vacuum() performs). The snapshot is now
        // stale relative to storage.
        _ = try await storage.upsertChunk(
            ChunkRecord(hash: "out-of-band", model: "m", embeddings: Data(), counts: [1])
        )

        let reopened = try await Indexer(storage: storage, config: .testing())
        #expect(await reopened.didUseSnapshotFastPath == false)
        // Fell back to full rehydration, which still reconstructs every chunk.
        let ledger = try await reopened.ledgerContents()
        #expect(ledger.count == 6)
        // Stale snapshot must have been cleared during the mismatch handling.
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }

    // MARK: - 4. Corrupt-payload fallback

    @Test("corrupt payload: fingerprint matches but payload is undecodable → fast path skipped")
    func corruptPayloadFallsBack() async throws {
        let (storage, _) = try await Self.populateAndFlush(chunkCount: 6)

        // Overwrite the snapshot with a record whose fingerprint matches current
        // storage but whose payload is truncated/garbage.
        let gens = try await storage.generations()
        let chunkCount = try await storage.chunkCount()
        let fp = LedgerSnapshotFingerprint.compute(chunkCount: chunkCount, generations: gens)
        let corrupt = LedgerSnapshotRecord(
            dims: Self.dims,
            chunkCount: fp.chunkCount,
            maxChunkID: fp.maxChunkID,
            totalEmbeddings: fp.totalEmbeddings,
            maxGenerationID: fp.maxGenerationID,
            generationCount: fp.generationCount,
            // Claims a large chunk count in its 4-byte header but has no body.
            payload: Data([0xFF, 0xFF, 0xFF, 0x7F])
        )
        try await storage.saveLedgerSnapshot(corrupt)

        // init must not throw — the decode failure is swallowed and the walk runs.
        let reopened = try await Indexer(storage: storage, config: .testing())
        #expect(await reopened.didUseSnapshotFastPath == false)
        let ledger = try await reopened.ledgerContents()
        #expect(ledger.count == 6)
        // The corrupt snapshot must have been cleared on the failed load attempt.
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }

    @Test("v1 (pre-issue-#137) snapshot payload falls back to full rehydration, not misparsed")
    func v1FormatSnapshotFallsBack() async throws {
        let (storage, originalLedger) = try await Self.populateAndFlush(chunkCount: 6)

        // Reconstruct the exact v1 payload layout (issue #136): no magic, no
        // version byte, no per-token tag — just chunkCount, then per chunk
        // chunkID + tokenCount + tokenCount*dims Float32LE, straight from the
        // ledger captured right after flush.
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
        let sortedChunkIDs = originalLedger.keys.sorted()
        appendU32(UInt32(sortedChunkIDs.count))
        for chunkID in sortedChunkIDs {
            let rows = originalLedger[chunkID]!
            appendI64(chunkID)
            appendU32(UInt32(rows.count))
            for row in rows {
                for v in row { appendU32(v.bitPattern) }
            }
        }

        let gens = try await storage.generations()
        let chunkCount = try await storage.chunkCount()
        let fp = LedgerSnapshotFingerprint.compute(chunkCount: chunkCount, generations: gens)
        let v1Record = LedgerSnapshotRecord(
            dims: Self.dims,
            chunkCount: fp.chunkCount,
            maxChunkID: fp.maxChunkID,
            totalEmbeddings: fp.totalEmbeddings,
            maxGenerationID: fp.maxGenerationID,
            generationCount: fp.generationCount,
            payload: v1
        )
        try await storage.saveLedgerSnapshot(v1Record)

        // init must not throw — the version-mismatch decode failure is
        // swallowed (same as any other corrupt/unusable snapshot) and the
        // full rehydration walk runs instead, producing the correct ledger.
        let reopened = try await Indexer(storage: storage, config: .testing())
        #expect(await reopened.didUseSnapshotFastPath == false)
        let ledger = try await reopened.ledgerContents()
        #expect(ledger == originalLedger)
        // The unusable v1 snapshot must have been cleared on the failed load.
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }

    // MARK: - Write-time consistency guard

    @Test("write guard: ledger inconsistent with storage → no snapshot persisted at persistSnapshot")
    func inconsistentLedgerSkipsSnapshotWrite() async throws {
        let (storage, _) = try await Self.populateAndFlush(chunkCount: 6)
        // A valid snapshot exists after the flush.
        #expect(try await storage.loadLedgerSnapshot() != nil)

        // Reopen so the indexer holds the full (consistent) ledger.
        let indexer = try await Indexer(storage: storage, config: .testing())
        #expect(await indexer.didUseSnapshotFastPath == true)

        // Mutate storage out-of-band so the committed embedding count no longer
        // matches the in-memory ledger (simulates a crash-left partial state
        // rewritten directly behind the indexer's back).
        let gens = try await storage.generations()
        for g in gens {
            try await storage.updateGenerationEmbeddingCount(id: g.id, count: 0)
        }

        // persistSnapshot() must detect the divergence and refuse to write,
        // clearing any prior snapshot instead.
        try await indexer.persistSnapshot()
        #expect(try await storage.loadLedgerSnapshot() == nil,
                "an inconsistent ledger must not be persisted as a snapshot")
    }

    // MARK: - Empty index

    @Test("empty index: no generations → no fast path, no snapshot written")
    func emptyIndexNoSnapshot() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing())
        // No adds, no flush — an empty flush is a no-op and writes no snapshot.
        _ = try await indexer.flush()
        #expect(await indexer.didUseSnapshotFastPath == false)
        #expect(try await storage.loadLedgerSnapshot() == nil)
    }
}

/// `true` when compiled with optimisations (`swift test -c release`); mirrors
/// `PerformanceTests.isReleaseBuild`. The wall-clock comparison below is too
/// noisy to defend under a debug build.
private let isReleaseBuild: Bool = {
    var debug = false
    assert({ debug = true; return true }())
    return !debug
}()

/// Release-gated wall-clock evidence that the snapshot fast path is
/// measurably faster than the full rehydration walk (issue #136 acceptance
/// criterion 7). The correctness of "which path was taken" is asserted by the
/// boolean-flag tests above; this suite only documents the speedup and asserts
/// the loose, non-flaky invariant that the fast path is not slower.
@Suite("Indexer Snapshot Fast Path Speedup", .serialized,
       .enabled(if: isReleaseBuild,
                "Snapshot speedup is release-only (run with `swift test -c release`)"))
struct IndexerSnapshotSpeedupTests {

    @Test("fast-path init is measurably faster than the full rehydration walk")
    func fastPathBeatsFullWalk() async throws {
        let dims = 64
        let chunkCount = 3_000
        let tokensPerChunk = 4
        let trials = 7

        // Build a corpus large enough that the full walk does real work.
        // Note (issue #137 / ADR 035): the full walk itself is now a cheap
        // O(pairs) integrity walk (no per-token Q4 dequant — that eager
        // reconstruction was removed), so its absolute cost — and therefore
        // its margin versus the snapshot fast path, which is also O(pairs) —
        // is much smaller than when this test was written for ADR 034. A
        // single-shot wall-clock comparison is thin enough now to be flipped
        // by scheduler contention when this suite runs alongside the rest of
        // the package's tests, so this measures min-of-`trials` for each
        // phase instead of one sample: scheduler/GC jitter only ever adds
        // delay, never subtracts it, so the minimum across repeated trials
        // is the most contention-resistant estimate of the true cost.
        let storage = InMemoryStorage()
        try await storage.open()
        let builder = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        for c in 1...chunkCount {
            var flat = [Float](repeating: 0, count: tokensPerChunk * dims)
            var s = UInt64(c) &* 2_654_435_761 &+ 17
            for i in 0..<flat.count {
                s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                flat[i] = Float(s >> 40) / Float(1 << 24) - 0.5
            }
            try await builder.add(chunkID: Int64(c), embeddings: flat, dims: dims)
        }
        _ = try await builder.flush()
        // A valid snapshot now exists.
        #expect(try await storage.loadLedgerSnapshot() != nil)

        var fastSamples: [TimeInterval] = []
        var fullSamples: [TimeInterval] = []
        for _ in 0..<trials {
            // `persistSnapshot()` re-encodes `builder`'s (unchanged) ledger
            // and re-arms the on-disk snapshot, since the previous trial's
            // fast-path load consumed + cleared it (invalidate-on-load).
            try await builder.persistSnapshot()
            #expect(try await storage.loadLedgerSnapshot() != nil)

            let fastStart = Date()
            let fast = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
            fastSamples.append(Date().timeIntervalSince(fastStart))
            #expect(await fast.didUseSnapshotFastPath == true)

            // Snapshot now absent after invalidate-on-load — this init takes
            // the full walk.
            #expect(try await storage.loadLedgerSnapshot() == nil)
            let fullStart = Date()
            let full = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
            fullSamples.append(Date().timeIntervalSince(fullStart))
            #expect(await full.didUseSnapshotFastPath == false)
        }

        let fastElapsed = fastSamples.min()!
        let fullElapsed = fullSamples.min()!
        let ratio = fullElapsed / max(fastElapsed, 1e-9)
        print("[PerfTest] snapshot fast-path(min of \(trials))=\(String(format: "%.4f", fastElapsed))s full-walk(min of \(trials))=\(String(format: "%.4f", fullElapsed))s speedup=\(String(format: "%.2f", ratio))×")

        // Loose, non-flaky invariant: the fast path must not be slower than the
        // full walk. The `print` above documents the actual speedup.
        #expect(fastElapsed <= fullElapsed,
                "snapshot fast path (\(fastElapsed)s) should not be slower than the full walk (\(fullElapsed)s)")
    }
}
