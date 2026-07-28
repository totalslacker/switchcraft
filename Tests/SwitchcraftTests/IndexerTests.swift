import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("LSM Indexer")
struct IndexerTests {

    // MARK: - Test helpers

    /// Converts a `ContinuousClock.Duration` to fractional seconds via its
    /// `(seconds, attoseconds)` components (mirrors `PerformanceTests.nanoseconds(_:)`).
    static func seconds(_ duration: ContinuousClock.Duration) -> TimeInterval {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    /// Build a unit-norm random `[Float]` row of length `dims`.
    static func randomRow(dims: Int, rng: inout SplitMix64) -> [Float] {
        var v = [Float](repeating: 0, count: dims)
        for j in 0..<dims {
            let u = Float(rng.next()) / Float(UInt64.max)
            v[j] = u - 0.5
        }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            for j in 0..<dims { v[j] /= norm }
        }
        return v
    }

    /// Add `chunkCount` chunks each carrying `tokensPerChunk` random
    /// unit-norm rows. ChunkIDs are assigned 1, 2, 3, … in order.
    static func populate(
        indexer: Indexer,
        chunkCount: Int,
        tokensPerChunk: Int,
        dims: Int,
        rng: inout SplitMix64
    ) async throws -> [(chunkID: Int64, rows: [[Float]])] {
        var added = [(Int64, [[Float]])]()
        for c in 0..<chunkCount {
            let chunkID = Int64(c + 1)
            var rows = [[Float]]()
            var flat = [Float]()
            flat.reserveCapacity(tokensPerChunk * dims)
            for _ in 0..<tokensPerChunk {
                let row = randomRow(dims: dims, rng: &rng)
                rows.append(row)
                flat.append(contentsOf: row)
            }
            try await indexer.add(chunkID: chunkID, embeddings: flat, dims: dims)
            added.append((chunkID, rows))
        }
        return added
    }

    /// Float32 LE encoder/decoder round-trip used to check `bucket.center`.
    static func decodeCenter(_ data: Data) -> [Float] {
        Indexer.decodeFloat32LE(data)
    }

    static func l2Norm(_ v: [Float]) -> Float {
        sqrt(v.reduce(0) { $0 + $1 * $1 })
    }

    // MARK: - Single-flush invariants (Acceptance #1, #2, #3)

    @Test("flush produces a generation whose chunk-ID range covers all adds")
    func generationCoversAddedChunks() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        var rng = SplitMix64(seed: 1)

        _ = try await Self.populate(
            indexer: indexer, chunkCount: 6, tokensPerChunk: 4, dims: 8, rng: &rng
        )
        try await indexer.flush()

        let gens = try await storage.generations()
        #expect(gens.count == 1)
        let g = gens[0]
        #expect(g.minChunkID == 1)
        #expect(g.maxChunkID == 6)
        #expect(g.numEmbeddings == 24)
    }

    @Test("flush writes exactly k buckets and assigns every token to one bucket")
    func bucketCountAndCoverage() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        var rng = SplitMix64(seed: 2)

        let chunks = try await Self.populate(
            indexer: indexer, chunkCount: 8, tokensPerChunk: 4, dims: 8, rng: &rng
        )
        try await indexer.flush()

        let gens = try await storage.generations()
        let buckets = try await storage.buckets(forGeneration: gens[0].id)

        // Expected k matches Witchcraft's formula on the new generation's m.
        let m = chunks.reduce(0) { $0 + $1.rows.count }
        var kExpected = max(Int((16.0 * Double(m).squareRoot()).rounded()), 1)
        // Training set is sqrt-sample per chunk: tokens_per_chunk=4 → 2 per chunk.
        let trainM = chunks.count * Int(Double(4).squareRoot().rounded(.up))
        if trainM < kExpected {
            kExpected = max(trainM / 4, 1)
        }
        kExpected = min(kExpected, trainM)
        #expect(buckets.count == kExpected)

        // Every (chunkID, tokenOffset) pair appears in exactly one bucket.
        var seen = Set<IndexPair>()
        for bucket in buckets {
            let pairs = try IndicesCodec.decode(bucket.indices)
            for p in pairs {
                #expect(seen.insert(p).inserted, "duplicate pair \(p)")
            }
        }
        let expectedPairs: Set<IndexPair> = Set(chunks.flatMap { c in
            (0..<c.rows.count).map { IndexPair(chunkID: UInt32(c.chunkID), tokenOffset: UInt32($0)) }
        })
        #expect(seen == expectedPairs)
    }

    @Test("decoded residual + center reproduces the original embedding within Q4 MSE bound")
    func residualRoundTripWithinMSEBound() async throws {
        let dims = 16
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        var rng = SplitMix64(seed: 3)

        let chunks = try await Self.populate(
            indexer: indexer, chunkCount: 6, tokensPerChunk: 4, dims: dims, rng: &rng
        )
        try await indexer.flush()

        // Build a (chunkID, tokenOffset) → original-row lookup.
        var originals = [IndexPair: [Float]]()
        for c in chunks {
            for (i, row) in c.rows.enumerated() {
                originals[IndexPair(chunkID: UInt32(c.chunkID), tokenOffset: UInt32(i))] = row
            }
        }

        let gens = try await storage.generations()
        let buckets = try await storage.buckets(forGeneration: gens[0].id)

        // Q4 (4-bit, μ-law companded) round-trip MSE is bounded above by
        // about 0.05 per dim for this codec on residuals from L2-normalised
        // vectors. Use a slack of 0.1 to allow for the test's small dim.
        let perDimToleranceSquared: Float = 0.1
        for bucket in buckets {
            let center = Self.decodeCenter(bucket.center)
            #expect(center.count == dims)

            let pairs = try IndicesCodec.decode(bucket.indices)
            let decoded = Q4Codec.decodeResiduals(bucket.residuals)
            #expect(decoded.count == pairs.count * dims)

            for (i, p) in pairs.enumerated() {
                let original = try #require(originals[p])
                var sse: Float = 0
                for j in 0..<dims {
                    let approx = decoded[i * dims + j] + center[j]
                    let err = approx - original[j]
                    sse += err * err
                }
                let mse = sse / Float(dims)
                #expect(mse < perDimToleranceSquared,
                        "pair \(p) MSE \(mse) exceeded tolerance")
            }
        }
    }

    @Test("indices in each bucket are sorted ascending by (chunkID, tokenOffset)")
    func indicesAreAscending() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        var rng = SplitMix64(seed: 4)

        _ = try await Self.populate(
            indexer: indexer, chunkCount: 8, tokensPerChunk: 4, dims: 8, rng: &rng
        )
        try await indexer.flush()
        let gens = try await storage.generations()
        let buckets = try await storage.buckets(forGeneration: gens[0].id)

        for bucket in buckets {
            let pairs = try IndicesCodec.decode(bucket.indices)
            var prev: IndexPair? = nil
            for p in pairs {
                if let q = prev {
                    let strictlyAfter = (p.chunkID > q.chunkID)
                        || (p.chunkID == q.chunkID && p.tokenOffset > q.tokenOffset)
                    #expect(strictlyAfter, "pairs not ascending: \(q) then \(p)")
                }
                prev = p
            }
        }
    }

    @Test("centers are L2-normalised Float32 vectors of length dims")
    func centersAreL2Normalised() async throws {
        let dims = 16
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 4))
        var rng = SplitMix64(seed: 5)

        _ = try await Self.populate(
            indexer: indexer, chunkCount: 6, tokensPerChunk: 4, dims: dims, rng: &rng
        )
        try await indexer.flush()
        let gens = try await storage.generations()
        let buckets = try await storage.buckets(forGeneration: gens[0].id)

        for bucket in buckets {
            #expect(bucket.center.count == dims * 4) // Float32 LE
            let center = Self.decodeCenter(bucket.center)
            #expect(center.count == dims)
            let n = Self.l2Norm(center)
            #expect(abs(n - 1.0) < 1e-4, "center norm \(n) not ≈ 1")
        }
    }

    // MARK: - Cascade (Acceptance #4)

    @Test("cascade collapses lower levels into the next when capacity is exceeded")
    func cascadeCollapsesLowerLevels() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        // testing config: l0Cap=4, fanout=2 → cap(0) = 8, cap(1) = 16, cap(2) = 32.
        let config = IndexerConfig.testing(l0Capacity: 4, lsmFanout: 2)
        let indexer = try await Indexer(storage: storage, config: config)
        var rng = SplitMix64(seed: 6)

        // Each flush adds 4 single-token chunks. Track expected level
        // after each flush by replaying the cascade walk on paper:
        //   flush 1: pending=4, total=4, cap(0)=8 → L0 (size 4)
        //   flush 2: pending=4, total=8, cap(0)=8 → L0 (size 8)
        //   flush 3: pending=4, total=12 > 8, cap(1)=16 → L1 (size 12)
        //   flush 4: pending=4, total=4+0=4 → L0 (size 4); L1 untouched
        //   flush 5: pending=4, total=8 → L0 (size 8)
        //   flush 6: pending=4, total=12 → L0=12 > 8 → L1: 12+12=24 > 16 → L2 (size 24)
        let dims = 8
        var nextChunkID: Int64 = 1
        for flushIdx in 1...6 {
            for _ in 0..<4 {
                let row = Self.randomRow(dims: dims, rng: &rng)
                try await indexer.add(chunkID: nextChunkID, embeddings: row, dims: dims)
                nextChunkID += 1
            }
            try await indexer.flush()

            let gens = try await storage.generations()
            switch flushIdx {
            case 1:
                #expect(gens.count == 1)
                #expect(gens.first?.level == 0)
                #expect(gens.first?.numEmbeddings == 4)
            case 2:
                #expect(gens.count == 1)
                #expect(gens.first?.level == 0)
                #expect(gens.first?.numEmbeddings == 8)
            case 3:
                #expect(gens.count == 1)
                #expect(gens.first?.level == 1)
                #expect(gens.first?.numEmbeddings == 12)
            case 4:
                #expect(gens.count == 2)
                let l0 = gens.first { $0.level == 0 }
                let l1 = gens.first { $0.level == 1 }
                #expect(l0?.numEmbeddings == 4)
                #expect(l1?.numEmbeddings == 12)
            case 5:
                #expect(gens.count == 2)
                let l0 = gens.first { $0.level == 0 }
                let l1 = gens.first { $0.level == 1 }
                #expect(l0?.numEmbeddings == 8)
                #expect(l1?.numEmbeddings == 12)
            case 6:
                // L0 and L1 both deleted; one new L2 with all 24 embeddings.
                #expect(gens.count == 1)
                #expect(gens.first?.level == 2)
                #expect(gens.first?.numEmbeddings == 24)
            default:
                Issue.record("unhandled flush index")
            }
        }
    }

    // MARK: - Determinism (Acceptance #5)

    @Test("two builds with the same input + seed produce byte-identical bucket blobs")
    func deterministicBuilds() async throws {
        let dims = 16
        let chunks = 12
        let tokensPer = 4
        let config = IndexerConfig.testing(l0Capacity: 8, lsmFanout: 2, seed: 12345)

        // Generate a fixed input set independent of the indexers' RNG.
        var setupRng = SplitMix64(seed: 999)
        var inputs = [(Int64, [Float])]()
        for c in 0..<chunks {
            var flat = [Float]()
            for _ in 0..<tokensPer {
                flat.append(contentsOf: Self.randomRow(dims: dims, rng: &setupRng))
            }
            inputs.append((Int64(c + 1), flat))
        }

        func buildBucketBlobs() async throws -> [(Data, Data, Data)] {
            let storage = InMemoryStorage()
            try await storage.open()
            let indexer = try await Indexer(storage: storage, config: config)
            for (chunkID, flat) in inputs {
                try await indexer.add(chunkID: chunkID, embeddings: flat, dims: dims)
            }
            try await indexer.flush()

            let gens = try await storage.generations()
            #expect(gens.count == 1)
            let buckets = try await storage.buckets(forGeneration: gens[0].id)
            return buckets.map { ($0.center, $0.indices, $0.residuals) }
        }

        let a = try await buildBucketBlobs()
        let b = try await buildBucketBlobs()
        #expect(a.count == b.count)
        for (i, pair) in zip(a, b).enumerated() {
            #expect(pair.0.0 == pair.1.0, "bucket \(i) center differs")
            #expect(pair.0.1 == pair.1.1, "bucket \(i) indices differ")
            #expect(pair.0.2 == pair.1.2, "bucket \(i) residuals differ")
        }
    }

    // MARK: - clearIndex (Acceptance #6)

    @Test("clearIndex empties generations and buckets but leaves documents and chunks")
    func clearIndexPreservesDocsAndChunks() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing(l0Capacity: 4, lsmFanout: 2))
        var rng = SplitMix64(seed: 7)

        // Insert some documents & chunks directly so we can verify they
        // remain after clearIndex.
        try await storage.upsertDocument(DocumentRecord(
            uuid: "doc-1", date: Date(), hash: "h1", body: "hello world"
        ))
        let chunk = try await storage.upsertChunk(ChunkRecord(
            hash: "h1", model: "m", embeddings: Data()
        ))
        #expect(chunk.id > 0)

        // Build an index over some random embeddings.
        _ = try await Self.populate(
            indexer: indexer, chunkCount: 4, tokensPerChunk: 4, dims: 8, rng: &rng
        )
        try await indexer.flush()
        #expect(try await storage.generations().isEmpty == false)

        try await indexer.clearIndex()

        #expect(try await storage.generations().isEmpty)
        #expect(try await storage.documentCount() == 1)
        #expect(try await storage.chunkCount() == 1)

        // After clearIndex, the indexer can rebuild over fresh inputs.
        let dims2 = 8
        let row = Self.randomRow(dims: dims2, rng: &rng)
        try await indexer.add(chunkID: 1, embeddings: row, dims: dims2)
        try await indexer.flush()
        #expect(try await storage.generations().count == 1)
    }

    // MARK: - Empty flush (Acceptance #7)

    @Test("flush with zero buffered embeddings produces no new generation")
    func emptyFlushIsNoop() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .testing())

        try await indexer.flush()
        #expect(try await storage.generations().isEmpty)

        // A second flush should also be a no-op.
        try await indexer.flush()
        #expect(try await storage.generations().isEmpty)
    }

    // MARK: - Performance smoke (Acceptance #9)

    @Test("indexing 5000 single-token 128-dim chunks completes within 5s",
          .timeLimit(.minutes(1)))
    func performanceSmoke() async throws {
        let dims = 128
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(storage: storage, config: .production)
        var rng = SplitMix64(seed: 8)

        // Pre-generate inputs so the timing measures only the indexer.
        var rows = [[Float]]()
        rows.reserveCapacity(5000)
        for _ in 0..<5000 {
            rows.append(Self.randomRow(dims: dims, rng: &rng))
        }

        let clock = ContinuousClock()
        let start = clock.now
        for (i, row) in rows.enumerated() {
            try await indexer.add(chunkID: Int64(i + 1), embeddings: row, dims: dims)
        }
        try await indexer.flush()
        let elapsed = Self.seconds(clock.now - start)

        let gens = try await storage.generations()
        #expect(gens.count == 1)
        // Acceptance criterion: < 5s (release). Debug limit is 10s.
        //
        // Root cause of prior debug slowness (issue #97): the k-means argmax
        // inner loop (`KMeans.assignAll`) ran 5000×1131=5.655M bounds-checked
        // iterations per call (7 calls per flush = 33.93M total). In debug
        // mode Swift disables vectorisation → ~5–9s elapsed. It is now replaced
        // by `vDSP_maxvi` (Accelerate), which is always SIMD-compiled and
        // executes the same argmax in ~15ms regardless of build config.
        //
        // Measurement evidence (#97, 11 runs each, M3 Max, sequential debug mode):
        //   Pre-fix (plain Swift argmax): median=6.515s, p95=9.611s — nearly
        //     no headroom against the 10s limit; 36.5s spike observed under load.
        //   Post-fix (vDSP_maxvi):        wall-clock median=3.688s (upper bound
        //     on elapsed); all 11 runs passed elapsed<10s. The argmax step is
        //     now ~15ms; remaining elapsed is dominated by 5000 async add() hops.
        //
        // Re-measured for issue #146 (11 consecutive *full-suite* debug runs —
        // isolated `--filter` runs can't reproduce Swift Testing's default
        // intra-process parallel scheduling, which is the actual contention
        // source, not CI-vs-local variance): see ADR 039 for full data. The
        // 10s/5s limits were left unchanged — the #97 baseline held with real
        // headroom; the prior "ADR 012, ~1.35× CI-runner factor" comment above
        // this one cited a discarded debug-mode calibration figure computed for
        // a *different*, release-mode-only suite (search latency, not an async
        // add-loop) and did not actually apply here — removed as inaccurate.
        #if DEBUG
        let limit: TimeInterval = 10.0
        #else
        let limit: TimeInterval = 5.0
        #endif
        print("[PerfSmoke] elapsed=\(String(format: "%.3f", elapsed))s limit=\(limit)s")
        #expect(elapsed < limit, "indexing took \(elapsed)s — expected < \(limit)s")
    }
}
