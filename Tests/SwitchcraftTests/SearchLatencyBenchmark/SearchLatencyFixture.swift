// SPDX-License-Identifier: Apache-2.0
//
// Realistic-scale search-latency benchmark fixture (issue #140 / ADR 035).
//
// Builds a multi-generation LSM tree directly via `storage.insertGeneration`
// / `storage.insertBucket` — bypassing `Indexer`'s real k-means clustering
// entirely — so the fixture is cheap to construct while still exercising
// the real per-dimension arithmetic (residual decode, dot products) the
// perf profile depends on. See `docs/Plan.md` and ADR 035 "Fixture
// construction strategy" for why this bypass is the right call: real
// k-means at the issue's target scale (~53K clusters for the 11.1M-token
// generation) was flagged in Research as untested and possibly
// prohibitively slow to build even once.
//
// ## Scale: reduced from the issue's ~15,700 chunks / ~11.1M embeddings
//
// The issue's own Plan risk section explicitly allows this: "if [fixture
// build time] proves too slow for a single test run, batch size / total
// embedding count can be tuned down (still exercising the same
// per-dimension arithmetic) without invalidating the ratio assertion,
// since the ratio is scale-invariant by construction." Building the full
// ~11.1M-embedding × 768-dim generation (with k ≈ 53,300 k-means-formula
// buckets) means generating, normalising, and Q4-encoding ~8.5 billion
// floats — impractical for a routine `swift test` run in Debug, let alone
// CI. This fixture keeps the perf-relevant lever (768 dims — "the perf
// profile depends on per-dim operations", per the Specify-stage decision)
// and the two-generation (large + small) LSM shape from the reported
// workload, but scales chunk/token counts down by roughly 20×. The ratio
// and sustained-workload assertions hold at this scale for the same
// reason they'd hold at full scale: both are relative measurements
// (post/pre elapsed ratio; last-quartile-p95 vs first-quartile-median),
// not absolute-time thresholds.

import Foundation
@testable import SwitchcraftCore

enum SearchLatencyFixture {

    /// T5-base dimensionality — matches the issue's Q1 answer. Do not
    /// reduce this to speed up the fixture; low-dim mocks don't exercise
    /// the same per-dimension decode/dot-product cost profile real
    /// queries hit (Specify-stage Q1 decision).
    static let dims = 768

    struct GenerationSpec {
        let level: Int
        let chunkCount: Int
        let tokensPerChunk: Int
    }

    /// Mirrors the reported workload's two active generations: gen 124
    /// (level 3, the overwhelming majority of embeddings) and gen 127
    /// (level 1, a small recent L0→L1 cascade). Ratio ≈ 285:1 here vs.
    /// ≈ 236:1 in the original report — close enough to preserve the
    /// "one dominant generation + one small recent one" shape.
    static let defaultGenerationSpecs: [GenerationSpec] = [
        GenerationSpec(level: 3, chunkCount: 750, tokensPerChunk: 740),
        GenerationSpec(level: 1, chunkCount: 65, tokensPerChunk: 30),
    ]

    /// Witchcraft/production k-means coefficient (`k = round(16 * sqrt(n))`,
    /// per `IndexerConfig.production`). Even though this fixture bypasses
    /// k-means, sizing bucket count via the same formula keeps average
    /// bucket occupancy (and therefore per-bucket decode cost) realistic.
    static let kCoefficient = 16.0

    struct Built: Sendable {
        let storage: InMemoryStorage
        let totalEmbeddings: Int
        let totalChunks: Int
        let bucketCountsByGeneration: [Int]
    }

    // MARK: - Deterministic Gaussian generation

    /// Box-Muller transform over two `SplitMix64` draws. Reproducible for
    /// a fixed seed and call sequence — the fixed-seed requirement from
    /// the issue's Q1 answer (`SplitMix64(seed: 42)`).
    static func nextGaussian(_ rng: inout SplitMix64) -> Float {
        let u1 = Double(rng.next() >> 11) * (1.0 / 9_007_199_254_740_992.0)  // 2^-53
        let u2 = Double(rng.next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
        let r = (-2.0 * Foundation.log(max(u1, 1e-300))).squareRoot()
        return Float(r * Foundation.cos(2.0 * Double.pi * u2))
    }

    /// One unit-norm `dims`-length vector — the hypersphere geometry XTR
    /// MaxSim operates on (Specify-stage Q1 answer).
    static func randomUnitVector(dims: Int, rng: inout SplitMix64) -> [Float] {
        var v = [Float](repeating: 0, count: dims)
        var sumSq: Float = 0
        for j in 0..<dims {
            let g = nextGaussian(&rng)
            v[j] = g
            sumSq += g * g
        }
        let scale: Float = sumSq > 0 ? 1.0 / sumSq.squareRoot() : 1
        for j in 0..<dims { v[j] *= scale }
        return v
    }

    // MARK: - Fixture construction

    /// Build the fixture. Each bucket's centre is its first-assigned
    /// token's own embedding (not the zero vector — an all-buckets-share-
    /// one-centre fixture makes every query/centroid dot product
    /// identically 0, which degenerates top-k centroid selection into
    /// "first k buckets by index, identical for every query token,"
    /// producing an unrealistically tiny `selected` set). Real per-bucket
    /// embeddings give real, varied similarity so different query tokens
    /// select different top-k centroids, as in production. This is still
    /// not nearest-centroid assignment — token→bucket assignment is
    /// deterministic round-robin (`globalTokenIndex % k`), not k-means —
    /// the fixture only needs *a* real, distinct centre per bucket so
    /// `center + residual` reconstruction round-trips with realistic
    /// selection behaviour; it does not need clustering-accurate
    /// residual magnitudes, since this is a timing fixture, not a
    /// retrieval-quality fixture (Plan's explicit call).
    static func build(
        generationSpecs: [GenerationSpec] = defaultGenerationSpecs,
        kCoefficient: Double = kCoefficient,
        seed: UInt64 = 42
    ) async throws -> Built {
        let storage = InMemoryStorage()
        try await storage.open()
        var rng = SplitMix64(seed: seed)

        var totalEmbeddings = 0
        var totalChunks = 0
        var bucketCountsByGeneration: [Int] = []

        for spec in generationSpecs {
            let genEmbeddings = spec.chunkCount * spec.tokensPerChunk
            let k = max(1, Int((kCoefficient * Double(genEmbeddings).squareRoot()).rounded()))
            bucketCountsByGeneration.append(k)

            var bucketPairs = [[IndexPair]](repeating: [], count: k)
            var bucketResiduals = [Data](repeating: Data(), count: k)
            // Real per-bucket centre, populated lazily from each bucket's
            // first-assigned token embedding — NOT the zero vector. A
            // shared zero centre makes every bucket's query similarity
            // identically 0, which degenerates the top-k centroid sort
            // into "first k buckets by index, identical for every query
            // token" (verified during implementation: this collapsed
            // `selected` to ~64 buckets instead of the realistic
            // ~700-1000 the config.k/query-token-count math predicts).
            // Real embeddings give real (if noisy, since round-robin
            // assignment isn't nearest-centroid) similarity variation, so
            // different query tokens select different top-k buckets, as
            // in the production system.
            var bucketCenters = [[Float]?](repeating: nil, count: k)

            var minChunkID = Int64.max
            var maxChunkID = Int64.min
            var globalTokenIndex = 0

            for chunkIndex in 0..<spec.chunkCount {
                try Task.checkCancellation()
                let hash = "fixture-L\(spec.level)-c\(chunkIndex)"
                let chunk = try await storage.upsertChunk(
                    ChunkRecord(
                        hash: hash, model: "fixture", embeddings: Data(),
                        counts: [spec.tokensPerChunk]
                    )
                )
                let uuid = "fixture-doc-L\(spec.level)-\(chunkIndex)"
                try await storage.upsertDocument(
                    DocumentRecord(
                        uuid: uuid,
                        date: Date(timeIntervalSince1970: 0),
                        metadata: Data(),
                        hash: hash,
                        body: "synthetic fixture chunk \(chunkIndex) at level \(spec.level)",
                        lens: [spec.tokensPerChunk]
                    )
                )
                minChunkID = min(minChunkID, chunk.id)
                maxChunkID = max(maxChunkID, chunk.id)
                totalChunks += 1

                for tokenOffset in 0..<spec.tokensPerChunk {
                    let embedding = randomUnitVector(dims: dims, rng: &rng)
                    let bucketIdx = globalTokenIndex % k
                    let center: [Float]
                    if let existing = bucketCenters[bucketIdx] {
                        center = existing
                    } else {
                        center = embedding
                        bucketCenters[bucketIdx] = embedding
                    }
                    var residual = [Float](repeating: 0, count: dims)
                    for d in 0..<dims { residual[d] = embedding[d] - center[d] }
                    bucketPairs[bucketIdx].append(
                        IndexPair(chunkID: UInt32(chunk.id), tokenOffset: UInt32(tokenOffset))
                    )
                    bucketResiduals[bucketIdx].append(Q4Codec.encodeResiduals(residual))
                    globalTokenIndex += 1
                    totalEmbeddings += 1
                }
            }

            let genRecord = GenerationRecord(
                level: spec.level,
                numEmbeddings: genEmbeddings,
                minChunkID: minChunkID,
                maxChunkID: maxChunkID,
                created: Date(timeIntervalSince1970: 0)
            )
            let insertedGen = try await storage.insertGeneration(genRecord)

            for b in 0..<k {
                guard !bucketPairs[b].isEmpty else { continue }
                let centerData = Indexer.encodeFloat32LE(bucketCenters[b] ?? [Float](repeating: 0, count: dims))
                let bucket = BucketRecord(
                    generationID: insertedGen.id,
                    center: centerData,
                    indices: IndicesCodec.encode(bucketPairs[b]),
                    residuals: bucketResiduals[b]
                )
                _ = try await storage.insertBucket(bucket)
            }
        }

        return Built(
            storage: storage,
            totalEmbeddings: totalEmbeddings,
            totalChunks: totalChunks,
            bucketCountsByGeneration: bucketCountsByGeneration
        )
    }

    // MARK: - Query construction

    /// A synthetic query of `distinctTokenCount + duplicateTokenCount`
    /// rows: `distinctTokenCount` independent random unit vectors, plus
    /// `duplicateTokenCount` verbatim repeats of earlier rows. Mirrors
    /// the issue's own observation that natural-language queries (8
    /// English words → 15-20 T5 sub-word tokens) commonly repeat
    /// sub-word fragments, and exercises `searchOptimized`'s query-token
    /// dedup path.
    static func makeQuery(
        dims: Int,
        rng: inout SplitMix64,
        distinctTokenCount: Int = 15,
        duplicateTokenCount: Int = 3
    ) -> [Float] {
        var rows: [[Float]] = []
        rows.reserveCapacity(distinctTokenCount)
        for _ in 0..<distinctTokenCount {
            rows.append(randomUnitVector(dims: dims, rng: &rng))
        }
        var flat: [Float] = []
        flat.reserveCapacity((distinctTokenCount + duplicateTokenCount) * dims)
        for row in rows { flat.append(contentsOf: row) }
        for i in 0..<duplicateTokenCount {
            flat.append(contentsOf: rows[i % rows.count])
        }
        return flat
    }

    static func makeQuery(dims: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return makeQuery(dims: dims, rng: &rng)
    }
}
