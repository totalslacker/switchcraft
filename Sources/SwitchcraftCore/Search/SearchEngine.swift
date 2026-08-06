// SPDX-License-Identifier: Apache-2.0
import Foundation
import Accelerate

/// Read-only XTR-Warp search engine over any `SwitchcraftStorage`.
///
/// Mirrors upstream Witchcraft's `match_centroids` algorithm
/// (`src/lib.rs`), with one structural deviation: ADR 005 stores bucket
/// pairs as `(chunkID, tokenOffset)` rather than upstream's
/// `(documentRowID, subIdx)`. The engine bridges this by mapping each
/// scored chunk back to its owning documents through
/// `storage.documents(forChunkHash:)` (chunks are content-deduplicated;
/// multiple documents may share a chunk).
///
/// Pipeline:
///
///  1. For each generation, compute `query · centroidᵀ` for every bucket
///     centre (Accelerate `cblas_sgemm`).
///  2. Per (generation, query token): pick up to `k` centroids in
///     descending similarity, accumulating candidate-token cumulative
///     count until `>= tPrime`. The centroid that crosses the budget IS
///     included. `cumsum` resets per query token.
///  3. When a query token's top-`k` centroids in a generation do not
///     exhaust `tPrime`, record `missing[q] = max(missing[q], score of
///     the k-th centroid)` as the per-token baseline.
///  4. Decode every selected bucket once (`IndicesCodec` + `Q4Codec`)
///     and reconstruct candidate token embeddings as `centre + residual`.
///  5. Compute `query · candidateᵀ` (Accelerate) for all candidates.
///  6. Per query token, take the maximum dot product across all of a
///     document's candidate tokens, seeded with `missing[q]`.
///  7. Document score = mean over query tokens of the per-token max.
///     Filter via `StorageFilter.matches(_:)` and threshold; sort by
///     `(-score, uuid)` ascending; return prefix `topK`.
///
/// The actor is read-only — it never mutates storage. All inner loops
/// run sequentially to keep `cblas_sgemm` summation order deterministic
/// for the same inputs. See `adrs/006-search-constants.md` and
/// `adrs/007-search-vs-index-responsibility.md`.
public actor SearchEngine {

    /// Errors thrown by `SearchEngine` when input shapes are malformed
    /// or a pre-SQLite deadline check fires.
    public enum Error: Swift.Error, Sendable, Equatable {
        /// `dims` was zero, negative, or odd. Q4 residuals pack two
        /// floats per byte so `dims` must be a positive even number.
        case invalidDims(Int)
        /// Embedding buffer length is not a multiple of `dims`.
        case ragged(count: Int, dims: Int)
        /// A bucket's residual byte count does not match the number of
        /// pairs in its indices blob — almost always corruption or a
        /// mismatched dims.
        case bucketSizeMismatch(pairs: Int, residualBytes: Int, dims: Int)
        /// A bucket's centre blob was not `dims * 4` bytes.
        case centerSizeMismatch(bytes: Int, expected: Int)
        /// A pre-SQLite deadline check at the top of `searchHybrid`
        /// found the budget already exhausted. Translated to
        /// `SwitchcraftStoreError.searchTimedOut` by the caller.
        case deadlineExceeded(elapsed: Duration)
    }

    private let storage: any SwitchcraftStorage
    public let config: SearchConfig

    public init(
        storage: any SwitchcraftStorage,
        config: SearchConfig = SearchConfig()
    ) {
        self.storage = storage
        self.config = config
    }

    // MARK: - Public API

    /// Run the full vector-search pipeline.
    ///
    /// - Parameters:
    ///   - queryEmbeddings: row-major `n × dims` query token embeddings.
    ///   - dims: vector dimensionality. Must be positive.
    ///   - topK: maximum number of `SearchHit`s to return.
    ///   - filter: post-aggregation filter applied to candidate documents.
    ///
    /// - Returns: at most `topK` hits sorted by `score` descending; ties
    ///   broken by `uuid` lexicographic ascending. Returns `[]` when the
    ///   query is empty, the index has no embeddings, or every candidate
    ///   document is filtered out / falls below the threshold.
    public func search(
        queryEmbeddings: [Float],
        dims: Int,
        topK: Int,
        filter: StorageFilter = .all
    ) async throws -> [SearchHit] {
        if config.legacySequentialSearch {
            return try await searchLegacy(
                queryEmbeddings: queryEmbeddings, dims: dims, topK: topK, filter: filter
            )
        }
        return try await searchOptimized(
            queryEmbeddings: queryEmbeddings, dims: dims, topK: topK, filter: filter
        )
    }

    /// Pre-#140 reference implementation: fully sequential centroid scan
    /// and bucket decode, no query-token deduplication. Kept as a
    /// benchmark-only comparison path (see `SearchConfig.legacySequentialSearch`
    /// and ADR 035) and as an equivalence oracle in `SearchDeterminismTests`.
    /// Do not add new features here — this path is frozen by construction
    /// so it stays a faithful "before" baseline.
    private func searchLegacy(
        queryEmbeddings: [Float],
        dims: Int,
        topK: Int,
        filter: StorageFilter
    ) async throws -> [SearchHit] {
        guard dims > 0, dims % 2 == 0 else { throw Error.invalidDims(dims) }
        guard queryEmbeddings.count % dims == 0 else {
            throw Error.ragged(count: queryEmbeddings.count, dims: dims)
        }
        let n = queryEmbeddings.count / dims
        guard n > 0, topK > 0 else { return [] }

        // 1. Per-query-token centroid scan, accumulating selection
        //    decisions and the missing baseline.
        var missing = [Float](repeating: -.infinity, count: n)
        // Selected bucket ids, deduplicated, in first-seen order. Full
        // records are fetched in one batch after the scan loop completes
        // (see ADR 041) — the scan phase only needs `center` and residual
        // token count, not `indices`/`residuals`.
        var selectedIDs: [Int64] = []
        var seenBuckets = Set<Int64>()

        let generations = try await storage.generations()
        for gen in generations {
            let buckets = try await storage.scanBuckets(forGeneration: gen.id)
            if buckets.isEmpty { continue }

            // Build the centroid matrix and the per-bucket token count.
            var centersFlat = [Float]()
            centersFlat.reserveCapacity(buckets.count * dims)
            var bucketTokenCounts = [Int]()
            bucketTokenCounts.reserveCapacity(buckets.count)
            for bucket in buckets {
                let center = try Self.decodeCenter(bucket.center, dims: dims)
                centersFlat.append(contentsOf: center)
                let tokens = bucket.residualByteCount / (dims / 2)
                bucketTokenCounts.append(tokens)
            }

            // sims: n × numCentroids, row-major.
            let numCentroids = buckets.count
            let sims = Self.matmulQueryTimesRowMajorTranspose(
                queryEmbeddings: queryEmbeddings,
                n: n,
                rows: centersFlat,
                rowCount: numCentroids,
                dims: dims
            )

            // 2. Per query token: top-k with cumulative-token budget.
            for q in 0..<n {
                let row = q * numCentroids
                // Sort indices by descending similarity, breaking ties
                // by ascending centroid index. The total order makes
                // the result deterministic regardless of sort stability.
                var order = Array(0..<numCentroids)
                order.sort { a, b in
                    let sa = sims[row + a]
                    let sb = sims[row + b]
                    if sa != sb { return sa > sb }
                    return a < b
                }

                let limit = min(config.k, numCentroids)
                var cumsum = 0
                var crossed = false
                for j in 0..<limit {
                    let idx = order[j]
                    let bucketID = buckets[idx].id
                    if seenBuckets.insert(bucketID).inserted {
                        selectedIDs.append(bucketID)
                    }
                    cumsum += bucketTokenCounts[idx]
                    if cumsum >= config.tPrime {
                        crossed = true
                        break
                    }
                }
                if !crossed && limit > 0 {
                    let kthScore = sims[row + order[limit - 1]]
                    if kthScore > missing[q] {
                        missing[q] = kthScore
                    }
                }
            }
        }

        // No candidate buckets at all → no hits. (missing[q] is still
        // -inf so the threshold filter would drop everything anyway.)
        if selectedIDs.isEmpty { return [] }

        // 2.5. Fetch full records (indices + residuals) only for the
        //      buckets selection actually picked, then reconstruct in
        //      `selectedIDs` order — not fetch-return order — so decode
        //      output stays byte-identical to the pre-two-phase-split
        //      implementation (ADR 041). Ids that vanished between scan
        //      and fetch (e.g. a concurrent vacuum) are silently dropped,
        //      matching the existing tolerance for a generation vanishing
        //      between `storage.generations()` and the scan call above.
        let fetchedBuckets = try await storage.buckets(ids: selectedIDs)
        var bucketsByID: [Int64: BucketRecord] = [:]
        bucketsByID.reserveCapacity(fetchedBuckets.count)
        for bucket in fetchedBuckets { bucketsByID[bucket.id] = bucket }
        let selected: [BucketRecord] = selectedIDs.compactMap { bucketsByID[$0] }
        if selected.isEmpty { return [] }

        // 3. Decode every selected bucket and reconstruct candidate
        //    token embeddings as centre + residual.
        var candidatesFlat = [Float]()
        var candidateChunkIDs = [UInt32]()
        for bucket in selected {
            let center = try Self.decodeCenter(bucket.center, dims: dims)
            let pairs = try IndicesCodec.decode(bucket.indices)
            let residuals = Q4Codec.decodeResiduals(bucket.residuals)
            guard residuals.count == pairs.count * dims else {
                throw Error.bucketSizeMismatch(
                    pairs: pairs.count,
                    residualBytes: bucket.residuals.count,
                    dims: dims
                )
            }
            candidatesFlat.reserveCapacity(candidatesFlat.count + pairs.count * dims)
            candidateChunkIDs.reserveCapacity(candidateChunkIDs.count + pairs.count)
            for (i, pair) in pairs.enumerated() {
                let off = i * dims
                for j in 0..<dims {
                    candidatesFlat.append(center[j] + residuals[off + j])
                }
                candidateChunkIDs.append(pair.chunkID)
            }
        }

        let candidateCount = candidateChunkIDs.count
        if candidateCount == 0 { return [] }

        // 4. Score: candidate × queryᵀ → (candidateCount × n).
        //    Equivalent to `query · candidateᵀ` transposed; we lay out
        //    candidate-major so per-candidate rows are contiguous when
        //    we group them by document below.
        let scores = Self.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: candidatesFlat,
            n: candidateCount,
            rows: queryEmbeddings,
            rowCount: n,
            dims: dims
        )

        // 5. Group candidates by chunkID, then resolve to documents.
        //    Per-document per-q-token max, seeded with missing[q].
        var perDocMax: [String: [Float]] = [:]
        var chunkIDsToResolve: [UInt32] = []
        var chunkOffsets: [UInt32: [Int]] = [:]
        for i in 0..<candidateCount {
            chunkOffsets[candidateChunkIDs[i], default: []].append(i)
        }
        chunkIDsToResolve = Array(chunkOffsets.keys)
        // Sorted for deterministic iteration order over chunk groups.
        chunkIDsToResolve.sort()

        for chunkID in chunkIDsToResolve {
            let candidateRows = chunkOffsets[chunkID]!
            guard let chunk = try await storage.chunk(id: Int64(chunkID)) else {
                continue
            }
            let docs = try await storage.documents(forChunkHash: chunk.hash)
            if docs.isEmpty { continue }

            // Per-q-token max across the candidate tokens of this chunk.
            var chunkMax = [Float](repeating: -.infinity, count: n)
            for row in candidateRows {
                let base = row * n
                for q in 0..<n {
                    let s = scores[base + q]
                    if s > chunkMax[q] { chunkMax[q] = s }
                }
            }

            for doc in docs {
                if !filter.matches(doc) { continue }
                if perDocMax[doc.uuid] == nil {
                    var seeded = [Float](repeating: 0, count: n)
                    for q in 0..<n {
                        let missingVal = missing[q]
                        seeded[q] = missingVal == -.infinity ? -.infinity : missingVal
                    }
                    perDocMax[doc.uuid] = seeded
                }
                // Update the per-document max in place via the
                // dictionary's subscript modify accessor — avoids
                // copying the [Float] buffer per chunk per document.
                for q in 0..<n {
                    if chunkMax[q] > perDocMax[doc.uuid]![q] {
                        perDocMax[doc.uuid]![q] = chunkMax[q]
                    }
                }
            }
        }

        if perDocMax.isEmpty { return [] }

        // 6. Aggregate per-q-token maxima → mean → SearchHit.
        let scaler = 1.0 / Float(n)
        var hits: [SearchHit] = []
        hits.reserveCapacity(perDocMax.count)
        for (uuid, perToken) in perDocMax {
            // If any q-token had no evidence at all (-inf both in
            // candidates and in missing), the document is dropped — its
            // score is undefined.
            var sum: Float = 0
            var anyMissing = false
            for q in 0..<n {
                let v = perToken[q]
                if v == -.infinity { anyMissing = true; break }
                sum += v
            }
            if anyMissing { continue }
            let score = sum * scaler
            if score < config.threshold { continue }
            hits.append(SearchHit(uuid: uuid, score: score))
        }

        // 7. Deterministic ordering by (-score, uuid) ascending. The
        //    comparator defines a total order, so determinism does not
        //    depend on Swift's sort being stable.
        hits.sort { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.uuid < rhs.uuid
        }
        if hits.count > topK {
            return Array(hits.prefix(topK))
        }
        return hits
    }

    /// Default (post-#140) implementation. Behaviourally identical to
    /// `searchLegacy` — same selection rule, same MEAN aggregation, same
    /// `(-score, uuid)` total order — but:
    ///
    ///  1. Query tokens that are bit-exact duplicates of an earlier token
    ///     are collapsed to a single representative before centroid
    ///     selection and scoring; the representative's contribution is
    ///     multiplied by its duplicate count when computing the mean.
    ///     Provably a no-op transformation (ADR 028 already establishes
    ///     that repeated query tokens are a no-op under MEAN aggregation;
    ///     this is that fact applied in the collapse direction).
    ///  2. Decoding the deduplicated candidate bucket set (`center +
    ///     residual` reconstruction, the dominant per-search cost per
    ///     issue #140's own profiling) is parallelised across
    ///     `TaskGroup` child tasks that touch no actor-isolated state,
    ///     with each child's output written into an order-preserving
    ///     slot so concatenation reproduces the exact sequential byte
    ///     order — preserving ADR 006(f)'s bit-identical output
    ///     guarantee. See ADR 035 for the full design and measurements.
    private func searchOptimized(
        queryEmbeddings: [Float],
        dims: Int,
        topK: Int,
        filter: StorageFilter
    ) async throws -> [SearchHit] {
        guard dims > 0, dims % 2 == 0 else { throw Error.invalidDims(dims) }
        guard queryEmbeddings.count % dims == 0 else {
            throw Error.ragged(count: queryEmbeddings.count, dims: dims)
        }
        let n = queryEmbeddings.count / dims
        guard n > 0, topK > 0 else { return [] }

        // 0. Query-token dedup: collapse bit-exact-equal rows to a single
        //    representative, preserving first-occurrence order. `m <= n`
        //    unique rows drive every downstream computation; `duplicateCount[u]`
        //    is folded back in at the final mean-aggregation step so the
        //    output is identical to processing all `n` rows individually.
        var uniqueRowsFlat = [Float]()
        uniqueRowsFlat.reserveCapacity(queryEmbeddings.count)
        var duplicateCount = [Int]()
        var seen: [[Float]: Int] = [:]
        seen.reserveCapacity(n)
        for q in 0..<n {
            let start = q * dims
            let row = Array(queryEmbeddings[start..<(start + dims)])
            if let u = seen[row] {
                duplicateCount[u] += 1
            } else {
                let u = duplicateCount.count
                seen[row] = u
                uniqueRowsFlat.append(contentsOf: row)
                duplicateCount.append(1)
            }
        }
        let m = duplicateCount.count

        // 1. Per-unique-query-token centroid scan, accumulating selection
        //    decisions and the missing baseline.
        var missing = [Float](repeating: -.infinity, count: m)
        // Selected bucket ids, deduplicated, in first-seen order. Full
        // records are fetched in one batch after the scan loop completes
        // (see ADR 041) — the scan phase only needs `center` and residual
        // token count, not `indices`/`residuals`.
        var selectedIDs: [Int64] = []
        var seenBuckets = Set<Int64>()

        let generations = try await storage.generations()
        for gen in generations {
            try Task.checkCancellation()
            let buckets = try await storage.scanBuckets(forGeneration: gen.id)
            if buckets.isEmpty { continue }

            // Build the centroid matrix and the per-bucket token count.
            var centersFlat = [Float]()
            centersFlat.reserveCapacity(buckets.count * dims)
            var bucketTokenCounts = [Int]()
            bucketTokenCounts.reserveCapacity(buckets.count)
            for bucket in buckets {
                let center = try Self.decodeCenter(bucket.center, dims: dims)
                centersFlat.append(contentsOf: center)
                let tokens = bucket.residualByteCount / (dims / 2)
                bucketTokenCounts.append(tokens)
            }

            // sims: m × numCentroids, row-major.
            let numCentroids = buckets.count
            let sims = Self.matmulQueryTimesRowMajorTranspose(
                queryEmbeddings: uniqueRowsFlat,
                n: m,
                rows: centersFlat,
                rowCount: numCentroids,
                dims: dims
            )

            // 2. Per unique query token: top-k with cumulative-token budget.
            for u in 0..<m {
                let row = u * numCentroids
                // Sort indices by descending similarity, breaking ties
                // by ascending centroid index. The total order makes
                // the result deterministic regardless of sort stability.
                var order = Array(0..<numCentroids)
                order.sort { a, b in
                    let sa = sims[row + a]
                    let sb = sims[row + b]
                    if sa != sb { return sa > sb }
                    return a < b
                }

                let limit = min(config.k, numCentroids)
                var cumsum = 0
                var crossed = false
                for j in 0..<limit {
                    let idx = order[j]
                    let bucketID = buckets[idx].id
                    if seenBuckets.insert(bucketID).inserted {
                        selectedIDs.append(bucketID)
                    }
                    cumsum += bucketTokenCounts[idx]
                    if cumsum >= config.tPrime {
                        crossed = true
                        break
                    }
                }
                if !crossed && limit > 0 {
                    let kthScore = sims[row + order[limit - 1]]
                    if kthScore > missing[u] {
                        missing[u] = kthScore
                    }
                }
            }
        }

        // No candidate buckets at all → no hits. (missing[u] is still
        // -inf so the threshold filter would drop everything anyway.)
        if selectedIDs.isEmpty { return [] }

        try Task.checkCancellation()

        // 2.5. Fetch full records (indices + residuals) only for the
        //      buckets selection actually picked, then reconstruct in
        //      `selectedIDs` order — not fetch-return order — so decode
        //      output stays byte-identical to the pre-two-phase-split
        //      implementation (ADR 041). Ids that vanished between scan
        //      and fetch (e.g. a concurrent vacuum) are silently dropped,
        //      matching the existing tolerance for a generation vanishing
        //      between `storage.generations()` and the scan call above.
        let fetchedBuckets = try await storage.buckets(ids: selectedIDs)
        var bucketsByID: [Int64: BucketRecord] = [:]
        bucketsByID.reserveCapacity(fetchedBuckets.count)
        for bucket in fetchedBuckets { bucketsByID[bucket.id] = bucket }
        let selected: [BucketRecord] = selectedIDs.compactMap { bucketsByID[$0] }
        if selected.isEmpty { return [] }

        try Task.checkCancellation()

        // 3. Decode every selected bucket and reconstruct candidate token
        //    embeddings as centre + residual, in parallel chunks that
        //    concatenate back in original order (deterministic output).
        let (candidatesFlat, candidateChunkIDs) = try await Self.decodeSelectedBuckets(
            selected, dims: dims
        )

        try Task.checkCancellation()

        let candidateCount = candidateChunkIDs.count
        if candidateCount == 0 { return [] }

        // 4. Score: candidate × uniqueQueryᵀ → (candidateCount × m).
        //    Equivalent to `query · candidateᵀ` transposed; we lay out
        //    candidate-major so per-candidate rows are contiguous when
        //    we group them by document below.
        let scores = Self.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: candidatesFlat,
            n: candidateCount,
            rows: uniqueRowsFlat,
            rowCount: m,
            dims: dims
        )

        // 5. Group candidates by chunkID, then resolve to documents.
        //    Per-document per-unique-q-token max, seeded with missing[u].
        var perDocMax: [String: [Float]] = [:]
        var chunkIDsToResolve: [UInt32] = []
        var chunkOffsets: [UInt32: [Int]] = [:]
        for i in 0..<candidateCount {
            chunkOffsets[candidateChunkIDs[i], default: []].append(i)
        }
        chunkIDsToResolve = Array(chunkOffsets.keys)
        // Sorted for deterministic iteration order over chunk groups.
        chunkIDsToResolve.sort()

        for chunkID in chunkIDsToResolve {
            let candidateRows = chunkOffsets[chunkID]!
            guard let chunk = try await storage.chunk(id: Int64(chunkID)) else {
                continue
            }
            let docs = try await storage.documents(forChunkHash: chunk.hash)
            if docs.isEmpty { continue }

            // Per-unique-q-token max across the candidate tokens of this chunk.
            var chunkMax = [Float](repeating: -.infinity, count: m)
            for row in candidateRows {
                let base = row * m
                for u in 0..<m {
                    let s = scores[base + u]
                    if s > chunkMax[u] { chunkMax[u] = s }
                }
            }

            for doc in docs {
                if !filter.matches(doc) { continue }
                if perDocMax[doc.uuid] == nil {
                    var seeded = [Float](repeating: 0, count: m)
                    for u in 0..<m {
                        let missingVal = missing[u]
                        seeded[u] = missingVal == -.infinity ? -.infinity : missingVal
                    }
                    perDocMax[doc.uuid] = seeded
                }
                // Update the per-document max in place via the
                // dictionary's subscript modify accessor — avoids
                // copying the [Float] buffer per chunk per document.
                for u in 0..<m {
                    if chunkMax[u] > perDocMax[doc.uuid]![u] {
                        perDocMax[doc.uuid]![u] = chunkMax[u]
                    }
                }
            }
        }

        if perDocMax.isEmpty { return [] }

        // 6. Aggregate per-unique-q-token maxima → mean over the original
        //    `n` query tokens → SearchHit. Each unique token's maximum is
        //    weighted by its duplicate count so the sum is identical to
        //    summing all `n` original (non-deduplicated) token maxima.
        let scaler = 1.0 / Float(n)
        var hits: [SearchHit] = []
        hits.reserveCapacity(perDocMax.count)
        for (uuid, perToken) in perDocMax {
            // If any unique token had no evidence at all (-inf both in
            // candidates and in missing), the document is dropped — its
            // score is undefined.
            var sum: Float = 0
            var anyMissing = false
            for u in 0..<m {
                let v = perToken[u]
                if v == -.infinity { anyMissing = true; break }
                sum += v * Float(duplicateCount[u])
            }
            if anyMissing { continue }
            let score = sum * scaler
            if score < config.threshold { continue }
            hits.append(SearchHit(uuid: uuid, score: score))
        }

        // 7. Deterministic ordering by (-score, uuid) ascending. The
        //    comparator defines a total order, so determinism does not
        //    depend on Swift's sort being stable.
        hits.sort { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.uuid < rhs.uuid
        }
        if hits.count > topK {
            return Array(hits.prefix(topK))
        }
        return hits
    }

    /// Run vector search and full-text search in tandem and fuse their
    /// ranked lists via Reciprocal Rank Fusion (equal-weight, 1-indexed).
    ///
    /// Pipeline:
    ///  1. Vector candidates: call `search(...)` with `topK = perSourceBudget`.
    ///  2. FTS candidates: call `storage.searchFullText(...)` with
    ///     `limit = perSourceBudget`. The same `filter` is pushed into both
    ///     sub-calls as a performance optimisation.
    ///  3. For every uuid present in either list, compute
    ///     `rrf = Σ 1 / (rrfK + rank)` over its sources (1-indexed).
    ///  4. Apply `filter` to the union of uuids by re-fetching each
    ///     `DocumentRecord` and evaluating `filter.matches(_:)` — the
    ///     correctness gate against any future divergence between the
    ///     two backends' native filter lowerings and `StorageFilter`.
    ///  5. Sort by `(rrf DESC, uuid ASC)` and return the prefix `topK`.
    ///
    /// Empty-input fallbacks (still RRF-formatted scores):
    /// - No query embeddings (`queryEmbeddings.count == 0`): FTS-only.
    /// - Empty / whitespace-only `queryText`: vector-only.
    /// - Both empty: returns `[]`.
    ///
    /// All sub-calls are awaited sequentially — no `TaskGroup` — so the
    /// fused output is byte-identical for the same inputs.
    ///
    /// - Parameters:
    ///   - queryEmbeddings: row-major `n × dims` query token embeddings.
    ///     Pass an empty array to skip vector search.
    ///   - dims: vector dimensionality. Ignored when `queryEmbeddings`
    ///     is empty.
    ///   - queryText: raw query string for the FTS source.
    ///     Whitespace-only and empty strings skip FTS search.
    ///   - topK: maximum number of fused hits to return.
    ///   - filter: pushed to both sources and re-applied to the union.
    ///   - config: RRF tuning. See `HybridConfig` and ADR 008.
    public func searchHybrid(
        queryEmbeddings: [Float],
        dims: Int,
        queryText: String,
        topK: Int,
        filter: StorageFilter = .all,
        config: HybridConfig = HybridConfig(),
        deadlineContext: SearchDeadlineContext? = nil
    ) async throws -> HybridSearchResult {
        guard topK > 0 else {
            return HybridSearchResult(
                hits: [],
                timing: HybridSearchTiming(vectorDuration: .zero, bm25Duration: .zero, mergeDuration: .zero)
            )
        }

        // Pre-phase deadline check. Must run before any storage access.
        if let ctx = deadlineContext, ctx.isExpired {
            throw Error.deadlineExceeded(elapsed: ctx.elapsed)
        }

        // Note: this method does NOT arm or disarm the backend's progress
        // handler. Callers that want progress-handler timeout enforcement
        // must call `await storage.configureSearchDeadline(ctx)` before
        // calling this method and `await storage.configureSearchDeadline(nil)`
        // after (on both success and error paths). `SwitchcraftStore.search()`
        // owns the arm/disarm lifecycle. Direct callers of `searchHybrid`
        // are responsible for the same contract.

        let trimmedQuery = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasVector = !queryEmbeddings.isEmpty
        let hasText = !trimmedQuery.isEmpty
        if !hasVector && !hasText {
            return HybridSearchResult(
                hits: [],
                timing: HybridSearchTiming(vectorDuration: .zero, bm25Duration: .zero, mergeDuration: .zero)
            )
        }

        // 1. Vector candidates (skipped if no embeddings).
        // Check both task cancellation and wall-time deadline before the
        // vector search (BLAS matmuls can run long on large indices).
        let vectorStageStart = ContinuousClock.now
        try Task.checkCancellation()
        if let ctx = deadlineContext, ctx.isExpired {
            throw Error.deadlineExceeded(elapsed: ctx.elapsed)
        }
        let vectorHits: [SearchHit]
        if hasVector {
            vectorHits = try await search(
                queryEmbeddings: queryEmbeddings,
                dims: dims,
                topK: config.perSourceBudget,
                filter: filter
            )
        } else {
            vectorHits = []
        }
        let vectorDuration = ContinuousClock.now - vectorStageStart

        // 2. FTS candidates (skipped if no text).
        let bm25StageStart = ContinuousClock.now
        try Task.checkCancellation()
        if let ctx = deadlineContext, ctx.isExpired {
            throw Error.deadlineExceeded(elapsed: ctx.elapsed)
        }
        let ftsHits: [FullTextHit]
        if hasText {
            ftsHits = try await storage.searchFullText(
                query: trimmedQuery,
                limit: config.perSourceBudget,
                filter: filter
            )
        } else {
            ftsHits = []
        }
        let bm25Duration = ContinuousClock.now - bm25StageStart

        // 3. Build per-uuid union with 1-indexed ranks and per-source raw
        //    scores. Dictionary iteration order is unspecified, but the
        //    final `(score DESC, uuid ASC)` sort below is a total order
        //    so the fused output is deterministic regardless.
        let mergeStageStart = ContinuousClock.now
        try Task.checkCancellation()
        if let ctx = deadlineContext, ctx.isExpired {
            throw Error.deadlineExceeded(elapsed: ctx.elapsed)
        }
        struct Provenance {
            var vectorRank: Int?
            var vectorScore: Float?
            var ftsRank: Int?
            var ftsScore: Float?
        }
        var union: [String: Provenance] = [:]
        union.reserveCapacity(vectorHits.count + ftsHits.count)

        for (i, hit) in vectorHits.enumerated() {
            var prov = union[hit.uuid] ?? Provenance()
            prov.vectorRank = i + 1
            prov.vectorScore = hit.score
            union[hit.uuid] = prov
        }
        for (i, hit) in ftsHits.enumerated() {
            var prov = union[hit.uuid] ?? Provenance()
            prov.ftsRank = i + 1
            prov.ftsScore = hit.score
            union[hit.uuid] = prov
        }

        if union.isEmpty {
            let mergeDuration = ContinuousClock.now - mergeStageStart
            return HybridSearchResult(
                hits: [],
                timing: HybridSearchTiming(
                    vectorDuration: vectorDuration,
                    bm25Duration: bm25Duration,
                    mergeDuration: mergeDuration
                )
            )
        }

        // 4. Final filter pass on the union. The same filter has already
        //    been pushed into both sub-calls; this is the correctness
        //    gate against any divergence between a backend's native
        //    filter lowering and `StorageFilter.matches`.
        let kFloat = Float(config.rrfK)
        var hits: [HybridHit] = []
        hits.reserveCapacity(union.count)
        for (uuid, prov) in union {
            if case .all = filter {
                // Skip the per-uuid document fetch when there is no
                // filter to apply.
            } else {
                guard let document = try await storage.document(uuid: uuid),
                      filter.matches(document)
                else {
                    continue
                }
            }

            var score: Float = 0
            if let r = prov.vectorRank { score += 1 / (kFloat + Float(r)) }
            if let r = prov.ftsRank    { score += 1 / (kFloat + Float(r)) }

            hits.append(HybridHit(
                uuid: uuid,
                score: score,
                vectorRank: prov.vectorRank,
                vectorScore: prov.vectorScore,
                ftsRank: prov.ftsRank,
                ftsScore: prov.ftsScore
            ))
        }

        // 5. Sort by (rrf DESC, uuid ASC) — total order, deterministic
        //    regardless of the underlying sort's stability.
        hits.sort { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.uuid < rhs.uuid
        }
        let mergeDuration = ContinuousClock.now - mergeStageStart
        let timing = HybridSearchTiming(
            vectorDuration: vectorDuration,
            bm25Duration: bm25Duration,
            mergeDuration: mergeDuration
        )
        if hits.count > topK {
            return HybridSearchResult(hits: Array(hits.prefix(topK)), timing: timing)
        }
        return HybridSearchResult(hits: hits, timing: timing)
    }

    /// Score already-embedded passage token vectors directly against the
    /// query, without any storage or centroid lookup. Pure in-memory
    /// MaxSim utility — for each passage, compute per-q-token max dot
    /// product across the passage's tokens and take the mean across q.
    ///
    /// - Parameters:
    ///   - queryEmbeddings: row-major `n × dims` query token embeddings.
    ///   - dims: vector dimensionality.
    ///   - passages: list of per-passage row-major `[Float]` token
    ///     embeddings, each a multiple of `dims`.
    ///
    /// - Returns: one score per passage, in input order.
    public func score(
        queryEmbeddings: [Float],
        dims: Int,
        passages: [[Float]]
    ) async throws -> [Float] {
        guard dims > 0, dims % 2 == 0 else { throw Error.invalidDims(dims) }
        guard queryEmbeddings.count % dims == 0 else {
            throw Error.ragged(count: queryEmbeddings.count, dims: dims)
        }
        let n = queryEmbeddings.count / dims
        var out = [Float](repeating: 0, count: passages.count)
        if n == 0 { return out }
        let scaler = 1.0 / Float(n)

        for (i, passage) in passages.enumerated() {
            guard passage.count % dims == 0 else {
                throw Error.ragged(count: passage.count, dims: dims)
            }
            let m = passage.count / dims
            if m == 0 {
                out[i] = 0
                continue
            }
            // (m × n) = passage × queryᵀ.
            let scores = Self.matmulQueryTimesRowMajorTranspose(
                queryEmbeddings: passage,
                n: m,
                rows: queryEmbeddings,
                rowCount: n,
                dims: dims
            )
            // Per-q-token max across passage tokens, then mean across q.
            var perTokenMax = [Float](repeating: -.infinity, count: n)
            for row in 0..<m {
                let base = row * n
                for q in 0..<n {
                    let s = scores[base + q]
                    if s > perTokenMax[q] { perTokenMax[q] = s }
                }
            }
            var sum: Float = 0
            for q in 0..<n { sum += perTokenMax[q] }
            out[i] = sum * scaler
        }
        return out
    }

    // MARK: - Helpers

    /// Decode `bucket.center` (Float32 LE bytes) into `[Float]` of
    /// length `dims`, validating the byte count.
    static func decodeCenter(_ data: Data, dims: Int) throws -> [Float] {
        let expected = dims * 4
        guard data.count == expected else {
            throw Error.centerSizeMismatch(bytes: data.count, expected: expected)
        }
        return Indexer.decodeFloat32LE(data)
    }

    /// Decode one contiguous slice of `selected` buckets into flat
    /// candidate token embeddings (`center + residual`, vDSP-added) and
    /// their owning chunk IDs. Pure function of its inputs — touches no
    /// actor state — so it is safe to run inside a `TaskGroup` child task.
    private static func decodeBucketChunk(
        _ buckets: ArraySlice<BucketRecord>, dims: Int
    ) throws -> (flat: [Float], chunkIDs: [UInt32]) {
        var candidatesFlat = [Float]()
        var candidateChunkIDs = [UInt32]()
        for bucket in buckets {
            let center = try Self.decodeCenter(bucket.center, dims: dims)
            let pairs = try IndicesCodec.decode(bucket.indices)
            let residuals = Q4Codec.decodeResiduals(bucket.residuals)
            guard residuals.count == pairs.count * dims else {
                throw Error.bucketSizeMismatch(
                    pairs: pairs.count,
                    residualBytes: bucket.residuals.count,
                    dims: dims
                )
            }
            let base = candidatesFlat.count
            candidatesFlat.append(contentsOf: repeatElement(0, count: pairs.count * dims))
            candidateChunkIDs.reserveCapacity(candidateChunkIDs.count + pairs.count)
            center.withUnsafeBufferPointer { centerPtr in
                residuals.withUnsafeBufferPointer { residualPtr in
                    candidatesFlat.withUnsafeMutableBufferPointer { outPtr in
                        for i in 0..<pairs.count {
                            let residualOff = i * dims
                            let outOff = base + residualOff
                            vDSP_vadd(
                                centerPtr.baseAddress!, 1,
                                residualPtr.baseAddress! + residualOff, 1,
                                outPtr.baseAddress! + outOff, 1,
                                vDSP_Length(dims)
                            )
                        }
                    }
                }
            }
            for pair in pairs {
                candidateChunkIDs.append(pair.chunkID)
            }
        }
        return (candidatesFlat, candidateChunkIDs)
    }

    /// Decode every bucket in `selected` and reconstruct candidate token
    /// embeddings as `centre + residual`, splitting the (deduplicated)
    /// bucket set into contiguous chunks decoded concurrently via
    /// `TaskGroup`. Chunk results are concatenated back in their original
    /// `selected`-order (not completion order), so the output is
    /// byte-identical to a fully sequential decode — see ADR 006(f) and
    /// ADR 035. Small bucket sets skip the `TaskGroup` entirely; the
    /// per-task dispatch overhead isn't worth it below the threshold.
    static func decodeSelectedBuckets(
        _ selected: [BucketRecord], dims: Int
    ) async throws -> (flat: [Float], chunkIDs: [UInt32]) {
        let sequentialThreshold = 64
        if selected.count <= sequentialThreshold {
            return try decodeBucketChunk(selected[...], dims: dims)
        }

        let processorCount = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunkCount = min(processorCount, selected.count)
        let chunkSize = (selected.count + chunkCount - 1) / chunkCount
        let ranges: [Range<Int>] = stride(from: 0, to: selected.count, by: chunkSize).map { start in
            start..<min(start + chunkSize, selected.count)
        }

        var decodedByIndex = [(flat: [Float], chunkIDs: [UInt32])?](repeating: nil, count: ranges.count)
        try await withThrowingTaskGroup(of: (Int, (flat: [Float], chunkIDs: [UInt32])).self) { group in
            for (idx, range) in ranges.enumerated() {
                let slice = selected[range]
                group.addTask {
                    try Task.checkCancellation()
                    let decoded = try Self.decodeBucketChunk(slice, dims: dims)
                    return (idx, decoded)
                }
            }
            for try await (idx, decoded) in group {
                decodedByIndex[idx] = decoded
            }
        }

        var totalTokens = 0
        for decoded in decodedByIndex { totalTokens += decoded!.chunkIDs.count }
        var flat = [Float]()
        flat.reserveCapacity(totalTokens * dims)
        var chunkIDs = [UInt32]()
        chunkIDs.reserveCapacity(totalTokens)
        for decoded in decodedByIndex {
            flat.append(contentsOf: decoded!.flat)
            chunkIDs.append(contentsOf: decoded!.chunkIDs)
        }
        return (flat, chunkIDs)
    }

    /// Compute `query × rowsᵀ` as an `(n × rowCount)` row-major matrix.
    ///
    /// Wraps `cblas_sgemm` so callers can reuse the kernel for both the
    /// per-generation centroid pass and the candidate-scoring pass.
    /// The Swift-level call shape is deterministic (single sgemm with
    /// fixed argument order); BLAS internal reduction order is not
    /// controlled here and may vary across architectures, so callers
    /// must compare results with an absolute tolerance rather than
    /// expecting bit-exact equality.
    internal static func matmulQueryTimesRowMajorTranspose(
        queryEmbeddings: [Float],
        n: Int,
        rows: [Float],
        rowCount: Int,
        dims: Int
    ) -> [Float] {
        precondition(queryEmbeddings.count == n * dims)
        precondition(rows.count == rowCount * dims)
        precondition(
            n <= Int(Int32.max) && rowCount <= Int(Int32.max) && dims <= Int(Int32.max),
            "search dimensions (n=\(n), rows=\(rowCount), dims=\(dims)) must fit in Int32"
        )
        var out = [Float](repeating: 0, count: n * rowCount)
        if n == 0 || rowCount == 0 { return out }
        queryEmbeddings.withUnsafeBufferPointer { qPtr in
            rows.withUnsafeBufferPointer { rPtr in
                out.withUnsafeMutableBufferPointer { oPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(n), Int32(rowCount), Int32(dims),
                        1.0,
                        qPtr.baseAddress, Int32(dims),
                        rPtr.baseAddress, Int32(dims),
                        0.0,
                        oPtr.baseAddress, Int32(rowCount)
                    )
                }
            }
        }
        return out
    }
}
