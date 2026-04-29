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

    /// Errors thrown by `SearchEngine` when input shapes are malformed.
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
        guard dims > 0, dims % 2 == 0 else { throw Error.invalidDims(dims) }
        guard queryEmbeddings.count % dims == 0 else {
            throw Error.ragged(count: queryEmbeddings.count, dims: dims)
        }
        let n = queryEmbeddings.count / dims
        guard n > 0, topK > 0 else { return [] }

        // 1. Per-query-token centroid scan, accumulating selection
        //    decisions and the missing baseline.
        var missing = [Float](repeating: -.infinity, count: n)
        // Selected (generationID, bucket) records, deduplicated.
        var selected: [BucketRecord] = []
        var seenBuckets = Set<Int64>()

        let generations = try await storage.generations()
        for gen in generations {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            if buckets.isEmpty { continue }

            // Build the centroid matrix and the per-bucket token count.
            var centersFlat = [Float]()
            centersFlat.reserveCapacity(buckets.count * dims)
            var bucketTokenCounts = [Int]()
            bucketTokenCounts.reserveCapacity(buckets.count)
            for bucket in buckets {
                let center = try Self.decodeCenter(bucket.center, dims: dims)
                centersFlat.append(contentsOf: center)
                let tokens = bucket.residuals.count / (dims / 2)
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
                        selected.append(buckets[idx])
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
                        let m = missing[q]
                        seeded[q] = m == -.infinity ? -.infinity : m
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

    /// Compute `query × rowsᵀ` as an `(n × rowCount)` row-major matrix.
    ///
    /// Wraps `cblas_sgemm` so callers can reuse the kernel for both the
    /// per-generation centroid pass and the candidate-scoring pass.
    /// Sequential evaluation is preserved so summation order — and
    /// therefore the float-bit-equality determinism the spec mandates —
    /// is stable across runs.
    fileprivate static func matmulQueryTimesRowMajorTranspose(
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
