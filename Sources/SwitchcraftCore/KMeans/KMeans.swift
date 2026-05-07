// SPDX-License-Identifier: Apache-2.0
import Foundation
import Accelerate

/// Spherical k-means clustering for L2-normalised row vectors.
///
/// Port of upstream Witchcraft's `kmeans` in `lib.rs` (see `NOTICE`).
/// Algorithm:
///
///  1. Initialise `k` centroids by sampling `k` distinct rows of `data`.
///  2. For `maxIterations` passes:
///     - Assign each row to the centroid with the largest dot product
///       (equivalent to nearest-by-cosine for unit vectors).
///     - Sum the rows assigned to each cluster.
///     - L2-normalise each sum to produce the new centroid. Empty
///       clusters reseed from a randomly chosen row.
///
/// No convergence check — iteration count is fixed by the caller.
public enum KMeans {

    /// Output of one `cluster(...)` run: the centroids and the
    /// per-row cluster assignments.
    public struct Result: Sendable {
        /// Row-major `clusters * dims` centroids, L2-normalised.
        public let centroids: [Float]
        /// Cluster assignment for each input row, length `data.count / dims`.
        public let assignments: [Int]
    }

    /// Run spherical k-means.
    ///
    /// - Parameters:
    ///   - data: row-major `m * dims` array of L2-normalised vectors.
    ///   - dims: vector dimensionality (n).
    ///   - clusters: target number of clusters (k). Must satisfy `k ≤ m`.
    ///   - maxIterations: number of passes. Witchcraft's caller uses ~25.
    ///   - rng: source of randomness for initialisation and empty-cluster
    ///     reseeding. Inject a `SplitMix64(seed:)` for deterministic runs.
    public static func cluster<R: RandomNumberGenerator>(
        data: [Float],
        dims: Int,
        clusters k: Int,
        maxIterations: Int = 25,
        rng: inout R
    ) -> Result {
        precondition(dims > 0, "dims must be positive")
        precondition(k > 0, "clusters must be positive")
        precondition(data.count % dims == 0, "data length must be a multiple of dims")
        let m = data.count / dims
        precondition(k <= m, "clusters (\(k)) must be <= number of rows (\(m))")

        // 1. Init: sample k distinct row indices without replacement.
        let indices = sampleDistinct(count: k, from: m, rng: &rng)
        var centroids = [Float](repeating: 0, count: k * dims)
        for (slot, row) in indices.enumerated() {
            let src = row * dims
            let dst = slot * dims
            for j in 0..<dims {
                centroids[dst + j] = data[src + j]
            }
        }

        var assignments = [Int](repeating: 0, count: m)

        for _ in 0..<maxIterations {
            // 2a. Assignment by argmax dot product. Computed as a single
            // Accelerate sgemm (data × centroidsᵀ) followed by per-row
            // argmax — same arithmetic as the naive double loop, ~30×
            // faster on Apple silicon for typical (m, k, dims) sizes.
            assignAll(
                data: data, dims: dims, centroids: centroids, k: k,
                into: &assignments
            )

            // 2b. Sum rows into per-cluster accumulators.
            var sums = [Float](repeating: 0, count: k * dims)
            var counts = [Int](repeating: 0, count: k)
            for row in 0..<m {
                let c = assignments[row]
                counts[c] += 1
                let src = row * dims
                let dst = c * dims
                for j in 0..<dims {
                    sums[dst + j] += data[src + j]
                }
            }

            // 2c. Normalise; reseed empty clusters from a random row.
            for c in 0..<k {
                let dst = c * dims
                let source: ArraySlice<Float>
                if counts[c] > 0 {
                    source = sums[dst..<dst + dims]
                } else {
                    let reseedRow = sampleDistinct(count: 1, from: m, rng: &rng)[0]
                    let s = reseedRow * dims
                    source = data[s..<s + dims]
                }
                let norm = sqrt(source.reduce(0) { $0 + $1 * $1 })
                if norm > 0 {
                    var i = dst
                    for v in source {
                        centroids[i] = v / norm
                        i += 1
                    }
                } else {
                    var i = dst
                    for v in source {
                        centroids[i] = v
                        i += 1
                    }
                }
            }

        }

        // Final assignments against the converged centroids so callers can
        // build buckets without re-running the assignment kernel.
        assignAll(
            data: data, dims: dims, centroids: centroids, k: k,
            into: &assignments
        )

        return Result(centroids: centroids, assignments: assignments)
    }

    /// Assign each row of `data` to the nearest centroid by dot product.
    /// Used to compute residuals or filter candidates outside of training.
    public static func assign(
        data: [Float],
        dims: Int,
        centroids: [Float]
    ) -> [Int] {
        precondition(dims > 0)
        precondition(data.count % dims == 0)
        precondition(centroids.count % dims == 0)
        let m = data.count / dims
        let k = centroids.count / dims
        precondition(k > 0)

        var out = [Int](repeating: 0, count: m)
        assignAll(data: data, dims: dims, centroids: centroids, k: k, into: &out)
        return out
    }

    // MARK: - Helpers

    /// Compute `scores = data × centroidsᵀ` (m × k, row-major) via
    /// Accelerate's sgemm, then write the per-row argmax into
    /// `assignments`. Same algorithm as the naive nested loop, but
    /// vectorised — necessary for the indexer's 5,000-embedding perf
    /// gate (acceptance criteria require <5s).
    private static func assignAll(
        data: [Float],
        dims: Int,
        centroids: [Float],
        k: Int,
        into assignments: inout [Int]
    ) {
        let m = data.count / dims
        guard m > 0 else { return }
        precondition(
            m <= Int(Int32.max) && k <= Int(Int32.max) && dims <= Int(Int32.max),
            "k-means dimensions (m=\(m), k=\(k), dims=\(dims)) must fit in Int32 for cblas_sgemm"
        )
        var scores = [Float](repeating: 0, count: m * k)
        data.withUnsafeBufferPointer { dataPtr in
            centroids.withUnsafeBufferPointer { centroidsPtr in
                scores.withUnsafeMutableBufferPointer { scoresPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(m), Int32(k), Int32(dims),
                        1.0,
                        dataPtr.baseAddress, Int32(dims),
                        centroidsPtr.baseAddress, Int32(dims),
                        0.0,
                        scoresPtr.baseAddress, Int32(k)
                    )
                }
            }
        }
        // Use vDSP_maxvi for per-row argmax: Accelerate's implementation is
        // always SIMD-compiled (pre-built dylib), so debug mode pays the same
        // cost as release. The naive Swift loop above was ~5s in debug mode
        // for m=5000, k=1131 — leaving essentially no headroom against the
        // 10s budget. vDSP_maxvi reduces this to ~15ms in debug mode.
        scores.withUnsafeBufferPointer { scoresPtr in
            let base = scoresPtr.baseAddress!
            for row in 0..<m {
                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(base + row * k, 1, &maxVal, &maxIdx, vDSP_Length(k))
                assignments[row] = Int(maxIdx)
            }
        }
    }

    /// Reservoir-style sampling without replacement (Floyd's algorithm).
    private static func sampleDistinct<R: RandomNumberGenerator>(
        count: Int, from population: Int, rng: inout R
    ) -> [Int] {
        precondition(count <= population)
        var picked = Set<Int>()
        picked.reserveCapacity(count)
        var ordered = [Int]()
        ordered.reserveCapacity(count)
        // Floyd's algorithm: for j = N-K..<N, pick t in [0, j], add t if new else add j.
        for j in (population - count)..<population {
            let t = Int(rng.next() % UInt64(j + 1))
            if picked.insert(t).inserted {
                ordered.append(t)
            } else {
                picked.insert(j)
                ordered.append(j)
            }
        }
        return ordered
    }
}
