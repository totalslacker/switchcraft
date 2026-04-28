import Foundation

/// Spherical k-means clustering for L2-normalised row vectors.
///
/// Port of upstream Witchcraft's `kmeans` in `lib.rs`. Algorithm:
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
            // 2a. Assignment by argmax dot product.
            for row in 0..<m {
                var bestCluster = 0
                var bestScore: Float = -.infinity
                let rowOffset = row * dims
                for c in 0..<k {
                    let cOffset = c * dims
                    var s: Float = 0
                    for j in 0..<dims {
                        s += data[rowOffset + j] * centroids[cOffset + j]
                    }
                    if s > bestScore {
                        bestScore = s
                        bestCluster = c
                    }
                }
                assignments[row] = bestCluster
            }

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
        for row in 0..<m {
            assignments[row] = nearestCentroid(
                row: row, dims: dims, data: data, centroids: centroids, k: k
            )
        }

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
        for row in 0..<m {
            out[row] = nearestCentroid(
                row: row, dims: dims, data: data, centroids: centroids, k: k
            )
        }
        return out
    }

    // MARK: - Helpers

    @inline(__always)
    private static func nearestCentroid(
        row: Int, dims: Int, data: [Float], centroids: [Float], k: Int
    ) -> Int {
        var bestCluster = 0
        var bestScore: Float = -.infinity
        let rowOffset = row * dims
        for c in 0..<k {
            let cOffset = c * dims
            var s: Float = 0
            for j in 0..<dims {
                s += data[rowOffset + j] * centroids[cOffset + j]
            }
            if s > bestScore {
                bestScore = s
                bestCluster = c
            }
        }
        return bestCluster
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
