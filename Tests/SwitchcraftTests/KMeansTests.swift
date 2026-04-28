import Foundation
import Testing
import SwitchcraftCore

@Suite("K-Means Clustering")
struct KMeansTests {

    // MARK: - Helpers

    /// Build `count` unit vectors of length `dims` distributed near `mean`,
    /// then L2-normalise. Uses a SplitMix64 for reproducibility.
    static func makeCluster(
        around mean: [Float], count: Int, jitter: Float, rng: inout SplitMix64
    ) -> [Float] {
        let dims = mean.count
        var rows = [Float]()
        rows.reserveCapacity(count * dims)
        for _ in 0..<count {
            var v = [Float](repeating: 0, count: dims)
            for j in 0..<dims {
                let u = Float(rng.next()) / Float(UInt64.max) // [0, 1]
                v[j] = mean[j] + (u - 0.5) * 2 * jitter
            }
            // L2-normalise
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            if norm > 0 {
                for j in 0..<dims { v[j] /= norm }
            }
            rows.append(contentsOf: v)
        }
        return rows
    }

    static func l2Norm(_ v: ArraySlice<Float>) -> Float {
        sqrt(v.reduce(0) { $0 + $1 * $1 })
    }

    // MARK: - Tests

    @Test("centroids have shape clusters × dims")
    func shape() {
        var rng = SplitMix64(seed: 1)
        let data = Self.makeCluster(around: [1, 0], count: 20, jitter: 0.05, rng: &rng)
        let result = KMeans.cluster(data: data, dims: 2, clusters: 3, maxIterations: 5, rng: &rng)
        #expect(result.centroids.count == 3 * 2)
        #expect(result.assignments.count == 20)
    }

    @Test("centroids are L2-normalised")
    func unitCentroids() {
        var rng = SplitMix64(seed: 2)
        let data = Self.makeCluster(around: [1, 0, 0], count: 30, jitter: 0.1, rng: &rng)
        let result = KMeans.cluster(data: data, dims: 3, clusters: 4, maxIterations: 10, rng: &rng)
        for c in 0..<4 {
            let slice = result.centroids[(c * 3)..<((c + 1) * 3)]
            #expect(abs(Self.l2Norm(slice) - 1.0) < 1e-5)
        }
    }

    @Test("assignments fall in [0, clusters)")
    func assignmentRange() {
        var rng = SplitMix64(seed: 3)
        let data = Self.makeCluster(around: [1, 0], count: 50, jitter: 0.2, rng: &rng)
        let k = 4
        let result = KMeans.cluster(data: data, dims: 2, clusters: k, maxIterations: 5, rng: &rng)
        for a in result.assignments {
            #expect(a >= 0 && a < k)
        }
    }

    @Test("recovers three well-separated clusters")
    func recoversClusters() {
        var rng = SplitMix64(seed: 42)
        // Three distinct directions on the unit circle.
        let centersTrue: [[Float]] = [
            [1, 0],
            [-0.5, 0.866_025], // 120°
            [-0.5, -0.866_025], // 240°
        ]
        var data = [Float]()
        var groundTruth = [Int]()
        let perCluster = 30
        for (idx, mean) in centersTrue.enumerated() {
            let rows = Self.makeCluster(around: mean, count: perCluster, jitter: 0.05, rng: &rng)
            data.append(contentsOf: rows)
            groundTruth.append(contentsOf: Array(repeating: idx, count: perCluster))
        }

        let result = KMeans.cluster(
            data: data, dims: 2, clusters: 3, maxIterations: 25, rng: &rng
        )

        // For each true cluster, all of its points should land in the same
        // recovered cluster (modulo permutation).
        for trueIdx in 0..<3 {
            let assigned = (0..<groundTruth.count)
                .filter { groundTruth[$0] == trueIdx }
                .map { result.assignments[$0] }
            let unique = Set(assigned)
            #expect(unique.count == 1, "true cluster \(trueIdx) split across \(unique)")
        }

        // And different true clusters should get different recovered clusters.
        let recoveredFor = (0..<3).map { trueIdx in
            result.assignments[trueIdx * perCluster]
        }
        #expect(Set(recoveredFor).count == 3)
    }

    @Test("two runs with the same seed produce identical centroids")
    func deterministicGivenSeed() {
        var rng1 = SplitMix64(seed: 7)
        let data = Self.makeCluster(around: [1, 0, 0, 0], count: 40, jitter: 0.1, rng: &rng1)

        var rngA = SplitMix64(seed: 7)
        var rngB = SplitMix64(seed: 7)
        let a = KMeans.cluster(data: data, dims: 4, clusters: 5, maxIterations: 10, rng: &rngA)
        let b = KMeans.cluster(data: data, dims: 4, clusters: 5, maxIterations: 10, rng: &rngB)
        #expect(a.centroids == b.centroids)
        #expect(a.assignments == b.assignments)
    }

    @Test("assign() agrees with the trained model's own assignments")
    func assignConsistency() {
        var rng = SplitMix64(seed: 11)
        let data = Self.makeCluster(around: [1, 0, 0], count: 60, jitter: 0.15, rng: &rng)
        let result = KMeans.cluster(data: data, dims: 3, clusters: 5, maxIterations: 15, rng: &rng)
        let assigned = KMeans.assign(data: data, dims: 3, centroids: result.centroids)
        #expect(assigned == result.assignments)
    }

    @Test("k = m places each point in its own cluster")
    func kEqualsM() {
        var rng = SplitMix64(seed: 13)
        let data: [Float] = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]
        let result = KMeans.cluster(
            data: data, dims: 3, clusters: 3, maxIterations: 10, rng: &rng
        )
        #expect(Set(result.assignments).count == 3)
    }

    @Test("respects maxIterations = 1")
    func minimalIterations() {
        var rng = SplitMix64(seed: 17)
        let data = Self.makeCluster(around: [1, 0], count: 16, jitter: 0.1, rng: &rng)
        // Should not crash and should produce normalised centroids.
        let result = KMeans.cluster(
            data: data, dims: 2, clusters: 4, maxIterations: 1, rng: &rng
        )
        for c in 0..<4 {
            let slice = result.centroids[(c * 2)..<((c + 1) * 2)]
            #expect(abs(Self.l2Norm(slice) - 1.0) < 1e-5)
        }
    }
}
