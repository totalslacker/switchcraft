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

    // MARK: - Cross-stack reference parity (issue #28 / ADR 013)

    /// Validate that the committed Witchcraft reference centroids are
    /// well-formed: shape matches the JSON index, and every centroid is
    /// L2-normalised within FP tolerance (Witchcraft writes
    /// L2-normalised centroids per its kmeans definition).
    ///
    /// Skips when `reference_centroids.{bin,json}` are not present in
    /// the test bundle (fresh checkout where fixtures haven't been
    /// regenerated yet — see `adrs/013-reference-fixture-provenance.md`
    /// and `scripts/witchcraft-fixture-export.patch`).
    @Test(
        "reference centroids fixture is well-formed (L2-normalised, dims-consistent)",
        .enabled(if: ReferenceCentroidsFixture.isAvailable)
    )
    func referenceCentroidsStructure() throws {
        let index = try ReferenceCentroidsFixture.loadIndex()
        let blob = try ReferenceCentroidsFixture.loadBlob()

        let centroids = ReferenceCentroidsFixture.centroids(index: index, blob: blob)
        let dims = index.dims
        guard let clusters = index.centroids.clusters else {
            Issue.record("centroids region missing clusters count")
            return
        }
        #expect(centroids.count == clusters * dims)

        for c in 0..<clusters {
            let slice = centroids[(c * dims)..<((c + 1) * dims)]
            let norm = Self.l2Norm(slice)
            #expect(
                abs(norm - 1.0) < 1e-3,
                "centroid \(c) not L2-normalised: ‖·‖ = \(norm)"
            )
        }
    }

    /// Cross-implementation k-means parity: re-run Swift KMeans on the
    /// SAME float inputs Witchcraft fed to its kmeans, and verify each
    /// Swift centroid finds a UNIQUE near-match in the reference centroid
    /// set (one-to-one assignment, no reuse).
    ///
    /// We compare via greedy one-to-one matching rather than a fixed
    /// permutation because k-means cluster ordering is implementation-
    /// defined. Forbidding reuse catches collapsed/duplicate Swift
    /// centroids that a many-to-one match would miss. Threshold is 0.99
    /// (per the plan-stage risk note: cluster count for the 33-fact
    /// corpus is small enough that Lloyd's iteration may converge to
    /// different local minima across Swift and Rust). ADR 013 documents
    /// the chosen tolerance.
    @Test(
        "Swift k-means on the reference inputs matches Witchcraft centroids (cosine ≥ 0.99)",
        .enabled(if: ReferenceCentroidsFixture.isAvailable)
    )
    func referenceCentroidsParity() throws {
        let index = try ReferenceCentroidsFixture.loadIndex()
        let blob = try ReferenceCentroidsFixture.loadBlob()

        let dims = index.dims
        guard
            let inputRows = index.inputs.rows,
            let clusters = index.centroids.clusters
        else {
            Issue.record("centroids fixture missing rows/clusters metadata")
            return
        }
        let inputs = ReferenceCentroidsFixture.inputs(index: index, blob: blob)
        let referenceCentroids = ReferenceCentroidsFixture.centroids(
            index: index, blob: blob
        )
        #expect(inputs.count == inputRows * dims)
        #expect(referenceCentroids.count == clusters * dims)

        var rng = SplitMix64(seed: 0xC02_8C0DE_CAFE)
        let result = KMeans.cluster(
            data: inputs,
            dims: dims,
            clusters: clusters,
            maxIterations: 25,
            rng: &rng
        )

        // Build the full cosine matrix between Swift centroids (rows) and
        // reference centroids (cols). Inputs are pre-normalised
        // (Witchcraft's kmeans expects L2-normalised vectors) so cosine
        // == dot.
        var cosine = [Float](repeating: 0, count: clusters * clusters)
        for s in 0..<clusters {
            for r in 0..<clusters {
                var dot: Float = 0
                for d in 0..<dims {
                    dot += result.centroids[s * dims + d]
                        * referenceCentroids[r * dims + d]
                }
                cosine[s * clusters + r] = dot
            }
        }

        // Greedy one-to-one assignment: repeatedly pick the highest
        // remaining (Swift, reference) pair, lock both out, and record
        // the cosine. After `clusters` rounds every Swift centroid is
        // matched to a distinct reference centroid. The minimum matched
        // cosine is the parity statistic.
        var swiftUsed = [Bool](repeating: false, count: clusters)
        var refUsed = [Bool](repeating: false, count: clusters)
        var minMatchedCosine: Float = 1.0
        for _ in 0..<clusters {
            var best: Float = -.infinity
            var bestS = -1
            var bestR = -1
            for s in 0..<clusters where !swiftUsed[s] {
                for r in 0..<clusters where !refUsed[r] {
                    let c = cosine[s * clusters + r]
                    if c > best {
                        best = c
                        bestS = s
                        bestR = r
                    }
                }
            }
            #expect(bestS >= 0 && bestR >= 0, "greedy matching exhausted before all clusters paired")
            swiftUsed[bestS] = true
            refUsed[bestR] = true
            if best < minMatchedCosine { minMatchedCosine = best }
        }
        #expect(
            minMatchedCosine >= 0.99,
            "minimum one-to-one matched cosine = \(minMatchedCosine) < 0.99"
        )
    }
}
