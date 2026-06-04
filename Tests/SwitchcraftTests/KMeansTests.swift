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

    /// Regression test for the vDSP_maxvi argmax path (issue #97).
    /// Uses a manually-verifiable 3-row × 3-centroid score matrix where
    /// each row's maximum is unambiguous: the expected assignments are
    /// [2, 0, 1] by inspection.
    @Test("assign() picks the maximum-score centroid per row (vDSP_maxvi regression)")
    func assignAllPicksMaximum() {
        // dims=3, m=3 rows, k=3 centroids
        // data rows are standard basis vectors; centroids are also standard
        // basis vectors (permuted), so dot products are identity-matrix-like.
        let dims = 3
        let data: [Float] = [
            0.0, 0.0, 1.0, // row 0: nearest to centroid 2 (dot=1 with c2)
            1.0, 0.0, 0.0, // row 1: nearest to centroid 0 (dot=1 with c0)
            0.0, 1.0, 0.0, // row 2: nearest to centroid 1 (dot=1 with c1)
        ]
        let centroids: [Float] = [
            1.0, 0.0, 0.0, // centroid 0: x-axis
            0.0, 1.0, 0.0, // centroid 1: y-axis
            0.0, 0.0, 1.0, // centroid 2: z-axis
        ]
        let assignments = KMeans.assign(data: data, dims: dims, centroids: centroids)
        #expect(assignments == [2, 0, 1])
    }

    /// Verifies vDSP_maxvi tie-break semantics: when two centroids produce
    /// equal dot products, the lower-index centroid wins (first-occurrence).
    /// This locks in deterministic behavior across Accelerate/OS versions and
    /// matches the prior plain-Swift left-to-right argmax loop.
    @Test("assign() picks the first maximum-score centroid on a tie (vDSP_maxvi tie-break)")
    func assignTieBreakPicksFirst() {
        // dims=2, m=1 row, k=3 centroids.
        // data=[1,0]; centroids c0=[1,0], c1=[0,1], c2=[1,0].
        // dot products: c0→1.0, c1→0.0, c2→1.0.
        // Tie between c0 and c2 — first occurrence (index 0) must win.
        let dims = 2
        let data: [Float] = [1.0, 0.0]
        let centroids: [Float] = [
            1.0, 0.0, // centroid 0 — tied for max
            0.0, 1.0, // centroid 1 — lower score
            1.0, 0.0, // centroid 2 — tied for max (but higher index)
        ]
        let assignments = KMeans.assign(data: data, dims: dims, centroids: centroids)
        #expect(assignments == [0], "tie should resolve to the first maximum (index 0), got \(assignments)")
    }

    // MARK: - Tiling regression (ADR 027 / issue #112)

    /// Reference argmax: naive per-row dot-product loop, no BLAS. Used to
    /// cross-check the tiled SGEMM path for correctness.
    static func naiveAssign(data: [Float], dims: Int, centroids: [Float]) -> [Int] {
        let m = data.count / dims
        let k = centroids.count / dims
        return (0..<m).map { row in
            var bestIdx = 0
            var bestDot = Float.leastNormalMagnitude
            for c in 0..<k {
                var dot: Float = 0
                for d in 0..<dims {
                    dot += data[row * dims + d] * centroids[c * dims + d]
                }
                if dot > bestDot { bestDot = dot; bestIdx = c }
            }
            return bestIdx
        }
    }

    /// Tiling parity for m = assignBatchSize + 1 (4 097): forces two SGEMM
    /// calls (one full tile + one row). Verifies per-row argmax is identical
    /// to the naive reference on orthogonal-centroid data.
    @Test("assign() tiling: m = B+1 matches naive argmax (issue #112)")
    func tiledAssignOnePlusOneBatch() {
        let dims = 8
        // k orthogonal unit basis vectors (first k standard-basis directions).
        let k = dims
        var centroids = [Float](repeating: 0, count: k * dims)
        for c in 0..<k { centroids[c * dims + c] = 1.0 }

        // 4 097 rows: each row's strongest component is at index i % k.
        let m = 4_097
        var rng = SplitMix64(seed: 0xAB_CD_EF)
        var data = [Float]()
        data.reserveCapacity(m * dims)
        for i in 0..<m {
            let dominant = i % k
            var v = [Float](repeating: 0, count: dims)
            v[dominant] = 0.95
            for j in 0..<dims where j != dominant {
                v[j] = Float(rng.next() & 0xFFFF) / Float(1 << 16) * 0.05
            }
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            for j in 0..<dims { v[j] /= norm }
            data.append(contentsOf: v)
        }

        let got = KMeans.assign(data: data, dims: dims, centroids: centroids)
        let ref = Self.naiveAssign(data: data, dims: dims, centroids: centroids)
        #expect(got == ref, "tiled assign() diverged from naive reference at m=\(m)")
    }

    /// Tiling parity for m = 2 × assignBatchSize + 100 (8 292): forces three
    /// SGEMM calls. Same correctness check as above.
    @Test("assign() tiling: m = 2B+100 matches naive argmax (issue #112)")
    func tiledAssignThreeBatches() {
        let dims = 4
        let k = 4
        var centroids = [Float](repeating: 0, count: k * dims)
        for c in 0..<k { centroids[c * dims + c] = 1.0 }

        let m = 4_096 * 2 + 100
        var rng = SplitMix64(seed: 0xDEAD_BEEF)
        var data = [Float]()
        data.reserveCapacity(m * dims)
        for i in 0..<m {
            let dominant = i % k
            var v = [Float](repeating: 0, count: dims)
            v[dominant] = 0.9
            for j in 0..<dims where j != dominant {
                v[j] = Float(rng.next() & 0xFF) / Float(1 << 8) * 0.1
            }
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            for j in 0..<dims { v[j] /= norm }
            data.append(contentsOf: v)
        }

        let got = KMeans.assign(data: data, dims: dims, centroids: centroids)
        let ref = Self.naiveAssign(data: data, dims: dims, centroids: centroids)
        #expect(got == ref, "tiled assign() diverged from naive reference at m=\(m)")
    }

    /// Edge case: m = 1 (single row — always one tile, trivially correct).
    @Test("assign() tiling: m=1 edge case (issue #112)")
    func tiledAssignSingleRow() {
        let dims = 3
        let data: [Float] = [0, 0, 1]
        let centroids: [Float] = [1, 0, 0,   0, 1, 0,   0, 0, 1]
        let got = KMeans.assign(data: data, dims: dims, centroids: centroids)
        #expect(got == [2])
    }

    /// Edge case: m = assignBatchSize exactly (exactly one full tile).
    @Test("assign() tiling: m = B exactly (issue #112)")
    func tiledAssignExactlyOneBatch() {
        let dims = 2
        let centroids: [Float] = [1, 0,   0, 1]
        let m = 4_096
        var data = [Float]()
        data.reserveCapacity(m * dims)
        // Odd rows → centroid 0, even rows → centroid 1.
        for i in 0..<m {
            if i % 2 == 0 { data.append(contentsOf: [0.99, 0.14]) }
            else          { data.append(contentsOf: [0.14, 0.99]) }
        }
        // Normalise.
        for i in 0..<m {
            let norm = sqrt(data[i*2]*data[i*2] + data[i*2+1]*data[i*2+1])
            data[i*2] /= norm; data[i*2+1] /= norm
        }
        let got = KMeans.assign(data: data, dims: dims, centroids: centroids)
        let expected = (0..<m).map { $0 % 2 == 0 ? 0 : 1 }
        #expect(got == expected)
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
