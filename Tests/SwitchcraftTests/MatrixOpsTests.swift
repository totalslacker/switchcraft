import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("Matrix Operations")
struct MatrixOpsTests {

    // MARK: - Helpers

    /// Naive triple-loop reference implementation of `query × rowsᵀ`.
    /// Produces an `(n × rowCount)` row-major matrix.
    static func naiveMatmulQTimesRowsT(
        query: [Float],
        n: Int,
        rows: [Float],
        rowCount: Int,
        dims: Int
    ) -> [Float] {
        precondition(query.count == n * dims)
        precondition(rows.count == rowCount * dims)
        var out = [Float](repeating: 0, count: n * rowCount)
        for i in 0..<n {
            for j in 0..<rowCount {
                var acc: Float = 0
                for d in 0..<dims {
                    acc += query[i * dims + d] * rows[j * dims + d]
                }
                out[i * rowCount + j] = acc
            }
        }
        return out
    }

    /// Build a unit-norm random row of length `dims` using the supplied
    /// seeded RNG, so test inputs are fully reproducible.
    static func randomUnitRow(dims: Int, rng: inout SplitMix64) -> [Float] {
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

    /// Build `count` concatenated unit-norm rows of length `dims`.
    static func randomUnitRows(count: Int, dims: Int, rng: inout SplitMix64) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(count * dims)
        for _ in 0..<count {
            out.append(contentsOf: randomUnitRow(dims: dims, rng: &rng))
        }
        return out
    }

    /// Element-wise absolute-tolerance comparison.
    static func expectClose(
        _ actual: [Float],
        _ expected: [Float],
        tolerance: Float = 1e-5
    ) {
        #expect(actual.count == expected.count)
        for i in 0..<min(actual.count, expected.count) {
            let diff = abs(actual[i] - expected[i])
            #expect(diff < tolerance, "index \(i): |\(actual[i]) - \(expected[i])| = \(diff) >= \(tolerance)")
        }
    }

    // MARK: - Tests

    @Test("matmulQueryTimesRowMajorTranspose correctness within 1e-5")
    func matmulCorrectness() {
        let n = 4
        let rowCount = 8
        let dims = 64
        var rng = SplitMix64(seed: 0xA1B2_C3D4_E5F6_0789)

        let query = Self.randomUnitRows(count: n, dims: dims, rng: &rng)
        let rows = Self.randomUnitRows(count: rowCount, dims: dims, rng: &rng)

        let actual = SearchEngine.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: query, n: n,
            rows: rows, rowCount: rowCount, dims: dims
        )
        let expected = Self.naiveMatmulQTimesRowsT(
            query: query, n: n,
            rows: rows, rowCount: rowCount, dims: dims
        )

        Self.expectClose(actual, expected, tolerance: 1e-5)
    }

    @Test("KMeans.assign batched matches single-row evaluation")
    func batchedAssignMatchesSingleRow() {
        let m = 16
        let dims = 32
        let k = 8
        var rng = SplitMix64(seed: 0xDEAD_BEEF_CAFE_F00D)

        let data = Self.randomUnitRows(count: m, dims: dims, rng: &rng)
        let centroids = Self.randomUnitRows(count: k, dims: dims, rng: &rng)

        let batched = KMeans.assign(data: data, dims: dims, centroids: centroids)
        #expect(batched.count == m)

        for i in 0..<m {
            let row = Array(data[(i * dims)..<((i + 1) * dims)])
            let single = KMeans.assign(data: row, dims: dims, centroids: centroids)
            #expect(single.count == 1)
            #expect(single[0] == batched[i], "row \(i): batched=\(batched[i]) single=\(single[0])")
        }
    }

    @Test("matmulQueryTimesRowMajorTranspose matches naive on Q4-decoded inputs within 1e-5")
    func matmulOnQ4DecodedInputs() {
        let n = 3
        let rowCount = 6
        let dims = 32
        var rng = SplitMix64(seed: 0x0123_4567_89AB_CDEF)

        // Generate a residual buffer (typical residual range), even count
        // so it satisfies Q4Codec.encodeResiduals' precondition.
        let totalQuery = n * dims
        let totalRows = rowCount * dims
        precondition(totalQuery % 2 == 0 && totalRows % 2 == 0)

        func makeResiduals(_ count: Int) -> [Float] {
            var v = [Float](repeating: 0, count: count)
            for i in 0..<count {
                let u = Float(rng.next()) / Float(UInt64.max) // [0, 1]
                v[i] = (u - 0.5) * 0.5 // ~[-0.25, 0.25]
            }
            return v
        }

        let queryResiduals = makeResiduals(totalQuery)
        let rowResiduals = makeResiduals(totalRows)

        let queryDecoded = Q4Codec.decodeResiduals(Q4Codec.encodeResiduals(queryResiduals))
        let rowsDecoded = Q4Codec.decodeResiduals(Q4Codec.encodeResiduals(rowResiduals))

        #expect(queryDecoded.count == totalQuery)
        #expect(rowsDecoded.count == totalRows)

        let actual = SearchEngine.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: queryDecoded, n: n,
            rows: rowsDecoded, rowCount: rowCount, dims: dims
        )
        let expected = Self.naiveMatmulQTimesRowsT(
            query: queryDecoded, n: n,
            rows: rowsDecoded, rowCount: rowCount, dims: dims
        )

        Self.expectClose(actual, expected, tolerance: 1e-5)
    }
}
