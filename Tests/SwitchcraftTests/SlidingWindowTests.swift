import Foundation
import Testing
@testable import SwitchcraftCoreML

@Suite("SlidingWindow planner & merge")
struct SlidingWindowTests {

    // MARK: - plan

    @Test("T <= windowSize uses a single window starting at 0")
    func planFitsInOneWindow() {
        #expect(SlidingWindow.plan(tokenCount: 0, windowSize: 512, stride: 256) == [0])
        #expect(SlidingWindow.plan(tokenCount: 1, windowSize: 512, stride: 256) == [0])
        #expect(SlidingWindow.plan(tokenCount: 200, windowSize: 512, stride: 256) == [0])
        #expect(SlidingWindow.plan(tokenCount: 512, windowSize: 512, stride: 256) == [0])
    }

    @Test("T just over windowSize: last window left-shifts to T-windowSize")
    func planLastWindowShiftedSmall() {
        // T=600 with stride 256: stride-aligned starts that fully fit are 0
        // (since 256 + 512 = 768 > 600). Last start must be 600 - 512 = 88.
        let starts = SlidingWindow.plan(tokenCount: 600, windowSize: 512, stride: 256)
        #expect(starts == [0, 88])
    }

    @Test("T = windowSize + stride aligns the second window without shift")
    func planNoShiftWhenAligned() {
        // T=768 with stride 256: starts 0 and 256 both fit (256 + 512 = 768).
        // Last start = 768 - 512 = 256, equal to the stride start, so no shift.
        let starts = SlidingWindow.plan(tokenCount: 768, windowSize: 512, stride: 256)
        #expect(starts == [0, 256])
    }

    @Test("T = 1024 produces three contiguous windows covering [0, 1024)")
    func planThreeWindows() {
        let starts = SlidingWindow.plan(tokenCount: 1024, windowSize: 512, stride: 256)
        // Stride-aligned fits: 0, 256, 512. Last start = 1024 - 512 = 512, matches.
        #expect(starts == [0, 256, 512])
        // Coverage: every position in [0, 1024) is in at least one window.
        for pos in 0..<1024 {
            #expect(starts.contains(where: { pos >= $0 && pos < $0 + 512 }))
        }
    }

    @Test("T larger than full stride pattern shifts the tail without dropping previous starts")
    func planTailShiftsWithoutCollision() {
        // T = 900: stride-aligned fits = 0, 256 (since 256+512=768 ≤ 900).
        // Next stride 512+512=1024 > 900, so loop stops at 256. Last start =
        // 900 - 512 = 388. Result: [0, 256, 388]. The tail shift introduces a
        // new start that doesn't collide with the previous (256).
        let starts = SlidingWindow.plan(tokenCount: 900, windowSize: 512, stride: 256)
        #expect(starts == [0, 256, 388])
        // Strictly increasing.
        for i in 1..<starts.count {
            #expect(starts[i] > starts[i - 1])
        }
    }

    @Test("plan with stride == windowSize disables overlap")
    func planNoOverlap() {
        let starts = SlidingWindow.plan(tokenCount: 1024, windowSize: 512, stride: 512)
        #expect(starts == [0, 512])
    }

    // MARK: - merge

    /// Build a single-window normalised buffer where row `i` is a
    /// unit vector along axis `i % dims`. Raw norms are `5.0` everywhere
    /// (well above MIN_NORM) so the filter keeps every position.
    private static func makeAxisAlignedWindow(
        windowSize: Int, dims: Int
    ) -> (normalised: [Float], rawNorms: [Float]) {
        var v = [Float](repeating: 0, count: windowSize * dims)
        let n = [Float](repeating: 5.0, count: windowSize)
        for row in 0..<windowSize {
            v[row * dims + (row % dims)] = 1
        }
        return (v, n)
    }

    @Test("merge with one window returns first T rows verbatim")
    func mergeSingleWindow() {
        let dims = 4
        let windowSize = 8
        let T = 5
        let (normalised, rawNorms) = Self.makeAxisAlignedWindow(
            windowSize: windowSize, dims: dims
        )
        let result = SlidingWindow.merge(
            windowStarts: [0],
            windowSize: windowSize,
            tokenCount: T,
            dims: dims,
            windowNormalised: [normalised],
            windowRawNorms: [rawNorms],
            minNorm: 1.0
        )
        #expect(result.keptPositions == [0, 1, 2, 3, 4])
        #expect(result.embeddings.count == T * dims)
        // Row 0 = [1, 0, 0, 0]; row 1 = [0, 1, 0, 0]; etc.
        for row in 0..<T {
            for d in 0..<dims {
                let v = result.embeddings[row * dims + d]
                let expected: Float = (row % dims == d) ? 1 : 0
                #expect(abs(v - expected) < 1e-6)
            }
        }
    }

    @Test("mean-merge of two windows in their overlap renormalises to unit length")
    func mergeMeanOverlapRenormalises() {
        let dims = 4
        let windowSize = 4
        // Two windows. Window 0 starts at 0, window 1 starts at 2. Overlap
        // is [2, 4).
        // Window 0 row 2 = [1, 0, 0, 0] (axis 0 unit)
        // Window 1 row 0 (= position 2) = [0, 1, 0, 0] (axis 1 unit)
        // Merged position 2 should be ([1, 1, 0, 0] / 2) renormalised to
        // unit length: [√½, √½, 0, 0].
        var w0 = [Float](repeating: 0, count: windowSize * dims)
        var w1 = [Float](repeating: 0, count: windowSize * dims)
        w0[0 * dims + 0] = 1 // pos 0 → axis 0
        w0[1 * dims + 1] = 1 // pos 1 → axis 1
        w0[2 * dims + 0] = 1 // pos 2 → axis 0 (overlap)
        w0[3 * dims + 1] = 1 // pos 3 → axis 1 (overlap)
        w1[0 * dims + 1] = 1 // pos 2 → axis 1 (overlap)
        w1[1 * dims + 0] = 1 // pos 3 → axis 0 (overlap)
        w1[2 * dims + 2] = 1 // pos 4 → axis 2
        w1[3 * dims + 3] = 1 // pos 5 → axis 3
        let n0 = [Float](repeating: 5.0, count: windowSize)
        let n1 = [Float](repeating: 5.0, count: windowSize)

        let result = SlidingWindow.merge(
            windowStarts: [0, 2],
            windowSize: windowSize,
            tokenCount: 6,
            dims: dims,
            windowNormalised: [w0, w1],
            windowRawNorms: [n0, n1],
            minNorm: 1.0
        )
        #expect(result.keptPositions == [0, 1, 2, 3, 4, 5])
        #expect(result.embeddings.count == 6 * dims)

        // Merged pos 2 = renormalise(mean([1,0,0,0], [0,1,0,0])) = [√½, √½, 0, 0].
        let halfSqrt = (0.5 as Float).squareRoot()
        let pos2 = Array(result.embeddings[(2 * dims)..<(3 * dims)])
        #expect(abs(pos2[0] - halfSqrt) < 1e-6)
        #expect(abs(pos2[1] - halfSqrt) < 1e-6)
        #expect(abs(pos2[2]) < 1e-6)
        #expect(abs(pos2[3]) < 1e-6)

        // Each kept row is unit length.
        for row in 0..<6 {
            var s: Float = 0
            for d in 0..<dims {
                let v = result.embeddings[row * dims + d]
                s += v * v
            }
            #expect(abs(s.squareRoot() - 1) < 1e-6)
        }
    }

    @Test("rows with mean raw norm < minNorm are dropped from output")
    func mergeNormFilterDropsLowSignal() {
        let dims = 2
        let windowSize = 4
        let T = 4
        // All rows are valid unit vectors but only positions 0 and 2 have
        // raw norm above 1.0; positions 1 and 3 have raw norm 0.27 (the
        // </s>/padding regime from Witchcraft).
        let normalised: [Float] = [
            1, 0,   // pos 0
            0, 1,   // pos 1 (will be filtered)
            1, 0,   // pos 2
            0, 1,   // pos 3 (will be filtered)
        ]
        let rawNorms: [Float] = [5.0, 0.27, 5.0, 0.27]

        let result = SlidingWindow.merge(
            windowStarts: [0],
            windowSize: windowSize,
            tokenCount: T,
            dims: dims,
            windowNormalised: [normalised],
            windowRawNorms: [rawNorms],
            minNorm: 1.0
        )
        #expect(result.keptPositions == [0, 2])
        #expect(result.embeddings == [1, 0, 1, 0])
    }

    @Test("empty input (T=0) returns no rows")
    func mergeEmpty() {
        let result = SlidingWindow.merge(
            windowStarts: [0],
            windowSize: 4,
            tokenCount: 0,
            dims: 2,
            windowNormalised: [[Float](repeating: 0, count: 8)],
            windowRawNorms: [[Float](repeating: 0, count: 4)],
            minNorm: 1.0
        )
        #expect(result.embeddings.isEmpty)
        #expect(result.keptPositions.isEmpty)
    }

    @Test("plan + merge round-trip on a 600-token left-shifted layout covers every position")
    func mergeRoundTripWithShiftedLastWindow() {
        let dims = 2
        let windowSize = 512
        let T = 600
        let starts = SlidingWindow.plan(
            tokenCount: T, windowSize: windowSize, stride: 256
        )
        #expect(starts == [0, T - windowSize])

        // Build per-window outputs where each position `pos` in [0, T) gets
        // the unit vector along axis (pos % dims). Both windows agree on
        // their overlap, so merging should recover identity.
        var windows: [[Float]] = []
        var rawNorms: [[Float]] = []
        for start in starts {
            var w = [Float](repeating: 0, count: windowSize * dims)
            var n = [Float](repeating: 0.27, count: windowSize)
            for localRow in 0..<windowSize {
                let pos = start + localRow
                if pos < T {
                    w[localRow * dims + (pos % dims)] = 1
                    n[localRow] = 5.0
                }
            }
            windows.append(w)
            rawNorms.append(n)
        }

        let result = SlidingWindow.merge(
            windowStarts: starts,
            windowSize: windowSize,
            tokenCount: T,
            dims: dims,
            windowNormalised: windows,
            windowRawNorms: rawNorms,
            minNorm: 1.0
        )
        #expect(result.keptPositions.count == T)
        #expect(result.embeddings.count == T * dims)
        for pos in 0..<T {
            let row = Array(result.embeddings[(pos * dims)..<((pos + 1) * dims)])
            for d in 0..<dims {
                let expected: Float = (pos % dims == d) ? 1 : 0
                #expect(abs(row[d] - expected) < 1e-6)
            }
        }
    }
}
