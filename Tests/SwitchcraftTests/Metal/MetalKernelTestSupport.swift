// SPDX-License-Identifier: Apache-2.0
//
// Test support for the Phase 2 search-path Metal kernels (umbrella #50).
// Mirrors `Tests/SwitchcraftTests/Support/CoreMLAsset.swift` in spirit:
// a single `isAvailable` computed property that gates the Metal suites
// via Swift Testing's `.enabled(if:)` trait, so a host without Metal
// (or a `SWITCHCRAFT_FORCE_ACCELERATE`-overridden test run) skips the
// suites cleanly without `XCTSkip` noise.
//
// Subsequent kernel sub-issues add reusable correctness assertions and
// fixture builders here. The scaffolding sub-issue ships `isAvailable`,
// `cosineSimilarity`, and `maxAbsError` — the minimum viable surface
// for sub-issue #2 to drop in its first kernel test.

#if canImport(Metal)

import Foundation
import Metal
@testable import SwitchcraftCore

/// Probes the host for Metal support. Used as `.enabled(if: …)` on the
/// Metal test suites so they skip cleanly on no-GPU hosts and on test
/// runs that have set `SWITCHCRAFT_FORCE_ACCELERATE` to exercise the
/// fallback path.
enum MetalAvailability {
    /// `true` iff `MTLCreateSystemDefaultDevice()` returns a device and
    /// `SWITCHCRAFT_FORCE_ACCELERATE` is unset (or set to `0`). The env
    /// var name is owned by `MetalContext.forceAccelerateEnvVar` so the
    /// test gate cannot drift out of lockstep with the production check.
    static var isAvailable: Bool {
        if isForceAccelerateSet() { return false }
        return MTLCreateSystemDefaultDevice() != nil
    }

    private static func isForceAccelerateSet() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[
            MetalContext.forceAccelerateEnvVar
        ] else {
            return false
        }
        return !raw.isEmpty && raw != "0"
    }
}

/// Reusable correctness helpers for kernel tests. Assertions live in the
/// caller's `#expect` so test failures point at the kernel-specific
/// callsite rather than at this support file.
enum MetalCorrectness {
    /// Cosine similarity between two equal-length float vectors.
    /// Returns 1.0 for identical-direction vectors, 0.0 for orthogonal,
    /// -1.0 for opposite. Returns 1.0 when both vectors are zero
    /// (degenerate case where cosine is undefined; treated as "equal").
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count,
                     "cosineSimilarity inputs must be equal length")
        var dot: Double = 0, nA: Double = 0, nB: Double = 0
        for i in 0..<a.count {
            let ai = Double(a[i]), bi = Double(b[i])
            dot += ai * bi
            nA += ai * ai
            nB += bi * bi
        }
        let denom = nA.squareRoot() * nB.squareRoot()
        return denom > 0 ? dot / denom : 1.0
    }

    /// Largest absolute element-wise error between two equal-length
    /// float vectors. Used alongside cosine similarity because kernels
    /// can pass cosine on a near-zero vector while still producing
    /// catastrophic outliers.
    static func maxAbsError(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count,
                     "maxAbsError inputs must be equal length")
        var maxErr: Double = 0
        for i in 0..<a.count {
            let d = abs(Double(a[i]) - Double(b[i]))
            if d > maxErr { maxErr = d }
        }
        return maxErr
    }
}

#endif // canImport(Metal)
