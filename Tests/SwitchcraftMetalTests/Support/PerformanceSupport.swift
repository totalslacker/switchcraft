// SPDX-License-Identifier: Apache-2.0
//
// Performance-test helpers for the SwitchcraftMetal suites.
//
// **Duplicate of the release-build detection / nanoseconds / percentile
// helpers from `Tests/SwitchcraftTests/Performance/MetalPerformanceTests.swift`**
// per the issue #60 plan §"Tests + perf helpers duplicated, not extracted"
// and Q4 resolution. Refactor trigger: same as
// `MetalKernelTestSupport.swift` — extract a shared target if duplicated
// four-plus times after #61–#63 land. Until then, mirror any change.

#if canImport(Metal)

import Foundation

/// `true` when compiled with optimisations (`swift test -c release`).
/// Mirrors the `assert`-trick used in `Tests/SwitchcraftTests/Performance/`.
let isReleaseBuildMetal: Bool = {
    var debug = false
    assert({ debug = true; return true }())
    return !debug
}()

enum MetalPerf {
    /// Convert a `ContinuousClock.Duration` to an integer nanosecond
    /// count. Used as the unit in percentile sampling.
    static func nanoseconds(_ duration: ContinuousClock.Duration) -> UInt64 {
        let parts = duration.components
        let secNs = UInt64(max(parts.seconds, 0)) * 1_000_000_000
        let attoNs = UInt64(max(parts.attoseconds, 0)) / 1_000_000_000
        return secNs &+ attoNs
    }

    /// Index into a sorted (ascending) sample array for the requested
    /// percentile, using nearest-rank with ceiling. Mirrors ADR 012's
    /// percentile arithmetic used elsewhere in the project.
    static func percentileIndex(_ p: Double, count: Int) -> Int {
        precondition(count > 0, "percentileIndex requires count > 0")
        let raw = Int((p * Double(count)).rounded(.up)) - 1
        return min(max(raw, 0), count - 1)
    }
}

#endif // canImport(Metal)
