// SPDX-License-Identifier: Apache-2.0
//
// Wall-clock benchmark harness for the issue #49 prototype. Uses
// `mach_absolute_time` so timings reflect the same clock the OS uses for
// scheduling and don't include CPU-time accounting overhead. Reports p50,
// p95, mean, and min latency over `iterations` runs (after `warmup` warm-up
// runs that aren't recorded).
//
// `BenchmarkResult` is `Codable` so the suite can dump the full table to
// JSON under `docs/investigations/metal-matmul-feasibility-figures/` for
// commit.

#if canImport(Darwin)

import Foundation
import Darwin.Mach

/// One row of the latency table.
public struct BenchmarkResult: Codable, Sendable {
    public let backend: String
    public let shape: String
    public let m: Int
    public let k: Int
    public let n: Int
    public let iterations: Int
    public let warmup: Int
    public let p50Nanos: UInt64
    public let p95Nanos: UInt64
    public let meanNanos: UInt64
    public let minNanos: UInt64
    /// Optional tile shape for the custom kernel. `nil` for cblas / MPS.
    public let tile: MetalTile?
    public let gflops: Double

    /// Convenience formatter for human-readable rendering in the report.
    public var p50Millis: Double { Double(p50Nanos) / 1_000_000.0 }
    public var p95Millis: Double { Double(p95Nanos) / 1_000_000.0 }
    public var meanMillis: Double { Double(meanNanos) / 1_000_000.0 }
}

public enum BenchmarkHarness {
    /// Mach-time conversion factor (nanoseconds-per-tick). Cached for the
    /// process lifetime; the first call costs one syscall.
    private static let timebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t()
        mach_timebase_info(&t)
        return t
    }()

    /// Run `body` `iterations + warmup` times, discarding the warmup samples,
    /// and aggregate p50/p95/mean/min in nanoseconds.
    public static func measure(
        backend: String,
        shape: String,
        m: Int, k: Int, n: Int,
        iterations: Int,
        warmup: Int,
        tile: MetalTile? = nil,
        body: () throws -> Void
    ) rethrows -> BenchmarkResult {
        for _ in 0..<warmup { try body() }
        var samples = [UInt64]()
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = mach_absolute_time()
            try body()
            let end = mach_absolute_time()
            samples.append(end - start)
        }
        let nanos = samples.map { ticksToNanos($0) }.sorted()
        let p50 = nanos[nanos.count / 2]
        let p95 = nanos[Int(Double(nanos.count) * 0.95)]
        let mean = UInt64(nanos.reduce(0, +) / UInt64(nanos.count))
        let minV = nanos.first ?? 0
        let flops = 2.0 * Double(m) * Double(k) * Double(n)
        // Use min-latency for GFLOPS so the number reflects best-case throughput
        // rather than average; matches how MPS's own published numbers are quoted.
        let gflops = flops / (Double(minV) / 1.0) // flops / nanos = Gflops
        return BenchmarkResult(
            backend: backend,
            shape: shape,
            m: m, k: k, n: n,
            iterations: iterations,
            warmup: warmup,
            p50Nanos: p50,
            p95Nanos: p95,
            meanNanos: mean,
            minNanos: minV,
            tile: tile,
            gflops: gflops
        )
    }

    private static func ticksToNanos(_ ticks: UInt64) -> UInt64 {
        ticks &* UInt64(timebase.numer) / UInt64(timebase.denom)
    }
}

/// Codable container for the full benchmark output (committed to figures dir).
public struct BenchmarkRun: Codable, Sendable {
    public let hardware: HardwareInfo
    public let runDate: String
    public let results: [BenchmarkResult]

    public init(hardware: HardwareInfo, runDate: String, results: [BenchmarkResult]) {
        self.hardware = hardware
        self.runDate = runDate
        self.results = results
    }

    public struct HardwareInfo: Codable, Sendable {
        public let chip: String
        public let osVersion: String
        public let memoryGB: Int?

        public init(chip: String, osVersion: String, memoryGB: Int?) {
            self.chip = chip
            self.osVersion = osVersion
            self.memoryGB = memoryGB
        }
    }
}

#endif
