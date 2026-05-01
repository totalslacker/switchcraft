// SPDX-License-Identifier: Apache-2.0
//
// Benchmark suite for issue #49. Times Accelerate (`cblas_sgemm` via the
// production `SearchEngine` path), `MPSMatrixMultiplication`, and the custom
// Metal kernel on the four T5-base shapes. Emits a `BenchmarkRun` JSON dump
// to `docs/investigations/metal-matmul-feasibility-figures/results.json` (or
// the path supplied via `SWITCHCRAFT_METAL_PROTO_RESULTS_DIR`) so the
// feasibility report can be regenerated from committed code + seed.
//
// Gate: `SWITCHCRAFT_METAL_PROTO_BENCH=1` AND release build. The release-mode
// gate matches the existing `PerformanceTests` policy — debug builds are too
// noisy to defend a latency floor against, and gating prevents accidental
// CI runs that would consume GPU minutes.

import Foundation
import Testing
@testable import SwitchcraftCore
import SwitchcraftMetalProto

@Suite("Matmul benchmark", .serialized,
       .enabled(if: MetalProtoGate.benchmarkEnabled,
                "Set SWITCHCRAFT_METAL_PROTO_BENCH=1 and run with `swift test -c release` to benchmark"))
struct MatmulBenchmarkTests {

    static let warmup: Int = 10
    static let iterations: Int = 100
    /// Default tile used for the headline custom-kernel numbers. The
    /// threadgroup-sweep test (`ThreadgroupSweepTests`) is the source of
    /// truth for which tile is fastest; this default lines up with the
    /// observed best on M1 / M1 Ultra at the time of writing.
    static let defaultTile = MetalTile(16, 16)

    @Test("benchmark all backends across all T5-base shapes")
    func benchmarkAll() throws {
        var results: [BenchmarkResult] = []

        let metal = try MetalMatmul()
        let mps = try MPSMatmul()

        // Pre-compile the custom-kernel pipeline so the first benchmark
        // iteration doesn't pay the function-constant + pipeline-creation cost.
        _ = try metal.pipeline(for: Self.defaultTile)

        for shape in T5BaseShapes.all {
            let (a, b) = BenchmarkInputs.makeAB(for: shape)
            let bT = AccelerateMatmul.transposeB(b, k: shape.k, n: shape.n)

            // Accelerate (cblas_sgemm via SearchEngine).
            results.append(BenchmarkHarness.measure(
                backend: "accelerate_cblas_sgemm",
                shape: shape.name,
                m: shape.m, k: shape.k, n: shape.n,
                iterations: Self.iterations,
                warmup: Self.warmup
            ) {
                _ = AccelerateMatmul.matmulPretransposed(
                    a: a, m: shape.m, k: shape.k, bT: bT, n: shape.n
                )
            })

            // MPSMatrixMultiplication.
            results.append(try BenchmarkHarness.measure(
                backend: "mps_matrix_multiplication",
                shape: shape.name,
                m: shape.m, k: shape.k, n: shape.n,
                iterations: Self.iterations,
                warmup: Self.warmup
            ) {
                _ = try mps.matmul(
                    a: a, aShape: (shape.m, shape.k),
                    b: b, bShape: (shape.k, shape.n)
                )
            })

            // Custom Metal kernel (default tile).
            results.append(try BenchmarkHarness.measure(
                backend: "custom_metal_fp32",
                shape: shape.name,
                m: shape.m, k: shape.k, n: shape.n,
                iterations: Self.iterations,
                warmup: Self.warmup,
                tile: Self.defaultTile
            ) {
                _ = try metal.matmul(
                    a: a, aShape: (shape.m, shape.k),
                    b: b, bShape: (shape.k, shape.n),
                    tile: Self.defaultTile
                )
            })
        }

        try Self.writeResults(results, fileName: "results.json")

        // Print a one-line summary per row so the test output is informative
        // even without inspecting the JSON file.
        for r in results {
            let tile = r.tile.map { " tile=\($0.tgM)×\($0.tgN)" } ?? ""
            print(String(format: "  %@ %@%@: p50=%.3f ms p95=%.3f ms mean=%.3f ms (%.1f GFLOPS)",
                         r.backend, r.shape, tile,
                         r.p50Millis, r.p95Millis, r.meanMillis, r.gflops))
        }
    }

    static func writeResults(_ results: [BenchmarkResult], fileName: String) throws {
        let dir = MetalProtoGate.resultsDirOverride
            ?? defaultFiguresDir()
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true
        )
        let url = dir.appendingPathComponent(fileName)

        let run = BenchmarkRun(
            hardware: detectHardware(),
            runDate: ISO8601DateFormatter().string(from: Date()),
            results: results
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(run)
        try data.write(to: url)
        print("benchmark results written to \(url.path)")
    }

    private static func defaultFiguresDir() -> URL {
        // Walk up from the build dir to the repo root, then into the figures
        // path. The test's working dir is the package root under SwiftPM's
        // standard layout, so `FileManager.default.currentDirectoryPath` is
        // sufficient for the common case.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return cwd
            .appendingPathComponent("docs")
            .appendingPathComponent("investigations")
            .appendingPathComponent("metal-matmul-feasibility-figures")
    }

    static func detectHardware() -> BenchmarkRun.HardwareInfo {
        let chip = sysctlString("machdep.cpu.brand_string") ?? "unknown"
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        let physMem = ProcessInfo.processInfo.physicalMemory
        let memGB = Int(physMem / (1024 * 1024 * 1024))
        return BenchmarkRun.HardwareInfo(
            chip: chip, osVersion: osVersion, memoryGB: memGB
        )
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: Int = 0
        sysctlbyname(name, nil, &size, nil, 0)
        guard size > 0 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        let rc = sysctlbyname(name, &bytes, &size, nil, 0)
        guard rc == 0 else { return nil }
        let unsigned = bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: unsigned, as: UTF8.self)
    }
}
