// SPDX-License-Identifier: Apache-2.0
//
// Threadgroup-tile sweep on the largest T5-base shape (`ffn_up`,
// `[512, 768] × [768, 3072]`). Emits a `threadgroup-sweep.json` next to
// `results.json` so the feasibility report can show which tile shape the
// custom kernel does best on. Same gating as the main benchmark suite.

import Foundation
import Testing
@testable import SwitchcraftCore
import SwitchcraftMetalProto

@Suite("Threadgroup sweep", .serialized,
       .enabled(if: MetalProtoGate.benchmarkEnabled,
                "Set SWITCHCRAFT_METAL_PROTO_BENCH=1 and run with `swift test -c release` to benchmark"))
struct ThreadgroupSweepTests {

    static let warmup: Int = 5
    static let iterations: Int = 50

    @Test("sweep all tile shapes on the largest T5 shape")
    func sweepLargestShape() throws {
        let metal = try MetalMatmul()
        let shape = T5BaseShapes.largest
        let (a, b) = BenchmarkInputs.makeAB(for: shape)

        var results: [BenchmarkResult] = []
        for tile in ThreadgroupSweep.tiles {
            // Pre-build pipeline so iteration timing is steady-state only.
            _ = try metal.pipeline(for: tile)

            let result = try BenchmarkHarness.measure(
                backend: "custom_metal_fp32",
                shape: shape.name,
                m: shape.m, k: shape.k, n: shape.n,
                iterations: Self.iterations,
                warmup: Self.warmup,
                tile: tile
            ) {
                _ = try metal.matmul(
                    a: a, aShape: (shape.m, shape.k),
                    b: b, bShape: (shape.k, shape.n),
                    tile: tile
                )
            }
            results.append(result)
            print(String(format: "  tile %d×%d on %@: p50=%.3f ms p95=%.3f ms (%.1f GFLOPS)",
                         tile.tgM, tile.tgN, shape.name,
                         result.p50Millis, result.p95Millis, result.gflops))
        }

        try MatmulBenchmarkTests.writeResults(results, fileName: "threadgroup-sweep.json")
    }
}
