// SPDX-License-Identifier: Apache-2.0
//
// Issue #49 acceptance criterion: custom-Metal kernel must reach
// ≥0.99999 cosine similarity vs the production `cblas_sgemm` path on all
// four T5-base shapes. Gated on `SWITCHCRAFT_METAL_PROTO=1` so the default
// `swift test` invocation skips the suite cleanly.

import Foundation
import Testing
@testable import SwitchcraftCore
import SwitchcraftMetalProto

@Suite("MetalMatmul correctness",
       .enabled(if: MetalProtoGate.correctnessEnabled,
                "Set SWITCHCRAFT_METAL_PROTO=1 to run the Metal matmul prototype tests"))
struct MetalMatmulCorrectnessTests {

    @Test("custom kernel matches cblas within 0.99999 cosine on all four T5 shapes",
          arguments: T5BaseShapes.all)
    func customKernelMatchesCblas(shape: MatmulShape) throws {
        let kernel = try MetalMatmul()
        let (a, b) = BenchmarkInputs.makeAB(for: shape)

        let expected = AccelerateMatmul.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n)
        )
        let actual = try kernel.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n),
            tile: MetalTile(16, 16)
        )

        #expect(actual.count == expected.count, "shape \(shape.name): output size mismatch")
        let cosine = CosineSimilarity.compute(actual, expected)
        #expect(cosine >= 0.99999,
                "shape \(shape.name): cosine \(cosine) < 0.99999")
    }

    /// Smoke test: each tile shape in the threadgroup-sweep produces results
    /// that match cblas. If a tile shape is broken (e.g. (8,4) hits an
    /// indexing bug) we want the sweep test to flag it before the benchmark
    /// happily reports nonsense numbers.
    @Test("every threadgroup tile produces correct output on the smallest shape",
          arguments: ThreadgroupSweep.tiles)
    func everyTileIsCorrect(tile: MetalTile) throws {
        let kernel = try MetalMatmul()
        let shape = T5BaseShapes.tokenProj // smallest: M·K·N keeps this fast
        let (a, b) = BenchmarkInputs.makeAB(for: shape)

        let expected = AccelerateMatmul.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n)
        )
        let actual = try kernel.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n),
            tile: tile
        )
        let cosine = CosineSimilarity.compute(actual, expected)
        #expect(cosine >= 0.99999,
                "tile \(tile.tgM)×\(tile.tgN): cosine \(cosine) < 0.99999")
    }
}

@Suite("MPSMatmul correctness",
       .enabled(if: MetalProtoGate.correctnessEnabled,
                "Set SWITCHCRAFT_METAL_PROTO=1 to run the Metal matmul prototype tests"))
struct MPSMatmulCorrectnessTests {

    @Test("MPS matches cblas within 0.99999 cosine on all four T5 shapes",
          arguments: T5BaseShapes.all)
    func mpsMatchesCblas(shape: MatmulShape) throws {
        let kernel = try MPSMatmul()
        let (a, b) = BenchmarkInputs.makeAB(for: shape)

        let expected = AccelerateMatmul.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n)
        )
        let actual = try kernel.matmul(
            a: a, aShape: (shape.m, shape.k),
            b: b, bShape: (shape.k, shape.n)
        )

        #expect(actual.count == expected.count, "shape \(shape.name): output size mismatch")
        let cosine = CosineSimilarity.compute(actual, expected)
        #expect(cosine >= 0.99999,
                "shape \(shape.name): cosine \(cosine) < 0.99999")
    }
}
