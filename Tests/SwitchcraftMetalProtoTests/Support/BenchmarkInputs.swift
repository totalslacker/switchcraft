// SPDX-License-Identifier: Apache-2.0
//
// Reproducible matrix generators for the prototype's correctness and
// benchmark suites. Uses `SwitchcraftCore.SplitMix64` (the same PRNG the
// production matrix-ops tests use) with fixed per-shape seeds so the same
// inputs are produced across runs and machines.

import Foundation
@testable import SwitchcraftCore
import SwitchcraftMetalProto

enum BenchmarkInputs {
    /// Per-shape seeds. Keeping these explicit (rather than hashing the
    /// shape) keeps the inputs stable across Swift toolchain versions.
    static func seedA(for shape: MatmulShape) -> UInt64 {
        switch shape.name {
        case "qkv_proj":   return 0xC0FF_EE00_0000_0001
        case "ffn_up":     return 0xC0FF_EE00_0000_0002
        case "ffn_down":   return 0xC0FF_EE00_0000_0003
        case "token_proj": return 0xC0FF_EE00_0000_0004
        default:           return 0xC0FF_EE00_DEAD_BEEF
        }
    }

    static func seedB(for shape: MatmulShape) -> UInt64 {
        switch shape.name {
        case "qkv_proj":   return 0xBADC_0FFE_0000_0001
        case "ffn_up":     return 0xBADC_0FFE_0000_0002
        case "ffn_down":   return 0xBADC_0FFE_0000_0003
        case "token_proj": return 0xBADC_0FFE_0000_0004
        default:           return 0xBADC_0FFE_DEAD_BEEF
        }
    }

    /// Generate `count` floats in `[-0.5, 0.5)` from the supplied generator.
    /// Matches the value range used in `MatrixOpsTests.randomUnitRow` before
    /// the unit-norm step (we don't normalise here — matmul correctness
    /// doesn't depend on it and skipping the sqrt keeps generation cheap).
    static func makeFloats(count: Int, rng: inout SplitMix64) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let u = Float(rng.next()) / Float(UInt64.max)
            out[i] = u - 0.5
        }
        return out
    }

    /// Build the `(A, B)` input pair for a shape with deterministic seeds.
    static func makeAB(for shape: MatmulShape) -> (a: [Float], b: [Float]) {
        var rngA = SplitMix64(seed: seedA(for: shape))
        var rngB = SplitMix64(seed: seedB(for: shape))
        let a = makeFloats(count: shape.m * shape.k, rng: &rngA)
        let b = makeFloats(count: shape.k * shape.n, rng: &rngB)
        return (a, b)
    }
}

/// Reference matmul computed via the production `cblas_sgemm` path
/// (`SearchEngine.matmulQueryTimesRowMajorTranspose`). The production helper
/// computes `A · Bᵀ`, so to obtain `A · B` we pre-transpose `B` and then
/// pass it as the "rows" argument. The underlying `cblas_sgemm` invocation
/// is the same — same flop count, same call shape — which is the property
/// the benchmark harness cares about.
enum AccelerateMatmul {
    static func matmul(
        a: [Float], aShape: (Int, Int),
        b: [Float], bShape: (Int, Int)
    ) -> [Float] {
        let (m, k) = aShape
        let (kB, n) = bShape
        precondition(k == kB, "shape mismatch")
        // Transpose B (k × n row-major) → b' (n × k row-major).
        var bT = [Float](repeating: 0, count: n * k)
        for r in 0..<k {
            for c in 0..<n {
                bT[c * k + r] = b[r * n + c]
            }
        }
        return SearchEngine.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: a, n: m,
            rows: bT, rowCount: n, dims: k
        )
    }

    /// In-place variant the benchmark harness uses to avoid timing the
    /// transpose. Pass `bT` already laid out as `n × k` row-major.
    static func matmulPretransposed(
        a: [Float], m: Int, k: Int,
        bT: [Float], n: Int
    ) -> [Float] {
        return SearchEngine.matmulQueryTimesRowMajorTranspose(
            queryEmbeddings: a, n: m,
            rows: bT, rowCount: n, dims: k
        )
    }

    /// Pre-transpose `B` once for benchmark hot-loop reuse.
    static func transposeB(_ b: [Float], k: Int, n: Int) -> [Float] {
        var bT = [Float](repeating: 0, count: n * k)
        for r in 0..<k {
            for c in 0..<n {
                bT[c * k + r] = b[r * n + c]
            }
        }
        return bT
    }
}

/// Cosine similarity between two equal-length float vectors. Returns 1.0
/// for identical-direction vectors, 0.0 for orthogonal, -1.0 for opposite.
/// Used by the prototype correctness tests against the cblas reference.
enum CosineSimilarity {
    static func compute(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count)
        var dot: Double = 0, nA: Double = 0, nB: Double = 0
        for i in 0..<a.count {
            let ai = Double(a[i]), bi = Double(b[i])
            dot += ai * bi
            nA += ai * ai
            nB += bi * bi
        }
        let denom = (nA.squareRoot() * nB.squareRoot())
        return denom > 0 ? dot / denom : 1.0
    }
}

/// Suite-level gates. Mirrors `Tests/SwitchcraftTests/Support/CoreMLAsset.swift`
/// in spirit: an env var must be set, otherwise the suite skips cleanly so
/// the default `swift test` invocation stays green.
enum MetalProtoGate {
    static let correctnessEnvVar = "SWITCHCRAFT_METAL_PROTO"
    static let benchmarkEnvVar = "SWITCHCRAFT_METAL_PROTO_BENCH"

    static var correctnessEnabled: Bool {
        envFlagSet(correctnessEnvVar) || envFlagSet(benchmarkEnvVar)
    }

    static var benchmarkEnabled: Bool {
        envFlagSet(benchmarkEnvVar) && isReleaseBuild
    }

    static let isReleaseBuild: Bool = {
        var debug = false
        assert({ debug = true; return true }())
        return !debug
    }()

    private static func envFlagSet(_ name: String) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name] else { return false }
        return !raw.isEmpty && raw != "0"
    }

    /// Optional override for where benchmark JSON gets written. When unset,
    /// the suite falls back to the path inside the worktree (the default
    /// committed location). Tests that need to run under sandboxed / read-only
    /// filesystems can point this at a tmp dir.
    static var resultsDirOverride: URL? {
        guard let raw = ProcessInfo.processInfo.environment["SWITCHCRAFT_METAL_PROTO_RESULTS_DIR"],
              !raw.isEmpty
        else { return nil }
        return URL(fileURLWithPath: raw, isDirectory: true)
    }
}
