// SPDX-License-Identifier: Apache-2.0
//
// T5-base matmul shapes under test. Issue #49 lists four representative
// projections from the encoder forward pass; this file enumerates them as
// `(name, M, K, N)` tuples so the prototype's correctness, benchmark, and
// threadgroup-sweep suites all share a single source of truth.
//
// All shapes treat `A` as row-major `M × K` and `B` as row-major `K × N`,
// producing a row-major `M × N` output. Sequence length is fixed at the T5
// default of 512 tokens; batch is 1.

import Foundation

/// One row in the shape table.
public struct MatmulShape: Sendable, Equatable, Codable {
    /// Human-readable label (used in benchmark output).
    public let name: String
    /// Rows of `A`.
    public let m: Int
    /// Inner dimension (cols of `A` = rows of `B`).
    public let k: Int
    /// Cols of `B` (also cols of the output).
    public let n: Int

    public init(name: String, m: Int, k: Int, n: Int) {
        self.name = name
        self.m = m
        self.k = k
        self.n = n
    }

    /// Total floating-point operations for `C = A · B` (`2·M·K·N`).
    public var flops: Double {
        2.0 * Double(m) * Double(k) * Double(n)
    }
}

public enum T5BaseShapes {
    /// `[512, 768] × [768, 768]` — q/k/v projection.
    public static let qkv = MatmulShape(name: "qkv_proj", m: 512, k: 768, n: 768)
    /// `[512, 768] × [768, 3072]` — `wi_0` / `wi_1` FFN up-projection.
    public static let ffnUp = MatmulShape(name: "ffn_up", m: 512, k: 768, n: 3072)
    /// `[512, 3072] × [3072, 768]` — `wo` FFN down-projection.
    public static let ffnDown = MatmulShape(name: "ffn_down", m: 512, k: 3072, n: 768)
    /// `[512, 768] × [768, 128]` — final token projection.
    public static let tokenProj = MatmulShape(name: "token_proj", m: 512, k: 768, n: 128)

    /// All four shapes in declaration order. The "largest" shape used by the
    /// go/no-go bar and the threadgroup-sweep is `ffnUp` (3072-wide output).
    public static let all: [MatmulShape] = [qkv, ffnUp, ffnDown, tokenProj]

    /// The shape used to evaluate the go/no-go bar and as the threadgroup-sweep
    /// target. `ffnUp` is the largest by FLOPs and the most common projection
    /// in T5's forward pass.
    public static let largest: MatmulShape = ffnUp
}
