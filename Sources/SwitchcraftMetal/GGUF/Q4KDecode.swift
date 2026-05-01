// SPDX-License-Identifier: Apache-2.0
//
// CPU dequantisation reference for ggml's Q4_K format.
//
// This is a one-to-one transliteration of `dequantize_row_q4_K` from
// `ggml/src/ggml-quants.c` at the pinned upstream commit
// `b70770970e84c30a007b3859a453768b3ece2d3d` (recorded in
// `docs/porting/ggml-t5.md`). The Swift implementation mirrors the C
// FP32 multiply/add sequence exactly so the output is bit-equal to the
// upstream code under IEEE-754 round-to-nearest-even — Swift does not
// auto-fuse FP32 ops into FMA, and we deliberately avoid `simd`
// shortcuts and `@_transparent` here. See ADR 016 §"Bit-equal Q4_K
// decode" for the rationale.
//
// This routine is **not on the production read path** — the GGUFReader
// uploads Q4_K bytes raw and the matmul kernel (sub-issue #60)
// dequantises on the fly inside the GPU pipeline. `dequantise(...)`
// exists for:
//
//   1. The round-trip parity test in this sub-issue (decoded values
//      match the upstream FP32 dump bit-for-bit).
//   2. The CPU reference path the matmul kernel's parity test (#60)
//      will compare against.

#if canImport(Metal)

import Foundation

/// Pure-Swift CPU reference for ggml's `dequantize_row_q4_K`.
@_spi(SwitchcraftMetal)
public enum Q4KDecode {
    /// Number of weights packed into one Q4_K super-block.
    public static let elementsPerSuperBlock = 256
    /// Bytes per super-block: `2*sizeof(half) + 12 + 128 = 144`.
    public static let bytesPerSuperBlock = 144

    /// Dequantise a contiguous run of Q4_K super-blocks into FP32.
    ///
    /// - Parameters:
    ///   - blocks: Raw Q4_K bytes — must be a whole number of
    ///     super-blocks (`blocks.count % 144 == 0`).
    ///   - out: Output buffer; must have capacity for
    ///     `(blocks.count / 144) * 256` floats.
    /// - Precondition: `out.count == (blocks.count / 144) * 256`.
    public static func dequantise(
        blocks: UnsafeRawBufferPointer,
        into out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(
            blocks.count % bytesPerSuperBlock == 0,
            "Q4_K block byte count must be a multiple of 144 (got \(blocks.count))"
        )
        let nb = blocks.count / bytesPerSuperBlock
        precondition(
            out.count == nb * elementsPerSuperBlock,
            "Output buffer must have capacity \(nb * elementsPerSuperBlock) (got \(out.count))"
        )

        let base = blocks.baseAddress!.assumingMemoryBound(to: UInt8.self)

        var yIndex = 0
        for i in 0..<nb {
            let blk = base.advanced(by: i * bytesPerSuperBlock)

            // Super-block scale `d` and min `dmin`, FP16 little-endian.
            let dHalf = Float16(
                bitPattern: UInt16(blk[0]) | (UInt16(blk[1]) << 8)
            )
            let dminHalf = Float16(
                bitPattern: UInt16(blk[2]) | (UInt16(blk[3]) << 8)
            )
            let d = Float(dHalf)
            let dmin = Float(dminHalf)

            // 12 bytes of packed 6-bit sub-scales / mins.
            let scales = blk.advanced(by: 4)
            // 128 bytes of 4-bit packed weights.
            var qOffset = 4 + 12

            // 8 sub-blocks of 32 elements each, processed in pairs of
            // 64 elements (matches upstream's `j += 64` loop). `is` is
            // the sub-block pair counter.
            var isIdx = 0
            for _ in stride(from: 0, to: elementsPerSuperBlock, by: 64) {
                var sc: UInt8 = 0
                var m: UInt8 = 0

                getScaleMinK4(j: isIdx + 0, scales: scales, sc: &sc, m: &m)
                let d1 = d * Float(sc)
                let m1 = dmin * Float(m)

                getScaleMinK4(j: isIdx + 1, scales: scales, sc: &sc, m: &m)
                let d2 = d * Float(sc)
                let m2 = dmin * Float(m)

                let q = base.advanced(by: i * bytesPerSuperBlock + qOffset)

                // First 32 elements of the pair: low nibble of `q[l]`.
                for l in 0..<32 {
                    let nibble = Float(q[l] & 0x0F)
                    out[yIndex] = d1 * nibble - m1
                    yIndex += 1
                }
                // Next 32 elements: high nibble of `q[l]`.
                for l in 0..<32 {
                    let nibble = Float(q[l] >> 4)
                    out[yIndex] = d2 * nibble - m2
                    yIndex += 1
                }

                qOffset += 32
                isIdx += 2
            }
        }
    }

    /// Convenience wrapper that allocates and returns a fresh `[Float]`.
    public static func dequantise(blocks: Data) -> [Float] {
        let nb = blocks.count / bytesPerSuperBlock
        var out = [Float](repeating: 0, count: nb * elementsPerSuperBlock)
        out.withUnsafeMutableBufferPointer { dst in
            blocks.withUnsafeBytes { src in
                dequantise(blocks: src, into: dst)
            }
        }
        return out
    }

    /// Direct port of `get_scale_min_k4` from `ggml-quants.c`.
    ///
    /// Unpacks the 6-bit scale and 6-bit min for sub-block `j ∈ [0, 8)`
    /// from the 12-byte `scales` packing. The j<4 branch reads the low
    /// 6 bits of two adjacent bytes; the j>=4 branch stitches a low
    /// nibble with two high bits from a third byte.
    @inline(__always)
    private static func getScaleMinK4(
        j: Int,
        scales: UnsafePointer<UInt8>,
        sc: inout UInt8,
        m: inout UInt8
    ) {
        if j < 4 {
            sc = scales[j] & 63
            m = scales[j + 4] & 63
        } else {
            sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
            m = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4)
        }
    }
}

#endif // canImport(Metal)
