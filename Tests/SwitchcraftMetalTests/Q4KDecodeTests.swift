// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the pure-Swift Q4_K dequantisation reference
// (issue #59 sub-issue 2). Hand-crafted super-blocks exercise the
// scale/min unpack and the 4-bit weight unpack against the formula
// documented in `docs/porting/ggml-t5.md` §"Dequantisation".
//
// These tests run unconditionally — no GGUF asset required.

#if canImport(Metal)

import Foundation
import Testing
@_spi(SwitchcraftMetal) import SwitchcraftMetal

@Suite("Q4_K decode reference")
struct Q4KDecodeTests {

    /// Build a single Q4_K super-block with the given:
    ///
    ///   - `d`: super-block scale (FP16)
    ///   - `dmin`: super-block min (FP16)
    ///   - `subScales`: 8 sub-block 6-bit scales (each 0..63)
    ///   - `subMins`: 8 sub-block 6-bit mins (each 0..63)
    ///   - `weightsLow`: 128 low-nibble weights (one per byte position
    ///     in `qs[]`, applied to the first 32-element half of each
    ///     sub-block pair)
    ///   - `weightsHigh`: 128 high-nibble weights (applied to the
    ///     second 32-element half)
    static func makeBlock(
        d: Float16,
        dmin: Float16,
        subScales: [UInt8],
        subMins: [UInt8],
        weightsLow: [UInt8],
        weightsHigh: [UInt8]
    ) -> Data {
        precondition(subScales.count == 8)
        precondition(subMins.count == 8)
        precondition(weightsLow.count == 128)
        precondition(weightsHigh.count == 128)
        for v in subScales { precondition(v < 64) }
        for v in subMins { precondition(v < 64) }
        for v in weightsLow { precondition(v < 16) }
        for v in weightsHigh { precondition(v < 16) }

        var bytes = [UInt8](repeating: 0, count: 144)

        // FP16 d (little-endian).
        let dBits = d.bitPattern
        bytes[0] = UInt8(dBits & 0xFF)
        bytes[1] = UInt8((dBits >> 8) & 0xFF)
        // FP16 dmin (little-endian).
        let dminBits = dmin.bitPattern
        bytes[2] = UInt8(dminBits & 0xFF)
        bytes[3] = UInt8((dminBits >> 8) & 0xFF)

        // Pack the 8 sub-block 6-bit scales/mins into 12 bytes using
        // the same layout `get_scale_min_k4` decodes:
        //   for j ∈ [0,4): scales[j] holds the scale's low 6 bits in
        //       its low 6 bits; mins[j-0] holds the min's low 6 bits in
        //       scales[j+4]'s low 6 bits.
        //   for j ∈ [4,8): scales[j-4]'s high 2 bits hold the scale's
        //       high 2 bits; scales[j+4]'s low 4 bits hold the scale's
        //       low 4 bits. Similarly for mins.
        var s = [UInt8](repeating: 0, count: 12)
        // First 4 sub-blocks (low 6 bits in scales[0..4] and scales[4..8]).
        for j in 0..<4 {
            s[j] |= subScales[j] & 0x3F
            s[j + 4] |= subMins[j] & 0x3F
        }
        // Last 4 sub-blocks (split-encoded).
        for j in 4..<8 {
            // scale: low 4 bits → scales[j+4] low nibble;
            //        high 2 bits → scales[j-4] bits 6..7
            let sc = subScales[j] & 0x3F
            s[j + 4] |= sc & 0x0F
            s[j - 4] |= (sc >> 4) << 6
            // min:   low 4 bits → scales[j+4] high nibble;
            //        high 2 bits → scales[j-0] bits 6..7
            let mn = subMins[j] & 0x3F
            s[j + 4] |= (mn & 0x0F) << 4
            s[j - 0] |= (mn >> 4) << 6
        }
        for k in 0..<12 { bytes[4 + k] = s[k] }

        // 128 bytes of qs[]: low nibble = weightsLow[i], high nibble =
        // weightsHigh[i].
        for i in 0..<128 {
            bytes[4 + 12 + i] = (weightsHigh[i] << 4) | (weightsLow[i] & 0x0F)
        }

        return Data(bytes)
    }

    @Test("zero-block decodes to all zeros")
    func zeroBlock() {
        let block = Self.makeBlock(
            d: 0, dmin: 0,
            subScales: [UInt8](repeating: 0, count: 8),
            subMins: [UInt8](repeating: 0, count: 8),
            weightsLow: [UInt8](repeating: 0, count: 128),
            weightsHigh: [UInt8](repeating: 0, count: 128)
        )
        let out = Q4KDecode.dequantise(blocks: block)
        #expect(out.count == 256)
        for v in out { #expect(v == 0) }
    }

    @Test("constant-scale block reproduces upstream formula")
    func constantScaleBlock() {
        // Spread the per-sub-block parameters so the test catches
        // sub-block addressing bugs (j<4 vs j>=4 split-byte unpack).
        let dHalf = Float16(0.125)
        let dminHalf = Float16(0.0625)
        let subScales: [UInt8] = [3, 7, 11, 15, 19, 23, 27, 31]
        let subMins: [UInt8] = [2, 4, 6, 8, 10, 12, 14, 16]

        // Use a deterministic pattern so off-by-one errors in the
        // low/high nibble unpack are visible: low nibbles ramp 0..15
        // repeatedly, high nibbles ramp 15..0.
        var weightsLow = [UInt8](repeating: 0, count: 128)
        var weightsHigh = [UInt8](repeating: 0, count: 128)
        for i in 0..<128 {
            weightsLow[i] = UInt8(i % 16)
            weightsHigh[i] = UInt8(15 - (i % 16))
        }

        let block = Self.makeBlock(
            d: dHalf,
            dmin: dminHalf,
            subScales: subScales,
            subMins: subMins,
            weightsLow: weightsLow,
            weightsHigh: weightsHigh
        )
        let out = Q4KDecode.dequantise(blocks: block)
        #expect(out.count == 256)

        // Compute the expected values via the reference formula. The
        // upstream code processes 64 elements per outer loop iteration
        // (two adjacent sub-blocks); the first 32 elements of each
        // 64-block use sub-block scale `is`, the next 32 use sub-block
        // scale `is+1`. Within each 32-element half the weights come
        // from `weightsLow[qOffset+l]` (first half) and
        // `weightsHigh[qOffset+l]` (second half), where `qOffset`
        // advances by 32 per outer iteration.
        let d = Float(dHalf)
        let dmin = Float(dminHalf)
        var expected = [Float](repeating: 0, count: 256)
        var yIdx = 0
        var qOffset = 0
        var isIdx = 0
        while yIdx < 256 {
            let sc1 = subScales[isIdx]
            let mn1 = subMins[isIdx]
            let d1 = d * Float(sc1)
            let m1 = dmin * Float(mn1)
            let sc2 = subScales[isIdx + 1]
            let mn2 = subMins[isIdx + 1]
            let d2 = d * Float(sc2)
            let m2 = dmin * Float(mn2)
            for l in 0..<32 {
                let nibble = Float(weightsLow[qOffset + l] & 0x0F)
                expected[yIdx] = d1 * nibble - m1
                yIdx += 1
            }
            for l in 0..<32 {
                let nibble = Float(weightsHigh[qOffset + l] & 0x0F)
                expected[yIdx] = d2 * nibble - m2
                yIdx += 1
            }
            qOffset += 32
            isIdx += 2
        }

        // Bit-equal check — Swift's straight-line FP32 ops match the
        // reference closure exactly because both compute the same
        // sequence of multiplies and subtracts.
        for i in 0..<256 {
            #expect(out[i].bitPattern == expected[i].bitPattern,
                    "element \(i): got \(out[i]), expected \(expected[i])")
        }
    }

    @Test("multi-block decode produces the right element count")
    func multiBlockSizing() {
        var combined = Data()
        let zeros = [UInt8](repeating: 0, count: 8)
        let zeroLow = [UInt8](repeating: 0, count: 128)
        for _ in 0..<3 {
            combined.append(Self.makeBlock(
                d: 0, dmin: 0,
                subScales: zeros, subMins: zeros,
                weightsLow: zeroLow, weightsHigh: zeroLow
            ))
        }
        let out = Q4KDecode.dequantise(blocks: combined)
        #expect(out.count == 3 * 256)
    }
}

#endif // canImport(Metal)
