// SPDX-License-Identifier: Apache-2.0

// IEEE 754 half-precision → Float32 decoder.
//
// `Float16` is unavailable on x86_64 macOS (Swift compile-time error), which
// breaks Release-config universal-binary archive builds. This pure-Swift
// decoder is used on all architectures (Strategy B) so the same code path is
// CI-verified on arm64 and the same binary compiles for x86_64.
//
// Reference: IEEE 754-2008 §3.3 — half-precision format:
//   bit 15    : sign
//   bits 14–10: biased exponent (bias = 15; all-zeros = subnormal/zero,
//               all-ones = infinity/NaN)
//   bits  9–0 : mantissa fraction (10 explicit bits)
internal func float32FromFloat16Bits(_ bits: UInt16) -> Float {
    let sign: UInt32     = UInt32(bits & 0x8000) << 16   // bit 31 of Float32
    let exp16: UInt32    = UInt32((bits >> 10) & 0x1F)
    let mantissa: UInt32 = UInt32(bits & 0x3FF)

    if exp16 == 0 {
        guard mantissa != 0 else { return Float(bitPattern: sign) }  // ±zero

        // Subnormal: value = mantissa × 2^(−24).
        // Normalise: find position of the highest set bit (0-indexed from LSB)
        // so the value can be expressed as 1.frac × 2^(p−24).
        let p     = 31 - mantissa.leadingZeroBitCount        // 0 for 0x0001, 9 for 0x03FF
        let exp32 = UInt32(p + 103)                          // rebias: p−24 + 127 = p+103
        let frac  = (mantissa - (1 << p)) << (23 - p)        // strip implicit 1, extend to 23 bits
        return Float(bitPattern: sign | (exp32 << 23) | frac)

    } else if exp16 == 31 {
        guard mantissa != 0 else {
            return Float(bitPattern: sign | 0x7F800000)      // ±infinity
        }
        // NaN: propagate payload best-effort (10-bit → 23-bit, shift left 13).
        return Float(bitPattern: sign | 0x7F800000 | (mantissa << 13))

    } else {
        // Normal: rebias exponent (FP16 bias = 15 → FP32 bias = 127: add 112).
        // Extend 10-bit mantissa to 23-bit FP32 mantissa by shifting left 13.
        let exp32 = exp16 + 112
        return Float(bitPattern: sign | (exp32 << 23) | (mantissa << 13))
    }
}
