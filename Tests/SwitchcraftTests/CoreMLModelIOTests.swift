// SPDX-License-Identifier: Apache-2.0
import Testing
@testable import SwitchcraftCoreML

// Tests for the pure-Swift IEEE 754 half-precision → Float32 decoder.
//
// Strategy B was chosen for the Float16/x86_64 fix: a single UInt16-based
// decode path is used on all architectures, with no #if arch(arm64)
// conditional. This means the exact code that runs on x86_64 archive builds
// is CI-verified on Apple Silicon. The tests below exercise that shared path.
//
// CI coverage note: GitHub Actions macOS runners are Apple Silicon. The x86_64
// *compile* path can be verified locally with `arch -x86_64 swift build -c release`.
// Because there is no architecture branch in float32FromFloat16Bits, the arm64
// test results are directly representative of x86_64 behaviour.
@Suite("Float16 codec")
struct Float16CodecTests {

    // Verifies the 11 representative bit patterns required by the issue spec.
    @Test("Known bit patterns decode to correct Float32 values")
    func knownPatterns() {
        // ±zero
        let posZero = float32FromFloat16Bits(0x0000)
        #expect(posZero.isZero)
        #expect(posZero.sign == .plus)

        let negZero = float32FromFloat16Bits(0x8000)
        #expect(negZero.isZero)
        #expect(negZero.sign == .minus)

        // Normals
        #expect(float32FromFloat16Bits(0x3C00) == 1.0)   // +1.0
        #expect(float32FromFloat16Bits(0xBC00) == -1.0)  // -1.0

        // Subnormals
        // Smallest positive subnormal: 1 × 2^(-24) = 5.9604644775390625e-8
        #expect(float32FromFloat16Bits(0x0001) == Float(bitPattern: 0x3380_0000))
        // Largest subnormal: 1023 × 2^(-24) ≈ 6.0975552e-5
        #expect(float32FromFloat16Bits(0x03FF) == Float(bitPattern: 0x387F_C000))
        // Smallest normal: 2^(-14) ≈ 6.103515625e-5
        #expect(float32FromFloat16Bits(0x0400) == Float(bitPattern: 0x3880_0000))

        // Largest normal: 65504
        #expect(float32FromFloat16Bits(0x7BFF) == 65504.0)

        // Infinities
        #expect(float32FromFloat16Bits(0x7C00) == .infinity)
        #expect(float32FromFloat16Bits(0xFC00) == -.infinity)

        // NaN (payload 0x200 — quiet NaN)
        let nan = float32FromFloat16Bits(0x7E00)
        #expect(nan.isNaN)
    }

    // On arm64 the native Float16 type is available as an exhaustive oracle.
    // This test sweeps all 65 536 half-precision bit patterns and verifies that
    // float32FromFloat16Bits produces an identical Float32 value.
    //
    // The #if guard is intentional: Float16(bitPattern:) is unavailable on
    // x86_64, so this cross-validation can only run where the hardware type
    // exists. The knownPatterns test above covers correctness on x86_64.
    #if arch(arm64)
    @Test("Decoder matches native Float16 for all 65536 bit patterns (arm64 oracle)")
    func matchesNativeFloat16() {
        for bits in UInt16.min...UInt16.max {
            let native = Float(Float16(bitPattern: bits))
            let decoded = float32FromFloat16Bits(bits)
            if native.isNaN {
                #expect(decoded.isNaN,
                    "bits=\(String(bits, radix: 16, uppercase: true)): expected NaN, got \(decoded)")
            } else {
                #expect(decoded == native,
                    "bits=\(String(bits, radix: 16, uppercase: true)): expected \(native), got \(decoded)")
            }
        }
    }
    #endif
}
