// SPDX-License-Identifier: Apache-2.0
/// SplitMix64 — small, fast, seedable 64-bit PRNG.
///
/// Used as the default seedable RNG in algorithms that take an `inout
/// RandomNumberGenerator`, and in tests where reproducibility matters.
/// Quality is sufficient for non-cryptographic use; not for security.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    public var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
