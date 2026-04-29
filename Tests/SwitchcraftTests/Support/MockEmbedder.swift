import Foundation
import CryptoKit
import SwitchcraftCore

/// Deterministic synthetic embedder for tests.
///
/// Produces stable per-token unit vectors derived from a SHA-256 of
/// `"\(text)|\(tokenIndex)"`. Tokens are whitespace-split. Whitespace-only
/// or empty input returns an empty array.
///
/// - Same input ⇒ byte-identical output, across runs and processes (the
///   per-process random seed of `Foundation.Hasher` is deliberately
///   avoided).
/// - `dims` defaults to 128, matching Witchcraft's projection dimension.
struct MockEmbedder: Embedder, Sendable {
    let dims: Int
    let modelIdentifier: String

    init(dims: Int = 128, modelIdentifier: String? = nil) {
        precondition(dims > 0 && dims % 2 == 0,
                     "MockEmbedder dims must be positive and even")
        self.dims = dims
        self.modelIdentifier = modelIdentifier ?? "mock-embedder-v1-d\(dims)"
    }

    func encode(_ text: String) async throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [] }
        let tokens = trimmed.split(whereSeparator: { $0.isWhitespace })
        if tokens.isEmpty { return [] }

        var out = [Float]()
        out.reserveCapacity(tokens.count * dims)
        for (i, _) in tokens.enumerated() {
            // Per-token seed: first 8 bytes of SHA-256("text|i") LE.
            let seedString = "\(text)|\(i)"
            let digest = SHA256.hash(data: Data(seedString.utf8))
            var seed: UInt64 = 0
            for (b, byte) in digest.prefix(8).enumerated() {
                seed |= UInt64(byte) << (8 * b)
            }
            var rng = SplitMix64(seed: seed)
            // Fill `dims` floats in (-1, 1) and normalise to unit length.
            var vec = [Float](repeating: 0, count: dims)
            var sumSq: Float = 0
            for j in 0..<dims {
                let bits = UInt32(truncatingIfNeeded: rng.next())
                // Map UInt32 to [-1, 1) float using upper 24 bits to keep
                // mantissa-exact stability.
                let scaled = Float(bits >> 8) / Float(1 << 24)
                let v = scaled * 2 - 1
                vec[j] = v
                sumSq += v * v
            }
            // Avoid division by zero; on the impossible all-zero draw,
            // fall back to a one-hot vector.
            let norm = sumSq > 0 ? 1 / sumSq.squareRoot() : 0
            if norm == 0 {
                vec[0] = 1
            } else {
                for j in 0..<dims { vec[j] *= norm }
            }
            out.append(contentsOf: vec)
        }
        return out
    }
}
