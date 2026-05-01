// SPDX-License-Identifier: Apache-2.0
//
// T5 relative-position attention bias for the Phase 2 Metal embedder
// (issue #64). Direct port of HuggingFace transformers'
// `T5Attention._relative_position_bucket` + the ggml/Candle
// "compute once per encode, broadcast across layers" pattern.
//
// See `docs/porting/ggml-t5.md` §"Relative-position attention bucket"
// for the formula reference and ADR 017 §"Per-op precision routing"
// for the FP32-accumulator contract.

#if canImport(Metal)

import Foundation

/// Pure-Swift builder for T5's relative-position attention bias.
///
/// The encoder forward pass uses a per-head learned bias added to the
/// pre-softmax attention logits. The bias is keyed by a logarithmic
/// bucket of the relative offset between query and key positions.
///
/// Two-stage construction:
///   1. **`bucket(relativePosition:)`** computes the bucket index for
///      one signed offset, matching transformers' encoder defaults
///      (`bidirectional=true`, `num_buckets=32`, `max_distance=128`).
///   2. **`buildBiasMatrix(table:numHeads:seqLen:)`** assembles the
///      `(numHeads × seqLen × seqLen)` FP32 bias buffer for one
///      window. The same buffer is broadcast across all 12 attention
///      layers in `T5MetalEmbedder.encodeWindow`.
@_spi(SwitchcraftMetal)
public enum T5RelativePositionBias {

    /// Number of buckets used by the encoder (`relative_attention_num_buckets`
    /// in `xtr-base-en/config.json`). The bidirectional split below
    /// halves this to 16.
    public static let numBuckets: Int = 32

    /// Maximum distance covered by the logarithmic span
    /// (`relative_attention_max_distance` in `xtr-base-en/config.json`).
    public static let maxDistance: Int = 128

    /// Compute the bucket index for a signed relative position
    /// `relativePosition = key_index - query_index`.
    ///
    /// Bidirectional encoder variant: positive offsets occupy the
    /// upper half of the bucket range, negative offsets the lower
    /// half. Within each half, half the buckets are exact 1:1 (small
    /// offsets) and the other half are logarithmically spaced up to
    /// `maxDistance`.
    ///
    /// The arithmetic mirrors transformers' Python verbatim
    /// (`_relative_position_bucket`) — keep it that way for parity
    /// debugging.
    public static func bucket(relativePosition: Int) -> Int {
        // bidirectional == true → halve buckets, encode sign in the
        // upper-half offset.
        let halfBuckets = numBuckets / 2  // 16
        let isPositive = relativePosition > 0
        let relativeBuckets = isPositive ? halfBuckets : 0
        let absPos = abs(relativePosition)

        let maxExact = halfBuckets / 2  // 8
        if absPos < maxExact {
            return relativeBuckets + absPos
        }

        // Logarithmic span: `max_exact + log(absPos / max_exact) /
        // log(maxDistance / max_exact) * (halfBuckets - max_exact)`.
        // Matches PyTorch's `.to(torch.long)` which truncates toward
        // zero for non-negative values; clamp at `halfBuckets - 1` so
        // out-of-range offsets land in the last bucket.
        let logRatio = log(Double(absPos) / Double(maxExact))
        let logSpan = log(Double(maxDistance) / Double(maxExact))
        let scaled = logRatio / logSpan * Double(halfBuckets - maxExact)
        let bucket = maxExact + Int(scaled)
        let clamped = min(bucket, halfBuckets - 1)
        return relativeBuckets + clamped
    }

    /// Pre-compute the `(seqLen × seqLen)` bucket-index table for a
    /// fixed window length. Independent of the per-head weights — the
    /// table is reusable across encodes if the seqLen is constant
    /// (which it is for `T5MetalEmbedder` — always `windowSize`).
    public static func bucketTable(seqLen: Int) -> [Int32] {
        var table = [Int32](repeating: 0, count: seqLen * seqLen)
        for q in 0..<seqLen {
            for k in 0..<seqLen {
                let rp = k - q
                table[q * seqLen + k] = Int32(bucket(relativePosition: rp))
            }
        }
        return table
    }

    /// Assemble the per-head FP32 bias buffer for one encode call.
    ///
    /// - Parameters:
    ///   - bucketTable: pre-computed `(seqLen × seqLen)` bucket indices
    ///     (see `bucketTable(seqLen:)`).
    ///   - weights: GGUF-loaded `relative_attention_bias.weight`,
    ///     row-major `(numBuckets × numHeads)` (HuggingFace
    ///     `Embedding(num_buckets, num_heads)` → on-disk fastest-
    ///     varying-first reverses to `[num_heads, num_buckets]`, but
    ///     the row-major Swift view is `weights[bucket * numHeads + head]`).
    ///     Pass the un-reversed shape explicitly so callers know which
    ///     layout they hand in — see `T5MetalEmbedder` weight binding.
    ///   - numHeads: number of attention heads (12 for T5-base).
    ///   - seqLen: window length (typically 512).
    /// - Returns: FP32 row-major buffer of shape
    ///   `(numHeads × seqLen × seqLen)` ready to be uploaded to a
    ///   `MTLBuffer` and passed as the `bias:` argument to
    ///   `SoftmaxKernel`. The flat layout (`B = numHeads * seqLen`,
    ///   `N = seqLen`) matches the softmax kernel's bias shape.
    public static func buildBiasMatrix(
        bucketTable: [Int32],
        weights: [Float],
        numHeads: Int,
        seqLen: Int
    ) -> [Float] {
        precondition(bucketTable.count == seqLen * seqLen)
        precondition(weights.count == numBuckets * numHeads)
        precondition(numHeads > 0)
        precondition(seqLen > 0)

        var bias = [Float](repeating: 0, count: numHeads * seqLen * seqLen)
        for h in 0..<numHeads {
            let headBase = h * seqLen * seqLen
            for q in 0..<seqLen {
                let rowBase = headBase + q * seqLen
                for k in 0..<seqLen {
                    let bIdx = Int(bucketTable[q * seqLen + k])
                    bias[rowBase + k] = weights[bIdx * numHeads + h]
                }
            }
        }
        return bias
    }
}

#endif // canImport(Metal)
