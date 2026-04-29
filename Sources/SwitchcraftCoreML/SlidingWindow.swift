import Foundation

/// Pure-Swift planner and merger for the long-input sliding-window
/// strategy used by `T5CoreMLEmbedder`.
///
/// The strategy matches Witchcraft's `src/embedder.rs`:
///
/// - **Window size** is fixed (the CoreML model's static sequence length,
///   default `512` tokens). Inputs of length `T <= window` use a single
///   window starting at offset `0`.
/// - For `T > window`, windows step by `stride` (default `256`) until
///   the next window would run past `T`. The **last window is shifted
///   left** so it starts at `T - window` and ends exactly at `T`. This
///   avoids encoding padding rows in the tail and keeps every position
///   covered by at least one real (non-padding) window slot.
/// - When two windows overlap, the per-token outputs at overlapping
///   positions are **averaged** (mean-merge). Averaging two unit vectors
///   that point in slightly different directions yields a vector shorter
///   than 1, so the merged vector is re-normalised to unit length before
///   it leaves the merger. The merger receives both the L2-normalised
///   vectors and the pre-normalisation L2 norms; it returns per-position
///   re-normalised unit vectors plus per-position averaged raw norms (the
///   latter only for the threshold filter — never returned to callers).
/// - After merging, any position whose **mean raw L2 norm** is below
///   `minNorm` (default `1.0`, the `MIN_NORM` constant from Witchcraft)
///   is dropped from the output. Content tokens have norms ≥ ~5.0;
///   `</s>` and padding tokens have norms ≈ 0.27, so the threshold
///   removes low-signal positions while keeping all real content.
///
/// Extracting the window math here keeps it unit-testable on every
/// platform without loading a CoreML model.
public enum SlidingWindow {

    /// Plan the window starts for an input of `tokenCount` tokens.
    ///
    /// - `windowSize`: maximum tokens per inference (CoreML input length).
    ///   Must be `> 0`.
    /// - `stride`: step between consecutive window starts. Must be
    ///   `> 0` and `<= windowSize`.
    /// - Returns: a non-empty array of window-start offsets in
    ///   `[0, max(0, tokenCount - windowSize)]`. The first start is
    ///   always `0`. The last start is always `max(0, tokenCount - windowSize)`
    ///   when `tokenCount > windowSize` (the left-shift rule). Successive
    ///   starts strictly increase.
    public static func plan(
        tokenCount: Int,
        windowSize: Int = 512,
        stride: Int = 256
    ) -> [Int] {
        precondition(windowSize > 0, "windowSize must be positive")
        precondition(stride > 0 && stride <= windowSize,
                     "stride must be in 1...windowSize")

        if tokenCount <= 0 {
            return [0]
        }
        if tokenCount <= windowSize {
            return [0]
        }

        // Generate stride-aligned starts that fully fit (start + windowSize <= T).
        // Then shift the final window left to T - windowSize so it ends at T
        // without padding. If that shift collapses onto the previous start, drop
        // the previous start to keep the sequence strictly increasing.
        let lastStart = tokenCount - windowSize
        var starts: [Int] = []
        var s = 0
        while s + windowSize <= tokenCount {
            starts.append(s)
            s += stride
        }
        if starts.last != lastStart {
            // Drop any starts >= lastStart (defensive — shouldn't happen with
            // the loop condition above, but keep the invariant explicit).
            while let tail = starts.last, tail >= lastStart {
                starts.removeLast()
            }
            starts.append(lastStart)
        }
        return starts
    }

    /// Merge per-window encoder outputs into a single per-token stream
    /// and apply the pre-normalisation L2 norm threshold filter.
    ///
    /// Inputs:
    /// - `windowStarts`: as returned by `plan(...)`.
    /// - `windowSize`: the model's fixed input length.
    /// - `tokenCount`: original (un-padded) token count `T`. Only the
    ///   first `T` rows of any window are merged; rows beyond `T` are
    ///   padding (only possible when `T < windowSize`).
    /// - `dims`: embedding dimension (e.g. 128).
    /// - `windowNormalised`: per-window normalised vectors, row-major
    ///   `windowSize × dims`. One entry per window in `windowStarts`.
    /// - `windowRawNorms`: per-window pre-normalisation L2 norms, one
    ///   `Float` per row (`windowSize` per window).
    /// - `minNorm`: drop positions whose merged raw norm is below this
    ///   threshold (default `1.0`).
    ///
    /// Output: a row-major `m × dims` `[Float]` where `m` is the number
    /// of positions kept after the norm filter, plus the indices of the
    /// kept positions (in the original 0..<T space).
    public static func merge(
        windowStarts: [Int],
        windowSize: Int,
        tokenCount: Int,
        dims: Int,
        windowNormalised: [[Float]],
        windowRawNorms: [[Float]],
        minNorm: Float = 1.0
    ) -> (embeddings: [Float], keptPositions: [Int]) {
        precondition(windowStarts.count == windowNormalised.count,
                     "windowStarts and windowNormalised count mismatch")
        precondition(windowStarts.count == windowRawNorms.count,
                     "windowStarts and windowRawNorms count mismatch")
        precondition(windowSize > 0)
        precondition(dims > 0)
        precondition(tokenCount >= 0)

        if tokenCount == 0 {
            return ([], [])
        }

        var sumNormalised = [Float](repeating: 0, count: tokenCount * dims)
        var sumRawNorm = [Float](repeating: 0, count: tokenCount)
        var counts = [Int](repeating: 0, count: tokenCount)

        for (wi, start) in windowStarts.enumerated() {
            let normalised = windowNormalised[wi]
            let rawNorms = windowRawNorms[wi]
            precondition(normalised.count == windowSize * dims,
                         "windowNormalised[\(wi)] must be windowSize * dims floats")
            precondition(rawNorms.count == windowSize,
                         "windowRawNorms[\(wi)] must be windowSize floats")

            // Window covers positions [start, start + windowSize); only the
            // overlap with [0, tokenCount) is real input.
            let lo = max(0, start)
            let hi = min(tokenCount, start + windowSize)
            if lo >= hi { continue }

            for pos in lo..<hi {
                let localRow = pos - start
                counts[pos] += 1
                sumRawNorm[pos] += rawNorms[localRow]
                let srcOffset = localRow * dims
                let dstOffset = pos * dims
                for d in 0..<dims {
                    sumNormalised[dstOffset + d] += normalised[srcOffset + d]
                }
            }
        }

        var kept: [Int] = []
        kept.reserveCapacity(tokenCount)
        var output: [Float] = []
        output.reserveCapacity(tokenCount * dims)
        for pos in 0..<tokenCount {
            let c = counts[pos]
            if c == 0 { continue } // unreachable for tokenCount > 0 plans
            let invC = 1 / Float(c)
            let meanRawNorm = sumRawNorm[pos] * invC
            if meanRawNorm < minNorm { continue }
            kept.append(pos)
            let srcOffset = pos * dims
            // Mean of unit vectors → renormalise to unit length for the
            // downstream cosine-similarity scorer. The pre-norm filter
            // already used `meanRawNorm` to decide whether to keep this
            // position; the returned vector is unit-normalised.
            var sumSq: Float = 0
            for d in 0..<dims {
                let v = sumNormalised[srcOffset + d] * invC
                sumSq += v * v
            }
            let invLen: Float = sumSq > 0 ? 1 / sumSq.squareRoot() : 0
            if invLen == 0 {
                // Degenerate: averaged to the zero vector (only possible
                // when overlapping windows produced exactly antipodal
                // unit vectors). Drop the position rather than emit NaN.
                kept.removeLast()
                continue
            }
            for d in 0..<dims {
                output.append(sumNormalised[srcOffset + d] * invC * invLen)
            }
        }
        return (output, kept)
    }
}
