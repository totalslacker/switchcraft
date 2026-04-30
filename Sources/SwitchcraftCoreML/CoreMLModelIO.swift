// SPDX-License-Identifier: Apache-2.0
import Foundation

#if canImport(CoreML)
import CoreML

/// Output names produced by `scripts/convert-xtr-to-coreml.py`. The Swift
/// wrapper looks these up by name on every prediction.
enum CoreMLOutputName {
    /// Raw projected `[1, sequence, 128]` FP16 tensor — output of the
    /// `768 → 128` projection layer, before L2 normalisation. Used by
    /// the Swift caller to compute per-token L2 norms for the
    /// pre-normalisation `MIN_NORM` filter.
    static let rawProjected = "raw_projected"
    /// Unit-normalised `[1, sequence, 128]` FP16 tensor — the final
    /// per-token embeddings ready to be sliced and returned.
    static let normalised = "normalised"
}

/// Input feature name produced by `scripts/convert-xtr-to-coreml.py`.
enum CoreMLInputName {
    /// `[1, 512]` Int32 token IDs (HuggingFace `tokenizer.json` IDs;
    /// pad-token ID = 0).
    static let inputIDs = "input_ids"
}

/// Helpers for building `MLMultiArray` inputs and reading Float16
/// outputs into row-major `[Float]` buffers.
enum CoreMLModelIO {

    /// Build a `[1, windowSize]` Int32 `MLMultiArray` from a window of
    /// token IDs. Positions beyond `tokens.count` are padded with
    /// `padTokenID` (default `0`, matching the XTR-base-en pad token).
    static func makeInputIDs(
        tokens: ArraySlice<Int32>,
        windowSize: Int,
        padTokenID: Int32 = 0
    ) throws -> MLMultiArray {
        precondition(tokens.count <= windowSize,
                     "tokens slice must fit in the window")
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: windowSize)],
            dataType: .int32
        )
        // MLMultiArray.dataPointer is a raw `Int32` buffer for `.int32` dtype.
        let ptr = array.dataPointer.bindMemory(
            to: Int32.self, capacity: windowSize
        )
        var i = 0
        for tok in tokens {
            ptr[i] = tok
            i += 1
        }
        while i < windowSize {
            ptr[i] = padTokenID
            i += 1
        }
        return array
    }

    /// Read a `[1, windowSize, dims]` Float16 multi-array into a flat
    /// row-major `[Float]` of length `windowSize * dims`. The function
    /// widens FP16 → FP32 at the boundary (Embedder always returns
    /// `[Float]`).
    ///
    /// Strides are honoured so the function works with non-contiguous
    /// CoreML output buffers.
    static func readEmbeddingTensor(
        _ array: MLMultiArray,
        windowSize: Int,
        dims: Int
    ) -> [Float] {
        precondition(array.shape.count == 3,
                     "expected rank-3 [1, sequence, dims] tensor, got shape \(array.shape)")
        precondition(array.shape[0].intValue == 1,
                     "expected batch dim 1, got \(array.shape[0])")
        precondition(array.shape[1].intValue == windowSize,
                     "expected sequence dim \(windowSize), got \(array.shape[1])")
        precondition(array.shape[2].intValue == dims,
                     "expected feature dim \(dims), got \(array.shape[2])")

        var out = [Float](repeating: 0, count: windowSize * dims)
        let strideSeq = array.strides[1].intValue
        let strideDim = array.strides[2].intValue

        switch array.dataType {
        case .float16:
            // Float16 is bit-identical to Swift's Float16 (since macOS 11).
            let base = array.dataPointer.bindMemory(
                to: Float16.self, capacity: array.count
            )
            for s in 0..<windowSize {
                for d in 0..<dims {
                    let idx = s * strideSeq + d * strideDim
                    out[s * dims + d] = Float(base[idx])
                }
            }
        case .float32:
            let base = array.dataPointer.bindMemory(
                to: Float.self, capacity: array.count
            )
            for s in 0..<windowSize {
                for d in 0..<dims {
                    let idx = s * strideSeq + d * strideDim
                    out[s * dims + d] = base[idx]
                }
            }
        case .float64:
            let base = array.dataPointer.bindMemory(
                to: Double.self, capacity: array.count
            )
            for s in 0..<windowSize {
                for d in 0..<dims {
                    let idx = s * strideSeq + d * strideDim
                    out[s * dims + d] = Float(base[idx])
                }
            }
        default:
            preconditionFailure(
                "unsupported CoreML output dtype \(array.dataType.rawValue) — convert-xtr-to-coreml.py emits FP16"
            )
        }
        return out
    }

    /// Read a `[1, windowSize, dims]` FP16 raw-projection tensor and
    /// return the per-row L2 norm (length `windowSize`). The norms are
    /// computed in FP32 so the threshold comparison is stable.
    static func readRowL2Norms(
        _ array: MLMultiArray,
        windowSize: Int,
        dims: Int
    ) -> [Float] {
        let flat = readEmbeddingTensor(array, windowSize: windowSize, dims: dims)
        var norms = [Float](repeating: 0, count: windowSize)
        for s in 0..<windowSize {
            var sum: Float = 0
            for d in 0..<dims {
                let v = flat[s * dims + d]
                sum += v * v
            }
            norms[s] = sum.squareRoot()
        }
        return norms
    }
}
#endif
