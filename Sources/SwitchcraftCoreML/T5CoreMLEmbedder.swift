import Foundation
import SwitchcraftCore

#if canImport(CoreML)
import CoreML

/// `Embedder` conformance backed by a CoreML-converted T5 encoder + 768→128
/// projection layer (`google/xtr-base-en`). Tokenises with the injected
/// `Tokenizer`, runs sliding-window inference with stride 256 / window 512
/// for inputs longer than the model's fixed sequence length, drops
/// low-signal positions whose pre-normalisation L2 norm is below `minNorm`
/// (default 1.0, matching Witchcraft's `MIN_NORM`), and returns flat
/// row-major `[Float]` of length `m × dims`.
///
/// The CoreML model is loaded once at init and reused across calls. The
/// type is an `actor` because `MLModel.prediction(from:)` is synchronous
/// and not documented thread-safe across concurrent callers; the actor
/// gives exclusive access automatically and matches the rest of the
/// codebase's concurrency pattern (`Indexer`, `SearchEngine`,
/// `SwitchcraftStore`).
///
/// ### Asset distribution
///
/// The `.mlpackage` is **not committed** to the repository (~80 MB; Git
/// LFS is incompatible with SwiftPM's resolver). Run
/// `scripts/convert-xtr-to-coreml.py` once to produce it, place it at
/// the path of your choice, and pass the path to `init(modelURL:...)`.
/// Tests skip when the asset is absent (see `CoreMLAsset`).
public actor T5CoreMLEmbedder: Embedder {

    // MARK: - Stored state

    /// Embedding dimensionality. Matches XTR-base-en's projection output.
    public nonisolated let dims: Int

    /// Stable identifier recorded on every `ChunkRecord.model`. Defaults
    /// to `"google/xtr-base-en@v1"`; callers wanting precise revision
    /// tracking pass `"google/xtr-base-en@<sha>"` at init.
    public nonisolated let modelIdentifier: String

    /// Pre-normalisation L2 norm threshold. Positions whose merged raw
    /// norm is below this value are dropped. `1.0` matches Witchcraft.
    public nonisolated let minNorm: Float

    /// Maximum tokens per inference (the CoreML model's fixed sequence
    /// length). `512` for the canonical XTR-base-en CoreML asset.
    public nonisolated let windowSize: Int

    /// Stride between consecutive window starts when the input exceeds
    /// `windowSize`. `256` matches Witchcraft's sliding-window stride.
    public nonisolated let stride: Int

    private let tokenizer: Tokenizer
    private let model: MLModel

    // MARK: - Init

    /// Load a CoreML model from a `.mlpackage` URL (or compiled `.mlmodelc`).
    public init(
        modelURL: URL,
        tokenizer: Tokenizer,
        computeUnits: MLComputeUnits = .all,
        modelIdentifier: String = "google/xtr-base-en@v1",
        dims: Int = 128,
        windowSize: Int = 512,
        stride: Int = 256,
        minNorm: Float = 1.0
    ) async throws {
        precondition(dims > 0 && dims % 2 == 0,
                     "dims must be positive and even (Q4 codec packs two nibbles per byte)")
        precondition(windowSize > 0)
        precondition(stride > 0 && stride <= windowSize)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        // .mlpackage paths must be compiled before MLModel(contentsOf:) can load
        // them; .mlmodelc paths are already compiled. `MLModel.compileModel(at:)`
        // is async on macOS 13+ / iOS 16+; awaiting it inside the actor's init
        // is fine because the actor isn't observable until init returns.
        let compiledURL: URL
        let pathExt = modelURL.pathExtension.lowercased()
        if pathExt == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try await MLModel.compileModel(at: modelURL)
        }

        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.tokenizer = tokenizer
        self.dims = dims
        self.modelIdentifier = modelIdentifier
        self.minNorm = minNorm
        self.windowSize = windowSize
        self.stride = stride
    }

    /// Convenience init that resolves the model URL from a `Bundle`.
    public init(
        bundle: Bundle,
        resourceName: String,
        resourceExtension: String? = "mlpackage",
        tokenizer: Tokenizer,
        computeUnits: MLComputeUnits = .all,
        modelIdentifier: String = "google/xtr-base-en@v1",
        dims: Int = 128,
        windowSize: Int = 512,
        stride: Int = 256,
        minNorm: Float = 1.0
    ) async throws {
        guard let url = bundle.url(
            forResource: resourceName,
            withExtension: resourceExtension
        ) else {
            throw T5CoreMLEmbedderError.modelNotFoundInBundle(
                bundle: bundle.bundleIdentifier ?? bundle.bundlePath,
                resource: resourceName,
                extension: resourceExtension ?? ""
            )
        }
        try await self.init(
            modelURL: url,
            tokenizer: tokenizer,
            computeUnits: computeUnits,
            modelIdentifier: modelIdentifier,
            dims: dims,
            windowSize: windowSize,
            stride: stride,
            minNorm: minNorm
        )
    }

    // MARK: - Embedder

    public func encode(_ text: String) async throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [] }

        let tokens = try tokenizer.encode(text, addSpecialTokens: true)
        if tokens.isEmpty { return [] }

        let starts = SlidingWindow.plan(
            tokenCount: tokens.count,
            windowSize: windowSize,
            stride: stride
        )

        var perWindowNormalised: [[Float]] = []
        var perWindowRawNorms: [[Float]] = []
        perWindowNormalised.reserveCapacity(starts.count)
        perWindowRawNorms.reserveCapacity(starts.count)

        for start in starts {
            let lo = start
            let hi = min(start + windowSize, tokens.count)
            let slice = tokens[lo..<hi]
            let inputArray = try CoreMLModelIO.makeInputIDs(
                tokens: slice,
                windowSize: windowSize
            )
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                CoreMLInputName.inputIDs: MLFeatureValue(multiArray: inputArray)
            ])
            let output = try model.prediction(from: provider)

            guard
                let normalisedFV = output.featureValue(for: CoreMLOutputName.normalised),
                let normalisedArray = normalisedFV.multiArrayValue
            else {
                throw T5CoreMLEmbedderError.missingOutput(name: CoreMLOutputName.normalised)
            }
            guard
                let rawFV = output.featureValue(for: CoreMLOutputName.rawProjected),
                let rawArray = rawFV.multiArrayValue
            else {
                throw T5CoreMLEmbedderError.missingOutput(name: CoreMLOutputName.rawProjected)
            }

            let normalised = CoreMLModelIO.readEmbeddingTensor(
                normalisedArray,
                windowSize: windowSize,
                dims: dims
            )
            let rawNorms = CoreMLModelIO.readRowL2Norms(
                rawArray,
                windowSize: windowSize,
                dims: dims
            )
            perWindowNormalised.append(normalised)
            perWindowRawNorms.append(rawNorms)
        }

        let merged = SlidingWindow.merge(
            windowStarts: starts,
            windowSize: windowSize,
            tokenCount: tokens.count,
            dims: dims,
            windowNormalised: perWindowNormalised,
            windowRawNorms: perWindowRawNorms,
            minNorm: minNorm
        )
        return merged.embeddings
    }
}

/// Errors surfaced by `T5CoreMLEmbedder`.
public enum T5CoreMLEmbedderError: Error, Sendable, Equatable {
    /// `init(bundle:resourceName:...)` could not resolve a URL for the
    /// given resource name in the given bundle.
    case modelNotFoundInBundle(bundle: String, resource: String, extension: String)
    /// The CoreML model produced a feature dictionary that did not
    /// contain the expected output. Check that the `.mlpackage` was
    /// produced by `scripts/convert-xtr-to-coreml.py` (which emits
    /// `raw_projected` and `normalised`).
    case missingOutput(name: String)
}

#endif
