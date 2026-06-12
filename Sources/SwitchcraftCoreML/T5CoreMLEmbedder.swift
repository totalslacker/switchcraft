// SPDX-License-Identifier: Apache-2.0
import Foundation
import os
import SwitchcraftCore

#if canImport(CoreML)
import CoreML

private let coreMLLogger = Logger(subsystem: "com.switchcraft.coreml", category: "CoreML")

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
/// ### Crash safety
///
/// CoreML internally raises Objective-C exceptions from failure sites such
/// as `MLE5BindEmptyMemoryObjectToPort`. Swift's do-catch does not intercept
/// ObjC exceptions — they previously propagated to std::terminate. Each call
/// to `predictor.predict(input:)` is now wrapped in `catchingNSException`,
/// which converts ObjC exceptions into `CoreMLNativeError.nativeException`
/// thrown values so callers can recover. A re-entrancy guard serialises
/// concurrent encode calls to prevent concurrent access to CoreML-internal
/// resources. Pass `failureLogURL` at init to receive structured crash
/// telemetry (JSONL) for each caught exception.
///
/// ### ANE IOSurface pool mitigation
///
/// Under sustained bulk-inference load the ANE IOSurface buffer pool exhausts
/// after ~388+ encodes on high-variance corpora, causing every subsequent
/// inference to fail with "Failed to allocate E5 buffer object." Three defence
/// layers work together:
///
/// 1. `autoreleasepool` drains CoreML-internal ObjC buffers after each window.
/// 2. Proactive model reload every `reloadInterval` encodes flushes accumulated
///    ANE resources before the pool is exhausted. Each reload takes 1–3 s on
///    ANE-capable hardware (CPU recompile is faster); tune `reloadInterval`
///    to balance stall frequency against pool pressure for your workload.
/// 3. On IOSurface failure: force-reload + ANE retry. If the ANE retry also
///    fails, the original error is logged in `failureLogURL` (when set) with
///    `category: "error"` and rethrown.
///
/// See ADR 021 for the full rationale.
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

    /// Maximum total token count across all sliding windows that the embedder
    /// will accept in a single `encode` call. Inputs that tokenise to more
    /// tokens than this value are handled according to `overflowPolicy`.
    ///
    /// Default `8 * windowSize` (4,096 for the standard 512-token window),
    /// yielding ~15 windows at stride 256 — well below the ~577-window burst
    /// that exhausted the ANE IOSurface pool. Must be ≥ `windowSize`.
    public nonisolated let maxInputTokens: Int

    /// Controls behaviour when a tokenised input exceeds `maxInputTokens`.
    /// `.truncate` (default) silently clips to the prefix; `.reject` throws
    /// `EmbedderError.inputTooLarge(actual:max:)`.
    public nonisolated let overflowPolicy: EmbedderOverflowPolicy

    private let tokenizer: Tokenizer
    private var predictor: any MLPredictor
    /// Recreates the main predictor on demand; used by proactive reload to
    /// flush accumulated ANE IOSurface resources.
    private let predictorFactory: @Sendable () throws -> any MLPredictor
    private let failureLogURL: URL?
    private var callCount: Int = 0
    /// Number of `encode` calls between proactive model reloads.
    ///
    /// Real-world evidence shows ANE IOSurface pool exhaustion at ~388 calls on
    /// high-variance corpora; 150 provides comfortable margin below that point.
    /// Model reload with ANE compilation takes 1–3 s on Apple Silicon — tune
    /// this value upward on workloads where that periodic stall is unacceptable,
    /// or downward on hardware where the pool is smaller.
    private let reloadInterval: Int

    // Re-entrancy guard: serialises concurrent encode calls so CoreML-internal
    // resources released at prediction-end are not raced by a second call.
    private var inFlight: Bool = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    // MARK: - Init

    /// Load a CoreML model from a `.mlpackage` URL (or compiled `.mlmodelc`).
    ///
    /// - Parameters:
    ///   - modelURL: filesystem URL of the `.mlpackage` or `.mlmodelc`.
    ///     `.mlpackage` paths are compiled in-process before load.
    ///   - tokenizer: HuggingFace `Tokenizer` matching the model's vocab.
    ///   - computeUnits: CoreML compute selector (CPU/GPU/ANE).
    ///   - modelIdentifier: stable string written to every `ChunkRecord.model`.
    ///   - dims: projection output dimensionality. Must be positive and even.
    ///   - windowSize: maximum tokens per inference (model fixed length).
    ///   - stride: sliding-window stride; must satisfy `0 < stride ≤ windowSize`.
    ///   - minNorm: pre-normalisation L2 norm threshold; positions below
    ///     this are dropped.
    ///   - failureLogURL: optional file URL for structured crash telemetry.
    ///     When non-nil and a `CoreMLNativeError.nativeException` is caught
    ///     during prediction, one JSONL row is appended (fields: timestamp,
    ///     name, reason, inputLength, callStack, category). The file is created
    ///     if absent. Pass `nil` (default) to disable all logging and file I/O.
    ///   - reloadInterval: how many `encode` calls to allow between proactive
    ///     model reloads. Each reload recreates the `MLModel` to flush
    ///     accumulated ANE IOSurface resources. Default `150` — see the stored
    ///     property doc-comment for tuning guidance. Existing callers that omit
    ///     this parameter are unaffected.
    ///   - maxInputTokens: maximum total token count accepted per `encode` call.
    ///     Inputs exceeding this limit are handled by `overflowPolicy`. Default
    ///     `8 * windowSize`. Must be ≥ `windowSize`.
    ///   - overflowPolicy: `.truncate` (default) clips oversized inputs to the
    ///     first `maxInputTokens` tokens; `.reject` throws
    ///     `EmbedderError.inputTooLarge(actual:max:)`.
    /// - Throws: any error from `MLModel.compileModel(at:)` or `MLModel(contentsOf:)`.
    public init(
        modelURL: URL,
        tokenizer: Tokenizer,
        computeUnits: MLComputeUnits = .all,
        modelIdentifier: String = "google/xtr-base-en@v1",
        dims: Int = 128,
        windowSize: Int = 512,
        stride: Int = 256,
        minNorm: Float = 1.0,
        failureLogURL: URL? = nil,
        reloadInterval: Int = 150,
        maxInputTokens: Int? = nil,
        overflowPolicy: EmbedderOverflowPolicy = .truncate
    ) async throws {
        precondition(dims > 0 && dims % 2 == 0,
                     "dims must be positive and even (Q4 codec packs two nibbles per byte)")
        precondition(windowSize > 0)
        precondition(stride > 0 && stride <= windowSize)
        precondition(reloadInterval > 0, "reloadInterval must be positive (used as modulo divisor)")
        let resolvedMaxInputTokens = maxInputTokens ?? 8 * windowSize
        precondition(resolvedMaxInputTokens >= windowSize,
                     "maxInputTokens must be >= windowSize (got \(resolvedMaxInputTokens) < \(windowSize))")

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

        // Capture value types so the closures are @Sendable without relying on
        // MLModelConfiguration's Sendability (it is a class; we avoid capturing
        // the instance and instead capture its scalar property).
        let capturedCompiledURL = compiledURL
        let capturedComputeUnits = computeUnits

        let factory: @Sendable () throws -> any MLPredictor = {
            let config = MLModelConfiguration()
            config.computeUnits = capturedComputeUnits
            return try MLModel(contentsOf: capturedCompiledURL, configuration: config)
        }

        self.predictorFactory = factory
        self.predictor = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.tokenizer = tokenizer
        self.dims = dims
        self.modelIdentifier = modelIdentifier
        self.minNorm = minNorm
        self.windowSize = windowSize
        self.stride = stride
        self.failureLogURL = failureLogURL
        self.reloadInterval = reloadInterval
        self.maxInputTokens = resolvedMaxInputTokens
        self.overflowPolicy = overflowPolicy
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
        minNorm: Float = 1.0,
        failureLogURL: URL? = nil,
        reloadInterval: Int = 150,
        maxInputTokens: Int? = nil,
        overflowPolicy: EmbedderOverflowPolicy = .truncate
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
            minNorm: minNorm,
            failureLogURL: failureLogURL,
            reloadInterval: reloadInterval,
            maxInputTokens: maxInputTokens,
            overflowPolicy: overflowPolicy
        )
    }

    /// Test-only init: inject an `MLPredictor` stub without loading a real model.
    ///
    /// `internal` — access from test targets via `@testable import SwitchcraftCoreML`.
    internal init(
        predictor: any MLPredictor,
        tokenizer: Tokenizer,
        dims: Int = 128,
        windowSize: Int = 512,
        stride: Int = 256,
        minNorm: Float = 1.0,
        modelIdentifier: String = "stub@v0",
        failureLogURL: URL? = nil,
        maxInputTokens: Int? = nil,
        overflowPolicy: EmbedderOverflowPolicy = .truncate
    ) {
        precondition(dims > 0 && dims % 2 == 0,
                     "dims must be positive and even")
        precondition(windowSize > 0)
        precondition(stride > 0 && stride <= windowSize)
        let resolvedMaxInputTokens = maxInputTokens ?? 8 * windowSize
        precondition(resolvedMaxInputTokens >= windowSize,
                     "maxInputTokens must be >= windowSize")
        let capturedPredictor = predictor
        self.predictorFactory = { capturedPredictor }
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.dims = dims
        self.windowSize = windowSize
        self.stride = stride
        self.minNorm = minNorm
        self.modelIdentifier = modelIdentifier
        self.failureLogURL = failureLogURL
        self.reloadInterval = 150
        self.maxInputTokens = resolvedMaxInputTokens
        self.overflowPolicy = overflowPolicy
    }

    /// Test-only init: inject a factory for predictor lifecycle testing.
    ///
    /// Use this variant when the test must verify model reload behaviour or
    /// the Layer 3 reactive reload + ANE retry path. The factory is called
    /// once during init and again on each proactive or reactive reload.
    ///
    /// `internal` — access from test targets via `@testable import SwitchcraftCoreML`.
    internal init(
        predictorFactory: @escaping @Sendable () throws -> any MLPredictor,
        tokenizer: Tokenizer,
        dims: Int = 128,
        windowSize: Int = 512,
        stride: Int = 256,
        minNorm: Float = 1.0,
        modelIdentifier: String = "stub@v0",
        failureLogURL: URL? = nil,
        reloadInterval: Int = 150,
        maxInputTokens: Int? = nil,
        overflowPolicy: EmbedderOverflowPolicy = .truncate
    ) throws {
        precondition(dims > 0 && dims % 2 == 0,
                     "dims must be positive and even")
        precondition(windowSize > 0)
        precondition(stride > 0 && stride <= windowSize)
        precondition(reloadInterval > 0, "reloadInterval must be positive (used as modulo divisor)")
        let resolvedMaxInputTokens = maxInputTokens ?? 8 * windowSize
        precondition(resolvedMaxInputTokens >= windowSize,
                     "maxInputTokens must be >= windowSize")
        self.predictorFactory = predictorFactory
        self.predictor = try predictorFactory()
        self.tokenizer = tokenizer
        self.dims = dims
        self.windowSize = windowSize
        self.stride = stride
        self.minNorm = minNorm
        self.modelIdentifier = modelIdentifier
        self.failureLogURL = failureLogURL
        self.reloadInterval = reloadInterval
        self.maxInputTokens = resolvedMaxInputTokens
        self.overflowPolicy = overflowPolicy
    }

    // MARK: - Embedder

    /// Encode `text` to a flat row-major `m × dims` `[Float]`. Returns
    /// an empty array for empty / whitespace-only input.
    ///
    /// ObjC exceptions from CoreML are converted to `CoreMLNativeError`
    /// and thrown rather than crashing the host process. IOSurface allocation
    /// failures trigger a reactive model reload + ANE retry (see class doc-comment);
    /// callers only receive an error if the ANE retry also fails.
    ///
    /// - Throws: `EmbedderError.inputTooLarge(actual:max:)` when the token
    ///   count exceeds `maxInputTokens` and `overflowPolicy` is `.reject`;
    ///   `T5CoreMLEmbedderError.missingOutput` if the CoreML model does not
    ///   produce the expected feature dictionary;
    ///   `CoreMLNativeError.nativeException` if CoreML raises an internal
    ///   ObjC exception that the embedder cannot recover from; any
    ///   tokenizer-originated error.
    public func encode(_ text: String) async throws -> [Float] {
        try await _encodeImpl(text, minSurfaceFormLength: 0)
    }

    /// Encode a query string with optional short-token filtering. See ADR 028.
    public func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float] {
        try await _encodeImpl(text, minSurfaceFormLength: minSurfaceFormLength)
    }

    // MARK: - Private helpers

    /// Shared implementation for `encode` and `encodeQuery`. Contains the
    /// re-entrancy guard, proactive reload, sliding-window inference, and
    /// optional surface-form length filter.
    ///
    /// Both `encode()` and `encodeQuery()` are thin wrappers here to avoid
    /// re-entrancy deadlock: calling `self.encode()` from `encodeQuery()` while
    /// `inFlight == true` would queue on the waiters list and deadlock.
    private func _encodeImpl(_ text: String, minSurfaceFormLength: Int) async throws -> [Float] {
        // Re-entrancy guard: if another encode is in progress, queue and wait.
        if inFlight {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                waiters.append(cont)
            }
        }
        inFlight = true
        defer {
            if let next = waiters.first {
                waiters = Array(waiters.dropFirst())
                inFlight = false
                next.resume()
            } else {
                inFlight = false
            }
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return [] }

        var tokens = try tokenizer.encode(text, addSpecialTokens: true)
        if tokens.isEmpty { return [] }

        // Overflow guard: prevent oversized inputs from generating hundreds of
        // sliding windows and exhausting the ANE IOSurface buffer pool (ADR 022).
        if tokens.count > maxInputTokens {
            switch overflowPolicy {
            case .truncate:
                tokens = Array(tokens.prefix(maxInputTokens))
            case .reject:
                throw EmbedderError.inputTooLarge(actual: tokens.count, max: maxInputTokens)
            }
        }

        // Proactive model reload: recreate the predictor every reloadInterval
        // encodes to flush accumulated ANE IOSurface resources.
        // Counter increments only for real inference calls (whitespace-only inputs
        // are excluded so transient noise doesn't skew the reload cadence).
        callCount += 1
        if callCount % reloadInterval == 0 {
            do {
                self.predictor = try predictorFactory()
                coreMLLogger.info(
                    "T5CoreMLEmbedder: reloaded model at encode #\(self.callCount, privacy: .public) (reloadInterval=\(self.reloadInterval, privacy: .public))"
                )
            } catch {
                coreMLLogger.error(
                    "T5CoreMLEmbedder: model reload failed at encode #\(self.callCount, privacy: .public): \(error, privacy: .public)"
                )
                // Keep existing predictor; if it also fails, the error surfaces normally.
            }
        }

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

            let output = try predictWindow(
                provider: provider,
                inputLength: text.count,
                windowTokenCount: hi - lo
            )

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

        // Surface-form length filter (ADR 028): drop query token positions whose
        // decoded surface form is too short to be discriminative. At default
        // minSurfaceFormLength=0 the fast path returns all embeddings unchanged.
        guard minSurfaceFormLength > 0 else {
            return merged.embeddings
        }

        var filtered: [Float] = []
        filtered.reserveCapacity(merged.keptPositions.count * dims)
        for (i, pos) in merged.keptPositions.enumerated() {
            let tokenID = tokens[pos]
            let surfaceForm = (try? tokenizer.decode([tokenID])) ?? ""
            if surfaceForm.count > minSurfaceFormLength {
                let offset = i * dims
                filtered.append(contentsOf: merged.embeddings[offset..<(offset + dims)])
            }
        }
        return filtered
    }


    /// Run one window prediction with autoreleasepool drainage and reactive reload + ANE retry.
    private func predictWindow(
        provider: MLDictionaryFeatureProvider,
        inputLength: Int,
        windowTokenCount: Int
    ) throws -> any MLFeatureProvider {
        let windowStart = Date()
        do {
            let result = try autoreleasepool {
                try catchingNSException { try self.predictor.predict(input: provider) }
            }
            let elapsed = Int(Date().timeIntervalSince(windowStart) * 1000)
            coreMLLogger.info(
                "predictWindow: \(windowTokenCount, privacy: .public) tokens, \(elapsed, privacy: .public)ms"
            )
            return result
        } catch let nativeError as CoreMLNativeError {
            guard isIOSurfaceExhaustion(nativeError) else {
                if let logURL = failureLogURL {
                    logNativeException(nativeError, inputLength: inputLength, to: logURL)
                }
                throw nativeError
            }

            // Layer 3 — Reactive reload + ANE retry: force-reload the predictor
            // and retry on ANE.
            do {
                self.predictor = try predictorFactory()
                let retryResult = try autoreleasepool {
                    try catchingNSException { try self.predictor.predict(input: provider) }
                }
                let elapsed = Int(Date().timeIntervalSince(windowStart) * 1000)
                coreMLLogger.info(
                    "predictWindow (ANE retry): \(windowTokenCount, privacy: .public) tokens, \(elapsed, privacy: .public)ms"
                )
                return retryResult
            } catch {
                if let logURL = failureLogURL {
                    logNativeException(nativeError, inputLength: inputLength, to: logURL)
                }
                throw nativeError
            }
        }
    }

    private func isIOSurfaceExhaustion(_ error: CoreMLNativeError) -> Bool {
        guard case .nativeException(_, let reason, _) = error else { return false }
        return reason.contains("IOSurface") || reason.contains("E5 buffer")
    }

    private func logNativeException(
        _ error: CoreMLNativeError,
        inputLength: Int,
        to url: URL
    ) {
        guard case .nativeException(let name, let reason, let callStack) = error else { return }

        let firstFrames = callStack.prefix(5)

        coreMLLogger.error(
            "🔴 [COREML-CRASH] name=\(name, privacy: .public) reason=\(reason, privacy: .public) input_len=\(inputLength, privacy: .public)"
        )
        for frame in firstFrames {
            coreMLLogger.error("\(frame, privacy: .public)")
        }

        appendJSONLRow(
            name: name,
            reason: reason,
            inputLength: inputLength,
            callStack: Array(firstFrames),
            category: "error",
            to: url
        )
    }

    private func appendJSONLRow(
        name: String,
        reason: String,
        inputLength: Int,
        callStack: [String],
        category: String = "error",
        cpuErrorName: String? = nil,
        cpuErrorReason: String? = nil,
        cpuCallStack: [String]? = nil,
        to url: URL
    ) {
        let formatter = ISO8601DateFormatter()
        let timestamp = formatter.string(from: Date())

        let entry = CoreMLFailureLogEntry(
            callStack: callStack,
            category: category,
            cpuCallStack: cpuCallStack,
            cpuErrorName: cpuErrorName,
            cpuErrorReason: cpuErrorReason,
            inputLength: inputLength,
            name: name,
            reason: reason,
            timestamp: timestamp
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(entry),
              var line = String(data: data, encoding: .utf8) else { return }
        line += "\n"
        guard let lineData = line.data(using: .utf8) else { return }

        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: lineData)
    }
}

// JSONL row appended by `appendJSONLRow`; key order is sorted by `JSONEncoder.outputFormatting = .sortedKeys`.
private struct CoreMLFailureLogEntry: Encodable {
    let callStack: [String]
    let category: String
    let cpuCallStack: [String]?
    let cpuErrorName: String?
    let cpuErrorReason: String?
    let inputLength: Int
    let name: String
    let reason: String
    let timestamp: String
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
