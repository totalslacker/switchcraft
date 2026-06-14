// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Encodes text into per-token embedding vectors used by the index and
/// search pipelines.
///
/// Implementations adapt a model (T5/CoreML, llama.cpp, a remote API, a
/// deterministic mock) to the shape Switchcraft consumes. The contract:
///
/// - `encode(_:)` returns a row-major `n × dims` `[Float]` where `n` is
///   the number of tokens. The buffer length MUST be a multiple of `dims`.
/// - The function MUST be deterministic for the same input — the index
///   and search engine rely on byte-identical embeddings to produce
///   reproducible results across runs.
/// - Empty or whitespace-only text MAY return an empty array.
/// - `modelIdentifier` is recorded on every persisted `ChunkRecord.model`
///   so future reads can detect mismatches with the model used at write
///   time. Pick a stable identifier (model file hash, tag, version).
///
/// `Sendable` is required because `SwitchcraftStore` (an actor) holds a
/// reference across `await` boundaries.
public protocol Embedder: Sendable {
    /// Dimensionality of token embeddings produced by this model. Must be
    /// positive and even (the Q4 residual codec packs two nibbles per byte).
    var dims: Int { get }

    /// Stable identifier for the model. Recorded on `ChunkRecord.model`.
    var modelIdentifier: String { get }

    /// Maximum number of tokens the embedder will process in a single `encode`
    /// call. Inputs that tokenise to more tokens than this limit are handled
    /// according to the conformer's configured `EmbedderOverflowPolicy`.
    ///
    /// Conformers should expose this as a `nonisolated let` stored property so
    /// callers can read the limit without entering an actor.
    var maxInputTokens: Int { get }

    /// Encode `text` into a flat row-major `n × dims` per-token embedding
    /// matrix. Returns an empty array for empty / whitespace-only text.
    ///
    /// - Throws: `EmbedderError.inputTooLarge(actual:max:)` when the token
    ///   count exceeds `maxInputTokens` and the overflow policy is `.reject`.
    func encode(_ text: String) async throws -> [Float]

    /// Encode a *query* string, optionally filtering out token embeddings whose
    /// decoded surface form is too short to be discriminative.
    ///
    /// The default implementation ignores `minSurfaceFormLength` and forwards
    /// to `encode(_:)`, providing Witchcraft-compatible behaviour for embedders
    /// that do not have access to token surface forms.
    ///
    /// Token-aware embedders (e.g. `T5CoreMLEmbedder`) override this method to
    /// suppress common short subword fragments (e.g. `"t"`, `"le"`, `"by"`)
    /// that inflate noise scores at corpus scale. See ADR 028.
    ///
    /// - Parameters:
    ///   - text: The query string to encode.
    ///   - minSurfaceFormLength: Keep only token positions whose decoded
    ///     surface form has `count > minSurfaceFormLength`. Pass `0` (default)
    ///     to disable filtering and preserve Witchcraft parity.
    /// - Throws: Same conditions as `encode(_:)`.
    func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float]

    /// Release any accumulated runtime resources (e.g. CoreML ANE IOSurface
    /// pool) and prepare the embedder for the next batch of inference calls.
    ///
    /// This is an explicit, consumer-controlled Layer 0 flush. Callers
    /// performing sustained bulk-inference workloads (indexing thousands of
    /// documents in a single process) should call `resetState()` between
    /// batches to prevent ANE IOSurface pool exhaustion. See ADR 031.
    ///
    /// **Behaviour contract:**
    /// - The reset is functionally transparent: `encode(s)` returns
    ///   byte-identical `[Float]` before and after a `resetState()` call for
    ///   the same input string `s`. Model weights and tokenization are
    ///   unaffected; only runtime resource pools are flushed.
    /// - Concurrent `encode()` calls already in-flight when `resetState()` is
    ///   called complete before the reset executes. `encode()` calls that
    ///   arrive after the reset starts queue and resume after it finishes.
    /// - On `T5CoreMLEmbedder`: completes in ≤ 5 s on M-series hardware
    ///   (dominated by CoreML model reload). Budget for this latency between
    ///   batches.
    /// - On `T5CoreMLEmbedder`: if the internal model reload fails, the error
    ///   propagates to the caller. The old (stale) model reference is retained;
    ///   subsequent `encode()` calls will use it and may also fail.
    /// - On `T5CoreMLEmbedder`: calling `resetState()` resets the internal
    ///   proactive-reload counter to 0. The next Layer 2 proactive reload fires
    ///   `reloadInterval` encodes after the explicit reset.
    /// - The default implementation is a no-op. `MockEmbedder`,
    ///   `T5MetalEmbedder`, and other conformers that do not accumulate
    ///   runtime IOSurface state inherit this no-op at zero cost.
    ///
    /// - Throws: Errors from the underlying resource-reset mechanism (e.g.
    ///   model reload failure in `T5CoreMLEmbedder`). The default no-op
    ///   never throws.
    func resetState() async throws
}

// MARK: - Default implementations

public extension Embedder {
    func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float] {
        try await encode(text)
    }

    func resetState() async throws {
        // No-op: most embedders do not accumulate runtime resource state.
    }
}
