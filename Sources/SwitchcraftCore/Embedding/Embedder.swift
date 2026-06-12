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
}

// MARK: - Default implementations

public extension Embedder {
    func encodeQuery(_ text: String, minSurfaceFormLength: Int) async throws -> [Float] {
        try await encode(text)
    }
}
