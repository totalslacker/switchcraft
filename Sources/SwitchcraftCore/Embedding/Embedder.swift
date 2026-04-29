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

    /// Encode `text` into a flat row-major `n × dims` per-token embedding
    /// matrix. Returns an empty array for empty / whitespace-only text.
    func encode(_ text: String) async throws -> [Float]
}
