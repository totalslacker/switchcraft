// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Controls how an `Embedder` handles inputs whose token count exceeds `maxInputTokens`.
///
/// The two policies match the primary consumer trade-offs for retrieval workloads:
///
/// - `.truncate` (default): Silently clips the token sequence to the first
///   `maxInputTokens` tokens and encodes the prefix. This is the established
///   convention (Hugging Face `truncation=True, max_length=...`) and is appropriate
///   for search, classification, and bulk-index pipelines where prefix content is
///   informative and silent data loss is acceptable.
///
/// - `.reject`: Throws `EmbedderError.inputTooLarge(actual:max:)` so the caller can
///   decide whether to skip, summarise, or split the input. Use this when silent
///   truncation would violate the application's correctness guarantees.
public enum EmbedderOverflowPolicy: Sendable, Equatable, Hashable {
    /// Silently truncate the token sequence to `maxInputTokens` elements and encode
    /// the prefix. No error is thrown; embeddings are returned for the truncated input.
    case truncate

    /// Throw `EmbedderError.inputTooLarge(actual:max:)` without calling the underlying
    /// model. The caller is responsible for splitting, summarising, or skipping the input.
    case reject
}
