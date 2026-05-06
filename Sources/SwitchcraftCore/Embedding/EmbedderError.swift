// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Errors thrown by `Embedder` implementations.
public enum EmbedderError: Error, Sendable, Equatable {
    /// The tokenised input length exceeded the embedder's `maxInputTokens` limit
    /// and the configured overflow policy is `.reject`.
    ///
    /// - Parameters:
    ///   - actual: The number of tokens produced by the tokenizer for the input.
    ///   - max: The embedder's `maxInputTokens` limit.
    case inputTooLarge(actual: Int, max: Int)
}
