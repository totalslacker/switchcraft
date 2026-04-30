// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Metaspace pre-tokenizer: replaces leading whitespace of each word with ▁ (U+2581).
struct MetaspacePreTokenizer: Sendable {
    let replacement: Character    // U+2581 (▁)
    let prependScheme: PrependScheme

    enum PrependScheme: String, Sendable {
        case always, first, never
    }

    init(from json: [String: Any]) throws {
        if let rep = json["replacement"] as? String, let ch = rep.unicodeScalars.first {
            replacement = Character(ch)
        } else {
            replacement = "▁"
        }
        if let scheme = json["prepend_scheme"] as? String {
            prependScheme = PrependScheme(rawValue: scheme) ?? .always
        } else if let addPrefix = json["add_prefix_space"] as? Bool {
            prependScheme = addPrefix ? .always : .never
        } else {
            prependScheme = .always
        }
    }

    /// Split `s` into Metaspace-encoded pre-tokens: each word has its leading whitespace
    /// replaced by a single ▁ character. Returns an empty array for empty or all-whitespace input.
    func tokenize(_ s: String) -> [String] {
        guard !s.isEmpty else { return [] }
        var input = s
        if prependScheme == .always {
            input = " " + input
        }
        let tokens = splitAttachingLeadingWhitespace(input)
        return tokens.compactMap { token -> String? in
            guard let firstNonSpace = token.firstIndex(where: { !$0.isWhitespace }) else {
                return nil   // all-whitespace segment with no following word — drop it
            }
            return String(replacement) + String(token[firstNonSpace...])
        }
    }

    /// Split `s` so that each whitespace run is prepended to the following word.
    /// Returns an array of (whitespace + word) strings.
    private func splitAttachingLeadingWhitespace(_ s: String) -> [String] {
        var result: [String] = []
        var pendingWhitespace = ""
        var currentWord = ""

        for ch in s {
            if ch.isWhitespace {
                if !currentWord.isEmpty {
                    result.append(pendingWhitespace + currentWord)
                    pendingWhitespace = ""
                    currentWord = ""
                }
                pendingWhitespace.append(ch)
            } else {
                currentWord.append(ch)
            }
        }
        if !currentWord.isEmpty {
            result.append(pendingWhitespace + currentWord)
        }
        // Trailing whitespace with no following word is dropped (normalizer already stripped it)
        return result
    }
}
