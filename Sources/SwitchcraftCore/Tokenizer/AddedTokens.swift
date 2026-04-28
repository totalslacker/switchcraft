import Foundation

/// A single entry from `tokenizer.json`'s `added_tokens` array. These are tokens
/// the tokenizer is contractually required to emit at a fixed ID whenever they
/// appear in the input, regardless of what the underlying model would have
/// segmented to. HuggingFace handles them via a pre-segmentation pass before
/// normalization (for `normalized: false` entries) or before pre-tokenization
/// (for `normalized: true` entries).
///
/// For `xtr-base-en` every entry has `normalized: false, single_word: false,
/// lstrip: false, rstrip: false`, and the entry's `id` happens to coincide with
/// the same string in the Unigram vocab — so Viterbi alone would emit the same
/// IDs by coincidence. The pre-segmentation pass makes this correct for any
/// future tokenizer where that coincidence does not hold.
struct AddedToken: Sendable, Equatable {
    let id: Int32
    let content: String
    let normalized: Bool
    let singleWord: Bool
    let lstrip: Bool
    let rstrip: Bool
    let special: Bool
}

/// One piece of a string after splitting on added-token matches: either an
/// added-token ID to emit directly, or a raw substring still owed to the rest
/// of the encode pipeline (normalize → pre-tokenize → Viterbi).
enum AddedTokenSpan: Sendable, Equatable {
    case added(Int32)
    case text(String)
}

/// Linear-scan splitter: at each position, tries the configured `AddedToken`
/// entries in descending content-length order so longer matches win on
/// overlap. Honors per-entry `single_word`, `lstrip`, and `rstrip` flags per
/// HuggingFace `added_vocabulary.rs` semantics. The `normalized` flag is the
/// caller's responsibility: build one matcher from `normalized: false` entries
/// for pre-normalization splitting and a separate matcher from
/// `normalized: true` entries for post-normalization splitting.
struct AddedTokenMatcher: Sendable {
    private let entries: [AddedToken]

    init(entries: [AddedToken]) {
        // Empty-content entries would loop forever in the splitter; defensive
        // skip in case of malformed JSON.
        let nonEmpty = entries.filter { !$0.content.isEmpty }
        self.entries = nonEmpty.sorted { $0.content.count > $1.content.count }
    }

    var isEmpty: Bool { entries.isEmpty }

    /// Split `text` into a sequence of `.added(id)` and `.text(raw)` spans.
    /// When no added-token match occurs anywhere, returns `[.text(text)]`
    /// (or `[]` for empty input) so the caller's encode pipeline is byte-
    /// identical to a no-pre-segmentation path.
    func split(_ text: String) -> [AddedTokenSpan] {
        if text.isEmpty { return [] }
        if entries.isEmpty { return [.text(text)] }

        var result: [AddedTokenSpan] = []
        var unmatchedStart = text.startIndex
        var pos = text.startIndex

        while pos < text.endIndex {
            if let match = matchAt(pos, in: text, unmatchedStart: unmatchedStart) {
                if match.startIndex > unmatchedStart {
                    result.append(.text(String(text[unmatchedStart..<match.startIndex])))
                }
                result.append(.added(match.id))
                unmatchedStart = match.endIndex
                pos = match.endIndex
            } else {
                pos = text.index(after: pos)
            }
        }

        if unmatchedStart < text.endIndex {
            result.append(.text(String(text[unmatchedStart..<text.endIndex])))
        }

        return result
    }

    private struct Match {
        let id: Int32
        let startIndex: String.Index
        let endIndex: String.Index
    }

    private func matchAt(
        _ pos: String.Index,
        in text: String,
        unmatchedStart: String.Index
    ) -> Match? {
        for entry in entries {
            let content = entry.content
            guard let end = text.index(pos, offsetBy: content.count, limitedBy: text.endIndex) else {
                continue
            }
            guard text[pos..<end] == content else { continue }

            if entry.singleWord {
                // Left boundary: if the content begins with a word char, the
                // character immediately before the match (if any) must NOT be
                // a word char. Mirror logic on the right.
                if let firstScalar = content.unicodeScalars.first, isWordChar(firstScalar),
                   pos > text.startIndex {
                    let beforeIdx = text.index(before: pos)
                    if let beforeScalar = text[beforeIdx].unicodeScalars.first,
                       isWordChar(beforeScalar) {
                        continue
                    }
                }
                if let lastScalar = content.unicodeScalars.last, isWordChar(lastScalar),
                   end < text.endIndex {
                    if let afterScalar = text[end].unicodeScalars.first,
                       isWordChar(afterScalar) {
                        continue
                    }
                }
            }

            var matchStart = pos
            var matchEnd = end

            if entry.lstrip {
                while matchStart > unmatchedStart {
                    let prev = text.index(before: matchStart)
                    if text[prev].isWhitespace {
                        matchStart = prev
                    } else {
                        break
                    }
                }
            }

            if entry.rstrip {
                while matchEnd < text.endIndex && text[matchEnd].isWhitespace {
                    matchEnd = text.index(after: matchEnd)
                }
            }

            return Match(id: entry.id, startIndex: matchStart, endIndex: matchEnd)
        }
        return nil
    }
}

/// HuggingFace's default `is_word_char` matches ASCII alphanumerics plus
/// underscore for `\w`-style boundaries. `xtr-base-en` does not exercise the
/// `single_word` flag, so this stays ASCII-only until a tokenizer that needs
/// broader Unicode word semantics arrives. See HF `added_vocabulary.rs`.
@inline(__always)
private func isWordChar(_ scalar: Unicode.Scalar) -> Bool {
    let v = scalar.value
    return (v >= 0x30 && v <= 0x39)   // 0-9
        || (v >= 0x41 && v <= 0x5A)   // A-Z
        || (v >= 0x61 && v <= 0x7A)   // a-z
        || v == 0x5F                  // _
}
