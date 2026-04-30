// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Swift port of the HuggingFace `tokenizers` Unigram (SentencePiece) pipeline for
/// `google/xtr-base-en`. Loads a `tokenizer.json` and produces token IDs that match
/// the upstream Rust crate bit-exactly.
///
/// Pipeline: Normalizer (Precompiled → Strip → Replace) → Metaspace pre-tokenizer →
/// Unigram model (trie-based Viterbi) → TemplateProcessing post-processor → Decoder.
public struct Tokenizer: Sendable {
    private let normalizer: any Normalizer
    private let preTokenizer: MetaspacePreTokenizer
    private let model: UnigramModel
    private let postProcessor: TemplatePostProcessor
    private let metaspaceDecoder: MetaspaceDecoder
    private let idToToken: [Int32: String]
    private let specialTokenIDs: Set<Int32>
    /// All `added_tokens` entries from the source `tokenizer.json`. Exposed at
    /// internal visibility so the test target's `@testable import` can iterate
    /// them for the parity guard test (every entry must round-trip to its own
    /// declared id with `addSpecialTokens: false`).
    let addedTokens: [AddedToken]
    private let preNormMatcher: AddedTokenMatcher
    private let postNormMatcher: AddedTokenMatcher

    /// Load a tokenizer from a HuggingFace `tokenizer.json` file at `path`.
    ///
    /// - Throws: `TokenizerError.fileNotFound` if `path` does not exist,
    ///   `TokenizerError.ioError` if the file cannot be read,
    ///   `TokenizerError.malformedJSON` if the JSON is invalid,
    ///   `TokenizerError.missingField` / `.unsupportedComponent` if the
    ///   pipeline shape is not the expected Unigram + Metaspace stack.
    public init(contentsOf path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw TokenizerError.fileNotFound(path: path)
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            // The file exists per the check above; a failure here is an I/O or
            // permissions problem, not a missing file. Preserve the underlying
            // cause for diagnostics rather than papering over it as fileNotFound.
            let nsError = error as NSError
            if nsError.domain == NSCocoaErrorDomain,
               nsError.code == CocoaError.fileReadNoSuchFile.rawValue {
                throw TokenizerError.fileNotFound(path: path)
            }
            throw TokenizerError.ioError(path: path, underlying: error)
        }
        try self.init(data: data)
    }

    /// Load a tokenizer from a raw `tokenizer.json` payload.
    ///
    /// - Throws: same errors as `init(contentsOf:)` other than the
    ///   filesystem-related cases.
    public init(data: Data) throws {
        let raw: Any
        do {
            raw = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw TokenizerError.malformedJSON(underlying: error)
        }
        guard let root = raw as? [String: Any] else {
            throw TokenizerError.malformedJSON(underlying: ParseError.rootNotObject)
        }

        // Normalizer
        guard let normJSON = root["normalizer"] as? [String: Any] else {
            throw TokenizerError.missingField("normalizer")
        }
        normalizer = try buildNormalizer(normJSON)

        // PreTokenizer (Metaspace)
        guard let preJSON = root["pre_tokenizer"] as? [String: Any] else {
            throw TokenizerError.missingField("pre_tokenizer")
        }
        guard (preJSON["type"] as? String) == "Metaspace" else {
            throw TokenizerError.unsupportedComponent("pre_tokenizer must be Metaspace")
        }
        preTokenizer = try MetaspacePreTokenizer(from: preJSON)

        // Model (Unigram)
        guard let modelJSON = root["model"] as? [String: Any] else {
            throw TokenizerError.missingField("model")
        }
        guard (modelJSON["type"] as? String) == "Unigram" else {
            throw TokenizerError.unsupportedComponent("model must be Unigram")
        }
        let unigram = try UnigramModel(modelJSON: modelJSON)
        model = unigram

        // Parse the full `added_tokens` array, including all per-entry flags.
        // These drive (a) the post-processor's name→id template lookup,
        // (b) pre-segmentation matching during encode, and (c) the decode-side
        // id→string map. PR #7 only kept `content → id`; that was correct for
        // xtr-base-en by coincidence (every added-token id also lives in the
        // Unigram vocab at the same id). Issue #8 makes the priority match
        // explicit so any future tokenizer behaves correctly without relying
        // on that coincidence.
        var addedTokens: [AddedToken] = []
        var addedTokenIDs: [String: Int32] = [:]
        if let added = root["added_tokens"] as? [[String: Any]] {
            for entry in added {
                guard let name = entry["content"] as? String,
                      let idInt = entry["id"] as? Int else {
                    continue
                }
                let id = Int32(idInt)
                addedTokenIDs[name] = id
                addedTokens.append(AddedToken(
                    id: id,
                    content: name,
                    normalized: entry["normalized"] as? Bool ?? false,
                    singleWord: entry["single_word"] as? Bool ?? false,
                    lstrip: entry["lstrip"] as? Bool ?? false,
                    rstrip: entry["rstrip"] as? Bool ?? false,
                    special: entry["special"] as? Bool ?? false
                ))
            }
        }
        self.addedTokens = addedTokens
        self.preNormMatcher = AddedTokenMatcher(entries: addedTokens.filter { !$0.normalized })
        self.postNormMatcher = AddedTokenMatcher(entries: addedTokens.filter { $0.normalized })
        self.specialTokenIDs = Set(addedTokens.lazy.filter { $0.special }.map { $0.id })

        // PostProcessor (TemplateProcessing)
        guard let postJSON = root["post_processor"] as? [String: Any] else {
            throw TokenizerError.missingField("post_processor")
        }
        guard (postJSON["type"] as? String) == "TemplateProcessing" else {
            throw TokenizerError.unsupportedComponent("post_processor must be TemplateProcessing")
        }
        postProcessor = try TemplatePostProcessor(from: postJSON, addedTokenIDs: addedTokenIDs)

        // Decoder (Metaspace)
        guard let decJSON = root["decoder"] as? [String: Any] else {
            throw TokenizerError.missingField("decoder")
        }
        guard (decJSON["type"] as? String) == "Metaspace" else {
            throw TokenizerError.unsupportedComponent("decoder must be Metaspace")
        }
        metaspaceDecoder = MetaspaceDecoder(from: decJSON)

        // Build ID → token map for decoding from the union of vocab + added
        // tokens, so future tokenizers whose added-token IDs fall outside the
        // Unigram vocab range still round-trip through `decode`. For
        // xtr-base-en every added-token id already exists in the vocab with
        // the same content; the union is identity here, but added entries
        // take precedence on collision because they are the public canonical
        // form.
        var reverse: [Int32: String] = [:]
        reverse.reserveCapacity(unigram.vocabCount + addedTokens.count)
        for i in 0..<unigram.vocabCount {
            let id = Int32(i)
            if let tok = unigram.token(for: id) {
                reverse[id] = tok
            }
        }
        for entry in addedTokens {
            reverse[entry.id] = entry.content
        }
        idToToken = reverse
    }

    /// Encode `text` to token IDs via the full pipeline. When `addSpecialTokens` is
    /// `true` (default), the post-processor template is applied, appending `</s>` for
    /// xtr-base-en. Set to `false` for raw token streams.
    ///
    /// Pipeline (per HuggingFace `added_vocabulary.rs`):
    ///   1. Pre-normalization split on `normalized: false` added tokens.
    ///      Matching spans are emitted as their declared id directly,
    ///      bypassing normalization and Viterbi.
    ///   2. Each surviving raw text span is normalized.
    ///   3. Post-normalization split on `normalized: true` added tokens (no-op
    ///      for xtr-base-en, where every entry has `normalized: false`).
    ///   4. Each surviving normalized span is metaspace pre-tokenized and
    ///      Viterbi-encoded.
    ///   5. The post-processor template wraps the full id stream.
    ///
    /// When neither matcher fires (no added-token substrings in the input)
    /// the path is byte-identical to `normalize → metaspace → Viterbi → post`.
    public func encode(_ text: String, addSpecialTokens: Bool = true) throws -> [Int32] {
        var modelIDs: [Int32] = []
        for preNormSpan in preNormMatcher.split(text) {
            switch preNormSpan {
            case .added(let id):
                modelIDs.append(id)
            case .text(let raw):
                let normalized = normalizer.normalize(raw)
                for postNormSpan in postNormMatcher.split(normalized) {
                    switch postNormSpan {
                    case .added(let id):
                        modelIDs.append(id)
                    case .text(let segment):
                        let preTokens = preTokenizer.tokenize(segment)
                        modelIDs.append(contentsOf: model.encode(preTokens))
                    }
                }
            }
        }
        return postProcessor.process(modelIDs, addSpecialTokens: addSpecialTokens)
    }

    /// Decode token IDs back to text via the Metaspace decoder. IDs whose
    /// `added_tokens` entry has `special: true` are dropped (matches
    /// HuggingFace's `decode(skip_special_tokens=True)` default). Filtering by
    /// the declared `special` flag — rather than by token-string shape — is
    /// what makes this correct for tokenizers whose specials are not
    /// `<…>`-shaped (e.g. BERT-style `[CLS]`, `[SEP]`).
    public func decode(_ ids: [Int32]) throws -> String {
        var tokens: [String] = []
        tokens.reserveCapacity(ids.count)
        for id in ids {
            if specialTokenIDs.contains(id) { continue }
            if let tok = idToToken[id] {
                tokens.append(tok)
            }
        }
        return metaspaceDecoder.decode(tokens)
    }

    private enum ParseError: Error {
        case rootNotObject
    }
}

