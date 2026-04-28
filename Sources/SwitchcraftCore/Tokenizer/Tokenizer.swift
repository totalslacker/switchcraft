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

        // Added tokens → name → ID lookup (special tokens like </s>, <pad>, <extra_id_N>).
        // The Unigram vocab already contains these at their canonical IDs, but
        // post-processor templates reference them by string name.
        var addedTokenIDs: [String: Int32] = [:]
        if let added = root["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if let name = entry["content"] as? String,
                   let id = entry["id"] as? Int {
                    addedTokenIDs[name] = Int32(id)
                }
            }
        }

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

        // Build ID → token map for decoding.
        var reverse: [Int32: String] = [:]
        reverse.reserveCapacity(unigram.vocabCount)
        for i in 0..<unigram.vocabCount {
            let id = Int32(i)
            if let tok = unigram.token(for: id) {
                reverse[id] = tok
            }
        }
        idToToken = reverse
    }

    /// Encode `text` to token IDs via the full pipeline. When `addSpecialTokens` is
    /// `true` (default), the post-processor template is applied, appending `</s>` for
    /// xtr-base-en. Set to `false` for raw token streams.
    public func encode(_ text: String, addSpecialTokens: Bool = true) throws -> [Int32] {
        let normalized = normalizer.normalize(text)
        let preTokens = preTokenizer.tokenize(normalized)
        let modelIDs = model.encode(preTokens)
        return postProcessor.process(modelIDs, addSpecialTokens: addSpecialTokens)
    }

    /// Decode token IDs back to text via the Metaspace decoder.
    public func decode(_ ids: [Int32]) throws -> String {
        var tokens: [String] = []
        tokens.reserveCapacity(ids.count)
        for id in ids {
            if let tok = idToToken[id] {
                if tok.hasPrefix("<") && tok.hasSuffix(">") {
                    continue
                }
                tokens.append(tok)
            }
        }
        return metaspaceDecoder.decode(tokens)
    }

    private enum ParseError: Error {
        case rootNotObject
    }
}

