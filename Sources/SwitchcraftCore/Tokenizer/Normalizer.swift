// SPDX-License-Identifier: Apache-2.0
import Foundation

protocol Normalizer: Sendable {
    func normalize(_ s: String) -> String
}

// MARK: - Sequence

struct SequenceNormalizer: Normalizer {
    let normalizers: [any Normalizer]

    func normalize(_ s: String) -> String {
        normalizers.reduce(s) { $1.normalize($0) }
    }
}

// MARK: - Precompiled (DoubleArray charsmap)

struct PrecompiledNormalizer: Normalizer {
    private let trie: DoubleArrayTrie

    init(trie: DoubleArrayTrie) {
        self.trie = trie
    }

    func normalize(_ s: String) -> String {
        var result = ""
        result.reserveCapacity(s.utf8.count)
        // Iterate actual grapheme clusters (Swift Character = extended grapheme cluster)
        for grapheme in s {
            let gStr = String(grapheme)
            let gBytes = Array(gStr.utf8)
            if gBytes.count < 6 {
                if let replacement = lookup(bytes: gBytes[...], exactLength: gBytes.count) {
                    result += replacement
                    continue
                }
            }
            // Fall back to individual Unicode scalars
            for scalar in gStr.unicodeScalars {
                let sBytes = Array(scalar.utf8)
                if let replacement = lookup(bytes: sBytes[...], exactLength: sBytes.count) {
                    result += replacement
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        return result
    }

    private func lookup(bytes: ArraySlice<UInt8>, exactLength: Int) -> String? {
        let matches = trie.commonPrefixSearch(bytes)
        guard let match = matches.first(where: { $0.length == exactLength }) else { return nil }
        return trie.replacementString(at: match.value) ?? ""
    }
}

// MARK: - Strip

struct StripNormalizer: Normalizer {
    let stripLeft: Bool
    let stripRight: Bool

    func normalize(_ s: String) -> String {
        var result = s
        if stripLeft { result = String(result.drop(while: { $0.isWhitespace })) }
        if stripRight { result = String(result.reversed().drop(while: { $0.isWhitespace }).reversed()) }
        return result
    }
}

// MARK: - Replace (regex → literal)

struct ReplaceNormalizer: Normalizer {
    private let regex: NSRegularExpression
    private let replacement: String

    init(pattern: String, replacement: String) throws {
        self.regex = try NSRegularExpression(pattern: pattern)
        self.replacement = replacement
    }

    func normalize(_ s: String) -> String {
        let range = NSRange(s.startIndex..., in: s)
        return regex.stringByReplacingMatches(in: s, range: range, withTemplate: NSRegularExpression.escapedTemplate(for: replacement))
    }
}

// MARK: - Factory

extension Normalizer where Self == SequenceNormalizer {
    static func fromJSON(_ json: [String: Any]) throws -> any Normalizer {
        guard let type = json["type"] as? String else {
            throw TokenizerError.missingField("normalizer.type")
        }
        switch type {
        case "Sequence":
            guard let normalizers = json["normalizers"] as? [[String: Any]] else {
                throw TokenizerError.missingField("normalizer.normalizers")
            }
            let built = try normalizers.map { try buildNormalizer($0) }
            return SequenceNormalizer(normalizers: built)
        default:
            return try buildNormalizer(json)
        }
    }
}

func buildNormalizer(_ json: [String: Any]) throws -> any Normalizer {
    guard let type = json["type"] as? String else {
        throw TokenizerError.missingField("normalizer.type")
    }
    switch type {
    case "Sequence":
        guard let normalizers = json["normalizers"] as? [[String: Any]] else {
            throw TokenizerError.missingField("normalizer.normalizers")
        }
        let built = try normalizers.map { try buildNormalizer($0) }
        return SequenceNormalizer(normalizers: built)

    case "Precompiled":
        guard let b64 = json["precompiled_charsmap"] as? String else {
            throw TokenizerError.missingField("precompiled_charsmap")
        }
        guard let binaryData = Data(base64Encoded: b64) else {
            throw TokenizerError.malformedJSON(underlying: NormalizerBuildError.invalidBase64)
        }
        let trie = try DoubleArrayTrie(data: binaryData)
        return PrecompiledNormalizer(trie: trie)

    case "Strip":
        let stripLeft = json["strip_left"] as? Bool ?? false
        let stripRight = json["strip_right"] as? Bool ?? false
        return StripNormalizer(stripLeft: stripLeft, stripRight: stripRight)

    case "Replace":
        guard let patternObj = json["pattern"] as? [String: Any] else {
            throw TokenizerError.missingField("normalizer.pattern")
        }
        let pattern: String
        if let regex = patternObj["Regex"] as? String {
            pattern = regex
        } else if let str = patternObj["String"] as? String {
            pattern = NSRegularExpression.escapedPattern(for: str)
        } else {
            throw TokenizerError.unsupportedNormalizer("Replace pattern must be Regex or String")
        }
        let content = json["content"] as? String ?? ""
        return try ReplaceNormalizer(pattern: pattern, replacement: content)

    case "NFC":
        return PassthroughNormalizer() // Swift strings are NFC by default

    case "NFKC":
        return NFKCNormalizer()

    default:
        throw TokenizerError.unsupportedNormalizer(type)
    }
}

private enum NormalizerBuildError: Error {
    case invalidBase64
}

struct PassthroughNormalizer: Normalizer {
    func normalize(_ s: String) -> String { s }
}

struct NFKCNormalizer: Normalizer {
    func normalize(_ s: String) -> String {
        s.precomposedStringWithCompatibilityMapping
    }
}
