import Foundation

// MARK: - ByteTrie

/// Flat-array byte-keyed trie for vocabulary token lookup.
/// Built once at init; read-only after construction (Sendable-safe).
final class ByteTrieNode: @unchecked Sendable {
    var children: [UInt8: ByteTrieNode] = [:]
    var ids: [Int] = []

    init() {}
}

struct ByteTrie: Sendable {
    private let root: ByteTrieNode

    init() { root = ByteTrieNode() }

    func insert(_ utf8: [UInt8], id: Int) {
        var node = root
        for (i, byte) in utf8.enumerated() {
            if node.children[byte] == nil {
                node.children[byte] = ByteTrieNode()
            }
            node = node.children[byte]!
            if i == utf8.count - 1 {
                node.ids.append(id)
            }
        }
    }

    /// Returns all (length, id) pairs where length bytes of `bytes` match a vocabulary token.
    func commonPrefixSearch(_ bytes: ArraySlice<UInt8>) -> [(length: Int, id: Int)] {
        var results: [(length: Int, id: Int)] = []
        var node = root
        for (offset, byte) in zip(0..., bytes) {
            guard let child = node.children[byte] else { break }
            let length = offset + 1
            for id in child.ids {
                results.append((length: length, id: id))
            }
            node = child
        }
        return results
    }
}

// MARK: - Viterbi DP node

private struct BestPathNode {
    var tokenID: Int32          // ADR 003(c): Int32 at the API boundary
    var bestPathScore: Double   // ADR 003(a): Double for log-probability accumulators
    var startsAt: Int?          // nil = unreachable
}

// MARK: - UnigramModel

struct UnigramModel: Sendable {
    private struct VocabEntry: Sendable {
        let token: String
        let score: Double
    }

    private let vocab: [VocabEntry]
    private let unkID: Int32
    private let unkScore: Double
    private let trie: ByteTrie

    // Hardcoded per the HuggingFace tokenizers Rust implementation — not parsed from JSON
    private let fuseUnk: Bool = true

    init(modelJSON: [String: Any]) throws {
        guard let vocabArray = modelJSON["vocab"] as? [[Any]] else {
            throw TokenizerError.missingField("model.vocab")
        }
        var entries: [VocabEntry] = []
        entries.reserveCapacity(vocabArray.count)
        for pair in vocabArray {
            guard pair.count >= 2,
                  let token = pair[0] as? String,
                  let score = pair[1] as? Double else {
                throw TokenizerError.malformedJSON(underlying: UnigramError.malformedVocabEntry)
            }
            entries.append(VocabEntry(token: token, score: score))
        }
        vocab = entries
        unkID = Int32(modelJSON["unk_id"] as? Int ?? 2)
        let minScore = entries.map(\.score).min() ?? -10.0
        unkScore = minScore - 10.0

        // Build byte-trie from vocabulary (ADR 003(b): index by bytes, not Characters)
        let t = ByteTrie()
        for (id, entry) in entries.enumerated() {
            let bytes = Array(entry.token.utf8)
            if !bytes.isEmpty {
                t.insert(bytes, id: id)
            }
        }
        trie = t
    }

    /// Encode pre-tokenized strings into token IDs via trie-based Viterbi DP.
    /// Each string in `preTokens` is segmented independently.
    func encode(_ preTokens: [String]) -> [Int32] {
        var result: [Int32] = []
        for token in preTokens {
            result.append(contentsOf: viterbi(token))
        }
        return result
    }

    /// Vocab ID for an exact-match token string, or nil.
    func id(for token: String) -> Int32? {
        for (i, e) in vocab.enumerated() where e.token == token {
            return Int32(i)
        }
        return nil
    }

    /// Token string for an ID.
    func token(for id: Int32) -> String? {
        let i = Int(id)
        guard i >= 0, i < vocab.count else { return nil }
        return vocab[i].token
    }

    // MARK: - Trie-based Viterbi (encode_optimized)

    private func viterbi(_ input: String) -> [Int32] {
        // ADR 003(b): materialise UTF-8 bytes upfront; all indexing via Int
        let bytes = Array(input.utf8)
        guard !bytes.isEmpty else { return [] }

        var bestPath = [BestPathNode](
            repeating: BestPathNode(tokenID: 0, bestPathScore: 0.0, startsAt: nil),
            count: bytes.count + 1
        )
        bestPath[0].startsAt = 0  // position 0 is always reachable

        for startsAt in 0..<bytes.count {
            guard bestPath[startsAt].startsAt != nil || startsAt == 0 else { continue }
            let available = bestPath[startsAt].bestPathScore
            var hasSingleNode = false
            let mblen = utf8ScalarByteLength(bytes[startsAt])

            for (length, vocabID) in trie.commonPrefixSearch(bytes[startsAt...]) {
                let keyPos = startsAt + length
                let score = vocab[vocabID].score + available
                if bestPath[keyPos].startsAt == nil || score > bestPath[keyPos].bestPathScore {
                    bestPath[keyPos] = BestPathNode(
                        tokenID: Int32(vocabID),
                        bestPathScore: score,
                        startsAt: startsAt
                    )
                }
                if length == mblen { hasSingleNode = true }
            }

            if !hasSingleNode {
                let keyPos = startsAt + mblen
                let score = unkScore + available
                if bestPath[keyPos].startsAt == nil || score > bestPath[keyPos].bestPathScore {
                    bestPath[keyPos] = BestPathNode(
                        tokenID: unkID,
                        bestPathScore: score,
                        startsAt: startsAt
                    )
                }
            }
        }

        // Backtrack with fuse_unk
        var ids: [Int32] = []
        var endsAt = bytes.count
        while endsAt > 0 {
            let node = bestPath[endsAt]
            guard let start = node.startsAt else { break }
            if fuseUnk && node.tokenID == unkID,
               let last = ids.last, last == unkID {
                // Consecutive unk tokens: already have one unk in ids; skip adding another
            } else {
                ids.append(node.tokenID)
            }
            endsAt = start
        }
        ids.reverse()
        return ids
    }

    @inline(__always)
    private func utf8ScalarByteLength(_ leadByte: UInt8) -> Int {
        switch leadByte {
        case 0x00...0x7F: return 1
        case 0xC0...0xDF: return 2
        case 0xE0...0xEF: return 3
        default: return 4
        }
    }
}

private enum UnigramError: Error {
    case malformedVocabEntry
}
