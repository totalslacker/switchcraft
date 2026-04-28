import Foundation

/// DoubleArray trie decoder for the `precompiled_charsmap` binary embedded in
/// HuggingFace tokenizer.json Precompiled normalizer entries.
///
/// Binary format (spm_precompiled / huggingface):
///   [u32 LE: trie_size_bytes] [u32 × (trie_size_bytes/4): nodes] [UTF-8: string blob]
struct DoubleArrayTrie: Sendable {
    private let nodes: [UInt32]
    private let stringBlob: [UInt8]

    init(data: Data) throws {
        guard data.count >= 4 else {
            throw TokenizerError.malformedJSON(underlying: DoubleArrayError.truncatedHeader)
        }
        let trieBytes = Int(data.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian })
        guard trieBytes % 4 == 0, data.count >= 4 + trieBytes else {
            throw TokenizerError.malformedJSON(underlying: DoubleArrayError.truncatedTrie)
        }
        let nodeCount = trieBytes / 4
        var nodeArray = [UInt32](repeating: 0, count: nodeCount)
        data.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.advanced(by: 4)
            _ = memcpy(&nodeArray, base, trieBytes)
        }
        nodes = nodeArray
        let blobStart = 4 + trieBytes
        stringBlob = Array(data[blobStart...])
    }

    /// Returns (matchLength, stringBlobOffset) for all prefixes of `bytes` found in the trie.
    /// `matchLength` is the number of bytes consumed; `stringBlobOffset` indexes into `stringBlob`.
    func commonPrefixSearch(_ bytes: ArraySlice<UInt8>) -> [(length: Int, value: Int)] {
        var results: [(length: Int, value: Int)] = []
        var nodePos = 0
        for (i, c) in bytes.enumerated() {
            if c == 0 { break }
            nodePos ^= Int(c)
            guard nodePos < nodes.count else { return results }
            let unit = nodes[nodePos]
            guard Self.label(unit) == UInt32(c) else { return results }
            nodePos ^= Int(Self.offset(unit))
            if Self.hasLeaf(unit) {
                guard nodePos < nodes.count else { return results }
                let val = Int(Self.value(nodes[nodePos]))
                results.append((length: i + 1, value: val))
            }
        }
        return results
    }

    /// Returns the null-terminated replacement string stored at `offset` in the string blob,
    /// or `nil` if the offset is out of range.
    func replacementString(at offset: Int) -> String? {
        guard offset < stringBlob.count else { return nil }
        var end = offset
        while end < stringBlob.count && stringBlob[end] != 0 { end += 1 }
        return String(bytes: stringBlob[offset..<end], encoding: .utf8)
    }

    // MARK: - Bit-field accessors (match spm_precompiled Rust exactly)

    @inline(__always)
    private static func hasLeaf(_ node: UInt32) -> Bool {
        (node >> 8) & 1 == 1
    }

    @inline(__always)
    private static func value(_ node: UInt32) -> UInt32 {
        node & 0x7FFFFFFF
    }

    @inline(__always)
    private static func label(_ node: UInt32) -> UInt32 {
        node & 0x800000FF
    }

    @inline(__always)
    private static func offset(_ node: UInt32) -> UInt32 {
        (node >> 10) << ((node & (1 << 9)) >> 6)
    }
}

private enum DoubleArrayError: Error {
    case truncatedHeader
    case truncatedTrie
}
