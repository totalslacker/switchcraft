import Foundation
import Testing
@testable import SwitchcraftCore

// MARK: - Fixture loading

private struct TokenizerFixture: Decodable {
    let input: String
    let ids: [Int32]
    let noSpecial: [Int32]
}

private enum FixtureLoader {
    static let tokenizer: Tokenizer = {
        guard let url = Bundle.module.url(forResource: "xtr-base-en.tokenizer", withExtension: "json", subdirectory: "Fixtures") else {
            fatalError("xtr-base-en.tokenizer.json missing from test bundle")
        }
        do {
            let data = try Data(contentsOf: url)
            return try Tokenizer(data: data)
        } catch {
            fatalError("Failed to load tokenizer: \(error)")
        }
    }()

    static let fixtures: [TokenizerFixture] = {
        guard let url = Bundle.module.url(forResource: "xtr-base-en.tokenizer-fixtures", withExtension: "json", subdirectory: "Fixtures") else {
            fatalError("xtr-base-en.tokenizer-fixtures.json missing from test bundle")
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode([TokenizerFixture].self, from: data)
        } catch {
            fatalError("Failed to load fixtures: \(error)")
        }
    }()
}

// MARK: - Integration: bit-exact match against upstream Rust reference

@Suite("Tokenizer / fixture parity")
struct TokenizerFixtureTests {
    @Test("encode matches reference IDs (with special tokens)", arguments: 0..<FixtureLoader.fixtures.count)
    func encodeWithSpecial(index: Int) throws {
        let fx = FixtureLoader.fixtures[index]
        let got = try FixtureLoader.tokenizer.encode(fx.input, addSpecialTokens: true)
        #expect(got == fx.ids, "input=\(fx.input.debugDescription)")
    }

    @Test("encode matches reference IDs (no special tokens)", arguments: 0..<FixtureLoader.fixtures.count)
    func encodeWithoutSpecial(index: Int) throws {
        let fx = FixtureLoader.fixtures[index]
        let got = try FixtureLoader.tokenizer.encode(fx.input, addSpecialTokens: false)
        #expect(got == fx.noSpecial, "input=\(fx.input.debugDescription)")
    }
}

// MARK: - Edge cases

@Suite("Tokenizer / edge cases")
struct TokenizerEdgeCaseTests {
    @Test("empty input returns just </s> when special tokens enabled")
    func emptyInput() throws {
        let ids = try FixtureLoader.tokenizer.encode("", addSpecialTokens: true)
        #expect(ids == [1])
    }

    @Test("empty input returns empty array when special tokens disabled")
    func emptyInputNoSpecial() throws {
        let ids = try FixtureLoader.tokenizer.encode("", addSpecialTokens: false)
        #expect(ids.isEmpty)
    }

    @Test("whitespace-only input collapses to just </s>")
    func whitespaceOnly() throws {
        let ids = try FixtureLoader.tokenizer.encode("   \t\n", addSpecialTokens: true)
        #expect(ids == [1])
    }

    @Test("control characters are stripped by nmt_nfkc")
    func controlCharsStripped() throws {
        let ids = try FixtureLoader.tokenizer.encode("\u{01}\u{02}\u{03}", addSpecialTokens: true)
        #expect(ids == [1])
    }

    @Test("zero-width space normalizes to a regular space")
    func zeroWidthSpaceNormalized() throws {
        let plain = try FixtureLoader.tokenizer.encode("hello world", addSpecialTokens: true)
        let zws   = try FixtureLoader.tokenizer.encode("hello\u{200B}world", addSpecialTokens: true)
        #expect(plain == zws)
    }

    @Test("NFKC compatibility decomposition: ℌ → H")
    func nfkcDecomposition() throws {
        let canonical = try FixtureLoader.tokenizer.encode("Hello", addSpecialTokens: true)
        let decomp    = try FixtureLoader.tokenizer.encode("ℌello", addSpecialTokens: true)
        #expect(canonical == decomp)
    }

    @Test("CJK characters fuse into a single unk under fuse_unk")
    func cjkFusedUnk() throws {
        let ids = try FixtureLoader.tokenizer.encode("日本語テキスト", addSpecialTokens: true)
        // [▁, <unk>, </s>] — fuse_unk merges the run of unknown CJK bytes
        #expect(ids == [3, 2, 1])
    }

    @Test("emoji becomes unk in middle of mostly-known tokens")
    func emojiUnk() throws {
        let ids = try FixtureLoader.tokenizer.encode("cat 🐱 dog", addSpecialTokens: true)
        #expect(ids == [1712, 3, 2, 1782, 1])
    }

    @Test("literal ▁ in input is treated as a word boundary")
    func literalMetaspace() throws {
        let ids = try FixtureLoader.tokenizer.encode("▁literal metaspace", addSpecialTokens: true)
        #expect(ids == [26998, 10531, 6633, 1])
    }

    @Test("very long input (>2048 chars) tokenizes within budget")
    func longInput() throws {
        let input = String(repeating: "a", count: 2049)
        let start = Date()
        let ids = try FixtureLoader.tokenizer.encode(input, addSpecialTokens: true)
        let elapsed = Date().timeIntervalSince(start)
        #expect(elapsed < 5.0, "long input took \(elapsed)s; exceeds 5s budget")
        #expect(ids.first == 3)         // ▁
        #expect(ids.last == 1)          // </s>
        #expect(ids.count == 2 + 2049)  // ▁ + 2049 'a' tokens + </s>
    }

    @Test("encode is deterministic across repeated calls")
    func deterministic() throws {
        let inputs = ["hello world", "Hello, world!", "naïve café résumé", "日本"]
        for s in inputs {
            let a = try FixtureLoader.tokenizer.encode(s, addSpecialTokens: true)
            let b = try FixtureLoader.tokenizer.encode(s, addSpecialTokens: true)
            #expect(a == b)
        }
    }
}

// MARK: - Decode round-trip

@Suite("Tokenizer / decode")
struct TokenizerDecodeTests {
    @Test("decode reverses Metaspace encoding for simple text")
    func decodeSimple() throws {
        let ids = try FixtureLoader.tokenizer.encode("hello world", addSpecialTokens: true)
        let text = try FixtureLoader.tokenizer.decode(ids)
        #expect(text == "hello world")
    }

    @Test("decode skips special tokens like </s>")
    func decodeSkipsSpecial() throws {
        let ids = try FixtureLoader.tokenizer.encode("Hello, world!", addSpecialTokens: true)
        let text = try FixtureLoader.tokenizer.decode(ids)
        #expect(text == "Hello, world!")
    }

    @Test("decode of empty IDs returns empty string")
    func decodeEmpty() throws {
        let text = try FixtureLoader.tokenizer.decode([])
        #expect(text == "")
    }
}

// MARK: - DoubleArrayTrie unit tests

@Suite("DoubleArrayTrie")
struct DoubleArrayTrieTests {
    @Test("ℌ (U+210C) normalizes to H via Precompiled charsmap")
    func normalizeMathH() throws {
        // Sanity-check the underlying Precompiled normalizer through the full Tokenizer:
        // bit-exact normalization is the easiest path. The DoubleArray decoder is
        // exercised transitively, plus directly via the fixture suite.
        let ids = try FixtureLoader.tokenizer.encode("ℌello", addSpecialTokens: false)
        let h   = try FixtureLoader.tokenizer.encode("Hello", addSpecialTokens: false)
        #expect(ids == h)
    }

    @Test("zero-width space deletes via Precompiled charsmap")
    func zwsDeletes() throws {
        let withZWS = try FixtureLoader.tokenizer.encode("hello\u{200B}world", addSpecialTokens: false)
        let plain   = try FixtureLoader.tokenizer.encode("hello world", addSpecialTokens: false)
        #expect(withZWS == plain)
    }
}

// MARK: - Added-token parity guard
//
// Every entry in `added_tokens` must round-trip to its declared id when encoded
// in isolation with `addSpecialTokens: false`. For xtr-base-en this passes
// without code changes today (the added-token IDs coincide with Unigram vocab
// IDs), but the assertion exists so any future tokenizer that lacks the
// coincidence surfaces immediately rather than silently diverging from the
// HuggingFace reference. When porting a new tokenizer, extend the fixture
// loader so this test runs against every loaded tokenizer's `added_tokens`.

@Suite("Tokenizer / added-token parity")
struct AddedTokenParityTests {
    @Test("every added_tokens entry encodes to exactly [id]",
          arguments: 0..<FixtureLoader.tokenizer.addedTokens.count)
    func eachAddedTokenRoundTripsToItsID(index: Int) throws {
        let entry = FixtureLoader.tokenizer.addedTokens[index]
        let ids = try FixtureLoader.tokenizer.encode(entry.content, addSpecialTokens: false)
        #expect(ids == [entry.id],
                "added token \(entry.content.debugDescription) (id=\(entry.id)) encoded to \(ids)")
    }
}

// MARK: - Component-level: AddedTokenMatcher

@Suite("AddedTokenMatcher")
struct AddedTokenMatcherTests {
    private static func entry(
        _ content: String,
        id: Int32,
        normalized: Bool = false,
        singleWord: Bool = false,
        lstrip: Bool = false,
        rstrip: Bool = false
    ) -> AddedToken {
        AddedToken(
            id: id, content: content, normalized: normalized,
            singleWord: singleWord, lstrip: lstrip, rstrip: rstrip, special: true
        )
    }

    @Test("empty input yields empty output")
    func emptyInput() {
        let m = AddedTokenMatcher(entries: [Self.entry("<a>", id: 100)])
        #expect(m.split("") == [])
    }

    @Test("no matches returns single text span")
    func noMatchSingleSpan() {
        let m = AddedTokenMatcher(entries: [Self.entry("<a>", id: 100)])
        #expect(m.split("hello world") == [.text("hello world")])
    }

    @Test("empty entries set returns single text span")
    func noEntries() {
        let m = AddedTokenMatcher(entries: [])
        #expect(m.split("anything goes") == [.text("anything goes")])
    }

    @Test("substring match anywhere by default")
    func basicMatch() {
        let m = AddedTokenMatcher(entries: [Self.entry("<a>", id: 100)])
        #expect(m.split("x<a>y") == [.text("x"), .added(100), .text("y")])
    }

    @Test("adjacent matches with no gap produce no empty text spans")
    func adjacentMatches() {
        let m = AddedTokenMatcher(entries: [
            Self.entry("<a>", id: 100),
            Self.entry("<b>", id: 200),
        ])
        #expect(m.split("<a><b>") == [.added(100), .added(200)])
    }

    @Test("overlapping candidates: longer match wins")
    func longerWins() {
        let m = AddedTokenMatcher(entries: [
            Self.entry("<ab>", id: 100),
            Self.entry("<abc>", id: 200),
        ])
        #expect(m.split("<abc>") == [.added(200)])
        #expect(m.split("<abc><ab>") == [.added(200), .added(100)])
    }

    @Test("partial-prefix near-miss falls through to text")
    func partialPrefix() {
        let m = AddedTokenMatcher(entries: [Self.entry("<extra>", id: 100)])
        #expect(m.split("<extralong>") == [.text("<extralong>")])
        #expect(m.split("<extr") == [.text("<extr")])
    }

    @Test("single_word: false matches mid-word")
    func singleWordFalseMatchesMidWord() {
        let m = AddedTokenMatcher(entries: [Self.entry("foo", id: 100, singleWord: false)])
        #expect(m.split("xfooy") == [.text("x"), .added(100), .text("y")])
    }

    @Test("single_word: true rejects mid-word match")
    func singleWordTrueRejectsMidWord() {
        let m = AddedTokenMatcher(entries: [Self.entry("foo", id: 100, singleWord: true)])
        #expect(m.split("xfooy") == [.text("xfooy")])
    }

    @Test("single_word: true accepts at word boundary")
    func singleWordTrueAcceptsBoundary() {
        let m = AddedTokenMatcher(entries: [Self.entry("foo", id: 100, singleWord: true)])
        #expect(m.split("foo") == [.added(100)])
        #expect(m.split(" foo ") == [.text(" "), .added(100), .text(" ")])
        #expect(m.split("(foo)") == [.text("("), .added(100), .text(")")])
    }

    @Test("single_word: true with non-word content has no boundary requirement")
    func singleWordTrueNonWordContent() {
        let m = AddedTokenMatcher(entries: [Self.entry("<x>", id: 100, singleWord: true)])
        #expect(m.split("a<x>b") == [.text("a"), .added(100), .text("b")])
    }

    @Test("lstrip consumes adjacent leading whitespace")
    func lstripConsumesLeading() {
        let m = AddedTokenMatcher(entries: [Self.entry("<x>", id: 100, lstrip: true)])
        #expect(m.split("a   <x>b") == [.text("a"), .added(100), .text("b")])
    }

    @Test("rstrip consumes adjacent trailing whitespace")
    func rstripConsumesTrailing() {
        let m = AddedTokenMatcher(entries: [Self.entry("<x>", id: 100, rstrip: true)])
        #expect(m.split("a<x>   b") == [.text("a"), .added(100), .text("b")])
    }

    @Test("lstrip+rstrip consume both sides")
    func lstripRstripBothSides() {
        let m = AddedTokenMatcher(entries: [
            Self.entry("<x>", id: 100, lstrip: true, rstrip: true)
        ])
        #expect(m.split("a   <x>   b") == [.text("a"), .added(100), .text("b")])
    }

    @Test("empty-content entries are skipped at construction")
    func emptyContentSkipped() {
        let m = AddedTokenMatcher(entries: [Self.entry("", id: 100)])
        #expect(m.isEmpty)
        #expect(m.split("anything") == [.text("anything")])
    }
}

// MARK: - Component-level: Metaspace pre-tokenizer

@Suite("MetaspacePreTokenizer")
struct MetaspaceTests {
    @Test("simple split prepends ▁ to each word")
    func simpleSplit() throws {
        let pre = try MetaspacePreTokenizer(from: ["replacement": "▁", "prepend_scheme": "always"])
        let tokens = pre.tokenize("hello world")
        #expect(tokens == ["▁hello", "▁world"])
    }

    @Test("leading whitespace is preserved as ▁ prefix")
    func leadingWhitespace() throws {
        let pre = try MetaspacePreTokenizer(from: ["replacement": "▁", "prepend_scheme": "always"])
        let tokens = pre.tokenize(" leading")
        #expect(tokens == ["▁leading"])
    }

    @Test("empty input produces no tokens")
    func emptyInputProducesNothing() throws {
        let pre = try MetaspacePreTokenizer(from: ["replacement": "▁", "prepend_scheme": "always"])
        let tokens = pre.tokenize("")
        #expect(tokens.isEmpty)
    }
}
