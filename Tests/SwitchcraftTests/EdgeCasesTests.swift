// Edge cases suite for SwitchcraftStore's public API.
//
// Covers corner cases not exercised by SwitchcraftStoreTests' happy
// paths: empty query, empty index, single-character query, very long
// documents, Unicode bodies (CJK, emoji, RTL), duplicate-add with the
// same id and same body, and removing the only document in the index.
//
// All tests use MockEmbedder + InMemoryStorage so no model assets or
// SQLite assets are required. The store is constructed with the
// `init(storage:embedder:config:)` initialiser (rather than the
// `.sqlite()` factory) so the duplicate-add test can read
// `documentCount()` / `chunkCount()` directly from the test-owned
// storage — `SwitchcraftStore.storage` is private.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

@Suite("Edge Cases")
struct EdgeCasesTests {

    // MARK: - Helpers

    /// Construct a fresh `(SwitchcraftStore, InMemoryStorage)` pair with
    /// `MockEmbedder(dims: 32)` and the given config. The test retains
    /// the storage reference so it can assert `documentCount()` and
    /// `chunkCount()` directly.
    private static func makeStoreWithStorage(
        config: StoreConfig = .default
    ) async throws -> (SwitchcraftStore, InMemoryStorage) {
        let storage = InMemoryStorage()
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: 32),
            config: config
        )
        return (store, storage)
    }

    // MARK: - Requirement 1: empty query

    @Test("empty query returns [] without throwing")
    func emptyQueryReturnsEmpty() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()
        try await store.add(id: "doc-a", body: "alpha keyword")

        let hits = try await store.search(query: "", topK: 10)
        #expect(hits.isEmpty)

        try await store.shutdown()
    }

    // MARK: - Requirement 2: empty index

    @Test("search on an empty index returns []")
    func emptyIndexReturnsEmpty() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()

        let hits = try await store.search(query: "anything", topK: 10)
        #expect(hits.isEmpty)

        try await store.shutdown()
    }

    // MARK: - Requirement 3: single-character query

    @Test("single-character query does not throw")
    func singleCharacterQueryDoesNotThrow() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()
        try await store.add(id: "doc-a", body: "a quick brown fox")

        // Result content is not asserted — FTS5 may treat single
        // characters as below-threshold, and the vector path may or
        // may not surface a match. The contract is no throw.
        _ = try await store.search(query: "a", topK: 10)

        try await store.shutdown()
    }

    // MARK: - Requirement 4: very long document

    @Test("long document (10k+ tokens) flushes and is searchable")
    func longDocumentRoundTrip() async throws {
        // Build a body with exactly 10,000 unique whitespace-separated
        // tokens so MockEmbedder produces a 10,000-vector chunk. Each
        // token is unique so we can search for one and assert presence.
        let tokenCount = 10_000
        var builder = ""
        builder.reserveCapacity(tokenCount * 8)
        for i in 0..<tokenCount {
            if i > 0 { builder.append(" ") }
            builder.append("tok\(i)")
        }
        let body = builder

        // Configure l0 large enough that the 10k-vector flush completes
        // without triggering a real k-means cascade. The spec only
        // requires `index()` to flush without error — not to cascade —
        // and the production default (l0=1024, k≈1600) would be slow
        // under debug.
        let config = StoreConfig(
            indexer: IndexerConfig(l0Capacity: 16_384, lsmFanout: 2)
        )
        let (store, _) = try await Self.makeStoreWithStorage(config: config)

        try await store.add(id: "long-doc", body: body)
        try await store.index()

        // Pick a token from late in the body to also exercise the
        // tail of the embedding ledger.
        let hits = try await store.search(query: "tok9999", topK: 10)
        #expect(hits.contains(where: { $0.uuid == "long-doc" }))

        try await store.shutdown()
    }

    // MARK: - Requirement 5: CJK body

    @Test("CJK body can be added and self-search returns the document")
    func cjkBodyRoundTrip() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()

        // A run of Han characters with no whitespace. SQLite FTS5's
        // default `unicode61` tokenizer groups consecutive Han into one
        // token, and MockEmbedder whitespace-splits to a single token
        // — so both retrieval paths align on the same single-token
        // representation.
        let body = "你好世界欢迎光临"
        try await store.add(id: "doc-cjk", body: body)

        let hits = try await store.search(query: body, topK: 10)
        #expect(hits.contains(where: { $0.uuid == "doc-cjk" }))

        try await store.shutdown()
    }

    // MARK: - Requirement 6: emoji body

    @Test("emoji body add + search does not throw")
    func emojiBodyAddSearchNoThrow() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()

        let body = "🎉🚀✨🌟🔥"
        try await store.add(id: "doc-emoji", body: body)

        // Self-search and a generic ASCII query: the contract is
        // no-throw. Emoji tokenisation under FTS5 is implementation-
        // dependent, so result presence is not asserted.
        _ = try await store.search(query: body, topK: 10)
        _ = try await store.search(query: "celebrate", topK: 10)

        try await store.shutdown()
    }

    // MARK: - Requirement 7: RTL body

    @Test("RTL (Arabic + Hebrew) body add + search does not throw")
    func rtlBodyAddSearchNoThrow() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()

        let arabic = "مرحبا بالعالم"
        let hebrew = "שלום עולם"
        try await store.add(id: "doc-arabic", body: arabic)
        try await store.add(id: "doc-hebrew", body: hebrew)

        _ = try await store.search(query: arabic, topK: 10)
        _ = try await store.search(query: hebrew, topK: 10)

        try await store.shutdown()
    }

    // MARK: - Requirement 8: duplicate add, same id + same body

    @Test("duplicate add with same id and same body keeps documentCount=1, chunkCount=1")
    func duplicateAddSameIdSameBody() async throws {
        let (store, storage) = try await Self.makeStoreWithStorage()

        try await store.add(id: "dup", body: "hello world")
        try await store.add(id: "dup", body: "hello world")

        let docCount = try await storage.documentCount()
        let chunkCount = try await storage.chunkCount()
        #expect(docCount == 1)
        #expect(chunkCount == 1)

        try await store.shutdown()
    }

    // MARK: - Requirement 9: remove the only document

    @Test("remove of the only document leaves an empty index")
    func removeOnlyDocumentLeavesEmptyIndex() async throws {
        let (store, _) = try await Self.makeStoreWithStorage()

        try await store.add(id: "only", body: "lonely keyword in an otherwise empty index")
        try await store.remove(id: "only")

        let hits = try await store.search(query: "lonely keyword", topK: 10)
        #expect(hits.isEmpty)

        try await store.shutdown()
    }
}
