import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("Hybrid Search")
struct HybridSearchTests {

    // MARK: - Helpers

    static func makeDocument(
        uuid: String,
        body: String,
        metadata: [String: String] = [:]
    ) -> DocumentRecord {
        let metadataData: Data
        if metadata.isEmpty {
            metadataData = Data()
        } else {
            metadataData = (try? JSONSerialization.data(withJSONObject: metadata)) ?? Data()
        }
        return DocumentRecord(
            uuid: uuid,
            date: Date(timeIntervalSince1970: 0),
            metadata: metadataData,
            hash: "hash-\(uuid)",
            body: body,
            lens: []
        )
    }

    /// Storage seeded with documents only — no buckets / generations.
    /// Vector search against this storage returns `[]`, so it isolates
    /// the FTS path.
    static func makeFtsOnlyStorage(
        _ docs: [(uuid: String, body: String)]
    ) async throws -> InMemoryStorage {
        let storage = InMemoryStorage()
        try await storage.open()
        for d in docs {
            try await storage.upsertDocument(makeDocument(uuid: d.uuid, body: d.body))
        }
        return storage
    }

    /// Build a synthetic index with a vector pipeline that ranks a known
    /// document at rank 1 for the chosen query token. Mirrors
    /// `SearchEngineTests.makeSyntheticIndex` but exposes per-doc bodies
    /// so tests can also drive the FTS source.
    static func makeIndexedStorage(
        documents: [(uuid: String, body: String)],
        tokensPerDoc: Int = 8,
        dims: Int = 16,
        seed: UInt64
    ) async throws -> (
        storage: InMemoryStorage,
        rowsByDoc: [String: [[Float]]]
    ) {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(
            storage: storage,
            config: IndexerConfig.testing(l0Capacity: 1024, lsmFanout: 4)
        )
        var rng = SplitMix64(seed: seed)

        var rowsByDoc: [String: [[Float]]] = [:]
        for spec in documents {
            var rows = [[Float]]()
            var flat = [Float]()
            for _ in 0..<tokensPerDoc {
                let row = SearchEngineTests.randomRow(dims: dims, rng: &rng)
                rows.append(row)
                flat.append(contentsOf: row)
            }
            rowsByDoc[spec.uuid] = rows

            let hash = "h-\(spec.uuid)"
            let chunk = try await storage.upsertChunk(
                ChunkRecord(
                    hash: hash,
                    model: "test",
                    embeddings: Data(),
                    counts: [tokensPerDoc]
                )
            )

            try await storage.upsertDocument(
                DocumentRecord(
                    uuid: spec.uuid,
                    date: Date(timeIntervalSince1970: 0),
                    metadata: Data(),
                    hash: hash,
                    body: spec.body,
                    lens: [tokensPerDoc]
                )
            )
            try await indexer.add(chunkID: chunk.id, embeddings: flat, dims: dims)
        }
        try await indexer.flush()
        return (storage, rowsByDoc)
    }

    // MARK: - Empty-input fallbacks

    @Test("both empty: returns []")
    func bothEmptyReturnsEmpty() async throws {
        let storage = try await Self.makeFtsOnlyStorage([
            ("a", "alpha"),
            ("b", "beta"),
        ])
        let engine = SearchEngine(storage: storage)

        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "   ",
            topK: 10
        )
        #expect(hits.isEmpty)
    }

    @Test("no embeddings: FTS-only path with RRF-formatted scores")
    func emptyEmbeddingsUsesFtsOnly() async throws {
        let storage = try await Self.makeFtsOnlyStorage([
            ("alpha", "lorem ipsum uniquealpha"),
            ("beta",  "lorem ipsum uniquebeta"),
        ])
        let engine = SearchEngine(storage: storage)

        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "uniquealpha",
            topK: 10
        )
        // Only one document contains "uniquealpha"; rank=1.
        #expect(hits.count == 1)
        #expect(hits[0].uuid == "alpha")
        #expect(hits[0].vectorRank == nil)
        #expect(hits[0].vectorScore == nil)
        #expect(hits[0].ftsRank == 1)
        #expect(hits[0].score == 1 / Float(61))
    }

    @Test("whitespace-only text: vector-only path with RRF-formatted scores")
    func whitespaceTextUsesVectorOnly() async throws {
        let dims = 16
        let (storage, rowsByDoc) = try await Self.makeIndexedStorage(
            documents: (0..<8).map { ("doc-\(String(format: "%02d", $0))", "body-\($0)") },
            dims: dims,
            seed: 11
        )
        let engine = SearchEngine(storage: storage)
        let target = "doc-03"
        let queryToken = rowsByDoc[target]![2]

        let hits = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "   \n  ",
            topK: 5
        )
        #expect(!hits.isEmpty)
        #expect(hits[0].uuid == target)
        #expect(hits[0].vectorRank == 1)
        #expect(hits[0].ftsRank == nil)
        #expect(hits[0].ftsScore == nil)
        #expect(hits[0].score == 1 / Float(61))
    }

    // MARK: - Single-source recall

    @Test("vector finds the doc when FTS misses (no body match)")
    func vectorOnlyRecallWhenFtsFindsNothing() async throws {
        let dims = 16
        let (storage, rowsByDoc) = try await Self.makeIndexedStorage(
            documents: (0..<6).map { ("doc-\($0)", "body \($0)") },
            dims: dims,
            seed: 13
        )
        let engine = SearchEngine(storage: storage)
        let target = "doc-2"
        let queryToken = rowsByDoc[target]![1]

        let hits = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "no-such-term-anywhere",
            topK: 5
        )
        // FTS returns nothing → only vector contributes; target is rank 1.
        #expect(hits[0].uuid == target)
        #expect(hits[0].ftsRank == nil)
        #expect(hits[0].vectorRank == 1)
    }

    @Test("FTS rescues a doc that vector ranks low")
    func ftsBoostsRareTokenDoc() async throws {
        let dims = 16
        // 6 documents with random bodies; only "rare" contains "needle".
        let docs: [(uuid: String, body: String)] = [
            ("doc-0", "alpha bravo charlie"),
            ("doc-1", "delta echo foxtrot"),
            ("doc-2", "golf hotel india"),
            ("doc-3", "juliet kilo lima"),
            ("doc-4", "mike november oscar"),
            ("rare",  "papa quebec needle"),
        ]
        let (storage, rowsByDoc) = try await Self.makeIndexedStorage(
            documents: docs, dims: dims, seed: 17
        )
        let engine = SearchEngine(storage: storage)
        // Pick a query unrelated to "rare" — use doc-0's token.
        let queryToken = rowsByDoc["doc-0"]![0]

        let hits = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "needle",
            topK: 10
        )
        // "rare" must surface from FTS even though vector likely
        // ranks doc-0 first.
        #expect(hits.contains { $0.uuid == "rare" })
        let rareHit = hits.first { $0.uuid == "rare" }!
        #expect(rareHit.ftsRank == 1)
    }

    // MARK: - Dual-source fusion

    @Test("doc present in both sources beats single-source-only docs")
    func dualSourceDominance() async throws {
        let dims = 16
        // Three docs. The first ("both") will be vector rank 1 (we'll
        // query with one of its tokens) AND FTS rank 1 (only it
        // contains the query word).
        let docs: [(uuid: String, body: String)] = [
            ("both",     "specialword alpha beta"),
            ("vec-only", "gamma delta epsilon"),
            ("fts-only", "specialword zeta eta"),
        ]
        let (storage, rowsByDoc) = try await Self.makeIndexedStorage(
            documents: docs, dims: dims, seed: 23
        )
        let engine = SearchEngine(storage: storage)
        let queryToken = rowsByDoc["both"]![0]

        let hits = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "specialword",
            topK: 10
        )

        // "both" appears in vector AND fts → highest fused score.
        #expect(hits[0].uuid == "both")
        let bothHit = hits[0]
        #expect(bothHit.vectorRank != nil)
        #expect(bothHit.ftsRank != nil)
        // Score must equal the sum of the two per-source contributions.
        let expected =
            1 / (Float(60) + Float(bothHit.vectorRank!)) +
            1 / (Float(60) + Float(bothHit.ftsRank!))
        #expect(bothHit.score == expected)
    }

    @Test("topK truncates the fused list")
    func topKTruncates() async throws {
        let storage = try await Self.makeFtsOnlyStorage(
            (0..<5).map { ("uuid-\($0)", "common token-\($0)") }
        )
        let engine = SearchEngine(storage: storage)

        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "common",
            topK: 3
        )
        #expect(hits.count == 3)
    }

    // MARK: - Filter

    @Test("StorageFilter.uuidIn excludes documents from both sources")
    func filterUuidInExcludes() async throws {
        let storage = try await Self.makeFtsOnlyStorage([
            ("keep",   "common everywhere"),
            ("drop",   "common everywhere"),
            ("keep-2", "common everywhere"),
        ])
        let engine = SearchEngine(storage: storage)

        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "common",
            topK: 10,
            filter: .uuidIn(Set(["keep", "keep-2"]))
        )
        let uuids: Set<String> = Set(hits.map { $0.uuid })
        #expect(uuids == ["keep", "keep-2"])
    }

    // MARK: - Formula reference

    @Test("RRF formula matches hand-computed reference (1-indexed, k=60)")
    func rrfFormulaLocked() async throws {
        // FTS-only: bodies crafted so the term-overlap ranking is
        // deterministic and known: each body has one shared token plus
        // a unique distinguishing token in lexicographic order. The
        // overlap score is the same for every document that contains
        // "shared", so the uuid tie-break orders them.
        let storage = try await Self.makeFtsOnlyStorage([
            ("aaa", "shared other"),
            ("bbb", "shared other"),
            ("ccc", "shared other"),
        ])
        let engine = SearchEngine(storage: storage)

        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "shared",
            topK: 10,
            config: HybridConfig(rrfK: 60, perSourceBudget: 50)
        )
        #expect(hits.map { $0.uuid } == ["aaa", "bbb", "ccc"])
        // Reference scores: 1/(60+1), 1/(60+2), 1/(60+3).
        #expect(hits[0].score == 1 / Float(61))
        #expect(hits[1].score == 1 / Float(62))
        #expect(hits[2].score == 1 / Float(63))
        #expect(hits[0].ftsRank == 1)
        #expect(hits[1].ftsRank == 2)
        #expect(hits[2].ftsRank == 3)
    }

    @Test("custom rrfK is honoured")
    func customRrfK() async throws {
        let storage = try await Self.makeFtsOnlyStorage([("only", "term")])
        let engine = SearchEngine(storage: storage)
        let hits = try await engine.searchHybrid(
            queryEmbeddings: [],
            dims: 16,
            queryText: "term",
            topK: 10,
            config: HybridConfig(rrfK: 10, perSourceBudget: 50)
        )
        #expect(hits.count == 1)
        #expect(hits[0].score == 1 / Float(11))
    }

    // MARK: - Determinism

    @Test("same inputs produce byte-identical hits across runs")
    func determinism() async throws {
        let dims = 16
        let docs: [(uuid: String, body: String)] = (0..<10).map {
            ("doc-\(String(format: "%02d", $0))", "body keyword token-\($0)")
        }
        let (storage, rowsByDoc) = try await Self.makeIndexedStorage(
            documents: docs, dims: dims, seed: 99
        )
        let engine = SearchEngine(storage: storage)
        let queryToken = rowsByDoc["doc-04"]![3]

        let first = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "keyword",
            topK: 5
        )
        let second = try await engine.searchHybrid(
            queryEmbeddings: queryToken,
            dims: dims,
            queryText: "keyword",
            topK: 5
        )
        #expect(first == second)
    }
}
