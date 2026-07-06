// SPDX-License-Identifier: Apache-2.0
//
// Determinism regression suite for the #140 search-latency refactor
// (ADR 006(f) amendment, ADR 035).
//
// `SearchEngine.search` is parallelised internally (`TaskGroup`-based
// bucket decode) and deduplicates query tokens as of ADR 035, while
// `searchLegacy` (reachable via `SearchConfig.legacySequentialSearch`)
// remains the frozen pre-#140 sequential/no-dedup baseline. This suite
// is the regression backstop for both properties the ADR amendment
// claims:
//
//   1. Repeated identical `search()` calls against unchanged storage
//      produce byte-identical `[SearchHit]` output (no nondeterminism
//      from TaskGroup completion order).
//   2. `searchOptimized` (default) and `searchLegacy` produce
//      equivalent output for the same input, including queries with
//      duplicated tokens (dedup correctness).
//   3. Concurrent overlapping `search()` calls into the same
//      `SearchEngine` do not corrupt results (actor reentrancy safety —
//      the parallel decode's child tasks touch no actor-isolated
//      state, but this is the end-to-end backstop for that claim).

import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("SearchDeterminism")
struct SearchDeterminismTests {

    // MARK: - Fixture

    /// Corpus large enough that at least one generation's selected
    /// bucket set exceeds `decodeSelectedBuckets`'s sequential-fallback
    /// threshold (64), so these tests actually exercise the `TaskGroup`
    /// decode path rather than only its sequential fallback.
    static func makeLargeSyntheticIndex(dims: Int, seed: UInt64) async throws -> (
        storage: InMemoryStorage,
        rowsByDoc: [String: [[Float]]]
    ) {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(
            storage: storage,
            config: IndexerConfig.testing(l0Capacity: 1024, lsmFanout: 4, seed: seed)
        )
        var rng = SplitMix64(seed: seed)

        var rowsByDoc: [String: [[Float]]] = [:]
        let docCount = 200
        let tokensPerDoc = 25
        for d in 0..<docCount {
            let uuid = String(format: "doc-%04d", d)
            var rows = [[Float]]()
            var flat = [Float]()
            for _ in 0..<tokensPerDoc {
                let row = SearchEngineTests.randomRow(dims: dims, rng: &rng)
                rows.append(row)
                flat.append(contentsOf: row)
            }
            rowsByDoc[uuid] = rows

            let hash = "h-\(uuid)"
            let chunk = try await storage.upsertChunk(
                ChunkRecord(hash: hash, model: "test", embeddings: Data(), counts: [tokensPerDoc])
            )
            try await storage.upsertDocument(
                DocumentRecord(
                    uuid: uuid,
                    date: Date(timeIntervalSince1970: 0),
                    metadata: Data(),
                    hash: hash,
                    body: "body of \(uuid)",
                    lens: [tokensPerDoc]
                )
            )
            try await indexer.add(chunkID: chunk.id, embeddings: flat, dims: dims)
        }
        try await indexer.flush()
        return (storage, rowsByDoc)
    }

    /// Build a synthetic query of `tokenCount` rows drawn from indexed
    /// document tokens, with some rows repeated verbatim so dedup logic
    /// is exercised.
    static func makeQueryWithDuplicates(
        rowsByDoc: [String: [[Float]]], tokenCount: Int
    ) -> [Float] {
        let allRows = rowsByDoc.values.flatMap { $0 }
        var query: [Float] = []
        // Half the tokens are distinct rows; the other half repeats the
        // first few, so `duplicateCount` is non-trivial for several rows.
        let distinctCount = max(1, tokenCount / 2)
        for i in 0..<tokenCount {
            let row = allRows[(i % distinctCount) % allRows.count]
            query.append(contentsOf: row)
        }
        return query
    }

    // MARK: - 1. Repeated identical searches are byte-identical

    @Test("N repeated identical search() calls produce byte-identical [SearchHit]")
    func repeatedSearchesAreByteIdentical() async throws {
        let dims = 24
        let (storage, rowsByDoc) = try await Self.makeLargeSyntheticIndex(dims: dims, seed: 101)
        let engine = SearchEngine(storage: storage)
        let query = Self.makeQueryWithDuplicates(rowsByDoc: rowsByDoc, tokenCount: 20)

        var results: [[SearchHit]] = []
        for _ in 0..<8 {
            let hits = try await engine.search(queryEmbeddings: query, dims: dims, topK: 20)
            results.append(hits)
        }

        #expect(!results[0].isEmpty, "expected non-empty hits from a query built from indexed tokens")
        for hits in results.dropFirst() {
            #expect(hits == results[0], "repeated search() calls diverged — TaskGroup decode is not deterministic")
        }
    }

    // MARK: - 2. searchOptimized vs searchLegacy equivalence

    @Test("searchOptimized and searchLegacy agree on plain queries (no duplicate tokens)")
    func optimizedMatchesLegacyPlainQuery() async throws {
        let dims = 24
        let (storage, rowsByDoc) = try await Self.makeLargeSyntheticIndex(dims: dims, seed: 202)

        let allRows = rowsByDoc.values.flatMap { $0 }
        var query: [Float] = []
        for i in 0..<16 { query.append(contentsOf: allRows[i * 7 % allRows.count]) }

        let optimizedEngine = SearchEngine(storage: storage, config: SearchConfig())
        let legacyEngine = SearchEngine(
            storage: storage, config: SearchConfig(legacySequentialSearch: true)
        )

        let optimizedHits = try await optimizedEngine.search(queryEmbeddings: query, dims: dims, topK: 20)
        let legacyHits = try await legacyEngine.search(queryEmbeddings: query, dims: dims, topK: 20)

        #expect(!legacyHits.isEmpty)
        #expect(optimizedHits == legacyHits,
                "searchOptimized diverged from searchLegacy on a duplicate-free query")
    }

    /// Unlike the duplicate-free case (bit-exact — see
    /// `optimizedMatchesLegacyPlainQuery`), a duplicated-token query
    /// makes `searchOptimized` compute each duplicate's contribution as
    /// `v * Float(duplicateCount)` where `searchLegacy` computes it as
    /// `duplicateCount` separate floating-point additions interleaved
    /// with other query tokens' contributions. Both are the same real
    /// number mathematically (ADR 028's no-op proof, applied in the
    /// collapse direction — see ADR 035), but IEEE-754 addition is not
    /// associative, so the two summation orders can round to adjacent
    /// floats. This is the same class of arithmetic-reordering
    /// tolerance the project already accepts for BLAS reduction order
    /// (`matmulQueryTimesRowMajorTranspose`'s doc comment) and for
    /// cross-implementation parity (±0.01–0.025 in the NFCorpus / facts
    /// suites) — compare with a tight tolerance, not bit-exact equality.
    static let dedupScoreTolerance: Float = 1e-4

    @Test("searchOptimized and searchLegacy agree on a query with duplicated tokens")
    func optimizedMatchesLegacyWithDuplicateTokens() async throws {
        let dims = 24
        let (storage, rowsByDoc) = try await Self.makeLargeSyntheticIndex(dims: dims, seed: 303)
        let query = Self.makeQueryWithDuplicates(rowsByDoc: rowsByDoc, tokenCount: 20)

        let optimizedEngine = SearchEngine(storage: storage, config: SearchConfig())
        let legacyEngine = SearchEngine(
            storage: storage, config: SearchConfig(legacySequentialSearch: true)
        )

        let optimizedHits = try await optimizedEngine.search(queryEmbeddings: query, dims: dims, topK: 20)
        let legacyHits = try await legacyEngine.search(queryEmbeddings: query, dims: dims, topK: 20)

        #expect(!legacyHits.isEmpty)
        #expect(optimizedHits.map(\.uuid) == legacyHits.map(\.uuid),
                "searchOptimized's query-token dedup changed the ranked document order vs searchLegacy")
        for (optimizedHit, legacyHit) in zip(optimizedHits, legacyHits) {
            #expect(optimizedHit.uuid == legacyHit.uuid)
            let message = "score for \(optimizedHit.uuid) diverged beyond float-reordering tolerance: optimized=\(optimizedHit.score) legacy=\(legacyHit.score)"
            #expect(abs(optimizedHit.score - legacyHit.score) < Self.dedupScoreTolerance, "\(message)")
        }
    }

    // MARK: - 3. Concurrent overlapping searches don't corrupt results

    @Test("concurrent overlapping search() calls into the same SearchEngine don't corrupt results")
    func concurrentOverlappingSearchesAreSafe() async throws {
        let dims = 24
        let (storage, rowsByDoc) = try await Self.makeLargeSyntheticIndex(dims: dims, seed: 404)
        let engine = SearchEngine(storage: storage)

        // Two distinct queries, interleaved across many concurrent tasks,
        // each asserting it only ever sees its own query's expected result.
        let queryA = Self.makeQueryWithDuplicates(rowsByDoc: rowsByDoc, tokenCount: 18)
        let queryB = Self.makeQueryWithDuplicates(rowsByDoc: rowsByDoc, tokenCount: 22)

        let referenceA = try await engine.search(queryEmbeddings: queryA, dims: dims, topK: 20)
        let referenceB = try await engine.search(queryEmbeddings: queryB, dims: dims, topK: 20)
        #expect(!referenceA.isEmpty)
        #expect(!referenceB.isEmpty)

        try await withThrowingTaskGroup(of: (Bool, [SearchHit]).self) { group in
            for i in 0..<24 {
                let useA = i % 2 == 0
                let query = useA ? queryA : queryB
                group.addTask {
                    let hits = try await engine.search(queryEmbeddings: query, dims: dims, topK: 20)
                    return (useA, hits)
                }
            }
            for try await (useA, hits) in group {
                let expected = useA ? referenceA : referenceB
                #expect(hits == expected,
                        "concurrent search() call returned a result inconsistent with its own query — possible actor reentrancy corruption")
            }
        }
    }
}
