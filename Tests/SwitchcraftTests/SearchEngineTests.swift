import Foundation
import Testing
@testable import SwitchcraftCore

@Suite("Search Engine")
struct SearchEngineTests {

    // MARK: - Helpers

    /// Build a unit-norm random row of length `dims`.
    static func randomRow(dims: Int, rng: inout SplitMix64) -> [Float] {
        var v = [Float](repeating: 0, count: dims)
        for j in 0..<dims {
            let u = Float(rng.next()) / Float(UInt64.max)
            v[j] = u - 0.5
        }
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            for j in 0..<dims { v[j] /= norm }
        }
        return v
    }

    static func l2Normalise(_ row: [Float]) -> [Float] {
        let n = sqrt(row.reduce(0) { $0 + $1 * $1 })
        guard n > 0 else { return row }
        return row.map { $0 / n }
    }

    /// Build a synthetic in-memory index and return:
    /// - storage: the populated InMemoryStorage
    /// - rowsByDoc: per-document the rows that were indexed
    /// - chunkID: the chunkID assigned to each document
    static func makeSyntheticIndex(
        documents docs: Int,
        tokensPerDoc: Int,
        dims: Int,
        seed: UInt64
    ) async throws -> (
        storage: InMemoryStorage,
        rowsByDoc: [String: [[Float]]],
        chunkIDs: [String: Int64]
    ) {
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(
            storage: storage,
            config: IndexerConfig.testing(l0Capacity: 1024, lsmFanout: 4)
        )
        var rng = SplitMix64(seed: seed)

        var rowsByDoc: [String: [[Float]]] = [:]
        var chunkIDs: [String: Int64] = [:]
        for d in 0..<docs {
            let uuid = String(format: "doc-%03d", d)
            var rows = [[Float]]()
            var flat = [Float]()
            for _ in 0..<tokensPerDoc {
                let row = randomRow(dims: dims, rng: &rng)
                rows.append(row)
                flat.append(contentsOf: row)
            }
            rowsByDoc[uuid] = rows

            // Each document gets one chunk that holds all of its tokens.
            let hash = "h-\(uuid)"
            let chunk = try await storage.upsertChunk(
                ChunkRecord(
                    hash: hash,
                    model: "test",
                    embeddings: Data(),
                    counts: [tokensPerDoc]
                )
            )
            chunkIDs[uuid] = chunk.id

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
        return (storage, rowsByDoc, chunkIDs)
    }

    // MARK: - Top-hit retrieval

    @Test("a query token equal to an indexed token retrieves its owning document at rank 1")
    func selfRetrievalRetrievesOwner() async throws {
        let dims = 16
        let (storage, rowsByDoc, _) = try await Self.makeSyntheticIndex(
            documents: 30, tokensPerDoc: 8, dims: dims, seed: 42
        )

        let engine = SearchEngine(storage: storage)
        let target = "doc-007"
        let query = rowsByDoc[target]![3]  // pick token 3 of the target doc

        let hits = try await engine.search(
            queryEmbeddings: query,
            dims: dims,
            topK: 5
        )
        #expect(hits.first?.uuid == target)
    }

    @Test("a query equal to the mean of a document's tokens scores that document highest")
    func meanQueryRetrievesOwner() async throws {
        let dims = 16
        let (storage, rowsByDoc, _) = try await Self.makeSyntheticIndex(
            documents: 25, tokensPerDoc: 8, dims: dims, seed: 7
        )

        let engine = SearchEngine(storage: storage)
        let target = "doc-012"
        let rows = rowsByDoc[target]!
        var mean = [Float](repeating: 0, count: dims)
        for row in rows {
            for j in 0..<dims { mean[j] += row[j] }
        }
        let inv = 1.0 / Float(rows.count)
        for j in 0..<dims { mean[j] *= inv }
        let query = Self.l2Normalise(mean)

        let hits = try await engine.search(
            queryEmbeddings: query,
            dims: dims,
            topK: 5
        )
        #expect(hits.first?.uuid == target)
        // Score must be strictly the maximum across hits.
        if let top = hits.first {
            for hit in hits.dropFirst() {
                #expect(top.score >= hit.score)
            }
        }
    }

    // MARK: - topK and filter

    @Test("topK strictly bounds the result count")
    func topKBoundsResults() async throws {
        let dims = 16
        let (storage, rowsByDoc, _) = try await Self.makeSyntheticIndex(
            documents: 30, tokensPerDoc: 8, dims: dims, seed: 3
        )
        let engine = SearchEngine(storage: storage)
        let query = rowsByDoc["doc-000"]![0]

        let three = try await engine.search(queryEmbeddings: query, dims: dims, topK: 3)
        #expect(three.count <= 3)

        let one = try await engine.search(queryEmbeddings: query, dims: dims, topK: 1)
        #expect(one.count <= 1)
    }

    @Test("StorageFilter.uuidIn excludes documents not in the set")
    func filterExcludes() async throws {
        let dims = 16
        let (storage, rowsByDoc, _) = try await Self.makeSyntheticIndex(
            documents: 20, tokensPerDoc: 6, dims: dims, seed: 11
        )
        let engine = SearchEngine(storage: storage)
        let query = rowsByDoc["doc-005"]![0]

        // Allowed set excludes doc-005, the natural top hit.
        let allowed: Set<String> = ["doc-001", "doc-002", "doc-003"]
        let hits = try await engine.search(
            queryEmbeddings: query,
            dims: dims,
            topK: 10,
            filter: .uuidIn(allowed)
        )
        for hit in hits {
            #expect(allowed.contains(hit.uuid))
        }
        #expect(hits.contains { $0.uuid == "doc-005" } == false)
    }

    // MARK: - MaxSim correctness

    @Test("MaxSim score equals the locked mean-of-per-token-max formula")
    func maxSimScoreFormula() async throws {
        // Build a tiny deterministic index: one doc with 4 tokens.
        // Pick query tokens that exactly match two of those tokens; with
        // the remaining query tokens kept at -infinity baseline, the
        // document score must equal mean(dot products).
        let dims = 8
        let storage = InMemoryStorage()
        try await storage.open()
        let indexer = try await Indexer(
            storage: storage,
            config: IndexerConfig.testing(l0Capacity: 1024, lsmFanout: 4)
        )

        var rng = SplitMix64(seed: 100)
        let rows = (0..<4).map { _ in Self.randomRow(dims: dims, rng: &rng) }
        var flat = [Float]()
        for row in rows { flat.append(contentsOf: row) }

        let chunk = try await storage.upsertChunk(
            ChunkRecord(hash: "only-chunk", model: "test", embeddings: Data(), counts: [4])
        )
        try await storage.upsertDocument(
            DocumentRecord(
                uuid: "the-doc",
                date: Date(timeIntervalSince1970: 0),
                hash: "only-chunk",
                body: "x",
                lens: [4]
            )
        )
        try await indexer.add(chunkID: chunk.id, embeddings: flat, dims: dims)
        try await indexer.flush()

        // Two query tokens, each matching one of the indexed rows.
        var query = [Float]()
        query.append(contentsOf: rows[0])
        query.append(contentsOf: rows[2])

        let engine = SearchEngine(storage: storage)
        let hits = try await engine.search(queryEmbeddings: query, dims: dims, topK: 1)
        let hit = try #require(hits.first)
        #expect(hit.uuid == "the-doc")

        // Reconstructed candidate ≈ centroid + residual, lossy by Q4. The
        // engine reports `mean_q max_t (q · approxToken)`. Since the
        // query tokens equal the original rows, the per-q max picks
        // approxToken[0] for q=0 and approxToken[2] for q=1. Compute
        // the same approximation and compare.
        let gens = try await storage.generations()
        var approxByPair = [IndexPair: [Float]]()
        for gen in gens {
            let buckets = try await storage.buckets(forGeneration: gen.id)
            for bucket in buckets {
                let center = Indexer.decodeFloat32LE(bucket.center)
                let pairs = try IndicesCodec.decode(bucket.indices)
                let resid = Q4Codec.decodeResiduals(bucket.residuals)
                for (i, p) in pairs.enumerated() {
                    var v = [Float](repeating: 0, count: dims)
                    for j in 0..<dims {
                        v[j] = center[j] + resid[i * dims + j]
                    }
                    approxByPair[p] = v
                }
            }
        }
        // Expected per-q-token max = max over all reconstructed tokens of
        // (q · approxToken). Compute by enumerating all approx tokens.
        let allApprox = (0..<4).compactMap {
            approxByPair[IndexPair(chunkID: UInt32(chunk.id), tokenOffset: UInt32($0))]
        }

        func dot(_ a: [Float], _ b: [Float]) -> Float {
            var s: Float = 0
            for i in 0..<a.count { s += a[i] * b[i] }
            return s
        }
        let q0Scores = allApprox.map { dot(rows[0], $0) }
        let q1Scores = allApprox.map { dot(rows[2], $0) }
        let q0Max = q0Scores.max()!
        let q1Max = q1Scores.max()!
        let expected = (q0Max + q1Max) / 2.0

        // The per-q max should pick the matching reconstructed token —
        // i.e. token 0 for q=rows[0], token 2 for q=rows[2]. Q4 quantisation
        // is lossy, so this isn't guaranteed mathematically; this is a
        // sanity check that the test corpus is well-behaved.
        #expect(q0Scores.firstIndex(of: q0Max) == 0)
        #expect(q1Scores.firstIndex(of: q1Max) == 2)

        #expect(abs(hit.score - expected) < 1e-5,
                "hit.score \(hit.score) vs expected \(expected)")
    }

    // MARK: - Determinism

    @Test("identical searches over identical storage produce bit-identical results")
    func determinism() async throws {
        let dims = 16
        let (storage, rowsByDoc, _) = try await Self.makeSyntheticIndex(
            documents: 20, tokensPerDoc: 6, dims: dims, seed: 99
        )
        let engine = SearchEngine(storage: storage)
        let query = rowsByDoc["doc-009"]![1]

        let first = try await engine.search(queryEmbeddings: query, dims: dims, topK: 5)
        let second = try await engine.search(queryEmbeddings: query, dims: dims, topK: 5)

        #expect(first.count == second.count)
        for (a, b) in zip(first, second) {
            #expect(a.uuid == b.uuid)
            #expect(a.score.bitPattern == b.score.bitPattern,
                    "scores differ at \(a.uuid): \(a.score) vs \(b.score)")
        }
    }

    // MARK: - score()

    @Test("score(passages) ranks the passage that exactly matches the query highest")
    func scorePassagesPicksMatchingPassage() async throws {
        let dims = 16
        var rng = SplitMix64(seed: 5)
        let queryTokens = (0..<3).map { _ in Self.randomRow(dims: dims, rng: &rng) }

        var query = [Float]()
        for row in queryTokens { query.append(contentsOf: row) }

        // Passage A: exactly the query tokens (highest possible score).
        var passageA = [Float]()
        for row in queryTokens { passageA.append(contentsOf: row) }

        // Passage B: random tokens.
        var passageB = [Float]()
        for _ in 0..<5 {
            passageB.append(contentsOf: Self.randomRow(dims: dims, rng: &rng))
        }

        // Passage C: contains one matching token plus padding.
        var passageC = [Float]()
        passageC.append(contentsOf: queryTokens[0])
        for _ in 0..<4 {
            passageC.append(contentsOf: Self.randomRow(dims: dims, rng: &rng))
        }

        let storage = InMemoryStorage()
        try await storage.open()
        let engine = SearchEngine(storage: storage)
        let scores = try await engine.score(
            queryEmbeddings: query,
            dims: dims,
            passages: [passageA, passageB, passageC]
        )
        #expect(scores.count == 3)
        #expect(scores[0] >= scores[1])
        #expect(scores[0] >= scores[2])
    }
}
