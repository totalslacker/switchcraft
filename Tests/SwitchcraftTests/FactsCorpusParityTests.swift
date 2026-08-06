import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Cross-implementation parity gate against upstream Witchcraft's
/// 33-fact end-to-end corpus.
///
/// Asserts that, for the same indexed corpus, Switchcraft's
/// `SearchEngine.search` (the pure-vector pipeline — see ADR 008 for why
/// the hybrid surface is *not* a parity gate) matches the reference
/// values produced by Witchcraft's `match_centroids` for every query
/// in `Tests/Fixtures/facts_corpus.json` under the relaxed FP32-vs-Q4K
/// shape documented in ADR 010(h):
///
/// - Top-1 doc-id: strict equality with Witchcraft.
/// - Top-3 doc-id set: set equality (membership, not order).
/// - Per-doc score: within ±0.025 of the Witchcraft reference for that
///   doc-id (matched by doc-id, not by rank).
/// - Doc order beyond rank 3: best-effort, no assertion.
///
/// k=32, tPrime=40000, and the asset-gate trait are spec'd by
/// ADR 006 / ADR 010.
///
/// ### `SharedStore` lifecycle convention
///
/// The 33 facts and the model load (~1–3 s for ANE compilation) are
/// amortised across every test in the suite via a single
/// `SharedStore.shared` actor. The actor holds an in-memory SQLite
/// `SwitchcraftStore`, an `Embedder`, and a `SearchEngine` over the
/// same `SwitchcraftStorage` instance the store uses internally. It is
/// **never explicitly shut down** — Swift Testing has no equivalent of
/// `tearDownAll`, the store is read-only after the initial index, and
/// the `:memory:` SQLite handle cleans up at process exit.
///
/// Future asset-gated suites that need a one-time-built index should
/// follow the same pattern: build the heavy resources lazily inside a
/// `static let shared` actor; do not try to tear them down.
///
/// ### Why a separate `SearchEngine` instance
///
/// `SwitchcraftStore.searchEngine` is `private`. To keep production
/// source unchanged, the suite constructs its own `SearchEngine` over
/// the same `SQLiteStorage` it hands to the store. Both engines are
/// read-only views over the same backing storage after indexing
/// completes — concurrency-safe by construction.
@Suite(
    "33-Fact Corpus Parity (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable)
)
struct FactsCorpusParityTests {

    // MARK: - Fixture model

    struct FactsCorpusFixture: Decodable {
        let provenance: Provenance
        let facts: [String]
        let queries: [Query]

        struct Provenance: Decodable {
            let witchcraftCommit: String
            let generatedAt: String
            let computeUnits: String
            let k: Int
            let tPrime: Int
            let threshold: Float
        }

        struct Query: Decodable {
            let query: String
            let expected: [Expected]
        }

        struct Expected: Decodable {
            let docId: String
            let score: Float
        }
    }

    static let fixture: FactsCorpusFixture = {
        guard
            let url = Bundle.module.url(
                forResource: "facts_corpus",
                withExtension: "json",
                subdirectory: "Fixtures"
            )
        else {
            fatalError("facts_corpus.json missing from test bundle")
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(FactsCorpusFixture.self, from: data)
        } catch {
            fatalError("Failed to decode facts_corpus.json: \(error)")
        }
    }()

    // MARK: - SharedStore

    /// One MLModel load + one 33-fact index per test run, shared
    /// across every per-query test in the suite.
    actor SharedStore {
        static let shared = SharedStore()

        private var built: Built?

        struct Built {
            let embedder: T5CoreMLEmbedder
            let store: SwitchcraftStore
            let storage: SQLiteStorage
            let searchEngine: SearchEngine
        }

        func get() async throws -> Built {
            if let built { return built }

            let modelURL = CoreMLAsset.url!  // gate ensures non-nil
            let tokenizerURL = Bundle.module.url(
                forResource: "xtr-base-en.tokenizer",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
            let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
            let embedder = try await T5CoreMLEmbedder(
                modelURL: modelURL,
                tokenizer: tokenizer
            )

            let storage = SQLiteStorage(path: ":memory:")
            let store = try await SwitchcraftStore(
                storage: storage,
                embedder: embedder,
                config: .default
            )

            for (index, fact) in FactsCorpusParityTests.fixture.facts.enumerated() {
                try await store.add(id: "\(index)", body: fact)
            }
            // Explicit flush at the parity gate so we are not relying on
            // `search`'s implicit flush. See plan: explicit-is-better-
            // than-implicit at this site.
            try await store.index()

            let searchEngine = SearchEngine(
                storage: storage,
                config: StoreConfig.default.search
            )

            let result = Built(
                embedder: embedder,
                store: store,
                storage: storage,
                searchEngine: searchEngine
            )
            built = result
            return result
        }
    }

    // MARK: - Per-query parity tests

    /// Cross-implementation tolerance.
    ///
    /// Switchcraft runs FP32-compute / FP16-outputs CoreML; Witchcraft
    /// runs Q4K GGUF. Matched-precision FP16-vs-Q4K is unattainable on
    /// the T5 encoder graph (full FP16 compute produces NaN — see fix
    /// #31 / ADR 010(c) amendment). Tolerance bands are calibrated to
    /// the measured FP32-vs-Q4K drift (worst observed Δ ≈ 0.024 on the
    /// duplicated body in `FACTS[0]`/`FACTS[16]`).
    private static let scoreTolerance: Float = 0.025
    private static let topKSetCheckRanks = 3

    private static func assertParity(queryIndex: Int) async throws {
        let built = try await SharedStore.shared.get()
        let queryFixture = Self.fixture.queries[queryIndex]
        let queryEmbeddings = try await built.embedder.encode(queryFixture.query)

        let hits = try await built.searchEngine.search(
            queryEmbeddings: queryEmbeddings,
            dims: built.embedder.dims,
            topK: queryFixture.expected.count,
            filter: .all
        )

        let actualIds = hits.map(\.uuid)
        let expectedIds = queryFixture.expected.map(\.docId)

        // Top-1 must match Witchcraft exactly. This is the primary
        // parity gate.
        #expect(
            actualIds.first == expectedIds.first,
            """
            Top-1 for "\(queryFixture.query)" was \
            \(actualIds.first ?? "<nil>") — expected \
            \(expectedIds.first ?? "<nil>")
            """
        )

        // Top-K (K=3) document set must match — membership only, order
        // is allowed to permute under cross-implementation precision
        // drift. Beyond rank K the result order is best-effort.
        //
        // Derive K from the expected side and assert separately that
        // Switchcraft returned at least K hits, so a short result list
        // surfaces as a hard failure rather than a silently-weakened
        // set check.
        let k = min(Self.topKSetCheckRanks, expectedIds.count)
        #expect(
            actualIds.count >= k,
            """
            Query "\(queryFixture.query)" returned only \(actualIds.count) \
            hits — expected at least \(k) to validate top-\(k) set parity.
            """
        )
        let actualTopSet = Set(actualIds.prefix(k))
        let expectedTopSet = Set(expectedIds.prefix(k))
        #expect(
            actualTopSet == expectedTopSet,
            """
            Top-\(k) set for "\(queryFixture.query)" was \(actualTopSet) — \
            expected \(expectedTopSet). Ordering may differ within the \
            top-\(k); membership must not.
            """
        )

        // Per-doc score parity within tolerance, matched by doc-id (not
        // by rank). Comparing by rank would compare scores for
        // *different* documents whenever rank-ordering permutes within
        // the top-K — masking real drift on one doc and falsely
        // flagging another. Build a doc-id → expected-score map and
        // assert each returned hit's score against the reference for
        // its own doc-id.
        let expectedScoreByDocId: [String: Float] = Dictionary(
            uniqueKeysWithValues: queryFixture.expected.map { ($0.docId, $0.score) }
        )
        for hit in hits {
            guard let expected = expectedScoreByDocId[hit.uuid] else {
                // Doc returned by Switchcraft but not in the Witchcraft
                // reference top-K. The set-equality check above already
                // flags this for ranks < K; for ranks ≥ K it's
                // best-effort (no assertion per ADR 010(h)).
                continue
            }
            #expect(
                abs(hit.score - expected) <= Self.scoreTolerance,
                """
                Query "\(queryFixture.query)" doc \(hit.uuid): score \
                \(hit.score) drifted from Witchcraft reference \(expected) \
                by more than ±\(Self.scoreTolerance) (Δ=\(hit.score - expected)).
                """
            )
        }
    }

    @Test("Query 'a lake with funny colors' parity (top-1 strict, top-3 set, scores ±0.025)")
    func lakeWithFunnyColors() async throws {
        try await Self.assertParity(queryIndex: 0)
    }

    @Test("Query 'A group of flamingos' parity (top-1 strict, top-3 set, scores ±0.025)")
    func groupOfFlamingos() async throws {
        try await Self.assertParity(queryIndex: 1)
    }

    @Test("Query 'facts about fruits and berries' parity (top-1 strict, top-3 set, scores ±0.025)")
    func factsAboutFruitsAndBerries() async throws {
        try await Self.assertParity(queryIndex: 2)
    }

    // MARK: - Public RRF surface smoke test

    /// Exercises `SwitchcraftStore.search` (the public hybrid surface)
    /// for every query. Asserts only the top-1 doc-id matches the
    /// expected top-1 from the fixture — no score assertion, because
    /// RRF scores are not commensurate with Witchcraft's by ADR 008.
    @Test("Public RRF surface returns expected top-1 doc per query")
    func publicSurfaceTop1() async throws {
        let built = try await SharedStore.shared.get()
        for query in Self.fixture.queries {
            guard let expectedTop = query.expected.first else {
                Issue.record("Fixture query '\(query.query)' has no expected results")
                continue
            }
            let hits = try await built.store.search(
                query: query.query,
                topK: 1,
                filter: .all
            ).hits
            #expect(
                hits.first?.uuid == expectedTop.docId,
                """
                RRF top-1 for "\(query.query)" was \
                \(hits.first?.uuid ?? "<nil>") — expected \(expectedTop.docId)
                """
            )
        }
    }
}
#endif
