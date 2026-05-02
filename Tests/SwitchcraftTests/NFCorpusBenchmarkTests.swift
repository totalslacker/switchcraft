import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Cross-implementation NDCG@10 quality gate against the NFCorpus test
/// split.
///
/// Asserts that, for the full NFCorpus test corpus, Switchcraft's
/// vector-only `SearchEngine.search` (the parity site — Witchcraft has
/// no BM25, so the public RRF surface is not a parity gate per
/// ADR 008) achieves macro-averaged NDCG@10 within Witchcraft's
/// published [0.31, 0.33] band.
///
/// The 0.31–0.33 band comes from upstream Witchcraft's `make
/// nfcorpus-score` output and is also the target cited in
/// `docs/Plan.md` ("Performance Expectations" / NDCG@10 row). A future
/// model build that drifts outside this band should fail this test
/// loudly rather than have its bounds widened — that's the parity
/// gate's whole point per ADR 006.
///
/// ### Asset gating
///
/// Double-gated: requires both the CoreML model
/// (`SWITCHCRAFT_XTR_MLPACKAGE`) and the NFCorpus directory
/// (`SWITCHCRAFT_NFCORPUS_DIR` containing `nfcorpus.tsv`,
/// `questions.test.tsv`, `qrels.test.json`). When either is absent
/// the suite skips cleanly; CI sets neither. See
/// `Tests/SwitchcraftTests/Support/NFCorpusDataset.swift` and
/// `Tests/SwitchcraftTests/Support/CoreMLAsset.swift` for the asset
/// contracts; the README documents the developer setup.
///
/// ### Runtime
///
/// One-time index build encodes ~3,633 abstracts via CoreML T5,
/// roughly ~6 minutes on Apple Silicon. Followed by ~325 query
/// encodes. The suite is opt-in (`swift test --filter
/// NFCorpusBenchmark`) precisely because of this cost.
@Suite(
    "NFCorpus Benchmark (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable && NFCorpusDataset.isAvailable)
)
struct NFCorpusBenchmarkTests {

    // MARK: - Tunables

    /// Witchcraft's published NDCG@10 band on the NFCorpus test split.
    /// Source: `make nfcorpus-score` output / `docs/Plan.md`
    /// "Performance Expectations" row. Do not widen without
    /// justification; per ADR 006 this is the parity gate.
    private static let ndcgLowerBound: Double = 0.31
    private static let ndcgUpperBound: Double = 0.33

    private static let topK: Int = 10

    // MARK: - SharedStore

    /// One-time model load + corpus index, amortised across every
    /// test in the suite via the same `static let shared` actor
    /// pattern documented in `FactsCorpusParityTests.SharedStore`.
    actor SharedStore {
        static let shared = SharedStore()

        private var built: Built?

        struct Built {
            let embedder: T5CoreMLEmbedder
            let store: SwitchcraftStore
            let storage: SQLiteStorage
            let searchEngine: SearchEngine
            let queries: [(queryId: String, text: String)]
            let qrels: [String: [String: Int]]
        }

        func get() async throws -> Built {
            if let built { return built }

            let modelURL = CoreMLAsset.url!  // gate ensures non-nil
            let datasetDir = NFCorpusDataset.url!  // gate ensures non-nil
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

            let collectionMap = try loadCollectionMap(
                from: datasetDir.appendingPathComponent(NFCorpusDataset.collectionMapFileName)
            )
            let corpus = try loadCorpus(
                from: datasetDir.appendingPathComponent(NFCorpusDataset.corpusFileName),
                collectionMap: collectionMap
            )
            for row in corpus {
                try await store.add(id: row.docId, body: row.text)
            }
            try await store.index()

            let queries = try loadQueries(
                from: datasetDir.appendingPathComponent(NFCorpusDataset.queriesFileName)
            )
            let qrels = try loadQrels(
                from: datasetDir.appendingPathComponent(NFCorpusDataset.qrelsFileName)
            )

            let searchEngine = SearchEngine(
                storage: storage,
                config: StoreConfig.default.search
            )

            let result = Built(
                embedder: embedder,
                store: store,
                storage: storage,
                searchEngine: searchEngine,
                queries: queries,
                qrels: qrels
            )
            built = result
            return result
        }
    }

    // MARK: - The parity gate

    @Test("NDCG@10 ∈ [0.31, 0.33]")
    func ndcgAt10WithinBand() async throws {
        let built = try await SharedStore.shared.get()

        var perQueryNDCG: [Double] = []
        perQueryNDCG.reserveCapacity(built.queries.count)

        for query in built.queries {
            // Match upstream `score.py` (pytrec_eval): macro-average
            // is over queries that appear in qrels. Queries returned
            // by Switchcraft but absent from qrels are excluded.
            guard let relevant = built.qrels[query.queryId] else { continue }

            let queryEmbeddings = try await built.embedder.encode(query.text)
            if queryEmbeddings.isEmpty {
                // Empty query after tokenisation contributes 0 — same
                // as a missing run line in pytrec_eval.
                perQueryNDCG.append(0)
                continue
            }
            let hits = try await built.searchEngine.search(
                queryEmbeddings: queryEmbeddings,
                dims: built.embedder.dims,
                topK: Self.topK,
                filter: .all
            )
            let retrievedIds = hits.map(\.uuid)
            perQueryNDCG.append(ndcg10(retrieved: retrievedIds, relevant: relevant))
        }

        #expect(!perQueryNDCG.isEmpty, "No qrels matched any query — fixture mismatch")
        let macroNDCG = perQueryNDCG.reduce(0, +) / Double(perQueryNDCG.count)

        #expect(
            macroNDCG >= Self.ndcgLowerBound && macroNDCG <= Self.ndcgUpperBound,
            """
            Macro NDCG@10 = \(macroNDCG) — outside Witchcraft's published \
            band [\(Self.ndcgLowerBound), \(Self.ndcgUpperBound)]. \
            Per ADR 006 this is a parity gate; investigate before \
            relaxing the bounds.
            """
        )
    }
}

// MARK: - NDCG@10

/// Standard IR NDCG@10:
///   DCG = Σ_{i=1..10} (2^rel_i − 1) / log₂(i + 1)
///   IDCG = DCG of the top-10 relevant grades sorted descending
///   NDCG = DCG / IDCG, returning 0 when IDCG is 0 (no relevant docs).
private func ndcg10(retrieved: [String], relevant: [String: Int]) -> Double {
    var dcg = 0.0
    for (i, docId) in retrieved.prefix(10).enumerated() {
        let grade = relevant[docId] ?? 0
        if grade > 0 {
            dcg += (pow(2.0, Double(grade)) - 1.0) / log2(Double(i + 2))
        }
    }
    let idealGrades = relevant.values.sorted(by: >).prefix(10)
    var idcg = 0.0
    for (i, grade) in idealGrades.enumerated() {
        idcg += (pow(2.0, Double(grade)) - 1.0) / log2(Double(i + 2))
    }
    guard idcg > 0 else { return 0 }
    return dcg / idcg
}

// MARK: - Fixture loaders

private struct NFCorpusRow {
    let docId: String
    let text: String
}

/// Parse `collection_map.json`: `{ "numeric_id": "MED-*" }`.
/// Witchcraft's pre-staged corpus TSV uses numeric row indices; the map
/// translates them to the `MED-*` IDs used in qrels.
private func loadCollectionMap(from url: URL) throws -> [String: String] {
    let data = try Data(contentsOf: url)
    guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: String] else {
        throw NFCorpusFixtureError.qrelsNotNestedDict
    }
    return dict
}

/// Parse the corpus TSV. Witchcraft's pre-staged format is two columns:
/// `numeric_id \t text`. The numeric_id is mapped to a `MED-*` doc ID via
/// the `collectionMap` from `collection_map.json` (required for qrel lookup).
private func loadCorpus(from url: URL, collectionMap: [String: String]) throws -> [NFCorpusRow] {
    let raw = try String(contentsOf: url, encoding: .utf8)
    var rows: [NFCorpusRow] = []
    for line in raw.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard cols.count >= 2 else { continue }
        let numericId = String(cols[0])
        let text = String(cols[1...].joined(separator: "\t"))
        if numericId.isEmpty || text.isEmpty { continue }
        guard let docId = collectionMap[numericId] else { continue }
        rows.append(NFCorpusRow(docId: docId, text: text))
    }
    return rows
}

/// Parse the queries TSV. Columns: `query-id \t query`.
private func loadQueries(from url: URL) throws -> [(queryId: String, text: String)] {
    let raw = try String(contentsOf: url, encoding: .utf8)
    var queries: [(queryId: String, text: String)] = []
    for line in raw.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
        guard cols.count >= 2 else { continue }
        let qid = String(cols[0])
        let text = String(cols[1...].joined(separator: "\t"))
        if qid.isEmpty || text.isEmpty { continue }
        queries.append((queryId: qid, text: text))
    }
    return queries
}

/// Parse pytrec_eval-style qrels: `{ qid: { docid: grade } }`. Grades
/// are integers 0–3 in NFCorpus. Other JSON shapes (BeIR's TSV-as-JSON,
/// flat lists) are rejected with a clear error; the canonical fetch
/// source pins this shape.
private func loadQrels(from url: URL) throws -> [String: [String: Int]] {
    let data = try Data(contentsOf: url)
    let parsed = try JSONSerialization.jsonObject(with: data, options: [])
    guard let topDict = parsed as? [String: Any] else {
        throw NFCorpusFixtureError.qrelsNotNestedDict
    }
    var result: [String: [String: Int]] = [:]
    result.reserveCapacity(topDict.count)
    for (qid, value) in topDict {
        guard let inner = value as? [String: Any] else {
            throw NFCorpusFixtureError.qrelsNotNestedDict
        }
        var grades: [String: Int] = [:]
        grades.reserveCapacity(inner.count)
        for (docId, gradeAny) in inner {
            // Accept Int / NSNumber / numeric String. JSONSerialization
            // bridges JSON numbers to NSNumber, so the NSNumber path
            // covers both integer and float inputs.
            if let g = gradeAny as? NSNumber {
                grades[docId] = g.intValue
            } else if let s = gradeAny as? String, let g = Int(s) {
                grades[docId] = g
            } else {
                throw NFCorpusFixtureError.qrelsGradeNotInt(qid: qid, docId: docId)
            }
        }
        result[qid] = grades
    }
    return result
}

private enum NFCorpusFixtureError: Error, CustomStringConvertible {
    case qrelsNotNestedDict
    case qrelsGradeNotInt(qid: String, docId: String)

    var description: String {
        switch self {
        case .qrelsNotNestedDict:
            return "qrels.test.json must be a nested-dict { qid: { docid: grade } } — see scripts/fetch-nfcorpus.sh"
        case .qrelsGradeNotInt(let qid, let docId):
            return "qrels.test.json grade for (\(qid), \(docId)) is not an integer"
        }
    }
}

#endif
