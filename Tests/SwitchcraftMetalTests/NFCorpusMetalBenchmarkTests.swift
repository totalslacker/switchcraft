// SPDX-License-Identifier: Apache-2.0
//
// Cross-implementation NDCG@10 quality gate for `T5MetalEmbedder`.
// Issue #65 (umbrella #57).
//
// Sibling of `Tests/SwitchcraftTests/NFCorpusBenchmarkTests.swift` —
// asserts that for the full NFCorpus test split, Switchcraft's
// vector-only `SearchEngine.search` (the parity site — Witchcraft has
// no BM25, so the public RRF surface is not a parity gate per ADR 008)
// achieves macro-averaged NDCG@10 within Witchcraft's published
// [0.31, 0.33] band when running through `T5MetalEmbedder` instead of
// `T5CoreMLEmbedder`.
//
// Triple-gated: `SWITCHCRAFT_XTR_GGUF` (the GGUF asset),
// `SWITCHCRAFT_NFCORPUS_DIR` (the dataset directory), and a Metal-
// capable host. When any gate is unset the suite skips cleanly. CI
// sets none.
//
// One-time index build encodes ~3,633 abstracts via Metal T5 — multi-
// minute on Apple Silicon at the ~50 ms / 512-token target. This is
// why every gate is opt-in.

#if canImport(Metal)

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite
@_spi(SwitchcraftMetal) import SwitchcraftCore
@_spi(SwitchcraftMetal) import SwitchcraftMetal

// ASSET-GATED: requires SWITCHCRAFT_XTR_GGUF + SWITCHCRAFT_NFCORPUS_DIR — skips in CI
@Suite(
    "NFCorpus Metal Benchmark (asset-gated)",
    .enabled(if: GGUFAsset.isAvailable
                 && NFCorpusDatasetMetal.isAvailable
                 && MetalAvailability.isAvailable,
             "Set SWITCHCRAFT_XTR_GGUF + SWITCHCRAFT_NFCORPUS_DIR; requires Metal.")
)
struct NFCorpusMetalBenchmarkTests {

    // MARK: - Tunables

    /// Metal-specific NDCG@10 band.
    ///
    /// **Lower bound (0.31)** — Witchcraft's published floor from `make
    /// nfcorpus-score`. Any score below 0.31 indicates an algorithmic
    /// regression; do NOT lower this without a root-cause fix.
    ///
    /// **Upper bound (0.34)** — Calibrated (issue #75, 2026-05-02).
    /// Observed NDCG@10 = 0.336 with `xtr-v3.gguf` (Q4K) on the full
    /// NFCorpus test split. Metal's FP32-throughout arithmetic (ADR 014
    /// §precision policy) produces slightly higher quality than ggml's
    /// mixed-precision reference (Witchcraft's published ceiling: 0.33).
    /// This divergence is intentional; the upper bound = observed + ~0.004
    /// headroom. Do NOT raise the lower bound or drop the upper below 0.33.
    private static let ndcgLowerBound: Double = 0.31
    private static let ndcgUpperBound: Double = 0.34

    private static let topK: Int = 10

    // MARK: - SharedStore

    /// One-time model load + corpus index, amortised across every test
    /// in the suite via the same `static let shared` actor pattern as the
    /// CoreML sibling.
    actor SharedStore {
        static let shared = SharedStore()

        private var built: Built?

        struct Built {
            let embedder: T5MetalEmbedder
            let store: SwitchcraftStore
            let storage: SQLiteStorage
            let searchEngine: SearchEngine
            let queries: [(queryId: String, text: String)]
            let qrels: [String: [String: Int]]
        }

        func get() async throws -> Built {
            if let built { return built }

            let modelURL = GGUFAsset.url!  // gate ensures non-nil
            let datasetDir = NFCorpusDatasetMetal.url!  // gate ensures non-nil
            let tokenizerURL = try #require(
                TokenizerFixture.url,
                "tokenizer fixture missing — Tests/Fixtures/xtr-base-en.tokenizer.json absent"
            )
            let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
            let embedder = try await T5MetalEmbedder(
                modelURL: modelURL,
                tokenizer: tokenizer
            )

            let storage = SQLiteStorage(path: ":memory:")
            let store = try await SwitchcraftStore(
                storage: storage,
                embedder: embedder,
                config: .default
            )

            let collectionMap = try loadCollectionMapMetal(
                from: datasetDir.appendingPathComponent(NFCorpusDatasetMetal.collectionMapFileName)
            )
            let corpus = try loadCorpusMetal(
                from: datasetDir.appendingPathComponent(NFCorpusDatasetMetal.corpusFileName),
                collectionMap: collectionMap
            )
            for row in corpus {
                try await store.add(id: row.docId, body: row.text)
            }
            try await store.index()

            let queries = try loadQueriesMetal(
                from: datasetDir.appendingPathComponent(NFCorpusDatasetMetal.queriesFileName)
            )
            let qrels = try loadQrelsMetal(
                from: datasetDir.appendingPathComponent(NFCorpusDatasetMetal.qrelsFileName)
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

    @Test("NDCG@10 ∈ [0.31, 0.33] with T5MetalEmbedder")
    func ndcgAt10WithinBand() async throws {
        let built = try await SharedStore.shared.get()

        var perQueryNDCG: [Double] = []
        perQueryNDCG.reserveCapacity(built.queries.count)

        for query in built.queries {
            // Match upstream `score.py` (pytrec_eval): macro-average is
            // over queries that appear in qrels. Queries returned by
            // Switchcraft but absent from qrels are excluded.
            guard let relevant = built.qrels[query.queryId] else { continue }

            let queryEmbeddings = try await built.embedder.encode(query.text)
            if queryEmbeddings.isEmpty {
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
            perQueryNDCG.append(ndcg10Metal(retrieved: retrievedIds, relevant: relevant))
        }

        #expect(!perQueryNDCG.isEmpty, "No qrels matched any query — fixture mismatch")
        let macroNDCG = perQueryNDCG.reduce(0, +) / Double(perQueryNDCG.count)

        print("NFCorpusMetalBenchmark macro NDCG@10 = \(macroNDCG)")
        #expect(
            macroNDCG >= Self.ndcgLowerBound && macroNDCG <= Self.ndcgUpperBound,
            """
            Macro NDCG@10 = \(macroNDCG) outside Metal band \
            [\(Self.ndcgLowerBound), \(Self.ndcgUpperBound)]. \
            Score below 0.31 indicates a kernel regression — bisect \
            against the PyTorch FP32 fixture in T5MetalEmbedderTests. \
            Score above 0.34 means Metal is outperforming the calibrated \
            ceiling; recalibrate the upper bound (issue #75, 2026-05-02: \
            observed 0.336, upper set to 0.34).
            """
        )
    }
}

#endif // canImport(Metal)
