// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

// Corpus design:
//   - 5 relevant documents: distinct excerpts about Bartleby the Scrivener.
//   - 200 noise documents: synthetic text engineered to saturate the
//     "t", "le", "by" subword fragments that the T5 SentencePiece tokenizer
//     extracts from "bartleby" → ["bar", "t", "le", "by"].
//
// Without the filter (queryMinSurfaceFormLength=0), common fragments
// ("t", "le", "by") fire on nearly every noise document, compressing the
// gap between relevant and noise scores.  Precision@5 should be ≤ 0.2
// (the single top-ranked relevant doc).
//
// With the filter (queryMinSurfaceFormLength=2), those fragments are
// suppressed; only the specific "bar" embedding steers the ranking, and
// the relevant Bartleby documents rise to fill ≥3 of the top 5 positions.
//
// precision@5 = |{relevant docs in top 5}| / 5
//   - baseline (no filter):  < 0.4  (at most 1 of 5 positions is relevant)
//   - with filter:           ≥ 0.6  (at least 3 of 5 positions are relevant)
//
// Note: the 200-document scale was chosen as the minimum corpus size that
// reliably triggers the noise floor for this T5 checkpoint. If the
// testBaselineNoiseDegradation assertion fails, raise noiseDocumentCount.
@Suite(
    "Noise Floor Precision (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable)
)
struct NoiseFloorPrecisionTests {

    // MARK: - Corpus constants

    /// IDs for the 5 relevant (Bartleby-themed) documents.
    private static let relevantIDs: [String] = (0..<5).map { "bartleby-doc-\($0)" }

    /// Number of noise documents. Must be large enough to push the noise
    /// floor above the relevant documents' scores when the filter is off.
    private static let noiseDocumentCount = 200

    // MARK: - Relevant document bodies

    private static let bartlebyTexts: [String] = [
        """
        Bartleby was a scrivener employed in a Wall Street law office. He began \
        his duties efficiently, copying legal documents with great care. But soon \
        Bartleby acquired a strange habit of replying to every request with the \
        quiet phrase: I would prefer not to. Bartleby's passive refusals baffled \
        his employer, who felt unable to dismiss this peculiar, immovable man.
        """,
        """
        In the story of Bartleby the Scrivener by Herman Melville, the protagonist \
        named Bartleby works as a copyist in a Manhattan office. Bartleby gradually \
        refuses to perform his duties, repeating only that he would prefer not to \
        comply. Bartleby becomes a symbol of passive resistance against the demands \
        of commerce and routine.
        """,
        """
        Bartleby remained motionless in the corner of the office on Wall Street. \
        When asked to perform any task, Bartleby replied with the same quiet refusal. \
        The lawyer who employed Bartleby could not understand why Bartleby behaved \
        this way, yet felt a profound sympathy for the inscrutable copyist.
        """,
        """
        The tale of Bartleby explores themes of alienation, isolation, and the \
        dehumanising effects of modern office work. Bartleby's quiet rebellion \
        against a copying routine in a nineteenth century office has made Bartleby \
        one of American literature's most memorable characters. Bartleby's final \
        fate in the Tombs prison is both inevitable and deeply moving.
        """,
        """
        Herman Melville published Bartleby the Scrivener in 1853 in Putnam's Monthly \
        Magazine. Bartleby works in a Wall Street office copying legal documents by \
        hand. Bartleby increasingly refuses all requests from his employer, leaving \
        only the refrain I would prefer not to. Bartleby remains one of the most \
        enigmatic figures in the American literary canon.
        """,
    ]

    // MARK: - Noise document generation

    /// Produce noise text saturated with words containing "t", "le", and "by"
    /// subword fragments. These are the three problematic fragments from the
    /// "bartleby" decomposition that inflate noise scores when unfiltered.
    private static func noiseDocument(index: Int) -> String {
        let templates: [String] = [
            "The table nearby was stacked with bottles. By the time the people arrived it was little use to reset the whole setup. Whereby the nearby establishment delivered bottles to the table by bicycle at the settlement.",
            "At the settlement nearby little tables were set beside the bottle collection. It was a simple but valuable setup. By late afternoon the bottles at the table had settled into a suitable and stable arrangement by the wall.",
            "Nearby the little table stood bottles of a notable vintage. The notable label indicated a subtle and gentle blend. By simply tasting it people could tell the little bottle held a remarkable but simple recipe available at the outlet.",
            "The suitable settlement at the table included bottles placed neatly. People at the nearby establishment were able to stable the subtle blend. By itself the bottle at the table was the best little item available by the outlet.",
            "Bottles were stacked on little tables at the nearby settlement. By notable effort the establishment had assembled a remarkable and stable collection. Little by little the table filled with bottles until it was completely full by nightfall.",
            "The notable table at the nearby outlet had little bottles available by the gallon. By the time the settlement was complete it was little trouble to bottle the gentle blend. The table by the outlet was suitable for settling bottles.",
            "At the bottle outlet nearby the little table was suitable for tasting. By gentle handling the settlement of bottles at the table was notable. Little bubbles settled by the outlet noticeably affecting the subtle taste available.",
            "Bottles at the settlement were little by little collected and placed at the table nearby. The notable outlet by the table was a stable and suitable place. By gentle effort the little collection at the table became notable and complete.",
        ]
        let template = templates[index % templates.count]
        // Add index-specific variation to prevent all noise docs having identical embeddings.
        return "Entry \(index) at site \(index % 50): \(template)"
    }

    // MARK: - Shared resources

    actor SharedResources {
        static let shared = SharedResources()
        private var built: Built?

        struct Built {
            let embedder: T5CoreMLEmbedder
        }

        func get() async throws -> Built {
            if let built { return built }
            let modelURL = CoreMLAsset.url!
            let tokenizerURL = Bundle.module.url(
                forResource: "xtr-base-en.tokenizer",
                withExtension: "json",
                subdirectory: "Fixtures"
            )!
            let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)
            let embedder = try await T5CoreMLEmbedder(modelURL: modelURL, tokenizer: tokenizer)
            let result = Built(embedder: embedder)
            built = result
            return result
        }
    }

    // MARK: - Helpers

    private func buildLargeStore(queryMinSurfaceFormLength: Int) async throws -> SwitchcraftStore {
        let res = try await SharedResources.shared.get()
        let storage = SQLiteStorage(path: ":memory:")
        let searchConfig = SearchConfig(queryMinSurfaceFormLength: queryMinSurfaceFormLength)
        let config = StoreConfig(search: searchConfig)

        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: res.embedder,
            config: config
        )

        for (i, id) in Self.relevantIDs.enumerated() {
            try await store.add(
                id: id,
                title: "Bartleby the Scrivener",
                body: Self.bartlebyTexts[i]
            )
        }

        for i in 0..<Self.noiseDocumentCount {
            try await store.add(
                id: "noise-\(i)",
                body: Self.noiseDocument(index: i)
            )
        }

        try await store.index()
        return store
    }

    private func precision(at k: Int, hits: [HybridHit]) -> Double {
        let relevantInTopK = hits.prefix(k).filter { Self.relevantIDs.contains($0.uuid) }.count
        return Double(relevantInTopK) / Double(k)
    }

    private func hitsDescription(_ hits: [HybridHit]) -> String {
        hits.map { h in
            let vs = h.vectorScore.map { String(format: "%.4f", $0) } ?? "nil"
            return "\(h.uuid)(vs=\(vs))"
        }.joined(separator: ", ")
    }

    // MARK: - Tests

    /// Pins the noise-floor regression boundary: without the filter, common
    /// subword fragments from "bartleby" flood into noise documents and
    /// compress the precision gap. Precision@5 should stay low (≤ 0.2, i.e.
    /// only the rank-1 relevant document makes the top 5).
    ///
    /// If this test fails (precision@5 ≥ 0.4 without filter), the 200-doc
    /// corpus is insufficient to trigger the noise floor with this model
    /// checkpoint. Increase `noiseDocumentCount` until it fails as expected.
    @Test("Baseline: noise-floor degrades precision@5 without surface-form filter")
    func baselineNoiseDegradation() async throws {
        let store = try await buildLargeStore(queryMinSurfaceFormLength: 0)
        let hits = try await store.search(query: "bartleby", topK: 5).hits

        let p5 = precision(at: 5, hits: hits)
        try await store.shutdown()
        #expect(
            p5 < 0.4,
            """
            Noise-floor baseline not reproduced: precision@5 = \(p5) is not < 0.4 \
            without the surface-form filter. The \(Self.noiseDocumentCount)-document \
            noise corpus may be insufficient to trigger the floor for this model \
            checkpoint. Increase noiseDocumentCount and re-run. \
            Top-5 hits: \(hitsDescription(hits))
            """
        )
    }

    /// With `queryMinSurfaceFormLength = 2`, the common single- and
    /// two-character fragments ("t", "le", "by") are suppressed from the
    /// "bartleby" query representation. Only the more-specific "bar" token
    /// steers the search, and the five Bartleby documents should reclaim ≥3
    /// of the top-5 positions (precision@5 ≥ 0.6).
    @Test("Fix: queryMinSurfaceFormLength=2 raises precision@5 ≥ 0.6")
    func filterImprovesPrecision() async throws {
        let store = try await buildLargeStore(queryMinSurfaceFormLength: 2)
        let hits = try await store.search(query: "bartleby", topK: 5).hits

        let p5 = precision(at: 5, hits: hits)
        try await store.shutdown()
        #expect(
            p5 >= 0.6,
            """
            Surface-form filter did not improve precision@5 to ≥ 0.6: got \(p5). \
            Expected ≥3 of the 5 Bartleby documents in the top 5. \
            Top-5 hits: \(hitsDescription(hits))
            """
        )
    }

    /// Verifies that enabling the surface-form filter does not break top-1
    /// accuracy at small scale (35 documents). The single Bartleby target
    /// must remain at rank 1 even with "t"/"le"/"by" suppressed.
    @Test("Top-1 preserved at 35-doc scale with queryMinSurfaceFormLength=2")
    func top1PreservedAtSmallScale() async throws {
        let res = try await SharedResources.shared.get()
        let storage = SQLiteStorage(path: ":memory:")
        let searchConfig = SearchConfig(queryMinSurfaceFormLength: 2)
        let config = StoreConfig(search: searchConfig)

        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: res.embedder,
            config: config
        )

        let primaryID = Self.relevantIDs[0]
        try await store.add(
            id: primaryID,
            title: "Bartleby the Scrivener",
            body: Self.bartlebyTexts[0]
        )

        for i in 0..<34 {
            try await store.add(id: "noise-\(i)", body: Self.noiseDocument(index: i))
        }

        try await store.index()
        let hits = try await store.search(query: "bartleby", topK: 5).hits

        let p1 = precision(at: 1, hits: hits)
        try await store.shutdown()
        #expect(
            p1 == 1.0,
            """
            Top-1 not preserved at 35-doc scale with filter: precision@1 = \(p1). \
            Target document "\(primaryID)" should rank first. \
            Top-5 hits: \(hitsDescription(hits))
            """
        )
    }
}
#endif
