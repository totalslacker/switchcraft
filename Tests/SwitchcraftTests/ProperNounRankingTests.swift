import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

#if canImport(CoreML)
import CoreML
@testable import SwitchcraftCoreML

/// Regression test for issue #105: XTR vector pipeline mis-ranks proper-noun
/// queries when a document's title is not prepended at embedding time.
///
/// **Pathology (testPathologyExists)**: Without title prepending, a document
/// titled "Bartleby, the Scrivener" (Melville story excerpt, zero occurrences
/// of "alfred") outranks a document titled "Alfred - Productivity App for
/// macOS" (body dense with "Alfred") in the XTR vector pipeline for query
/// "alfred". This is caused by embedding-space proximity (H1) and/or the
/// Alfred document's tokens missing the top-k centroid scan window (H2).
///
/// **Fix (testTitlePrependFixesRanking)**: With title prepending (ADR 025),
/// the Alfred document's title tokens appear at position 0 of the T5 encoder
/// context — the position closest to the minimal-context shape of a standalone
/// query token — and land in the top-k centroid scan. Alfred outranks Bartleby.
///
/// Both tests use an in-memory SQLite store and share a single T5CoreMLEmbedder
/// instance to amortise the model load (~1–3 s on ANE). They are asset-gated:
/// set `SWITCHCRAFT_XTR_MLPACKAGE=<path>` to enable.
@Suite(
    "Proper Noun Ranking (asset-gated)",
    .enabled(if: CoreMLAsset.isAvailable)
)
struct ProperNounRankingTests {

    // MARK: - Synthetic corpus

    /// Document ID for the Alfred app document.
    private static let alfredID = "alfred-app"
    /// Document ID for the Bartleby document.
    private static let bartlebyID = "bartleby"

    /// Title for the Alfred document (used in the fix test only).
    private static let alfredTitle = "Alfred - Productivity App for macOS"

    /// Body for the Alfred document — intentionally avoids using the word
    /// "alfred" so that without title prepend the document scores near-baseline
    /// for the query "alfred", but with the title prepended (ADR 025) the title
    /// token "Alfred" drives the score up. This is exactly the H3 failure mode.
    private static let alfredBody = """
        This productivity application for macOS boosts your efficiency through \
        hotkeys, keywords, and text expansion. The powerpack unlocks powerful \
        workflows that automate repetitive tasks. It integrates with macOS \
        Spotlight and adds features like a clipboard manager, snippets, and \
        password manager integration. This tool has been the go-to choice for \
        Mac power users for over a decade. Download it free and discover why \
        millions of Mac users rely on it daily. Custom workflows let you \
        control the launcher with triggers and actions. The companion mobile app \
        brings the experience to your phone so you can trigger workflows from \
        anywhere. The application runs natively on macOS entirely on-device.
        """

    /// Title for the Bartleby document (used in the fix test only).
    private static let bartlebyTitle = "Bartleby, the Scrivener"

    /// Body for the Bartleby document — opening of Melville's story. Contains
    /// zero occurrences of "alfred". The T5 embedding-space proximity between
    /// certain literary proper-noun tokens and the "alfred" query token is the
    /// H1 pathology this corpus is designed to reproduce.
    private static let bartlebyBody = """
        I am a rather elderly man. The nature of my avocations, for the last \
        thirty years, has brought me into more than ordinary contact with what \
        would seem an interesting and somewhat singular set of men, of whom, as \
        yet, nothing that I know of has ever been written — I mean the \
        law-copyists, or scriveners. I have known very many of them, \
        professionally and privately, and, if I pleased, could relate divers \
        histories at which good-natured gentlemen might smile and sentimental \
        souls might weep. But I waive the biographies of all other scriveners, \
        for a few passages in the life of Bartleby, who was a scrivener of the \
        strangest I ever saw, or heard of. While, of other law-copyists, I \
        might write the complete life, of Bartleby nothing of that sort can be \
        done. I believe that no materials exist for a full and satisfactory \
        biography of this man. It is an irreparable loss to literature. Bartleby \
        was one of those beings of whom nothing is ascertainable except from the \
        original sources, and, in his case, those are very small. What my own \
        astonished eyes saw of Bartleby, that is all I know of him, except, \
        indeed, one vague report, which will appear in the sequel.
        """

    // MARK: - Sparse-title corpus

    /// Document ID for the sparse-title Bartleby document.
    private static let sparseBartlebyID = "sparse-bartleby"

    /// Title for the sparse-title document — a single word so `title.count < 20`
    /// triggers the constructive fallback in ADR 025 §(h).
    private static let sparseBartlebyTitle = "Bartleby"

    /// Body for the sparse-title document. Simulates a homework-help landing page
    /// whose title is just "Bartleby" while the body references the name ~55 times.
    // "Bartleby" appears approximately 55 times in this body.
    private static let sparseBartlebyBody = """
        Bartleby is the go-to resource for students studying classic American \
        literature. Need help understanding Bartleby the Scrivener? Bartleby has \
        comprehensive summaries, character analysis, and critical essays about Bartleby.

        Search Bartleby for any passage from Bartleby the Scrivener and find instant \
        explanations. Bartleby makes it easy to understand why Bartleby says \
        "I would prefer not to", what Bartleby symbolises in a nineteenth-century \
        office setting, and how Bartleby fits into American short fiction. Bartleby \
        provides free access to the full text of Bartleby the Scrivener alongside \
        Bartleby study guides written by expert teachers.

        Students use Bartleby to write better essays about Bartleby. Whether your \
        assignment covers Bartleby's passive resistance, Bartleby's relationship with \
        the narrator, or the ending of Bartleby in the Tombs, Bartleby has a study \
        guide for every aspect of the story. Bartleby also provides citation help so \
        you can quote Bartleby correctly in MLA, APA, or Chicago format.

        Bartleby covers all the major themes in Bartleby the Scrivener: alienation, \
        dehumanisation, and the silent rebellion of Bartleby against the demands of \
        commerce. Bartleby situates the character Bartleby within Melville's broader \
        work and in the context of nineteenth-century America. Bartleby helps you \
        compare Bartleby with characters from other Melville stories.

        Millions of students visit Bartleby every semester. Whether you are reading \
        Bartleby for the first time or writing a doctoral thesis about Bartleby, \
        Bartleby provides the resources you need. Bartleby is the most trusted \
        academic resource for understanding Bartleby the Scrivener and the world \
        Bartleby inhabited.

        Why is the site called Bartleby? Because Bartleby the Scrivener represents \
        the spirit of quiet, determined inquiry. Just as Bartleby preferred not to \
        do what others demanded, Bartleby the site gives you freedom to explore \
        Bartleby on your own terms. Find Bartleby study guides, Bartleby essay \
        prompts, and Bartleby quizzes — all free on Bartleby.
        """

    /// Ten distinct filler paragraphs about topics unrelated to "bartleby" or
    /// "alfred". Cycled across the 52 filler documents.
    private static let fillerCorpus: [String] = [
        "The Amazon River is the largest river in the world by discharge volume. It flows through South America, primarily through Brazil, Peru, and Colombia. The Amazon basin contains the world's largest tropical rainforest, home to an extraordinary diversity of plant and animal species.",
        "Pasta carbonara is a classic Roman dish made with eggs, hard cheese, cured pork, and black pepper. The sauce is created by combining beaten eggs and finely grated Pecorino Romano with cooked guanciale. The heat from the pasta cooks the eggs into a creamy sauce without scrambling them.",
        "The FIFA World Cup is the most prestigious international football tournament in the world. It is held every four years and involves national teams competing from every continent. Brazil holds the record for the most World Cup victories, having won the tournament five times.",
        "The Great Wall of China is one of the most impressive architectural achievements in human history. Construction began as early as the seventh century BC and continued for centuries under various dynasties. The wall stretches thousands of kilometres across northern China, crossing mountains, deserts, and plains.",
        "Photosynthesis is the process by which plants, algae, and certain bacteria convert sunlight into chemical energy stored as glucose. The process occurs in chloroplasts and requires carbon dioxide and water as raw materials. Oxygen is released as a byproduct, making photosynthesis essential to life on Earth.",
        "Mount Everest is the highest mountain on Earth, with a summit elevation of 8,848 metres above sea level. It is located in the Himalayas on the border between Nepal and Tibet. The first confirmed ascent was made on 29 May 1953 by Edmund Hillary and Tenzing Norgay.",
        "Traditional French baguettes are made from simple ingredients: flour, water, yeast, and salt. The dough undergoes a long fermentation process to develop flavour and texture. Baguettes are characterised by their long, thin shape and crispy golden crust with a soft, chewy interior.",
        "Wimbledon is the oldest tennis tournament in the world, held annually at the All England Club in London. Players compete on grass courts, which favour a fast, low-bouncing game style. The tournament is renowned for its strict dress code requiring all players to wear white.",
        "The Roman Empire at its height controlled territories stretching from Britain to Egypt and from Spain to Mesopotamia. Roman engineering achievements such as aqueducts, roads, and amphitheatres remain visible across Europe and North Africa today.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. They form when massive stars collapse at the end of their life cycle. The first direct image of a black hole was captured in 2019 by the Event Horizon Telescope.",
    ]

    /// 52 filler documents generated by cycling through `fillerCorpus`.
    private static var fillerDocuments: [(id: String, body: String)] {
        (0..<52).map { i in ("filler-\(i)", fillerCorpus[i % fillerCorpus.count]) }
    }

    // MARK: - SharedResources

    /// One MLModel load shared across both tests.
    actor SharedResources {
        static let shared = SharedResources()

        private var built: Built?

        struct Built {
            let embedder: T5CoreMLEmbedder
            let tokenizer: Tokenizer
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
            let embedder = try await T5CoreMLEmbedder(
                modelURL: modelURL,
                tokenizer: tokenizer
            )
            let result = Built(embedder: embedder, tokenizer: tokenizer)
            built = result
            return result
        }
    }

    // MARK: - Helpers

    /// Build a fresh in-memory store, index the two synthetic documents
    /// (optionally with titles), flush, and return the search results for
    /// query "alfred".
    private func rankingsForAlfred(useTitles: Bool) async throws -> [HybridHit] {
        let res = try await SharedResources.shared.get()
        let storage = SQLiteStorage(path: ":memory:")
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: res.embedder,
            config: .default
        )

        if useTitles {
            try await store.add(
                id: Self.alfredID,
                title: Self.alfredTitle,
                body: Self.alfredBody
            )
            try await store.add(
                id: Self.bartlebyID,
                title: Self.bartlebyTitle,
                body: Self.bartlebyBody
            )
        } else {
            try await store.add(id: Self.alfredID, body: Self.alfredBody)
            try await store.add(id: Self.bartlebyID, body: Self.bartlebyBody)
        }

        try await store.index()
        return try await store.search(query: "alfred", topK: 10)
    }

    /// Build a fresh in-memory store, index the sparse-Bartleby document
    /// (title "Bartleby", 8 chars) and 52 unrelated filler documents, flush,
    /// and return search results for the given query.
    private func rankingsForSparseTitleQuery(_ query: String) async throws -> [HybridHit] {
        let res = try await SharedResources.shared.get()
        let storage = SQLiteStorage(path: ":memory:")
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: res.embedder,
            config: .default
        )

        try await store.add(
            id: Self.sparseBartlebyID,
            title: Self.sparseBartlebyTitle,
            body: Self.sparseBartlebyBody
        )
        for doc in Self.fillerDocuments {
            try await store.add(id: doc.id, body: doc.body)
        }

        try await store.index()
        return try await store.search(query: query, topK: 10)
    }

    // MARK: - Tests

    /// Verifies the pathology: without title prepending, the Bartleby document
    /// outranks the Alfred document in the XTR vector pipeline for query "alfred".
    ///
    /// If this test fails (Alfred already outranks Bartleby without title), the
    /// synthetic corpus does not reproduce the H1/H2 bug with this T5 model
    /// checkpoint. Adjust the Bartleby body content and re-run.
    @Test("Pathology: without title prepend Bartleby outranks Alfred in vector pipeline")
    func pathologyExists() async throws {
        let hits = try await rankingsForAlfred(useTitles: false)

        guard
            let alfredHit = hits.first(where: { $0.uuid == Self.alfredID }),
            let bartlebyHit = hits.first(where: { $0.uuid == Self.bartlebyID })
        else {
            let desc = hits.map { h in "\(h.uuid)=vr\(h.vectorRank.map { String($0) } ?? "nil")" }
                .joined(separator: ", ")
            Issue.record("One or both synthetic documents missing from search results — corpus may be too small to produce a merged index generation. hits: \(desc)")
            return
        }

        let alfredVR = alfredHit.vectorRank ?? Int.max
        let bartlebyVR = bartlebyHit.vectorRank ?? Int.max

        // If this assertion fails, the synthetic corpus does not reproduce the
        // bug; the T5 model may already rank Alfred correctly without a title.
        // Adjust the Bartleby body (e.g. use a longer Melville excerpt) so its
        // tokens are closer to "alfred" in T5 embedding space, then re-run.
        #expect(
            bartlebyVR < alfredVR,
            """
            Pathology not reproduced: expected Bartleby (vr=\(bartlebyVR)) to \
            outrank Alfred (vr=\(alfredVR)) without title prepend, but Alfred \
            already ranks higher. Adjust the Bartleby corpus text so the bug is \
            reproducible, then re-run. Scores: alfred=\(alfredHit.vectorScore.map { String($0) } ?? "nil"), \
            bartleby=\(bartlebyHit.vectorScore.map { String($0) } ?? "nil").
            """
        )
    }

    /// Verifies the fix: with title prepending (ADR 025), the Alfred document
    /// outranks the Bartleby document in the XTR vector pipeline for query "alfred".
    ///
    /// This is the primary regression gate for R2. If this test fails even
    /// after pathologyExists() passes, H1 (T5 embedding-space proximity) may
    /// dominate and cannot be fixed without model retraining — escalate to the
    /// issue tracker.
    @Test("Fix: with title prepend Alfred outranks Bartleby in vector pipeline")
    func titlePrependFixesRanking() async throws {
        let hits = try await rankingsForAlfred(useTitles: true)

        guard
            let alfredHit = hits.first(where: { $0.uuid == Self.alfredID }),
            let bartlebyHit = hits.first(where: { $0.uuid == Self.bartlebyID })
        else {
            let desc = hits.map { h in "\(h.uuid)=vr\(h.vectorRank.map { String($0) } ?? "nil")" }
                .joined(separator: ", ")
            Issue.record("One or both synthetic documents missing from search results. hits: \(desc)")
            return
        }

        let alfredVR = alfredHit.vectorRank ?? Int.max
        let bartlebyVR = bartlebyHit.vectorRank ?? Int.max

        #expect(
            alfredVR < bartlebyVR,
            """
            Title prepend did not fix ranking: Alfred (vr=\(alfredVR)) should \
            rank above Bartleby (vr=\(bartlebyVR)) after title prepend. \
            Scores: alfred=\(alfredHit.vectorScore.map { String($0) } ?? "nil"), \
            bartleby=\(bartlebyHit.vectorScore.map { String($0) } ?? "nil"). \
            If pathologyExists() also fails, the corpus doesn't reproduce the bug. \
            If pathologyExists() passes but this fails, H1 dominates — escalate.
            """
        )
    }

    /// Verifies R1: a document with a sparse (single-word) title and a body
    /// containing the query term 50+ times ranks in the top 5 by vector rank
    /// against a corpus of 52 unrelated filler documents.
    ///
    /// This is the primary regression gate for the constructive title fallback
    /// introduced in ADR 025 §(h) / issue #118. The test should fail before
    /// the fix is applied and pass after.
    @Test("Fix: sparse-title document ranks in top 5 for body-match query")
    func sparseTitleRanksHighForBodyQuery() async throws {
        let hits = try await rankingsForSparseTitleQuery("bartleby")

        guard let hit = hits.first(where: { $0.uuid == Self.sparseBartlebyID }) else {
            let desc = hits.map { h in "\(h.uuid)=vr\(h.vectorRank.map { String($0) } ?? "nil")" }
                .joined(separator: ", ")
            Issue.record(
                "sparse-bartleby document missing from search results. hits: \(desc)"
            )
            return
        }

        let vr = hit.vectorRank ?? Int.max
        #expect(
            vr <= 5,
            """
            Sparse-title document ranked vr=\(vr) for query "bartleby"; expected ≤5. \
            Score: \(hit.vectorScore.map { String($0) } ?? "nil"). \
            If this fails before the ADR 025 §(h) fix is applied, the pathology is \
            being correctly reproduced. Apply the fix and re-run.
            """
        )
    }

    /// Cross-regression guard: adding a sparse-title document to the same index
    /// as a rich-title document must not hurt either group's ranking.
    ///
    /// Verifies:
    /// - Alfred (rich title, 35 chars, `title.count ≥ 20`) ranks ≤5 for "alfred".
    /// - Sparse-Bartleby (sparse title, 8 chars, `title.count < 20`) ranks ≤5
    ///   for "bartleby", in the same index.
    @Test("Fix: sparse-title and rich-title documents coexist without cross-regression")
    func sparseTitleAndRichTitleCoexist() async throws {
        let res = try await SharedResources.shared.get()
        let storage = SQLiteStorage(path: ":memory:")
        let store = try await SwitchcraftStore(
            storage: storage,
            embedder: res.embedder,
            config: .default
        )

        try await store.add(
            id: Self.alfredID,
            title: Self.alfredTitle,
            body: Self.alfredBody
        )
        try await store.add(
            id: Self.sparseBartlebyID,
            title: Self.sparseBartlebyTitle,
            body: Self.sparseBartlebyBody
        )
        for doc in Self.fillerDocuments {
            try await store.add(id: doc.id, body: doc.body)
        }
        try await store.index()

        // Rich-title document (Alfred) must still rank top 5 for "alfred".
        let alfredHits = try await store.search(query: "alfred", topK: 10)
        if let alfredHit = alfredHits.first(where: { $0.uuid == Self.alfredID }) {
            let vr = alfredHit.vectorRank ?? Int.max
            #expect(
                vr <= 5,
                "Alfred (rich title) ranked vr=\(vr) for \"alfred\"; expected ≤5. Score: \(alfredHit.vectorScore.map { String($0) } ?? "nil")."
            )
        } else {
            Issue.record("Alfred document missing from \"alfred\" search results.")
        }

        // Sparse-title document (sparse-bartleby) must also rank top 5 for "bartleby".
        let bartlebyHits = try await store.search(query: "bartleby", topK: 10)
        if let bartlebyHit = bartlebyHits.first(where: { $0.uuid == Self.sparseBartlebyID }) {
            let vr = bartlebyHit.vectorRank ?? Int.max
            #expect(
                vr <= 5,
                "Sparse-Bartleby (sparse title) ranked vr=\(vr) for \"bartleby\"; expected ≤5. Score: \(bartlebyHit.vectorScore.map { String($0) } ?? "nil")."
            )
        } else {
            Issue.record("Sparse-bartleby document missing from \"bartleby\" search results.")
        }
    }
}
#endif
