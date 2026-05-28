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

    /// Body for the Alfred document — dense with "Alfred" so BM25 also
    /// ranks it highly and the vector-pipeline behaviour is the discriminator.
    private static let alfredBody = """
        Alfred is a productivity application for macOS that boosts your \
        efficiency with hotkeys, keywords, and text expansion. Alfred's \
        powerpack unlocks powerful workflows that let you automate repetitive \
        tasks. Alfred integrates with macOS Spotlight and adds Alfred-specific \
        features like a clipboard manager, snippets, and 1Password integration. \
        Alfred has been the go-to productivity tool for Mac power users for over \
        a decade. Download Alfred free from alfredapp.com and discover why \
        Alfred is loved by millions of Mac users worldwide. Alfred workflows let \
        you control Alfred with custom triggers and actions. Alfred Remote brings \
        Alfred to your iPhone and iPad so you can trigger Alfred workflows from \
        anywhere. Alfred is built natively for macOS and runs entirely on-device.
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
}
#endif
