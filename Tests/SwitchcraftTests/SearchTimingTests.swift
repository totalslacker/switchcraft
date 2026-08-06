// Per-stage search timing instrumentation tests for issue #148.
//
// Covers requirement 7 of the issue:
//  - Each stage's duration is independently observable (a slow BM25 or
//    vector stage isn't attributed to the other).
//  - `cascadeCompactionOccurred` is true/false as expected.
//  - Best-effort partial timing survives on the `searchTimedOut` path.
//
// All injected delays are millisecond-scale (`DelayInjectingStorage` /
// `DelayedEmbedder`), and cascade tests use a tiny `l0Capacity` — no
// hardcoded seconds-scale sleeps anywhere in this file.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

@Suite("SearchTiming")
struct SearchTimingTests {

    // MARK: - Happy path: every stage is populated

    @Test("successful search populates every timing field")
    func successfulSearchPopulatesAllFields() async throws {
        let store = try await SwitchcraftStore(
            storage: InMemoryStorage(),
            embedder: MockEmbedder(dims: 8)
        )
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        // A generous explicit deadline avoids spurious searchTimedOut
        // under full-suite concurrent load, where scheduling contention
        // alone (unrelated to this test's own injected delays) can push
        // even cheap work past the 5s default (see SearchTimeoutTests'
        // same pattern for `deadline: .seconds(30)`).
        let result = try await store.search(query: "apples", topK: 5, deadline: .seconds(30))

        #expect(!result.hits.isEmpty)
        #expect(result.timing.flushDuration != nil)
        #expect(result.timing.embedDuration != nil)
        #expect(result.timing.vectorDuration != nil)
        #expect(result.timing.bm25Duration != nil)
        #expect(result.timing.mergeDuration != nil)
    }

    // MARK: - Independent observability

    @Test("a slow BM25 stage does not inflate vectorDuration")
    func slowBm25DoesNotInflateVector() async throws {
        let delayStorage = DelayInjectingStorage()
        let embedder = MockEmbedder(dims: 8)
        let store = try await SwitchcraftStore(storage: delayStorage, embedder: embedder)

        try await store.add(id: "doc-1", body: "apples oranges grapes")
        try await store.index()

        await delayStorage.setFullTextDelay(.milliseconds(100))

        // A generous explicit deadline avoids spurious searchTimedOut
        // under full-suite concurrent load, where scheduling contention
        // alone (unrelated to this test's own injected delays) can push
        // even cheap work past the 5s default (see SearchTimeoutTests'
        // same pattern for `deadline: .seconds(30)`).
        let result = try await store.search(query: "apples", topK: 5, deadline: .seconds(30))

        let bm25 = try #require(result.timing.bm25Duration)
        let vector = try #require(result.timing.vectorDuration)
        #expect(bm25 >= .milliseconds(80), "bm25Duration should reflect the injected delay")
        #expect(vector < .milliseconds(50), "vectorDuration must not absorb the FTS delay")
    }

    @Test("a slow vector stage does not inflate bm25Duration")
    func slowVectorDoesNotInflateBm25() async throws {
        let delayStorage = DelayInjectingStorage()
        let embedder = MockEmbedder(dims: 8)
        let store = try await SwitchcraftStore(storage: delayStorage, embedder: embedder)

        try await store.add(id: "doc-1", body: "apples oranges grapes")
        try await store.index()

        await delayStorage.setBucketDelay(.milliseconds(100))

        // A generous explicit deadline avoids spurious searchTimedOut
        // under full-suite concurrent load, where scheduling contention
        // alone (unrelated to this test's own injected delays) can push
        // even cheap work past the 5s default (see SearchTimeoutTests'
        // same pattern for `deadline: .seconds(30)`).
        let result = try await store.search(query: "apples", topK: 5, deadline: .seconds(30))

        let vector = try #require(result.timing.vectorDuration)
        let bm25 = try #require(result.timing.bm25Duration)
        #expect(vector >= .milliseconds(80), "vectorDuration should reflect the injected delay")
        #expect(bm25 < .milliseconds(50), "bm25Duration must not absorb the vector delay")
    }

    @Test("a slow embed stage does not inflate flushDuration or searchHybrid sub-stages")
    func slowEmbedDoesNotInflateOtherStages() async throws {
        let embedder = DelayedEmbedder(dims: 8, delay: .milliseconds(100))
        let store = try await SwitchcraftStore(
            storage: InMemoryStorage(),
            embedder: embedder
        )
        // Seed via a plain MockEmbedder-backed add would require a second
        // store; instead add through this store directly — encode() also
        // carries the injected delay, which only affects add()'s own
        // wall time, not this test's search() timing fields.
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        // A generous explicit deadline avoids spurious searchTimedOut
        // under full-suite concurrent load, where scheduling contention
        // alone (unrelated to this test's own injected delays) can push
        // even cheap work past the 5s default (see SearchTimeoutTests'
        // same pattern for `deadline: .seconds(30)`).
        let result = try await store.search(query: "apples", topK: 5, deadline: .seconds(30))

        let embed = try #require(result.timing.embedDuration)
        let flush = try #require(result.timing.flushDuration)
        #expect(embed >= .milliseconds(80), "embedDuration should reflect the injected delay")
        #expect(flush < .milliseconds(50), "flushDuration must not absorb the embed delay")
    }

    // MARK: - Cascade compaction detection

    @Test("cascadeCompactionOccurred is true when search's flush crosses l0Capacity")
    func cascadeDetectedThroughSearch() async throws {
        // cap(0) = 4 * 2 = 8. Three add+search rounds of 4 tokens each
        // cascade on round 3 (4 + 8 pending/L0 tokens > cap 8), mirroring
        // CompactionEventTests.cascadeTriggerReturnValue but exercised
        // through store.search() instead of indexer.flush() directly.
        let config = StoreConfig.testing(indexer: .testing(l0Capacity: 4, lsmFanout: 2))
        let store = try await SwitchcraftStore(
            storage: InMemoryStorage(),
            embedder: MockEmbedder(dims: 8),
            config: config
        )

        // Explicit generous deadlines avoid spurious searchTimedOut under
        // full-suite concurrent load (see the comment in the independent-
        // observability tests above for why).

        // Round 1: 4 tokens → L0 has 4 (below cap 8, manual).
        try await store.add(id: "doc-1", body: "aa bb cc dd")
        let r1 = try await store.search(query: "aa", topK: 5, deadline: .seconds(30))
        #expect(r1.timing.cascadeCompactionOccurred == false)
        #expect(r1.timing.cascadeCompactionDuration == nil)

        // Round 2: 4 more tokens → L0 has 8 (at cap 8, still manual).
        try await store.add(id: "doc-2", body: "ee ff gg hh")
        let r2 = try await store.search(query: "ee", topK: 5, deadline: .seconds(30))
        #expect(r2.timing.cascadeCompactionOccurred == false)
        #expect(r2.timing.cascadeCompactionDuration == nil)

        // Round 3: 4 more tokens → total 4 (pending) + 8 (L0) = 12 > cap 8 → cascade.
        try await store.add(id: "doc-3", body: "ii jj kk ll")
        let r3 = try await store.search(query: "ii", topK: 5, deadline: .seconds(30))
        #expect(r3.timing.cascadeCompactionOccurred == true)
        let cascadeDuration = try #require(r3.timing.cascadeCompactionDuration)
        #expect(cascadeDuration == r3.timing.flushDuration)
    }

    @Test("cascadeCompactionOccurred is false for an ordinary (non-cascading) search")
    func noCascadeForOrdinarySearch() async throws {
        let store = try await SwitchcraftStore(
            storage: InMemoryStorage(),
            embedder: MockEmbedder(dims: 8)
        )
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        // A generous explicit deadline avoids spurious searchTimedOut
        // under full-suite concurrent load, where scheduling contention
        // alone (unrelated to this test's own injected delays) can push
        // even cheap work past the 5s default (see SearchTimeoutTests'
        // same pattern for `deadline: .seconds(30)`).
        let result = try await store.search(query: "apples", topK: 5, deadline: .seconds(30))

        #expect(result.timing.cascadeCompactionOccurred == false)
        #expect(result.timing.cascadeCompactionDuration == nil)
    }

    // MARK: - Partial timing on timeout

    @Test("a forced searchTimedOut carries flushDuration but not later stages")
    func partialTimingOnTimeout() async throws {
        let store = try await SwitchcraftStore(
            storage: InMemoryStorage(),
            embedder: MockEmbedder(dims: 8)
        )
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        do {
            _ = try await store.search(query: "apples", topK: 5, deadline: .zero)
            Issue.record("Expected searchTimedOut but search completed normally")
        } catch let e as SwitchcraftStoreError {
            guard case .searchTimedOut(_, let partialTiming) = e else {
                Issue.record("Wrong SwitchcraftStoreError case: \(e)")
                return
            }
            #expect(partialTiming.flushDuration != nil)
            #expect(partialTiming.embedDuration == nil)
            #expect(partialTiming.vectorDuration == nil)
            #expect(partialTiming.bm25Duration == nil)
            #expect(partialTiming.mergeDuration == nil)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}
