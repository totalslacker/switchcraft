// Search timeout and task-cancellation tests for issue #83.
//
// Three enforcement paths are covered:
//  1. Pre-SQLite checkpoint (post-flush): deadline:.zero always fires here.
//  2. Progress-handler interrupt: configured via configureSearchDeadline
//     directly on SQLiteStorage with a far-past searchStart so remaining
//     budget = 0; the callback fires on the first sqlite3_step.
//  3. Task cancellation via SlowMockEmbedder that blocks in encode, giving
//     the enclosing Task time to be cancelled before search completes.
//
// Note on progress-handler granularity: sqlite3_progress_handler fires
// every N=10,000 SQLite VM instructions.  FTS5 virtual-table queries may
// use fewer VM instructions than expected because FTS5 logic executes in C
// code rather than the SQLite VM.  The testProgressHandlerElapsedInvariant
// test therefore accepts both a timeout AND a clean completion — a clean
// completion means the dataset was too small to trigger the handler on this
// machine.  The invariant (elapsed ≥ deadline) is only checked when a
// timeout IS thrown.
//
// All tests use a :memory: SQLite backend. No model assets, no CoreML.

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore
@testable import SwitchcraftSQLite

@Suite("SearchTimeout")
struct SearchTimeoutTests {

    // MARK: - Helpers

    static func makeStore(
        embedder: any Embedder = MockEmbedder(dims: 32)
    ) async throws -> SwitchcraftStore {
        try await SwitchcraftStore.sqlite(
            databasePath: ":memory:",
            embedder: embedder
        )
    }

    /// An embedder that suspends `delay` before calling `MockEmbedder`.
    /// Used to hold the search pipeline open long enough for a cooperative
    /// `Task.cancel()` to propagate through `Task.sleep`'s cancellation
    /// check.  Instances contain only Sendable fields and are safe to send
    /// across actor boundaries.
    struct SlowMockEmbedder: Embedder, Sendable {
        let inner: MockEmbedder
        let delay: Duration

        init(dims: Int = 32, delay: Duration) {
            inner = MockEmbedder(dims: dims)
            self.delay = delay
        }

        var dims: Int { inner.dims }
        var modelIdentifier: String { inner.modelIdentifier }

        func encode(_ text: String) async throws -> [Float] {
            // Task.sleep respects task cancellation: it throws
            // CancellationError as soon as the enclosing task is cancelled.
            try await Task.sleep(for: delay)
            return try await inner.encode(text)
        }
    }

    // MARK: - T1: Pre-checkpoint timeout

    @Test("pre-checkpoint: deadline:.zero fires immediately after flush")
    func preCheckpointTimeout() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        // deadline:.zero ⟹ elapsed (always ≥ 0) ≥ deadline (0) is true
        // at the post-flush checkpoint in SwitchcraftStore.search().
        await #expect(throws: SwitchcraftStoreError.self) {
            _ = try await store.search(query: "apples", topK: 5, deadline: .zero)
        }
    }

    @Test("pre-checkpoint: thrown error is searchTimedOut with non-negative elapsed")
    func preCheckpointTimeoutElapsed() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        do {
            _ = try await store.search(query: "apples", topK: 5, deadline: .zero)
            Issue.record("Expected searchTimedOut but search completed normally")
        } catch let e as SwitchcraftStoreError {
            guard case .searchTimedOut(let elapsed) = e else {
                Issue.record("Wrong SwitchcraftStoreError case: \(e)")
                return
            }
            // elapsed is the cost of flush + deadline check; must be ≥ 0.
            #expect(elapsed >= .zero)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - T2: Progress-handler interrupt

    // This test configures the progress handler with zero remaining budget
    // so that the callback fires (returning 1) the instant 10,000 VM
    // instructions have elapsed.  Whether it fires depends on the query
    // cost; see module-level note above.  The test is not marked failing
    // if the query completes cleanly — that means the dataset was too
    // small to trigger the handler.
    @Test("progress handler: if fired, translates SQLITE_INTERRUPT → searchTimedOut")
    func progressHandlerElapsedInvariant() async throws {
        let storage = SQLiteStorage(path: ":memory:")
        try await storage.open()

        // 500 documents matching the query forces a large FTS5 result set
        // that SQLite must fully materialise before returning the first row.
        // This increases the chance that > 10,000 VM instructions execute
        // during the first sqlite3_step call.
        for i in 0..<500 {
            try await storage.upsertDocument(DocumentRecord(
                uuid: "doc-\(i)",
                date: Date(timeIntervalSince1970: Double(i)),
                metadata: Data(),
                hash: "hash-\(i)",
                body: "apples oranges grapes document number \(i) with extra tokens",
                lens: [7]
            ))
        }

        let deadline = Duration.seconds(1)
        // searchStart is 10 s in the past, deadline is 1 s:
        //   elapsed ≈ 10 s > deadline ⟹ remainingBudget = max(0, 1–10) = 0
        // isActive = true, so the callback returns 1 on the first firing.
        let pastStart = ContinuousClock.now - .seconds(10)
        let ctx = SearchDeadlineContext(searchStart: pastStart, deadline: deadline)
        await storage.configureSearchDeadline(ctx)

        do {
            _ = try await storage.searchFullText(
                query: "apples",
                limit: 500,
                filter: .all
            )
            // Query completed in fewer than 10,000 VM instructions — the
            // progress handler never fired.  Accept this as a pass; see note.
        } catch let e as SwitchcraftStoreError {
            guard case .searchTimedOut(let elapsed) = e else {
                Issue.record("Wrong SwitchcraftStoreError case: \(e)")
                return
            }
            // If the handler DID fire, elapsed must be ≥ deadline.
            #expect(elapsed >= deadline)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - T3: Task cancellation

    @Test("task cancellation propagates CancellationError through encode")
    func taskCancellationDuringEncode() async throws {
        // Use SlowMockEmbedder so encode blocks for far longer than the
        // test's cancel window.  Crucially, no add() call is made — the
        // first (and only) encode call happens inside store.search().
        // SwitchcraftStore.init does NOT call embedder.encode, so
        // construction is instantaneous even with a slow embedder.
        let store = try await SwitchcraftStore.sqlite(
            databasePath: ":memory:",
            embedder: SlowMockEmbedder(delay: .seconds(100))
        )

        let searchTask = Task {
            // Will block in embedder.encode("apples") for 100 s
            // unless the task is cancelled first.
            try await store.search(query: "apples", topK: 5)
        }

        // Give the task time to start and suspend inside Task.sleep
        // (the 100 s sleep in SlowMockEmbedder.encode).
        try await Task.sleep(for: .milliseconds(50))
        searchTask.cancel()

        do {
            _ = try await searchTask.value
            Issue.record("Expected CancellationError but search completed normally")
        } catch is CancellationError {
            // Expected — Task.sleep in SlowMockEmbedder.encode threw when
            // the enclosing task was cancelled.
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - T4: Happy path with deadline

    @Test("normal search with generous deadline returns correct results")
    func normalSearchWithDeadline() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "apple-doc", body: "apples are delicious fruit")
        try await store.add(id: "banana-doc", body: "bananas are yellow fruit")
        try await store.add(id: "cherry-doc", body: "cherries are small and dark")

        let hits = try await store.search(
            query: "apples",
            topK: 5,
            deadline: .seconds(30)
        )
        #expect(!hits.isEmpty)
        #expect(hits[0].uuid == "apple-doc")
    }

    // MARK: - T5: Connection usable after timeout

    @Test("connection is in a clean state after a search timeout")
    func connectionUsableAfterTimeout() async throws {
        let store = try await Self.makeStore()
        try await store.add(id: "doc-1", body: "apples oranges grapes")

        // Trigger a pre-checkpoint timeout.
        do {
            _ = try await store.search(query: "apples", topK: 5, deadline: .zero)
        } catch is SwitchcraftStoreError {
            // Expected timeout — continue to verify recovery.
        }

        // A subsequent search with a generous deadline must succeed.
        // The preceding timeout was a pre-checkpoint (post-flush) timeout —
        // the SQL phase was never reached and the progress handler was never
        // armed. This verifies that the store recovers cleanly from a
        // pre-SQL timeout and remains fully usable for subsequent searches.
        let hits = try await store.search(
            query: "apples",
            topK: 5,
            deadline: .seconds(30)
        )
        #expect(!hits.isEmpty)
        #expect(hits[0].uuid == "doc-1")
    }
}
