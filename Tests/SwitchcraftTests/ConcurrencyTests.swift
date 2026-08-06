// Concurrency correctness suite for SwitchcraftStore + SearchEngine.
//
// Verifies higher-level invariants under genuine concurrent access:
// final state consistency, absence of Indexer.Error.ledgerOutOfSync
// under shared storage, and deterministic (bit-exact) results from
// concurrent searches over a static corpus.
//
// All scenarios use InMemoryStorage + MockEmbedder(dims: 32) and
// IndexerConfig.testing(l0Capacity: 4, lsmFanout: 2) to force cascades
// with few documents. No SQLite, no model assets, no time-based
// synchronisation (no sleep, no DispatchSemaphore) — only Swift
// Concurrency primitives.
//
// NOTE: bodies are made unique per task (loop index in body) so chunk
// hashes differ across concurrent adds. Identical bodies across
// concurrent add() calls would race the chunk pre-check in
// SwitchcraftStore.add and could double-buffer embeddings under one
// chunkID, eventually throwing Indexer.Error.ledgerOutOfSync.
//
// Run 100× to check for flakiness:
//   for i in $(seq 1 100); do swift test --filter ConcurrencyTests || { echo "FAILED on run $i"; exit 1; }; done

import Foundation
import Testing
@testable import Switchcraft
@testable import SwitchcraftCore

@Suite("ConcurrencyTests")
struct ConcurrencyTests {

    // MARK: - Helpers

    /// Build a body that embeds the loop index so each concurrent add
    /// has a unique content hash. The shared keyword guarantees the
    /// post-run search returns hits.
    private static func body(_ i: Int) -> String {
        "document number \(i) about apples and oranges and grapes"
    }

    private static let sharedKeyword = "apples"

    private static func makeStore(
        storage: InMemoryStorage
    ) async throws -> SwitchcraftStore {
        try await SwitchcraftStore(
            storage: storage,
            embedder: MockEmbedder(dims: 32),
            config: .testing(indexer: .testing(l0Capacity: 4, lsmFanout: 2))
        )
    }

    // MARK: - R1

    @Test("R1: concurrent add + search on a single store reaches consistent final state")
    func concurrentReadWriteSingleStore() async throws {
        let storage = InMemoryStorage()
        let store = try await Self.makeStore(storage: storage)

        let addCount = 20
        let searchCount = 10

        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<addCount {
                group.addTask {
                    try await store.add(id: "doc-\(i)", body: Self.body(i))
                }
            }
            for _ in 0..<searchCount {
                group.addTask {
                    _ = try await store.search(query: Self.sharedKeyword, topK: 5).hits
                }
            }
            try await group.waitForAll()
        }

        let count = try await storage.documentCount()
        #expect(count == addCount)

        let finalHits = try await store.search(query: Self.sharedKeyword, topK: 10).hits
        #expect(!finalHits.isEmpty)

        try await store.shutdown()
    }

    // MARK: - R2

    @Test("R2: reader (storeB) does not block writer (storeA) over shared InMemoryStorage")
    func readerDoesNotBlockWriterTwoStores() async throws {
        let storage = InMemoryStorage()
        let storeA = try await Self.makeStore(storage: storage)
        let storeB = try await Self.makeStore(storage: storage)

        let addCount = 20
        let readerCount = 10

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Writer: add 20 docs then index().
            group.addTask {
                for i in 0..<addCount {
                    try await storeA.add(id: "doc-\(i)", body: Self.body(i))
                }
                try await storeA.index()
            }
            // Concurrent readers on storeB.
            for _ in 0..<readerCount {
                group.addTask {
                    _ = try await storeB.search(query: Self.sharedKeyword, topK: 5).hits
                }
            }
            try await group.waitForAll()
        }

        // Final reads should see the fully-flushed corpus.
        let finalHits = try await storeB.search(query: Self.sharedKeyword, topK: 10).hits
        #expect(!finalHits.isEmpty)

        let count = try await storage.documentCount()
        #expect(count == addCount)

        try await storeA.shutdown()
        try await storeB.shutdown()
    }

    // MARK: - R3

    @Test("R3: interleaved add + search under cascading flush completes without errors")
    func noErrorsInterleavedAddSearchUnderLargeFlush() async throws {
        let storage = InMemoryStorage()
        let store = try await Self.makeStore(storage: storage)

        let addCount = 30
        let searchCount = 15

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Launch order: 5 adds first, then alternate add/search for
            // the remaining 25 adds and 15 searches. TaskGroup execution
            // order is scheduler-controlled; this is a launch-order
            // interleave only (no time-based barrier).
            for i in 0..<5 {
                group.addTask {
                    try await store.add(id: "doc-\(i)", body: Self.body(i))
                }
            }
            var addIdx = 5
            var searchIdx = 0
            while addIdx < addCount || searchIdx < searchCount {
                if addIdx < addCount {
                    let i = addIdx
                    group.addTask {
                        try await store.add(id: "doc-\(i)", body: Self.body(i))
                    }
                    addIdx += 1
                }
                if searchIdx < searchCount {
                    group.addTask {
                        _ = try await store.search(query: Self.sharedKeyword, topK: 5).hits
                    }
                    searchIdx += 1
                }
            }
            try await group.waitForAll()
        }

        let count = try await storage.documentCount()
        #expect(count == addCount)

        let finalHits = try await store.search(query: Self.sharedKeyword, topK: 10).hits
        #expect(!finalHits.isEmpty)

        try await store.shutdown()
    }

    // MARK: - R4

    @Test("R4: N concurrent searches across two stores produce bit-identical results")
    func bitExactFloatEqualityAcrossConcurrentSearches() async throws {
        let storage = InMemoryStorage()
        let storeA = try await Self.makeStore(storage: storage)
        let storeB = try await Self.makeStore(storage: storage)

        let docCount = 10
        for i in 0..<docCount {
            try await storeA.add(id: "doc-\(i)", body: Self.body(i))
        }
        try await storeA.index()

        let searchCount = 10  // 5 from each store
        let query = "apples and oranges"

        var results: [[HybridHit]] = try await withThrowingTaskGroup(
            of: [HybridHit].self
        ) { group in
            for _ in 0..<(searchCount / 2) {
                group.addTask {
                    try await storeA.search(query: query, topK: 10).hits
                }
            }
            for _ in 0..<(searchCount / 2) {
                group.addTask {
                    try await storeB.search(query: query, topK: 10).hits
                }
            }
            var collected: [[HybridHit]] = []
            collected.reserveCapacity(searchCount)
            for try await hits in group {
                collected.append(hits)
            }
            return collected
        }

        #expect(results.count == searchCount)
        #expect(!results[0].isEmpty)

        // Pairwise equality (Hashable/Equatable HybridHit).
        let reference = results.removeFirst()
        for hits in results {
            #expect(hits == reference)
        }

        try await storeA.shutdown()
        try await storeB.shutdown()
    }
}
