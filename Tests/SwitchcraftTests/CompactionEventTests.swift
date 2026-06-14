// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import SwitchcraftCore
@testable import Switchcraft

// MARK: - Thread-safe event collector for tests

private actor EventCollector {
    var events: [CompactionEvent] = []
    func append(_ event: CompactionEvent) { events.append(event) }
    func all() -> [CompactionEvent] { events }
}

// MARK: - Tests

@Suite("CompactionEvent")
struct CompactionEventTests {

    // MARK: - Helpers

    /// Add `count` single-token chunks to `indexer`, returning the next available chunkID.
    static func addChunks(
        to indexer: Indexer,
        count: Int,
        startID: Int64 = 1,
        dims: Int,
        rng: inout SplitMix64
    ) async throws -> Int64 {
        var nextID = startID
        for _ in 0..<count {
            var flat = [Float](repeating: 0, count: dims)
            for j in 0..<dims { flat[j] = Float(rng.next()) / Float(UInt64.max) }
            let norm = sqrt(flat.reduce(0) { $0 + $1 * $1 })
            if norm > 0 { for j in 0..<dims { flat[j] /= norm } }
            try await indexer.add(chunkID: nextID, embeddings: flat, dims: dims)
            nextID += 1
        }
        return nextID
    }

    // MARK: - Indexer.flush() return-value tests

    @Test("manual trigger: flush with pending embeddings returns CompactionEvent with trigger == .manual")
    func manualTriggerReturnValue() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        // l0Capacity=8 → cap(0) = 16; 4 single-token chunks will not cascade.
        let config = IndexerConfig.testing(l0Capacity: 8, lsmFanout: 2)
        let indexer = try await Indexer(storage: storage, config: config)
        var rng = SplitMix64(seed: 42)
        let dims = 8
        let tokenCount = 4

        _ = try await Self.addChunks(to: indexer, count: tokenCount, dims: dims, rng: &rng)
        let event = try await indexer.flush()

        let e = try #require(event, "expected non-nil CompactionEvent for non-empty flush")
        #expect(e.trigger == .manual)
        #expect(e.inputSegments == 1, "no prior gens — pending buffer is the only input segment")
        #expect(e.inputBytes == UInt64(tokenCount * dims * 4))
        #expect(abs(e.startedAt.timeIntervalSinceNow) < 10)
    }

    @Test("cascade trigger: flush that promotes past L0 capacity returns CompactionEvent with trigger == .cascade")
    func cascadeTriggerReturnValue() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        // cap(0) = 4 * 2 = 8. Three flushes of 4 tokens trigger cascade on flush 3.
        let config = IndexerConfig.testing(l0Capacity: 4, lsmFanout: 2)
        let indexer = try await Indexer(storage: storage, config: config)
        var rng = SplitMix64(seed: 99)
        let dims = 8

        // Flush 1: 4 tokens → L0 with 4 tokens (below cap 8, manual).
        _ = try await Self.addChunks(to: indexer, count: 4, startID: 1, dims: dims, rng: &rng)
        _ = try await indexer.flush()

        // Flush 2: 4 more tokens → L0 with 8 tokens (at cap 8, still manual).
        _ = try await Self.addChunks(to: indexer, count: 4, startID: 5, dims: dims, rng: &rng)
        _ = try await indexer.flush()

        // Flush 3: 4 more tokens → total=4+8=12 > cap(0)=8 → cascade to L1.
        _ = try await Self.addChunks(to: indexer, count: 4, startID: 9, dims: dims, rng: &rng)
        let event = try await indexer.flush()

        let e = try #require(event, "expected non-nil CompactionEvent for cascade flush")
        #expect(e.trigger == .cascade)
        // Merged gens: 1 (the existing L0 gen) + pending buffer = inputSegments 2.
        #expect(e.inputSegments == 2)
        // total = 4 (pending) + 8 (L0) = 12 tokens.
        #expect(e.inputBytes == UInt64(12 * dims * 4))
    }

    @Test("no-op flush: flushing an empty pending buffer returns nil")
    func noOpFlushReturnsNil() async throws {
        let storage = InMemoryStorage()
        try await storage.open()
        let config = IndexerConfig.testing()
        let indexer = try await Indexer(storage: storage, config: config)

        // No indexer.add() calls — buffer is empty.
        let event = try await indexer.flush()
        #expect(event == nil, "flush on empty pending buffer should return nil")
    }

    // MARK: - SwitchcraftStore.onCompactionEvent tests

    @Test("SwitchcraftStore: callback fires once per real flush with correct fields")
    func storeCallbackFires() async throws {
        let storage = InMemoryStorage()
        let embedder = MockEmbedder(dims: 8)
        // l0Capacity=8 → cap(0)=16; a single doc with a few tokens won't cascade.
        let config = StoreConfig(indexer: .testing(l0Capacity: 8, lsmFanout: 2))
        let store = try await SwitchcraftStore(
            storage: storage, embedder: embedder, config: config
        )

        let collector = EventCollector()
        await store.setOnCompactionEvent { event in
            await collector.append(event)
        }

        // "one two three" → 3 tokens via MockEmbedder.
        try await store.add(id: "doc-1", body: "one two three")
        try await store.index()

        let received = await collector.all()
        #expect(received.count == 1, "callback should fire exactly once per flush")
        let e = try #require(received.first)
        #expect(e.trigger == .manual)
        #expect(e.inputSegments >= 1)
        #expect(e.inputBytes > 0)
    }

    @Test("SwitchcraftStore: callback does not fire for no-op index() call")
    func storeCallbackSilentOnNoOp() async throws {
        let storage = InMemoryStorage()
        let embedder = MockEmbedder(dims: 8)
        let store = try await SwitchcraftStore(
            storage: storage, embedder: embedder
        )

        let collector = EventCollector()
        await store.setOnCompactionEvent { event in
            await collector.append(event)
        }

        // No documents added — index() is a no-op flush.
        try await store.index()

        let received = await collector.all()
        #expect(received.isEmpty, "callback must not fire for no-op flush")
    }
}
