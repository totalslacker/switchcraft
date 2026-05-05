// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import SwitchcraftCore
import SwitchcraftSQLite

// MARK: - Test Suite

@Suite("SQLiteStorage WAL Concurrency")
struct SQLiteStorageConcurrencyTests {

    // Corpus size for slow reads. Target: ≥100ms FTS scan on Apple silicon.
    // Increase to 10_000 if CI runners produce sub-100ms reads.
    private static let seedCount: Int = 5_000

    // MARK: - Helpers

    private static func makeTemporaryDB() async throws -> (SQLiteStorage, URL) {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("sc-concurrency-\(UUID().uuidString).sqlite3")
        let storage = SQLiteStorage(path: url.path)
        try await storage.open()
        for i in 0..<seedCount {
            try await storage.upsertDocument(DocumentRecord(
                uuid: "doc-\(i)",
                date: Date(timeIntervalSince1970: Double(i)),
                hash: "seed-hash-\(i)",
                body: "the quick brown fox jumps over the lazy dog " +
                      "document number \(i) with additional filler text so " +
                      "the full text search takes longer when scanning the corpus"
            ))
        }
        return (storage, url)
    }

    private static func cleanupDB(url: URL) {
        let fm = FileManager.default
        try? fm.removeItem(at: url)
        try? fm.removeItem(atPath: url.path + "-wal")
        try? fm.removeItem(atPath: url.path + "-shm")
    }

    private static func makeWriteDoc(uuid: String) -> DocumentRecord {
        DocumentRecord(uuid: uuid, date: Date(), hash: "wh-\(uuid)", body: "write doc \(uuid)")
    }

    /// Returns the median wall time (seconds) of `iterations` invocations.
    private static func measureMedian(
        iterations: Int = 3,
        _ work: () async throws -> Void
    ) async throws -> Double {
        var times: [Double] = []
        for _ in 0..<iterations {
            let start = Date()
            try await work()
            times.append(Date().timeIntervalSince(start))
        }
        let sorted = times.sorted()
        return sorted[sorted.count / 2]
    }

    // MARK: - Test [1]: Liveness

    /// Verifies that a write completes while a slow FTS read is still in flight.
    ///
    /// Under the old single-actor design the write queues behind the read and
    /// finishes only after the read completes, so `writeEnd ≥ readEnd`.
    /// With the writer/reader split, the write completes concurrently so
    /// `writeEnd < readEnd`.
    @Test("Write completes while slow FTS read is in flight (WAL liveness)")
    func livenessTest() async throws {
        let (storage, url) = try await Self.makeTemporaryDB()
        defer {
            Self.cleanupDB(url: url)
            Task { try? await storage.close() }
        }

        // Start the slow FTS scan on the reader actor.
        async let slowRead: [FullTextHit] = storage.searchFullText(
            query: "the",
            limit: Self.seedCount,
            filter: .all
        )

        // Give the cooperative scheduler a chance to begin the scan.
        for _ in 0..<8 { await Task.yield() }

        // Issue a write on the writer actor and record when it finishes.
        try await storage.upsertDocument(Self.makeWriteDoc(uuid: "liveness-write"))
        let writeEnd = Date()

        // Wait for the read to finish and record its completion time.
        _ = try await slowRead
        let readEnd = Date()

        // The write must finish before the read completes: writer and reader
        // run on separate actors/connections. Under the old single-actor design
        // writeEnd would be ≥ readEnd.
        #expect(writeEnd < readEnd, "Write should finish before slow read — may have queued behind the reader")
    }

    // MARK: - Test [2]: Performance assertion

    /// Microbench: concurrent read+write must complete faster than the sum of
    /// isolated read + isolated write × 0.7.
    ///
    /// Under true WAL concurrency: T_both ≈ max(T_read, T_write).
    /// The 0.7 factor is a generous floor per the spec.
    @Test("Concurrent read+write is faster than sum of isolated operations")
    func performanceAssertionTest() async throws {
        let (storage, url) = try await Self.makeTemporaryDB()
        defer {
            Self.cleanupDB(url: url)
            Task { try? await storage.close() }
        }

        // Isolated read (3 iterations, median).
        var readIter = 0
        let tRead = try await Self.measureMedian(iterations: 3) {
            readIter += 1
            _ = try await storage.searchFullText(query: "the", limit: Self.seedCount, filter: .all)
        }

        // Isolated write: 100 sequential upserts (3 iterations, median).
        var writeIter = 0
        let tWrite = try await Self.measureMedian(iterations: 3) {
            writeIter += 1
            for j in 0..<100 {
                let uuid = "write-\(writeIter)-\(j)"
                try await storage.upsertDocument(Self.makeWriteDoc(uuid: uuid))
            }
        }

        // Concurrent: read + 100 writes via async let (3 iterations, median).
        var bothIter = 0
        let tBoth = try await Self.measureMedian(iterations: 3) {
            bothIter += 1
            let iter = bothIter
            async let readTask: [FullTextHit] = storage.searchFullText(
                query: "the", limit: Self.seedCount, filter: .all
            )
            for j in 0..<100 {
                let uuid = "both-\(iter)-\(j)"
                try await storage.upsertDocument(Self.makeWriteDoc(uuid: uuid))
            }
            _ = try await readTask
        }

        let threshold = tRead + tWrite * 0.7
        let msg = "T_both=\(tBoth) T_read=\(tRead) T_write=\(tWrite) threshold=\(threshold)"
        #expect(tBoth < threshold, Comment(rawValue: msg))
    }

    // MARK: - Test [4]: SafariUnfucker stall regression

    /// Reproduces the original stall in miniature: a bulk-write loop of 50
    /// upserts must not be blocked by a concurrent slow FTS scan.
    ///
    /// This is the SafariUnfucker scenario where ~1,178/6,511 chunks were
    /// indexed and a search held the storage actor, freezing the indexer.
    @Test("Bulk write loop is not stalled by a concurrent slow FTS scan")
    func safariUnfuckerRegressionTest() async throws {
        let (storage, url) = try await Self.makeTemporaryDB()
        defer {
            Self.cleanupDB(url: url)
            Task { try? await storage.close() }
        }

        let bulkCount = 50

        // Baseline: 50 sequential writes in isolation.
        var isoCounter = 0
        let tIsolated = try await Self.measureMedian(iterations: 1) {
            for _ in 0..<bulkCount {
                isoCounter += 1
                try await storage.upsertDocument(Self.makeWriteDoc(uuid: "iso-\(isoCounter)"))
            }
        }

        // Concurrent: 50 writes + slow FTS scan running in parallel.
        // We time only the writes task; the FTS scan runs alongside.
        let tConcurrentWrites = try await withThrowingTaskGroup(of: Double.self) { group -> Double in
            group.addTask {
                // Slow FTS scan — holds the reader actor for its duration.
                _ = try await storage.searchFullText(
                    query: "the", limit: Self.seedCount, filter: .all
                )
                return 0.0
            }

            let batchCounter = MutableBox(0)
            group.addTask {
                let t = Date()
                for _ in 0..<bulkCount {
                    let n = batchCounter.increment()
                    try await storage.upsertDocument(Self.makeWriteDoc(uuid: "conc-\(n)"))
                }
                return Date().timeIntervalSince(t)
            }

            var writeDuration = 0.0
            for try await result in group {
                if result > 0 { writeDuration = result }
            }
            return writeDuration
        }

        let ceiling = tIsolated * 1.5
        let msg = "concurrent writes took \(tConcurrentWrites)s, isolated was \(tIsolated)s, ceiling=\(ceiling)s"
        #expect(tConcurrentWrites < ceiling, Comment(rawValue: msg))
    }
}

// Simple sendable reference box for capturing mutable counters in task groups.
private final class MutableBox: @unchecked Sendable {
    private var value: Int
    private let lock = NSLock()

    init(_ initial: Int = 0) { self.value = initial }

    @discardableResult
    func increment() -> Int {
        lock.lock()
        defer { lock.unlock() }
        value += 1
        return value
    }
}
