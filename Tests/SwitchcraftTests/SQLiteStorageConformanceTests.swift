import Foundation
import Testing
import SQLite3
import SwitchcraftCore
import SwitchcraftSQLite
import SwitchcraftStorageTesting

@Suite("SQLiteStorage Conformance")
struct SQLiteStorageConformanceTests {

    @Test("In-memory SQLiteStorage satisfies the SwitchcraftStorage contract")
    func inMemoryConformance() async throws {
        try await StorageConformance.runAll {
            SQLiteStorage(path: ":memory:")
        }
    }

    @Test("On-disk SQLiteStorage satisfies the SwitchcraftStorage contract")
    func onDiskConformance() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-conformance-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        try await StorageConformance.runAll {
            SQLiteStorage(path: path)
        }
    }

    @Test("Reopening an on-disk store preserves committed data")
    func onDiskPersistence() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-persist-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            let storage = SQLiteStorage(path: path)
            try await storage.open()
            try await storage.upsertDocument(DocumentRecord(
                uuid: "persist-1",
                date: Date(timeIntervalSince1970: 12345),
                metadata: Data(),
                hash: "h",
                body: "persisted body",
                lens: [3, 4]
            ))
            try await storage.close()
        }

        let storage = SQLiteStorage(path: path)
        try await storage.open()
        let fetched = try await storage.document(uuid: "persist-1")
        #expect(fetched?.body == "persisted body")
        #expect(fetched?.lens == [3, 4])
        try await storage.close()
    }

    /// `scanBuckets`'s `residualByteCount` (issue #151 / ADR 041) must match
    /// the full `buckets(forGeneration:)` fetch's `residuals.count` exactly,
    /// for buckets with varied residual sizes, in both the `.inMemory` and
    /// `.fileBacked` `SQLiteStorage` modes — the scan-shaped query reads the
    /// blob's length header via SQL `length(residuals)`, never the payload,
    /// so this proves that shortcut still yields the correct byte count.
    @Test("scanBuckets residualByteCount matches full fetch", arguments: [false, true])
    func scanBucketsResidualByteCountMatchesFullFetch(onDisk: Bool) async throws {
        let path: String
        if onDisk {
            path = NSTemporaryDirectory().appending("switchcraft-scan-bytes-\(UUID().uuidString).sqlite3")
        } else {
            path = ":memory:"
        }
        defer { if onDisk { try? FileManager.default.removeItem(atPath: path) } }

        let storage = SQLiteStorage(path: path)
        try await storage.open()

        let gen = try await storage.insertGeneration(
            GenerationRecord(level: 0, numEmbeddings: 0, minChunkID: 0, maxChunkID: 0, created: Date())
        )
        let residualSizes = [0, 1, 4, 4096]
        var inserted: [BucketRecord] = []
        for (i, size) in residualSizes.enumerated() {
            let bucket = try await storage.insertBucket(
                BucketRecord(
                    generationID: gen.id,
                    center: Data(repeating: UInt8(i), count: 8),
                    indices: Data(),
                    residuals: Data(repeating: UInt8(i), count: size)
                )
            )
            inserted.append(bucket)
        }

        let scanned = try await storage.scanBuckets(forGeneration: gen.id)
        let full = try await storage.buckets(forGeneration: gen.id)
        #expect(scanned.count == full.count)
        let fullByID = Dictionary(uniqueKeysWithValues: full.map { ($0.id, $0) })
        for record in scanned {
            #expect(record.residualByteCount == fullByID[record.id]?.residuals.count)
            #expect(record.center == fullByID[record.id]?.center)
        }

        try await storage.close()
    }

    /// Verify that opening a V0-schema database (no `title` column) triggers the
    /// V0→V1 migration so that title-based FTS works after upgrading.
    @Test("V0→V1 migration adds title column and rebuilds FTS")
    func v0ToV1Migration() async throws {
        let path = NSTemporaryDirectory().appending("switchcraft-v0-\(UUID().uuidString).sqlite3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // 1. Create a V0 database directly via C API (no title column, old FTS schema).
        var db: OpaquePointer?
        let openRc = sqlite3_open_v2(
            path, &db,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_URI,
            nil
        )
        #expect(openRc == SQLITE_OK)
        let v0DDL = [
            "PRAGMA journal_mode = WAL",
            """
            CREATE TABLE document (
                uuid TEXT PRIMARY KEY,
                date REAL NOT NULL,
                metadata BLOB NOT NULL DEFAULT (x''),
                hash TEXT NOT NULL,
                body TEXT NOT NULL,
                lens TEXT NOT NULL DEFAULT ''
            )
            """,
            "CREATE VIRTUAL TABLE document_fts USING fts5(body, content='document', content_rowid='rowid')",
            """
            CREATE TRIGGER document_ai AFTER INSERT ON document BEGIN
                INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
            END
            """,
            """
            CREATE TRIGGER document_ad AFTER DELETE ON document BEGIN
                INSERT INTO document_fts(document_fts, rowid, body) VALUES ('delete', old.rowid, old.body);
            END
            """,
            """
            CREATE TRIGGER document_au AFTER UPDATE ON document BEGIN
                INSERT INTO document_fts(document_fts, rowid, body) VALUES ('delete', old.rowid, old.body);
                INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
            END
            """,
            "INSERT INTO document (uuid, date, metadata, hash, body, lens) VALUES ('existing-1', 0, x'', 'h1', 'old body text', '')",
        ]
        for sql in v0DDL {
            let rc = sqlite3_exec(db, sql, nil, nil, nil)
            #expect(rc == SQLITE_OK, "V0 DDL failed: \(sql)")
        }
        sqlite3_close_v2(db)

        // 2. Open via SQLiteStorage — should trigger V0→V1 migration.
        let storage = SQLiteStorage(path: path)
        try await storage.open()

        // 3. Existing document round-trips correctly after migration.
        let existing = try await storage.document(uuid: "existing-1")
        #expect(existing?.body == "old body text")
        #expect(existing?.title == nil)

        // 4. New documents with titles can be inserted and found via FTS.
        try await storage.upsertDocument(DocumentRecord(
            uuid: "migrated-titled",
            date: Date(timeIntervalSince1970: 1),
            hash: "h2",
            body: "homework help",
            title: "Bartleby"
        ))

        let hits = try await storage.searchFullText(query: "bartleby", limit: 10, filter: .all)
        #expect(hits.contains { $0.uuid == "migrated-titled" }, "title-matched document must appear after migration")

        try await storage.close()
    }
}
