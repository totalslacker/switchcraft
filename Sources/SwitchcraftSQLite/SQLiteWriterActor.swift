// SPDX-License-Identifier: Apache-2.0
import Foundation
import SQLite3
import SwitchcraftCore

/// Internal actor that owns the writer connection for a file-backed SQLite
/// database opened in WAL mode. Serialises all mutating operations; the
/// companion `SQLiteReaderActor` runs concurrently on its own connection,
/// enabling WAL's one-writer + N-readers concurrency.
actor SQLiteWriterActor {
    private let path: String
    private var connection: SQLiteConnection?

    init(path: String) {
        self.path = path
    }

    // MARK: - Lifecycle

    func open() throws {
        if connection != nil { return }
        let conn = try SQLiteConnection(path: path)
        try conn.execute("PRAGMA foreign_keys = ON")
        try conn.execute("PRAGMA journal_mode = WAL")

        // Detect existing database before applying schema DDL.
        let isExistingDB = try Self.tableExists(conn, name: "document")

        for ddl in Schema.statements {
            try conn.execute(ddl)
        }

        // V0→V1 migration: add `title` column to `document` and rebuild FTS.
        // This is the first schema migration; subsequent migrations should
        // bump the version gate and add their own branch.
        // FTS rebuild on large corpora may be slow (seconds); acceptable for v1.
        if isExistingDB {
            let hasTitleColumn = try Self.columnExists(conn, table: "document", column: "title")
            if !hasTitleColumn {
                try Self.migrateV0toV1(conn)
            }
        }
        // Stamp version 1 on both new installs and migrated databases.
        try conn.execute("PRAGMA user_version = 1")

        connection = conn
    }

    // MARK: - Schema migration helpers

    private static func tableExists(_ conn: SQLiteConnection, name: String) throws -> Bool {
        let stmt = try conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        )
        try stmt.bind([.text(name)])
        return try stmt.step()
    }

    private static func columnExists(_ conn: SQLiteConnection, table: String, column: String) throws -> Bool {
        let stmt = try conn.prepare("PRAGMA table_info(\(table))")
        while try stmt.step() {
            if stmt.columnText(1) == column { return true }
        }
        return false
    }

    private static func migrateV0toV1(_ conn: SQLiteConnection) throws {
        try conn.transaction {
            // Add title column to base table.
            try conn.execute("ALTER TABLE document ADD COLUMN title TEXT")

            // Drop old triggers and FTS virtual table (FTS5 doesn't support ADD COLUMN).
            try conn.execute("DROP TRIGGER IF EXISTS document_au")
            try conn.execute("DROP TRIGGER IF EXISTS document_ad")
            try conn.execute("DROP TRIGGER IF EXISTS document_ai")
            try conn.execute("DROP TABLE IF EXISTS document_fts")

            // Recreate FTS and triggers with V1 DDL (title first — ADR 026).
            try conn.execute("""
                CREATE VIRTUAL TABLE document_fts
                    USING fts5(title, body, content='document', content_rowid='rowid')
                """)
            try conn.execute("""
                CREATE TRIGGER document_ai AFTER INSERT ON document BEGIN
                    INSERT INTO document_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
                END
                """)
            try conn.execute("""
                CREATE TRIGGER document_ad AFTER DELETE ON document BEGIN
                    INSERT INTO document_fts(document_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
                END
                """)
            try conn.execute("""
                CREATE TRIGGER document_au AFTER UPDATE ON document BEGIN
                    INSERT INTO document_fts(document_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
                    INSERT INTO document_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
                END
                """)

            // Rebuild FTS content from the base table.
            try conn.execute("INSERT INTO document_fts(rowid, title, body) SELECT rowid, title, body FROM document")
        }
    }

    func close() {
        connection = nil
    }

    /// Run `PRAGMA wal_checkpoint(TRUNCATE)` on the writer connection,
    /// flushing all pending WAL pages to the main database file and resetting
    /// the WAL header. No-op if the connection is not open.
    func walCheckpoint() throws {
        guard let conn = connection else { return }
        try conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    }

    // MARK: - Documents

    func upsertDocument(_ document: DocumentRecord) throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            INSERT INTO document (uuid, date, metadata, hash, body, lens, title)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET
                date = excluded.date,
                metadata = excluded.metadata,
                hash = excluded.hash,
                body = excluded.body,
                lens = excluded.lens,
                title = excluded.title
            """)
        let titleBinding: SQLValue = document.title.map { .text($0) } ?? .null
        try stmt.bind([
            .text(document.uuid),
            .real(Codecs.encode(document.date)),
            .blob(document.metadata),
            .text(document.hash),
            .text(document.body),
            .text(Codecs.encode(document.lens)),
            titleBinding,
        ])
        try stmt.step()
    }

    func deleteDocument(uuid: String) throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("DELETE FROM document WHERE uuid = ?")
        try stmt.bind([.text(uuid)])
        try stmt.step()
    }

    // MARK: - Chunks

    func upsertChunk(_ record: ChunkRecord) throws -> ChunkRecord {
        let conn = try requireConnection()

        // Writer-side dedup: check-then-insert is atomic within the writer
        // because WAL guarantees only one concurrent writer. No TOCTOU risk.
        if let existing = try chunk(hashLookup: record.hash) {
            return existing
        }

        let insert = try conn.prepare("""
            INSERT INTO chunk (hash, model, embeddings, counts)
            VALUES (?, ?, ?, ?)
            """)
        try insert.bind([
            .text(record.hash),
            .text(record.model),
            .blob(record.embeddings),
            .text(Codecs.encode(record.counts)),
        ])
        try insert.step()

        var inserted = record
        inserted.id = conn.lastInsertRowID
        return inserted
    }

    // MARK: - Generations

    func insertGeneration(_ generation: GenerationRecord) throws -> GenerationRecord {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            INSERT INTO generation (level, num_embeddings, min_chunk_id, max_chunk_id, created)
            VALUES (?, ?, ?, ?, ?)
            """)
        try stmt.bind([
            .int(Int64(generation.level)),
            .int(Int64(generation.numEmbeddings)),
            .int(generation.minChunkID),
            .int(generation.maxChunkID),
            .real(Codecs.encode(generation.created)),
        ])
        try stmt.step()
        var inserted = generation
        inserted.id = conn.lastInsertRowID
        return inserted
    }

    func deleteGeneration(id: Int64) throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("DELETE FROM generation WHERE id = ?")
        try stmt.bind([.int(id)])
        try stmt.step()
    }

    func updateGenerationEmbeddingCount(id: Int64, count: Int) throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("UPDATE generation SET num_embeddings = ? WHERE id = ?")
        try stmt.bind([.int(Int64(count)), .int(id)])
        try stmt.step()
    }

    /// Atomically delete `losingGenerationID` (FK-cascades its buckets) and,
    /// if `survivingBuckets` is non-empty, insert a new generation + buckets.
    /// All operations run inside a single `BEGIN`/`COMMIT`/`ROLLBACK` block.
    func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) throws -> GenerationRecord? {
        let conn = try requireConnection()
        return try conn.transaction {
            // Delete the loser (FK cascade removes its buckets).
            let delStmt = try conn.prepare("DELETE FROM generation WHERE id = ?")
            try delStmt.bind([.int(losingGenerationID)])
            try delStmt.step()

            guard !survivingBuckets.isEmpty else { return nil }

            // Insert the surviving generation record.
            let genStmt = try conn.prepare("""
                INSERT INTO generation (level, num_embeddings, min_chunk_id, max_chunk_id, created)
                VALUES (?, ?, ?, ?, ?)
                """)
            try genStmt.bind([
                .int(Int64(survivingRecord.level)),
                .int(Int64(survivingRecord.numEmbeddings)),
                .int(survivingRecord.minChunkID),
                .int(survivingRecord.maxChunkID),
                .real(Codecs.encode(survivingRecord.created)),
            ])
            try genStmt.step()
            let newGenID = conn.lastInsertRowID

            // Insert each surviving bucket with the new generationID.
            for bucket in survivingBuckets {
                let bucketStmt = try conn.prepare("""
                    INSERT INTO bucket (generation_id, center, indices, residuals)
                    VALUES (?, ?, ?, ?)
                    """)
                try bucketStmt.bind([
                    .int(newGenID),
                    .blob(bucket.center),
                    .blob(bucket.indices),
                    .blob(bucket.residuals),
                ])
                try bucketStmt.step()
            }

            var newGen = survivingRecord
            newGen.id = newGenID
            return newGen
        }
    }

    // MARK: - Buckets

    func insertBucket(_ bucket: BucketRecord) throws -> BucketRecord {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            INSERT INTO bucket (generation_id, center, indices, residuals)
            VALUES (?, ?, ?, ?)
            """)
        try stmt.bind([
            .int(bucket.generationID),
            .blob(bucket.center),
            .blob(bucket.indices),
            .blob(bucket.residuals),
        ])
        try stmt.step()
        var inserted = bucket
        inserted.id = conn.lastInsertRowID
        return inserted
    }

    // MARK: - Clear

    func clear() throws {
        let conn = try requireConnection()
        try conn.transaction {
            try conn.execute("DELETE FROM bucket")
            try conn.execute("DELETE FROM generation")
            try conn.execute("DELETE FROM chunk")
            try conn.execute("DELETE FROM document")
            try conn.execute("DELETE FROM sqlite_sequence")
        }
    }

    // MARK: - Helpers

    private func chunk(hashLookup hash: String) throws -> ChunkRecord? {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT id, hash, model, embeddings, counts
            FROM chunk
            WHERE hash = ?
            """)
        try stmt.bind([.text(hash)])
        guard try stmt.step() else { return nil }
        return decodeChunk(stmt)
    }

    private func requireConnection() throws -> SQLiteConnection {
        guard let connection else {
            throw SQLiteError(code: 1, message: "storage is not open")
        }
        return connection
    }

    private func decodeDocument(_ stmt: SQLiteStatement) -> DocumentRecord {
        DocumentRecord(
            uuid: stmt.columnText(0),
            date: Codecs.decodeDate(stmt.columnDouble(1)),
            metadata: stmt.columnBlob(2),
            hash: stmt.columnText(3),
            body: stmt.columnText(4),
            title: stmt.columnTextOptional(6),
            lens: Codecs.decodeInts(stmt.columnText(5))
        )
    }

    private func decodeChunk(_ stmt: SQLiteStatement) -> ChunkRecord {
        ChunkRecord(
            id: stmt.columnInt64(0),
            hash: stmt.columnText(1),
            model: stmt.columnText(2),
            embeddings: stmt.columnBlob(3),
            counts: Codecs.decodeInts(stmt.columnText(4))
        )
    }
}
