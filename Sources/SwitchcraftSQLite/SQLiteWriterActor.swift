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
        for ddl in Schema.statements {
            try conn.execute(ddl)
        }
        connection = conn
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
            INSERT INTO document (uuid, date, metadata, hash, body, lens)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET
                date = excluded.date,
                metadata = excluded.metadata,
                hash = excluded.hash,
                body = excluded.body,
                lens = excluded.lens
            """)
        try stmt.bind([
            .text(document.uuid),
            .real(Codecs.encode(document.date)),
            .blob(document.metadata),
            .text(document.hash),
            .text(document.body),
            .text(Codecs.encode(document.lens)),
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
