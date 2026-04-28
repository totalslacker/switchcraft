import Foundation
import SwitchcraftCore

/// SQLite-backed implementation of `SwitchcraftStorage`.
///
/// All access is serialised through actor isolation; the underlying sqlite3
/// handle is single-threaded by construction. Pass `":memory:"` as the path
/// for an ephemeral store (useful in tests).
public actor SQLiteStorage: SwitchcraftStorage {
    private let path: String
    private var connection: SQLiteConnection?

    public init(path: String) {
        self.path = path
    }

    // MARK: - Lifecycle

    public func open() async throws {
        if connection != nil { return }
        let conn = try SQLiteConnection(path: path)
        try conn.execute("PRAGMA foreign_keys = ON")
        try conn.execute("PRAGMA journal_mode = WAL")
        for ddl in Schema.statements {
            try conn.execute(ddl)
        }
        connection = conn
    }

    public func close() async throws {
        connection = nil
    }

    public func clear() async throws {
        let conn = try requireConnection()
        try conn.transaction {
            try conn.execute("DELETE FROM bucket")
            try conn.execute("DELETE FROM generation")
            try conn.execute("DELETE FROM chunk")
            try conn.execute("DELETE FROM document")
            // Reset AUTOINCREMENT counters so id sequences start at 1 again.
            try conn.execute("DELETE FROM sqlite_sequence")
        }
    }

    // MARK: - Documents

    public func upsertDocument(_ document: DocumentRecord) async throws {
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

    public func deleteDocument(uuid: String) async throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("DELETE FROM document WHERE uuid = ?")
        try stmt.bind([.text(uuid)])
        try stmt.step()
    }

    public func document(uuid: String) async throws -> DocumentRecord? {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT uuid, date, metadata, hash, body, lens
            FROM document
            WHERE uuid = ?
            """)
        try stmt.bind([.text(uuid)])
        guard try stmt.step() else { return nil }
        return decodeDocument(stmt)
    }

    public func documents(matching filter: StorageFilter) async throws -> [DocumentRecord] {
        let conn = try requireConnection()
        let clause = filter.lower()
        let stmt = try conn.prepare("""
            SELECT uuid, date, metadata, hash, body, lens
            FROM document
            WHERE \(clause.sql)
            """)
        try stmt.bind(clause.bindings)
        var results: [DocumentRecord] = []
        while try stmt.step() {
            results.append(decodeDocument(stmt))
        }
        return results
    }

    public func documentCount() async throws -> Int {
        try scalarInt("SELECT COUNT(*) FROM document")
    }

    // MARK: - Chunks

    public func upsertChunk(_ record: ChunkRecord) async throws -> ChunkRecord {
        let conn = try requireConnection()

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

    public func chunk(hash: String) async throws -> ChunkRecord? {
        try chunk(hashLookup: hash)
    }

    public func chunk(id: Int64) async throws -> ChunkRecord? {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT id, hash, model, embeddings, counts
            FROM chunk
            WHERE id = ?
            """)
        try stmt.bind([.int(id)])
        guard try stmt.step() else { return nil }
        return decodeChunk(stmt)
    }

    public func chunkCount() async throws -> Int {
        try scalarInt("SELECT COUNT(*) FROM chunk")
    }

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

    // MARK: - Generations

    public func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord {
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

    public func generations() async throws -> [GenerationRecord] {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT id, level, num_embeddings, min_chunk_id, max_chunk_id, created
            FROM generation
            ORDER BY id ASC
            """)
        var rows: [GenerationRecord] = []
        while try stmt.step() {
            rows.append(GenerationRecord(
                id: stmt.columnInt64(0),
                level: Int(stmt.columnInt64(1)),
                numEmbeddings: Int(stmt.columnInt64(2)),
                minChunkID: stmt.columnInt64(3),
                maxChunkID: stmt.columnInt64(4),
                created: Codecs.decodeDate(stmt.columnDouble(5))
            ))
        }
        return rows
    }

    public func deleteGeneration(id: Int64) async throws {
        let conn = try requireConnection()
        let stmt = try conn.prepare("DELETE FROM generation WHERE id = ?")
        try stmt.bind([.int(id)])
        try stmt.step()
    }

    // MARK: - Buckets

    public func insertBucket(_ bucket: BucketRecord) async throws -> BucketRecord {
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

    public func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord] {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT id, generation_id, center, indices, residuals
            FROM bucket
            WHERE generation_id = ?
            ORDER BY id ASC
            """)
        try stmt.bind([.int(generationID)])
        var rows: [BucketRecord] = []
        while try stmt.step() {
            rows.append(BucketRecord(
                id: stmt.columnInt64(0),
                generationID: stmt.columnInt64(1),
                center: stmt.columnBlob(2),
                indices: stmt.columnBlob(3),
                residuals: stmt.columnBlob(4)
            ))
        }
        return rows
    }

    // MARK: - Full-text Search

    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        guard limit > 0, !query.trimmingCharacters(in: .whitespaces).isEmpty else {
            return []
        }
        let conn = try requireConnection()
        let clause = filter.lower(tableAlias: "d")

        // bm25() returns lower = more relevant; negate so higher = better.
        let sql = """
            SELECT d.uuid, -bm25(document_fts) AS score
            FROM document_fts
            JOIN document d ON d.rowid = document_fts.rowid
            WHERE document_fts MATCH ? AND \(clause.sql)
            ORDER BY score DESC
            LIMIT ?
            """
        let stmt = try conn.prepare(sql)
        var bindings: [SQLValue] = [.text(query)]
        bindings.append(contentsOf: clause.bindings)
        bindings.append(.int(Int64(limit)))
        try stmt.bind(bindings)

        var hits: [FullTextHit] = []
        while try stmt.step() {
            hits.append(FullTextHit(
                uuid: stmt.columnText(0),
                score: Float(stmt.columnDouble(1))
            ))
        }
        return hits
    }

    // MARK: - Helpers

    private func requireConnection() throws -> SQLiteConnection {
        guard let connection else {
            throw SQLiteError(code: 1, message: "storage is not open")
        }
        return connection
    }

    private func scalarInt(_ sql: String) throws -> Int {
        let conn = try requireConnection()
        let stmt = try conn.prepare(sql)
        guard try stmt.step() else { return 0 }
        return Int(stmt.columnInt64(0))
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
