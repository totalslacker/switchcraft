// SPDX-License-Identifier: Apache-2.0
import Foundation
import SQLite3
import SwitchcraftCore

/// Internal actor that owns the read-only connection for a file-backed SQLite
/// database opened in WAL mode. Runs concurrently with `SQLiteWriterActor`
/// because WAL allows N concurrent readers alongside one writer.
///
/// The connection is opened with `SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX`.
/// It does NOT set `journal_mode = WAL` (WAL is a database-file property set
/// once by the writer) and does NOT run schema DDL (tables already exist when
/// the reader opens).
actor SQLiteReaderActor {
    private let path: String
    private var connection: SQLiteConnection?

    init(path: String) {
        self.path = path
    }

    // MARK: - Lifecycle

    func open() throws {
        if connection != nil { return }
        let conn = try SQLiteConnection(
            path: path,
            flags: SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX
        )
        try conn.execute("PRAGMA foreign_keys = ON")
        connection = conn
    }

    func close() {
        connection = nil
    }

    // MARK: - Documents

    func document(uuid: String) throws -> DocumentRecord? {
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

    func documents(matching filter: StorageFilter) throws -> [DocumentRecord] {
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

    func documents(forChunkHash hash: String) throws -> [DocumentRecord] {
        let conn = try requireConnection()
        let stmt = try conn.prepare("""
            SELECT uuid, date, metadata, hash, body, lens
            FROM document
            WHERE hash = ?
            """)
        try stmt.bind([.text(hash)])
        var results: [DocumentRecord] = []
        while try stmt.step() {
            results.append(decodeDocument(stmt))
        }
        return results
    }

    func documentCount() throws -> Int {
        try scalarInt("SELECT COUNT(*) FROM document")
    }

    // MARK: - Chunks

    func chunk(hash: String) throws -> ChunkRecord? {
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

    func chunk(id: Int64) throws -> ChunkRecord? {
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

    func chunkCount() throws -> Int {
        try scalarInt("SELECT COUNT(*) FROM chunk")
    }

    // MARK: - Generations

    func generations() throws -> [GenerationRecord] {
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

    // MARK: - Buckets

    func buckets(forGeneration generationID: Int64) throws -> [BucketRecord] {
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

    func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) throws -> [FullTextHit] {
        guard limit > 0, !query.trimmingCharacters(in: .whitespaces).isEmpty else {
            return []
        }
        let ftsTerms = query
            .split { !$0.isLetter && !$0.isNumber }
            .map { "\"\(String($0))\"" }
        guard !ftsTerms.isEmpty else { return [] }
        let ftsQuery = ftsTerms.joined(separator: " ")

        let conn = try requireConnection()
        let clause = filter.lower(tableAlias: "d")

        let sql = """
            SELECT d.uuid, -bm25(document_fts) AS score
            FROM document_fts
            JOIN document d ON d.rowid = document_fts.rowid
            WHERE document_fts MATCH ? AND \(clause.sql)
            ORDER BY score DESC, d.uuid ASC
            LIMIT ?
            """
        let stmt = try conn.prepare(sql)
        var bindings: [SQLValue] = [.text(ftsQuery)]
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

    // MARK: - Search Deadline

    func configureProgressHandler(_ ctx: SearchDeadlineContext?) {
        guard let conn = connection else { return }
        if let ctx {
            let sqlPhaseStart = ContinuousClock.now
            let elapsed = sqlPhaseStart - ctx.searchStart
            conn.progressState.sqlPhaseStart = sqlPhaseStart
            conn.progressState.remainingBudget = max(.zero, ctx.deadline - elapsed)
            conn.progressState.isActive = true
        } else {
            conn.progressState.isActive = false
        }
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
