// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwitchcraftCore
import Switchcraft

/// SQLITE_INTERRUPT result code (decimal 9). Defined here rather than
/// relying on the C-header name to avoid brittle cross-module constant
/// references.
private let sqliteInterruptCode: Int32 = 9

/// SQLite-backed implementation of `SwitchcraftStorage`.
///
/// For **file-backed** paths, storage is split across two independent actors:
/// - `SQLiteWriterActor` — owns the WAL-mode writer connection; handles all
///   mutating operations.
/// - `SQLiteReaderActor` — owns a read-only connection; handles all queries.
///
/// Because WAL allows one writer + N concurrent readers at the SQLite file
/// level, a long-running search no longer blocks ongoing indexing writes.
///
/// For **in-memory** paths (`:memory:`, `file::memory:`, or any URI with
/// `mode=memory`), a single connection is retained directly by this actor,
/// matching the original single-actor behaviour. WAL concurrency is a
/// file-backed-only property and does not apply to in-memory databases.
public actor SQLiteStorage: SwitchcraftStorage {
    private let path: String
    private var mode: Mode = .closed

    private enum Mode {
        case closed
        /// Single-connection path for in-memory SQLite databases.
        case inMemory(SQLiteConnection)
        /// Dual-actor path for file-backed WAL databases.
        case fileBacked(SQLiteWriterActor, SQLiteReaderActor)
    }

    /// Deadline context for the currently running search call. Set by
    /// `configureSearchDeadline` at the top of `SearchEngine.searchHybrid`
    /// and cleared when the call completes. Provides `searchStart` for
    /// computing total elapsed time when translating `SQLITE_INTERRUPT`.
    private var currentDeadlineContext: SearchDeadlineContext?

    private let ftsTitleWeight: Float

    public init(path: String, ftsTitleWeight: Float = 3.0) {
        self.path = path
        self.ftsTitleWeight = ftsTitleWeight
    }

    // MARK: - In-memory path detection

    static func isInMemoryPath(_ path: String) -> Bool {
        path == ":memory:"
            || path.contains(":memory:")
            || path.contains("mode=memory")
    }

    // MARK: - Lifecycle

    public func open() async throws {
        guard case .closed = mode else { return }

        if Self.isInMemoryPath(path) {
            let conn = try SQLiteConnection(path: path)
            try conn.execute("PRAGMA foreign_keys = ON")
            try conn.execute("PRAGMA journal_mode = WAL")
            for ddl in Schema.statements {
                try conn.execute(ddl)
            }
            mode = .inMemory(conn)
        } else {
            let writer = SQLiteWriterActor(path: path)
            let reader = SQLiteReaderActor(path: path, ftsTitleWeight: ftsTitleWeight)
            // Writer opens first: sets WAL mode and runs schema DDL.
            do {
                try await writer.open()
            } catch {
                throw error
            }
            // Reader opens second. On failure, close the writer and rethrow.
            do {
                try await reader.open()
            } catch {
                await writer.close()
                throw error
            }
            mode = .fileBacked(writer, reader)
        }
    }

    public func close() async throws {
        switch mode {
        case .closed:
            break
        case .inMemory:
            mode = .closed
        case .fileBacked(let writer, let reader):
            mode = .closed
            await writer.close()
            await reader.close()
        }
    }

    /// Flush pending WAL pages to the main database file and reset the WAL.
    /// Returns `.complete` for in-memory or closed stores (no WAL to truncate).
    public func walCheckpoint() async throws -> CheckpointResult {
        switch mode {
        case .closed, .inMemory:
            return .complete
        case .fileBacked(let writer, _):
            return try await writer.walCheckpoint()
        }
    }

    public func clear() async throws {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            try conn.transaction {
                try conn.execute("DELETE FROM bucket")
                try conn.execute("DELETE FROM generation")
                try conn.execute("DELETE FROM chunk")
                try conn.execute("DELETE FROM document")
                try conn.execute("DELETE FROM sqlite_sequence")
            }
        case .fileBacked(let writer, _):
            try await writer.clear()
        }
    }

    // MARK: - Documents

    public func upsertDocument(_ document: DocumentRecord) async throws {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
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
        case .fileBacked(let writer, _):
            try await writer.upsertDocument(document)
        }
    }

    public func deleteDocument(uuid: String) async throws {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("DELETE FROM document WHERE uuid = ?")
            try stmt.bind([.text(uuid)])
            try stmt.step()
        case .fileBacked(let writer, _):
            try await writer.deleteDocument(uuid: uuid)
        }
    }

    public func document(uuid: String) async throws -> DocumentRecord? {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT uuid, date, metadata, hash, body, lens, title
                FROM document
                WHERE uuid = ?
                """)
            try stmt.bind([.text(uuid)])
            guard try stmt.step() else { return nil }
            return decodeDocument(stmt)
        case .fileBacked(_, let reader):
            return try await reader.document(uuid: uuid)
        }
    }

    public func documents(matching filter: StorageFilter) async throws -> [DocumentRecord] {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let clause = filter.lower()
            let stmt = try conn.prepare("""
                SELECT uuid, date, metadata, hash, body, lens, title
                FROM document
                WHERE \(clause.sql)
                """)
            try stmt.bind(clause.bindings)
            var results: [DocumentRecord] = []
            while try stmt.step() {
                results.append(decodeDocument(stmt))
            }
            return results
        case .fileBacked(_, let reader):
            return try await reader.documents(matching: filter)
        }
    }

    public func documents(forChunkHash hash: String) async throws -> [DocumentRecord] {
        try Task.checkCancellation()
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT uuid, date, metadata, hash, body, lens, title
                FROM document
                WHERE hash = ?
                """)
            try stmt.bind([.text(hash)])
            var results: [DocumentRecord] = []
            do {
                while try stmt.step() {
                    results.append(decodeDocument(stmt))
                }
            } catch {
                try translateIfInterrupt(error)
            }
            return results
        case .fileBacked(_, let reader):
            do {
                return try await reader.documents(forChunkHash: hash)
            } catch {
                try translateIfInterrupt(error)
            }
        }
    }

    public func documentCount() async throws -> Int {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("SELECT COUNT(*) FROM document")
            guard try stmt.step() else { return 0 }
            return Int(stmt.columnInt64(0))
        case .fileBacked(_, let reader):
            return try await reader.documentCount()
        }
    }

    public func indexedURLs() async throws -> Set<String> {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("SELECT uuid FROM document")
            var result = Set<String>()
            while try stmt.step() {
                result.insert(stmt.columnText(0))
            }
            return result
        case .fileBacked(_, let reader):
            return try await reader.indexedURLs()
        }
    }

    // MARK: - Chunks

    public func upsertChunk(_ record: ChunkRecord) async throws -> ChunkRecord {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            if let existing = try inMemoryChunkByHash(conn, hash: record.hash) {
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
        case .fileBacked(let writer, _):
            return try await writer.upsertChunk(record)
        }
    }

    public func chunk(hash: String) async throws -> ChunkRecord? {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            return try inMemoryChunkByHash(conn, hash: hash)
        case .fileBacked(_, let reader):
            return try await reader.chunk(hash: hash)
        }
    }

    public func chunk(id: Int64) async throws -> ChunkRecord? {
        try Task.checkCancellation()
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT id, hash, model, embeddings, counts
                FROM chunk
                WHERE id = ?
                """)
            try stmt.bind([.int(id)])
            do {
                guard try stmt.step() else { return nil }
            } catch {
                try translateIfInterrupt(error)
            }
            return decodeChunk(stmt)
        case .fileBacked(_, let reader):
            do {
                return try await reader.chunk(id: id)
            } catch {
                try translateIfInterrupt(error)
            }
        }
    }

    public func chunkCount() async throws -> Int {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("SELECT COUNT(*) FROM chunk")
            guard try stmt.step() else { return 0 }
            return Int(stmt.columnInt64(0))
        case .fileBacked(_, let reader):
            return try await reader.chunkCount()
        }
    }

    public func allChunks() async throws -> [ChunkRecord] {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT id, hash, model, embeddings, counts
                FROM chunk
                ORDER BY id ASC
                """)
            var rows: [ChunkRecord] = []
            while try stmt.step() {
                rows.append(decodeChunk(stmt))
            }
            return rows
        case .fileBacked(_, let reader):
            return try await reader.allChunks()
        }
    }

    public func chunkBucketRefCount(_ chunkID: Int64) async throws -> Int {
        guard let u32 = UInt32(exactly: chunkID) else { return 0 }
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            // Fast path: range check across generations.
            let rangeStmt = try conn.prepare("""
                SELECT id FROM generation
                WHERE min_chunk_id <= ? AND max_chunk_id >= ?
                ORDER BY id ASC
                """)
            try rangeStmt.bind([.int(chunkID), .int(chunkID)])
            var generationIDs: [Int64] = []
            while try rangeStmt.step() {
                generationIDs.append(rangeStmt.columnInt64(0))
            }
            if generationIDs.isEmpty { return 0 }
            // Slow path: decode all blobs and accumulate total pair count.
            var total = 0
            for genID in generationIDs {
                let bucketStmt = try conn.prepare("""
                    SELECT indices FROM bucket WHERE generation_id = ?
                    """)
                try bucketStmt.bind([.int(genID)])
                while try bucketStmt.step() {
                    let blob = bucketStmt.columnBlob(0)
                    let pairs = try IndicesCodec.decode(blob)
                    total += pairs.filter { $0.chunkID == u32 }.count
                }
            }
            return total
        case .fileBacked(_, let reader):
            return try await reader.chunkBucketRefCount(chunkID)
        }
    }

    // MARK: - Generations

    public func insertGeneration(_ generation: GenerationRecord) async throws -> GenerationRecord {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
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
        case .fileBacked(let writer, _):
            return try await writer.insertGeneration(generation)
        }
    }

    public func generations() async throws -> [GenerationRecord] {
        try Task.checkCancellation()
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT id, level, num_embeddings, min_chunk_id, max_chunk_id, created
                FROM generation
                ORDER BY id ASC
                """)
            var rows: [GenerationRecord] = []
            do {
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
            } catch {
                try translateIfInterrupt(error)
            }
            return rows
        case .fileBacked(_, let reader):
            do {
                return try await reader.generations()
            } catch {
                try translateIfInterrupt(error)
            }
        }
    }

    public func deleteGeneration(id: Int64) async throws {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("DELETE FROM generation WHERE id = ?")
            try stmt.bind([.int(id)])
            try stmt.step()
        case .fileBacked(let writer, _):
            try await writer.deleteGeneration(id: id)
        }
    }

    public func updateGenerationEmbeddingCount(id: Int64, count: Int) async throws {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("UPDATE generation SET num_embeddings = ? WHERE id = ?")
            try stmt.bind([.int(Int64(count)), .int(id)])
            try stmt.step()
        case .fileBacked(let writer, _):
            try await writer.updateGenerationEmbeddingCount(id: id, count: count)
        }
    }

    public func replaceGeneration(
        losingGenerationID: Int64,
        survivingRecord: GenerationRecord,
        survivingBuckets: [BucketRecord]
    ) async throws -> GenerationRecord? {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
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
        case .fileBacked(let writer, _):
            return try await writer.replaceGeneration(
                losingGenerationID: losingGenerationID,
                survivingRecord: survivingRecord,
                survivingBuckets: survivingBuckets
            )
        }
    }

    // MARK: - Buckets

    public func insertBucket(_ bucket: BucketRecord) async throws -> BucketRecord {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
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
        case .fileBacked(let writer, _):
            return try await writer.insertBucket(bucket)
        }
    }

    public func buckets(forGeneration generationID: Int64) async throws -> [BucketRecord] {
        try Task.checkCancellation()
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            let stmt = try conn.prepare("""
                SELECT id, generation_id, center, indices, residuals
                FROM bucket
                WHERE generation_id = ?
                ORDER BY id ASC
                """)
            try stmt.bind([.int(generationID)])
            var rows: [BucketRecord] = []
            do {
                while try stmt.step() {
                    rows.append(BucketRecord(
                        id: stmt.columnInt64(0),
                        generationID: stmt.columnInt64(1),
                        center: stmt.columnBlob(2),
                        indices: stmt.columnBlob(3),
                        residuals: stmt.columnBlob(4)
                    ))
                }
            } catch {
                try translateIfInterrupt(error)
            }
            return rows
        case .fileBacked(_, let reader):
            do {
                return try await reader.buckets(forGeneration: generationID)
            } catch {
                try translateIfInterrupt(error)
            }
        }
    }

    // MARK: - Full-text Search

    public func searchFullText(
        query: String,
        limit: Int,
        filter: StorageFilter
    ) async throws -> [FullTextHit] {
        switch mode {
        case .closed:
            throw SQLiteError(code: 1, message: "storage is not open")
        case .inMemory(let conn):
            guard limit > 0, !query.trimmingCharacters(in: .whitespaces).isEmpty else {
                return []
            }
            // FTS5's MATCH grammar treats punctuation (".", "(", ":", quotes, etc.)
            // as syntax. Sanitise by splitting on alphanumeric boundaries, then
            // double-quoting each token so FTS5 reserved keywords (AND/OR/NOT/NEAR)
            // can't be parsed as operators.
            let ftsTerms = query
                .split { !$0.isLetter && !$0.isNumber }
                .map { "\"\(String($0))\"" }
            guard !ftsTerms.isEmpty else { return [] }
            let ftsQuery = ftsTerms.joined(separator: " ")
            let clause = filter.lower(tableAlias: "d")
            // bm25 column weights: title first (ADR 026 column-order constraint), body second.
            // bm25() returns lower = more relevant; negate so higher = better.
            // Tie-break on uuid ascending for deterministic ordering (see ADR 008).
            let titleW = String(format: "%.6g", ftsTitleWeight)
            let sql = """
                SELECT d.uuid, -bm25(document_fts, \(titleW), 1.0) AS score
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
            do {
                // Check cancellation before each step so that a cancelled task
                // does not have to wait for an in-progress sqlite3_step to
                // return before the CancellationError is observed.
                while true {
                    try Task.checkCancellation()
                    guard try stmt.step() else { break }
                    hits.append(FullTextHit(
                        uuid: stmt.columnText(0),
                        score: Float(stmt.columnDouble(1))
                    ))
                }
            } catch {
                try translateIfInterrupt(error)
            }
            return hits
        case .fileBacked(_, let reader):
            do {
                return try await reader.searchFullText(query: query, limit: limit, filter: filter)
            } catch {
                try translateIfInterrupt(error)
            }
        }
    }

    // MARK: - Search Deadline

    public func configureSearchDeadline(_ ctx: SearchDeadlineContext?) async {
        currentDeadlineContext = ctx
        func arm(_ conn: SQLiteConnection) {
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
        switch mode {
        case .closed:
            break
        case .inMemory(let conn):
            arm(conn)
        case .fileBacked(_, let reader):
            await reader.configureProgressHandler(ctx)
        }
    }

    // MARK: - In-memory helpers

    private func inMemoryChunkByHash(_ conn: SQLiteConnection, hash: String) throws -> ChunkRecord? {
        let stmt = try conn.prepare("""
            SELECT id, hash, model, embeddings, counts
            FROM chunk
            WHERE hash = ?
            """)
        try stmt.bind([.text(hash)])
        guard try stmt.step() else { return nil }
        return decodeChunk(stmt)
    }

    /// Translate a `SQLiteError` with code `SQLITE_INTERRUPT` into
    /// `SwitchcraftStoreError.searchTimedOut` using the stored deadline
    /// context. For all other errors, rethrows unchanged.
    ///
    /// Translation only happens when `currentDeadlineContext` is non-nil.
    /// If an interrupt arrives with no active deadline context (shouldn't
    /// happen in normal use, but possible if a caller bypasses the search
    /// pipeline), the original `SQLiteError` is rethrown unchanged rather
    /// than emitting a misleading `searchTimedOut(elapsed: .zero)`.
    private func translateIfInterrupt(_ error: Error) throws -> Never {
        if let sqliteError = error as? SQLiteError,
           sqliteError.code == sqliteInterruptCode,
           let ctx = currentDeadlineContext {
            throw SwitchcraftStoreError.searchTimedOut(elapsed: ContinuousClock.now - ctx.searchStart)
        }
        throw error
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
