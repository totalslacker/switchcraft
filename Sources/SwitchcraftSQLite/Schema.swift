// SPDX-License-Identifier: Apache-2.0
/// SQL DDL for the SQLite reference backend.
///
/// The schema mirrors `docs/Plan.md`. FTS5 maintenance is wired via triggers
/// on `document` so that backends only have to issue document-level inserts
/// and deletes.
///
/// Schema version history (tracked via PRAGMA user_version):
///   0 — original schema (no `title` column)
///   1 — added `title TEXT` to `document`; added `title` as first column in
///       `document_fts`. The `bm25(document_fts, w, 1.0)` call in
///       `SQLiteReaderActor.searchFullText` maps w to `title` and 1.0 to
///       `body`. This column order is a hard constraint — see ADR 026.
enum Schema {
    static let statements: [String] = [
        // Documents (V1 schema includes nullable title column)
        """
        CREATE TABLE IF NOT EXISTS document (
            uuid TEXT PRIMARY KEY,
            date REAL NOT NULL,
            metadata BLOB NOT NULL DEFAULT (x''),
            hash TEXT NOT NULL,
            body TEXT NOT NULL,
            lens TEXT NOT NULL DEFAULT '',
            title TEXT
        )
        """,

        // FTS5 over title and body (title first — see ADR 026 column-order constraint).
        // bm25(document_fts, titleWeight, 1.0) maps first argument to `title`.
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS document_fts
            USING fts5(title, body, content='document', content_rowid='rowid')
        """,

        // Triggers keep document_fts in sync.
        """
        CREATE TRIGGER IF NOT EXISTS document_ai AFTER INSERT ON document BEGIN
            INSERT INTO document_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS document_ad AFTER DELETE ON document BEGIN
            INSERT INTO document_fts(document_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS document_au AFTER UPDATE ON document BEGIN
            INSERT INTO document_fts(document_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
            INSERT INTO document_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
        END
        """,

        // Chunks
        """
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT NOT NULL UNIQUE,
            model TEXT NOT NULL,
            embeddings BLOB NOT NULL,
            counts TEXT NOT NULL DEFAULT ''
        )
        """,

        // Generations (LSM levels)
        """
        CREATE TABLE IF NOT EXISTS generation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            num_embeddings INTEGER NOT NULL,
            min_chunk_id INTEGER NOT NULL,
            max_chunk_id INTEGER NOT NULL,
            created REAL NOT NULL
        )
        """,

        // Buckets — cascade-deleted when their parent generation is removed.
        """
        CREATE TABLE IF NOT EXISTS bucket (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_id INTEGER NOT NULL REFERENCES generation(id) ON DELETE CASCADE,
            center BLOB NOT NULL,
            indices BLOB NOT NULL,
            residuals BLOB NOT NULL
        )
        """,

        "CREATE INDEX IF NOT EXISTS bucket_generation_idx ON bucket(generation_id)",

        // Single-row ledger snapshot slot (issue #136 / ADR 034). The
        // `CHECK (id = 1)` constraint enforces at most one snapshot; save is an
        // UPSERT on id=1. Created unconditionally on every open via
        // `CREATE TABLE IF NOT EXISTS`, so existing databases pick it up lazily
        // without a `PRAGMA user_version` bump (only the V0→V1 title migration,
        // which altered an existing table, needed conditional migration logic).
        """
        CREATE TABLE IF NOT EXISTS ledger_snapshot (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            dims INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            max_chunk_id INTEGER NOT NULL,
            total_embeddings INTEGER NOT NULL,
            max_generation_id INTEGER NOT NULL,
            generation_count INTEGER NOT NULL,
            payload BLOB NOT NULL
        )
        """,

        // Search calls `documents(forChunkHash:)` once per selected chunk
        // to bridge ADR 005's `(chunkID, tokenOffset)` bucket pairs back
        // to documents. Without this index that becomes a full table scan
        // per call and dominates wall time on large corpora.
        "CREATE INDEX IF NOT EXISTS document_hash_idx ON document(hash)",
    ]
}
