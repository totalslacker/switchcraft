# ADR 026 — FTS title field, V0→V1 schema migration, and BM25 title-weight config

**Status**: Accepted
**Date**: 2026-06-03
**Issue**: #108 (add title field to DocumentRecord / BM25 FTS schema for title-weight boosting)

---

## Context

ADR 025 introduced title-prepend at embedding time (placing the title at T5
position 0 to improve vector-pipeline ranking for short proper-noun queries).
Section (d) of that ADR deferred a complementary improvement: indexing the
title in the FTS/BM25 pipeline so that a document whose body is semantically
disjoint from its title can still be retrieved via title token matching.

The concrete failure case: query `bartleby` against a ~19,000-page corpus.
`bartleby.com` (a homework-help landing page whose title is "Bartleby" but
whose body contains no bartleby-adjacent tokens) falls **outside** the vector
pipeline's top-500 candidates (because the body embedding is dominated by
homework-help semantics) and lands only at FTS rank 5 via incidental URL token
matching. With RRF(K=60), a single-source rank-5 score `1/65 ≈ 0.0154` falls
below multi-source secondary documents, so `bartleby.com` is absent from the
fused top results despite being the canonical domain page.

The vector pipeline failure is fundamental (ADR 025 section (e) — H2 constant
tuning is deferred). The only actionable lever is the FTS pipeline: adding a
first-class `title` column to `document_fts` so that title-token matches
contribute a boosted BM25 signal independently of body content.

---

## Decision

### (a) DocumentRecord.title: String?

`DocumentRecord` gains a `title: String?` field (nil = no title). The existing
`body`, `hash`, `lens`, `uuid`, `date`, and `metadata` fields are unchanged.
All init parameters retain their defaults; existing callers that omit `title`
get byte-identical behaviour to before.

### (b) SQLite schema — V1

The `document` base table gains a nullable `title TEXT` column:

```sql
ALTER TABLE document ADD COLUMN title TEXT
```

The FTS5 virtual table is redefined with `title` as the **first** column:

```sql
CREATE VIRTUAL TABLE document_fts
    USING fts5(title, body, content='document', content_rowid='rowid')
```

All three triggers (`document_ai`, `document_ad`, `document_au`) are updated
to include `title` in every FTS insert/delete operation.

**Column ordering is a hard constraint.** The `bm25(document_fts, w1, w2)`
function maps its weight arguments positionally: `w1` → column 0 (`title`),
`w2` → column 1 (`body`). If this order ever changes in `Schema.swift` without
updating every `bm25()` call site in `SQLiteReaderActor` and `SQLiteStorage`,
the weights will silently apply to the wrong columns. Never reorder the FTS
columns without updating all `bm25()` call sites.

### (c) V0→V1 schema migration

SQLite FTS5 virtual tables do not support `ALTER TABLE ADD COLUMN`. Adding
`title` to `document_fts` requires dropping and recreating the virtual table.

Migration is detected in `SQLiteWriterActor.open()`:
1. Before running `Schema.statements`, check whether the `document` table
   exists (`isExistingDB`).
2. After running DDL (which is idempotent for non-FTS tables via `IF NOT
   EXISTS`), if `isExistingDB`, check whether the `title` column exists in
   `PRAGMA table_info(document)`.
3. If absent, run the migration in a single transaction:
   - `ALTER TABLE document ADD COLUMN title TEXT`
   - Drop old triggers and `document_fts`
   - Recreate FTS and triggers with V1 DDL
   - `INSERT INTO document_fts(rowid, title, body) SELECT rowid, title, body FROM document`
4. Set `PRAGMA user_version = 1` (on both fresh installs and migrated databases).

Fresh installs skip migration (the `document` table did not exist before DDL
ran) and go directly to `PRAGMA user_version = 1` after applying V1 DDL.

**FTS rebuild cost.** On large corpora (19,000+ pages), the FTS rebuild from
step (3d) above may take seconds. This blocks `storage.open()`. Acceptable for
v1; future migrations should consider incremental rebuild or background reindex
strategies for large datasets.

### (d) HybridConfig.ftsTitleWeight

The BM25 per-column weight for the `title` column lives in `HybridConfig`
(not `IndexerConfig`, which governs LSM-tree and k-means parameters). The
field is `ftsTitleWeight: Float` with default `3.0`.

This placement is correct because FTS scoring is a hybrid-search tunable
(analogous to `rrfK` and `perSourceBudget`), not an indexing tunable. All
future FTS scoring parameters should be added to `HybridConfig`.

The weight is forwarded at construction time:
- `SwitchcraftStore.sqlite(...)` → `SQLiteStorage(path:ftsTitleWeight:)`
- `SQLiteStorage` → `SQLiteReaderActor(path:ftsTitleWeight:)`
- `InMemoryStorage(ftsTitleWeight:)` (same default)

### (e) bm25() weight interpolation

SQLite FTS5's `bm25(table, w1, w2, ...)` does not accept bound parameters for
its weight arguments; the values must be literal SQL constants. The weight is
interpolated into the SQL string using `String(format: "%.6g", ftsTitleWeight)`.

Since `ftsTitleWeight` is a `Float` from trusted config (never user input),
the injection risk is negligible. The `%.6g` format avoids pathological
stringifications of NaN or Inf (which cannot appear in a `Float` produced by
the `precondition(>= 0)` guard, but are defended against anyway).

### (f) InMemoryStorage scoring

`InMemoryStorage.searchFullText` applies the weight as:

```
score = (bodyOverlap + titleOverlap * ftsTitleWeight) / queryTerms.count
```

where `overlap` is computed via case-insensitive token intersection. This is a
naive approximation of BM25 (no IDF, no term frequency weighting), consistent
with the pre-existing InMemory FTS semantics. Scores from InMemory and SQLite
backends are not numerically comparable; tests must assert ordering (rank), not
score magnitudes.

### (g) NFCorpus NDCG@10 gate

NFCorpus passages are indexed without `title`; `DocumentRecord.title == nil`
for all benchmark rows. FTS5 treats NULL as an empty string for BM25 purposes;
the `title` column contributes 0 BM25 signal for titleless documents. The
NFCorpus NDCG@10 result (0.31–0.33) is unaffected by this change.

---

## Consequences

- Documents stored without a title get byte-identical FTS and vector behaviour
  to before this change. Callers that do not pass `title` are fully unaffected.
- Documents with a non-nil `title` now receive a BM25 boost proportional to
  `ftsTitleWeight` (default 3×) for title-token matches, independent of body
  content.
- First time a pre-108 database is opened via `SQLiteStorage`, the migration
  runs automatically. For large corpora this may add seconds to `open()`.
- The FTS5 column order (`title`, `body`) is now a codebase invariant. Any
  future change to column order must update all `bm25()` call sites.
- `InMemoryStorage` scoring is a weighted token-overlap approximation; its
  scores are not comparable to SQLite BM25 scores.

---

## References

- ADR 008 (`008-hybrid-fusion.md`) — `HybridConfig` and RRF fusion
- ADR 019 (`019-sqlite-writer-reader-split.md`) — writer/reader actor split
- ADR 025 (`025-title-prepend-for-query-alignment.md`) — title-prepend at
  embedding time; section (d) deferred this work; superseded for FTS guidance
- Issue #108 — bartleby failure case, diagnostic data, requirements
- Issue #105 — original alfred/bartleby ranking inversion
