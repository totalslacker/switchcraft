# ADR 041 — Two-Phase Centroid Scan (Scan + Fetch)

## Status

Accepted

## Context

`totalslacker/switchcraft#151`, itself downstream of `totalslacker/WebBrain#344`
("Switchcraft search consistently exceeds 5s Release deadline on typical
queries"), found that `SearchEngine.searchOptimized` and `SearchEngine.searchLegacy`
each began the centroid scan by calling `storage.buckets(forGeneration:)` for
every generation and reading only two fields per returned `BucketRecord`:
`center` (needed — it's the matrix `cblas_sgemm` scores the query against)
and `residuals.count` (the blob's *length*, never its bytes). `indices` was
never touched at scan time at all. Both `indices` and `residuals` are needed
later, but only for the small subset of buckets selection actually picks —
typically well under 5% of scanned buckets on a realistic-scale index (see
the issue's measured live-index numbers: 72,773 buckets scanned per search,
797 MiB copied out of SQLite, ~35.5 MiB of which — the `center` data — was
actually used at that point).

`SQLiteStorage`'s (and `SQLiteReaderActor`'s) `buckets(forGeneration:)`
fetched all three columns (`center`, `indices`, `residuals`) eagerly via a
single `SELECT ... ORDER BY id ASC`, materializing every blob into a Swift
`Data` before either search path inspected anything. This issue splits that
single read into a lightweight scan phase and a targeted fetch phase.

This is a direct sequel to ADR 035 (#140, parallel bucket decode + query-token
dedup) — same two search paths, same "both paths must move together" oracle
coupling (`SearchDeterminismTests` uses `searchLegacy` as `searchOptimized`'s
equivalence oracle), but targeting I/O materialization waste rather than
compute/decode cost. It changes only what is read during the scan phase; the
ADR 035 parallel-decode machinery (`decodeSelectedBuckets`/`decodeBucketChunk`)
is untouched.

## Decision

### 1. A new scan-shaped record type and two new required protocol methods

`BucketScanRecord` (`Sources/SwitchcraftCore/Storage/Records.swift`) carries
`id`, `generationID`, `center: Data`, and `residualByteCount: Int` — the raw
byte length of the `residuals` blob, *not* a pre-divided token count. The
storage layer does not track `dims` per bucket or generation; `SearchEngine`
already divides `bucket.residuals.count / (dims / 2)` today (it receives
`dims` as a caller parameter) and keeps doing the same arithmetic against
`residualByteCount` instead — no behavior change to that calculation.

`SwitchcraftStorage` gains two required (not defaulted) protocol methods:

- `scanBuckets(forGeneration:) async throws -> [BucketScanRecord]` — the scan
  phase. Ordered `id ASC`, identically to `buckets(forGeneration:)`, because
  that order feeds `cblas_sgemm`'s summation order and is load-bearing for
  determinism (ADR 006(f), ADR 035).
- `buckets(ids: [Int64]) async throws -> [BucketRecord]` — the fetch phase.
  Returns full records for exactly the requested ids; return order is
  unspecified — callers needing a specific order reconstruct it themselves.

These were made required, not defaulted via the `extension SwitchcraftStorage`
opt-out pattern used elsewhere in this protocol (e.g. `saveLedgerSnapshot`'s
"never incorrect, just slower" default no-op). A default no-op doesn't make
sense for a correctness-critical read path — `SearchEngine`'s centroid scan
has no fallback behavior if a backend silently returned nothing.

### 2. `SQLiteStorage`/`SQLiteReaderActor`: `length(residuals)`, not a
   truncated full fetch

`scanBuckets`'s SQL is:

```sql
SELECT id, generation_id, center, length(residuals)
FROM bucket WHERE generation_id = ? ORDER BY id ASC
```

`length()` on a SQLite blob column reads the row's length header, not the
payload — this is the entire fix. The issue's own text calls out the failure
mode this guards against explicitly: "the change described in this issue is
meaningless if the SQLite implementation still fetches the bytes internally
and merely truncates the Swift-side struct." `BucketScanBytesReadTests`
(below) is the regression test for exactly this failure mode.

`buckets(ids:)` batches its `WHERE id IN (...)` fetch in chunks of 500
(`SQLiteReaderActor.bucketFetchBatchSize`) to stay well under SQLite's
`SQLITE_LIMIT_VARIABLE_NUMBER` (32766 in modern builds, as low as 999 on
older ones) across platforms — `selectedIDs` size is bounded in practice
(`k` centroids × generations × query tokens, deduplicated) but not provably
so, and this codebase doesn't control the exact SQLite build's limit.

Both of `SQLiteStorage`'s internal modes (`.inMemory` direct-SQL,
`.fileBacked` via `SQLiteReaderActor`, per ADR 019) implement both methods
independently, matching the existing shape of `buckets(forGeneration:)`.

### 3. `InMemoryStorage.buckets(ids:)`: linear scan, no new index

`InMemoryStorage` has no I/O cost to save (storage design tenet) — its
`buckets(ids:)` is a linear scan across `bucketsByGeneration.values` filtered
by a `Set` of requested ids, rather than a new `bucketsByID` dictionary
mirroring the existing `chunksByHash`/`chunksByID` dual-index pattern. A
second dict would need consistent bookkeeping across five existing mutation
sites (`insertBucket`, `deleteGeneration`, `replaceGeneration`,
`applyVacuumPlan`'s three sub-cases, `clear`) for zero production benefit —
`InMemoryStorage` is test/benchmark-only. `InMemoryStorageBucketsByIDsRegressionTests`
covers the three bucket-mutating paths (`deleteGeneration`, `replaceGeneration`,
`applyVacuumPlan`) explicitly.

### 4. `SearchEngine`: `selected: [BucketRecord]` becomes `selectedIDs: [Int64]`
   during scan, one fetch after

Both `searchLegacy` and `searchOptimized`'s scan loops now call
`storage.scanBuckets(forGeneration:)` instead of `storage.buckets(forGeneration:)`,
and accumulate `selectedIDs: [Int64]` (same `seenBuckets` dedup logic,
unchanged) instead of full `BucketRecord`s. After the scan loop completes
across all generations, one `storage.buckets(ids: selectedIDs)` call fetches
full records for exactly the deduplicated selected set. An id → record
dictionary then reconstructs `selected: [BucketRecord]` in `selectedIDs`
order — **not** fetch-return order, since `buckets(ids:)`'s return order is
unspecified by the protocol. Reconstructing in scan order is what keeps
`decodeSelectedBuckets` (ADR 035, unchanged) producing byte-identical output
to the pre-split implementation.

Both search paths move together in the same commit, per the issue's explicit
requirement: `SearchDeterminismTests` uses `searchLegacy` as `searchOptimized`'s
equivalence oracle, so a partial migration would silently compare two
different code shapes for the wrong reason.

### 5. Missing ids during fetch are silently dropped, not an error

The two-phase split introduces a narrow new race window: a concurrent
vacuum/compaction between the scan call and the fetch-by-id call could delete
a bucket that was just scanned. `SQLiteReaderActor` holds one persistent
read-only connection with no transaction spanning multiple calls, so this is
possible in principle. This is directionally the *same* race class the code
already tolerated before this change (a generation can vanish between
`storage.generations()` and `storage.buckets(forGeneration:)` in the scan
loop, silently treated as empty) — so `buckets(ids:)` returning fewer records
than requested is handled the same way: `selectedIDs.compactMap { byID[$0] }`,
no new `SearchEngine.Error` case. If every selected id vanished, `selected`
ends up empty and the function returns `[]`, exactly like the pre-existing
"no candidate buckets" path.

### 6. The new bytes-read benchmark assertion is timing-ratio, not a SQLite
   VFS byte-counting shim

No SQLite I/O-byte instrumentation exists anywhere in this codebase (no VFS
shim, no `sqlite3_status`/`sqlite3_stmt_status` usage) — building one is
disproportionate new infrastructure for this issue. `BucketScanBytesReadTests`
(`Tests/SwitchcraftTests/SearchLatencyBenchmark/BucketScanBytesReadTests.swift`)
instead builds two on-disk `SQLiteStorage` generations with equal bucket
counts (300 each) but a 200x per-bucket residual-size difference, and
compares median elapsed time (7 trials) for `scanBuckets` vs.
`buckets(forGeneration:)` across the two generations. The full-fetch ratio
(large/small) is asserted to exceed 3x — a **control** proving the
methodology can actually detect a residual-size effect at all — and the
scan-phase ratio is asserted to stay both under 3x and strictly less than the
full-fetch ratio.

Unlike `SearchLatencyBenchmarkTests` (gated behind
`SWITCHCRAFT_SEARCH_LATENCY_BENCHMARK=1` because its fixture is hundreds of
thousands of 768-dim embeddings), this fixture is ~600 buckets and a few MB
of residual bytes total — cheap enough (under 0.1s observed) to run
unconditionally on every `swift test`. The issue's own risk section notes
this assertion is "the only thing that will actually catch a regression of
this specific waste going forward"; gating it behind an opt-in env var most
contributors never set would defeat that purpose.

## Consequences

- **`SwitchcraftStorage` protocol surface grows by two required methods.**
  Any third-party backend conforming to this protocol must implement
  `scanBuckets(forGeneration:)` and `buckets(ids:)` — this is a
  source-breaking protocol change for external conformers, consistent with
  the storage design tenet that new/changed protocol surface must work
  identically across backends. No default no-op was viable (see Decision 1).
- **No behavior change to selection, scoring, or decode.** `selectedIDs`
  dedup logic, `cblas_sgemm` summation order, and `decodeSelectedBuckets`
  are all unchanged; only what's read during the scan phase changes. Output
  is bit-identical for identical inputs (verified by `SearchDeterminismTests`,
  unchanged).
- **Per-search-call read volume drops from O(all bucket bytes in scanned
  generations) to O(bucket count × (center size + 8 bytes)) for the scan
  phase, plus O(selected bucket bytes) for the fetch phase** — on the
  issue's measured live-index numbers, this is the difference between
  materializing 797 MiB and materializing roughly 35.5 MiB (`center`) plus
  whatever the selected ~5% of buckets' `indices`/`residuals` total.
  `SearchLatencyBenchmarkTests`' existing ratio benchmark improved
  incidentally (searchOptimized/searchLegacy speedup rose from the
  previously-measured range to 4.9x in Debug / 24x in Release on the same
  fixture) as a side effect of this change, though that benchmark's `>= 3x`
  gate was already passing before this issue and is not the primary
  evidence this ADR relies on — `BucketScanBytesReadTests` is.
- **New narrow race window** between scan and fetch (see Decision 5),
  tolerated the same way the pre-existing generations()/buckets(forGeneration:)
  race already was — no new error surface.
- **Out of scope / deferred**: the other WB#344-suspected contributors to
  the 5s timeout (flush/cascade compaction timing, CoreML embedder
  contention, `perSourceBudget` tuning) are unaffected by this change and
  remain tracked in WB#344, not here.
