# ADR 040 — Per-Stage Search Timing Instrumentation

## Status

Accepted

## Context

A downstream consumer report ("Switchcraft search consistently exceeds 5s
Release deadline on typical queries") found a live operator repro: search
times out at the 5s deadline on typical queries while a sibling backend
completes the same query in 2–3s, despite both paying the same CoreML
embedder cost. That report's research stage inspected this package and
found **no per-stage timing exists** inside `SwitchcraftStore.search()` or
`SearchEngine.searchHybrid()` — so the parent investigation cannot tell
whether the excess time is spent in the entry-point flush (which, per ADR
032/004, can synchronously trigger an L0 cascade compaction once
`IndexerConfig.l0Capacity` is crossed — a pure-Switchcraft cost a read-only
vector search never pays elsewhere), the CoreML embedder call, or inside
`searchHybrid`'s XTR-Warp retrieval itself.

This issue is purely observability: it makes the existing cost structure
measurable so the parent investigation can attribute the 5s overrun to a
specific stage. It does not change retrieval behavior or the cascade
compaction policy (ADR 004, ADR 032) — fixing that cost, if warranted, is
a follow-up issue, not this one — and it does not add logging or alerting
(the consuming sub-issue, SemanticHistory, owns that decision).

Note: the original issue text attributed "no local mitigation [for the
cascade cost]" to ADR 039. This package's Research stage found that
citation to be a mismatch — ADR 039 is about fixing flaky wall-clock
*test assertions*, not the cascade compaction cost — so it is not repeated
here. The underlying technical claim (cascade fires synchronously inside
`flush()`, observable via `CompactionEvent.trigger`) is independently
correct regardless of the citation error.

## Decision

### 1. Wrapper-struct return types, not sibling methods

`SearchEngine.searchHybrid()` now returns `HybridSearchResult` (`hits:
[HybridHit]`, `timing: HybridSearchTiming`) instead of `[HybridHit]`
directly; `SwitchcraftStore.search()` now returns `SearchResult` (`hits:
[HybridHit]`, `timing: SearchTiming`) instead of `[HybridHit]` directly.
This was chosen over adding parallel `searchWithTiming()` methods because
the issue's own requirement 4 suggests this shape, and a single return
type keeps the timing data from silently drifting out of sync with a
duplicate code path. The cost is a mechanical one-line `.hits` edit at
~79 existing call sites (mostly test call sites); this is the accepted
cost of the issue's "keep additive... minimal edits" requirement, not zero
edits.

### 2. `ContinuousClock` pairs at each call site, not a generic wrapper

Every stage boundary this issue measures is a `Task` suspension point
across a different actor's isolation domain (`SwitchcraftStore` →
`Indexer`, `SwitchcraftStore` → `Embedder`, `SwitchcraftStore` →
`SearchEngine` → `SwitchcraftStorage`). A generic `measure { await ... }`
timing wrapper would itself need to cross the same actor boundaries as the
work it measures, so each stage instead brackets its own work with a
plain local `let start = ContinuousClock.now` / `ContinuousClock.now -
start` pair, in the same actor context that performs the `await`. This
matches the existing `SearchDeadlineContext` precedent for
`ContinuousClock`-over-`Date()` in this codebase (ADR 039's clock mandate
extends the same reasoning to interval measurement, not just deadline
comparison).

### 3. `cascadeCompactionDuration` = `flushDuration` when cascade occurred

`Indexer.performFlush()` has no internal boundary separating cascade-only
time from ordinary L0-write time within a single call — a manual flush
and a cascade flush share the same code path. Rather than add a `duration`
field to `CompactionEvent` (a public struct's shape change, plus new
`ContinuousClock` instrumentation inside `Indexer`), `SwitchcraftStore.search()`
measures the whole `flushAndClearPending()` call once and reports that
same duration as `cascadeCompactionDuration` whenever the returned
`CompactionEvent.trigger == .cascade`. This is numerically identical to
adding the field to `CompactionEvent` and requires zero production changes
to `Indexer`.

`flushAndClearPending()` itself changed from `private func
flushAndClearPending() async throws` to `@discardableResult private func
flushAndClearPending() async throws -> (event: CompactionEvent?, duration:
Duration)`, so its other 5 existing call sites (in `add`/`remove`/`clear`/
`walCheckpoint`/`shutdown`) needed no edits.

### 4. Partial timing travels on the thrown error, not a shared actor property

`SwitchcraftStoreError.searchTimedOut` gained a `partialTiming: SearchTiming`
field (`searchTimedOut(elapsed: Duration, partialTiming: SearchTiming)`),
populated from a **call-local `var timing`** built up inside
`SwitchcraftStore.search()` as each stage completes, and attached to every
`searchTimedOut` throw (the two direct pre-SQLite deadline checks, and the
translated `SearchEngine.Error.deadlineExceeded` catch).

A shared "last call's timing" actor property was rejected: `SwitchcraftStore`
permits reentrancy at `await` points, so two concurrent `search()` calls
racing against a shared mutable property could let one call's diagnostic
data silently overwrite another's before either reads it. A call-local
variable is race-free by construction under reentrancy and needs no extra
synchronization — it directly matches how `searchTimedOut` already carried
`elapsed` before this issue.

One error path this issue did not anticipate needed its own fix: `SQLiteStorage.translateIfInterrupt(_:)`
(`Sources/SwitchcraftSQLite/SQLiteStorage.swift`) throws
`SwitchcraftStoreError.searchTimedOut` directly when a SQLite progress-handler
interrupt fires, entirely inside the storage layer with no visibility into
`SwitchcraftStore.search()`'s accumulated timing. That call site now
passes an empty `SearchTiming()` — the one error path where full partial-timing
coverage genuinely isn't feasible, consistent with the issue's requirement
6 ("full coverage of every error path is not required").

### 5. `SearchEngine.Error.deadlineExceeded` left unchanged

The issue's requirement 6 scopes partial-timing to the `searchTimedOut`
path. By the time `SwitchcraftStore.search()` catches and translates
`SearchEngine.Error.deadlineExceeded` into `searchTimedOut`, its local
`timing` already has flush/cascade/embed durations captured — sufficient
for the stated primary use case (diagnosing 5s store-level timeouts)
without widening `SearchEngine`'s own public error surface.

### 6. All `SearchTiming` fields except `cascadeCompactionOccurred` are optional

The two earliest throw points in `search()` (the post-flush and post-embed
deadline checks) genuinely predate later stages. `nil` is the honest value
for a stage that has not run yet — a zero-duration sentinel would be
indistinguishable from "this stage ran and took no time," which is not
true and would mislead the parent investigation's diagnosis.
`HybridSearchTiming`'s three fields stay non-optional `Duration`, since
`searchHybrid()` always fully measures (or `.zero`-stamps, on its
empty-input fallback paths) all three stages before any successful return.

## Consequences

- **`SwitchcraftStore.search()` and `SearchEngine.searchHybrid()`'s return
  types changed** — source-breaking for external consumers pattern-matching
  the bare `[HybridHit]` return (a one-line `.hits` fix). In-package, ~79
  call sites (mostly tests) were updated mechanically in the same commit.
- **`SwitchcraftStoreError.searchTimedOut`'s arity changed** from
  `searchTimedOut(elapsed:)` to `searchTimedOut(elapsed:partialTiming:)` —
  source-breaking for any external consumer pattern-matching the old
  single-value case. This is the only race-free way to deliver requirement
  6 (see §4); consumers need a one-line `case .searchTimedOut(let elapsed, _):`
  fix.
- **No behavior change to retrieval, flush, or cascade logic.** This ADR
  is purely additive instrumentation, consistent with the issue's stated
  scope.
- **Out of scope / deferred**: fixing the cascade compaction cost itself
  (a possible future issue, not this one); threshold-based logging or
  alerting on slow stages (owned by the consuming SemanticHistory
  sub-issue).
