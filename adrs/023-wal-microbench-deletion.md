# ADR 023 — Delete WAL Concurrency Microbench (`performanceAssertionTest`)

**Status:** Accepted
**Date:** 2026-05-07
**Issue:** [#96](https://github.com/totalslacker/switchcraft/issues/96)

## Context

`SQLiteStorageConcurrencyTests.performanceAssertionTest()` was introduced alongside the writer/reader actor split (ADR 019) to validate that concurrent reads and writes complete faster than their serial sum. The assertion was:

```
tBoth < tRead + tWrite × 0.7
```

where `tBoth` is the wall-clock time for an `async let`-launched FTS read run concurrently with 100 sequential writes (median of 3 iterations), and `tRead`/`tWrite` are isolated medians of the same operations.

The test was disabled on CI via `@Test(.disabled(if: ProcessInfo.processInfo.environment["CI"] != nil, "…"))` — the Swift Testing `.disabled(if:)` trait gated on whether the `CI` environment variable is set (which GitHub Actions sets automatically). That skip annotation was a policy violation; this ADR documents the decision to delete the test instead.

### Root Cause Analysis

Three independent root causes make the speedup premise unmeasurable on shared CI runners — and partially unmeasurable even locally:

**Root Cause 1 — No `.serialized` on the suite.** `@Suite("SQLiteStorage WAL Concurrency")` has no `.serialized` trait. Swift Testing runs all tests in the suite concurrently by default. All three sibling tests (`livenessTest`, `performanceAssertionTest`, `safariUnfuckerRegressionTest`) each seed a 5,000-document SQLite database at startup. Running them simultaneously creates APFS I/O and WAL journal contention that inflates `tBoth`. Local measurements confirmed: the test passes 8/8 runs in isolation but fails ~50% when run alongside its siblings.

**Root Cause 2 — Swift cooperative scheduler on 3-core CI runners.** The `async let` + sequential writes pattern does not guarantee true parallelism. The cooperative thread pool is under no obligation to schedule the FTS task and the write loop on separate threads simultaneously. On a 3-core runner under full test-suite load, the scheduler frequently serializes them:

```
tBoth ≈ tRead + tWrite + scheduling_overhead
```

which trivially exceeds the threshold. This is the fundamental reason CI sees failures even when the underlying WAL concurrency is working correctly.

**Root Cause 3 — 3-iteration median is noise-sensitive.** ADR 012 documents that ≥50 iterations are needed for p50 stability in timing-based tests. Using a median of 3 means a single outlier iteration can shift the result significantly, especially when the signal (concurrency speedup) is the same order of magnitude as the noise (scheduler jitter on a loaded runner).

### Why Rewriting Is Not Viable

Addressing Root Cause 1 (add `.serialized`) and Root Cause 3 (raise to 11+ iterations) still cannot fix Root Cause 2. On a 3-core cooperative-scheduler runner, `async let` + sequential actor calls may serialize regardless of sample count. No timing threshold survives serialization: when the scheduler elects not to parallelize the tasks, `tBoth ≈ tRead + tWrite`, which exceeds any threshold less than 1.0× (no speedup).

The spec explicitly states: "If no rewrite produces a stable signal on macOS GitHub Actions runners, deletion is mandatory." Deletion is mandatory.

### "Keep But Skip" Is Prohibited

Project policy (CLAUDE.md) explicitly forbids `.disabled(if: CI)`, `XCTSkipIf`, `XCTSkipUnless`, `#if !CI`, and equivalent annotations as a "fix" for timing-sensitive failures. The existing skip annotation was itself a policy violation. "Keep but skip" is not an acceptable resolution.

## Decision

Delete `performanceAssertionTest` and its preceding `MARK` comment from `SQLiteStorageConcurrencyTests.swift`.

The test is gone. It cannot be recovered from a CI skip or a weaker assertion.

## CI Coverage After Deletion

The actor-serialization regression that `performanceAssertionTest` was meant to catch — concurrent search holding the storage actor and blocking indexer writes — is covered by `safariUnfuckerRegressionTest`, which remains in the suite and runs on every CI push.

`safariUnfuckerRegressionTest` verifies that 50 sequential writes are not stalled by a concurrently-running FTS scan, using a 1.5× overhead ceiling over an isolated write baseline. This directly tests the stall scenario from the original SafariUnfucker bug (1,178/6,511 chunks indexed when a search froze the indexer).

**Caveat:** `safariUnfuckerRegressionTest` itself uses a timing-ratio assertion (`tConcurrentWrites < tIsolated * 1.5`) and has observed failures at approximately 13% rate in isolation under local scheduler jitter. This is a pre-existing reliability limitation of the test design (not introduced by this PR). A dedicated issue should evaluate whether to redesign this test's assertion to avoid timing ratios entirely (e.g., verify correct result counts and write completion order rather than wall-clock ratios). Until then, `safariUnfuckerRegressionTest` is the best available CI gate for the actor-serialization regression.

`livenessTest` was also CI-disabled at the time of this deletion (under a separate issue), but was subsequently fixed (commit `814f8e3`) to use `ContinuousClock` instead of `Date()`, which resolved its timing-precision failure mode. `livenessTest` now runs on CI and provides a complementary assertion: it verifies that a single write completes while a slow FTS scan is in flight (liveness), whereas `safariUnfuckerRegressionTest` verifies that bulk writes are not stalled (throughput). Together they provide the coverage pair originally described in the issue spec.

## Consequences

- One fewer test in the WAL concurrency suite.
- The `.disabled(if: CI)` annotation on `performanceAssertionTest` is gone (the function is deleted).
- The speedup metric (`tBoth < tRead + tWrite × 0.7`) is no longer actively measured anywhere. This is acceptable: the split architecture is validated structurally by ADR 019, the stall regression is what matters to users, and `safariUnfuckerRegressionTest` + `livenessTest` together catch regressions in that behavior.
- The WAL writer/reader split now has two active CI gates: `livenessTest` (liveness, ContinuousClock-based, robust) and `safariUnfuckerRegressionTest` (throughput, timing-ratio-based, known ~13% local flakiness from `iterations: 1` baseline). A follow-on issue should evaluate redesigning `safariUnfuckerRegressionTest`'s assertion to avoid timing ratios.
