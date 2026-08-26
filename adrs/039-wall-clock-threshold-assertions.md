# ADR 039 — Wall-Clock Threshold Assertions: Absolute Floors, `ContinuousClock`, and Multi-Sample Medians

## Status

Accepted

## Context

Issue #146 was filed after two full-suite `swift test` (debug) runs on a
shared, oversubscribed dev host (`load averages: 56.85 46.51 39.65` /
`hw.ncpu=20`, 16 concurrent logged-in users) each failed a *different*
wall-clock-threshold assertion that passed reliably in isolation:

- `SQLiteStorageConcurrencyTests.searchWriteStallRegressionTest` —
  `tConcurrentWrites < tIsolated * 1.5`, no absolute floor. A ~10-20ms
  baseline (`tIsolated`) makes a few milliseconds of scheduling jitter a
  large *relative* overshoot (observed: `0.030s` vs ceiling `0.028s`;
  `0.028s` vs ceiling `0.017s` on a second run).
- `IndexerTests.performanceSmoke` — `elapsed < 10.0` (debug). Observed
  `15.33s`, well above the `#97`-era post-fix median of `3.688s`.

Both tests used `Date()` for timing (subject to wall-clock/NTP
adjustments — `livenessTest` in the same suite was already fixed for this
in `#95`, but the fix wasn't applied to its neighbor). Investigation
(issue #146 Research stage) additionally found:

- `searchWriteStallRegressionTest`'s `tIsolated` was computed via
  `measureMedian(iterations: 1)` — a single sample mislabeled as a
  median, and a real, host-load-independent source of ceiling noise: the
  issue's own two data points show the ceiling itself swinging from
  `0.028s` to `0.017s` across runs, consistent with `tIsolated` jitter,
  not necessarily `tConcurrentWrites` degrading.
- `performanceSmoke`'s inline comment justified its 10s debug limit with
  "Per ADR 012, CI (macOS-15) is ~1.35× slower than local." That figure
  is real but comes from ADR 012's *release-mode search-latency*
  calibration for a wholly different suite
  (`Tests/SwitchcraftTests/Performance/PerformanceTests.swift`, gated out
  entirely in debug) — a different workload profile (search latency vs.
  an async `add()`-loop), different hardware baseline, and conflates
  CI-vs-local variance with the actual failure mode here (host
  oversubscription during a full-suite *local* run, not CI-runner
  slowdown). The citation was inaccurate and has been removed.
- Swift Testing parallelizes test execution by default, across suites and
  across `@Test` functions within a suite, unless a suite is
  `.serialized` or `swift test --no-parallel` is passed. Neither this
  repo's CI invocation nor most local runs pass `--no-parallel`. This
  means every full-suite run — CI or local — races these assertions
  against dozens of concurrently scheduled tests, independent of any
  *other* process on the host. This explains why `--filter`-isolated
  re-runs of either test pass reliably while full-suite runs
  intermittently don't: isolated runs have no intra-process contention,
  full-suite runs always do. (This ADR does not change that default —
  see "Out of scope" below.)

Two existing, un-generalized precedents for the fix shape already existed
in-repo:

- `IndexerConflictRecoveryTests.recoveryBatch_compactionBoundary_withinTwoXBaseline`
  — `ratio <= 2.0 || crossingElapsed < 2.0`, an OR'd ratio/absolute-ceiling
  design "so the test isn't fragile on very fast HW."
- `LazyLedgerBenchmarkTests` — debug asserts a ratio (proxy for algorithmic
  complexity under unvectorized debug arithmetic), release asserts an
  absolute ceiling derived from one measured value.
- ADR 012 §(e) itself documents a "tighten only with evidence, increase
  iterations before loosening" policy for a different suite.

## Decision

For wall-clock threshold assertions in this codebase — and as a pattern
for any new ones — apply, as warranted by the specific test's scale and
failure mode:

1. **`ContinuousClock`, never `Date()`, for any interval being compared
   against a threshold.** `Date()` is subject to wall-clock/NTP
   adjustments; `ContinuousClock` is monotonic and immune to them. This
   doesn't reduce contention-driven overshoot magnitude (the extra wall
   time under real CPU contention is real work, not a measurement
   artifact) — it only removes a distinct, unrelated noise source.
   Convert `ContinuousClock.Duration` to fractional seconds via its
   `(seconds, attoseconds)` components (see `PerformanceTests
   .nanoseconds(_:)` for the nanosecond-precision precedent this ADR's
   fixes mirror at second precision).
2. **An absolute floor alongside any ratio ceiling, for thresholds whose
   baseline can be small** (here: tens of milliseconds). Shape:
   `ceiling = max(baseline * ratio, baseline + fixedSlack)`. This is the
   `max()`-of-two-terms variant of `IndexerConflictRecoveryTests`' OR'd
   `ratio <= X || absolute < Y` shape — both express "don't let a tiny
   baseline turn ordinary scheduling jitter into a large relative
   failure," and either shape is acceptable for future use; pick
   whichever reads more naturally at the call site.
3. **Multi-sample medians, not single-sample "medians," for any baseline
   measurement feeding into (2).** A `measureMedian(iterations: 1)`-style
   single sample is not a median and is itself a real noise source,
   independent of host load. Per ADR 012 §(e)'s existing policy,
   increasing iteration count is the first lever to pull before touching
   threshold math.
4. **Fresh measurement evidence, never a guessed constant**, for any
   floor/ratio/limit value — see the two Fix sites below for the
   methodology and raw data this ADR's changes are based on.

### Fix 1 — `IndexerTests.performanceSmoke`

Timing switched from `Date()` to `ContinuousClock`. The inaccurate "ADR
012, ~1.35× CI-runner factor" comment was removed and replaced with a
citation to this ADR. **The 10s (debug) / 5s (release) limits were left
unchanged** — re-measurement (below) showed the `#97`-era `vDSP_maxvi`
fix headroom holds; no code regression was found, so no threshold change
or `performFlush()` root-causing was warranted.

**Methodology**: 14 consecutive **full-suite** `swift test --skip-build`
(debug) runs (not `--filter` — isolated runs cannot reproduce Swift
Testing's default intra-process parallel scheduling, the actual
contention source per the Context section above), on a host under
sustained heavy load throughout (`uptime` load averages ranged
`36.53–55.72` / `46.44–74.34` / `55.77–74.34` across the 1/5/15-minute
windows, all against `hw.ncpu=20` — i.e. ~2–3.7× oversubscribed for the
whole run, comparable to or worse than the original incident's
`56.85 46.51 39.65`, with 16 users logged in throughout).

| run | elapsed (s) | run | elapsed (s) |
|-----|-------------|-----|-------------|
| 1   | 6.180       | 8   | 4.585       |
| 2   | 3.667       | 9   | 4.086       |
| 3   | 6.513       | 10  | 3.338       |
| 4   | 3.745       | 11  | 3.079       |
| 5   | 5.531       | 12  | 2.649       |
| 6   | 4.455       | 13  | 2.461       |
| 7   | 4.403       | 14  | 2.579       |

- **14/14 passed** (all `< 10.0s`); max observed `6.513s` — ~35% margin
  below the limit even under sustained heavy load.
- Median (average of the two middle values, n=14): **3.916s**. p95
  (nearest-rank): **6.513s** (the observed max, at this sample size).
- Compared to `#97`'s post-fix baseline (median `3.688s`, all-pass at
  `<10s`): the new median is **+6%**, within run-to-run noise — no
  meaningful drift since `#97`, despite `#137`'s substantial rewrite of
  `performFlush()`'s materialization path landing in between. The
  `vDSP_maxvi` argmax fix `#97` shipped is confirmed still in place
  (`KMeans.assignAll`).
- **Conclusion**: the existing 10s/5s budget is sound with real headroom;
  the issue's originally observed `15.33s` overshoot is attributable to
  host oversubscription beyond this specific measurement window, not a
  code regression. No threshold change made.

### Fix 2 — `SQLiteStorageConcurrencyTests.searchWriteStallRegressionTest`

- `measureMedian`'s default `iterations` raised from `1` to `5` (an
  actual median now, not a mislabeled single sample) — used for
  `tIsolated`.
- Both `tIsolated` and `tConcurrentWrites` timing switched from `Date()`
  to `ContinuousClock`.
- Ceiling changed from `tIsolated * 1.5` (no floor) to
  `max(tIsolated * 1.5, tIsolated + 0.030)`.

**Deriving the 30ms floor**: across the same 14 full-suite runs used for
Fix 1 (same host-load conditions), `tIsolated` ranged `0.0053s–0.0216s`
and `tConcurrentWrites` ranged `0.0065s–0.0204s` — i.e. even under
sustained heavy contention, the concurrent run's measured overhead over
isolated never exceeded a few milliseconds at this corpus scale (50
writes + a 5,000-doc FTS scan). The ratio term (`tIsolated * 1.5`) never
bound in any of the 14 runs — the floor term was what actually gated the
ceiling in every case, confirming the original bug report's diagnosis
that the ratio alone is the wrong tool at this baseline scale.

Applying the new design to the two originally-reported failures
retroactively (reconstructing `tIsolated` from each run's reported
ceiling `= tIsolated_old * 1.5`):

| run | old tIsolated (1 sample) | old tConcurrentWrites | old ceiling (×1.5, no floor) | new ceiling (`max(×1.5, +30ms)`) | passes new design? |
|-----|--------------------------|------------------------|-------------------------------|-----------------------------------|---------------------|
| A   | ~0.01867s                | 0.030s                 | 0.028s (**fail**, 0.030 ≥ 0.028) | 0.04867s | ✅ 0.030 < 0.04867 |
| B   | ~0.01133s                | 0.028s                 | 0.017s (**fail**, 0.028 ≥ 0.017) | 0.04133s | ✅ 0.028 < 0.04133 |

Both original failures pass comfortably under the new design.

**Confirmation**: same 14 consecutive full-suite runs as Fix 1 (single
combined measurement pass; both fixes were live simultaneously) —
`searchWriteStallRegressionTest` **14/14 passed**. Worst-case margin
(`ceiling - tConcurrentWrites`) across all 14 runs was `0.0231s` (run 1:
ceiling `0.0402s` − `tConcurrentWrites` `0.0171s`); every other run's
margin was `≥ 0.028s`. No case came close to the new ceiling.

## Related, out-of-scope finding

Across the same 14-run set, `ConcurrencyTests` (R1–R4, a different suite
entirely) recorded `.searchTimedOut(elapsed: ~5.0–5.5s)` failures against
its 5s deadline in **2 of 14 runs** (run 3: all four of R1–R4 failed in
the same run; run 9: R3 alone). This is the same failure *class* (a
wall-clock threshold with insufficient margin against Swift Testing's
intra-process contention under host load) as this ADR's two fixes, and
recurs more clearly here than the single R3 observation issue #146's own
Research stage had available. Per issue #146's explicit scope boundary,
this is **not** fixed by this ADR — `ConcurrencyTests` was called out as
out of scope pending evidence of recurrence, and that evidence now
exists. **Recommendation: file a follow-up issue** applying this ADR's
methodology (`ContinuousClock` if not already used, evidence-based
floor/deadline re-validation) to `ConcurrencyTests`' `searchDeadline`
usage.

## Alternatives considered

- **Loosen `performanceSmoke`'s limit without measurement** — rejected
  per issue #146's explicit requirement and this project's standing
  policy: threshold loosening requires measurement evidence, and the
  evidence here showed no loosening was needed.
- **Disable Swift Testing's default intra-process parallelism**
  (`swift test --no-parallel` repo-wide, or per-suite `.serialized` on
  every suite) — this is the single largest lever available (see
  Context) and would likely eliminate this whole failure class, but its
  blast radius is all 274+ tests' CI/local runtime, not just the two
  sites in scope for issue #146. Documented here as a known structural
  factor and a candidate for a dedicated follow-up issue, not adopted by
  this ADR.
- **A test-skip mechanism** (`.disabled(if: CI)`, `XCTSkipIf`, `#if !CI`,
  runtime `isCI` checks) — forbidden by standing project policy
  regardless of evidence; not considered.

## Consequences

- `IndexerTests.performanceSmoke`: `ContinuousClock` timing, corrected
  comment citing this ADR; no threshold change.
- `SQLiteStorageConcurrencyTests`: `measureMedian` default iterations
  1→5; `ContinuousClock` timing in both `livenessTest` (already fixed,
  `#95`) and now `searchWriteStallRegressionTest`; absolute-floor term
  added to the latter's ceiling.
- Diagnostic `print()` statements added to both tests (`[PerfSmoke]`,
  `[StallRegression]`), matching the existing `[PerfTest]` precedent in
  `IndexerConflictRecoveryTests` — visible in `swift test` output for
  future investigations without needing to re-instrument on demand.
- No `SwitchcraftStorage` protocol change, no public API change, no
  production-code change — both fixes are entirely test-file-scoped.
- `ConcurrencyTests`' `searchTimedOut` recurrence documented above as a
  recommended (not required) follow-up issue.

## Relationship to prior ADRs

- **ADR 012** (performance regression thresholds, release-mode
  `PerformanceTests` suite only): this ADR's floor/ratio design and
  "measure before adjusting" methodology directly follow ADR 012 §(e)'s
  precedent, but for debug-mode tests in a different suite; ADR 012's
  own specific "~1.35×" CI-runner figure is *not* reused here (see
  Context — the prior misapplication of that figure is the bug this ADR
  fixes in `performanceSmoke`'s comment).
- **`#95`** (WAL liveness `ContinuousClock` fix): this ADR extends the
  same fix to `searchWriteStallRegressionTest`, the one test in
  `SQLiteStorageConcurrencyTests` that hadn't received it.
- **`#97`** (`performanceSmoke` `vDSP_maxvi` fix): this ADR's
  re-measurement confirms `#97`'s fix and methodology (≥11 runs, median +
  p95, before/after comparison) still hold; no new fix required at the
  `KMeans`/`performFlush()` level.

## Out of scope

- Disabling Swift Testing's default parallel test execution (see
  Alternatives).
- Any change to `ConcurrencyTests`' `searchDeadline`/timeout design (see
  "Related, out-of-scope finding").
- Any change to `IndexerConflictRecoveryTests` or `LazyLedgerBenchmarkTests`
  — both already implement a compliant variant of this pattern and needed
  no changes.
