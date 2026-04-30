# ADR 012 — Performance regression thresholds

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #26 (Phase 1 testing: performance regression floor)

This ADR records the threshold values, corpus configuration, and runner
assumptions for `Tests/SwitchcraftTests/Performance/PerformanceTests.swift`.
The suite is a **regression floor**, not a production target — it exists
to catch obvious regressions before Phase 2 optimisations (Metal
kernels, parallel bucket search) silently degrade search latency,
indexing throughput, or memory. The thresholds are deliberately loose so
they remain stable on the macos-15 GitHub Actions runner.

Cite this ADR before tightening any threshold or changing the corpus
configuration.

---

## (a) Thresholds

| Assertion              | Floor          | Observed (M1 Ultra, release) | Headroom |
|------------------------|----------------|-------------------------------|----------|
| Search p50 latency     | < 300 ms       | ~126 ms                       | ~2.4×    |
| Search p95 latency     | < 500 ms       | ~137 ms                       | ~3.6×    |
| Indexing throughput    | > 10 docs/s    | ~4,600 docs/s                 | ~460×    |
| Peak RSS during search | < 300 MB delta | well under 300 MB             | comfortable |

The latency floors **deviate from the spec's original 50 / 100 ms
numbers**. The spec arrived at those by treating them as 3–5× looser
than the production target in `docs/Plan.md` Performance Expectations
(p95 ≈ 25–35 ms). That production target is for the **upstream
Witchcraft Rust implementation on Apple M2 Max running NFCorpus** — not
for the current Swift port over a 5,000-doc `MockEmbedder(dims: 32)`
corpus. Empirically, a single hybrid search on this corpus takes
~126 ms p50 / ~137 ms p95 on M1 Ultra in release mode, so the spec's
floor would fail unconditionally. The thresholds adopted here (300 /
500 ms) are 2–4× the observed values — tight enough to catch a 2× cost
regression, loose enough to absorb scheduler hiccups on shared CI
runners.

The throughput and memory floors stay loose by the same intent the spec
expressed: a 460× headroom on throughput and a generous 300 MB RSS
ceiling are not regression detectors for small slowdowns; they catch
order-of-magnitude regressions and cliff edges (e.g. forgetting to free
a per-search buffer).

## (b) Runner assumptions

CI runs on `macos-15` GitHub Actions runners (~7 GB RAM, 3 cores). The
release-mode CI job in `.github/workflows/ci.yml` is the authoritative
perf gate. The debug-mode job **skips this suite entirely** via an
`assert`-based runtime detection (`isReleaseBuild`) wired through
Swift Testing's `.enabled(if:)` trait — debug-mode latency is too noisy
on shared CI runners to defend a meaningful floor against. Initial
calibration tried debug-mode floors (p50 < 300 ms; observed ~241 ms on
an M1 Ultra dev box), but the macos-15 CI runner came in at ~326 ms
p50 and the test failed; rather than loosening the floor 3–4× to
absorb debug-mode noise (which would defeat the regression-detection
purpose), the suite is gated to release where the floor remains
meaningful.

## (c) Corpus tuning

The shared fixture is **5,000 documents × ~10 whitespace-separated
tokens × `MockEmbedder(dims: 32)`**. The hard cap in the test header is
≤ 50 tokens per document; the chosen body produces ~10. Worst-case
in-memory float ledger:

```
5_000 docs × 50 tokens × 32 dims × 4 bytes  = 32_000_000 bytes ≈ 32 MB
```

A naive Witchcraft-shaped corpus of 5,000 docs × 700 tokens × 128 dims
would be ~1.8 GB of floats — well over the 300 MB ceiling. The reduced
parameters chosen here keep the in-memory ledger ~10× under the ceiling
even after `DocumentRecord` / `ChunkRecord` overhead, the BM25 index,
and the LSM generations produced by `IndexerConfig.production`
(`l0Capacity: 1024`, `lsmFanout: 16`). Reducing further would risk the
search engine taking trivially few centroid/bucket lookups per query
and the suite becoming insensitive to regressions; the chosen size
exercises the production cascade thresholds end-to-end.

`MockEmbedder(dims: 32)` was chosen to match the precedent in
`ConcurrencyTests`. Using a smaller `dims` would still exercise the
storage and search paths but would shrink the ledger arithmetic enough
to lose meaning as a memory floor.

## (d) Why `InMemoryStorage` (and `MockEmbedder`)

- **`InMemoryStorage`** is used so disk I/O does not confound latency
  or throughput measurements. SQLite wall-clock cost varies with the
  runner's filesystem (APFS journal flushes, free-space fragmentation,
  background `mds` activity); none of those are what this suite is
  trying to detect.
- **`MockEmbedder`** is used so the suite has no dependency on model
  assets and produces deterministic timings. Real `T5CoreMLEmbedder`
  inference cost is ~30–50 ms per document (per `docs/Plan.md`); a
  5,000-doc add cycle would take minutes on top of indexing, which
  inflates suite runtime without measuring anything new.

The trade-off is that the throughput number is **not** a measurement
of embedder speed — it measures the indexing pipeline (chunk dedup,
`storage.upsertChunk`, `indexer.add`, LSM cascades, `storage.upsertDocument`,
final `index()`). When real-model performance becomes a regression
target, a separate suite using `T5CoreMLEmbedder` and the gated
`CoreMLAsset` infrastructure would be the right vehicle.

## (e) Policy for tightening

The thresholds above are deliberately conservative to avoid flapping
before any CI history exists. The tightening policy:

1. **Collect CI history** for at least 20 runs in both debug and
   release modes. Record the observed p50, p95, throughput, and RSS
   delta per run.
2. **Set new floors at 1.5–2× the observed CI 95th percentile** of
   each metric. For example, if release-mode p95 latency is observed
   to sit at 50 ms ± 10 ms across 20 runs, the new p95 floor would be
   ~150 ms — still loose enough to absorb scheduler hiccups, tight
   enough to catch a 2–3× regression.
3. **Increase iterations before loosening.** If a percentile measurement
   flaps near the floor, the first response is to raise the iteration
   count (50 → 100 or 200) so the percentile reading is more stable.
   Loosening the threshold further is a last resort and requires this
   ADR to be amended.
4. **Sanitizer-aware adjustment.** If the test target ever runs under
   asan/ubsan in CI, the 300 MB peak-RSS ceiling will need a
   sanitizer-specific adjustment (asan typically inflates RSS by
   ~2–3×). The current ceiling assumes no sanitizers, which matches
   `.github/workflows/ci.yml` today.

## (f) Methodology details

- **Measurement clock**: `ContinuousClock` via `clock.now - start`. Not
  affected by wall-clock adjustments. Resolution is well below 1 ms on
  macOS.
- **Warm-up**: one discarded `search` call before the timed loop. The
  first call after `index()` may include lazy initialisation (cold
  caches, first BM25 lookup); discarding it gives a cleaner percentile
  reading without inflating the iteration count.
- **Iterations**: 50 successive `search` calls per the spec. The 25th
  ranked sample is the p50; the 47th (`floor(0.95 × 50)`) is the p95.
- **RSS sampling**: `task_info(mach_task_self_, MACH_TASK_BASIC_INFO,
  ...)` reading `mach_task_basic_info.resident_size`. No special
  entitlements required. Returns 0 on rare failure so an assertion
  fails rather than the test crashing. We sample baseline before the
  first timed search and the peak across the loop, then assert
  `peak − baseline < 300 MB`.
- **Suite isolation**: `@Suite(.serialized)` so the three test
  functions don't time each other. The 5,000-doc corpus is built once
  via a `static let buildOnce: Task<Fixture, Error>` and shared between
  the latency and memory tests; the throughput test builds its own
  corpus inline because it must measure the build.
- **Release-mode gating**: the suite is annotated
  `.enabled(if: isReleaseBuild, …)` and `isReleaseBuild` is computed
  via the standard `assert`-based runtime trick (closure side-effects
  visible only when assertions are enabled). Under `swift test`
  (debug) all three tests skip with a clear reason; under
  `swift test -c release` they execute. This eliminates ~3 minutes of
  noisy debug-mode runtime per CI run and removes the false-positive
  failure mode where debug latency on a shared runner exceeds a floor
  calibrated against release-mode numbers.

---

## Status & out-of-scope

- **Continuous benchmarking** — long-term performance dashboards are a
  Phase 2 concern. This ADR scopes only the static floor.
- **CI enforcement of release mode** — a follow-up CI config issue
  will decide whether the existing `swift test -c release` job is the
  authoritative perf gate or whether a dedicated, less-loaded job
  should run only the perf suite. Out of scope here.
- **Real-model performance suite** — separate work (would use
  `T5CoreMLEmbedder` behind the `CoreMLAsset` gate).
