# ADR 011 — Sliding-window strategy for long inputs

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #18 (Phase 1: T5/CoreML embedder)

This ADR records the policy `T5CoreMLEmbedder` uses to encode inputs
longer than the CoreML model's fixed sequence length, the overlap
merge rule, and the pre-normalisation L2 norm threshold that filters
low-signal positions out of the final embedding stream. Cite this ADR
before changing any of these constants — they are tuned for parity
with upstream Witchcraft (`src/embedder.rs`).

---

## (a) Window size = 512, stride = 256

The CoreML model exposes a **fixed** input shape `[1, 512]` (ADR 010).
This decision was made deliberately: dynamic shapes (`RangeDim`) work
in `coremltools` but have known edge-case bugs with multi-input
models, and `EnumeratedShapes` is fastest but limited. Fixed shape
gives the compiler the most opportunity to optimise on ANE.

The window stride is **`256`** tokens (half the window). Witchcraft
uses the same stride; halving the window size means each non-final
position is encoded by exactly two windows in the steady-state, which
gives the mean-merge enough redundancy to dilute window-edge effects
without blowing up the inference budget.

## (b) Last window left-shifted to `T - windowSize`

For `T > windowSize`, naive stride-only windowing leaves a tail with
fewer than 512 real tokens, which would be padded. Padding rows are
then encoded with low-signal embeddings whose pre-norm L2 norm is
~0.27 — they would be filtered by the `MIN_NORM` threshold anyway, so
encoding them is wasted work.

Instead, the **last window is shifted left** so it starts at
`T - windowSize` and ends exactly at `T`. Every position in `[0, T)` is
covered by at least one real (non-padding) window slot, and the tail
inference produces meaningful embeddings the merger can use.

This matches Witchcraft's behaviour. The implementation lives in
`SlidingWindow.plan(...)` in
`Sources/SwitchcraftCoreML/SlidingWindow.swift`; the planner is a
pure-Swift function with always-on unit tests
(`Tests/SwitchcraftTests/SlidingWindowTests.swift`) so the window math
can be exercised on every commit without requiring the CoreML asset.

For `T ≤ windowSize` the planner returns `[0]` — a single window
starting at offset 0, with positions beyond `T` filled by the pad
token. The norm filter strips those padding rows from the output.

## (c) Mean-merge with re-normalisation in overlaps

When two windows cover the same position, their per-token outputs are
**averaged** (mean-merge). This is the simplest policy that is
symmetric in window order, and it matches Witchcraft.

Averaging two unit vectors that point in slightly different directions
yields a vector shorter than 1, so the merged vector is
**re-normalised to unit length** before it leaves the merger. Callers
of `Embedder` consume unit vectors (cosine similarity is the search
engine's scoring primitive); emitting a non-unit vector would silently
distort scores.

The merger averages the **L2-normalised** vectors, not the raw
projections. Both choices preserve direction in the limit, but
averaging unit vectors keeps each window's contribution comparable
in magnitude — a window whose raw output happens to be larger does
not dominate just because of its norm.

## (d) Pre-normalisation L2 threshold = 1.0, applied post-merge

Each window also emits per-row pre-normalisation L2 norms (computed
from the `raw_projected` CoreML output in
`CoreMLModelIO.readRowL2Norms`). The merger averages these norms
across the windows that cover each position; positions whose merged
mean raw norm is below `1.0` are **dropped** from the output.

`1.0` is Witchcraft's `MIN_NORM` constant (`src/embedder.rs`). It is
empirically calibrated against XTR-base-en behaviour:

- Content tokens have raw norms ≥ ~5.0.
- `</s>` and pad tokens have raw norms ≈ 0.27.

Anything in between is rare. Dropping sub-threshold positions removes
the trailing `</s>` token (which has no semantic content) and any
incidental pad rows from the output stream, leaving a tighter `m × 128`
matrix whose every row contributes signal to the search engine.

The threshold is exposed at `T5CoreMLEmbedder.init` as `minNorm:
Float = 1.0`, so callers can tighten or loosen it without rebuilding
the model. Lowering it below `0.27` defeats the filter; raising it
much above `1.0` starts trimming weak content tokens.

The filter is applied **post-merge**, not per-window:

- Per-window filtering would risk dropping a position in window A
  whose other-window evidence (window B's same position with a higher
  raw norm) would have rescued it.
- Post-merge filtering uses the strongest available evidence for each
  position: the mean of every window's raw norm at that position.

This matches Witchcraft's evaluation order.

## (e) Determinism

Two `encode` calls with the same input on the same loaded
`T5CoreMLEmbedder` instance return float-equal output. This is the
contract the `Embedder` protocol requires (per ADR 009(b)) and it is
what the index and search engines rely on for reproducibility.

Determinism across compute-unit choices (`.all` vs `.cpuOnly`) is **not**
guaranteed — ANE and CPU may differ in last-bit float values. The
fixture-based parity test uses cosine similarity rather than exact
equality so it is robust to that drift.

## (f) Configuration parameters surfaced at init

`T5CoreMLEmbedder.init` exposes `windowSize`, `stride`, and `minNorm`
as parameters (with defaults `512`, `256`, `1.0`). The defaults are
the right values for the canonical XTR-base-en asset; non-default
values are useful only when:

- A future model has a different fixed sequence length.
- A benchmark wants to ablate the overlap policy (e.g. stride = 512 to
  disable overlap and measure the cost).
- A workload-specific corpus has unusually short or long content
  tokens whose raw norm distribution differs from the calibration set.

Surfacing them at the API boundary avoids hardcoding a future-fragile
constant inside the embedder. The pure-Swift `SlidingWindow` planner
takes the same parameters so its unit tests cover the parameterised
behaviour, not just the defaults.
