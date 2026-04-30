# SmoothQuant + FP16 feasibility on `google/xtr-base-en`

## Recommendation

**No-go.** SmoothQuant under both the spec-mandated forward-hook
formulation and the canonical LayerNorm-absorption formulation, swept
across α ∈ {0.3, 0.5, 0.7, 0.85} and a partial-smoothing carve-out,
fails to unblock FP16 conversion of `google/xtr-base-en` on
coremltools 7.2 — every smoothed `.mlpackage` produces all-zero
outputs at prediction time, yielding mean cosine similarity of 0.0
against the FP32 PyTorch reference. Per the existing
`docs/Plan.md` "If SmoothQuant fails" ladder, the applicable next
rung is **(1) INT8 weight-only quantisation** (~110 MB asset, FP32
compute, no ANE eligibility) — a size-only win that does not require
solving the FP16 path.

## Scope and methodology

The investigation answers the four questions identified in issue #43:

1. **Are encoder activation outliers concentrated in a small set of
   channels?** — SmoothQuant's prerequisite. **Yes**, in the FFN `wo`
   path; see "Outlier characterisation" below.
2. **What smoothing strength α works for T5-base?** — **None of the
   tested values produce a usable FP16 model.**
3. **Does smoothing in PyTorch survive coremltools' trace into FP16
   without re-introducing NaN?** — **Smoothing produces outputs that
   are not NaN-tagged but are uniformly zero**, which the existing
   `MIN_NORM`-of-1.0 sliding-window filter rejects on every position.
4. **Does the smoothed-then-converted model still meet the ≥0.999
   PyTorch cosine parity gate?** — **No.** Mean cosine = 0.0 on every
   attempted configuration.

Methodology is implemented in `scripts/investigate-smoothquant.py`.
High-level recipe:

- **Calibration corpus**: 3 of the 4 committed fixture inputs (the
  `whitespace` fixture is empty and skipped) + 12 inlined
  public-domain English passages = 15 items spanning 16 sliding
  windows of 512 tokens each. `--minimal-calibration` restricts to
  the spec-default fixtures only.
- **Profiling**: per-input-channel max-abs, mean-abs, and 99.9th
  percentile recorded at every encoder Linear via PyTorch
  forward-pre-hooks. The model is `T5DenseGatedActDense` (gated FFN:
  `wi_0` + `wi_1` → gelu → `wo`), not the ungated form the spec
  assumed, so the target set is **84 Linears** = 12 blocks × (q, k,
  v, o, wi_0, wi_1, wo).
- **SmoothQuant**: per-input-channel scales `s_j = max(|X_j|)^α /
  max(|W_j|)^(1−α)` clamped to [1e−5, 1e+5]. Applied two ways:
  (a) the spec-mandated forward-pre-hook formulation, with weight
  rescaling `W ← W·s` and a runtime `x ← x/s` hook; (b) a canonical
  absorption formulation that folds the geometric-mean of (q, k, v)
  scales — and (wi_0, wi_1) for the gated FFN — into the preceding
  `T5LayerNorm` weights, so the traced graph contains no extra div
  ops along the absorbable paths.
- **Conversion**: `compute_precision=ct.precision.FLOAT16`,
  `convert_to="mlprogram"`,
  `minimum_deployment_target=ct.target.macOS13`. Inputs `Int32 [1,
  512]`, outputs `Float16 [1, 512, 128]`. Identical to the production
  conversion script except for the precision flag (which is `FLOAT32`
  in production per ADR 010(c)).
- **Parity**: mean per-token cosine similarity vs un-smoothed FP32
  PyTorch reference, using the same sliding-window plan and
  `MIN_NORM=1.0` filter as `scripts/convert-xtr-to-coreml.py`. Each
  of the 3 non-empty fixture inputs is encoded through both pipelines.
- **Toolchain**: Python 3.11, `torch==2.2.2`, `transformers==4.41.2`,
  `coremltools==7.2`, `numpy<2.0`, `matplotlib==3.8.4`. Pinned in
  `scripts/requirements-investigation.txt`. The same toolchain
  reproduced the un-smoothed FP16 NaN per ADR 010(c).

## Outlier characterisation

SmoothQuant's prerequisite — that activation outliers are
concentrated in a small set of channels — is **strongly satisfied**
on this model, but the concentration is in a single class of layer:
the post-gated-activation FFN output projection (`ff.wo`).

Mean per-input-channel max-abs across the 16 calibration windows,
averaged over all 12 encoder blocks:

| Sublayer       | Mean max-\|x\| | Worst block max-\|x\| |
|----------------|----------------:|----------------------:|
| self_attn.q/k/v |             1.3 |                   2.4 |
| self_attn.o    |             8.1 |                  17.9 |
| ff.wi_0        |             2.3 |                   7.4 |
| ff.wi_1        |             2.3 |                   7.4 |
| **ff.wo**      |       **565.7** |             **2198.5** |

The `ff.wo` Linear's input (the post-gate FFN tensor `gelu(wi_0(x)) ·
wi_1(x)`) is two-to-three orders of magnitude hotter than every
other activation in the encoder. In the worst block (block 1) the
single largest channel reaches max-\|x\| = 2198.5 against a median of
2.93 — a concentration ratio of **749×**. The top-50% mass sits in
0.05–0.3% of channels in the four highest-magnitude blocks (1, 2, 8,
11). This is exactly the channel-correlated outlier pattern
SmoothQuant is designed to attack.

The remaining sublayers (self-attention q/k/v/o, FFN gate halves
wi_0/wi_1) have max-\|x\| values that comfortably fit FP16's ~65 504
ceiling without any smoothing and concentration ratios of order
10–25× — well-behaved.

Per-block heatmaps:
[`figures/pre-smoothing/block-XX.png`](smoothquant-feasibility-figures/pre-smoothing/) /
[summary](smoothquant-feasibility-figures/pre-smoothing-summary.png).

## α sensitivity

| Label | α | NaN-free? | Produces rows? | Mean cos vs FP32 PyTorch |
|---|---|---|---|---|
| hooks-alpha0.3 | 0.3 | yes | NO | 0.000000 |
| hooks-alpha0.5 | 0.5 | yes | NO | 0.000000 |
| hooks-alpha0.7 | 0.7 | yes | NO | 0.000000 |
| hooks-alpha0.85 | 0.85 | yes | NO | 0.000000 |
| carveout-projection-only-alpha0.3 | 0.3 | yes | NO | 0.000000 |
| absorbed-alpha0.3 | 0.3 | yes | NO | 0.000000 |

"NaN-free" means no NaN/Inf was observed in the raw or normalised
output tensors of the converted FP16 model. "Produces rows" tracks
whether any token position survived the `MIN_NORM`-of-1.0 filter
that the production sliding-window merge applies — and on every
attempted configuration, every position was rejected because the
post-conversion raw row norm is exactly zero. The cosine column is
0.0 because all PyTorch reference rows compare to a zero-row CoreML
output (per-fixture shapes are reported in
`out-dir/summary.json`).

The full raw-output values were inspected for the α=0.5 hook
attempt: the converted FP16 `.mlpackage` returns a strict zero
tensor of shape `[1, 512, 128]` for every position, with no NaN and
no Inf. The same model evaluated end-to-end in FP32 PyTorch produces
correct row norms (5.14, 4.78, 5.29, …) bit-comparable with the
un-smoothed reference — confirming that the SmoothQuant transform
itself is mathematically correct, and the failure is downstream in
coremltools' FP16 lowering.

α curve shape: there is no curve. The same zero output is produced
across all four α values, both formulations, and the carve-out, so
the per-layer α-tuning rung in `docs/Plan.md`'s "Before declaring
no-go" subsection has nothing to climb.

## Conversion outcome

NaN-free **and** zero-output on every attempt. coremltools emits a
runtime warning during the `Running MIL default pipeline` step on
every attempt:

```
.../coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py:894:
RuntimeWarning: overflow encountered in cast
  return input_var.val.astype(dtype=string_to_nptype(dtype_val))
```

This warning is also present (but tolerated) when the production
script runs at FP32 compute precision, so it is not by itself the
smoking gun. The smoking gun is downstream: tracing T5's RMSNorm
(`x * rsqrt(mean(x²) + eps) * weight`) into FP16 ops causes
`mean(x²)` to overflow when any input element exceeds ~256 (since
256² = 65 536 > FP16 max 65 504). Once any `mean(x²)` overflows to
+∞ in FP16, `rsqrt(+∞) = 0`, the entire layer's output is zero, and
zero propagates through every downstream block.

SmoothQuant scales the *Linear* inputs and outputs but does not
touch the RMSNorm. The hot tensor that enters block-i+1's RMSNorm —
which is the residual sum of all prior blocks' contributions —
inherits all the magnitude that `ff.wo`'s output adds to the
residual stream, and `ff.wo` produces the same output magnitude
post- and pre-smoothing (smoothing is a graph-equivalent
re-parameterisation; it changes intra-Linear arithmetic, not the
Linear's downstream-visible result). The RMSNorm overflow is
therefore unaffected.

This is the failure mode ADR 010(c) described originally and is
unchanged by SmoothQuant: the technique re-distributes magnitude
inside the Linear, not across the residual stream that feeds
RMSNorm. Forward hooks vs. LayerNorm absorption, partial vs. full
smoothing, and α tuning all leave the residual-stream magnitude
identical, which is why every variant produces the same zero output.

Practical implication: SmoothQuant unblocks FP16 conversion only
when the bottleneck is **inside** a Linear (per-channel weight or
activation outliers that exceed FP16 range). On `google/xtr-base-en`
the bottleneck is **outside** the Linear — it is the residual-stream
magnitude entering the next RMSNorm. SmoothQuant has no lever for
that.

## Parity result

Per-fixture cosine similarity of CoreML FP16 (smoothed) vs PyTorch
FP32 (un-smoothed reference), all six attempts, all three non-empty
fixtures:

| Fixture | Reference rows | CoreML rows | Cosine |
|---|---:|---:|---:|
| short                       |    6 |    0 | 0.0 |
| paragraph                   |   47 |    0 | 0.0 |
| long_frankenstein_opening   |  554 |    0 | 0.0 |

Mean cosine = 0.0 for all six attempts. Gate is ≥0.999. **Fail by a
wide margin.** The reference rows are produced correctly by the
un-smoothed FP32 PyTorch pipeline (matching the committed
`Tests/Fixtures/xtr-base-en.embeddings.bin`); the CoreML row counts
are zero because every position's post-conversion raw norm is
strictly zero, falling below `MIN_NORM=1.0` and being filtered out.

## What this means for the SmoothQuant FP16 path

The strategic prize remains attractive (~80 MB asset, FP16 ANE-
eligible, tighter cross-stack parity) but **SmoothQuant alone is not
the lever that unlocks it on this model**. The FP16-blocking
bottleneck is the residual-stream magnitude that flows into each
block's RMSNorm — not per-Linear activation/weight extremes. Any
future attempt to land the FP16 path will need a technique that
clips, normalises, or otherwise bounds the residual stream itself
(or carves the RMSNorm out into FP32 in a way coremltools will
respect — previously attempted under ADR 010(c) without success).

Per the `docs/Plan.md` "If SmoothQuant fails" ladder (already on
`main` from commit `53b25f4`), the recommended next move is **rung
1: INT8 weight-only quantisation** via
`coremltools.optimize.coreml.linear_quantize_weights()`. That gives
a ~110 MB asset (vs. current ~430 MB) on FP32 compute — a size win
without disturbing the precision contract. Rung 2 (custom MIL pass
to clip/promote outlier sites) and rung 3 (switch model) remain
available if a future investigation explicitly targets the
RMSNorm-overflow bottleneck.

ADR 010(c) is **not** amended by this investigation; the precision
contract on `main` remains FP32 compute / FP16 outputs / ~430 MB.

## References

- Xiao, Lin, Seznec, Wu, Demouth, and Han, "SmoothQuant: Accurate
  and Efficient Post-Training Quantization for Large Language
  Models", NeurIPS 2023 — `https://arxiv.org/abs/2211.10438`.
- ADR 010(c) — current FP32-compute / FP16-output precision contract.
  Unchanged by this investigation.
- ADR 014(g) — "When the asymmetry can collapse — future ratchet
  conditions." This investigation is exactly the path 014(g)
  anticipates ("model surgery to clip outliers pre-conversion"); the
  result is that SmoothQuant alone is not the sufficient lever.
- Issue #43 — investigation spec.
- Issue #31 — original FP16 conversion bug.
- `scripts/investigate-smoothquant.py` — investigation script.
- `scripts/convert-xtr-to-coreml.py` — production conversion script;
  not modified by this issue.
- Reproducing this report: `pip install -r
  scripts/requirements-investigation.txt && python
  scripts/investigate-smoothquant.py`. Outputs land under `--out-dir`
  (default `/tmp/switchcraft-smoothquant-investigation/`); committed
  figures live under
  `docs/investigations/smoothquant-feasibility-figures/`.
