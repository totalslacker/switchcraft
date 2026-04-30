# Custom MIL pass FP32-promotion feasibility on `google/xtr-base-en`

## Recommendation

**No-go.** Targeted FP32 promotion of the residual-add → RMSNorm
sub-graph (and the L2-norm projection tail) does not unblock FP16
conversion of `google/xtr-base-en` on coremltools 7.2: every
attempted FP32 island short of "everything except Linears" produces
NaN-filled outputs and mean cosine 0.0 against the FP32 PyTorch
reference, while the only promotion island that does pass parity
(Island E, 26.6% of ops at FP32, 268.6 MB asset) is functionally a
full-FP32 build with FP16 Linear weights — not the strategic ~80 MB /
FP16 / ANE-eligible prize this investigation set out to unlock. Per
the `docs/Plan.md` "If SmoothQuant fails" ladder, the applicable next
move is to **stay on the current FP32-compute / FP16-output
configuration (ADR 010(c), ~430 MB)** and treat rung 1 (INT8
weight-only quantisation) as the available size-only win, since it
does not require solving the FP16 path.

## Scope and methodology

The investigation is implemented in
`scripts/investigate-mil-fp32-promote.py`. Highlights:

- **Parity inputs**: the three non-empty fixture inputs from
  `Tests/Fixtures/xtr-base-en.embeddings.json`; matches the production
  parity-check shape.
- **Pattern matching strategy** (resolves the spec's open question):
  structural matching via a custom `AbstractGraphPass` inserted
  between `common::fuse_layernorm_or_instancenorm` (index 36) and
  `common::add_fp16_cast` (index 54) of the default pipeline. The
  pass walks the post-fusion MIL program, identifies promotion
  candidates by op-type signatures, and stores them in a
  closure-shared `set[id(op)]` that a paired `op_selector` reads when
  `ct.transform.FP16ComputePrecision` decides whether to insert FP16
  casts. No PyTorch-side model surgery: the matcher is robust because
  the RMSNorm-decomposed signature has no later fusion pass that
  rewrites it (see "Pattern identification").
- **Island sweep**, in order of growing FP32 scope:
  **A** RMSNorm 6-op cluster only (anchor: `rsqrt`); **B** A +
  residual `add`; **C** B + L2-norm tail; **D** C + upstream
  attention/FFN-output `add`; **E** escape hatch (everything except
  `linear`/`matmul`/`conv`).
- **Conversion** is identical to the production script except for the
  precision flag and inserted custom pass; outputs remain
  `Float16 [1, 512, 128]`.
- **Parity** is reported two ways: **tail-aligned cosine** (same shape
  contract as the production parity check; cosine forced to 0.0 when
  row counts disagree) and **position-aligned cosine** (cosine on the
  per-token-position intersection of MIN_NORM survivors, with
  NaN-bearing rows dropped). The latter is the more forgiving metric
  and is reported so a "0.0" claim cannot hide a near-pass.
- **Toolchain**: Python 3.11, `torch==2.2.2`, `transformers==4.41.2`,
  `coremltools==7.2`, `numpy<2.0`, `matplotlib==3.8.4` (pinned in
  `scripts/requirements-investigation.txt`; no version bump required).

## Pattern identification

After tracing the production-equivalent traceable module and running
the default pipeline up to (not through) `common::add_fp16_cast`,
the post-fusion MIL program contains 833 ops. Notable types:

| Op type      | Count | Notes |
|--------------|------:|-------|
| `const`      |   389 | weights + scalars; not promotion candidates |
| `linear`     |    85 | q/k/v/o + wi_0/wi_1/wo + 768→128 projection |
| `mul`        |    63 | RMSNorm tail muls (×2 each), gate path muls |
| `add`        |    62 | residual adds + RMSNorm eps adds + position-bias adds |
| `pow`        |    27 | 25× RMSNorm squarings + 1 L2-norm square + 1 L2-norm `pow(·, 0.5)` |
| `reduce_mean`|    25 | RMSNorm variance |
| `rsqrt`      |    25 | RMSNorm reciprocal-sqrt **(matcher anchor)** |
| `matmul`     |    24 | attention QKᵀ + AV |
| `reduce_sum` |     1 | L2-norm tail |
| `real_div`   |     1 | L2-norm tail division |
| `maximum`    |     1 | `clamp_min(1e-12)` lowering |
| `sqrt`       |     0 | `vector_norm` lowers its sqrt as `pow(·, 0.5)` |
| `layer_norm` |     0 | T5 RMSNorm doesn't fuse (see below) |

**Why `layer_norm` count is zero.** coremltools 7.2's
`fuse_layernorm_or_instancenorm` has four templates; *all four*
require a `sub` op (mean subtraction). T5's RMSNorm is variance-only,
so none match. The norm survives as the decomposed primitive sequence
— which is what makes the structural matcher tractable.

**Canonical RMSNorm cluster** (6 ops per anchor): `pow(·, 2) →
reduce_mean(axes=[-1]) → add(eps) → rsqrt → mul(input, ·) → mul(γ,
·)`. **L2-norm tail** (5 ops): `pow(raw, 2) → reduce_sum(axes=[-1],
keep_dims) → pow(·, 0.5) → maximum(·, 1e-12) → real_div(raw, ·)`.
The matcher accepts both `pow(x, 2)` and `mul(x, x)` for squaring; on
this trace, all 25 RMSNorm squarings are `pow(x, 2)`. The L2-norm
sqrt lowers as `pow(·, 0.5)`, not as a `sqrt` op.

**Matcher hit counts** (from `out-dir/matcher-hit-counts.json`):

| Island | Promoted ops | Composition |
|--------|-------------:|-------------|
| A | 150 | 25 RMSNorm clusters × 6 ops |
| B | 174 | A + 24 residual `add` ops (final RMSNorm has no residual feed) |
| C | 179 | B + 5 L2-norm tail ops |
| D | 179 | C + 0 — every residual `add`'s parent is `linear`/`matmul` |
| E | 331 | All non-Linear, non-const, non-cast ops |

Counts match the encoder structure: 12 blocks × 2 sub-layers + 1
final = 25 RMSNorms; 24 residual `add` ops because the first
sub-layer's residual feed is the embedding directly.

## Promotion outcome and parity result

Full sweep (from `out-dir/sweep.md`):

| Label             | Island        | NaN-free? | Promoted ops | Total ops | FP32 fraction | Asset MB | Tail-aligned cos | Position-aligned cos |
|-------------------|--------------:|:---------:|-------------:|----------:|--------------:|---------:|----------------:|---------------------:|
| fp16-no-promote   | (no promote)  | yes       |            0 |       829 |          0.0% |    215.4 |        0.000000 |             0.000000 |
| fp32-reference    | __fp32__      | yes       |            0 |       833 |          0.0% |    430.7 |        1.000000 |             1.000000 |
| island-A          | A             | **no**    |          150 |       929 |         16.1% |    215.5 |        0.000000 |             0.000000 |
| island-B          | B             | **no**    |          174 |       929 |         18.7% |    215.5 |        0.000000 |             0.000000 |
| island-C          | C             | **no**    |          179 |       935 |         19.1% |    215.5 |        0.000000 |             0.000000 |
| island-D          | D             | **no**    |          179 |       935 |         19.1% |    215.5 |        0.000000 |             0.000000 |
| island-E          | E             | yes       |          331 |      1243 |         26.6% |    268.6 |        1.000000 |             1.000000 |

(`Total ops` exceeds 833 in the FP16 columns because
`add_fp16_cast` injects FP16/FP32 cast ops at every precision
boundary; inserted casts count toward program length.)

**Two distinct failure modes.** The plain-FP16 baseline reproduces
ADR 010(c)'s known result: NaN-free, but every output position is
zero (RMSNorm's `mean(x²)` overflows in FP16, `rsqrt(+∞) = 0`, zero
propagates downstream; MIN_NORM=1.0 rejects every position with
`zero_rows` = 7/49/1024).

Islands A–D *change the failure mode but not the verdict*: the output
is no longer all-zeros, but the `normalised` output contains NaN in
every MIN_NORM-survivor row, leaving zero usable position overlap with
the reference. Per-fixture row counts are 7/49/581 — *more* than the
reference's 6/47/554 — meaning raw row norms now clear MIN_NORM, but
at least one component of every survivor is NaN.

The mechanism is the binding finding: **promoting the RMSNorm cluster
(or even RMSNorm + immediate residual `add`) does not address the
upstream FP16 saturation that produces the inf in the first place.**
The residual `add`'s input operands are FP16 outputs of the
attention/FFN sub-layer's Linears (and the post-gate `mul(gelu(wi_0),
wi_1)`). When `ff.wo`'s output has a channel > FP16 ceiling
(~65 504), it saturates to +∞ in FP16 *before* it reaches the
residual add. The custom pass inserts an FP16→FP32 cast at the
island boundary, but the cast input is already +∞, so the cast emits
+∞. FP32 RMSNorm of inf produces NaN/inf (`mean(inf²)+eps = inf`,
`rsqrt(inf) = 0`, `mul(inf, 0) = NaN`). NaN propagates through every
subsequent block and the L2-norm tail.

Island D specifically tested the
transitive-promotion-one-hop-further-back hypothesis: walk past the
residual `add` to also promote the upstream attention/FFN-output
`add` ops. On this graph that walk found *zero* additional ops —
every residual `add`'s second input is a `linear` or `matmul`
output directly, not another `add`. Pushing the FP32 island one
more hop therefore requires promoting some Linears, at which point
the carve-out is no longer "targeted" in any meaningful sense.

Island E confirms the upper bound: with every op except
`linear`/`matmul`/`conv` at FP32 (26.6% fraction), parity is
bit-perfect. The residual stream stays in FP32 throughout. But this
is essentially a full-FP32 build with FP16 Linear *weights* only:
asset 268.6 MB vs the FP32 reference's 430.7 MB is a 38% reduction
entirely from FP16-storing the Linear weights — *not* from
FP16-computing anything load-bearing. ANE eligibility unaffected
(still gated on the FP16-compute residual stream this configuration
explicitly avoids).

The ≥0.999 parity gate is missed by the maximum possible margin on
all four targeted islands (cosine 0.000000); ADR 010(h)'s ±0.025
cross-stack tolerance escape clause does not apply at any reasonable
interpretation. Island E clears the gate (0.99999998), but as
discussed it is not a *targeted* carve-out and therefore does not
satisfy the spec's go condition.

## Coverage data and asset size

Targeted islands all land at 215.5 MB on disk — the same as the
plain-FP16 baseline (215.4 MB) — because none promotes any Linear
*weights* to FP32, so the ~360 MB of weight bulk stays at FP16
storage cost. The 16–19% FP32 op fraction reflects only the small
handful of activation-path ops kept at FP32, which is essentially
free in storage. The spec's anchor estimate of "~90 MB for targeted
carve-outs vs ~430 MB full FP32" was not met: a full-FP16 conversion
already lands at ~215 MB on this 12-block T5 encoder + 768→128
projection, not ~90 MB.

Performance proxy: not measured. No targeted island passed
correctness, and the only passing island is functionally full-FP32 —
so a latency comparison would be either vacuous or
apples-to-oranges.

## What this means

**The custom-MIL-pass route does not unlock the strategic FP16 + ANE
prize on `google/xtr-base-en`**, for the same underlying reason
SmoothQuant did not: the FP16 overflow happens *inside* (or at the
*output* of) FP16 Linears, before any RMSNorm-cluster boundary a MIL
pass can isolate. A pass that promotes only the RMSNorm and residual
`add` receives values that have already saturated to +∞; the FP16→FP32
cast preserves the inf, and FP32 ops applied to inf produce NaN/inf.
Extending the FP32 island back through the producing Linears defeats
the purpose of being targeted.

This investigation does not amend ADR 010(c). The precision contract
on `main` remains FP32 compute / FP16 outputs / ~430 MB. Of the
`docs/Plan.md` "If SmoothQuant fails" ladder, **rung 0 (SmoothQuant,
#43)** and **rung 2 (custom MIL pass FP32 promotion, this report)**
are now both declared no-go. Viable next moves that do *not* require
solving FP16:

- **Rung 1 (INT8 weight-only quantisation)** —
  `coremltools.optimize.coreml.linear_quantize_weights()` produces a
  ~110 MB asset at FP32 compute precision, no ANE eligibility. A
  size-only win that does not touch any of the FP16 carve-outs ruled
  out.
- **Rung 3 (model swap)** — checkpoint without T5's variance-only
  RMSNorm magnitude or with FP16-safe FFN activations. Out of scope.
- **Future ratchet (weight pre-clip / activation surgery upstream)** —
  scale-down `wi_1` and matching scale-up of `wo` along the same
  input channels (mathematically equivalent but with smaller pre-`wo`
  activations). A SmoothQuant variant aimed at the residual stream
  rather than the Linear inputs, anticipated by ADR 014(g)'s "model
  surgery to clip outliers pre-conversion" phrasing. Not investigated.

ADR 014(g) names "a custom MIL pass" alongside model surgery and
distilled-model swaps as a ratchet condition for collapsing the
FP32/FP16 asymmetry. The custom-MIL-pass leg of that condition is
**falsified for this model**: the points where outliers actually
happen are inside FP16 Linears, not at points a MIL pass can isolate
without extending the FP32 island to the entire encoder body. The
ADR is not amended by this report; a future ADR-update PR may want
to refine the phrasing.

## References

- `docs/investigations/smoothquant-feasibility.md` (#43) — prior
  investigation; identical failure mode at a different point in the
  graph.
- ADR 010(c) — current FP32-compute / FP16-output contract. Unchanged.
- ADR 014(g) — names this rung as a ratchet condition; empirically
  falsified for this model.
- `scripts/investigate-mil-fp32-promote.py` — investigation script.
- `scripts/convert-xtr-to-coreml.py` — production conversion script;
  unchanged.
- coremltools 7.2: `mil.passes.defs.optimize_normalization`
  (`fuse_layernorm_or_instancenorm`, does not match T5 RMSNorm);
  `mil.passes.defs.quantization` (`FP16ComputePrecision`);
  `mil.passes.pass_pipeline` (`PassPipeline.insert_pass`);
  `mil.passes.pass_registry` (`register_pass`).
- Reproducing: `pip install -r scripts/requirements-investigation.txt
  && python scripts/investigate-mil-fp32-promote.py`. Outputs land
  under `--out-dir` (default
  `/tmp/switchcraft-mil-fp32-promote-investigation/`); committed
  figures under `docs/investigations/mil-fp32-promote-figures/`. Wall
  clock on Apple Silicon: ~4 minutes (~30 s × 7 islands plus model
  load and tracing).
