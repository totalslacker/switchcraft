# ADR 014 — Cross-stack precision asymmetry: why Switchcraft and Witchcraft sit on different points of the precision/size curve

**Status**: Accepted
**Date**: 2026-04-30
**Issue**: ad-hoc — captured during Phase 1 wrap-up to document the rationale behind the FP32-vs-Q4K cross-stack drift.

This ADR records *why* Switchcraft and Witchcraft run at different
numerical precisions for the same model. ADR 010(c) records what
Switchcraft does locally; ADR 010(h) records the resulting parity
tolerance. This ADR records the cross-stack reasoning — useful for
future contributors debugging cross-implementation drift, downstream
users wondering whether the two stacks are interchangeable, and any
future work that proposes moving either side on the precision curve.

---

## (a) The asymmetry, summarised

| | Switchcraft (Apple platforms) | Witchcraft (ggml ecosystem) |
|---|---|---|
| Weight **storage** precision | FP32 (or FP16 weights, FP32 compute — see below) | Q4K body, Q8 attention K/V |
| Sensitive-op **compute** precision (RMSNorm, residual stream, softmax) | FP32 | FP32 (independent of storage) |
| Other-op **compute** precision | FP32 | FP16/FP32 mix per ggml kernel selection |
| Output precision | FP16 at model port (widened to FP32 at the Swift boundary) | F32 at output |
| Asset size | ~430 MB | ~80 MB |
| Native quantisation toolchain | CoreML (`compute_precision`) — FP16/FP32 only, coarse-grained | ggml K-quants (Q4_K, Q5_K, Q6_K, Q8_0, F16, F32) — fine-grained per-op kernel selection |
| Bundle-size budget | Generous (apps, IDE plugins, server-side macOS) | Tight (embedded clients at scale) |

Both stacks consume the same `google/xtr-base-en` weights. The
asymmetry is **not** "different points on a single precision curve" —
that framing was an oversimplification. It is at minimum a
two-axis difference: **storage precision** (how weights sit on disk
and in memory) and **compute precision per op** (what numerical
type the kernel uses for the matmul, norm, softmax, etc.). ggml
decouples these axes; CoreML largely couples them. See section (i).

## (b) Why Switchcraft is FP32 compute, not FP16

The original Plan and the original ADR 010(c) targeted FP16 throughout
(asset size + ANE/GPU speed). What blocked it, in order of discovery
during issue #31:

1. **T5 has well-documented activation outliers in the encoder body.**
   Attention matmul → RMSNorm's `pow → reduce_mean → rsqrt` produces
   intermediate values that exceed FP16's ~65 504 dynamic range,
   creating NaN that contaminates downstream layers. The same pattern
   is documented in `huggingface/diffusers#8604` for Stable Diffusion's
   T5-family text encoder. This is a model-graph property, not a
   coremltools or Switchcraft defect.

2. **The L2-norm tail also overflows independently.** The unit-norm
   projection sums 128 squared values; with FP16 inputs, intermediate
   magnitudes during the sum can exceed range even if the encoder body
   were fine. Two separate FP16 pain points in the same graph.

3. **Per-op FP32 carve-outs do not fix it.** Issue #31 tried
   `coremltools.transform.FP16ComputePrecision(op_selector=…)` to keep
   `pow`, `rsqrt`, `reduce_sum`, `real_div` at FP32 while the rest of
   the graph stayed FP16. That stabilises the L2-norm overflow path
   but does **not** stabilise T5's encoder body. The same NaN
   reproduces against both coremltools 7.2 and 9.0. Something deeper
   in how coremltools traces T5's attention path resists targeted
   carve-outs — this is a known coremltools / T5-family graph
   interaction, not a Switchcraft-specific defect, and not currently
   fixable without either upstream coremltools work or model surgery.

4. **The smallest working carve-out is "all of it":** FP32 compute
   throughout. ADR 010(c) records this as the as-built local
   contract.

## (c) Why FP16 outputs are still safe

Output ports do not drive any further model computation; they just
carry final values from the model graph to Swift. The values that
survive the model are already L2-normalised to unit length, so they
sit firmly in `[-1, 1]`. Zero overflow risk. FP16 in `[-1, 1]` has
roughly 3 decimal digits of precision — more than enough for the
cosine-similarity / MaxSim scoring downstream callers use.

The Swift `CoreMLModelIO` reader widens FP16 → FP32 at the boundary,
so internal arithmetic in Switchcraft is full precision throughout.
Halving the per-token output buffer is a real but modest size win on
the wire and in memory.

## (d) Why Witchcraft uses Q4K weight **storage** (and what it does at compute time)

Witchcraft is built on the `ggml` / llama.cpp ecosystem and uses GGUF
**K-quants** — block-quantised formats designed specifically for LLM
inference at low bit widths.

A critical clarification this ADR previously elided: **K-quants are
storage formats, not compute formats.** When ggml executes a matmul
against Q4K-stored weights, the runtime dequantises those weights
on-the-fly to FP16 or FP32 just before the matmul kernel runs; the
matmul itself computes at the higher precision (with FP32 accumulator
in the typical Metal kernel). Q4K saves disk and memory bandwidth;
the actual numerical work happens at higher precision. The phrase
"Q4K matmul" should be read as "Q4K-stored weights, dequantised on
read, computed at higher precision" — not as "4-bit arithmetic in
the matmul kernel."

With that clarification:

1. **Q4K weight storage** stores weights as 4 bits per element with
   shared per-block scale and min, in 256-element super-blocks.
   Roughly 5–6× smaller on disk than FP16. ggml has heavily-optimised
   Metal kernels for the dequant-and-matmul fused path, so it's also
   fast on Apple Silicon. Witchcraft also runs on Macs, so Metal
   performance matters.

2. **Q8 K/V cache for attention** addresses a different concern:
   attention `Q · Kᵀ` feeds into softmax, which exponentially
   amplifies precision errors. The K/V cache stays at 8 bits to give
   the softmax better numerical headroom than 4 bits would, while
   the projection weights (`q_proj`, `k_proj`, `v_proj`) themselves
   stay at Q4K storage. This "Q4K body, Q8 K/V" recipe is standard
   ggml guidance, not a Witchcraft invention.

3. **The sensitive-op compute path is FP32 throughout.** This is the
   thing that lets Witchcraft survive on `google/xtr-base-en` where
   Switchcraft's CoreML FP16 path produces NaN. ggml's Metal backend
   has separate kernels for FP32 norm (`kernel_rms_norm_f32`) and
   FP16 norm (`kernel_rms_norm_f16`); the runtime dispatches the
   FP32 kernel for T5's RMSNorm regardless of how the weights are
   stored. The residual stream stays in FP32 throughout; `mean(x²)`
   for any reasonable activation (≤ ~10 000) has plenty of headroom
   in FP32 (~10³⁸ max). This is **mixed-precision compute, with
   independent storage precision** — and it is the architectural
   feature that enables Witchcraft's small asset on this exact graph.

4. **Bundle size was the hard constraint** for Witchcraft's
   deployment target (Dropbox internal client-side semantic search
   at scale). ~80 MB on user disk matters more there than the last
   0.5 % of accuracy. ggml's K-quants storage format is designed for
   exactly that point on the curve. The mixed-precision compute
   above is what keeps the small asset *correct*.

5. **Witchcraft *could* ship higher-precision storage but doesn't.**
   ggml supports F16 and F32 GGUF formats. Reasons not to:
   - Larger asset for negligible quality gain on retrieval tasks.
   - Their published NDCG@10 ∈ [0.31, 0.33] on NFCorpus is
     calibrated against Q4K + Q8 K/V, so changing the format
     invalidates the canonical reference number that downstream
     consumers (us) rely on for the cross-implementation parity gate.
   - No upstream pressure to change since their deployment values
     size and latency.

## (e) Why these are different points on the same curve

The two stacks are **not** at opposed engineering goals. They sit at
different points on the same precision / size / quality Pareto curve,
chosen by different deployment constraints:

- **Switchcraft's** Apple-platform budget allows ~430 MB; ANE/GPU
  FP32 paths are well-supported; CoreML's quantisation primitives are
  FP16/FP32 (no block quantisation); FP16 broke for graph-specific
  reasons that couldn't be carved out → **landed at maximum-precision
  FP32** for compute, FP16 only at the safe output port.
- **Witchcraft's** embedded-client budget mandates ~80 MB; ggml's
  K-quants give 5–6× compression with documented small accuracy cost;
  attention gets the Q8 special treatment that's the standard ggml
  recipe → **landed at minimum-acceptable-quality block quantisation**
  for asset density.

Same model weights, same retrieval algorithm — different
inference-time trade-off.

## (f) Consequences for cross-stack parity

The visible artefact is the ±0.025 score drift documented in
ADR 010(h) and the rank-set divergence on weak-signal queries (e.g.
"flamingos" in the 33-fact corpus, where MaxSim scores past rank 2
are nearly tied and small precision drift permutes the ordering).
Both consequences are recorded honestly in
`Tests/Fixtures/facts_corpus.json.provenance` and accepted in
`FactsCorpusParityTests`'s relaxed acceptance shape (top-1 strict /
top-3 set / per-doc score ±0.025 / rank-order beyond 3 best-effort).

This means Switchcraft's per-token embeddings are **not byte-equal**
to Witchcraft's. They are unit vectors in the same conceptual space,
within ±0.025 dot-product distance of each other. For retrieval the
difference is invisible at top-3; it shows up beyond that as low-rank
permutation that's already smaller than the model's noise floor on
borderline-tied queries.

The original `docs/Plan.md` "Bit-exact embedding validation against
Candle reference outputs" line was written before this asymmetry was
understood and has been amended to reflect what's actually
deliverable: PyTorch ≥0.999 cosine and Witchcraft GGUF ±0.025
cross-stack. Bit-exact equality is unreachable so long as the two
stacks sit at different precision points.

## (g) When the asymmetry can collapse — future ratchet conditions

The asymmetry is not a permanent feature. ADR 010(h) anticipates the
Witchcraft-side direction explicitly. Trigger conditions:

- **CoreML's FP16 path starts working on T5.** Two surgical sub-paths
  named in earlier versions of this ADR — pre-conversion activation
  smoothing (SmoothQuant) and a custom MIL pass that promotes the
  RMSNorm cluster + residual `add` to FP32 — are now **empirically
  falsified** for `google/xtr-base-en`:
  - SmoothQuant (#43, `docs/investigations/smoothquant-feasibility.md`)
    re-parameterises *inside* the Linear and cannot lower the
    residual-stream magnitude that overflows the next block's RMSNorm.
    Cosine vs FP32 PyTorch reference: 0.0 across α ∈ {0.3, 0.5, 0.7,
    0.85} and the absorbed-LayerNorm formulation.
  - Custom MIL pass (#46,
    `docs/investigations/mil-fp32-promote-feasibility.md`) cannot
    isolate the saturation point: the FP16 saturation happens *inside*
    `ff.wo` before any reachable carve-out boundary, so promoting the
    RMSNorm cluster to FP32 receives an already-saturated `+inf`
    input and produces NaN. Cosine: 0.0 on every targeted island; the
    only passing island is functionally full-FP32 with FP16 weight
    storage.

  The remaining sub-paths that have **not** been ruled out:
  1. **A future coremltools release** with substantially finer-grained
     compute-precision selection than 7.2 / 9.0 (e.g. true per-tensor
     FP32-promotion that survives MIL graph fusion).
  2. **Custom Metal kernels** that bypass CoreML's automated graph
     optimisation entirely — effectively a Swift port of ggml's T5
     inference path. This is the "Phase 2 — Custom Metal kernels for
     the T5 embedder" line in `docs/Plan.md`. Substantial undertaking.
     The Metal kernels themselves don't reach ANE (custom kernels
     run on GPU only; ANE is reachable only through CoreML's
     `compute_precision` pipeline). **Critically, this does not
     foreclose future ANE access** — the `Embedder` protocol seam
     (ADR 009) supports multiple conformances simultaneously, so a
     future `T5CoreMLFP16Embedder` could ship alongside Metal
     kernels if a coremltools release with finer-grained per-op
     precision control or other surgical advance unblocks the FP16
     path. Pursuing custom Metal kernels sets ANE aside for now;
     it does not surrender the ANE prize permanently.
  3. **Pre-conversion model surgery beyond SmoothQuant** that targets
     the residual-stream magnitude rather than per-Linear inputs —
     e.g., scale-down `ff.wi_1` and matching scale-up of `ff.wo` along
     the same input channels, mathematically equivalent but reducing
     `ff.wo`'s output magnitude. Conceptually plausible but novel; not
     a published recipe like SmoothQuant. Anticipated by the
     SmoothQuant report's "Future ratchet" footnote.
  4. **Migrating to a smaller distilled model** without T5's
     variance-only RMSNorm overflow problem. Loses Witchcraft
     comparability; only viable if upstream Witchcraft also moves.

- **Witchcraft starts shipping higher-precision GGUF** (Q5_K, Q6_K,
  Q8_0, F16, F32) for production-grade users. Same effect — closer
  storage-precision pair, tighter cross-stack tolerance, in the same
  commit that regenerates `Tests/Fixtures/reference_*.bin`.

- **Switchcraft adds INT8 weight-only quantisation** (#45, in flight)
  as a Phase 2 optimisation. This is **storage-only** — compute stays
  at FP32, NaN risk stays zero, ANE eligibility unchanged. It shrinks
  the asset to ~110 MB without affecting the precision contract.
  *Expands* the trade-off space (Switchcraft offers both FP32 and
  INT8w builds) without changing the FP32 build's parity contract.

When any of the unfalsified conditions trigger, ADR 010(h)'s
"tighten back toward ±0.01" note should be revisited in the same
commit that updates the references.

### Status update (2026-05-01) — sub-condition (g)(2) is being followed

Sub-condition (g)(2) above ("Custom Metal kernels … effectively a
Swift port of ggml's T5 inference path") has moved from *unfalsified
candidate* to *active port*. Umbrella issue #57 tracks the multi-PR
campaign; sub-issue #58 lands the porting catalogue at
[`docs/porting/ggml-t5.md`](../docs/porting/ggml-t5.md), which fixes
the upstream commit pins (ggml, llama.cpp, Candle, Witchcraft), the
artefact map (upstream → Swift target), the Q4_K block layout
reference, the per-op precision matrix (informational; ADR 017 promoted it
to normative in #64), the relative-position attention bucket
formula, the FFN activation determination
(`feed_forward_proj: "gated-gelu"` per `google/xtr-base-en/config.json`,
which corrects the issue-body assumption that classic T5-base ReLU
applies), the GGUF asset acquisition pipeline, and the tokeniser
disposition.

This work is **a port, not an investigation**. The kernels exist
upstream in production-tested form (ggml has shipped them for years;
Witchcraft uses Candle's port of the same kernels in production at
Dropbox). The remaining work — sub-issues #59–#65 — is translation,
not discovery. Contrast with sub-conditions (g)(1) (SmoothQuant, #43)
and (g)(2)'s sibling MIL-pass approach (#46): both *were* genuine
research questions and both produced empirical no-go reports. The
ggml port path was deliberately *not* gated behind another
investigation because the source code is the answer; we read it,
translate it, and parity-test against the existing fixtures.

The port does not foreclose future ANE access. Per the existing
sub-condition (g)(1) bullet and ADR 009, the `Embedder` protocol
seam supports multiple conformances; a future
`T5CoreMLFP16Embedder` may ship alongside `T5MetalEmbedder` if a
later coremltools release surfaces finer-grained per-op precision
control. The port sets ANE aside for now; it does not surrender the
ANE prize permanently.

When the port lands its NDCG@10 gate (sub-issue #65) and
`T5MetalEmbedder` ships as an additional conformance, the section
above should record that ratchet sub-condition (g)(2) has *fully
collapsed* — both stacks then have an FP16-storage Q4_K-storage
embedder available. ADR 010(h)'s "tighten back toward ±0.01" note
becomes actionable in the same commit, because Switchcraft and
Witchcraft will run the same kernels against the same GGUF asset
and the cross-stack precision pair becomes Q4_K-vs-Q4_K rather than
FP32-vs-Q4_K.

## (h) What this ADR is not

- It is **not** a directive to change either side's precision. Both
  stacks are at honest, deliberate points on the curve.
- It is **not** a substitute for ADR 010(c) (the Switchcraft-local
  decision) or ADR 010(h) (the resulting tolerance). Those record
  what Switchcraft does and what tolerance the parity tests apply.
  This ADR records *why the two stacks differ in the first place*.
- It is **not** a regression. Cross-stack parity gates were always
  documented as approximate (±0.025); this ADR explains why the
  approximation is intrinsic, not a bug to fix.

## (i) The deeper asymmetry: storage precision vs compute precision

Earlier sections of this ADR (and earlier discussions in the project
generally) framed the Switchcraft↔Witchcraft difference as "different
points on the same precision/size/quality curve." That framing is
not wrong but is incomplete. The two stacks differ on **at least two
independent axes**, and the second axis — *compute architecture* — is
why FP16 is reachable on Witchcraft and unreachable through CoreML on
Switchcraft.

### Two independent axes

1. **Weight storage precision.** How weights sit on disk and in
   memory. Q4K (4-bit + per-block scale) on Witchcraft; FP32 on
   Switchcraft today. This is the axis that determines asset size.
2. **Per-op compute precision.** What numerical type each kernel
   uses for its matmul, norm, softmax, accumulator, etc. Witchcraft
   runs FP32 on sensitive ops (RMSNorm, residual stream, attention
   softmax) regardless of how the weights are stored; Switchcraft
   today runs FP32 throughout because CoreML's `compute_precision`
   doesn't allow targeted carve-outs in a way that survives MIL
   graph fusion (per #43 and #46).

These axes are decoupled in ggml; they are largely coupled in CoreML
at the level of control we have access to. The earlier framing
implied a single "precision" column with one value per stack, which
is why the question "how is Rust able to do what Swift can't?"
appears paradoxical: how does Q4K beat FP16 numerically? It doesn't.
Q4K is a *storage* format, dequantised to FP16 or FP32 before any
arithmetic. The arithmetic Witchcraft does on the residual stream is
**FP32**, the same precision Switchcraft uses today — Witchcraft
just gets to carry weights at Q4K through to the matmul kernel and
have the compute happen at the higher precision.

### Why the architectures permit different choices

- **ggml's Metal backend** has separate kernels for each (op-type ×
  numerical-type) combination: `kernel_rms_norm_f32`,
  `kernel_rms_norm_f16`, `kernel_norm_f32`, `kernel_norm_f16`,
  `kernel_mul_mat_q4_K_f32`, `kernel_mul_mat_f32_f32`, etc. The
  runtime dispatches the kernel matching the inputs and the
  configured precision policy. There is no "compute_precision" flag
  for the whole graph; precision is per-op, chosen by the developer
  when they wire up the inference graph in C/Rust code.

- **CoreML** is at a level above this. You set
  `compute_precision=ct.precision.FLOAT16` (or `FLOAT32`) at
  conversion time and CoreML's optimiser plus runtime dispatches to
  ANE / GPU / CPU at whatever precision its automated optimisation
  passes have decided to use. The per-op selector
  (`coremltools.transform.FP16ComputePrecision(op_selector=…)`)
  exists, but operates on the post-fusion MIL graph after coremltools'
  own passes have re-shaped the boundaries — and as the MIL pass
  investigation (#46) demonstrated, the residual-add → RMSNorm chain
  can't be carved cleanly because the FP16 saturation happens *inside*
  the upstream Linear, before any reachable carve-out boundary. The
  cast at the boundary preserves the saturated value as `+inf`; FP32
  RMSNorm of `+inf` produces NaN; NaN propagates downstream.

So the architectural difference is concretely:

- **CoreML** — high-level, black-box, automated graph optimisation.
  Apple chooses kernels for you, runs MIL fusion + lowering passes,
  dispatches to ANE / GPU / CPU. Lots of magic; less control.
- **ggml** — low-level, kernel-by-kernel, manual graph construction.
  You wire up op kernels by hand; you choose precision per op. Less
  magic; more control.

For this specific T5 graph, the CoreML magic doesn't have a knob to
turn that fixes the FP16 overflow in a targeted way. ggml's lack of
magic means a developer just *picks* FP32 norm kernels and FP16
doesn't enter the picture for those sites.

### Implications

- The "FP16/ANE prize" path through CoreML may genuinely be
  unreachable for `google/xtr-base-en` without writing custom Metal
  kernels (Phase 2 hot-path scope, but expanded substantially), or
  waiting for a future coremltools release that surfaces fine-grained
  per-op precision control through the public API.
- ANE access is gated on FP16 compute, which CoreML provides via
  `compute_precision`. Custom Metal kernels run on GPU only; they
  themselves don't reach ANE — that prize is specifically tied to
  CoreML's `compute_precision` pipeline, which is currently blocked
  for `google/xtr-base-en`. **Important nuance**: pursuing the
  Metal-kernel path does *not* foreclose future ANE access. The
  `Embedder` protocol seam (#16, ADR 009) supports multiple
  conformances simultaneously; if a future coremltools release
  surfaces finer-grained precision control (or any other surgical
  advance unblocks the FP16 path on this graph), a parallel
  `T5CoreMLFP16Embedder` conformance can ship alongside a
  `T5MetalEmbedder` conformance and consumers can pick at init
  time. The Metal-kernel work sets ANE aside for now; it does not
  surrender the ANE prize permanently.
- INT8 weight-only quantisation (#45) is the only easy win currently
  available, and it operates entirely on **axis 1** (storage). It
  shrinks the asset from ~430 MB to ~110 MB without touching axis 2
  (compute precision) — and therefore without engaging any of the
  failure modes that ruled out SmoothQuant (#43) and the MIL pass
  (#46).
- The "precision/size/quality curve" framing in section (e) above is
  preserved as a useful first-order summary, but the deeper truth is
  that ggml and CoreML are not on the same curve at all — they are
  in **different inference-architecture regimes** with different
  knobs available. ggml has a knob CoreML doesn't expose, and that
  knob is what makes ~80 MB FP16-on-T5 work.

This (i) is the most common follow-up question this ADR has gotten
("how can Rust do what Swift can't?"), and the honest answer is
"not what we said earlier — actually, *both* stacks compute the
sensitive ops at FP32; the difference is that ggml lets the
developer specify that per op, whereas CoreML's coarse-grained
precision controls don't permit a targeted carve-out on this
specific graph."

## References

- ADR 010(c) — Switchcraft asset format and precision contract.
- ADR 010(h) — Parity-testing tolerance rationale and table.
- ADR 013 — Reference fixture provenance (the cross-stack fixtures
  this ADR's analysis applies to).
- Issue #31 — FP16 conversion debugging that established FP32 compute
  as the smallest working configuration.
- Issue #30 / PR #33 — Where the ±0.025 tolerance was empirically
  calibrated against the FP32-vs-Q4K precision pair.
- Issue #43 / PR #44 / `docs/investigations/smoothquant-feasibility.md`
  — falsifies the SmoothQuant pre-conversion-smoothing ratchet path.
- Issue #46 / PR #48 / `docs/investigations/mil-fp32-promote-feasibility.md`
  — falsifies the custom-MIL-pass FP32-promotion ratchet path; also
  the source for the storage-vs-compute distinction documented in
  section (i).
- Issue #45 — INT8 weight-only quantisation (in flight); the storage-
  axis win that is reachable today without touching axis 2.
- `huggingface/diffusers#8604` — same FP16 activation-outlier pattern
  in T5-family encoders.
- `Tests/Fixtures/facts_corpus.json.provenance` — runtime artefact
  recording the precision pair for any given parity run.
- ggml documentation on K-quants — for the Q4_K / Q8_0 / Q5_K /
  Q6_K storage-format rationale.
- `ggml-metal.m` (llama.cpp) — example of per-op kernel selection
  (`kernel_rms_norm_f32` vs `kernel_rms_norm_f16`,
  `kernel_mul_mat_q4_K_f32` etc.) that section (i) describes as
  ggml's compute-precision-axis lever.
