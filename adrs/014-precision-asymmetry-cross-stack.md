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
| Compute precision | FP32 internal | Q4K matmul, Q8 attention |
| Output precision | FP16 at model port (widened to FP32 at the Swift boundary) | F32 at output |
| Asset size | ~430 MB | ~80 MB |
| Native quantisation toolchain | CoreML (`compute_precision`); FP16/FP32 only | ggml K-quants (Q4_K, Q5_K, Q6_K, Q8_0, F16, F32) |
| Bundle-size budget | Generous (apps, IDE plugins, server-side macOS) | Tight (embedded clients at scale) |

Both stacks consume the same `google/xtr-base-en` weights. The
asymmetry is purely an inference-time choice driven by deployment
constraints — not a quality difference in the underlying model.

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

## (d) Why Witchcraft is Q4K + Q8 attention

Witchcraft is built on the `ggml` / llama.cpp ecosystem and uses GGUF
**K-quants** — block-quantised formats designed specifically for LLM
inference at low bit widths:

1. **Q4K** stores weights as 4 bits per element with shared per-block
   scale and min, in 256-element super-blocks. Roughly 5–6× smaller
   than FP16. ggml has heavily-optimised Metal kernels for Q4K
   dequant-and-matmul fused ops, so it's not just smaller, it's also
   fast on Apple Silicon (this matters because Witchcraft also runs
   on Macs).

2. **Q8 specifically for attention.** Attention scoring (`Q · Kᵀ`) is
   more sensitive to quantisation noise than feedforward layers
   because scores feed through softmax, which exponentially amplifies
   precision errors. Q8 (~8 bits/element, shared scale) keeps
   attention near-FP16 quality while the rest of the network stays at
   Q4K. This "Q4K body, Q8 attention" recipe is standard ggml
   guidance, not a Witchcraft invention.

3. **Bundle size was the hard constraint** for Witchcraft's deployment
   target (Dropbox internal client-side semantic search at scale).
   ~80 MB on user disk matters more there than the last 0.5 % of
   accuracy. ggml's K-quants are designed for exactly that point on
   the curve.

4. **Witchcraft *could* ship higher precision but doesn't.** ggml
   supports F16 and F32 GGUF formats. Reasons not to:
   - Larger asset for negligible quality gain on retrieval tasks.
   - Their published NDCG@10 ∈ [0.31, 0.33] on NFCorpus is
     calibrated against Q4K + Q8 attention, so changing the format
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

- **CoreML's FP16 path starts working on T5.** A future coremltools
  release, a custom MIL pass, model surgery to clip outliers
  pre-conversion, or migrating to a smaller distilled model without
  T5's outlier problem could let Switchcraft drop to ~80 MB at FP16
  throughout. This brings the precision pair closer (FP16 vs Q4K is
  closer than FP32 vs Q4K) and the cross-stack tolerance should
  tighten toward ±0.01 in the same commit that updates the
  `.mlpackage` and references.
- **Witchcraft starts shipping higher-precision GGUF** (Q5_K, Q6_K,
  Q8_0, F16, F32) for production-grade users. Same effect — closer
  precision pair, tighter tolerance, in the same commit that
  regenerates `Tests/Fixtures/reference_*.bin`.
- **Switchcraft adds INT8 / W4 quantisation** as a Phase 2 optimisation
  (per `docs/Plan.md` "Phase 2 production optimisation" list). This
  *expands* the trade-off space (Switchcraft would then offer both
  FP32 and quantised builds) without changing the FP32 build's parity
  contract.

When any of these trigger, ADR 010(h)'s "tighten back toward ±0.01"
note should be revisited in the same commit that updates the
references.

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

## References

- ADR 010(c) — Switchcraft asset format and precision contract.
- ADR 010(h) — Parity-testing tolerance rationale and table.
- ADR 013 — Reference fixture provenance (the cross-stack fixtures
  this ADR's analysis applies to).
- Issue #31 — FP16 conversion debugging that established FP32 compute
  as the smallest working configuration.
- Issue #30 / PR #33 — Where the ±0.025 tolerance was empirically
  calibrated against the FP32-vs-Q4K precision pair.
- `huggingface/diffusers#8604` — same FP16 activation-outlier pattern
  in T5-family encoders.
- `Tests/Fixtures/facts_corpus.json.provenance` — runtime artefact
  recording the precision pair for any given parity run.
- ggml documentation on K-quants — for the Q4K/Q8/Q5K/Q6K format
  rationale.
