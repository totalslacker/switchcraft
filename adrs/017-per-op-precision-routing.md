# ADR 017 — Per-op precision routing in `T5MetalEmbedder`

**Status**: Accepted
**Date**: 2026-05-01
**Issue**: #64 (sub-issue of umbrella #57, Phase 2 Metal embedder — port of ggml's T5 inference to Swift + Metal)

This ADR promotes the per-op precision matrix in
`docs/porting/ggml-t5.md` from **informational** (a catalogue first
guess) to **normative** (a load-bearing contract every kernel and
orchestrator change must respect). It is the natural follow-up to ADR
014(b)/(d)/(i) — the cross-stack precision-asymmetry analysis that
explained *why* the Switchcraft port runs FP32 on the sensitive ops —
now that the kernels (#60–#63), the FP32 matmul (#64), and the
orchestrator (`T5MetalEmbedder`, #64) have empirically confirmed the
matrix kernel-by-kernel.

ADR 014 owns the cross-stack precision rationale; this ADR ratifies the
per-op outcome on the SwitchcraftMetal side and pins it normatively
against future kernel additions, performance optimisations, or upstream
upgrades.

---

## (a) Context

`docs/porting/ggml-t5.md` §"Per-op precision matrix" was published as
**informational** because no per-kernel data existed yet to confirm
ggml's per-op recipe transferred cleanly to our hosting model:

- Kernels not yet ported → no parity numbers to anchor the matrix.
- Orchestrator not yet wired → no end-to-end FP32 reference to verify
  the cumulative drift across 12 layers stays within the 0.99999 cosine
  budget.

The catalogue's policy was: subsequent sub-issues *may* amend a row of
the matrix if the port surfaces a per-op requirement the first guess
got wrong; amendments must be recorded both in the catalogue and in a
follow-up ADR.

Sub-issues #60–#64 have now landed:

- #60 — `Q4KMatMul` Q4_K→FP16 dequant + FP32 accumulator parity ≥ 0.99999.
- #61 — Gain-fused RMSNorm FP32-throughout parity ≥ 0.99999.
- #62 — Softmax FP32-throughout parity ≥ 0.99999 (with optional bias).
- #63 — Element-wise residual add / gated-GELU / L2 norm FP32 parity.
- #64 — FP32 batched matmul + orchestrator integration. End-to-end
  cosine vs PyTorch FP32 reference is gated at ≥ 0.99999 per token in
  `T5MetalEmbedderTests` (asset-gated by `SWITCHCRAFT_XTR_GGUF`).

The empirical confirmation is sufficient to promote the matrix to
normative.

## (b) Decision — promote the per-op matrix to normative

The per-op precision matrix in `docs/porting/ggml-t5.md` §"Per-op
precision matrix" is **normative**. Every change to a kernel,
orchestrator, or precision-affecting code path must:

1. Either preserve the row's `Weight storage / Compute / Accumulator`
   triple unchanged, or
2. Land an amendment in the same PR that updates **both** the catalogue
   table **and** this ADR's §(c) "Per-op binding" table, with rationale
   tied to a parity test that demonstrates the change is non-regressive
   against the cross-stack ≥0.99999 cosine gate (or the relevant
   per-kernel parity test, whichever is tighter).

Pure performance optimisations that preserve the precision triple
(threadgroup tiling, simdgroup matmul, fused dequant) need no
amendment.

## (c) Per-op binding — sensitive-op carve-outs

The matrix in the catalogue is the canonical reference. This section
restates the sensitive-op rows, plus the new `2_Dense` and FP32-matmul
rows landed by #64, so the constraints have a single normative home
that survives even if the catalogue's table is later restructured.

| Op (#64-side caller) | Precision rule (storage / compute / accum) | Rationale |
|---|---|---|
| Token embedding gather | Q4_K bytes → CPU dequant → FP32 upload | ADR 014(b) — embedding lookup result feeds the FP32 residual stream. |
| Pre-attention RMSNorm (×12) + pre-FFN RMSNorm (×12) + final RMSNorm (×1) | FP32 / **FP32 — non-negotiable** / FP32 | ADR 014(b)(2) "RMSNorm `mean(x²)` overflows FP16 historically." Gain multiply fused into the kernel, no FP16 staging. |
| Q/K/V/O projections (×48) | Q4_K weights → FP16 staging → FP32 accum / FP32 output | ADR 014(b)(1) and the catalogue Q4KMatMul row; FP16 staging is internal to the kernel, externally FP32. |
| Q · Kᵀ matmul (×12, per-head) | FP32 / FP32 / FP32 | ADR 014(d) — small-K activation matmul into the FP32 softmax slot. **No 1/√d_k scaling** (T5 v1.1 omits it). Implemented via `FP32MatMulKernel` with row-stride args for per-head slicing; documented as a deviation in `docs/porting/ggml-t5.md` §"Deviations from upstream". |
| Relative-position bias add | FP32 / FP32 / FP32 | Bias buffer is built CPU-side per encode (~12 MB) and added pre-softmax via `SoftmaxKernel`'s optional `bias:` slot. |
| Softmax (×12) | FP32 / **FP32 — sensitive op** / FP32 | ADR 003 + ADR 014(d) — subtract-max, exp, reciprocal compound FP16 loss. |
| Softmax · V matmul (×12, per-head) | FP32 / FP32 / FP32 | Same FP32 carve-out justification as Q · Kᵀ. Output written into `[M, D]` interleaved layout via the row-stride arg so the O projection sees a contiguous input without a separate merge pass. |
| FFN gate `wi_0` (×12) + up `wi_1` (×12) | Q4_K / FP16 staging / FP32 | Same as Q4KMatMul row above. |
| Gated-GELU activation (×12) | FP32 / FP32 (`gelu_new`, tanh-based) / FP32 | ADR 014(b) — `tanh` saturates safely at FP32; FP16 amplifies precision loss in the cubic term. |
| FFN down `wo` (×12) | Q4_K / FP16 staging / FP32 | Same as Q4KMatMul row above. |
| Residual add (×24) | FP32 / **FP32 — residual stream stays FP32** / FP32 | ADR 014(b) — every residual sum carries the cumulative encoder state; FP16 would alias adjacent residual contributions. |
| 2_Dense projection (×1, 768→128) | **FP16 weights widened to FP32 once at init** / FP32 / FP32 | The catalogue planned a dedicated FP16 `ProjectionMatMul` kernel; the `T5MetalEmbedder` orchestrator instead widens the FP16 weight to FP32 at init (~384 KiB resident) and reuses `FP32MatMulKernel`. The activation path remains FP32; only the storage line of the catalogue's row is amended. |
| L2 normalisation (×1) | FP32 / **FP32 — sensitive op** / FP32 | ADR 014(b)(2) — `sum(x²)` over the projected row overflows FP16. |

The two amendments to the catalogue's first guess are:

1. **Q · Kᵀ and softmax · V matmuls**: the catalogue named a
   `Sources/SwitchcraftMetal/Kernels/Attention.swift` placeholder; the
   port consolidates these into the single `FP32MatMulKernel` that also
   runs the 2_Dense projection. The activation precision contract is
   unchanged — the kernel surface is narrower than the catalogue's
   first cut.
2. **2_Dense projection**: the catalogue planned a dedicated FP16
   matmul kernel; this ADR records the deviation to FP16→FP32 widening
   at init. Justification: the FP16 path is one matmul, ~384 KiB of
   widened weight, and would otherwise require an entire parallel
   kernel family (FP16 weight × FP32 activation × FP32 accum) shipping
   for a single op. The FP32-only path is correct, simpler, and
   cheaper to maintain. ADR 012's 300 MB peak-RSS ceiling is unaffected.

## (d) Future-kernel rules

A new sub-issue that adds a kernel to `SwitchcraftMetal/Kernels/` (or
an MSL specialisation in `SwitchcraftMetal/Shaders/`) **must**:

1. Land its precision triple in the catalogue's matrix in the same PR.
2. Cite this ADR in the PR description (along with ADR 014 if the row
   touches an FP32 carve-out).
3. Carry a parity test that gates ≥ 0.99999 cosine vs an
   FP64-accumulator pure-Swift reference, matching the convention set
   by #60–#63.
4. If the new kernel changes a row that other kernels already
   exercised, run the cross-stack `T5MetalEmbedderTests` parity gate
   end-to-end (not just the per-kernel test). The compounded budget
   over 12 layers is the binding constraint.

A sub-issue that *amends* a precision triple (rare; expected only when
upstream ggml/Candle revises its own per-op recipe) **must**:

1. Update both the catalogue table and §(c) "Per-op binding" above in
   the same PR.
2. Re-run the cross-stack parity gate; the amendment is rejected if
   end-to-end cosine drops below 0.99999.

## (e) Cross-references

- **ADR 003** — typing requirements / parity-arithmetic FP32 carve-outs.
  This ADR reuses ADR 003's FP32 sensitive-ops list as the basis for
  the carve-outs in §(c).
- **ADR 009** — `Embedder` protocol contract; `T5MetalEmbedder` is an
  additional conformance, not a public API change.
- **ADR 010(j)** — GGUF asset distribution; the Q4_K weight buffers
  this ADR routes are produced by the pipeline ADR 010(j) documents.
- **ADR 011** — sliding-window strategy reused by `T5MetalEmbedder`
  unchanged; precision routing applies per window.
- **ADR 014** — precision asymmetry cross-stack; this ADR ratifies the
  per-op outcome of ADR 014(b)/(d)/(i) on the SwitchcraftMetal side.
  The "Status update (2026-05-01)" line in ADR 014 originally referred
  to "ADR 016 (sub-issue #64)"; that reference is corrected to "ADR
  017" in the same PR that lands this ADR.
- **ADR 015** — `MetalContext` and dispatch foundation reused
  unchanged.
- **ADR 016** — GGUF asset distribution; the upstream of every weight
  this ADR routes.
- `docs/porting/ggml-t5.md` §"Per-op precision matrix" — the matrix
  this ADR promotes to normative; canonical row-by-row reference for
  every kernel.

## (f) Rejected alternatives

- **Lower the matrix to optional / "advisory"**. Rejected: the cross-
  stack ≥ 0.99999 gate is the contract every consumer of `T5MetalEmbedder`
  inherits; it cannot be honoured without per-op binding.
- **Move the matrix into per-kernel header comments only**. Rejected:
  the matrix is a graph-level invariant. Per-kernel comments are the
  *implementation* of the contract; the contract itself needs a single
  normative home that doesn't move when kernels are reorganised.
- **Ship the dedicated FP16 `ProjectionMatMul` kernel from the
  catalogue's first cut**. Rejected: one matmul does not justify a
  parallel kernel family; the FP16→FP32 widening at init is correct,
  simpler, and matches the rest of the activation path's precision.
