# Porting catalogue — ggml's T5 encoder inference → Swift + Metal

**Status**: Active — sub-issue #58 of umbrella #57. Catalogue is the
contract that subsequent sub-issues (#59–#65) consume.
**Date**: 2026-05-01
**Scope**: `google/xtr-base-en` encoder forward pass only. Decoder is
out of scope (XTR is encoder-only at inference time).

This document is a **porting catalogue**, not a research report. Per
ADR 014(g) ratchet sub-condition (g)(2), the path being followed is
"custom Metal kernels … effectively a Swift port of ggml's T5
inference path." The kernels exist upstream in production-tested form.
Our job is translation, not discovery. This catalogue maps each
upstream artefact to its Swift destination so per-kernel sub-issues
can be reviewed against a fixed reference.

The catalogue is consumed by sub-issues #59 (`SwitchcraftMetal`
target + GGUF reader + ADR 010(j)), #60 (Q4_K matmul), #61 (RMSNorm),
#62 (softmax + relpos bias), #63 (element-wise: residual add, gated-
GELU activation, L2 norm), #64 (`T5MetalEmbedder` orchestration +
ADR 016), and #65 (NDCG@10 gate, cross-stack tolerance retune,
Plan.md tick).

---

## Upstream commit pins

| Repo | Pin (SHA) | Role |
|---|---|---|
| `ggml-org/ggml` | `b70770970e84c30a007b3859a453768b3ece2d3d` (2026-05-01) | **Algorithmic reference.** Q4_K block layout, Metal kernel source, per-op precision recipe. |
| `ggml-org/llama.cpp` | `c3c15053925746c74fc2aaf6b864bd66665393c4` (2026-05-01) | **T5 wiring reference.** `src/llama-model.cpp` shows how the T5 encoder forward pass is composed from ggml ops. |
| `huggingface/candle` | `5bd5618c310aaa97a30c5cb20bb6957f164ce6f7` (2026-04-22) | **Behavioural reference.** Witchcraft runs Candle (not raw ggml), so `Tests/Fixtures/reference_embeddings.bin` was produced by Candle's Q4_K kernels. When the port doesn't bit-match the PyTorch reference, Candle is the next debugging target — not ggml. |
| `dropbox/witchcraft` | `6ad59e51cfc89bcfb20756e3f05cf9429b7cb55f` (2026-04-24) | **Fixture provenance.** Same pin as ADR 013 (a). Source of `Sources/SwitchcraftCore/Tokenizer/*` validation fixtures and the cross-stack `reference_embeddings.bin`. |

These SHAs were verified to exist via `gh api repos/<repo>/commits/<sha>`
on 2026-05-01. Reviewers verifying this catalogue's file/function
citations should `git checkout` the corresponding pin before reading
upstream source. Catalogue freshness becomes an audit signal — if a
later sub-issue cannot find a cited symbol at the pinned SHA, that's
a defect to flag, not a silent-update opportunity.

### Why four pins, not just ggml

Witchcraft does **not** run on raw ggml or llama.cpp. Per Witchcraft's
`Cargo.toml` at the pinned commit, the runtime is **Candle**
(`candle-core`/`candle-nn`/`candle-transformers` at git rev `5bd5618`)
with the `metal` feature enabled on Apple Silicon. Candle:

- Reads the same GGUF format ggml writes (Q4_K layout is identical;
  Candle borrowed the format from ggml).
- Implements its **own** Metal kernels in
  `candle-core/src/quantized/metal.rs`, not a copy of ggml's
  `ggml-metal.metal`.
- Implements the same per-op mixed-precision policy by analogous
  mechanism (separate kernels per dtype, dispatched per op).

Implication for this catalogue: the **algorithmic** reference is
correctly named as ggml (canonical source for Q4_K layout and the
mixed-precision recipe), but the **behavioural** reference for the
existing cross-stack fixture is Candle. The artefact map below is
ggml-primary; a "Candle behavioural-reference appendix" at the end
provides per-row Candle pointers for debugging.

---

## Artefact map

Each row maps an upstream ggml file/symbol to its destination Swift
target file under the planned `SwitchcraftMetal` library target
(introduced in #59). The "precision" column states the row's
**informational** precision contract — storage dtype / compute dtype /
accumulator dtype. **ADR 016 (sub-issue #64) promotes this matrix to
normative.** Until then, downstream sub-issues may amend a row's
precision contract if the kernel port surfaces a per-op requirement
the catalogue's first guess got wrong; such amendments must be
recorded both here and in ADR 016.

Swift destinations under `Sources/SwitchcraftMetal/` are *planned*
paths — the target itself is created by #59 (the SPI-uplift +
`@_spi(SwitchcraftMetal) public` exposure of `MetalContext` is also
landed by #59). Until that target lands, the paths in column 3 are
forward references, not existing files.

### Encoder ops

| Upstream (ggml/llama.cpp) | Swift destination | Precision (storage / compute / accum) |
|---|---|---|
| `gguf.h` reader (file format), invoked by `llama-model-loader.cpp` (llama.cpp) | `Sources/SwitchcraftMetal/GGUF/GGUFReader.swift` (#59) | n/a (file format; produces typed `MTLBuffer`s) |
| `dequantize_row_q4_K` in `src/ggml-quants.c`; Metal kernel `kernel_dequantize_q4_K` in `src/ggml-metal/ggml-metal.metal` | `Sources/SwitchcraftMetal/Kernels/Q4KDequant.swift` + `Sources/SwitchcraftMetal/Shaders/Q4KDequant.metal` (#60) | Q4_K → FP16 (compute) → FP32 (downstream consumer) |
| `kernel_mul_mat_q4_K_f32` in `src/ggml-metal/ggml-metal.metal` (Q4_K weights × FP32 activations, FP32 accumulator) | `Sources/SwitchcraftMetal/Kernels/Q4KMatMul.swift` + `Sources/SwitchcraftMetal/Shaders/Q4KMatMul.metal` (#60) | Q4_K weights / FP16 activation × dequantised FP16 weight (compute) / FP32 accumulator |
| Token embedding lookup — `kernel_get_rows_q4_K` (Metal) / equivalent CPU path in `ggml-quants.c` | folded into `Sources/SwitchcraftMetal/Embedder/T5MetalEmbedder.swift` (#64); reuses Q4KDequant for the row gather | Q4_K storage → FP32 output |
| `kernel_rms_norm_f32` in `src/ggml-metal/ggml-metal.metal` (T5 RMSNorm — variance only, no mean centering, no bias) | `Sources/SwitchcraftMetal/Kernels/RMSNorm.swift` + `Sources/SwitchcraftMetal/Shaders/RMSNorm.metal` (#61) | FP32 weight / FP32 compute / FP32 accumulator. **FP32 carve-out is non-negotiable** per ADR 003 + ADR 014(b). |
| Q/K/V/O projections — `kernel_mul_mat_q4_K_f32` (4 × 12 = 48 invocations per encode) | reuses `Q4KMatMul.swift` from #60 | as Q4KMatMul row above |
| Scaled dot-product attention — `kernel_mul_mm_f16` / `kernel_mul_mm_f32` for `Q · Kᵀ`, optional `Q8_0` K cache via `kernel_mul_mat_q8_0_f32` | `Sources/SwitchcraftMetal/Kernels/Attention.swift` + `Sources/SwitchcraftMetal/Shaders/Attention.metal` (#62) | FP16 K/V (Q8_0 K cache is a Witchcraft-side ggml choice not strictly required for our encode-only path; default to FP16 K/V) / FP32 compute / FP32 accumulator |
| Relative-position bias — T5 `_relative_position_bucket` (see "Relative-position attention bucket" below); added pre-softmax | `Sources/SwitchcraftMetal/Kernels/RelativePositionBias.swift` + `Sources/SwitchcraftMetal/Shaders/RelativePositionBias.metal` (#62) | bucket-table FP32 / FP32 compute / FP32 accumulator |
| `kernel_soft_max_f32` in `src/ggml-metal/ggml-metal.metal` (T5 attention softmax with optional mask + bias) | `Sources/SwitchcraftMetal/Kernels/Softmax.swift` + `Sources/SwitchcraftMetal/Shaders/Softmax.metal` (#62) | FP32 input / FP32 compute / FP32 accumulator. **Sensitive op — do not lower to FP16 even if it appears to work on a single fixture.** |
| FFN gate matmul (`wi_0`, gated branch) — `kernel_mul_mat_q4_K_f32` | reuses `Q4KMatMul.swift` from #60 | as Q4KMatMul row above |
| FFN up matmul (`wi_1`, paired branch) — `kernel_mul_mat_q4_K_f32` | reuses `Q4KMatMul.swift` from #60 | as Q4KMatMul row above |
| Gated-GELU activation — `gelu_new(wi_0(x)) * wi_1(x)` element-wise; ggml `kernel_gelu_f32` + element-wise multiply | `Sources/SwitchcraftMetal/Kernels/GatedGELU.swift` + `Sources/SwitchcraftMetal/Shaders/GatedGELU.metal` (#63) | FP32 input / FP32 compute (`gelu_new` formula uses `tanh`; no exp overflow at FP32 for typical activation magnitudes) / FP32 accumulator |
| FFN down matmul (`wo`) — `kernel_mul_mat_q4_K_f32` | reuses `Q4KMatMul.swift` from #60 | as Q4KMatMul row above |
| Residual add (`x_out = x_in + sublayer(LN(x_in))`) — ggml `kernel_add_f32` | `Sources/SwitchcraftMetal/Kernels/ResidualAdd.swift` + `Sources/SwitchcraftMetal/Shaders/ResidualAdd.metal` (#63) | FP32 / FP32 / FP32. **Residual stream stays at FP32 throughout** per ADR 003 + ADR 014(b). |
| Final encoder RMSNorm | reuses `RMSNorm.swift` from #61 | as RMSNorm row above |
| Projection matmul (the absorbed `2_Dense/Linear` from sentence-transformers; FP16 weights at GGUF read time, no quantisation) — `kernel_mul_mm_f16` | `Sources/SwitchcraftMetal/Kernels/ProjectionMatMul.swift` + `Sources/SwitchcraftMetal/Shaders/ProjectionMatMul.metal` (#64) | FP16 weight / FP32 compute / FP32 accumulator |
| L2 normalisation (unit norm per token) — element-wise `x / sqrt(sum(x²) + ε)`; ggml has no dedicated kernel for this combination, composed from `kernel_sqr_f32` + `kernel_sum_rows_f32` + element-wise scale | `Sources/SwitchcraftMetal/Kernels/L2Norm.swift` + `Sources/SwitchcraftMetal/Shaders/L2Norm.metal` (#63) | FP32 / FP32 / FP32. **Sensitive op** — `sum(x²)` over 768 elements has overflowed FP16 historically (ADR 014(b) point 2). |
| Orchestrator — encoder forward pass dispatch order, `MTLBuffer` lifecycle, sliding-window integration | `Sources/SwitchcraftMetal/Embedder/T5MetalEmbedder.swift` (#64) | n/a (orchestration) |

### Op-count reconciliation

The issue body's count "FFN matmuls ×24" assumed classic T5-base
(non-gated FFN, two matmuls per block). **`google/xtr-base-en` is a
T5 v1.1 derivative with gated-GELU FFN, three matmuls per block**
(`wi_0` gate, `wi_1` up, `wo` down). The corrected count is therefore
**3 × 12 = 36 FFN matmuls** in the per-op precision matrix (see
"FFN activation" below for the source of this correction).

The per-op precision matrix consequently covers: token embedding
lookup ×1, RMSNorm ×25 (12 pre-attention + 12 pre-FFN + 1 final),
Q/K/V/O projections ×48 (4 × 12), scaled dot-product attention ×12,
relative-position bias add ×12, softmax ×12, **FFN matmuls ×36**
(3 × 12, not the issue body's ×24), gated-GELU activation ×12,
residual add ×24 (12 post-attention + 12 post-FFN), final RMSNorm ×1
(folded into the ×25 above), projection matmul ×1, L2 normalisation
×1.

### Reuse from existing Switchcraft surface

These existing files are reused as-is by the embedder port. They are
**not** copied or duplicated under `SwitchcraftMetal/`:

- `Sources/SwitchcraftCore/Tokenizer/*` — token IDs are produced by
  Switchcraft's existing tokeniser (validated against
  `xtr-base-en.tokenizer.json`). See "Tokeniser disposition" below.
- `Sources/SwitchcraftCoreML/SlidingWindow.swift` — sliding-window
  decomposition for >512-token inputs (ADR 011). The port reuses
  this; it does not re-implement.
- `Sources/SwitchcraftCore/Metal/MetalContext.swift`,
  `MetalDispatch.swift`, `Shaders/MetalCoreShaders.metal` — the
  Metal foundation from #51 / ADR 015. The embedder port reuses
  `MetalContext` and pipeline cache; #59 lifts visibility to
  `@_spi(SwitchcraftMetal) public` rather than introducing a second
  context.
- `Sources/SwitchcraftCore/Embedding/Embedder.swift` — the protocol
  contract `T5MetalEmbedder` will satisfy (ADR 009). No public API
  change; just an additional conformance.

### Distinction from `Sources/SwitchcraftCore/Codec/Q4Codec.swift`

Switchcraft's existing `Q4Codec` is **not** Q4_K and **not** what
the GGUF reader produces. `Q4Codec` is a per-vector residual codec
with FP32 carve-outs for parity arithmetic (ADR 003) used by the
search-side residual decode path. The two formats share the digit
"4" and nothing else:

| | `Q4Codec` (search side) | Q4_K (embedder side) |
|---|---|---|
| Block size | per-vector | 256-element super-block |
| Scale | per-vector FP32 carve-out | 6-bit sub-block scales × 8 sub-blocks per super-block |
| Min/zero-point | none (residual is mean-centred) | per-super-block FP16 min |
| Use | bucket residual decoder | embedder weight storage |
| Source | original Switchcraft (Witchcraft's `packops.rs` port) | ggml `ggml-quants.{h,c}` |

Sub-issue #59's GGUF reader implementer must not conflate the two.
They live in different directories (`SwitchcraftCore/Codec` vs
`SwitchcraftMetal/GGUF`) for exactly this reason.

---

## Q4_K block layout reference

Pinned to `ggml/src/ggml-quants.h` and `ggml/src/ggml-quants.c` at
`b70770970e84c30a007b3859a453768b3ece2d3d`. The Q4_K format stores
**256 weights per super-block** as follows:

```c
// from ggml/src/ggml-quants.h
#define QK_K 256

typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // K_SCALE_SIZE = 12 — 6-bit packed
                                  //   scales and mins (8 sub-blocks × 6 bits each
                                  //   for scale + 8 × 6 bits for min, packed into 12 bytes)
    uint8_t qs[QK_K/2];           // 4-bit quantized weights, 256/2 = 128 bytes
} block_q4_K;
```

Super-block size = `2 * sizeof(ggml_half) + 12 + 128 = 4 + 12 + 128
= 144 bytes` per 256 weights = 4.5 bits/weight (vs 4 nominal — the
overhead is the per-super-block FP16 scale/min and the 6-bit
sub-block scales).

### Sub-block structure

Each 256-element super-block is divided into **8 sub-blocks of 32
elements**. Each sub-block has its own 6-bit scale and 6-bit min,
both packed into the `scales[12]` byte array via a custom layout
(see `get_scale_min_k4` in `ggml-quants.c`):

```c
// from ggml/src/ggml-quants.c — get_scale_min_k4
static inline void get_scale_min_k4(int j, const uint8_t * q,
                                    uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

For sub-block `j ∈ [0, 8)`:

- `j ∈ [0, 4)`: scale = low 6 bits of `scales[j]`; min = low 6 bits
  of `scales[j+4]`.
- `j ∈ [4, 8)`: scale = low 4 bits of `scales[j+4]` concatenated
  with high 2 bits of `scales[j-4]` (forming 6 bits); min = high 4
  bits of `scales[j+4]` concatenated with high 2 bits of `scales[j]`.

### Dequantisation

For element `i` in sub-block `j`:

```
sub_scale  = (sb_scale_6bit / 63.0) * d        // d is FP16 super-block scale
sub_min    = (sb_min_6bit   / 63.0) * dmin     // dmin is FP16 super-block min
weight[i]  = sub_scale * q4[i]  -  sub_min
```

where `q4[i]` is the unpacked 4-bit value at position `i`. The 128
bytes of `qs[]` pack two 4-bit values per byte; the two halves of
the super-block (first 128 weights vs second 128) interleave by
nibble, not by byte (see `dequantize_row_q4_K` in `ggml-quants.c`
for the canonical unpack sequence). Sub-issue #60 should follow that
unpack order to keep MaxSim scoring stable across the port.

The Metal version (`kernel_dequantize_q4_K` and the fused matmul
variant `kernel_mul_mat_q4_K_f32`) both live in
`ggml/src/ggml-metal/ggml-metal.metal` at the pinned commit. The
fused matmul kernel is the throughput-relevant target for #60; the
standalone dequant is useful for #60's parity tests against the C
reference.

This level of detail is intended to let #59's GGUF reader implementer
write Q4_K decode without re-deriving the layout from upstream
source. The C and Metal definitions remain canonical.

---

## Per-op precision matrix (informational)

**This matrix is informational. ADR 016 (sub-issue #64) promotes it
to normative.** Downstream sub-issues may amend a row's precision
contract if the port surfaces a per-op requirement the catalogue's
first guess got wrong; record amendments here and in ADR 016.

The precision contract per op aligns with ADR 003 (parity-arithmetic
FP32 carve-outs for sensitive ops) and ADR 014(b)/(d)/(i)
(why FP32 is mandatory for RMSNorm + residual + softmax + L2 norm
on this graph). For ops not in ADR 003's sensitive list, the
catalogue follows ggml's per-op kernel choices verbatim.

| Op | Count per encode | Weight storage | Compute | Accumulator |
|---|---|---|---|---|
| Token embedding lookup (`shared.weight` lookup) | 1 | Q4_K | FP16 dequant → FP32 | n/a (gather) |
| RMSNorm (12 pre-attn + 12 pre-FFN + 1 final) | 25 | FP32 (small, 768 floats × 25) | **FP32 — non-negotiable** | FP32 |
| Q/K/V/O projections | 48 (4 × 12 layers) | Q4_K | dequantised → FP16 | FP32 |
| Scaled dot-product attention `Q · Kᵀ / √d_k` | 12 | n/a (activations only) | FP32 (the divide-by-√d_k must be FP32 or roundoff disturbs softmax) | FP32 |
| Relative-position bias add | 12 | FP32 (32-bucket × 12-head table; ~12 KB) | FP32 | FP32 |
| Softmax | 12 | n/a | **FP32 — sensitive op** (ADR 003) | FP32 |
| FFN gate matmul `wi_0` | 12 | Q4_K | dequantised → FP16 | FP32 |
| FFN up matmul `wi_1` | 12 | Q4_K | dequantised → FP16 | FP32 |
| Gated-GELU `gelu_new(gate) * up` | 12 | n/a | FP32 (`gelu_new` uses `tanh`; FP32 has headroom) | FP32 |
| FFN down matmul `wo` | 12 | Q4_K | dequantised → FP16 | FP32 |
| Residual add | 24 (12 post-attn + 12 post-FFN) | n/a | **FP32 — residual stream stays FP32 throughout** (ADR 014(b)) | FP32 |
| Projection matmul (absorbed `2_Dense.weight`, 768 → 128) | 1 | FP16 (no Q4_K — small, ~96 KB at FP32) | FP16 | FP32 |
| L2 normalisation | 1 | n/a | **FP32 — sensitive op**, `sum(x²)` overflows FP16 (ADR 014(b) point 2) | FP32 |

**Total**: 184 op invocations per 512-token encode (or fewer per
sliding window — #64 dispatches per window, accumulating in FP32 at
the orchestrator).

The matrix matches ggml's per-op kernel choices for the same ops on
the same model. Where ggml exposes both `_f16` and `_f32` variants
of a kernel, the catalogue picks the variant ggml's runtime would
dispatch under the "FP32 sensitive ops" policy that produces
Witchcraft's reference output.

---

## Relative-position attention bucket implementation

T5's relative-position bias is **not** a sinusoidal positional
encoding. It's a per-head learned bias added to the pre-softmax
attention logits, indexed via a logarithmic bucketing scheme that
groups adjacent positions into the same bucket and spreads
exponentially for larger offsets.

### Formula (T5 reference)

From `transformers/src/transformers/models/t5/modeling_t5.py`
(`T5Attention._relative_position_bucket`), transcribed verbatim with
defaults from `google/xtr-base-en/config.json` (`num_buckets=32`,
`max_distance=128`, `bidirectional=True` for the encoder):

```python
def _relative_position_bucket(relative_position,
                              bidirectional=True,
                              num_buckets=32,
                              max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2  # → 16 for the encoder (bidirectional)
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = relative_position.abs()
    else:
        relative_position = -torch.min(relative_position,
                                       torch.zeros_like(relative_position))

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2  # → 8
    is_small = relative_position < max_exact

    # the other half are for logarithmically bigger bins in positions
    # up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position,
                                    relative_position_if_large)
    return relative_buckets
```

Bucketed result is in `[0, num_buckets)` = `[0, 32)`. The bias
table is `Embedding(num_buckets=32, num_heads=12)` (per-head learned),
gathered by bucket index, transposed to `(num_heads, q_len, k_len)`,
and added to the attention logits before softmax.

### ggml / Candle counterpart

ggml's T5 inference path (per `llama.cpp/src/llama-model.cpp` at the
pinned commit, search for `LLM_ARCH_T5`) computes the bucket table
**once per encode** (not per layer — it's input-position-dependent
but layer-invariant) and broadcasts the per-head bias across all
12 attention layers. This shared-bias pattern is also present in
Candle's `candle-transformers/src/models/t5.rs`
(`T5Attention::compute_bias`).

### Port shape (#62)

Sub-issue #62's `RelativePositionBias.swift` should:

1. Compute the bucket index table once per encoder forward pass (not
   per layer), shape `(q_len, k_len)`, dtype `int32`.
2. Gather the per-head bias from the 32 × 12 weight matrix loaded
   from GGUF, producing a `(num_heads, q_len, k_len)` FP32 tensor.
3. Add to the `Q · Kᵀ / √d_k` logits buffer pre-softmax.

The bucket function itself is integer arithmetic + a single FP32
log; no Q4 / FP16 / dtype subtleties. The catalogue records the
formula here so #62's reviewer doesn't have to re-derive it from
upstream source.

---

## FFN activation determination

**Resolved.** Read verbatim from `google/xtr-base-en/config.json` at
HuggingFace revision `main` on 2026-05-01:

```json
{
  "feed_forward_proj": "gated-gelu",
  "is_gated_act": true,
  "dense_act_fn": "gelu_new",
  "_name_or_path": "google/t5-v1_1-base",
  "d_model": 768,
  "d_ff": 2048,
  "num_layers": 12,
  "num_heads": 12,
  "d_kv": 64,
  "layer_norm_epsilon": 1e-06,
  "relative_attention_num_buckets": 32,
  "relative_attention_max_distance": 128,
  "vocab_size": 32128
}
```

This **resolves the open research question on #57**: `xtr-base-en`'s
FFN is **gated-GELU**, not classic ReLU T5-base. Two consequences:

1. **Three matmuls per FFN block, not two.** `wi_0` (gate) + `wi_1`
   (up) + `wo` (down). The per-op precision matrix uses 36 FFN
   matmul invocations (3 × 12), corrected from the issue body's
   pre-research estimate of 24. See "Op-count reconciliation"
   above.
2. **`xtr-base-en` is a T5 v1.1 derivative**, not classic T5-base.
   Layer-norm is **RMSNorm with no bias** (T5 family default);
   FFN dimensions are **768 → 2048 (gate+up) → 768 (down)**, not
   T5-base's 768 → 3072 → 768. The catalogue's per-op matrix uses
   these v1.1 dimensions. The activation `gelu_new` is the
   `0.5*x*(1+tanh(sqrt(2/π)*(x + 0.044715*x³)))` formulation
   (HuggingFace's `gelu_new`), which is what ggml's `kernel_gelu_f32`
   computes.

Sub-issue #63's gated-GELU kernel must apply `gelu_new` to the gate
branch, then element-wise multiply with the up branch — *not* a
single GELU on a single hidden tensor.

---

## GGUF asset acquisition pipeline

The Q4_K GGUF asset for `google/xtr-base-en` is **derived locally**
against pinned upstream tooling. There is no published downloadable
artefact with a stable SHA-256 fingerprint.

### Two-step pipeline (recommended)

Run from a Witchcraft checkout at `WITCHCRAFT_COMMIT =
6ad59e51cfc89bcfb20756e3f05cf9429b7cb55f`:

```bash
# 1. Download FP16 safetensors + Dense projection from HuggingFace,
#    absorb the projection, write xtr.safetensors (FP16).
python downloadweights.py

# 2. Quantise to Q4_K GGUF using Witchcraft's in-tree quantize-tool
#    (Candle-backed; pinned to candle-core 5bd5618).
cargo run -p quantize-tool xtr.safetensors assets/xtr.gguf
```

Output: `assets/xtr.gguf`, ~80 MB, Q4_K weights + FP16
projection matrix (the `2_Dense/Linear` is small and stays at FP16,
not Q4_K).

The same commit + pipeline produced
`Tests/Fixtures/reference_embeddings.{bin,json}` per ADR 013 (a),
which is what the cross-stack parity gate
(`CrossStackEmbeddingParityTests`) compares against. Pinning to this
exact pipeline keeps the post-port parity test honest — divergence
that surfaces in #65 cannot be attributed to a different
quantisation toolchain.

### SHA-256 fingerprint: omitted intentionally

Witchcraft's `quantize-tool` is **not certified bit-stable across
hosts**. Candle's Q4_K writer is deterministic for the same input,
but the input depends on the safetensors download bit-matching, the
host's float-arithmetic environment behaving identically, and the
specific `candle-core` revision that `quantize-tool` builds against
(`5bd5618` at the pinned commit). Pinning a SHA-256 risks false-
positive mismatch reports on developer machines.

Authoritative provenance is **`WITCHCRAFT_COMMIT` + Candle commit pin**
(both above). The runtime guard against drift is the cross-stack
parity test in `CrossStackEmbeddingParityTests`, not a file-level
SHA-256.

### Alternative pipeline (llama.cpp `quantize`)

The issue body named "quantise from F32 GGUF via `llama.cpp`'s
`quantize`" as a viable alternative. It **is** technically viable —
both Candle and llama.cpp write the same Q4_K layout — but the
output bytes differ (different rounding orders, different sub-block
scale fitting heuristics). Picking llama.cpp's pipeline would
require regenerating `Tests/Fixtures/reference_embeddings.bin` in
the same commit, which couples this work to a fixture refresh that
nothing else needs.

**The catalogue recommends the Witchcraft/Candle pipeline.** llama.cpp
is the algorithmic reference for the Q4_K decoder; Candle is the
generator of the bytes the decoder must read to reproduce the
existing fixture.

### Asset gating env var (preview)

Sub-issue #59 lands `SWITCHCRAFT_XTR_GGUF` as the env-var asset gate
for the Q4_K GGUF, mirroring `SWITCHCRAFT_XTR_MLPACKAGE` per ADR 010
(d). The semantics and resolution order land in **ADR 010(j)**
alongside the `SwitchcraftMetal` target. Out of scope for this
catalogue.

---

## Tokeniser disposition

`Sources/SwitchcraftCore/Tokenizer/*` (validated against
`Tests/Fixtures/xtr-base-en.tokenizer.json`, ADR 001) is the **source
of truth** for tokenisation. The Switchcraft tokeniser produces
upstream-identical token IDs to HuggingFace's reference and to
Witchcraft's `tokenizers` crate, and that property is locked in by
existing tests.

Any tokeniser fields embedded in the GGUF asset
(`tokenizer.ggml.model`, `tokenizer.ggml.tokens`, etc., emitted by
ggml's `convert-*-to-gguf.py` family of converters) are **ignored**
by the GGUF reader landed in #59. The reader extracts only the
weight tensors and the architecture metadata needed to wire up the
forward pass.

This is consistent with ADR 010: the model asset is weights; the
tokeniser is bundled separately as `xtr-base-en.tokenizer.json`.
Embedding the tokeniser in the GGUF would duplicate state that
already lives in a more accessible form.

---

## Candle behavioural-reference appendix

When a kernel port doesn't bit-match against the PyTorch FP32
reference (`Tests/Fixtures/xtr-base-en.embeddings.bin`), the next
debugging step is checking what Candle does — *not* what ggml does.
Witchcraft runs Candle, so the cross-stack fixture
(`reference_embeddings.bin`) was produced by Candle's kernels. If a
port matches Candle exactly but diverges from ggml, the resulting
Switchcraft output will still pass `CrossStackEmbeddingParityTests`.
If the reverse, it won't.

Per-row Candle pointers (commit `5bd5618c310a` of
`huggingface/candle`):

| ggml row | Candle equivalent |
|---|---|
| `kernel_dequantize_q4_K` / `dequantize_row_q4_K` | `candle-core/src/quantized/k_quants.rs::BlockQ4K::to_float` |
| `kernel_mul_mat_q4_K_f32` (Metal) | `candle-core/src/quantized/metal.rs::call_quantized_matmul_mv_t` (dispatched by dtype) |
| `kernel_rms_norm_f32` | `candle-nn/src/rms_norm.rs` (compute path) + Metal kernel in `candle-core/src/metal_kernels` |
| `kernel_soft_max_f32` (Metal) | `candle-core/src/metal_kernels/softmax.metal` |
| `kernel_gelu_f32` | `candle-nn/src/activation.rs::Activation::NewGelu` |
| `kernel_add_f32` (residual) | `candle-core/src/metal_kernels/binary.metal` |
| `kernel_get_rows_q4_K` (token embedding lookup) | `candle-core/src/quantized/metal.rs::call_quantized_index_select` |
| T5 wiring (`llama.cpp/src/llama-model.cpp` LLM_ARCH_T5 branch) | `candle-transformers/src/models/t5.rs` (whole file; encoder is `T5Stack` constructed in `T5EncoderModel::load`) |
| `_relative_position_bucket` (T5 attention) | `candle-transformers/src/models/t5.rs::T5Attention::relative_position_bucket` + `compute_bias` |

Sub-issues #60–#63 should treat ggml as the *primary* citation in
PR review notes (it has the canonical algorithmic source) and
Candle as the *secondary* check when a parity test fails. Sub-issue
#64's orchestration should mirror Candle's `T5EncoderModel` forward-
pass shape if the ggml T5 wiring is unclear — Candle is more
readable on the encoder-only path because it's not multiplexed with
decoder code.

---

## Cross-references

- Umbrella: #57 (Phase 2 — Port ggml's T5 inference to Swift + Metal).
- ADR 014(g) — amended in this same commit to record that ratchet
  sub-condition (g)(2) is being followed.
- ADR 003 — typing requirements / parity-arithmetic FP32 carve-outs;
  consistent with the per-op precision matrix above.
- ADR 009 — `Embedder` protocol; `T5MetalEmbedder` is an additional
  conformance, not a public API change.
- ADR 010 — embedder asset distribution; (a) revision policy, (d)
  env-var gating pattern, (f) `modelIdentifier` policy. **(j) lands
  with #59**, not here.
- ADR 011 — sliding window; reused as-is.
- ADR 013 — reference fixture provenance; same Witchcraft pin.
- ADR 015 — `MetalContext` and dispatch; foundation reused.
- ADR 016 — *forward reference.* Lands with #64 to promote this
  catalogue's per-op matrix to normative.
- `docs/Plan.md` — Phase 2 sub-issue checklist links here.
