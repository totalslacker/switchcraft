# ADR 010 — Embedder model + asset distribution

**Status**: Accepted
**Date**: 2026-04-29
**Issue**: #18 (Phase 1: T5/CoreML embedder)

This ADR records the canonical embedding model, the CoreML asset
format, the parity gates that asset must clear, and how the asset is
distributed to test machines and downstream consumers. Cite this ADR
before swapping the model, changing the conversion pipeline, or
introducing a new distribution channel.

---

## (a) Canonical model = `google/xtr-base-en`

Switchcraft Phase 1 uses [`google/xtr-base-en`](https://huggingface.co/google/xtr-base-en)
on HuggingFace. It is the same checkpoint upstream Witchcraft converts
to safetensors for its Candle-based runtime, so per-token embeddings
must be numerically equivalent (within FP precision) for the same
inputs.

The repository ships:

- `model.safetensors` — the T5-base encoder (~439 MB, FP32).
- `2_Dense/pytorch_model.bin` — a `768 → 128` weight-only `nn.Linear`
  (no bias, identity activation; ~395 kB).
- `tokenizer.json` — the same Unigram tokenizer Switchcraft already
  validates against in `Tests/Fixtures/xtr-base-en.tokenizer.json`.

Both `model.safetensors` and `2_Dense/pytorch_model.bin` MUST be
loaded by the conversion script; the projection is what produces the
128-dim output the index and search layers consume.

License is Apache 2.0, compatible with Switchcraft's release license.

## (b) Pinned HuggingFace revision

The conversion script (`scripts/convert-xtr-to-coreml.py`) takes a
`--revision <commit-sha>` argument so the asset is deterministically
reproducible. Record the SHA used to produce the canonical asset
shipped via release artefacts in this ADR's commit log; the
release-asset workflow MUST embed the SHA into the
`mlmodel.short_description` so a reviewer can confirm provenance with
`coremltools`.

If the upstream HuggingFace repo is renamed, gated, or removed, the
asset can be rebuilt from a mirror provided the SHA matches.

## (c) Asset format = `.mlpackage`, FP32 compute / FP16 outputs, dual output

The conversion script emits a `.mlpackage` (mlprogram backend) with:

- **Input** `input_ids` — `Int32 [1, 512]` (HuggingFace token IDs;
  pad-id = 0). Sequence length is fixed; sliding-window logic in the
  Swift wrapper handles longer inputs (ADR 011).
- **Output** `raw_projected` — `Float16 [1, 512, 128]` post-projection,
  pre-L2-norm. Used by the Swift wrapper to compute per-token L2
  norms for the `MIN_NORM` filter.
- **Output** `normalised` — `Float16 [1, 512, 128]` post-projection,
  post-L2-norm. The unit-vector tensor that callers ultimately
  consume.

Two outputs are emitted because:

1. The spec requires L2 normalisation **baked into the CoreML graph**
   so the Swift caller never re-normalises.
2. The spec also requires dropping tokens whose **pre-normalisation**
   L2 norm is below `MIN_NORM = 1.0`; that decision must be made on the
   raw projection, not on a post-norm vector (which is always 1.0 or
   undefined for zero vectors).

Emitting both vectors is simpler than emitting a per-token norm scalar
and is well-supported by `coremltools`.

### Precision contract: FP32 compute, FP16 outputs

The conversion script sets `compute_precision=ct.precision.FLOAT32`
and emits FP16-typed output ports. **Compute and weight storage are
both FP32; only the output dtype is FP16.** This deviates from the
"FP16 weights everywhere" intent of the original ADR (issue #18) and
inflates the `.mlpackage` from the originally-budgeted ~80 MB to
~430 MB.

Why we accept the size hit: the `.mlpackage` produced with
`compute_precision=ct.precision.FLOAT16` (the previous setting)
silently returns NaN-filled outputs at `MLModel.prediction(_:)` time
on this graph. Two precision-sensitive regions overflow / collapse
under blanket FP16:

1. **T5's known activation outliers** propagate through attention
   matmul and RMSNorm `pow → reduce_mean → rsqrt`, producing inf/NaN
   that contaminates downstream layers. Same pattern as
   `huggingface/diffusers#8604`.
2. **The output L2-normalisation** (`vector_norm` on a 128-dim
   projection) sums 128 squared values; with FP16 inputs this can
   exceed FP16's ~65 504 ceiling.

Per-op FP32 carve-outs around the L2 region were tried first
(coremltools' `FP16ComputePrecision(op_selector=…)` keeping `pow`,
`rsqrt`, `reduce_sum`, `real_div`, etc. at FP32). They fix the
output-region overflow but do not stabilise T5's encoder body —
the same NaN reproduces against coremltools 7.2 and 9.0. The smallest
working carve-out is "all of it": FP32 compute throughout.

Output-port FP16 is preserved so the Swift wrapper's existing
contract (`MLMultiArray` reads as `Float16 [1, 512, 128]`, widened
to FP32 at the boundary by `CoreMLModelIO`) keeps working unchanged.

A future revisit (separate issue) would compress weights to FP16
storage via a custom MIL pass while keeping compute at FP32 — that
would halve the asset to ~215 MB. Treat the current 430 MB size as
correctness-first; size optimisation is deferred. See issue #31 for
the full diagnostic and decision log.

## (d) Distribution = local placement + env-var test gate

The `.mlpackage` is **not committed** to the repository:

- ~80 MB exceeds reasonable git limits.
- Git LFS is incompatible with SwiftPM's dependency resolver — there
  is no SwiftPM-friendly path to ship a large binary alongside source
  in the same repository today.
- A SwiftPM `binaryTarget(url:checksum:)` pointing at a release ZIP is
  the long-term answer for downstream consumers, but is deferred to
  the open-source release milestone (out of scope for #18).

Until then, the asset lives **outside the repo**:

1. Run `python3 scripts/convert-xtr-to-coreml.py --revision <sha> ...`
   on a machine with the model weights. The script produces the
   `.mlpackage` and the `Tests/Fixtures/xtr-base-en.embeddings.{bin,json}`
   reference fixtures (small enough to commit).
2. Place the `.mlpackage` anywhere convenient. The convention
   documented in `README.md` is `Tests/Fixtures/xtr-base-en.mlpackage/`,
   but any path works.
3. Tests that need the asset read its path from
   `SWITCHCRAFT_XTR_MLPACKAGE`. When the env var is unset or points to
   a non-existent path, asset-gated suites are `.disabled` via
   `.enabled(if: CoreMLAsset.isAvailable)` — fresh checkouts stay
   green.

## (e) Conversion-script provenance and parity gate

`scripts/convert-xtr-to-coreml.py` is the single source of truth for
the asset. It:

1. Loads the encoder + projection.
2. Wraps them in a single `nn.Module` whose `forward` returns
   `(raw_projected, normalised)`.
3. Traces with `torch.jit.trace` and converts via `coremltools.convert`
   targeting `macOS13` minimum deployment.
4. Runs a **PyTorch ↔ CoreML parity check** on a fixed reference set
   (`FIXTURE_INPUTS`): mean cosine similarity must be `≥ 0.999` per
   fixture, otherwise the script exits non-zero. The parity gate is
   the primary correctness barrier; silent numerical drift in the
   conversion would degrade search quality without obvious test
   failures.
5. Emits `Tests/Fixtures/xtr-base-en.embeddings.bin` (raw FP32
   reference rows produced by the **PyTorch** model after the
   `MIN_NORM` filter) plus a JSON index (`xtr-base-en.embeddings.json`)
   describing each input string, byte offset, row count, and the
   `<model>@<sha>` tag.

The fixture set MUST cover (per the spec):

- a short sentence,
- a multi-sentence paragraph,
- a whitespace-only string (zero rows),
- a string longer than 512 tokens (exercises sliding-window).

The long-input fixture is the opening paragraph of Mary Shelley's
*Frankenstein* (Project Gutenberg eBook #84, public domain). The exact
text is hardcoded in the conversion script's `FIXTURE_INPUTS` list so
the fixture is reproducible from the script alone.

## (f) `modelIdentifier` policy

`T5CoreMLEmbedder.modelIdentifier` is recorded on every persisted
`ChunkRecord.model` (per ADR 009(d)) so future reads can detect
mismatches with the model that wrote them.

The default value is `"google/xtr-base-en@v1"` — short and
human-readable. Callers wanting precise revision tracking pass
`"google/xtr-base-en@<commit-sha>"` at init.

We deliberately avoid hashing the `.mlpackage` directory to derive the
identifier: it would add ~1s of startup cost on an 80 MB asset, and
the SHA derived from the local copy is not portable across machines
that may have re-compiled the model. A manually-passed tag is simpler
and gives callers control over the granularity of mismatch detection.

## (g) Compute units

`T5CoreMLEmbedder` exposes `computeUnits: MLComputeUnits` at init
(default `.all`). `.all` lets CoreML route through ANE / GPU / CPU as
it sees fit and is the right default on modern Apple Silicon.

Constrained or older hardware (Intel macOS, low-memory devices) may
prefer `.cpuOnly`; the parameter exists so callers can override
without subclassing or rebuilding the asset. Per-call float-equality
is only required *within the same loaded model instance* — switching
compute units across instances may produce last-bit drift, which is
acceptable.
