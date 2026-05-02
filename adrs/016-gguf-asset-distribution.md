# ADR 016 — GGUF asset distribution for the Phase 2 Metal embedder

**Status**: Accepted
**Date**: 2026-05-01
**Issue**: #59 (sub-issue of umbrella #57, Phase 2 Metal embedder — port of ggml's T5 inference to Swift + Metal)

This ADR records the shape of the GGUF v2 or v3 asset that the Phase 2
Metal embedder consumes, the env-var test gate (`SWITCHCRAFT_XTR_GGUF`), the
relationship to the existing CoreML asset gate (ADR 010), and the
deliberate carve-outs we make for Phase 2 — `MTLBuffer` storage mode,
metadata KV surface, bit-equal Q4_K decode, and the `@_spi` import
pattern.

ADR 010 owns the embedder asset distribution policy generally; this ADR
extends it with a (j) section dedicated to the GGUF asset (see ADR
010(j)). ADR 015 owns the `MetalContext` foundation that this
sub-issue's `@_spi(SwitchcraftMetal)` hatch widens. ADR 014(g)(2) is
the cross-stack precision ratchet under which the whole Phase 2 Metal
embedder sits.

---

## (a) Context

The Witchcraft Rust upstream loads `google/xtr-base-en` weights from a
Q4-quantised GGUF asset (~80 MB). Switchcraft has no GGUF reader today.
Without one, none of the kernel sub-issues (#60–#63) or the
orchestration sub-issue (#64) can be parity-tested against the actual
Witchcraft asset.

The catalogue (`docs/porting/ggml-t5.md`, landed via PR #66) pins the
upstream commit, the Q4_K block layout, the FFN activation choice,
and the asset-acquisition pipeline. This ADR codifies the
Switchcraft-side reader shape and distribution policy that
operationalise the catalogue.

## (b) Reader scope (Phase 2 minimum viable)

The reader supports exactly three tensor dtypes — the set XTR-base-en
needs:

- **Q4_K** — 4-bit K-quant, 256-element super-blocks (FP16 super-block
  scale + min, 12 bytes packed 6-bit sub-scales/mins, 128 bytes packed
  4-bit weights = 144 bytes per super-block). Used for all matmul
  weights.
- **F32** — small per-layer RMSNorm gain vectors and the relative-
  position bias table.
- **F32** — the `2_Dense` projection weight (768 × 128). Witchcraft's
  `quantize-tool` writes this tensor as FP32 after the carve-out in
  `scripts/witchcraft-fixture-export.patch` (see issue #74). The reader
  also accepts FP16 here as a fallback for assets produced without the
  carve-out.

Other ggml dtypes (Q5_K, Q6_K, Q8_0, etc.) throw
`GGUFError.unsupportedDType(name:raw:)` naming the offending tensor.
This is deliberate: Phase 2 ships exactly the format the canonical
asset uses; widening the reader to dtypes Switchcraft does not consume
adds maintenance burden without enabling any concrete consumer.

**GGUF version: v2 and v3 both accepted (amended by issue #74).** The
canonical Witchcraft pipeline uses `quantize-tool` built against Candle
rev `5bd5618`, which emits **GGUF v2**. The reader was initially strict
v3-only; issue #74 diagnosed this as defect (3) and relaxed the check to
accept v2 ∪ v3. The structural layouts are identical for Q4_K, F32, and
F16 — the only observable difference is the version field. Rejecting v2
would require owning the upstream writer or patching Candle, both of
which are worse than a one-line version-range check. `GGUFReader.loadedVersion`
exposes the on-disk version for future debugging; if v4 ever appears
and diverges structurally, the version field will surface the discrepancy
rather than silently producing corrupt tensors.

## (c) Lazy upload, eager parse

`GGUFReader.init(url:device:)` parses the header / KV / tensor-info
blocks synchronously (it is a one-time cost, no faster to defer).
Tensor `MTLBuffer` allocation and byte upload are deferred to the
first `tensor(_:)` call and cached thereafter. This supports two
shapes equally well:

- **Encoder-only tests** that touch ~3 named tensors (this sub-issue's
  round-trip parity test) pay only those uploads.
- **Full encoder load** at `T5MetalEmbedder` construction (#64) walks
  the parsed name list and warms every tensor up front.

## (d) `MTLBuffer` storage mode

Phase 2 uses `.storageModeShared` for every tensor buffer.
`.storageModePrivate` (with a one-shot blit at upload time) would put
the bytes in GPU-local memory and is plausibly faster for the matmul
kernel, but:

1. Phase 2 does not yet have measurable kernel performance — the
   matmul kernel lands in #60.
2. The Q4_K bytes are read-only on the GPU and read-only on the CPU
   (the reader and the round-trip parity test); shared mode is
   correct, just not necessarily peak.
3. Switching modes is a one-line change inside the reader.

The decision is to revisit in #60 once the matmul kernel can be
benchmarked end-to-end.

## (e) Metadata KV surface

GGUF v3 has 13 value types (`u8/i8/u16/i16/u32/i32/f32/bool/string/
array/u64/i64/f64`). The reader parses all 13 (so unknown KVs do not
desync the file offset) but exposes only the typed subset Switchcraft
uses:

- `string`
- `int64` (i32 promoted on read; i64 directly)
- `uint64` (u32 promoted; u64 directly)
- `float32`
- `float64`
- `bool`

`u8`/`i8`/`u16`/`i16` are skipped (no current consumer; promoting them
would add ambiguity to the typed surface). Arrays are recursively
consumed but not surfaced — the catalogue's "Tokeniser disposition"
forbids using GGUF tokeniser metadata anyway, and Switchcraft has no
other array-typed metadata consumer today. Adding a variant later is
non-breaking.

## (f) Bit-equal Q4_K decode

The CPU dequantisation reference (`Q4KDecode.dequantise(blocks:into:)`)
is a one-to-one transliteration of `dequantize_row_q4_K` from
`ggml/src/ggml-quants.c` at the catalogue's pinned commit
(`b70770970e84c30a007b3859a453768b3ece2d3d`). The implementation:

- Uses straight `Float` ops — no `simd` shortcuts.
- Avoids `@_transparent` and `@inline(__always)` on the hot loop body
  to keep the compiler from rearranging the multiply/add sequence.
- Reads the FP16 super-block scale / min via `Float16(bitPattern:)`
  followed by `Float(...)` widening (matches `GGML_FP16_TO_FP32` in
  the upstream code).

The expected result: the Swift output is bit-equal to the C reference
under IEEE-754 round-to-nearest-even because Swift compilers do not
auto-fuse FP32 multiplies and adds into FMA. The round-trip parity
test (`GGUFReaderParityTests`) asserts this directly when an FP32
reference dump is provided via `SWITCHCRAFT_XTR_GGUF_FP32_REF`.

If bit-equal ever proves unachievable in practice (e.g. the reference
dump path computes in a different op order, or a future Swift
compiler version starts auto-fusing FP32 ops on this target), the
test downgrades to a documented `≤ 1 ULP relative` gate **in the
test source**, not silently in CI config. The first-line investigation
in that case is to compare op orders against the reference dump's
producer (likely `llama.cpp`'s `gguf-dump --print-data` or a Candle
helper).

## (g) Distribution = local placement + env-var test gate

Mirrors ADR 010(d). The Q4_K asset is **not committed**:

- ~80 MB exceeds reasonable git limits (same constraint as the
  CoreML `.mlpackage`).
- Git LFS is incompatible with SwiftPM's resolver.
- A `binaryTarget(url:checksum:)` distribution channel is the long-
  term answer for downstream consumers but is deferred to the open-
  source release milestone — Phase 2 does not require it because the
  reader is exercised by env-gated tests only.

Until the binary-target work happens, the asset lives outside the
repo:

1. Run the Witchcraft `quantize-tool` (Candle-backed) against the
   FP32 weights to produce `assets/xtr.gguf`. The catalogue
   (`docs/porting/ggml-t5.md` §"Asset acquisition") documents the
   pipeline; no SHA-256 is pinned because Candle's quantiser is not
   certified bit-stable across hosts.
2. Place the GGUF anywhere convenient (e.g.
   `Tests/Fixtures/xtr-base-en.gguf`, but any path works).
3. Tests that need the asset read its path from
   `SWITCHCRAFT_XTR_GGUF`. When the env var is unset or points to a
   non-existent path, asset-gated suites are `.disabled` via
   `.enabled(if: GGUFAsset.isAvailable)` — fresh checkouts stay
   green.

The runtime parity guard against asset drift is the cross-stack
parity test landing in #65, not a file-level fingerprint.

## (h) `@_spi(SwitchcraftMetal)` import pattern

This sub-issue establishes the first use of `@_spi(...)` in the
codebase. The hatch widens `MetalContext` (and `MetalContextError`)
from `internal` to `@_spi(SwitchcraftMetal) public` so the new
`SwitchcraftMetal` target can reuse the existing pipeline cache
without re-foundation.

Consumers of the SPI must import with the matching annotation:

```swift
@_spi(SwitchcraftMetal) import SwitchcraftCore
```

Tests that touch `MetalContext` or `MetalContextError` use:

```swift
@_spi(SwitchcraftMetal) @testable import SwitchcraftCore
```

A missing annotation produces a "use of undeclared type" compile
error rather than the more familiar "use of internal symbol" — the
diagnostic is opaque and is the main ergonomic cost of the SPI hatch.
Sub-issues #60–#64 will all need this annotation; this ADR is the
documentation of record so the pattern does not get re-derived per
sub-issue.

The SPI hatch deliberately does **not** expand the stable public API
surface (ADR 009). External consumers cannot reach the SPI without
opting into Swift's underscore-prefixed import attribute, which is
not part of the public language surface.

## (i) Open follow-ups

- **`scripts/fetch-xtr-gguf.sh`** — a developer-convenience helper
  mirroring `scripts/fetch-nfcorpus.sh`. Out of scope for this
  sub-issue per the spec; tracked as a follow-up.
- **`binaryTarget(url:checksum:)` distribution** — same blocker as
  ADR 010(d); deferred to the open-source release milestone.
- **`.storageModePrivate`** — revisit in #60 once the Q4_K matmul
  kernel can be benchmarked end-to-end.
- **Reference FP32 dump producer script** — the round-trip parity
  test reads
  `SWITCHCRAFT_XTR_GGUF_FP32_REF` if set; the producer is currently
  documented but uncommitted. Tracked as a follow-up alongside the
  fetch script.
