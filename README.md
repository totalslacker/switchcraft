# Switchcraft

Switchcraft is a Swift package that brings token-level semantic search
(XTR-Warp / ColBERT-family) to native Apple platforms. It is a Swift port
of Dropbox's Rust [Witchcraft](https://github.com/dropbox/witchcraft) — the
reimplementation of [XTR-Warp](https://arxiv.org/abs/2501.17788) — and is
licensed under Apache 2.0.

## Status

**Pre-1.0. The public API may change.** Phase 1 is feature-complete; v0.1.0
is the first tagged release. Architectural decisions are recorded under
[`adrs/`](adrs/); the full implementation plan and progress log lives in
[`docs/Plan.md`](docs/Plan.md).

## Platform support

Switchcraft targets:

- macOS 13+
- iOS 16+
- visionOS 1+

`swift-tools-version: 6.0`. CI runs on macOS only; iOS/visionOS are supported
by the platform list but not exercised in CI yet.

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/totalslacker/switchcraft", from: "0.1.0")
```

…and depend on the products you need:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "Switchcraft", package: "switchcraft"),
        .product(name: "SwitchcraftSQLite", package: "switchcraft"),
        // Optional — only if you want the bundled CoreML embedder:
        .product(name: "SwitchcraftCoreML", package: "switchcraft"),
    ]
)
```

## Quickstart

Switchcraft is `Embedder`-agnostic. The snippet below uses a deterministic
toy embedder so it compiles and runs without any model assets — useful for
exploring the API surface. Production callers wire in
[`T5CoreMLEmbedder`](#coreml-setup) (see below).

```swift
import Switchcraft
import SwitchcraftSQLite

struct ToyEmbedder: Embedder {
    let dims = 16
    let modelIdentifier = "toy-embedder@v0"
    func encode(_ text: String) async throws -> [Float] {
        let tokens = text.lowercased().split(whereSeparator: \.isWhitespace)
        return tokens.flatMap { tok -> [Float] in
            // FNV-1a 64-bit over UTF-8 — stable across runs and Swift versions
            // (unlike Swift's seed-randomised String.hashValue).
            var h: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in tok.utf8 { h = (h ^ UInt64(byte)) &* 0x0000_0100_0000_01b3 }
            var v = (0..<dims).map { i -> Float in
                Float(Int(truncatingIfNeeded: (h &+ UInt64(i)) % 31)) - 15
            }
            let n = (v.map { $0 * $0 }.reduce(0, +)).squareRoot()
            if n > 0 { for i in v.indices { v[i] /= n } }
            return v
        }
    }
}

let store = try await SwitchcraftStore.sqlite(
    databasePath: ":memory:",
    embedder: ToyEmbedder()
)
try await store.add(id: "doc-a", body: "Apples and bananas are popular fruits.")
try await store.add(id: "doc-b", body: "Heavy rainfall is expected this evening.")
let hits = try await store.search(query: "red apples in the orchard", topK: 5)
print(hits.map(\.uuid))
try await store.shutdown()
```

For unit tests of your own code, the test target ships a deterministic
`MockEmbedder` (`Tests/SwitchcraftTests/Support/MockEmbedder.swift`) — copy
or vendor it as needed; it is intentionally test-target-only and not
exported as a public type (see ADR 009(j)).

## Package layout

| Library | Purpose |
|---------|---------|
| `Switchcraft` | Umbrella module: `SwitchcraftStore` + `Embedder` + `StoreConfig`. Most consumers `import Switchcraft`. |
| `SwitchcraftSQLite` | SQLite + FTS5 storage backend and the `SwitchcraftStore.sqlite(...)` factory. |
| `SwitchcraftCoreML` | `T5CoreMLEmbedder` — a real `Embedder` backed by the `google/xtr-base-en` CoreML model. |
| `SwitchcraftMetal` | Phase 2 Metal embedder support — GGUF v3 reader + (forthcoming) Q4_K matmul / RMSNorm / SDPA kernels and `T5MetalEmbedder`. Requires the Q4-quantised GGUF asset gated by `SWITCHCRAFT_XTR_GGUF`. |
| `SwitchcraftStorageTesting` | A reusable conformance suite for adopters writing custom `SwitchcraftStorage` backends. Test-support only. |

`SwitchcraftCore` is an internal target (re-exported by `Switchcraft`) and is
intentionally not exposed as a top-level product (per ADR 009(i)). A
backend lives in its own target so consumers only link the frameworks they
actually use (no SQLite linkage for in-memory stores; no CoreML linkage for
callers that bring their own embedder).

## CoreML setup

The pieces below are the minimum needed to run an end-to-end search with the
real `T5CoreMLEmbedder`.

### Variants

`T5CoreMLEmbedder` accepts any `.mlpackage` whose graph matches the
contract defined in [ADR 010(c)](adrs/010-embedder-model-and-asset-distribution.md).
Two variants are supported:

| Variant | Asset | Compute | Use case | Env var | `modelIdentifier` |
|---|---|---|---|---|---|
| **FP32 (default)** | `xtr-base-en.mlpackage` (~430 MB) | FP32 GPU/CPU | Maximum precision; production default | `SWITCHCRAFT_XTR_MLPACKAGE` | `google/xtr-base-en@v1` |
| **INT8 weight-only** | `xtr-base-en-int8w.mlpackage` (~110 MB) | FP32 GPU/CPU | Size-constrained (iOS, edge, OTA); opt-in | `SWITCHCRAFT_XTR_MLPACKAGE_INT8W` | `google/xtr-base-en@v1-int8w` |

The INT8w variant compresses Linear-op weights to INT8 with per-channel
scales; weights are dequantised back to FP32 just before each matmul,
so compute precision is unchanged and the within-stack parity contract
is mean cosine similarity ≥ 0.998 vs the PyTorch FP32 reference. It
ships **alongside** the FP32 default — neither variant replaces the
other. See [ADR 010(i)](adrs/010-embedder-model-and-asset-distribution.md)
for the full contract.

> **Important — `modelIdentifier`**: the two variants MUST be
> initialised with **different** `modelIdentifier` strings (recommended:
> the values in the table above). `T5CoreMLEmbedder` records the
> identifier verbatim on every persisted chunk; if the same identifier
> is used for both variants, chunks indexed under one cannot be
> distinguished from chunks indexed under the other and ADR 010(f)'s
> mismatch detection silently passes through. The API does not enforce
> distinct identifiers — it is a usage contract for operators.

Neither asset is committed to the repository: both exceed reasonable
git limits and Git LFS is incompatible with SwiftPM's resolver. See
[ADR 010(d)](adrs/010-embedder-model-and-asset-distribution.md) for
the full distribution rationale.

### 1. Build the CoreML assets

Producing the FP32 default is a one-time step. The INT8w variant is
optional and is produced by a second post-processing step against the
FP32 asset.

```bash
# 1. Install the conversion-script dependencies (used by both scripts).
pip install -r scripts/requirements-coreml.txt

# 2. Build the FP32 default. Use the HuggingFace commit SHA you want
#    pinned into the asset's metadata (recorded in ADR 010).
python3 scripts/convert-xtr-to-coreml.py \
    --revision <huggingface-commit-sha> \
    --tokenizer Tests/Fixtures/xtr-base-en.tokenizer.json \
    --out-mlpackage Tests/Fixtures/xtr-base-en.mlpackage \
    --out-fixtures Tests/Fixtures

# 3. (Optional) Build the INT8 weight-only sibling. Defaults --input
#    to $SWITCHCRAFT_XTR_MLPACKAGE; defaults --output to a sibling
#    `<input-stem>-int8w.mlpackage` next to the FP32 asset.
python3 scripts/quantize-mlpackage-int8w.py \
    --input Tests/Fixtures/xtr-base-en.mlpackage \
    --output Tests/Fixtures/xtr-base-en-int8w.mlpackage
```

The conversion script:
- Loads the encoder + the `2_Dense/` projection layer.
- Produces an FP32 `.mlpackage` whose graph emits both the raw projection
  (for the `MIN_NORM` filter) and the L2-normalised vectors.
- Runs a PyTorch ↔ CoreML parity check (mean cosine similarity ≥ 0.999)
  and aborts non-zero if it fails.
- Writes `Tests/Fixtures/xtr-base-en.embeddings.{bin,json}` — the
  PyTorch reference fixtures Swift integration tests compare both
  variants against.

The quantisation script:
- Applies `coremltools.optimize.coreml.linear_quantize_weights`
  (per-channel symmetric INT8) to the FP32 asset.
- Asserts that at least one weight tensor was actually quantised
  (a sanity check against silent no-op'ing).
- Runs an INT8w-vs-FP32 CoreML parity check (mean cosine similarity
  ≥ 0.998) and aborts non-zero if it fails.

### 2. Place the asset(s) and point the test suite at them

The conventional location is `Tests/Fixtures/`, but any path works.
The test suite reads `SWITCHCRAFT_XTR_MLPACKAGE` for the FP32 suite
and `SWITCHCRAFT_XTR_MLPACKAGE_INT8W` for the INT8w suite:

```bash
export SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage
# Optional — only needed to run CoreMLInt8wParityTests.
export SWITCHCRAFT_XTR_MLPACKAGE_INT8W=$PWD/Tests/Fixtures/xtr-base-en-int8w.mlpackage
swift test
```

When either env var is unset or points to a non-existent path, the
corresponding asset-gated tests skip cleanly via Swift Testing's
`.enabled(if:)` trait — fresh checkouts stay green regardless of which
variants are present.

### 3. Wire the embedder into a store

The same `T5CoreMLEmbedder` API loads both variants — pick a `modelURL`
and pass the matching `modelIdentifier`:

```swift
import Switchcraft
import SwitchcraftSQLite
import SwitchcraftCoreML

let tokenizer = try Tokenizer(contentsOf: "/path/to/xtr-base-en.tokenizer.json")

// FP32 default — maximum precision.
let fp32Embedder = try await T5CoreMLEmbedder(
    modelURL: URL(fileURLWithPath: "/path/to/xtr-base-en.mlpackage"),
    tokenizer: tokenizer,
    computeUnits: .all,                            // .cpuOnly on constrained HW
    modelIdentifier: "google/xtr-base-en@v1"       // recorded on every chunk
)

// INT8 weight-only — ~3.9× smaller, FP32 compute unchanged. Note the
// distinct modelIdentifier (REQUIRED — see "Variants" above).
let int8wEmbedder = try await T5CoreMLEmbedder(
    modelURL: URL(fileURLWithPath: "/path/to/xtr-base-en-int8w.mlpackage"),
    tokenizer: tokenizer,
    computeUnits: .all,
    modelIdentifier: "google/xtr-base-en@v1-int8w"
)

let store = try await SwitchcraftStore.sqlite(
    databasePath: "/path/to/store.db",             // or ":memory:"
    embedder: fp32Embedder                         // or int8wEmbedder
)

try await store.add(id: "doc-a", body: "Apples and bananas are popular fruits.")
try await store.add(id: "doc-b", body: "Heavy rainfall is expected this evening.")
try await store.index()

let hits = try await store.search(
    query: "I picked some red apples from the orchard.",
    topK: 5
)
// hits[0].uuid == "doc-a"

try await store.shutdown()
```

`T5CoreMLEmbedder` handles tokenisation, sliding-window inference for
inputs longer than 512 tokens, and the pre-normalisation L2 norm
filter that strips low-signal positions (see
[ADR 011](adrs/011-sliding-window-long-input-strategy.md)).

## Metal embedder setup

The `SwitchcraftMetal` target is the Phase 2 port of ggml's T5
inference to Swift + Metal (umbrella issue #57). It ships alongside —
not in place of — the CoreML embedder. Sub-issue #59 landed the GGUF
reader + Q4_K CPU dequantisation reference; sub-issues #60–#63 added
the kernels (Q4_K matmul, RMSNorm, softmax, residual add, gated-GELU,
L2 norm); sub-issue #64 lands the FP32 attention/projection matmul and
the `T5MetalEmbedder` orchestrator.

### The asset

The Metal embedder consumes a Q4-quantised GGUF v3 file (~80 MB)
produced by Witchcraft's `quantize-tool` against the upstream
`google/xtr-base-en` weights. The asset is **not committed** for the
same size and Git LFS reasons as the CoreML `.mlpackage`; see
[ADR 010(j)](adrs/010-embedder-model-and-asset-distribution.md) and
[ADR 016](adrs/016-gguf-asset-distribution.md) for the distribution
rationale.

The asset acquisition pipeline is documented in
[`docs/porting/ggml-t5.md`](docs/porting/ggml-t5.md) §"Asset
acquisition". In short: run the Witchcraft `quantize-tool` (Candle-
backed) against the FP32 weights to produce `xtr-base-en.gguf`, place
it anywhere convenient.

### Running the asset-gated tests

```bash
export SWITCHCRAFT_XTR_GGUF=$PWD/Tests/Fixtures/xtr-base-en.gguf
# Optional — enables bit-equal Q4_K decode parity vs an FP32 reference
# dump (see ADR 016 §"Bit-equal Q4_K decode").
# export SWITCHCRAFT_XTR_GGUF_FP32_REF=$PWD/Tests/Fixtures/xtr-base-en.fp32-ref.json
swift test --filter SwitchcraftMetalTests
```

When `SWITCHCRAFT_XTR_GGUF` is unset or points at a non-existent path,
the round-trip parity suite skips cleanly via Swift Testing's
`.enabled(if:)` trait — fresh checkouts stay green. Header-parsing,
mixed-dtype, and Q4_K decode unit tests run unconditionally on
in-memory fixtures.

### Using `T5MetalEmbedder`

```swift
import SwitchcraftCore
import SwitchcraftCoreML
@_spi(SwitchcraftMetal) import SwitchcraftMetal

let tokenizerURL = URL(fileURLWithPath: "Tests/Fixtures/xtr-base-en.tokenizer.json")
let tokenizer = try Tokenizer(contentsOf: tokenizerURL.path)

let ggufURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SWITCHCRAFT_XTR_GGUF"]!)

// `T5MetalEmbedder.init` throws `metalUnavailable` when Metal is
// unreachable (no GPU, `SWITCHCRAFT_FORCE_ACCELERATE=1`, library load
// fail). Catch the throw and fall back to the CoreML embedder so the
// app stays usable on hosts where Metal isn't viable.
let embedder: any Embedder
do {
    embedder = try await T5MetalEmbedder(modelURL: ggufURL, tokenizer: tokenizer)
} catch T5MetalEmbedderError.metalUnavailable {
    let mlpackageURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SWITCHCRAFT_XTR_MLPACKAGE"]!)
    embedder = try await T5CoreMLEmbedder(modelURL: mlpackageURL, tokenizer: tokenizer)
}

// Plug into SwitchcraftStore the same way as T5CoreMLEmbedder; the
// `Embedder` protocol contract is identical (ADR 009). The Metal
// embedder records `modelIdentifier = "google/xtr-base-en@v1+gguf"`
// to distinguish embeddings produced by the two paths (ADR 010(c)).
```

The Metal embedder is `@_spi(SwitchcraftMetal) public` rather than full
public — see [ADR 016 §"`@_spi(SwitchcraftMetal)` import pattern"](adrs/016-gguf-asset-distribution.md).
Per-op precision routing follows [ADR 017](adrs/017-per-op-precision-routing.md);
the cross-stack ≥ 0.99999 cosine gate against PyTorch FP32 is the
correctness contract.

## Running the tests

```bash
# Always-on suite (fixture-driven; no model asset required).
swift test

# Run only the sliding-window planner unit tests.
swift test --filter SlidingWindow

# Asset-gated integration suite (requires SWITCHCRAFT_XTR_MLPACKAGE).
SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage \
    swift test --filter T5CoreMLEmbedder
```

Performance-sensitive tests should be run in release configuration
(`swift test -c release`).

### Forcing the Accelerate fallback

The Phase 2 search-path Metal kernels (umbrella #50) install a
transparent fallback to the existing Accelerate path: any Metal
failure (no GPU, library load fail, dispatch error) silently routes
back to `cblas_sgemm` so callers see no behaviour change. The
`SWITCHCRAFT_FORCE_ACCELERATE` env var forces that fallback path even
when Metal is available, so tests can exercise it on Metal-capable
hosts:

```bash
SWITCHCRAFT_FORCE_ACCELERATE=1 swift test --filter Metal
```

The Metal test suites use the same env var as part of their
`.enabled(if:)` gating, so they skip cleanly when it is set. See
[ADR 015](adrs/015-metal-context-and-dispatch.md) for the rationale.

### NFCorpus NDCG@10 parity benchmark

`NFCorpusBenchmarkTests` is the cross-implementation quality gate:
it indexes the NFCorpus test split through Switchcraft and asserts
macro-averaged NDCG@10 lands in Witchcraft's published [0.31, 0.33]
band (per ADR 006).

The NFCorpus dataset is **not committed** to this repository — its
license is academic-use-only and incompatible with Switchcraft's
Apache 2.0 release intent. The suite is double-gated on
`SWITCHCRAFT_XTR_MLPACKAGE` *and* `SWITCHCRAFT_NFCORPUS_DIR`. When
either env var is unset (or the expected files are missing), the
benchmark skips cleanly. CI sets neither, so it never runs there.

To run it locally, obtain NFCorpus under whatever terms you accept
and place these three plaintext files into a directory:

- `nfcorpus.tsv` — corpus rows (`docid \t title \t body`)
- `questions.test.tsv` — dev queries (`query-id \t query`)
- `qrels.test.json` — pytrec_eval-style nested relevance judgments
  (`{ qid: { docid: grade } }`)

`scripts/fetch-nfcorpus.sh` is one developer convenience that pulls
these from upstream Witchcraft's pinned commit and decompresses them
in place — see [`scripts/README.md`](scripts/README.md) for details.

Then:

```bash
export SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage
export SWITCHCRAFT_NFCORPUS_DIR=/path/to/nfcorpus

swift test --filter NFCorpusBenchmark
```

Expect ~6 minutes on Apple Silicon for the one-time CoreML T5 index
build (~3,633 abstracts) before the assertion runs.

## License and attribution

Switchcraft is licensed under the [Apache License 2.0](LICENSE). It ports
algorithm and data-structure code from Dropbox/Witchcraft (Apache 2.0) and
uses model architecture and weights from `google/xtr-base-en` (Apache 2.0).
See [`NOTICE`](NOTICE) for the full third-party attribution required by
Apache 2.0 §4(d).

## References

- [`docs/Plan.md`](docs/Plan.md) — full implementation plan and
  progress log.
- [`adrs/`](adrs/) — architecture decisions (model + asset
  distribution, sliding-window, hybrid fusion, etc.).
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contributor and release process.
- [Witchcraft (Rust upstream)](https://github.com/dropbox/witchcraft).
- [XTR-Warp paper (SIGIR'25)](https://arxiv.org/abs/2501.17788).
- [XTR paper](https://arxiv.org/abs/2304.01982).
