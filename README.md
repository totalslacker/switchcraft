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
