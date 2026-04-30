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

### 1. Build the CoreML asset

The `google/xtr-base-en` `.mlpackage` is ~80 MB and is not committed to the
repository (Git LFS is incompatible with SwiftPM's resolver; see
[ADR 010](adrs/010-embedder-model-and-asset-distribution.md) for the full
distribution rationale).

To produce it once:

```bash
# 1. Install the conversion-script dependencies.
pip install -r scripts/requirements-coreml.txt

# 2. Run the conversion. Use the HuggingFace commit SHA you want pinned
#    into the asset's metadata (recorded in ADR 010).
python3 scripts/convert-xtr-to-coreml.py \
    --revision <huggingface-commit-sha> \
    --tokenizer Tests/Fixtures/xtr-base-en.tokenizer.json \
    --out-mlpackage Tests/Fixtures/xtr-base-en.mlpackage \
    --out-fixtures Tests/Fixtures
```

The script:
- Loads the encoder + the `2_Dense/` projection layer.
- Produces an `.mlpackage` whose graph emits both the raw projection
  (for the `MIN_NORM` filter) and the L2-normalised vectors.
- Runs a PyTorch ↔ CoreML parity check (mean cosine similarity ≥ 0.999)
  and aborts non-zero if it fails.
- Writes `Tests/Fixtures/xtr-base-en.embeddings.{bin,json}` — the
  reference fixtures Swift integration tests compare against.

### 2. Place the asset and point the test suite at it

The conventional location is `Tests/Fixtures/xtr-base-en.mlpackage/`,
but any path works. The test suite reads `SWITCHCRAFT_XTR_MLPACKAGE`:

```bash
export SWITCHCRAFT_XTR_MLPACKAGE=$PWD/Tests/Fixtures/xtr-base-en.mlpackage
swift test
```

When the env var is unset or points to a non-existent path,
asset-gated tests skip cleanly via Swift Testing's `.enabled(if:)`
trait — fresh checkouts stay green.

### 3. Wire the embedder into a store

```swift
import Switchcraft
import SwitchcraftSQLite
import SwitchcraftCoreML

let modelURL  = URL(fileURLWithPath: "/path/to/xtr-base-en.mlpackage")
let tokenizer = try Tokenizer(contentsOf: "/path/to/xtr-base-en.tokenizer.json")

let embedder = try await T5CoreMLEmbedder(
    modelURL: modelURL,
    tokenizer: tokenizer,
    computeUnits: .all,                            // .cpuOnly on constrained HW
    modelIdentifier: "google/xtr-base-en@v1"       // recorded on every chunk
)

let store = try await SwitchcraftStore.sqlite(
    databasePath: "/path/to/store.db",             // or ":memory:"
    embedder: embedder
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
