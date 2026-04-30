# Switchcraft

Switchcraft is a Swift port of [Witchcraft](https://github.com/dropbox/witchcraft)
(Dropbox, Apache 2.0), the Rust reimplementation of [XTR-Warp](https://arxiv.org/abs/2501.17788).
It brings token-level semantic search with sub-linear retrieval to
native Apple platforms (macOS, iOS, iPadOS, visionOS).

License: Apache 2.0.

## Status

Phase 1 — feature-complete locally. The full implementation plan lives
at [`docs/Plan.md`](docs/Plan.md); architectural decisions are recorded
under [`adrs/`](adrs/).

## Package layout

| Library | Purpose |
|---------|---------|
| `Switchcraft` | Umbrella module: `SwitchcraftStore` + `Embedder` + `StoreConfig`. Most consumers `import Switchcraft`. |
| `SwitchcraftCore` | Engine internals (indexer, search, codecs, tokenizer). Re-exported by `Switchcraft`. |
| `SwitchcraftSQLite` | SQLite + FTS5 storage backend and the `SwitchcraftStore.sqlite(...)` factory. |
| `SwitchcraftCoreML` | `T5CoreMLEmbedder` — a real `Embedder` backed by the `google/xtr-base-en` CoreML model. |

A backend lives in its own target so consumers only link the
frameworks they actually use (no SQLite linkage for in-memory stores;
no CoreML linkage for callers that bring their own embedder).

## Getting Started

The pieces below are the minimum needed to run an end-to-end search
with the real embedder. For experimenting without the model, plug in
`MockEmbedder` (see `Tests/SwitchcraftTests/Support/MockEmbedder.swift`).

### 1. Build the CoreML asset

The `google/xtr-base-en` `.mlpackage` is ~80 MB and is not committed
to the repository (Git LFS is incompatible with SwiftPM's resolver;
see [ADR 010](adrs/010-embedder-model-and-asset-distribution.md) for
the full distribution rationale).

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
import SwitchcraftCore

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

## References

- [`docs/Plan.md`](docs/Plan.md) — full implementation plan and
  progress log.
- [`adrs/`](adrs/) — architecture decisions (model + asset
  distribution, sliding-window, hybrid fusion, etc.).
- [Witchcraft (Rust upstream)](https://github.com/dropbox/witchcraft).
- [XTR-Warp paper (SIGIR'25)](https://arxiv.org/abs/2501.17788).
- [XTR paper](https://arxiv.org/abs/2304.01982).
