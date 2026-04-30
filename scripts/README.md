# Switchcraft Scripts

This directory holds developer-facing helpers that are not part of the
shipping Swift package: model conversion, tokenizer-fixture
regeneration, and reference-score generation for the cross-implementation
parity tests.

## `convert-xtr-to-coreml.py`

Converts the `google/xtr-base-en` PyTorch encoder into a `.mlpackage`
suitable for `T5CoreMLEmbedder`. See `README.md` (project root) for the
end-to-end "Getting Started" instructions and the
`SWITCHCRAFT_XTR_MLPACKAGE` env-var contract.

## `regenerate-tokenizer-fixtures.py`

Regenerates the cached HuggingFace tokenizer fixture used by
`TokenizerTests`. Re-run after upgrading the upstream tokenizer JSON.

## `fetch-nfcorpus.sh` — populating `$SWITCHCRAFT_NFCORPUS_DIR`

`NFCorpusBenchmarkTests` (`Tests/SwitchcraftTests/`) computes
NDCG@10 against the NFCorpus test split and asserts the result lands
in Witchcraft's published [0.31, 0.33] band. The dataset is **not
committed** to this repository: NFCorpus is academic-use-only and
incompatible with Switchcraft's Apache 2.0 release intent. Developers
who want to run the benchmark obtain NFCorpus locally under whatever
upstream terms they accept.

This script is one developer convenience. It pulls four `.zst` files
from upstream Witchcraft's pinned commit, decompresses them in place,
and writes plaintext into the directory referenced by
`$SWITCHCRAFT_NFCORPUS_DIR`.

### Prerequisites

- `bash`, `curl`, `zstd` (on macOS: `brew install zstd`).

### Usage

```bash
export SWITCHCRAFT_NFCORPUS_DIR="$HOME/datasets/nfcorpus"
mkdir -p "$SWITCHCRAFT_NFCORPUS_DIR"
./scripts/fetch-nfcorpus.sh
```

The script writes three required plaintext files plus one optional
file into `$SWITCHCRAFT_NFCORPUS_DIR`:

- `nfcorpus.tsv` — corpus rows (`docid \t title \t body`)
- `questions.test.tsv` — dev queries (`query-id \t query`)
- `qrels.test.json` — pytrec_eval-style nested-dict relevance judgments
- `collection_map.json` — optional; ignored by the Switchcraft suite
  (Switchcraft indexes by the original `MED-XXXX` doc-id directly)

Total size ~5–6 MB plaintext. With both `SWITCHCRAFT_XTR_MLPACKAGE`
and `SWITCHCRAFT_NFCORPUS_DIR` set, run the benchmark with:

```bash
swift test --filter NFCorpusBenchmark
```

Expect ~6 minutes on Apple Silicon for the one-time index build.

### License caveat

NFCorpus is third-party data; Switchcraft does not redistribute it and
takes no position on its licensing terms. By running this script you
are downloading data hosted by upstream Witchcraft — confirm that your
intended use is permitted before proceeding.

## `witchcraft-fact-dump.patch` — regenerating `Tests/Fixtures/facts_corpus.json`

`FactsCorpusParityTests` (`Tests/SwitchcraftTests/`) compares
Switchcraft's `SearchEngine.search` output against reference scores
produced by upstream Witchcraft for the same 33-fact corpus and same
three queries. The reference scores live in
`Tests/Fixtures/facts_corpus.json`. This patch regenerates that file.

### One-time setup

You need a working Rust toolchain and the Witchcraft assets (~62 MB
quantized GGUF + ~2 MB tokenizer + config). The Switchcraft Apple-side
toolchain is not required.

```bash
# Pinned Witchcraft commit. The fixture file's
# `provenance.witchcraftCommit` field must match this exactly.
WITCHCRAFT_COMMIT=6ad59e51cfc89bcfb20756e3f05cf9429b7cb55f

git clone https://github.com/dropbox/witchcraft /tmp/witchcraft
cd /tmp/witchcraft
git checkout "$WITCHCRAFT_COMMIT"

# Build assets/xtr.gguf, assets/config.json, assets/tokenizer.json.
# Drives uv + transformers + safetensors + the in-tree quantize-tool.
make download
```

Built and tested against Cargo 1.92.0 (Homebrew) and Python 3.13+. Apple
Silicon picks up the `metal` feature automatically; Intel Macs and
Linux pick up `fbgemm`. Either is fine for fixture generation.

### Apply the patch and run

```bash
# From inside the Witchcraft checkout, with the same commit checked out:
cd /tmp/witchcraft
git apply /path/to/switchcraft/scripts/witchcraft-fact-dump.patch

# The new test is `#[ignore]` so a normal `cargo test` does not run it.
WITCHCRAFT_COMMIT=$(git rev-parse HEAD) cargo test \
    --features t5-quantized \
    --lib \
    tests::tests::dump_facts_corpus_fixture \
    -- --nocapture --include-ignored
```

The test prints a JSON document on stdout, framed by the literal lines
`---SWITCHCRAFT-FIXTURE-BEGIN---` / `---SWITCHCRAFT-FIXTURE-END---`.
Extract that block and write it to the Switchcraft repo at
`Tests/Fixtures/facts_corpus.json`:

```bash
WITCHCRAFT_COMMIT=$(git rev-parse HEAD) cargo test \
    --features t5-quantized --lib \
    tests::tests::dump_facts_corpus_fixture \
    -- --nocapture --include-ignored 2>&1 \
| awk '/^---SWITCHCRAFT-FIXTURE-BEGIN---$/{flag=1; next} \
       /^---SWITCHCRAFT-FIXTURE-END---$/{flag=0} flag' \
> /path/to/switchcraft/Tests/Fixtures/facts_corpus.json
```

`WITCHCRAFT_COMMIT` must be set on this command too — the helper reads
the env var directly when emitting the `provenance.witchcraftCommit`
field, and falls back to the literal string `"UNKNOWN"` if unset.

Then:

```bash
cd /path/to/switchcraft
python3 -m json.tool < Tests/Fixtures/facts_corpus.json > /dev/null
```

If `json.tool` is silent the file parsed cleanly. Commit it.

### Why the patch is committed (instead of a standalone Cargo binary)

`tests.rs` already owns the canonical `FACTS` and `QUERIES` constants;
duplicating them in a separate binary would split the source of truth.
The patch adds one `#[ignore]`d `#[test]` and reuses the existing
constants verbatim.

### What the patch does, in one sentence

Adds a single `#[ignore]`d test that indexes the 33-fact corpus exactly
the way `test_end_to_end` does, runs every query through
`crate::match_centroids` (the pure-vector pipeline — no FTS, no RRF, no
post-processing roll), and prints the captured `(docId, score)` lists
between the marker lines for the runner to capture.

## `witchcraft-fixture-export.patch` — regenerating cross-stack reference fixtures

Three Switchcraft tests (`KMeansTests.referenceCentroidsParity`,
`Q4CodecTests.referenceResidualsParity`,
`CrossStackEmbeddingParityTests.referenceEmbeddingsParity`) compare
Switchcraft's k-means / Q4 codec / CoreML embedder against ground-truth
reference data produced by upstream Witchcraft. The reference data
lives in:

* `Tests/Fixtures/reference_centroids.{bin,json}`
* `Tests/Fixtures/reference_residuals.{bin,json}`
* `Tests/Fixtures/reference_embeddings.{bin,json}`

This patch regenerates all three fixture pairs in one Witchcraft test
run.

See `adrs/013-reference-fixture-provenance.md` for the full provenance,
tolerance policy, and rationale for the patch-based generator.

### Setup

Same one-time setup as `witchcraft-fact-dump.patch` above:

```bash
WITCHCRAFT_COMMIT=6ad59e51cfc89bcfb20756e3f05cf9429b7cb55f
git clone https://github.com/dropbox/witchcraft /tmp/witchcraft
cd /tmp/witchcraft
git checkout "$WITCHCRAFT_COMMIT"
make download                       # ~62 MB GGUF + tokenizer + config
git apply /path/to/switchcraft/scripts/witchcraft-fixture-export.patch
```

> The fact-dump patch and the fixture-export patch insert at the **same
> anchor** in `src/tests.rs`. Apply only one at a time. To regenerate
> both fact and fixture data, apply one, run, `git stash` or `git
> checkout src/tests.rs`, then apply the other.

> **`pub(crate)` visibility note**: the patch calls `crate::to_q4_bytes`
> and `crate::from_companded_q4_bytes` directly. If upstream renames or
> re-scopes those items, add a `#[cfg(test)] pub use` shim inside the
> patch (still a single-file diff). The `dump_reference_centroids_fixture`
> SQL also assumes `embedding.value` and `bucket.center` are the input/
> output BLOB columns — adjust the SELECT statements if the schema has
> moved.

### Run

The patch adds three `#[ignore]`d tests:

| Fixture                        | Test name                              |
|-------------------------------|----------------------------------------|
| `reference_centroids.{bin,json}` | `dump_reference_centroids_fixture`     |
| `reference_residuals.{bin,json}` | `dump_reference_residuals_fixture`     |
| `reference_embeddings.{bin,json}`| `dump_reference_embeddings_fixture`    |

Each test prints two stdout sections, framed by markers:

```
---SWITCHCRAFT-<NAME>-JSON-BEGIN---
…JSON metadata…
---SWITCHCRAFT-<NAME>-JSON-END---
---SWITCHCRAFT-<NAME>-BIN-BASE64-BEGIN---
…base64-encoded raw FP32 LE blob…
---SWITCHCRAFT-<NAME>-BIN-BASE64-END---
```

Run all three at once:

```bash
WITCHCRAFT_COMMIT=$(git rev-parse HEAD) cargo test \
    --features t5-quantized \
    --lib \
    -- --nocapture --include-ignored \
    tests::tests::dump_reference_centroids_fixture \
    tests::tests::dump_reference_residuals_fixture \
    tests::tests::dump_reference_embeddings_fixture \
    > /tmp/switchcraft-fixtures.out 2>&1
```

### Extract

Per-fixture extraction follows the same `awk`-between-markers pattern
as `witchcraft-fact-dump.patch`. JSON regions go to `.json` files,
base64 regions decode into `.bin` files:

```bash
DEST=/path/to/switchcraft/Tests/Fixtures
SRC=/tmp/switchcraft-fixtures.out

for NAME in CENTROIDS RESIDUALS EMBEDDINGS; do
    LOWER=$(echo "$NAME" | tr A-Z a-z)
    awk -v b="---SWITCHCRAFT-${NAME}-JSON-BEGIN---" \
        -v e="---SWITCHCRAFT-${NAME}-JSON-END---" \
        '$0==b{flag=1; next} $0==e{flag=0} flag' "$SRC" \
        > "$DEST/reference_${LOWER}.json"
    awk -v b="---SWITCHCRAFT-${NAME}-BIN-BASE64-BEGIN---" \
        -v e="---SWITCHCRAFT-${NAME}-BIN-BASE64-END---" \
        '$0==b{flag=1; next} $0==e{flag=0} flag' "$SRC" \
        | base64 --decode \
        > "$DEST/reference_${LOWER}.bin"
done
```

Validate:

```bash
cd /path/to/switchcraft
for f in reference_centroids reference_residuals reference_embeddings; do
    python3 -m json.tool < "Tests/Fixtures/${f}.json" > /dev/null
    test -s "Tests/Fixtures/${f}.bin"
done
swift test --filter "Reference"
```

If `swift test` is silent on the new tests, double-check the JSON files
contain a valid `provenance.witchcraftCommit` matching the pin above —
the loaders fail-fast on missing provenance fields.

### Why patch-based instead of a Cargo binary

ADR 013(b) is the long form. Short version: Witchcraft's `kmeans`, Q4
codec, and bucket-centroid SQLite accessors are `pub(crate)` and not
reachable from an external crate. Adding `#[ignore]`d tests inside
upstream `src/tests.rs` reuses the existing `FACTS` constants and
private-API access without requiring upstream to publish anything.

### Notes on score comparability

- `match_centroids` is the closest analogue of Switchcraft's
  `SearchEngine.search(queryEmbeddings:dims:topK:filter:)`. Both return
  raw MaxSim mean scores; `crate::search` (Witchcraft) and
  `SwitchcraftStore.search` (Switchcraft) layer FTS + RRF on top, which
  ADR 008 disclaims as a parity target.
- Switchcraft's CoreML T5 runs FP32 compute with FP16 output ports
  (see ADR 010(c) for why blanket FP16 was abandoned); Witchcraft's
  GGUF T5 is Q4K. Some per-token drift is expected; the parity test
  accepts ±0.01 per ADR 003. If a future model build pushes drift past
  that, the runner re-records `provenance.computeUnits` / re-pins
  `witchcraftCommit` and the tolerance discussion happens in the
  affected PR.
