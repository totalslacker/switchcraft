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
