#!/usr/bin/env bash
# fetch-nfcorpus.sh — fetch the NFCorpus test split for the
# `NFCorpusBenchmark` parity suite.
#
# The NFCorpus dataset is academic-use-only and is **not redistributed**
# by Switchcraft. This helper is one developer convenience for obtaining
# the data locally; you accept whatever upstream terms apply.
#
# Source: upstream Witchcraft's pre-staged zstd files. Switchcraft pins
# this source so the published NDCG@10 ∈ [0.31, 0.33] band remains
# defensible (different mirrors normalise whitespace / title-body
# composition slightly differently).
#
# Usage:
#   export SWITCHCRAFT_NFCORPUS_DIR=/abs/path/to/nfcorpus
#   ./scripts/fetch-nfcorpus.sh
#
# Requirements: bash, curl, zstd. (Run `brew install zstd` on macOS if
# needed.)

set -euo pipefail

WITCHCRAFT_COMMIT="${WITCHCRAFT_COMMIT:-6ad59e51cfc89bcfb20756e3f05cf9429b7cb55f}"
BASE_URL="https://raw.githubusercontent.com/dropbox/witchcraft/${WITCHCRAFT_COMMIT}"

# (remote-relative-path, local-filename) pairs. Files arrive zstd-compressed
# and are decompressed in place; the test suite consumes plaintext only.
FILES=(
    "datasets/nfcorpus.tsv.zst|nfcorpus.tsv"
    "testset/nfcorpus/questions.test.tsv.zst|questions.test.tsv"
    "testset/nfcorpus/qrels.test.json.zst|qrels.test.json"
    "testset/nfcorpus/collection_map.json.zst|collection_map.json"
)

if [ -z "${SWITCHCRAFT_NFCORPUS_DIR:-}" ]; then
    cat >&2 <<'EOF'
SWITCHCRAFT_NFCORPUS_DIR is unset. Set it to an absolute directory path
that you want NFCorpus extracted into. Example:

    export SWITCHCRAFT_NFCORPUS_DIR="$HOME/datasets/nfcorpus"
    mkdir -p "$SWITCHCRAFT_NFCORPUS_DIR"
    ./scripts/fetch-nfcorpus.sh

The directory will be populated with four required files:
nfcorpus.tsv, questions.test.tsv, qrels.test.json, collection_map.json
(totalling ~5–6 MB). collection_map.json maps numeric corpus IDs to
MED-* doc IDs used in qrels — the benchmark requires all four files.

Then point the test suite at it:

    export SWITCHCRAFT_XTR_MLPACKAGE=/path/to/xtr-base-en.mlpackage
    swift test --filter NFCorpusBenchmark
EOF
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "fetch-nfcorpus.sh: 'curl' not found in PATH" >&2
    exit 1
fi
if ! command -v zstd >/dev/null 2>&1; then
    echo "fetch-nfcorpus.sh: 'zstd' not found in PATH (try: brew install zstd)" >&2
    exit 1
fi

mkdir -p "$SWITCHCRAFT_NFCORPUS_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Fetching NFCorpus from witchcraft@${WITCHCRAFT_COMMIT}"
echo "  destination: $SWITCHCRAFT_NFCORPUS_DIR"

for entry in "${FILES[@]}"; do
    remote="${entry%|*}"
    local_name="${entry#*|}"
    url="${BASE_URL}/${remote}"
    compressed="${TMP}/${local_name}.zst"
    out="${SWITCHCRAFT_NFCORPUS_DIR}/${local_name}"

    echo "  → ${remote}"
    curl --fail --silent --show-error --location --output "$compressed" "$url"
    zstd --decompress --stdout --quiet --force "$compressed" >"$out"
done

echo "Done. Files written:"
for entry in "${FILES[@]}"; do
    local_name="${entry#*|}"
    out="${SWITCHCRAFT_NFCORPUS_DIR}/${local_name}"
    size=$(wc -c <"$out" | tr -d ' ')
    printf "  %s (%s bytes)\n" "$out" "$size"
done

cat <<EOF

Run the benchmark with:

    export SWITCHCRAFT_XTR_MLPACKAGE=/path/to/xtr-base-en.mlpackage
    export SWITCHCRAFT_NFCORPUS_DIR="$SWITCHCRAFT_NFCORPUS_DIR"
    swift test --filter NFCorpusBenchmark

Expect ~6 minutes on Apple Silicon for the one-time index build.
EOF
