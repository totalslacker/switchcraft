#!/usr/bin/env python3
"""Regenerate Switchcraft tokenizer fixture reference IDs from HuggingFace.

Switchcraft's tokenizer fixtures (Tests/Fixtures/*.tokenizer-fixtures.json)
contain (input, ids, noSpecial) triples that the Swift implementation must
reproduce bit-exact. The reference IDs come from the upstream HuggingFace
`tokenizers` Rust crate via its Python bindings, run against the same
`tokenizer.json` Switchcraft loads.

This script regenerates a fixture file by tokenizing a fixed list of inputs
through HuggingFace and emitting the JSON shape the Swift test loader expects.

Pinned dependency:
    pip install tokenizers==0.23.1

Invocation (from repo root):
    python3 scripts/regenerate-tokenizer-fixtures.py \
        --tokenizer Tests/Fixtures/xtr-base-en.tokenizer.json \
        --out Tests/Fixtures/xtr-base-en.tokenizer-fixtures.json

The input list is hard-coded below; edit INPUTS to add or change cases.

When porting a new tokenizer, copy this script, point --tokenizer at the new
tokenizer.json, --out at the new fixtures path, and add the new inputs.
The Switchcraft Swift suite must pass against the regenerated reference IDs
without code changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-not-found]


# Inputs covered by the fixture suite. The entries before the Issue #8 marker
# include the original PR #7 set plus three later pre-Issue #8 additions
# (`"a" * 2049`, `"cat \U0001F431 dog"`, and `"naïve café résumé"`).
# The remaining entries are the Issue #8 additions covering added-token
# pre-segmentation (positive matches, no-space-padding, negative controls
# that must NOT match, and overlapping-token longest-wins).
#
# Non-printable / invisible characters are written with explicit escape
# sequences so the file stays reviewable and editor-safe (no risk of an
# editor or formatter silently mangling them).
INPUTS: list[str] = [
    "",
    "   ",
    " leading space",
    "trailing space  ",
    "hello world",
    "Hello, world!",
    "▁literal metaspace",
    "ℌello",
    "日本語テキスト",
    "\x01\x02\x03",
    "hello\u200bworld",
    "a" * 2049,
    "cat \U0001F431 dog",
    "naïve café résumé",
    # --- Issue #8: added-token pre-segmentation ---
    # Positive: spaces around added tokens
    "prefix <extra_id_5> middle <extra_id_42> suffix",
    # Positive: no spaces between text and added token
    "a<extra_id_5>b",
    # Negative control: identifier-like text that is NOT a literal added-token
    # match. Must segment via Viterbi, not be intercepted as an added token.
    "extra_id_5 is not a token",
    # Negative control: closing > not adjacent to the would-be token
    "<extra_id_5text>",
    # Overlap: <extra_id_52> must win over <extra_id_5> at the same position
    # (descending-length-sort), and the trailing <extra_id_5> still matches.
    "<extra_id_52><extra_id_5>",
]


def encode(tok: Tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    return tok.encode(text, add_special_tokens=add_special_tokens).ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="path to tokenizer.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="path to write the fixtures JSON",
    )
    args = parser.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    fixtures = []
    for text in INPUTS:
        fixtures.append(
            {
                "input": text,
                "ids": encode(tok, text, add_special_tokens=True),
                "noSpecial": encode(tok, text, add_special_tokens=False),
            }
        )

    args.out.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(fixtures)} fixtures to {args.out}")


if __name__ == "__main__":
    main()
