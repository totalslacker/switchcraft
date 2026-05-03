#!/usr/bin/env python3
"""Dump per-layer encoder hidden states from google/xtr-base-en for Metal bisection.

Runs T5EncoderModel.forward(..., output_hidden_states=True) on the standard fixture
inputs and writes two artefacts:

    Tests/Fixtures/hidden_states.bin   — raw FP32 tensor bytes (little-endian)
    Tests/Fixtures/hidden_states.json  — index [{name, byteOffset, byteLength, rows, cols}]

Each fixture × each encoder layer produces one entry. The hidden state at layer ℓ
is the *output* of that layer (post-attention, post-FFN, post-residual, before the
next layer's pre-norm). Layer 0 is the embedding output before any encoder layer.
There are nLayers+1 = 13 entries per fixture (indices 0–12).

The JSON schema is distinct from xtr-base-en.embeddings.json (cols=128) — all entries
here have cols=768, the encoder hidden dimension.

Usage (from repo root):
    python3 scripts/dump-pytorch-hidden-states.py \\
        --model-id google/xtr-base-en \\
        --revision <huggingface-commit-sha> \\
        --tokenizer Tests/Fixtures/xtr-base-en.tokenizer.json \\
        --out Tests/Fixtures

Asset-gated: the resulting files are committed to the repo as reference fixtures
(same pattern as xtr-base-en.embeddings.bin). Their generation requires a Python
environment with the packages in scripts/requirements-coreml.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

WINDOW_SIZE = 512
PAD_ID = 0

FIXTURE_INPUTS: List[Tuple[str, str]] = [
    ("short", "the apple is red and round"),
    (
        "paragraph",
        "Witchcraft is a Rust reimplementation of XTR-Warp. It "
        "implements token-level semantic search with sub-linear "
        "retrieval. Switchcraft is its Swift port, targeting Apple "
        "platforms.",
    ),
    (
        "long_frankenstein_opening",
        (
            "You will rejoice to hear that no disaster has accompanied the "
            "commencement of an enterprise which you have regarded with "
            "such evil forebodings. I arrived here yesterday, and my first "
            "task is to assure my dear sister of my welfare and increasing "
            "confidence in the success of my undertaking. I am already far "
            "north of London, and as I walk in the streets of Petersburgh, "
            "I feel a cold northern breeze play upon my cheeks, which "
            "braces my nerves and fills me with delight. Do you understand "
            "this feeling? This breeze, which has travelled from the "
            "regions towards which I am advancing, gives me a foretaste of "
            "those icy climes. Inspirited by this wind of promise, my "
            "daydreams become more fervent and vivid. I try in vain to be "
            "persuaded that the pole is the seat of frost and desolation; "
            "it ever presents itself to my imagination as the region of "
            "beauty and delight. There, Margaret, the sun is for ever "
            "visible, its broad disk just skirting the horizon and "
            "diffusing a perpetual splendour. There—for with your leave, "
            "my sister, I will put some trust in preceding navigators—"
            "there snow and frost are banished; and, sailing over a calm "
            "sea, we may be wafted to a land surpassing in wonders and in "
            "beauty every region hitherto discovered on the habitable "
            "globe. Its productions and features may be without example, "
            "as the phenomena of the heavenly bodies undoubtedly are in "
            "those undiscovered solitudes. What may not be expected in a "
            "country of eternal light? I may there discover the wondrous "
            "power which attracts the needle and may regulate a thousand "
            "celestial observations that require only this voyage to "
            "render their seeming eccentricities consistent for ever. I "
            "shall satiate my ardent curiosity with the sight of a part "
            "of the world never before visited, and may tread a land "
            "never before imprinted by the foot of man. These are my "
            "enticements, and they are sufficient to conquer all fear of "
            "danger or death and to induce me to commence this laborious "
            "voyage with the joy a child feels when he embarks in a "
            "little boat, with his holiday mates, on an expedition of "
            "discovery up his native river. But supposing all these "
            "conjectures to be false, you cannot contest the inestimable "
            "benefit which I shall confer on all mankind, to the last "
            "generation, by discovering a passage near the pole to those "
            "countries, to reach which at present so many months are "
            "requisite; or by ascertaining the secret of the magnet, "
            "which, if at all possible, can only be effected by an "
            "undertaking such as mine."
        ),
    ),
]


def load_tokenizer(path: str):
    """Load the HuggingFace BPE tokenizer JSON used by Switchcraft."""
    import json as _json
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def tokenize(tok, text: str) -> List[int]:
    """Tokenize and truncate to WINDOW_SIZE. Returns int list of length ≤ WINDOW_SIZE."""
    enc = tok.encode(text, add_special_tokens=True)
    ids = enc.ids[:WINDOW_SIZE]
    return ids


def load_model(model_id: str, revision: str | None):
    """Load google/xtr-base-en T5EncoderModel from HuggingFace."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import T5EncoderModel

    cache = snapshot_download(repo_id=model_id, revision=revision)
    encoder = T5EncoderModel.from_pretrained(cache)
    encoder.eval()
    return encoder, cache


def dump_hidden_states(
    encoder,
    tokenizer_path: str,
    out_dir: Path,
):
    import torch

    tok = load_tokenizer(tokenizer_path)
    index: list[dict] = []
    blob_parts: list[bytes] = []
    byte_offset = 0

    for fixture_name, text in FIXTURE_INPUTS:
        ids = tokenize(tok, text)
        W = len(ids)
        padded = ids + [PAD_ID] * (WINDOW_SIZE - W)
        input_ids = torch.tensor([padded], dtype=torch.long)
        attention_mask = torch.tensor([[1] * W + [0] * (WINDOW_SIZE - W)], dtype=torch.long)

        with torch.no_grad():
            out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # out.hidden_states is a tuple of (nLayers+1) tensors, each [1, 512, 768].
        # hidden_states[0] = embedding output, [1] = after layer 0, ..., [12] = after layer 11.
        hidden_states = out.hidden_states  # (13,) of [1, 512, 768]

        for layer_idx, hs in enumerate(hidden_states):
            # Slice to real token positions only (first W rows).
            hs_np = hs[0, :W, :].float().numpy()  # [W, 768]
            raw = hs_np.astype("<f4").tobytes()  # explicit little-endian FP32
            byte_length = len(raw)

            entry = {
                "name": f"{fixture_name}_layer_{layer_idx}",
                "byteOffset": byte_offset,
                "byteLength": byte_length,
                "rows": W,
                "cols": 768,
            }
            index.append(entry)
            blob_parts.append(raw)
            byte_offset += byte_length

        print(f"  {fixture_name}: {W} tokens, {len(hidden_states)} layers")

    bin_path = out_dir / "hidden_states.bin"
    json_path = out_dir / "hidden_states.json"

    with open(bin_path, "wb") as f:
        for part in blob_parts:
            f.write(part)
    with open(json_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    print(f"Wrote {bin_path} ({byte_offset} bytes)")
    print(f"Wrote {json_path} ({len(index)} entries)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="google/xtr-base-en", help="HuggingFace model ID")
    p.add_argument("--revision", default=None, help="HuggingFace commit SHA or tag")
    p.add_argument(
        "--tokenizer",
        default="Tests/Fixtures/xtr-base-en.tokenizer.json",
        help="Path to the Switchcraft tokenizer JSON (HuggingFace BPE format)",
    )
    p.add_argument(
        "--out",
        default="Tests/Fixtures",
        help="Output directory for hidden_states.{bin,json}",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} (revision={args.revision or 'latest'}) ...")
    encoder, _ = load_model(args.model_id, args.revision)

    print("Dumping hidden states for fixtures:")
    dump_hidden_states(encoder, args.tokenizer, out_dir)


if __name__ == "__main__":
    main()
