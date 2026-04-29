#!/usr/bin/env python3
"""Convert the `google/xtr-base-en` T5 encoder + 768→128 projection layer
to a CoreML `.mlpackage` and emit the Swift-side reference fixtures.

The CoreML graph this script produces matches the contract that
`Sources/SwitchcraftCoreML/T5CoreMLEmbedder.swift` expects:

    Inputs:
        input_ids        — Int32 [1, 512] HuggingFace token IDs (pad = 0)

    Outputs:
        raw_projected    — Float16 [1, 512, 128]  (post-projection,
                                                   pre-L2-norm; used for
                                                   the MIN_NORM filter)
        normalised       — Float16 [1, 512, 128]  (post-projection,
                                                   post-L2-norm; what the
                                                   Swift caller returns)

The Swift wrapper computes per-token L2 norms from `raw_projected`, drops
positions whose merged raw norm is < 1.0 (Witchcraft's `MIN_NORM`), then
emits the corresponding rows of `normalised` after sliding-window merge.

After export, the script runs a parity check: the same reference token
sequences are encoded through both the original PyTorch model and the
converted CoreML model. Mean cosine similarity must be ≥ 0.999 or the
script aborts non-zero.

The script also writes `Tests/Fixtures/xtr-base-en.embeddings.bin`
(raw FP32 reference embeddings produced by the PyTorch model after the
MIN_NORM filter, in the exact shape the Swift `T5CoreMLEmbedder` will
return) plus `xtr-base-en.embeddings.json` describing each input string,
its byte offset, row count, and the model commit SHA used to produce it.

Pinned dependencies live in `scripts/requirements-coreml.txt`. Install
with:
    pip install -r scripts/requirements-coreml.txt

Invocation (from repo root):
    python3 scripts/convert-xtr-to-coreml.py \\
        --model-id google/xtr-base-en \\
        --revision <huggingface-commit-sha> \\
        --tokenizer Tests/Fixtures/xtr-base-en.tokenizer.json \\
        --out-mlpackage Tests/Fixtures/xtr-base-en.mlpackage \\
        --out-fixtures Tests/Fixtures

The `.mlpackage` itself is NOT committed to the repo (~80 MB). Set
`SWITCHCRAFT_XTR_MLPACKAGE=<path>` so the integration tests can find it;
otherwise asset-gated tests skip cleanly.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# These imports are heavy; defer until main() so --help works without them.


# ---------------------------------------------------------------------------
# Reference inputs used by the parity check and Swift fixtures.
#
# The fixture set MUST cover (per the spec):
#   - a short sentence
#   - a multi-sentence paragraph
#   - a whitespace-only string (must produce zero rows)
#   - a string longer than 512 tokens (exercises sliding-window)
#
# The long-input fixture is the opening paragraph of Mary Shelley's
# *Frankenstein* (Project Gutenberg eBook #84, public domain). The exact
# byte range and SHA-256 of this excerpt are recorded in
# adrs/010-embedder-model-and-asset-distribution.md so the fixture is
# reproducible.
# ---------------------------------------------------------------------------

FIXTURE_INPUTS: List[Tuple[str, str]] = [
    ("short", "the apple is red and round"),
    (
        "paragraph",
        "Witchcraft is a Rust reimplementation of XTR-Warp. It "
        "implements token-level semantic search with sub-linear "
        "retrieval. Switchcraft is its Swift port, targeting Apple "
        "platforms.",
    ),
    ("whitespace", "   \t\n  "),
    # Long-input excerpt — see ADR 010 for the SHA-256 of this string.
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

WINDOW_SIZE = 512
STRIDE = 256
DIMS = 128
MIN_NORM = 1.0
PAD_ID = 0


# ---------------------------------------------------------------------------
# CoreML graph
# ---------------------------------------------------------------------------

@dataclass
class ConvertedModel:
    """Container for the loaded PyTorch encoder + projection."""
    encoder: object  # transformers.T5EncoderModel
    projection: object  # torch.nn.Linear
    device: object  # torch.device

    def encode(self, input_ids):  # numpy or torch.LongTensor [1, T]
        import torch
        with torch.no_grad():
            ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            attn = (ids != PAD_ID).long()
            hidden = self.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
            projected = self.projection(hidden)
            return projected  # [1, T, 128]


def build_pytorch_pipeline(model_id: str, revision: str) -> ConvertedModel:
    """Load the T5 encoder + 2_Dense projection for `google/xtr-base-en`.

    The projection is a 768→128 weight-only linear (no bias), packaged
    under `2_Dense/pytorch_model.bin` in the HuggingFace repo. We replicate
    the SentenceTransformer pipeline manually so we can compose a single
    torch graph that coremltools can trace.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import T5EncoderModel
    from safetensors.torch import load_file as load_safetensors

    cache = snapshot_download(repo_id=model_id, revision=revision)
    encoder = T5EncoderModel.from_pretrained(cache)
    encoder.eval()

    # Load the 2_Dense projection. It's a single weight tensor, no bias.
    dense_path = Path(cache) / "2_Dense" / "pytorch_model.bin"
    if not dense_path.exists():
        # Some checkpoints use safetensors instead.
        alt = Path(cache) / "2_Dense" / "model.safetensors"
        if alt.exists():
            dense_state = load_safetensors(str(alt))
        else:
            raise FileNotFoundError(
                f"Could not find 2_Dense weights at {dense_path} or {alt}. "
                "Confirm the SentenceTransformer projection layer is present "
                "in the HuggingFace snapshot."
            )
    else:
        dense_state = torch.load(str(dense_path), map_location="cpu")

    # SentenceTransformers stores the linear weight under 'linear.weight'.
    weight_key = next(
        k for k in dense_state.keys() if k.endswith(".weight") or k == "weight"
    )
    weight = dense_state[weight_key]
    if weight.shape != (DIMS, 768):
        raise ValueError(
            f"Expected projection shape ({DIMS}, 768), got {tuple(weight.shape)}"
        )
    projection = torch.nn.Linear(768, DIMS, bias=False)
    projection.weight.data = weight.float()
    projection.eval()

    device = torch.device("cpu")
    encoder.to(device)
    projection.to(device)
    return ConvertedModel(encoder=encoder, projection=projection, device=device)


def build_traceable_module(pipeline: ConvertedModel):
    """Wrap encoder + projection + L2-norm in a single ``torch.nn.Module``
    that emits two outputs: raw_projected and normalised.

    The L2 normalisation is baked into the graph; the Swift caller never
    runs a separate normalisation pass.
    """
    import torch

    encoder = pipeline.encoder
    projection = pipeline.projection

    class XTREncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.projection = projection

        def forward(self, input_ids):
            attention_mask = (input_ids != PAD_ID).to(dtype=torch.long)
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            raw = self.projection(hidden)
            # L2 normalise along the last dim. eps mirrors PyTorch's default.
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)
            normalised = raw / norm
            return raw, normalised

    module = XTREncoder()
    module.eval()
    return module


def convert_to_coreml(traceable, out_path: Path, *, revision: str, model_id: str):
    """Convert the wrapped torch module to CoreML and save as .mlpackage."""
    import torch
    import coremltools as ct

    example_ids = torch.zeros((1, WINDOW_SIZE), dtype=torch.long)
    traced = torch.jit.trace(traceable, example_ids, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, WINDOW_SIZE),
                dtype="int32",
            ),
        ],
        outputs=[
            ct.TensorType(name="raw_projected", dtype="fp16"),
            ct.TensorType(name="normalised", dtype="fp16"),
        ],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = (
        f"Switchcraft XTR-base-en encoder + 768→128 projection. "
        f"Source: {model_id}@{revision}"
    )
    mlmodel.author = "Switchcraft project (Apache 2.0)"
    mlmodel.save(str(out_path))


# ---------------------------------------------------------------------------
# Tokenization, parity, fixtures
# ---------------------------------------------------------------------------

def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def tokenize(tokenizer, text: str) -> List[int]:
    """Tokenise with addSpecialTokens=True (matches the Swift wrapper)."""
    enc = tokenizer.encode(text, add_special_tokens=True)
    return list(enc.ids)


def slide(tokens: List[int]) -> List[int]:
    """Plan window starts identical to SlidingWindow.plan() in Swift."""
    if len(tokens) <= WINDOW_SIZE:
        return [0]
    starts: List[int] = []
    s = 0
    while s + WINDOW_SIZE <= len(tokens):
        starts.append(s)
        s += STRIDE
    last = len(tokens) - WINDOW_SIZE
    if not starts or starts[-1] != last:
        while starts and starts[-1] >= last:
            starts.pop()
        starts.append(last)
    return starts


def encode_pytorch(pipeline: ConvertedModel, tokenizer, text: str):
    """Run the same end-to-end pipeline the Swift wrapper runs, but in
    PyTorch. Used to (a) seed parity-check inputs and (b) emit the Swift
    reference fixtures.
    """
    import torch
    import numpy as np

    if not text.strip():
        return np.zeros((0, DIMS), dtype=np.float32)

    ids = tokenize(tokenizer, text)
    starts = slide(ids)
    T = len(ids)

    sum_normalised = np.zeros((T, DIMS), dtype=np.float64)
    sum_raw_norm = np.zeros((T,), dtype=np.float64)
    counts = np.zeros((T,), dtype=np.int64)

    for start in starts:
        end = min(start + WINDOW_SIZE, T)
        slice_ids = ids[start:end]
        padded = slice_ids + [PAD_ID] * (WINDOW_SIZE - len(slice_ids))
        with torch.no_grad():
            input_ids = torch.tensor([padded], dtype=torch.long)
            raw = pipeline.encode(input_ids)[0].cpu().numpy()  # [W, 128]
        for local_row in range(end - start):
            pos = start + local_row
            r = raw[local_row]
            counts[pos] += 1
            sum_raw_norm[pos] += np.linalg.norm(r)
            n = r / max(np.linalg.norm(r), 1e-12)
            sum_normalised[pos] += n

    out_rows = []
    for pos in range(T):
        c = counts[pos]
        if c == 0:
            continue
        mean_raw = sum_raw_norm[pos] / c
        if mean_raw < MIN_NORM:
            continue
        v = sum_normalised[pos] / c
        # Renormalise (Swift wrapper does the same after merge).
        n = np.linalg.norm(v)
        if n == 0:
            continue
        out_rows.append((v / n).astype(np.float32))
    if not out_rows:
        return np.zeros((0, DIMS), dtype=np.float32)
    return np.stack(out_rows, axis=0)


def encode_coreml(mlmodel, tokenizer, text: str):
    """Mirror of encode_pytorch but routed through the CoreML model.
    Used by the parity check.
    """
    import numpy as np

    if not text.strip():
        return np.zeros((0, DIMS), dtype=np.float32)

    ids = tokenize(tokenizer, text)
    starts = slide(ids)
    T = len(ids)

    sum_normalised = np.zeros((T, DIMS), dtype=np.float64)
    sum_raw_norm = np.zeros((T,), dtype=np.float64)
    counts = np.zeros((T,), dtype=np.int64)

    for start in starts:
        end = min(start + WINDOW_SIZE, T)
        slice_ids = ids[start:end]
        padded = slice_ids + [PAD_ID] * (WINDOW_SIZE - len(slice_ids))
        feed = {"input_ids": np.asarray([padded], dtype=np.int32)}
        out = mlmodel.predict(feed)
        raw = np.asarray(out["raw_projected"], dtype=np.float32).reshape(WINDOW_SIZE, DIMS)
        normalised = np.asarray(out["normalised"], dtype=np.float32).reshape(WINDOW_SIZE, DIMS)
        for local_row in range(end - start):
            pos = start + local_row
            counts[pos] += 1
            sum_raw_norm[pos] += np.linalg.norm(raw[local_row])
            sum_normalised[pos] += normalised[local_row]

    out_rows = []
    for pos in range(T):
        c = counts[pos]
        if c == 0:
            continue
        mean_raw = sum_raw_norm[pos] / c
        if mean_raw < MIN_NORM:
            continue
        v = sum_normalised[pos] / c
        n = np.linalg.norm(v)
        if n == 0:
            continue
        out_rows.append((v / n).astype(np.float32))
    if not out_rows:
        return np.zeros((0, DIMS), dtype=np.float32)
    return np.stack(out_rows, axis=0)


def cosine_similarity(a, b) -> float:
    import numpy as np
    if a.shape != b.shape or a.size == 0:
        return 1.0 if a.size == 0 == b.size else 0.0
    sims = (a * b).sum(axis=1) / (
        (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    )
    return float(sims.mean())


def emit_fixtures(
    pipeline: ConvertedModel,
    tokenizer,
    out_dir: Path,
    *,
    model_id: str,
    revision: str,
):
    """Emit Tests/Fixtures/xtr-base-en.embeddings.{bin,json}.

    Layout:
      embeddings.bin   — concatenated raw FP32 little-endian rows.
      embeddings.json  — { "model": "<id>@<sha>", "dims": 128,
                           "fixtures": [
                             {"name": "...", "input": "...",
                              "rows": <m>, "byteOffset": <o>,
                              "byteLength": <m * dims * 4>}
                           ] }
    """
    import numpy as np

    bin_path = out_dir / "xtr-base-en.embeddings.bin"
    idx_path = out_dir / "xtr-base-en.embeddings.json"

    fixtures = []
    blob_parts = []
    offset = 0
    for name, text in FIXTURE_INPUTS:
        embeddings = encode_pytorch(pipeline, tokenizer, text)
        rows = int(embeddings.shape[0])
        flat = embeddings.astype("<f4").tobytes()
        fixtures.append({
            "name": name,
            "input": text,
            "rows": rows,
            "byteOffset": offset,
            "byteLength": len(flat),
        })
        blob_parts.append(flat)
        offset += len(flat)

    bin_path.write_bytes(b"".join(blob_parts))
    idx_path.write_text(
        json.dumps(
            {
                "model": f"{model_id}@{revision}",
                "dims": DIMS,
                "windowSize": WINDOW_SIZE,
                "stride": STRIDE,
                "minNorm": MIN_NORM,
                "fixtures": fixtures,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {bin_path} ({offset} bytes, {len(fixtures)} fixtures)")
    print(f"Wrote {idx_path}")


def parity_check(pipeline: ConvertedModel, mlmodel, tokenizer) -> List[Tuple[str, float]]:
    """Encode every fixture through both pipelines and return per-fixture
    cosine similarities as ``[(name, sim), ...]``. Shape mismatches are
    reported as ``sim = 0.0`` so the caller's per-fixture gate trips.
    """
    results: List[Tuple[str, float]] = []
    for name, text in FIXTURE_INPUTS:
        if not text.strip():
            continue  # whitespace-only, no rows to compare.
        py = encode_pytorch(pipeline, tokenizer, text)
        cm = encode_coreml(mlmodel, tokenizer, text)
        if py.shape != cm.shape:
            print(
                f"  parity FAIL [{name}]: shapes differ "
                f"{py.shape} vs {cm.shape}",
                file=sys.stderr,
            )
            results.append((name, 0.0))
            continue
        sim = cosine_similarity(py, cm)
        print(f"  parity [{name}]: rows={py.shape[0]} mean cos-sim={sim:.6f}")
        results.append((name, sim))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert google/xtr-base-en to CoreML and emit Switchcraft fixtures.",
    )
    parser.add_argument("--model-id", default="google/xtr-base-en")
    parser.add_argument(
        "--revision",
        required=True,
        help="HuggingFace commit SHA (recorded in ADR 010).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("Tests/Fixtures/xtr-base-en.tokenizer.json"),
    )
    parser.add_argument(
        "--out-mlpackage",
        type=Path,
        default=Path("Tests/Fixtures/xtr-base-en.mlpackage"),
    )
    parser.add_argument(
        "--out-fixtures",
        type=Path,
        default=Path("Tests/Fixtures"),
    )
    parser.add_argument(
        "--parity-threshold",
        type=float,
        default=0.999,
        help="Mean cosine similarity gate (PyTorch vs CoreML). Default 0.999.",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help=(
            "Skip the CoreML parity check. Use only when iterating on "
            "fixture generation; never commit assets that have not "
            "passed parity."
        ),
    )
    args = parser.parse_args(argv)

    if not args.tokenizer.exists():
        parser.error(f"--tokenizer not found: {args.tokenizer}")
    args.out_fixtures.mkdir(parents=True, exist_ok=True)
    args.out_mlpackage.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} @ {args.revision}…")
    pipeline = build_pytorch_pipeline(args.model_id, args.revision)
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Tracing + converting to CoreML → {args.out_mlpackage}")
    traceable = build_traceable_module(pipeline)
    convert_to_coreml(
        traceable,
        args.out_mlpackage,
        revision=args.revision,
        model_id=args.model_id,
    )

    print(f"Emitting Swift reference fixtures → {args.out_fixtures}")
    emit_fixtures(
        pipeline,
        tokenizer,
        args.out_fixtures,
        model_id=args.model_id,
        revision=args.revision,
    )

    if args.skip_parity:
        print("Skipping parity check (--skip-parity).")
        return 0

    import coremltools as ct
    mlmodel = ct.models.MLModel(str(args.out_mlpackage))
    print("Running PyTorch ↔ CoreML parity check…")
    results = parity_check(pipeline, mlmodel, tokenizer)
    failures = [(name, s) for (name, s) in results if s < args.parity_threshold]
    if failures:
        for name, s in failures:
            print(
                f"PARITY FAILED [{name}]: cos-sim {s:.6f} < threshold "
                f"{args.parity_threshold}.",
                file=sys.stderr,
            )
        print(
            f"PARITY FAILED: {len(failures)} of {len(results)} fixtures below "
            f"threshold {args.parity_threshold}. Aborting; do not ship this asset.",
            file=sys.stderr,
        )
        return 1
    worst = min((s for _, s in results), default=1.0)
    print(f"Parity check passed (worst-fixture cos-sim: {worst:.6f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
