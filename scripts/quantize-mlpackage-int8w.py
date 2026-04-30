#!/usr/bin/env python3
"""Post-process an FP32 `.mlpackage` into an INT8 weight-only variant.

This is rung 1 of the "If SmoothQuant fails" ladder in `docs/Plan.md`.
SmoothQuant was investigated in issue #43 / PR #44 and definitively
returned no-go (residual-stream magnitudes entering RMSNorm overflow
FP16, which SmoothQuant's per-Linear re-parameterisation cannot fix —
see `docs/investigations/smoothquant-feasibility.md`). INT8 weight-only
quantisation sidesteps the FP16 question entirely: weights are stored
as INT8 with per-channel scales, dequantised back to FP32 just before
each matmul, and compute precision stays at FP32 throughout. The asset
shrinks (~430 MB → ~110 MB, ~3.9×); compute precision and the parity
contract are unchanged from the FP32 default.

This script is **not a replacement** for `scripts/convert-xtr-to-coreml.py`;
it is a separate post-processing step that takes the FP32 `.mlpackage`
that script produces and emits a sibling INT8w `.mlpackage`. The two
scripts share no source; the conversion script remains the single source
of truth for the FP32 asset and the committed PyTorch reference fixtures.

    Inputs:
        --input <path>      FP32 `.mlpackage` (defaults to
                            $SWITCHCRAFT_XTR_MLPACKAGE).
        --output <path>     Destination `.mlpackage` (defaults to
                            <input-stem>-int8w.mlpackage).
        --tokenizer <path>  Tokenizer JSON for the parity-check inputs
                            (defaults to
                            Tests/Fixtures/xtr-base-en.tokenizer.json).
        --threshold <float> Mean cosine similarity floor for the
                            INT8w-vs-FP32 CoreML parity check (default
                            0.998 — see ADR 010(i)).
        --skip-parity       Skip the parity check. Use only for iteration;
                            never ship an asset that has not passed parity.

    Outputs:
        <output>            INT8 weight-only `.mlpackage`. The asset is
                            written using
                            `coremltools.optimize.coreml.linear_quantize_weights`
                            with `linear_symmetric` mode (per-channel
                            symmetric INT8 — coremltools 7.2's default).
        stdout              Human-readable summary: weight-op count,
                            measured cosine similarity per fixture, final
                            disk size.

    Parity check:
        INT8w-CoreML output is compared against FP32-CoreML output (NOT
        against the original PyTorch model — this script never loads
        PyTorch). The committed PyTorch reference fixtures
        (`Tests/Fixtures/xtr-base-en.embeddings.{bin,json}`) remain the
        canonical ground truth for both variants; the Swift test suite
        (`CoreMLInt8wParityTests`) does the INT8w-vs-PyTorch check.

    modelIdentifier convention (IMPORTANT):
        The INT8w variant MUST be initialised with a distinct
        `modelIdentifier` from the FP32 variant — recommended
        `"google/xtr-base-en@v1-int8w"` rather than the FP32 default
        `"google/xtr-base-en@v1"`. This is a documentation contract, not
        enforced in code. ADR 010(f)'s mismatch-detection logic uses the
        identifier verbatim; if you index a corpus under one variant and
        query under the other with the same identifier, mismatches will
        be silently undetectable. See ADR 010(i) for the full rationale.

    Workflow:
        1. Run `scripts/convert-xtr-to-coreml.py` to produce the FP32
           `.mlpackage` and the committed reference fixtures.
        2. Run this script on that FP32 `.mlpackage` to produce the
           INT8w sibling.
        3. Set `SWITCHCRAFT_XTR_MLPACKAGE_INT8W=<path>` so the asset-gated
           Swift tests can find it; otherwise those tests skip cleanly.

Pinned dependencies live in `scripts/requirements-coreml.txt` (no new
deps beyond what the conversion script already requires; coremltools
7.2 ships `linear_quantize_weights`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Heavy imports deferred so --help works without them.


# ---------------------------------------------------------------------------
# Reference inputs for the in-script parity check.
#
# Kept short so the script runs in seconds and does NOT need the
# sliding-window planner. Long-input parity is exercised by the Swift
# test suite (CoreMLInt8wParityTests) using the committed reference
# fixtures.
# ---------------------------------------------------------------------------

PARITY_INPUTS: List[Tuple[str, str]] = [
    ("short", "the apple is red and round"),
    (
        "paragraph",
        "Witchcraft is a Rust reimplementation of XTR-Warp. It "
        "implements token-level semantic search with sub-linear "
        "retrieval. Switchcraft is its Swift port, targeting Apple "
        "platforms.",
    ),
]

WINDOW_SIZE = 512
DIMS = 128
PAD_ID = 0


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def quantize_mlpackage(input_path: Path, output_path: Path) -> int:
    """Apply INT8 weight-only quantisation to ``input_path`` and save the
    result at ``output_path``. Returns the number of weight tensors that
    were transformed (i.e. the count of ``constexpr_affine_dequantize``
    ops in the output graph).

    Uses ``coremltools.optimize.coreml.linear_quantize_weights`` with the
    library default ``OpLinearQuantizerConfig(mode='linear_symmetric',
    dtype=np.int8)`` — per-channel symmetric INT8 over each Conv / Linear
    weight tensor whose total element count exceeds the default
    ``weight_threshold`` (2048). The XTR T5 encoder + 768→128 projection
    are well above that threshold, so every Linear in the encoder body
    plus the projection layer are quantised.

    The output graph keeps FP32 compute precision: the saved
    ``.mlpackage`` replaces each weight ``const`` with a
    ``constexpr_affine_dequantize`` whose output feeds the original
    matmul / linear op. Activations and accumulators stay at FP32.
    """
    import numpy as np
    import coremltools as ct
    from coremltools.optimize.coreml import (
        linear_quantize_weights,
        OpLinearQuantizerConfig,
        OptimizationConfig,
    )

    print(f"Loading FP32 .mlpackage from {input_path}…")
    src = ct.models.MLModel(str(input_path))

    print("Applying linear_quantize_weights (per-channel symmetric INT8)…")
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        )
    )
    quantised = linear_quantize_weights(src, config)

    # Sanity check: the only op coremltools should have introduced is
    # `constexpr_affine_dequantize`. Anything else indicates the pass
    # transformed something unexpected (e.g. activation ops) and the
    # parity contract may have shifted.
    spec = quantised.get_spec()
    function = spec.mlProgram.functions["main"]
    block_key = next(iter(function.block_specializations.keys()))
    op_types = [
        op.type for op in function.block_specializations[block_key].operations
    ]
    dequantize_count = sum(1 for t in op_types if t == "constexpr_affine_dequantize")
    if dequantize_count == 0:
        raise RuntimeError(
            "No constexpr_affine_dequantize ops found after quantisation. "
            "linear_quantize_weights silently no-op'd; check that the "
            "input .mlpackage actually contains Linear/MatMul weights "
            "above the 2048-element threshold."
        )

    print(f"  → {dequantize_count} weight tensor(s) quantised.")

    # Forward the existing description / author so provenance carries
    # through. Note the variant suffix in the description so a reviewer
    # who later inspects the asset can confirm what it is.
    quantised.short_description = (
        (src.short_description or "")
        + " [INT8 weight-only variant; per-channel symmetric, FP32 compute]"
    ).strip()
    quantised.author = src.author or "Switchcraft project (Apache 2.0)"

    print(f"Saving INT8w .mlpackage → {output_path}")
    quantised.save(str(output_path))
    return dequantize_count


# ---------------------------------------------------------------------------
# Parity check (INT8w-CoreML vs FP32-CoreML)
# ---------------------------------------------------------------------------

def tokenize_padded(tokenizer, text: str) -> List[int]:
    """Tokenise ``text`` and return a length-``WINDOW_SIZE`` ID list,
    truncated or right-padded with PAD_ID. The script intentionally only
    runs single-window inputs — long-input parity is the Swift test's
    job."""
    enc = tokenizer.encode(text, add_special_tokens=True)
    ids = list(enc.ids)[:WINDOW_SIZE]
    return ids + [PAD_ID] * (WINDOW_SIZE - len(ids))


def encode_window(mlmodel, padded_ids: List[int]):
    """Run a single padded window through ``mlmodel`` and return
    ``(raw_projected, normalised)`` as FP32 numpy arrays of shape
    ``[WINDOW_SIZE, DIMS]``."""
    import numpy as np
    feed = {"input_ids": np.asarray([padded_ids], dtype=np.int32)}
    out = mlmodel.predict(feed)
    raw = np.asarray(out["raw_projected"], dtype=np.float32).reshape(
        WINDOW_SIZE, DIMS
    )
    normalised = np.asarray(out["normalised"], dtype=np.float32).reshape(
        WINDOW_SIZE, DIMS
    )
    return raw, normalised


def cosine_similarity(a, b) -> float:
    """Mean cosine similarity over rows of two ``[N, DIMS]`` arrays."""
    import numpy as np
    if a.shape != b.shape or a.size == 0:
        return 0.0
    sims = (a * b).sum(axis=1) / (
        (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    )
    return float(sims.mean())


def parity_check(
    fp32_model, int8w_model, tokenizer, threshold: float
) -> List[Tuple[str, float]]:
    """Run every entry in ``PARITY_INPUTS`` through both models and
    report mean cosine similarity over the unpadded prefix (the rows
    corresponding to real tokens — pad rows are dropped because their
    projection magnitudes are dominated by quantisation noise on a
    near-zero input).
    """
    results: List[Tuple[str, float]] = []
    for name, text in PARITY_INPUTS:
        padded = tokenize_padded(tokenizer, text)
        real_token_count = sum(1 for tid in padded if tid != PAD_ID)
        if real_token_count == 0:
            print(f"  parity [{name}]: skipped (empty after tokenisation)")
            continue
        _, fp32_norm = encode_window(fp32_model, padded)
        _, int8w_norm = encode_window(int8w_model, padded)
        # Compare only the real-token prefix; padded positions have
        # near-zero projections whose cosine is numerically meaningless.
        sim = cosine_similarity(
            fp32_norm[:real_token_count], int8w_norm[:real_token_count]
        )
        marker = "OK" if sim >= threshold else "FAIL"
        print(
            f"  parity [{name}]: tokens={real_token_count} "
            f"mean cos-sim={sim:.6f} (threshold={threshold:.3f}) [{marker}]"
        )
        results.append((name, sim))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_input() -> Path | None:
    raw = os.environ.get("SWITCHCRAFT_XTR_MLPACKAGE", "").strip()
    return Path(raw) if raw else None


def _default_output(input_path: Path) -> Path:
    # <stem>.mlpackage → <stem>-int8w.mlpackage in the same directory.
    stem = input_path.name
    if stem.endswith(".mlpackage"):
        stem = stem[: -len(".mlpackage")]
    return input_path.with_name(f"{stem}-int8w.mlpackage")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quantise an FP32 google/xtr-base-en .mlpackage to INT8 "
            "weight-only (per-channel symmetric). See ADR 010(i) for "
            "the contract; remember to use a distinct modelIdentifier "
            "(e.g. 'google/xtr-base-en@v1-int8w') for the resulting "
            "asset to avoid silent ChunkRecord.model collisions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modelIdentifier convention\n"
            "--------------------------\n"
            "The INT8w variant MUST be initialised with a distinct\n"
            "modelIdentifier from the FP32 variant. The FP32 default is\n"
            "'google/xtr-base-en@v1'; the recommended INT8w identifier is\n"
            "'google/xtr-base-en@v1-int8w'. Per ADR 010(f), every\n"
            "ChunkRecord persists the active modelIdentifier so future\n"
            "reads can detect a model swap. If you index under FP32 and\n"
            "later query under INT8w (or vice versa) using the same\n"
            "identifier, the mismatch is silently undetectable.\n"
            "\n"
            "Workflow\n"
            "--------\n"
            "  1. python3 scripts/convert-xtr-to-coreml.py --revision <sha>\n"
            "     (produces the FP32 .mlpackage + reference fixtures)\n"
            "  2. python3 scripts/quantize-mlpackage-int8w.py\n"
            "     (produces the INT8w sibling .mlpackage)\n"
            "  3. export SWITCHCRAFT_XTR_MLPACKAGE_INT8W=<path-to-int8w>\n"
            "     (asset-gated Swift tests CoreMLInt8wParityTests pick it up)\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input(),
        help=(
            "Path to the FP32 .mlpackage. Defaults to "
            "$SWITCHCRAFT_XTR_MLPACKAGE if set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination for the INT8w .mlpackage. Defaults to "
            "<input-stem>-int8w.mlpackage in the same directory as --input."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("Tests/Fixtures/xtr-base-en.tokenizer.json"),
        help=(
            "Tokenizer JSON used by the in-script parity check. The "
            "Swift test suite has its own committed reference fixtures; "
            "this path is only used to drive the in-script INT8w-vs-FP32 "
            "comparison."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.998,
        help=(
            "Mean cosine-similarity floor for the INT8w-vs-FP32 CoreML "
            "parity check. Default 0.998 per ADR 010(i). Raise to 0.999 "
            "if measured drift on your asset is reliably below 0.001."
        ),
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help=(
            "Skip the parity check. Use only when iterating; never ship "
            "an INT8w asset that has not passed parity."
        ),
    )
    args = parser.parse_args(argv)

    if args.input is None:
        parser.error(
            "--input is required when SWITCHCRAFT_XTR_MLPACKAGE is unset."
        )
    if not args.input.exists():
        parser.error(f"--input not found: {args.input}")
    if args.output is None:
        args.output = _default_output(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        print(
            f"NOTE: --output already exists at {args.output}; it will be "
            "overwritten.",
            file=sys.stderr,
        )

    quantize_mlpackage(args.input, args.output)

    if args.skip_parity:
        print("Skipping parity check (--skip-parity).")
        return 0

    if not args.tokenizer.exists():
        parser.error(
            f"--tokenizer not found: {args.tokenizer}. The parity check "
            "needs a tokenizer to encode reference inputs; pass --tokenizer "
            "or use --skip-parity."
        )

    import coremltools as ct
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    fp32_model = ct.models.MLModel(str(args.input))
    int8w_model = ct.models.MLModel(str(args.output))

    print("Running INT8w ↔ FP32 CoreML parity check…")
    results = parity_check(fp32_model, int8w_model, tokenizer, args.threshold)
    failures = [(name, s) for (name, s) in results if s < args.threshold]
    if failures:
        for name, s in failures:
            print(
                f"PARITY FAILED [{name}]: cos-sim {s:.6f} < threshold "
                f"{args.threshold}.",
                file=sys.stderr,
            )
        print(
            f"PARITY FAILED: {len(failures)} of {len(results)} fixtures "
            f"below threshold {args.threshold}. Aborting; do not ship "
            "this asset.",
            file=sys.stderr,
        )
        return 1
    worst = min((s for _, s in results), default=1.0)
    print(
        f"Parity check passed (worst-fixture cos-sim: {worst:.6f}, "
        f"threshold: {args.threshold:.3f})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
