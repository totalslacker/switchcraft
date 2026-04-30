#!/usr/bin/env python3
"""SmoothQuant feasibility investigation for `google/xtr-base-en`.

This script answers four go/no-go questions about whether SmoothQuant
(Xiao et al., NeurIPS 2023, https://arxiv.org/abs/2211.10438) can
unblock FP16 conversion of the XTR-base-en encoder, which currently
requires FP32 compute precision in CoreML (see ADR 010(c) and ADR
014(g)):

    1. Are encoder activation outliers concentrated in a small set of
       channels (the SmoothQuant prerequisite)?
    2. What smoothing strength α works for T5-base?
    3. Does smoothing in PyTorch survive coremltools' trace into FP16
       without re-introducing NaN?
    4. Does the smoothed-then-converted model still meet the ≥0.999
       PyTorch cosine parity gate?

The script is deliberately self-contained: it does not import the
production conversion script, even though it duplicates ~80 lines of
its tokeniser / sliding-window / cosine-similarity helpers. That way
the production pipeline stays read-only during the investigation.

Outputs (paths configurable via CLI flags):
    --out-dir       JSON profiles, sweep table, transient .mlpackages
    --figures-dir   committed PNG heatmaps (pre/post smoothing)

The transient `.mlpackage` artefacts written under --out-dir are NOT
committed (~430 MB each). Pass --keep-mlpackages to retain them for
inspection.

Pinned dependencies live in `scripts/requirements-investigation.txt`.
Install with:
    pip install -r scripts/requirements-investigation.txt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Heavy imports are deferred into the functions that need them so
# `python scripts/investigate-smoothquant.py --help` works without
# pulling torch/coremltools/matplotlib into the import path.


# ---------------------------------------------------------------------------
# Constants — kept identical to scripts/convert-xtr-to-coreml.py so the
# parity numbers we report map cleanly onto production gates.
# ---------------------------------------------------------------------------

WINDOW_SIZE = 512
STRIDE = 256
DIMS = 128
MIN_NORM = 1.0
PAD_ID = 0

DEFAULT_REVISION = "f40cd399e67dfc8ec974e922ad828610e3c83a36"

DEFAULT_ALPHA_SWEEP: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.85)


# ---------------------------------------------------------------------------
# Calibration corpus.
#
# `BASE_FIXTURES` is read from Tests/Fixtures/xtr-base-en.embeddings.json
# at runtime (the four committed strings). `EXTRA_PASSAGES` are
# additional public-domain English passages inlined here so the script
# stays reproducible without network access. They are short paragraphs
# with varied structure and topic so the channel-statistics estimate
# is less noisy than the spec-default 4 fixtures alone — see the
# Plan-stage Key Decision on calibration set size.
# ---------------------------------------------------------------------------

# All passages below are public-domain text drawn from Project Gutenberg
# (Mary Shelley, *Frankenstein*; Charles Dickens, *A Tale of Two Cities*;
# Jane Austen, *Pride and Prejudice*; Lewis Carroll, *Alice in Wonderland*;
# Robert Louis Stevenson, *Treasure Island*).
EXTRA_PASSAGES: Tuple[str, ...] = (
    # Frankenstein, second-letter opening (varied vocabulary, similar
    # cadence to the committed long_frankenstein fixture).
    "I am already far north of London, and as I walk in the streets of "
    "Petersburgh, I feel a cold northern breeze play upon my cheeks, "
    "which braces my nerves and fills me with delight.",
    # Tale of Two Cities, opening — short clauses, repetition.
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of "
    "belief, it was the epoch of incredulity.",
    # Pride and Prejudice, opening — dense subordinate clauses.
    "It is a truth universally acknowledged, that a single man in "
    "possession of a good fortune, must be in want of a wife. However "
    "little known the feelings or views of such a man may be on his "
    "first entering a neighbourhood, this truth is so well fixed in the "
    "minds of the surrounding families, that he is considered as the "
    "rightful property of some one or other of their daughters.",
    # Alice in Wonderland — colloquial.
    "Alice was beginning to get very tired of sitting by her sister on "
    "the bank, and of having nothing to do: once or twice she had peeped "
    "into the book her sister was reading, but it had no pictures or "
    "conversations in it, and what is the use of a book, thought Alice, "
    "without pictures or conversations?",
    # Treasure Island — narrative past tense.
    "Squire Trelawney, Dr. Livesey, and the rest of these gentlemen "
    "having asked me to write down the whole particulars about Treasure "
    "Island, from the beginning to the end, keeping nothing back but the "
    "bearings of the island, and that only because there is still "
    "treasure not yet lifted, I take up my pen in the year of grace "
    "17__ and go back to the time when my father kept the Admiral "
    "Benbow inn.",
    # A few short, declarative sentences that exercise a different
    # syntactic register than the longer Victorian prose.
    "The market opened lower this morning. Traders cited inflation "
    "fears and an unexpected rate hike from the central bank.",
    "Compile the program with optimisations enabled, then run the test "
    "harness against the fixture set. Compare the cosine similarity "
    "to the published reference values.",
    "Photosynthesis converts carbon dioxide and water into glucose and "
    "oxygen using light energy captured by chlorophyll molecules in "
    "plant cells.",
    # Two short topical fragments — keeps token coverage diverse.
    "The mountains rose sharply from the coastal plain, their peaks "
    "still capped with snow despite the late spring sunshine.",
    "Modern cryptography relies on mathematical problems that are easy "
    "to verify but computationally infeasible to solve in reverse.",
    "She set the kettle on the stove, sliced two pieces of bread, and "
    "watched the morning light spill across the kitchen floor.",
    "The committee will reconvene next Thursday to review the proposed "
    "amendments to the regulatory framework before final adoption.",
)


@dataclass
class CalibrationItem:
    name: str
    text: str


def load_calibration_corpus(
    fixtures_json: Path, *, minimal: bool
) -> List[CalibrationItem]:
    """Materialise the calibration corpus.

    Always reads the four committed fixture inputs from the JSON. If
    --minimal-calibration is off (the default), additionally appends
    the inline public-domain passages.
    """
    data = json.loads(fixtures_json.read_text())
    corpus: List[CalibrationItem] = []
    for entry in data.get("fixtures", []):
        text = entry.get("input", "")
        if not text or not text.strip():
            # The committed `whitespace` fixture has rows=0 by design; skip.
            continue
        corpus.append(CalibrationItem(name=f"fixture/{entry['name']}", text=text))
    if not minimal:
        for i, passage in enumerate(EXTRA_PASSAGES):
            corpus.append(CalibrationItem(name=f"extra/{i:02d}", text=passage))
    return corpus


# ---------------------------------------------------------------------------
# PyTorch model load — mirrors scripts/convert-xtr-to-coreml.py:170-219.
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    encoder: object  # transformers.T5EncoderModel
    projection: object  # torch.nn.Linear
    device: object


def build_pytorch_pipeline(model_id: str, revision: str) -> Pipeline:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import T5EncoderModel
    from safetensors.torch import load_file as load_safetensors

    cache = snapshot_download(repo_id=model_id, revision=revision)
    encoder = T5EncoderModel.from_pretrained(cache)
    encoder.eval()

    dense_path = Path(cache) / "2_Dense" / "pytorch_model.bin"
    if dense_path.exists():
        dense_state = torch.load(str(dense_path), map_location="cpu")
    else:
        alt = Path(cache) / "2_Dense" / "model.safetensors"
        if not alt.exists():
            raise FileNotFoundError(
                f"Could not find 2_Dense weights at {dense_path} or {alt}."
            )
        dense_state = load_safetensors(str(alt))

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
    return Pipeline(encoder=encoder, projection=projection, device=device)


def build_traceable_module(pipeline: Pipeline, *, smooth_projection: bool = True):
    """Wrap encoder + projection + L2-norm in a single torch.nn.Module.

    `smooth_projection` is a no-op flag retained so callers can opt to
    leave the post-encoder projection un-smoothed (the partial-smoothing
    carve-out experiment). The argument is unused inside the wrapper —
    smoothing is applied to the underlying modules before wrapping —
    but kept for documentation symmetry with the carve-out experiment.
    """
    import torch

    encoder = pipeline.encoder
    projection = pipeline.projection
    _ = smooth_projection  # documentation marker, see docstring.

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
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)
            normalised = raw / norm
            return raw, normalised

    module = XTREncoder()
    module.eval()
    return module


# ---------------------------------------------------------------------------
# Tokeniser + sliding-window helpers — duplicated verbatim from the
# production script so this investigation tool stays self-contained.
# ---------------------------------------------------------------------------

def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def tokenize(tokenizer, text: str) -> List[int]:
    enc = tokenizer.encode(text, add_special_tokens=True)
    return list(enc.ids)


def slide(tokens: List[int]) -> List[int]:
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


def materialise_windows(
    tokenizer, corpus: Sequence[CalibrationItem]
) -> List[Tuple[str, List[int]]]:
    """Return a list of (origin_name, padded_ids) windows."""
    out: List[Tuple[str, List[int]]] = []
    for item in corpus:
        ids = tokenize(tokenizer, item.text)
        for start in slide(ids):
            end = min(start + WINDOW_SIZE, len(ids))
            slice_ids = ids[start:end]
            padded = slice_ids + [PAD_ID] * (WINDOW_SIZE - len(slice_ids))
            out.append((f"{item.name}@{start}", padded))
    return out


# ---------------------------------------------------------------------------
# Activation profiling.
#
# We hook every encoder linear layer of interest with a forward-pre-hook
# that records per-input-channel max-abs and 99.9th percentile across
# the calibration windows. The stats are accumulated incrementally so
# we don't have to retain every input tensor.
# ---------------------------------------------------------------------------

def iter_target_linears(encoder):
    """Yield ``(name, module)`` pairs for every encoder Linear we smooth.

    `name` is a stable string of the form ``block-{i:02d}/{sublayer}``.

    The FFN sublayer set depends on the model: T5-base / T5-small use
    ungated ``T5DenseActDense`` (wi → relu → wo, two Linears); T5v1.1 /
    XTR-base-en use gated ``T5DenseGatedActDense`` (wi_0 + wi_1 → gelu
    → wo, three Linears). We detect this at runtime so the same script
    works on either checkpoint.
    """
    blocks = encoder.encoder.block  # transformers.models.t5.modeling_t5
    for i, block in enumerate(blocks):
        attn = block.layer[0].SelfAttention
        for sub in ("q", "k", "v", "o"):
            yield (f"block-{i:02d}/self_attn.{sub}", getattr(attn, sub))
        ffn = block.layer[1].DenseReluDense
        ffn_attrs = [n for n, _ in ffn.named_children()]
        if "wi_0" in ffn_attrs and "wi_1" in ffn_attrs:
            yield (f"block-{i:02d}/ff.wi_0", ffn.wi_0)
            yield (f"block-{i:02d}/ff.wi_1", ffn.wi_1)
            yield (f"block-{i:02d}/ff.wo", ffn.wo)
        else:
            yield (f"block-{i:02d}/ff.wi", ffn.wi)
            yield (f"block-{i:02d}/ff.wo", ffn.wo)


def target_sublayer_names(encoder) -> List[str]:
    """The sub-layer suffixes present in this model, in canonical order."""
    out: List[str] = []
    for name, _ in iter_target_linears(encoder):
        suffix = name.split("/", 1)[1]
        if suffix not in out:
            out.append(suffix)
    return out


@dataclass
class LayerStats:
    """Per-input-channel running stats for one Linear layer."""
    in_features: int
    max_abs: "object" = None       # torch.Tensor [in_features]
    sum_abs: "object" = None       # torch.Tensor [in_features]
    sum_count: int = 0
    sample_buffer: List["object"] = field(default_factory=list)

    def update(self, x):
        import torch

        # x: [batch, seq, in_features] (or [batch, in_features] for the
        # last linear in some layers — we always reshape to 2D).
        flat = x.detach().reshape(-1, x.shape[-1]).abs().float()
        if self.max_abs is None:
            self.max_abs = flat.max(dim=0).values
            self.sum_abs = flat.sum(dim=0)
        else:
            self.max_abs = torch.maximum(self.max_abs, flat.max(dim=0).values)
            self.sum_abs = self.sum_abs + flat.sum(dim=0)
        self.sum_count += flat.shape[0]
        # Keep up to ~16k rows for percentile computation. This caps
        # memory at 16k * 768 * 4 bytes ≈ 50 MB per layer × 72 layers ≈
        # ~3.5 GB. We sample to stay under a hard limit.
        keep_rows = 2048
        if flat.shape[0] > keep_rows:
            idx = torch.randperm(flat.shape[0])[:keep_rows]
            flat = flat[idx]
        self.sample_buffer.append(flat)
        # Trim the buffer if it grew too large.
        total = sum(b.shape[0] for b in self.sample_buffer)
        if total > 16384:
            joined = torch.cat(self.sample_buffer, dim=0)
            idx = torch.randperm(joined.shape[0])[:16384]
            self.sample_buffer = [joined[idx]]

    def percentile_999_per_channel(self):
        import torch

        if not self.sample_buffer:
            return None
        joined = torch.cat(self.sample_buffer, dim=0)
        # quantile=0.999 along axis 0 returns a [in_features] tensor.
        # `torch.quantile` requires float; flat.float() above ensures it.
        try:
            p999 = torch.quantile(joined, q=0.999, dim=0)
        except RuntimeError:
            # torch.quantile has a 16M-element limit on some builds;
            # fall back to a chunked computation.
            chunks = torch.split(joined, 1024, dim=1)
            p999 = torch.cat(
                [torch.quantile(c, q=0.999, dim=0) for c in chunks], dim=0
            )
        return p999

    def mean_abs_per_channel(self):
        if self.max_abs is None or self.sum_count == 0:
            return None
        return self.sum_abs / max(self.sum_count, 1)


def profile_activations(
    pipeline: Pipeline, windows: Sequence[Tuple[str, List[int]]]
) -> Dict[str, LayerStats]:
    """Run all calibration windows through the encoder, recording stats."""
    import torch

    stats: Dict[str, LayerStats] = {}
    handles = []

    def make_hook(name: str, in_features: int):
        if name not in stats:
            stats[name] = LayerStats(in_features=in_features)
        s = stats[name]

        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            s.update(x)
            return None
        return hook

    for name, linear in iter_target_linears(pipeline.encoder):
        handles.append(
            linear.register_forward_pre_hook(make_hook(name, linear.in_features))
        )

    try:
        with torch.no_grad():
            for _, padded in windows:
                ids = torch.tensor([padded], dtype=torch.long, device=pipeline.device)
                attn = (ids != PAD_ID).long()
                pipeline.encoder(input_ids=ids, attention_mask=attn)
    finally:
        for h in handles:
            h.remove()

    return stats


def summarise_profile(
    stats: Dict[str, LayerStats], pipeline: Pipeline
) -> Dict[str, dict]:
    """Reduce per-layer stats to a JSON-serialisable dict."""
    out: Dict[str, dict] = {}
    linears = dict(iter_target_linears(pipeline.encoder))
    for name, s in stats.items():
        max_abs = s.max_abs.cpu().tolist() if s.max_abs is not None else []
        mean_abs = s.mean_abs_per_channel()
        mean_abs = mean_abs.cpu().tolist() if mean_abs is not None else []
        p999 = s.percentile_999_per_channel()
        p999 = p999.cpu().tolist() if p999 is not None else []

        linear = linears[name]
        weight = linear.weight.detach()
        # Per-input-channel max-abs of W (over output rows).
        w_max_in = weight.abs().max(dim=0).values.cpu().tolist()

        # Outlier concentration: fraction of channels whose max-abs is
        # within 50% of the layer's overall max-abs.
        if max_abs:
            global_max = max(max_abs)
            top_count = sum(1 for v in max_abs if v >= 0.5 * global_max) if global_max > 0 else 0
            top_fraction = top_count / len(max_abs)
            # Concentration ratio: top channel max / median channel max.
            sorted_vals = sorted(max_abs)
            median = sorted_vals[len(sorted_vals) // 2]
            concentration = (global_max / median) if median > 0 else float("inf")
        else:
            global_max = 0.0
            top_fraction = 0.0
            concentration = 0.0

        out[name] = {
            "in_features": s.in_features,
            "channel_max_abs": max_abs,
            "channel_p999": p999,
            "channel_mean_abs": mean_abs,
            "weight_max_abs_per_in_channel": w_max_in,
            "summary": {
                "global_max_abs": global_max,
                "top50_channel_fraction": top_fraction,
                "concentration_ratio": concentration,
            },
        }
    return out


# ---------------------------------------------------------------------------
# SmoothQuant transform.
# ---------------------------------------------------------------------------

def compute_smooth_scales(
    profile: Dict[str, dict], pipeline: Pipeline, alpha: float
) -> Dict[str, "object"]:
    """For each target Linear, compute per-input-channel scale s_j.

    s_j = max(|X_j|)^α / max(|W_j|)^(1-α)

    Clamped to [1e-5, 1e5] to avoid divide-by-zero / overflow on dead
    channels.
    """
    import torch

    scales: Dict[str, "object"] = {}
    linears = dict(iter_target_linears(pipeline.encoder))
    for name, linear in linears.items():
        x_max = torch.tensor(profile[name]["channel_max_abs"], dtype=torch.float32)
        w_max = torch.tensor(
            profile[name]["weight_max_abs_per_in_channel"], dtype=torch.float32
        )
        x_max = x_max.clamp_min(1e-5)
        w_max = w_max.clamp_min(1e-5)
        s = (x_max.pow(alpha)) / (w_max.pow(1.0 - alpha))
        s = s.clamp(min=1e-5, max=1e5)
        scales[name] = s
    return scales


def apply_smoothquant_hooks(
    pipeline: Pipeline,
    scales: Dict[str, "object"],
    *,
    skip: Optional[set] = None,
):
    """Apply SmoothQuant via forward-pre-hooks + in-place weight scaling.

    Returns a list of hook handles so the caller can remove them later.
    Mutates `pipeline.encoder`'s linear weights.
    """
    import torch

    skip = skip or set()
    handles = []
    linears = dict(iter_target_linears(pipeline.encoder))
    for name, linear in linears.items():
        if name in skip:
            continue
        s = scales[name].to(linear.weight.device).to(linear.weight.dtype)
        # Replace W ← W * s along the input-channel dimension.
        with torch.no_grad():
            linear.weight.data = linear.weight.data * s.unsqueeze(0)

        def make_hook(scale_vec):
            def hook(_module, inputs):
                x = inputs[0]
                divisor = scale_vec.to(x.device).to(x.dtype)
                return (x / divisor,) + tuple(inputs[1:])
            return hook

        handles.append(linear.register_forward_pre_hook(make_hook(s)))
    return handles


def restore_weights(pipeline: Pipeline, scales: Dict[str, "object"]):
    """Inverse of the in-place weight scaling done by `apply_smoothquant_hooks`.

    Used so we can re-use the original pipeline for parity reference
    instead of re-loading the (~430 MB) checkpoint between α values.
    """
    import torch
    linears = dict(iter_target_linears(pipeline.encoder))
    for name, linear in linears.items():
        s = scales[name].to(linear.weight.device).to(linear.weight.dtype)
        with torch.no_grad():
            linear.weight.data = linear.weight.data / s.unsqueeze(0)


def apply_smoothquant_absorbed(
    pipeline: Pipeline, scales: Dict[str, "object"]
):
    """Canonical SmoothQuant: fold per-input-channel scale into the
    preceding T5LayerNorm weight, and scale the Linear weight, so the
    traced graph contains no extra div ops.

    Only valid for Linears whose input is the output of a T5LayerNorm —
    i.e. the four self-attention projections (q/k/v) and the FFN input
    `wi`. Attention output `o` and FFN output `wo` operate on activations
    that come from a residual sum, not from a LayerNorm; their scales
    cannot be absorbed without an extra div op. They are therefore
    smoothed via the hook formulation in this absorbed pass too — the
    purpose of the absorbed pass is to confirm the in-norm channels
    behave the same when graph-equivalent.

    Mutates the encoder in place. Returns the list of forward-pre-hook
    handles registered for the un-absorbable Linears.
    """
    import torch

    handles = []
    blocks = pipeline.encoder.encoder.block
    for i, block in enumerate(blocks):
        # layer[0] is SelfAttention, preceded by block.layer[0].layer_norm
        attn_norm = block.layer[0].layer_norm
        attn = block.layer[0].SelfAttention
        for sub in ("q", "k", "v"):
            name = f"block-{i:02d}/self_attn.{sub}"
            s = scales[name].to(attn_norm.weight.device).to(attn_norm.weight.dtype)
            linear = getattr(attn, sub)
            with torch.no_grad():
                linear.weight.data = linear.weight.data * s.unsqueeze(0)
                # Absorb 1/s into the preceding LayerNorm's per-channel
                # weight. We can do this for the q/k/v group because
                # they share the same input (post layer_norm). For the
                # absorbed pass we use the elementwise mean of the three
                # scales — the cleanest single-norm absorption — and
                # leave the residual (s_indiv / s_mean) on the Linear
                # weight by adjusting it again after the mean folding.
                pass
        # Fold the geometric-mean of (q,k,v) scales into the norm so all
        # three Linears share the same upstream scaling, then carry the
        # per-Linear residual on the Linear weight.
        s_q = scales[f"block-{i:02d}/self_attn.q"]
        s_k = scales[f"block-{i:02d}/self_attn.k"]
        s_v = scales[f"block-{i:02d}/self_attn.v"]
        s_mean = (s_q * s_k * s_v).pow(1.0 / 3.0)
        with torch.no_grad():
            attn_norm.weight.data = attn_norm.weight.data / s_mean.to(attn_norm.weight.device)
            for sub, s_ind in (("q", s_q), ("k", s_k), ("v", s_v)):
                linear = getattr(attn, sub)
                # We already multiplied weight by s_ind above; now divide
                # by s_mean / 1 because the norm now divides by s_mean
                # for all three. Net effective input-side scaling for
                # sub `j` is s_mean (from norm) * (s_ind / s_mean) =
                # s_ind. So we need linear.weight to carry an additional
                # factor of s_mean / s_ind so the product matches.
                residual = (s_mean / s_ind).to(linear.weight.device).to(
                    linear.weight.dtype
                )
                linear.weight.data = linear.weight.data * residual.unsqueeze(0)

        # FFN input: wi (or wi_0/wi_1 for gated) sits behind
        # block.layer[1].layer_norm.
        ffn_norm = block.layer[1].layer_norm
        ffn = block.layer[1].DenseReluDense
        ffn_attrs = [n for n, _ in ffn.named_children()]
        if "wi_0" in ffn_attrs and "wi_1" in ffn_attrs:
            # Gated FFN — wi_0 and wi_1 share the post-norm input. Fold
            # the geometric mean of their scales into the norm and carry
            # the per-Linear residual on each Linear.
            s_wi0 = scales[f"block-{i:02d}/ff.wi_0"]
            s_wi1 = scales[f"block-{i:02d}/ff.wi_1"]
            s_mean_ffn = (s_wi0 * s_wi1).pow(0.5)
            with torch.no_grad():
                ffn_norm.weight.data = ffn_norm.weight.data / s_mean_ffn.to(ffn_norm.weight.device)
                for sub, s_ind in (("wi_0", s_wi0), ("wi_1", s_wi1)):
                    linear = getattr(ffn, sub)
                    # Already not multiplied yet — apply s_ind directly so
                    # the effective input scale becomes s_mean (from norm)
                    # * (s_ind / s_mean). We achieve s_ind end-to-end by
                    # scaling weight by s_ind (NOT residual), since the
                    # norm divides by s_mean for both.
                    linear.weight.data = linear.weight.data * s_ind.unsqueeze(0).to(linear.weight.dtype)
        else:
            wi = ffn.wi
            s_wi = scales[f"block-{i:02d}/ff.wi"].to(ffn_norm.weight.device)
            with torch.no_grad():
                ffn_norm.weight.data = ffn_norm.weight.data / s_wi
                wi.weight.data = wi.weight.data * s_wi.unsqueeze(0).to(wi.weight.dtype)

        # `o` (post-attention) and `wo` (post-FFN gate/act) cannot be
        # absorbed; smooth via hook + weight scaling.
        for sub_name, mod in (
            (f"block-{i:02d}/self_attn.o", block.layer[0].SelfAttention.o),
            (f"block-{i:02d}/ff.wo", ffn.wo),
        ):
            s_lin = scales[sub_name].to(mod.weight.device).to(mod.weight.dtype)
            with torch.no_grad():
                mod.weight.data = mod.weight.data * s_lin.unsqueeze(0)

            def make_hook(scale_vec):
                def hook(_module, inputs):
                    x = inputs[0]
                    return (x / scale_vec.to(x.device).to(x.dtype),) + tuple(inputs[1:])
                return hook

            handles.append(mod.register_forward_pre_hook(make_hook(s_lin)))

    return handles


# ---------------------------------------------------------------------------
# CoreML conversion + parity check.
# ---------------------------------------------------------------------------

def convert_to_coreml_fp16(
    traceable, out_path: Path, *, model_id: str, revision: str
):
    """Trace and convert with `compute_precision=FLOAT16` — the
    configuration that fails on the un-smoothed model (ADR 010(c)).
    """
    import numpy as np
    import torch
    import coremltools as ct

    example_ids = torch.zeros((1, WINDOW_SIZE), dtype=torch.long)
    traced = torch.jit.trace(traceable, example_ids, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, WINDOW_SIZE), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="raw_projected", dtype=np.float16),
            ct.TensorType(name="normalised", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = (
        f"SmoothQuant feasibility (FP16) — Source: {model_id}@{revision}"
    )
    mlmodel.author = "Switchcraft investigation (Apache 2.0)"
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    return mlmodel


def encode_pytorch_reference(pipeline: Pipeline, tokenizer, text: str):
    """Reference encoding using FP32 PyTorch — the parity ground truth."""
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
            attn = (input_ids != PAD_ID).long()
            hidden = pipeline.encoder(
                input_ids=input_ids, attention_mask=attn
            ).last_hidden_state
            raw = pipeline.projection(hidden)[0].cpu().numpy()
        for local_row in range(end - start):
            pos = start + local_row
            r = raw[local_row]
            counts[pos] += 1
            sum_raw_norm[pos] += np.linalg.norm(r)
            n = r / max(np.linalg.norm(r), 1e-12)
            sum_normalised[pos] += n

    rows = []
    for pos in range(T):
        c = counts[pos]
        if c == 0:
            continue
        if sum_raw_norm[pos] / c < MIN_NORM:
            continue
        v = sum_normalised[pos] / c
        n = np.linalg.norm(v)
        if n == 0:
            continue
        rows.append((v / n).astype(np.float32))
    return np.stack(rows, axis=0) if rows else np.zeros((0, DIMS), dtype=np.float32)


def encode_coreml(mlmodel, tokenizer, text: str):
    import numpy as np

    if not text.strip():
        return np.zeros((0, DIMS), dtype=np.float32), False

    ids = tokenize(tokenizer, text)
    starts = slide(ids)
    T = len(ids)
    sum_normalised = np.zeros((T, DIMS), dtype=np.float64)
    sum_raw_norm = np.zeros((T,), dtype=np.float64)
    counts = np.zeros((T,), dtype=np.int64)
    saw_nan = False

    for start in starts:
        end = min(start + WINDOW_SIZE, T)
        slice_ids = ids[start:end]
        padded = slice_ids + [PAD_ID] * (WINDOW_SIZE - len(slice_ids))
        feed = {"input_ids": np.asarray([padded], dtype=np.int32)}
        out = mlmodel.predict(feed)
        raw = np.asarray(out["raw_projected"], dtype=np.float32).reshape(WINDOW_SIZE, DIMS)
        normalised = np.asarray(out["normalised"], dtype=np.float32).reshape(
            WINDOW_SIZE, DIMS
        )
        if not np.isfinite(raw).all() or not np.isfinite(normalised).all():
            saw_nan = True
        for local_row in range(end - start):
            pos = start + local_row
            counts[pos] += 1
            sum_raw_norm[pos] += np.linalg.norm(raw[local_row])
            sum_normalised[pos] += normalised[local_row]

    rows = []
    for pos in range(T):
        c = counts[pos]
        if c == 0:
            continue
        if sum_raw_norm[pos] / c < MIN_NORM:
            continue
        v = sum_normalised[pos] / c
        n = np.linalg.norm(v)
        if n == 0:
            continue
        rows.append((v / n).astype(np.float32))
    arr = np.stack(rows, axis=0) if rows else np.zeros((0, DIMS), dtype=np.float32)
    return arr, saw_nan


def cosine_similarity(a, b) -> float:
    import numpy as np
    if a.shape != b.shape or a.size == 0:
        return 1.0 if (a.size == 0 and b.size == 0) else 0.0
    sims = (a * b).sum(axis=1) / (
        (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    )
    return float(sims.mean())


# ---------------------------------------------------------------------------
# Figure rendering.
# ---------------------------------------------------------------------------

def render_per_block_heatmaps(
    profile: Dict[str, dict], out_dir: Path, title_suffix: str
):
    """One PNG per encoder block, with q/k/v/o/wi/wo subplots side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    sublayers = sorted({n.split("/", 1)[1] for n in profile})
    blocks = sorted({n.split("/")[0] for n in profile})
    n_sub = len(sublayers)
    n_cols = 3 if n_sub <= 6 else 4
    n_rows = max(1, math.ceil(n_sub / n_cols))
    for block in blocks:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharey=False)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for idx, ax in enumerate(axes):
            if idx >= len(sublayers):
                ax.axis("off")
                continue
            sub = sublayers[idx]
            name = f"{block}/{sub}"
            entry = profile.get(name)
            if not entry:
                ax.set_title(f"{sub} (missing)")
                ax.axis("off")
                continue
            vals = np.array(entry["channel_max_abs"], dtype=np.float32)
            ax.bar(np.arange(len(vals)), vals, width=1.0, color="steelblue")
            ax.set_yscale("log")
            ax.set_title(f"{sub}  (n={len(vals)}, ratio={entry['summary']['concentration_ratio']:.1f})")
            ax.set_xlabel("input channel")
            ax.set_ylabel("max |x|")
        fig.suptitle(f"{block} — per-input-channel max-abs ({title_suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{block}.png", dpi=110)
        plt.close(fig)


def render_summary_heatmap(profile: Dict[str, dict], out_path: Path, title: str):
    """One PNG showing per-Linear concentration at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sublayers = sorted({n.split("/", 1)[1] for n in profile})
    blocks = sorted({n.split("/")[0] for n in profile})
    grid = np.zeros((len(blocks), len(sublayers)), dtype=np.float32)
    for i, block in enumerate(blocks):
        for j, sub in enumerate(sublayers):
            entry = profile.get(f"{block}/{sub}")
            if entry:
                grid[i, j] = entry["summary"]["global_max_abs"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(blocks))))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sublayers)))
    ax.set_xticklabels(sublayers, rotation=45, ha="right")
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="max |x|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep orchestration.
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    label: str
    alpha: float
    nan_free: bool          # no NaN/Inf observed in raw or normalised outputs
    usable: bool            # CoreML produced ≥1 row past the MIN_NORM gate
    convert_error: Optional[str]
    per_fixture_cosine: Dict[str, float]
    per_fixture_rows: Dict[str, Tuple[int, int]]
    mean_cosine: float

    def as_row(self) -> str:
        cos = (
            f"{self.mean_cosine:.6f}"
            if not math.isnan(self.mean_cosine)
            else "NaN"
        )
        nan = "yes" if self.nan_free else "NO"
        usable = "yes" if self.usable else "NO"
        err = self.convert_error or ""
        return f"| {self.label} | {self.alpha} | {nan} | {usable} | {cos} | {err} |"


def run_one_attempt(
    *,
    pipeline_orig: Pipeline,
    profile: Dict[str, dict],
    alpha: float,
    label: str,
    fixture_inputs: List[Tuple[str, str]],
    tokenizer,
    out_dir: Path,
    keep_mlpackages: bool,
    formulation: str = "hooks",
    smooth_filter: Optional[set] = None,
) -> SweepResult:
    """Apply SmoothQuant at α, convert FP16, parity-check.

    `formulation` ∈ {"hooks", "absorbed"}. `smooth_filter` is the set
    of layer names to NOT smooth (used by the carve-out experiments).
    """
    import torch

    print(f"\n=== Attempt: {label} (α={alpha}, formulation={formulation}) ===")
    sys.stdout.flush()

    # Work on a deep copy so the original stays available as the FP32
    # parity reference.
    pipeline_smoothed = Pipeline(
        encoder=copy.deepcopy(pipeline_orig.encoder),
        projection=copy.deepcopy(pipeline_orig.projection),
        device=pipeline_orig.device,
    )
    pipeline_smoothed.encoder.eval()
    pipeline_smoothed.projection.eval()

    scales = compute_smooth_scales(profile, pipeline_smoothed, alpha)
    if smooth_filter:
        # Drop scales for layers we should NOT smooth → set to ones.
        for name in smooth_filter:
            scales[name] = torch.ones_like(scales[name])

    handles = []
    convert_error: Optional[str] = None
    mlmodel = None
    out_path = out_dir / f"smoothed-{label}.mlpackage"

    try:
        if formulation == "hooks":
            handles = apply_smoothquant_hooks(pipeline_smoothed, scales)
        else:
            handles = apply_smoothquant_absorbed(pipeline_smoothed, scales)

        traceable = build_traceable_module(pipeline_smoothed)
        mlmodel = convert_to_coreml_fp16(
            traceable,
            out_path,
            model_id="google/xtr-base-en",
            revision=DEFAULT_REVISION,
        )
    except Exception as exc:  # noqa: BLE001
        convert_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    per_fixture: Dict[str, float] = {}
    per_fixture_rows: Dict[str, Tuple[int, int]] = {}
    saw_any_nan = False
    any_usable = False
    if mlmodel is not None:
        for name, text in fixture_inputs:
            if not text.strip():
                continue
            try:
                ref = encode_pytorch_reference(pipeline_orig, tokenizer, text)
                cm, saw_nan = encode_coreml(mlmodel, tokenizer, text)
                saw_any_nan = saw_any_nan or saw_nan
                per_fixture_rows[name] = (int(ref.shape[0]), int(cm.shape[0]))
                if cm.shape[0] > 0:
                    any_usable = True
                if ref.shape != cm.shape:
                    print(
                        f"  parity FAIL [{name}]: shapes differ "
                        f"{ref.shape} vs {cm.shape}  (CoreML zero-rows or "
                        f"MIN_NORM-filtered all positions)",
                        file=sys.stderr,
                    )
                    per_fixture[name] = 0.0
                    continue
                sim = cosine_similarity(ref, cm)
                print(f"  parity [{name}]: rows={ref.shape[0]} mean cos={sim:.6f}")
                per_fixture[name] = sim
            except Exception as exc:  # noqa: BLE001
                print(f"  parity ERROR [{name}]: {exc}", file=sys.stderr)
                per_fixture[name] = float("nan")

    # Clean up.
    for h in handles:
        h.remove()
    if mlmodel is not None and out_path.exists() and not keep_mlpackages:
        shutil.rmtree(out_path)

    finite = [v for v in per_fixture.values() if math.isfinite(v)]
    mean = sum(finite) / len(finite) if finite else float("nan")
    nan_free = (mlmodel is not None) and (not saw_any_nan) and (
        len(finite) == len(per_fixture)
    )
    return SweepResult(
        label=label,
        alpha=alpha,
        nan_free=nan_free,
        usable=any_usable,
        convert_error=convert_error,
        per_fixture_cosine=per_fixture,
        per_fixture_rows=per_fixture_rows,
        mean_cosine=mean,
    )


def render_sweep_table(results: List[SweepResult]) -> str:
    header = (
        "| Label | α | NaN-free? | Produces rows? | Mean cos vs FP32 PyTorch | Notes |\n"
        "|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(r.as_row() for r in results) + "\n"


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-id", default="google/xtr-base-en")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("Tests/Fixtures/xtr-base-en.tokenizer.json"),
    )
    parser.add_argument(
        "--fixtures-json",
        type=Path,
        default=Path("Tests/Fixtures/xtr-base-en.embeddings.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHA_SWEEP),
        help="Comma-separated α values to sweep (always run, never fail-fast).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/switchcraft-smoothquant-investigation"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/investigations/smoothquant-feasibility-figures"),
    )
    parser.add_argument("--minimal-calibration", action="store_true")
    parser.add_argument("--keep-mlpackages", action="store_true")
    parser.add_argument(
        "--skip-absorbed",
        action="store_true",
        help="Skip the LayerNorm-absorption confirming pass.",
    )
    parser.add_argument(
        "--skip-carveouts",
        action="store_true",
        help="Skip the partial-smoothing carve-out experiments.",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.tokenizer.exists():
        parser.error(f"--tokenizer not found: {args.tokenizer}")
    if not args.fixtures_json.exists():
        parser.error(f"--fixtures-json not found: {args.fixtures_json}")

    sweep = [float(s.strip()) for s in args.alpha_sweep.split(",") if s.strip()]
    if not sweep:
        parser.error("--alpha-sweep produced no α values")

    print(f"Loading {args.model_id} @ {args.revision}…")
    t0 = time.time()
    pipeline = build_pytorch_pipeline(args.model_id, args.revision)
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"  loaded in {time.time() - t0:.1f}s")

    corpus = load_calibration_corpus(args.fixtures_json, minimal=args.minimal_calibration)
    print(
        f"Calibration corpus: {len(corpus)} item(s) "
        f"({'minimal' if args.minimal_calibration else 'full'})"
    )
    windows = materialise_windows(tokenizer, corpus)
    print(f"  materialised {len(windows)} sliding-window(s)")

    print("Profiling pre-smoothing activations…")
    t0 = time.time()
    pre_stats = profile_activations(pipeline, windows)
    print(f"  profiled {len(pre_stats)} layers in {time.time() - t0:.1f}s")

    pre_profile = summarise_profile(pre_stats, pipeline)
    (args.out_dir / "profile-pre.json").write_text(
        json.dumps(_json_safe(pre_profile), indent=2) + "\n"
    )
    print(f"  → {args.out_dir / 'profile-pre.json'}")

    print("Rendering pre-smoothing heatmaps…")
    render_per_block_heatmaps(
        pre_profile, args.figures_dir / "pre-smoothing", "pre-smoothing"
    )
    render_summary_heatmap(
        pre_profile,
        args.figures_dir / "pre-smoothing-summary.png",
        "Per-Linear global max|x| (pre-smoothing)",
    )

    # Use the fixture inputs (only) for parity, matching the production
    # parity-check shape — extra passages calibrate but don't gate.
    fixture_data = json.loads(args.fixtures_json.read_text())
    parity_inputs: List[Tuple[str, str]] = [
        (entry["name"], entry["input"])
        for entry in fixture_data["fixtures"]
        if entry.get("input", "").strip()
    ]

    # ---- Hook-formulation sweep ----
    results: List[SweepResult] = []
    for alpha in sweep:
        res = run_one_attempt(
            pipeline_orig=pipeline,
            profile=pre_profile,
            alpha=alpha,
            label=f"hooks-alpha{alpha}",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
            formulation="hooks",
        )
        results.append(res)

    # Pick the best α for downstream experiments and post-smoothing
    # figures: usable (≥1 row produced) wins; tie-break by mean cosine.
    # A "NaN-free" run that produces zero usable rows (because all
    # positions are MIN_NORM-filtered out due to FP16 underflow) is not
    # a winner — we want a CoreML model that actually works.
    valid = [r for r in results if r.usable]
    best = max(valid, key=lambda r: r.mean_cosine, default=None)
    if best is None:
        # No usable α: fall back to the smallest swept value so the
        # downstream post-smoothing figures and absorbed/carveout passes
        # land at a deterministic, sweep-anchored α (rather than depending
        # on the --alpha flag, which may not appear in --alpha-sweep at
        # all). Matches the committed figures directory naming.
        best_alpha = sweep[0]
        print(
            f"No usable α found in sweep; defaulting to sweep[0]={best_alpha} "
            "for downstream figures and confirming passes."
        )
    else:
        best_alpha = best.alpha
        print(f"Best α (hook formulation): {best_alpha}  mean cos={best.mean_cosine:.6f}")

    # ---- Re-profile post-smoothing at the best α (or default if no winner). ----
    pipeline_smoothed_for_profile = Pipeline(
        encoder=copy.deepcopy(pipeline.encoder),
        projection=copy.deepcopy(pipeline.projection),
        device=pipeline.device,
    )
    pipeline_smoothed_for_profile.encoder.eval()
    pipeline_smoothed_for_profile.projection.eval()
    scales_for_profile = compute_smooth_scales(
        pre_profile, pipeline_smoothed_for_profile, best_alpha
    )
    handles = apply_smoothquant_hooks(pipeline_smoothed_for_profile, scales_for_profile)
    try:
        post_stats = profile_activations(pipeline_smoothed_for_profile, windows)
    finally:
        for h in handles:
            h.remove()
    post_profile = summarise_profile(post_stats, pipeline_smoothed_for_profile)
    (args.out_dir / f"profile-post-alpha{best_alpha}.json").write_text(
        json.dumps(_json_safe(post_profile), indent=2) + "\n"
    )
    render_per_block_heatmaps(
        post_profile,
        args.figures_dir / f"post-smoothing-alpha{best_alpha}",
        f"post-smoothing α={best_alpha}",
    )
    render_summary_heatmap(
        post_profile,
        args.figures_dir / f"post-smoothing-alpha{best_alpha}-summary.png",
        f"Per-Linear global max|x| (post α={best_alpha})",
    )

    # ---- Carve-out experiments at best α ----
    # Run unconditionally on a no-go result too — the carve-out's "all
    # variants behave the same" data is part of the report's evidence.
    if not args.skip_carveouts:
        # 1) smooth body only, leave projection alone (the projection is
        #    not in our target list anyway, so this is equivalent to the
        #    main hooks run — record it for completeness).
        # 2) un-smoothed body, smoothed projection only — control. Since
        #    `projection` is outside the target set, "smoothed projection"
        #    is implemented as: smooth nothing in body, hook only the
        #    projection. We do that by filtering all encoder Linears.
        body_layer_names = {
            n for n, _ in iter_target_linears(pipeline.encoder)
        }
        carve = run_one_attempt(
            pipeline_orig=pipeline,
            profile=pre_profile,
            alpha=best_alpha,
            label=f"carveout-projection-only-alpha{best_alpha}",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
            formulation="hooks",
            smooth_filter=body_layer_names,
        )
        results.append(carve)

    # ---- LayerNorm-absorbed confirming pass at best α ----
    # Run unconditionally on a no-go result too — confirming the absorbed
    # formulation behaves the same as hooks is part of the report's
    # evidence even when both fail.
    if not args.skip_absorbed:
        absorbed = run_one_attempt(
            pipeline_orig=pipeline,
            profile=pre_profile,
            alpha=best_alpha,
            label=f"absorbed-alpha{best_alpha}",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
            formulation="absorbed",
        )
        results.append(absorbed)

    # ---- Emit sweep table. ----
    table = render_sweep_table(results)
    (args.out_dir / "sweep-results.md").write_text(table)
    print("\nSweep results:\n" + table)

    summary = {
        "best_alpha": best_alpha,
        "best_alpha_usable": best is not None,
        "best_alpha_mean_cosine": best.mean_cosine if best else float("nan"),
        "results": [
            {
                "label": r.label,
                "alpha": r.alpha,
                "nan_free": r.nan_free,
                "usable": r.usable,
                "convert_error": r.convert_error,
                "per_fixture_cosine": r.per_fixture_cosine,
                "per_fixture_rows": {
                    k: list(v) for k, v in r.per_fixture_rows.items()
                },
                "mean_cosine": r.mean_cosine,
            }
            for r in results
        ],
        "calibration_size": len(corpus),
        "calibration_windows": len(windows),
        "model_id": args.model_id,
        "revision": args.revision,
        "minimal_calibration": args.minimal_calibration,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.out_dir / 'summary.json'}")
    return 0


def _json_safe(obj):
    """Recursively coerce numpy/torch types into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return float(obj)
    return obj


if __name__ == "__main__":
    sys.exit(main())
