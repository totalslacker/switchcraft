#!/usr/bin/env python3
"""Custom MIL pass FP32-promotion feasibility investigation for
``google/xtr-base-en``.

This script answers whether a coremltools custom MIL pass — or
equivalently a structural ``op_selector`` passed to
``ct.transform.FP16ComputePrecision`` — can selectively promote the
residual-add → RMSNorm sub-graphs (and the L2-norm projection tail) to
FP32 while leaving the rest of the encoder at FP16, producing NaN-free
output that meets the ≥0.999 cosine parity gate.

It is the second-order follow-up to the SmoothQuant feasibility study
(issue #43, ``scripts/investigate-smoothquant.py``); SmoothQuant
diagnosed the FP16 failure as residual-stream magnitude entering each
block's RMSNorm, not per-Linear activation outliers, which this
investigation directly targets.

The script sweeps five FP32 promotion "islands" of increasing size:

  A. RMSNorm 6-op cluster only (``mul²/pow → reduce_mean → add(eps) →
     rsqrt → mul(x,·) → mul(γ,·)``).
  B. A + the residual ``add`` feeding each RMSNorm.
  C. B + the L2-norm projection tail.
  D. C + the upstream attention/FFN-output ``add`` ops (one transitive
     hop further back).
  E. Escape hatch: everything except ``linear``/``matmul``/``conv``.

Outputs (paths configurable via CLI flags):
    --out-dir       JSON profiles, sweep table, transient .mlpackages
    --figures-dir   committed PNG figures (op counts, parity, asset size)

The transient ``.mlpackage`` artefacts written under ``--out-dir`` are
NOT committed (~80–430 MB each). Pass ``--keep-mlpackages`` to retain
them for inspection.

Pinned dependencies live in ``scripts/requirements-investigation.txt``.
Install with:
    pip install -r scripts/requirements-investigation.txt
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# Heavy imports are deferred into the functions that need them so
# `python scripts/investigate-mil-fp32-promote.py --help` works without
# pulling torch / coremltools / matplotlib into the import path.


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

# Island identifiers in the order they are swept. The final two
# constants are anchor op-types used by the structural matcher.
ISLANDS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

# Op types that mark the centre / boundary of each FP32 island. The
# structural matcher walks parents/children from these anchors.
RMSNORM_ANCHOR_OP = "rsqrt"
RMSNORM_TAIL_MUL_OPS: Tuple[str, ...] = ("mul",)
RMSNORM_BODY_OPS: Tuple[str, ...] = (
    "pow",
    "mul",          # mul(x, x) variant of squaring
    "reduce_mean",
    "reduce_sum",   # some traces emit reduce_sum + real_div(N) instead of reduce_mean
    "real_div",
    "add",          # epsilon add
    "rsqrt",
)
L2_TAIL_ANCHOR_OPS: Tuple[str, ...] = ("real_div",)


# ---------------------------------------------------------------------------
# PyTorch model build — duplicated from
# scripts/convert-xtr-to-coreml.py so this investigation tool stays
# self-contained.
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


def build_traceable_module(pipeline: Pipeline):
    """Wrap encoder + projection + L2-norm in a single torch.nn.Module.

    Identical structure to ``scripts/convert-xtr-to-coreml.py``'s
    ``build_traceable_module``: emits ``raw_projected`` and
    ``normalised`` so the parity check uses the same outputs as
    production.
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
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)
            normalised = raw / norm
            return raw, normalised

    module = XTREncoder()
    module.eval()
    return module


# ---------------------------------------------------------------------------
# Tokeniser + sliding-window helpers — duplicated verbatim from the
# production script.
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


def load_parity_inputs(fixtures_json: Path) -> List[Tuple[str, str]]:
    """Read the three non-empty committed fixtures (whitespace skipped)."""
    data = json.loads(fixtures_json.read_text())
    out: List[Tuple[str, str]] = []
    for entry in data.get("fixtures", []):
        text = entry.get("input", "")
        if text and text.strip():
            out.append((entry["name"], text))
    return out


# ---------------------------------------------------------------------------
# Pass-ordering verification.
# ---------------------------------------------------------------------------

def verify_pass_pipeline_order() -> Dict[str, int]:
    """Confirm the canonical pipeline orders ``fuse_layernorm_or_instancenorm``
    before ``add_fp16_cast``, so a custom promotion pass inserted between
    them sees the post-fusion graph and is honoured before fp16 casts go in.
    """
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline

    pipeline = PassPipeline.get_pipeline("default")
    passes = list(pipeline.passes)
    if "common::fuse_layernorm_or_instancenorm" not in passes:
        raise RuntimeError(
            "common::fuse_layernorm_or_instancenorm not in default pipeline; "
            "coremltools API drift — investigation cannot proceed."
        )
    if "common::add_fp16_cast" not in passes:
        raise RuntimeError(
            "common::add_fp16_cast not in default pipeline; "
            "coremltools API drift — investigation cannot proceed."
        )
    fuse_idx = passes.index("common::fuse_layernorm_or_instancenorm")
    fp16_idx = passes.index("common::add_fp16_cast")
    if fuse_idx >= fp16_idx:
        raise RuntimeError(
            f"Pass ordering invariant violated: fuse_layernorm at {fuse_idx} "
            f">= add_fp16_cast at {fp16_idx}"
        )
    return {
        "fuse_layernorm_idx": fuse_idx,
        "add_fp16_cast_idx": fp16_idx,
        "n_passes": len(passes),
    }


# ---------------------------------------------------------------------------
# Graph introspection.
# ---------------------------------------------------------------------------

def iter_program_ops(prog) -> List["object"]:  # noqa: ANN001
    """Walk every Operation in every block of every function of a MIL program."""
    out: List["object"] = []

    def visit_block(block):
        for op in block.operations:
            out.append(op)
            for nested in getattr(op, "blocks", []) or []:
                visit_block(nested)

    for func in prog.functions.values():
        visit_block(func)
    return out


def op_input_producers(op) -> List["object"]:  # noqa: ANN001
    """Yield each producer Operation feeding ``op``'s inputs (deduplicated)."""
    seen: List["object"] = []
    seen_ids: Set[int] = set()
    for value in op.inputs.values():
        # Inputs may be a single Var or a list of Vars (variadic).
        if isinstance(value, (list, tuple)):
            iterable = value
        else:
            iterable = [value]
        for var in iterable:
            producer = getattr(var, "op", None)
            if producer is None or id(producer) in seen_ids:
                continue
            seen_ids.add(id(producer))
            seen.append(producer)
    return seen


def op_output_consumers(op) -> List["object"]:  # noqa: ANN001
    """Yield each Operation that consumes any output Var of ``op``."""
    seen: List["object"] = []
    seen_ids: Set[int] = set()
    outputs = op.outputs if hasattr(op, "outputs") else []
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    for var in outputs:
        for child in getattr(var, "child_ops", []) or []:
            if id(child) in seen_ids:
                continue
            seen_ids.add(id(child))
            seen.append(child)
    return seen


def summarise_op_types(prog) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for op in iter_program_ops(prog):
        counts[op.op_type] = counts.get(op.op_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Structural matcher — the heart of the investigation.
#
# Anchors:
#   - Each rsqrt op marks the centre of an RMSNorm decomposed cluster.
#   - The L2-norm tail anchor is a real_div whose denominator chains
#     through clamp/maximum/sqrt + reduce_sum/pow back to the
#     projection-output activation.
#
# From an rsqrt anchor, the cluster is built by walking parents through
# {add(eps), reduce_mean/reduce_sum/real_div, pow/mul²} and walking
# children through {mul(x, ·), mul(γ, ·)}. Walking is type-guarded so
# we do not over-grow into surrounding linear / matmul ops.
# ---------------------------------------------------------------------------

def _is_square(op) -> bool:
    """True if the op squares its input (pow(x, 2) or mul(x, x))."""
    if op.op_type == "pow":
        # pow(x, exponent). Exponent must be a constant 2.
        exp = op.inputs.get("y")
        if exp is None:
            return False
        val = getattr(exp, "val", None)
        try:
            return val is not None and float(val) == 2.0
        except (TypeError, ValueError):
            return False
    if op.op_type == "mul":
        # mul(x, x) where both inputs are the same SSA Var.
        x = op.inputs.get("x")
        y = op.inputs.get("y")
        return x is not None and y is not None and id(x) == id(y)
    return False


def _walk_rmsnorm_cluster(rsqrt_op) -> Set["object"]:  # noqa: ANN001
    """Return the set of Operations forming the RMSNorm cluster around
    ``rsqrt_op``. Intentionally local — does not traverse into linears
    or downstream blocks.
    """
    cluster: Set["object"] = {rsqrt_op}

    # --- Walk parents from rsqrt up to the squaring op. --------------
    # Expected chain: rsqrt ← add(eps) ← reduce_mean ← (pow/mul²)
    # or:             rsqrt ← add(eps) ← real_div(reduce_sum, N) ← pow.
    frontier = list(op_input_producers(rsqrt_op))
    visited: Set[int] = {id(rsqrt_op)}
    for op in frontier:
        visited.add(id(op))
    while frontier:
        op = frontier.pop()
        if op.op_type in {"add", "reduce_mean", "reduce_sum", "real_div"}:
            cluster.add(op)
            for parent in op_input_producers(op):
                if id(parent) not in visited:
                    visited.add(id(parent))
                    frontier.append(parent)
        elif _is_square(op):
            cluster.add(op)
            # do not walk further past the square — its input is the
            # un-normalised activation, which belongs to upstream
            # ops we treat separately by island.
        # constant operands (eps, divisor) are not promoted; their
        # producer ops are immutable consts.

    # --- Walk children from rsqrt down through the two muls. ---------
    # Expected chain: rsqrt → mul(x, rsqrt) → mul(γ, ·)
    children = op_output_consumers(rsqrt_op)
    seen: Set[int] = {id(rsqrt_op)}
    pending = list(children)
    hops = 0
    while pending and hops < 4:
        next_pending: List["object"] = []
        for op in pending:
            if id(op) in seen:
                continue
            seen.add(id(op))
            if op.op_type in RMSNORM_TAIL_MUL_OPS:
                cluster.add(op)
                next_pending.extend(op_output_consumers(op))
        pending = next_pending
        hops += 1

    return cluster


def _residual_add_for_cluster(cluster: Set["object"]) -> Optional["object"]:  # noqa: ANN001
    """Identify the residual ``add`` op that feeds the RMSNorm cluster.

    The squaring op's input is the un-normalised activation; that
    activation's producer is the residual sum if RMSNorm is preceded by
    a residual addition (T5's per-sublayer pattern).
    """
    for op in cluster:
        if not _is_square(op):
            continue
        for parent in op_input_producers(op):
            if parent.op_type == "add":
                return parent
    return None


def _is_sqrt_like(op) -> bool:
    """Recognise an L2-norm-style square root: either a literal ``sqrt``
    op, or a ``pow(x, 0.5)``. Recent coremltools traces of
    ``torch.linalg.vector_norm`` lower the square root as the latter.
    """
    if op.op_type == "sqrt":
        return True
    if op.op_type == "pow":
        exp = op.inputs.get("y")
        val = getattr(exp, "val", None)
        try:
            return val is not None and abs(float(val) - 0.5) < 1e-6
        except (TypeError, ValueError):
            return False
    return False


def _walk_l2_norm_tail(prog) -> Set["object"]:  # noqa: ANN001
    """Locate the projection-output L2-norm tail.

    Pattern after standard MIL lowering of
    ``raw / vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-12)``:

        pow(raw, 2)               # or mul(raw, raw)
            ↓
        reduce_sum(axes=[-1], keep_dims=True)
            ↓
        sqrt  /  pow(·, 0.5)      # coremltools 7.2 lowers .sqrt() as pow(·, 0.5)
            ↓
        maximum(·, 1e-12)         # clamp_min lowering
            ↓
        real_div(raw, ·)          # the model's "normalised" output

    Anchor on the ``real_div`` whose denominator chains through
    ``maximum → (sqrt|pow0.5) → reduce_sum/reduce_mean → pow²/mul²``.
    The single L2-norm tail in the production graph has exactly this
    shape.
    """
    cluster: Set["object"] = set()
    candidates: List["object"] = []
    for op in iter_program_ops(prog):
        if op.op_type != "real_div":
            continue
        y_var = op.inputs.get("y")
        if y_var is None:
            continue
        producer = getattr(y_var, "op", None)
        if producer is None:
            continue
        # walk denominator chain up to a sqrt-like op
        chain: List["object"] = [op]
        cursor = producer
        guard = 0
        while cursor is not None and guard < 8:
            chain.append(cursor)
            if _is_sqrt_like(cursor):
                upstream = op_input_producers(cursor)
                if upstream and upstream[0].op_type in {
                    "reduce_sum",
                    "reduce_mean",
                }:
                    chain.append(upstream[0])
                    sq_parents = op_input_producers(upstream[0])
                    if sq_parents and (
                        sq_parents[0].op_type == "pow" or _is_square(sq_parents[0])
                    ):
                        chain.append(sq_parents[0])
                candidates.append(op)
                for c in chain:
                    cluster.add(c)
                break
            cursor_parents = op_input_producers(cursor)
            cursor = cursor_parents[0] if cursor_parents else None
            guard += 1

    if not candidates:
        return set()
    return cluster


def _is_neutral_op(op) -> bool:
    """Ops that are safe to leave at FP16 even in the most aggressive
    Island E (everything-except-Linears) configuration. Constants and
    casts must stay at their assigned precision.
    """
    return op.op_type in {
        "const",
        "cast",
    }


def find_promotion_set(prog, island: str) -> Set[int]:  # noqa: ANN001
    """Return the set of ``id(op)`` for ops that should stay at FP32.

    Walks the post-fusion MIL program; the caller passes the same
    ``prog`` object that the precision pass will see (i.e. this runs as
    part of an inserted graph pass between
    ``common::fuse_layernorm_or_instancenorm`` and
    ``common::add_fp16_cast``).
    """
    if island not in ISLANDS:
        raise ValueError(f"Unknown island {island!r}; expected one of {ISLANDS}")

    promotion: Set[int] = set()

    # Island E first (if requested) is the simplest: keep everything
    # except linear-class ops at FP32. This is the upper bound of what
    # a "targeted" carve-out can become.
    if island == "E":
        for op in iter_program_ops(prog):
            if op.op_type in {"linear", "matmul", "conv", "conv_transpose"}:
                continue
            if _is_neutral_op(op):
                continue
            promotion.add(id(op))
        return promotion

    # Otherwise, build up from the RMSNorm cluster outward.
    rsqrt_anchors = [
        op for op in iter_program_ops(prog) if op.op_type == RMSNORM_ANCHOR_OP
    ]
    rmsnorm_clusters: List[Set["object"]] = []
    for anchor in rsqrt_anchors:
        cluster = _walk_rmsnorm_cluster(anchor)
        rmsnorm_clusters.append(cluster)

    for cluster in rmsnorm_clusters:
        for op in cluster:
            promotion.add(id(op))

    if island in {"B", "C", "D"}:
        for cluster in rmsnorm_clusters:
            residual = _residual_add_for_cluster(cluster)
            if residual is not None:
                promotion.add(id(residual))

    if island in {"C", "D"}:
        l2_cluster = _walk_l2_norm_tail(prog)
        for op in l2_cluster:
            promotion.add(id(op))

    if island == "D":
        # One more transitive hop: also promote the upstream add ops
        # that feed each residual add. This is the "FFN/attention output
        # add" — already-residual-summed activations from the previous
        # block. In practice, since each sublayer's residual is itself a
        # sum, this means promoting two adds per sublayer instead of one.
        for cluster in rmsnorm_clusters:
            residual = _residual_add_for_cluster(cluster)
            if residual is None:
                continue
            for parent in op_input_producers(residual):
                if parent.op_type == "add":
                    promotion.add(id(parent))

    return promotion


# ---------------------------------------------------------------------------
# Custom MIL pass that records the promotion set into a closure-shared
# dict so the FP16ComputePrecision selector can read it.
# ---------------------------------------------------------------------------

PROMOTION_REGISTRY: Dict[str, Set[int]] = {}
"""Keyed by an arbitrary tag so concurrent island sweeps don't collide."""

_REGISTERED_PASSES: Set[str] = set()


def make_promotion_pass(tag: str, island: str):
    """Build (and register, if needed) an AbstractGraphPass that
    populates ``PROMOTION_REGISTRY[tag]`` for the given island.

    Returns the registered pass name to be inserted into the pipeline.
    """
    from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
    from coremltools.converters.mil.mil.passes.pass_registry import register_pass

    pass_name = f"common::switchcraft_promote_{tag}"
    if pass_name in _REGISTERED_PASSES:
        return pass_name

    class _PromotionDiscoveryPass(AbstractGraphPass):
        def apply(self, prog):
            promo = find_promotion_set(prog, island)
            PROMOTION_REGISTRY[tag] = promo

    # The decorator-style register_pass takes a namespace. Use the
    # plain function form available in coremltools 7.2.
    namespace, name = pass_name.split("::", 1)
    register_pass(namespace=namespace, name=name)(_PromotionDiscoveryPass)
    _REGISTERED_PASSES.add(pass_name)
    return pass_name


def make_op_selector(tag: str) -> Callable:
    """Return an op_selector that returns False (keep at FP32) for ops
    in ``PROMOTION_REGISTRY[tag]`` and True (allow FP16 cast) elsewhere.
    """
    def selector(op) -> bool:
        promo = PROMOTION_REGISTRY.get(tag)
        if promo is None:
            return True
        return id(op) not in promo
    return selector


# ---------------------------------------------------------------------------
# CoreML conversion.
# ---------------------------------------------------------------------------

@dataclass
class ConversionRecord:
    label: str
    island: Optional[str]
    convert_seconds: float
    n_promoted: int
    n_total_ops: int
    op_counts: Dict[str, int]
    asset_size_bytes: int
    convert_error: Optional[str]
    saw_nan: bool
    # Per-fixture cosine over the production-shape (tail-aligned) row
    # arrays. Set to 0.0 when shapes differ — same convention as
    # scripts/convert-xtr-to-coreml.py:parity_check.
    per_fixture_cosine: Dict[str, float]
    # Per-fixture cosine computed over the *position intersection* of
    # PyTorch reference and CoreML MIN_NORM-survivors. This is the
    # honest go/no-go signal when the two filter masks differ slightly.
    per_fixture_aligned_cosine: Dict[str, float]
    per_fixture_aligned_overlap: Dict[str, int]
    per_fixture_rows: Dict[str, Tuple[int, int]]
    per_fixture_zero_rows: Dict[str, int]
    per_fixture_row_norm_p99: Dict[str, float]
    mean_cosine: float
    mean_aligned_cosine: float


def _convert_with_pipeline(traced_module, *, custom_pass_name: Optional[str],
                           selector: Optional[Callable], precision: str):
    """Run ``ct.convert`` with a configured pipeline.

    ``precision`` ∈ {"fp32", "fp16"}; selector and custom_pass_name are
    only meaningful for "fp16".
    """
    import numpy as np
    import coremltools as ct
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline

    pipeline = PassPipeline.get_pipeline("default")
    if precision == "fp16" and custom_pass_name is not None:
        # Insert the discovery pass immediately before add_fp16_cast.
        idx = pipeline.passes.index("common::add_fp16_cast")
        pipeline.insert_pass(index=idx, pass_name=custom_pass_name)

    if precision == "fp16" and selector is not None:
        compute_precision = ct.transform.FP16ComputePrecision(op_selector=selector)
    elif precision == "fp16":
        compute_precision = ct.precision.FLOAT16
    else:
        compute_precision = ct.precision.FLOAT32

    return ct.convert(
        traced_module,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, WINDOW_SIZE), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="raw_projected", dtype=np.float16),
            ct.TensorType(name="normalised", dtype=np.float16),
        ],
        compute_precision=compute_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        pass_pipeline=pipeline,
    )


def trace_module(traceable):
    import torch
    example_ids = torch.zeros((1, WINDOW_SIZE), dtype=torch.long)
    return torch.jit.trace(traceable, example_ids, strict=False)


# ---------------------------------------------------------------------------
# Parity scaffolding — duplicated from the production script.
# ---------------------------------------------------------------------------

def encode_pytorch_reference(pipeline: Pipeline, tokenizer, text: str):
    """Production-shape parity reference: applies MIN_NORM and returns
    only the surviving rows.
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


def encode_pytorch_position_aligned(pipeline: Pipeline, tokenizer, text: str):
    """Return per-token-position normalised vectors plus the MIN_NORM
    survival mask, indexed by absolute token position.

    Used by the cross-stack cosine comparison so that we can report
    cosine on the *intersection* of positions where both the FP32
    PyTorch reference and the candidate CoreML model produce a usable
    row, instead of dropping the whole fixture when the two filter
    masks differ by a single position.
    """
    import torch
    import numpy as np

    if not text.strip():
        return np.zeros((0, DIMS), dtype=np.float32), np.zeros((0,), dtype=bool)

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

    aligned = np.zeros((T, DIMS), dtype=np.float32)
    mask = np.zeros((T,), dtype=bool)
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
        aligned[pos] = (v / n).astype(np.float32)
        mask[pos] = True
    return aligned, mask


def encode_coreml(mlmodel, tokenizer, text: str):
    """Run CoreML predict over the sliding-window plan; track NaN +
    raw-row-norm percentile + position-aligned vectors so we can
    compare cosine on the intersection of MIN_NORM survivors with the
    PyTorch reference, not just on tail-aligned production rows.
    """
    import numpy as np

    empty = np.zeros((0, DIMS), dtype=np.float32)
    empty_mask = np.zeros((0,), dtype=bool)
    if not text.strip():
        return empty, False, 0, 0.0, empty, empty_mask

    ids = tokenize(tokenizer, text)
    starts = slide(ids)
    T = len(ids)
    sum_normalised = np.zeros((T, DIMS), dtype=np.float64)
    sum_raw_norm = np.zeros((T,), dtype=np.float64)
    counts = np.zeros((T,), dtype=np.int64)
    saw_nan = False
    per_window_norms: List[float] = []
    zero_rows = 0

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
            r_norm = float(np.linalg.norm(raw[local_row]))
            sum_raw_norm[pos] += r_norm
            sum_normalised[pos] += normalised[local_row]
            per_window_norms.append(r_norm)
            if r_norm == 0.0:
                zero_rows += 1

    rows: List["np.ndarray"] = []
    aligned = np.zeros((T, DIMS), dtype=np.float32)
    mask = np.zeros((T,), dtype=bool)
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
        unit = (v / n).astype(np.float32)
        aligned[pos] = unit
        mask[pos] = True
        rows.append(unit)
    arr = np.stack(rows, axis=0) if rows else np.zeros((0, DIMS), dtype=np.float32)
    p99 = float(np.percentile(per_window_norms, 99)) if per_window_norms else 0.0
    return arr, saw_nan, zero_rows, p99, aligned, mask


def cosine_similarity(a, b) -> float:
    import numpy as np
    if a.shape != b.shape or a.size == 0:
        return 1.0 if (a.size == 0 and b.size == 0) else 0.0
    sims = (a * b).sum(axis=1) / (
        (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    )
    return float(sims.mean())


# ---------------------------------------------------------------------------
# Asset-size helper.
# ---------------------------------------------------------------------------

def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Sweep orchestration.
# ---------------------------------------------------------------------------

def run_island(
    *,
    pipeline_orig: Pipeline,
    traced_module,
    island: Optional[str],
    label: str,
    fixture_inputs: List[Tuple[str, str]],
    tokenizer,
    out_dir: Path,
    keep_mlpackages: bool,
) -> ConversionRecord:
    """Convert with a given island promotion set, parity-check, return record.

    ``island=None`` means "no promotion, plain FP16" — the negative
    control. Set ``island='__fp32__'`` for a full-FP32 reference run.
    """
    print(f"\n=== {label} ===")
    sys.stdout.flush()
    out_path = out_dir / f"{label}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)

    convert_error: Optional[str] = None
    mlmodel = None
    n_promoted = 0
    n_total_ops = 0
    op_counts: Dict[str, int] = {}
    convert_seconds = 0.0
    asset_size = 0
    saw_any_nan = False

    t0 = time.time()
    try:
        if island == "__fp32__":
            mlmodel = _convert_with_pipeline(
                traced_module,
                custom_pass_name=None,
                selector=None,
                precision="fp32",
            )
        elif island is None:
            # plain FP16, no selector — the negative control.
            mlmodel = _convert_with_pipeline(
                traced_module,
                custom_pass_name=None,
                selector=None,
                precision="fp16",
            )
        else:
            tag = f"{label}-{int(time.time()*1000)}"
            pass_name = make_promotion_pass(tag, island)
            selector = make_op_selector(tag)
            mlmodel = _convert_with_pipeline(
                traced_module,
                custom_pass_name=pass_name,
                selector=selector,
                precision="fp16",
            )
            promo = PROMOTION_REGISTRY.get(tag, set())
            n_promoted = len(promo)
    except Exception as exc:  # noqa: BLE001
        convert_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    convert_seconds = time.time() - t0

    if mlmodel is not None:
        try:
            prog = mlmodel._mil_program  # noqa: SLF001
            ops = iter_program_ops(prog)
            n_total_ops = len(ops)
            op_counts = summarise_op_types(prog)
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: could not introspect mil_program: {exc}")
        try:
            mlmodel.short_description = f"FP32-promote investigation [{label}]"
            mlmodel.author = "Switchcraft investigation (Apache 2.0)"
            mlmodel.save(str(out_path))
            asset_size = directory_size_bytes(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: could not save mlpackage: {exc}")

    per_fixture: Dict[str, float] = {}
    per_fixture_aligned: Dict[str, float] = {}
    per_fixture_aligned_overlap: Dict[str, int] = {}
    per_fixture_rows: Dict[str, Tuple[int, int]] = {}
    per_fixture_zero_rows: Dict[str, int] = {}
    per_fixture_row_norm_p99: Dict[str, float] = {}

    import numpy as _np

    if mlmodel is not None:
        for name, text in fixture_inputs:
            if not text.strip():
                continue
            try:
                ref = encode_pytorch_reference(pipeline_orig, tokenizer, text)
                ref_aligned, ref_mask = encode_pytorch_position_aligned(
                    pipeline_orig, tokenizer, text
                )
                cm, saw_nan, zero_rows, p99, cm_aligned, cm_mask = encode_coreml(
                    mlmodel, tokenizer, text
                )
                saw_any_nan = saw_any_nan or saw_nan
                per_fixture_rows[name] = (int(ref.shape[0]), int(cm.shape[0]))
                per_fixture_zero_rows[name] = int(zero_rows)
                per_fixture_row_norm_p99[name] = float(p99)

                # Position-aligned cosine on the intersection of masks.
                if ref_mask.size and cm_mask.size and ref_mask.shape == cm_mask.shape:
                    overlap = ref_mask & cm_mask
                    n_overlap = int(overlap.sum())
                    if n_overlap > 0:
                        a = ref_aligned[overlap]
                        b = cm_aligned[overlap]
                        # Drop any rows whose CoreML vector contains
                        # NaN/Inf — they survived MIN_NORM only because
                        # the *raw* row norm was finite, but a NaN can
                        # leak into the normalised output via 0/0 or
                        # inf/inf in FP16. They're not usable rows.
                        finite_mask = _np.isfinite(b).all(axis=1)
                        a = a[finite_mask]
                        b = b[finite_mask]
                        n_finite = int(finite_mask.sum())
                        if n_finite > 0:
                            sims = (a * b).sum(axis=1) / (
                                (_np.linalg.norm(a, axis=1) * _np.linalg.norm(b, axis=1)) + 1e-12
                            )
                            per_fixture_aligned[name] = float(sims.mean())
                        else:
                            per_fixture_aligned[name] = 0.0
                        per_fixture_aligned_overlap[name] = n_finite
                    else:
                        per_fixture_aligned[name] = 0.0
                        per_fixture_aligned_overlap[name] = 0
                else:
                    per_fixture_aligned[name] = 0.0
                    per_fixture_aligned_overlap[name] = 0

                if ref.shape != cm.shape:
                    print(
                        f"  parity FAIL [{name}]: shapes differ "
                        f"{ref.shape} vs {cm.shape}  zero_rows={zero_rows}  "
                        f"aligned_cos={per_fixture_aligned[name]:.6f} "
                        f"(overlap={per_fixture_aligned_overlap[name]})",
                        file=sys.stderr,
                    )
                    per_fixture[name] = 0.0
                    continue
                sim = cosine_similarity(ref, cm)
                print(
                    f"  parity [{name}]: rows={ref.shape[0]} mean cos={sim:.6f} "
                    f"aligned_cos={per_fixture_aligned[name]:.6f} "
                    f"zero_rows={zero_rows} norm_p99={p99:.4f}"
                )
                per_fixture[name] = sim
            except Exception as exc:  # noqa: BLE001
                print(f"  parity ERROR [{name}]: {exc}", file=sys.stderr)
                per_fixture[name] = float("nan")
                per_fixture_aligned[name] = float("nan")

    if mlmodel is not None and out_path.exists() and not keep_mlpackages:
        shutil.rmtree(out_path)

    finite = [v for v in per_fixture.values() if math.isfinite(v)]
    mean = sum(finite) / len(finite) if finite else float("nan")
    aligned_finite = [v for v in per_fixture_aligned.values() if math.isfinite(v)]
    mean_aligned = (
        sum(aligned_finite) / len(aligned_finite) if aligned_finite else float("nan")
    )
    return ConversionRecord(
        label=label,
        island=island,
        convert_seconds=convert_seconds,
        n_promoted=n_promoted,
        n_total_ops=n_total_ops,
        op_counts=op_counts,
        asset_size_bytes=asset_size,
        convert_error=convert_error,
        saw_nan=saw_any_nan,
        per_fixture_cosine=per_fixture,
        per_fixture_aligned_cosine=per_fixture_aligned,
        per_fixture_aligned_overlap=per_fixture_aligned_overlap,
        per_fixture_rows=per_fixture_rows,
        per_fixture_zero_rows=per_fixture_zero_rows,
        per_fixture_row_norm_p99=per_fixture_row_norm_p99,
        mean_cosine=mean,
        mean_aligned_cosine=mean_aligned,
    )


def render_sweep_table(records: List[ConversionRecord]) -> str:
    header = (
        "| Label | Island | NaN-free? | Promoted ops | Total ops | "
        "FP32 fraction | Asset MB | Tail-aligned cos | Position-aligned cos | "
        "Notes |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for r in records:
        cos = (
            f"{r.mean_cosine:.6f}" if not math.isnan(r.mean_cosine) else "NaN"
        )
        aligned = (
            f"{r.mean_aligned_cosine:.6f}"
            if not math.isnan(r.mean_aligned_cosine)
            else "NaN"
        )
        nan = "no" if r.saw_nan else ("n/a" if r.n_total_ops == 0 else "yes")
        size_mb = r.asset_size_bytes / (1024 * 1024)
        frac = (r.n_promoted / r.n_total_ops) if r.n_total_ops else 0.0
        notes = r.convert_error or ""
        island = r.island if r.island is not None else "(no promote)"
        rows.append(
            f"| {r.label} | {island} | {nan} | {r.n_promoted} | {r.n_total_ops} | "
            f"{frac*100:.1f}% | {size_mb:.1f} | {cos} | {aligned} | {notes} |"
        )
    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Figure rendering.
# ---------------------------------------------------------------------------

def render_op_count_figure(records: List[ConversionRecord], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r.label for r in records]
    promoted = [r.n_promoted for r in records]
    fp16 = [max(r.n_total_ops - r.n_promoted, 0) for r in records]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
    x = list(range(len(labels)))
    ax.bar(x, fp16, label="FP16", color="steelblue")
    ax.bar(x, promoted, bottom=fp16, label="FP32 (promoted)", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MIL ops")
    ax.set_title("Op count by precision tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_parity_figure(records: List[ConversionRecord], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r.label for r in records]
    tail = [
        r.mean_cosine if not math.isnan(r.mean_cosine) else 0.0 for r in records
    ]
    aligned = [
        r.mean_aligned_cosine if not math.isnan(r.mean_aligned_cosine) else 0.0
        for r in records
    ]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, tail, width, label="Tail-aligned (production)", color="steelblue")
    ax.bar(x + width / 2, aligned, width, label="Position-aligned (intersection)", color="seagreen")
    ax.axhline(0.999, color="red", linestyle="--", label="≥0.999 gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean cosine vs FP32 PyTorch")
    ax.set_ylim(0, 1.05)
    ax.set_title("Parity by island")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_size_figure(records: List[ConversionRecord], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r.label for r in records]
    sizes = [r.asset_size_bytes / (1024 * 1024) for r in records]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
    ax.bar(labels, sizes, color="darkorange")
    ax.set_ylabel(".mlpackage size (MB)")
    ax.set_title("Asset size by island")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/switchcraft-mil-fp32-promote-investigation"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/investigations/mil-fp32-promote-figures"),
    )
    parser.add_argument(
        "--islands",
        type=str,
        default=",".join(ISLANDS),
        help="Comma-separated subset of islands to sweep (default A,B,C,D,E).",
    )
    parser.add_argument("--keep-mlpackages", action="store_true")
    parser.add_argument(
        "--skip-fp32-reference",
        action="store_true",
        help="Skip the FP32 reference conversion (saves ~3 min).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Suppress matplotlib figure rendering.",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.tokenizer.exists():
        parser.error(f"--tokenizer not found: {args.tokenizer}")
    if not args.fixtures_json.exists():
        parser.error(f"--fixtures-json not found: {args.fixtures_json}")

    islands = [s.strip().upper() for s in args.islands.split(",") if s.strip()]
    for isl in islands:
        if isl not in ISLANDS:
            parser.error(f"unknown island {isl!r}; expected subset of {ISLANDS}")

    # Determinism for any sub-sampling RNG that downstream code uses.
    import random as _random
    import torch as _torch
    _random.seed(0)
    _torch.manual_seed(0)

    print("Verifying coremltools pass pipeline ordering…")
    pipeline_info = verify_pass_pipeline_order()
    print(
        f"  fuse_layernorm @ {pipeline_info['fuse_layernorm_idx']}, "
        f"add_fp16_cast @ {pipeline_info['add_fp16_cast_idx']}, "
        f"total {pipeline_info['n_passes']} passes."
    )

    print(f"Loading {args.model_id} @ {args.revision}…")
    t0 = time.time()
    pipeline = build_pytorch_pipeline(args.model_id, args.revision)
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"  loaded in {time.time() - t0:.1f}s")

    parity_inputs = load_parity_inputs(args.fixtures_json)
    print(f"Parity inputs: {[name for name, _ in parity_inputs]}")

    traceable = build_traceable_module(pipeline)
    print("Tracing torch module…")
    t0 = time.time()
    traced = trace_module(traceable)
    print(f"  traced in {time.time() - t0:.1f}s")

    records: List[ConversionRecord] = []

    # ---- Plain FP16 baseline (negative control). -----------------------
    records.append(
        run_island(
            pipeline_orig=pipeline,
            traced_module=traced,
            island=None,
            label="fp16-no-promote",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
        )
    )

    # ---- FP32 reference (sanity / introspection). ----------------------
    fp32_record: Optional[ConversionRecord] = None
    if not args.skip_fp32_reference:
        fp32_record = run_island(
            pipeline_orig=pipeline,
            traced_module=traced,
            island="__fp32__",
            label="fp32-reference",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
        )
        records.append(fp32_record)
        # Dump the FP32-reference op counts and a few canonical
        # signature samples to JSON so the report has the post-fusion
        # graph shape on record.
        graph_dump = {
            "n_total_ops": fp32_record.n_total_ops,
            "op_counts": fp32_record.op_counts,
        }
        (args.out_dir / "fp32-graph-summary.json").write_text(
            json.dumps(graph_dump, indent=2) + "\n"
        )
        print(f"  → {args.out_dir / 'fp32-graph-summary.json'}")

    # ---- Pre-flight: dry-run the matcher on the FP32 reference graph
    # ---- so the report can document matcher hit counts independently
    # ---- of the per-island sweeps that re-trace.
    if fp32_record is not None and fp32_record.n_total_ops > 0:
        # Re-convert at FP32 just to inspect the matcher output. Cheap
        # enough since the heavy lifting is the trace.
        try:
            mlmodel_intro = _convert_with_pipeline(
                traced,
                custom_pass_name=None,
                selector=None,
                precision="fp32",
            )
            prog = mlmodel_intro._mil_program  # noqa: SLF001
            matcher_counts = {}
            for isl in ISLANDS:
                ids = find_promotion_set(prog, isl)
                matcher_counts[isl] = len(ids)
            (args.out_dir / "matcher-hit-counts.json").write_text(
                json.dumps(matcher_counts, indent=2) + "\n"
            )
            print(f"  matcher hit counts: {matcher_counts}")
            del mlmodel_intro
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: matcher dry-run failed: {exc}")

    # ---- Island sweep. -------------------------------------------------
    for isl in islands:
        rec = run_island(
            pipeline_orig=pipeline,
            traced_module=traced,
            island=isl,
            label=f"island-{isl}",
            fixture_inputs=parity_inputs,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            keep_mlpackages=args.keep_mlpackages,
        )
        records.append(rec)

    # ---- Emit results. -------------------------------------------------
    table = render_sweep_table(records)
    (args.out_dir / "sweep.md").write_text(table)
    print("\nSweep results:\n" + table)

    summary = {
        "model_id": args.model_id,
        "revision": args.revision,
        "pipeline_info": pipeline_info,
        "results": [
            {
                "label": r.label,
                "island": r.island,
                "convert_seconds": r.convert_seconds,
                "n_promoted": r.n_promoted,
                "n_total_ops": r.n_total_ops,
                "asset_size_bytes": r.asset_size_bytes,
                "convert_error": r.convert_error,
                "saw_nan": r.saw_nan,
                "per_fixture_cosine": r.per_fixture_cosine,
                "per_fixture_aligned_cosine": r.per_fixture_aligned_cosine,
                "per_fixture_aligned_overlap": r.per_fixture_aligned_overlap,
                "per_fixture_rows": {
                    k: list(v) for k, v in r.per_fixture_rows.items()
                },
                "per_fixture_zero_rows": r.per_fixture_zero_rows,
                "per_fixture_row_norm_p99": r.per_fixture_row_norm_p99,
                "mean_cosine": r.mean_cosine,
                "mean_aligned_cosine": r.mean_aligned_cosine,
                "op_counts_top": dict(list(r.op_counts.items())[:30]),
            }
            for r in records
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n"
    )
    print(f"Wrote {args.out_dir / 'summary.json'}")

    if not args.no_figures:
        render_op_count_figure(
            records, args.figures_dir / "op-count-by-precision.png"
        )
        render_parity_figure(records, args.figures_dir / "parity-by-island.png")
        render_size_figure(records, args.figures_dir / "asset-size-by-island.png")
        print(f"Wrote figures under {args.figures_dir}")

    return 0


def _json_safe(obj):
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
