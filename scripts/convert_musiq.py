#!/usr/bin/env python3.12
"""Convert pyiqa MUSIQ-AVA → CoreML .mlpackage with static patch-tensor input.

Output shape: [1, 193, 3075] (49 patches for scale 224 + 144 patches for
scale 384, each row = 3072 pixel values + [spatial_pos, scale_id, mask]).

Writes: ImageRater/MLModels/musiq-ava.mlpackage (committed into Xcode project).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import pyiqa

# ---------------------------------------------------------------------------
# Compatibility shim: coremltools 9.0 + numpy 2.x.
#
# Why: coremltools 9.0 `_cast` does `dtype(x.val)` where `x.val` can be a
# length-1 numpy array (rank-1 constant folded from an aten::Int node).  In
# numpy ≥ 2.0 this raises `TypeError: only 0-dimensional arrays can be
# converted to Python scalars`.  Upstream fix is trivial (use `.item()`) but
# not yet released; patch it here so the converter runs on any numpy 2.x host.
# Removing this shim once coremltools ships a fix is safe.
# ---------------------------------------------------------------------------
import coremltools.converters.mil.frontend.torch.ops as _ct_ops
import coremltools.converters.mil.mil.builder as _mb_mod
from coremltools.converters.mil.mil import Builder as mb

_orig_cast = _ct_ops._cast.__wrapped__ if hasattr(_ct_ops._cast, "__wrapped__") else None


def _patched_cast(context, node, dtype, dtype_name):
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if x.can_be_folded_to_const():
        if not isinstance(x.val, dtype):
            # Use .item() to safely extract scalar from any-shape numpy array
            val = x.val.item() if isinstance(x.val, np.ndarray) else x.val
            res = mb.const(val=dtype(val), name=node.name)
        else:
            res = x
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


# Patch the module-level _cast that _int and _bool both call
_ct_ops._cast = _patched_cast


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "ImageRater" / "MLModels" / "musiq-ava.mlpackage"

# Static shape constants matching the Swift preprocessor.
SEQ_LEN = 49 + 144               # scales [224, 384] → max_seq_len (7²) + (12²)
PATCH_DIM = 32 * 32 * 3          # 3072
ROW_DIM = PATCH_DIM + 3          # 3075 (+ spatial_pos, scale_id, mask)


class MUSIQBody(nn.Module):
    """MUSIQ forward skipping `get_multiscale_patches` (Swift provides patches)."""

    def __init__(self, musiq: nn.Module):
        super().__init__()
        self.patch_size = musiq.patch_size
        self.conv_root = musiq.conv_root
        self.gn_root = musiq.gn_root
        self.root_pool = musiq.root_pool
        self.block1 = musiq.block1
        self.embedding = musiq.embedding
        self.transformer_encoder = musiq.transformer_encoder
        self.head = musiq.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, 3075] — already preprocessed patches.
        # Use static constants to avoid aten::Int ops that coremltools 9 cannot convert.
        _B = 1
        _SEQ = SEQ_LEN

        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        # Keep mask as float — bool() produces an int op unsupported by coremltools 9.
        inputs_masks = x[:, :, -1]
        patches = x[:, :, :-3]

        patches = patches.reshape(-1, 3, self.patch_size, self.patch_size)
        f = self.conv_root(patches)
        f = self.gn_root(f)
        f = self.root_pool(f)
        f = self.block1(f)
        f = f.permute(0, 2, 3, 1).reshape(_B, _SEQ, -1)
        f = self.embedding(f)
        f = self.transformer_encoder(
            f, inputs_spatial_positions, inputs_scale_positions, inputs_masks
        )
        q = self.head(f[:, 0])
        q = q.reshape(_B, 1, -1).mean(dim=1)

        # dist_to_mos: E[score] over bins 1..10
        bins = torch.arange(1, q.shape[-1] + 1, dtype=q.dtype, device=q.device)
        mos = (q * bins).sum(dim=-1)
        return mos


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("Loading pyiqa MUSIQ-AVA...", flush=True)
    metric = pyiqa.create_metric("musiq-ava", device="cpu", as_loss=False)
    metric.eval()

    body = MUSIQBody(metric.net).eval()

    # Dummy input matching Swift tensor shape.
    dummy = torch.randn(1, SEQ_LEN, ROW_DIM, dtype=torch.float32)
    # Ensure scale_id plausible (0/1) and mask=1 to exercise full graph.
    dummy[:, :49, -2] = 0.0
    dummy[:, 49:, -2] = 1.0
    dummy[:, :, -1] = 1.0

    print("Tracing...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(body, dummy, strict=False)
        # Freeze to fold aten::Int constant-propagation nodes that coremltools 9
        # cannot convert when running under Torch 2.11.
        traced = torch.jit.freeze(traced)

    print("Converting to CoreML (fp32)...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="patch_tensor", shape=(1, SEQ_LEN, ROW_DIM), dtype=np.float32)],
        outputs=[ct.TensorType(name="mos", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "MUSIQ-AVA aesthetic quality predictor (on-device)"
    mlmodel.author = "Focal"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        import shutil
        shutil.rmtree(OUT_PATH)
    mlmodel.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
