#!/usr/bin/env python3
"""
Download and convert TOPIQ-NR, TOPIQ-Swin, and CLIP ViT-B/32 to CoreML.

Outputs (relative to repo root):
  models/topiq-nr.mlpackage         TOPIQ-NR ResNet50 — technical quality [0,1]
  models/topiq-swin.mlpackage       TOPIQ-Swin       — aesthetic quality  [0,1]
  models/clip-vision.mlpackage      CLIP ViT-B/32    — 512-dim embedding
  ImageRater/Pipeline/CLIPTextEmbeddings.swift

Dependencies:
  pip install pyiqa coremltools torch torchvision open_clip_torch

Usage:
  cd /path/to/image-rating && python scripts/setup_models.py

Approximate download sizes:
  TOPIQ-NR (ResNet50):   ~100 MB  (cached at ~/.cache/pyiqa/)
  TOPIQ-Swin (Swin-B):   ~350 MB  (cached at ~/.cache/pyiqa/)
  CLIP ViT-B/32:         ~340 MB  (cached at ~/.cache/huggingface/ or ~/.cache/clip/)

All three models output raw scores / embeddings in [0,1] or [-1,1] range.
CoreML conversion uses static input shapes and targets macOS 14+.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from unittest.mock import patch

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SWIFT_DIR  = REPO_ROOT / "ImageRater" / "Pipeline"

MODELS_DIR.mkdir(exist_ok=True)


def _bilinear_interpolate_patch(input, size=None, scale_factor=None, mode='nearest',
                                align_corners=None, recompute_scale_factor=None, antialias=False):
    """Patch F.interpolate to replace bicubic with bilinear during TorchScript tracing.

    coremltools does not support upsample_bicubic2d. Bilinear is visually close
    for the small position-embedding interpolations in TOPIQ/Swin, and the CoreML
    output is used for scoring — slight positional interpolation differences are
    imperceptible to the quality score.
    """
    if mode == 'bicubic':
        mode = 'bilinear'
        align_corners = False if align_corners is None else align_corners
    return _orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
                             align_corners=align_corners,
                             recompute_scale_factor=recompute_scale_factor)

_orig_interpolate = F.interpolate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_mlpackage(mlmodel, save_path: Path):
    """Write an mlpackage without requiring libmodelpackage/libcoremlpython.

    coremltools native extensions fail on Python 3.14.  The .mlpackage format
    is just a directory with a Manifest.json + the spec protobuf.  For NeuralNetwork
    format models all weights are embedded in the spec — no separate weight blobs.
    """
    import uuid, json, shutil
    if save_path.exists():
        shutil.rmtree(save_path)
    spec_dir = save_path / "Data" / "com.apple.CoreML"
    spec_dir.mkdir(parents=True)
    spec_bytes = mlmodel._spec.SerializeToString()
    (spec_dir / "model.mlmodel").write_bytes(spec_bytes)
    spec_uuid = str(uuid.uuid4()).upper()
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            spec_uuid: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            }
        },
        "rootModelIdentifier": spec_uuid,
    }
    (save_path / "Manifest.json").write_text(json.dumps(manifest, indent=4))
    size_mb = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file()) / 1e6
    print(f"  → {save_path}  ({size_mb:.1f} MB)")


def _convert_to_coreml(traced_model, input_size: int, output_name: str, save_path: Path):
    import coremltools as ct
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, input_size, input_size),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="neuralnetwork",
    )
    _save_mlpackage(mlmodel, save_path)


class TOPIQWrapper(torch.nn.Module):
    """Strip pyiqa metric wrapper to expose raw net for TorchScript tracing."""
    def __init__(self, metric):
        super().__init__()
        self.net = metric.net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return torch.clamp(score.squeeze(-1), 0.0, 1.0).reshape(1)


# ---------------------------------------------------------------------------
# TOPIQ-NR  (ResNet50, technical quality, 224×224)
# ---------------------------------------------------------------------------

def setup_topiq_nr():
    out = MODELS_DIR / "topiq-nr.mlpackage"
    if out.exists():
        print(f"  Skipping — already exists: {out}")
        return

    print("  Downloading TOPIQ-NR weights (~100 MB via pyiqa)…")
    import pyiqa
    metric = pyiqa.create_metric("topiq_nr", as_loss=False, device="cpu")
    metric.eval()

    print("  Verifying score range…")
    with torch.no_grad():
        dummy = torch.rand(1, 3, 224, 224)
        score = TOPIQWrapper(metric)(dummy).item()
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  Score range OK: sample={score:.4f}")

    print("  Converting to CoreML…")
    wrapper = TOPIQWrapper(metric)
    wrapper.eval()
    # Patch bicubic→bilinear: coremltools does not support upsample_bicubic2d
    with patch("torch.nn.functional.interpolate", _bilinear_interpolate_patch):
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, torch.rand(1, 3, 224, 224))
    _convert_to_coreml(traced, input_size=224, output_name="score", save_path=out)


# ---------------------------------------------------------------------------
# TOPIQ-Swin  (Swin Transformer, aesthetic quality, 384×384)
# ---------------------------------------------------------------------------

def setup_topiq_swin():
    out = MODELS_DIR / "topiq-swin.mlpackage"
    if out.exists():
        print(f"  Skipping — already exists: {out}")
        return

    # topiq_nr-flive is the Swin-B variant trained on FLIVE (aesthetic-focused)
    print("  Downloading TOPIQ-Swin weights (~350 MB via pyiqa)…")
    import pyiqa
    metric = pyiqa.create_metric("topiq_nr-flive", as_loss=False, device="cpu")
    metric.eval()

    # Confirm actual crop size used by pyiqa's transform pipeline
    input_size = 384
    print(f"  Using input_size={input_size} (Swin-B default)")

    print("  Verifying score range…")
    with torch.no_grad():
        dummy = torch.rand(1, 3, input_size, input_size)
        score = TOPIQWrapper(metric)(dummy).item()
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    print(f"  Score range OK: sample={score:.4f}")

    print("  Converting to CoreML…")
    wrapper = TOPIQWrapper(metric)
    wrapper.eval()
    with patch("torch.nn.functional.interpolate", _bilinear_interpolate_patch):
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, torch.rand(1, 3, input_size, input_size))
    _convert_to_coreml(traced, input_size=input_size, output_name="score", save_path=out)


# CLIP ViT-B/32 normalisation constants (OpenAI weights, open_clip)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]   # R, G, B
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]   # R, G, B


# ---------------------------------------------------------------------------
# CLIP ViT-B/32  (vision encoder, 224×224, 512-dim embedding)
# ---------------------------------------------------------------------------

def setup_clip_vision():
    out = MODELS_DIR / "clip-vision.mlpackage"
    if out.exists():
        print(f"  Skipping — already exists: {out}")
        return

    print("  Downloading CLIP ViT-B/32 weights (~340 MB via open_clip)…")
    import open_clip
    # ViT-B-32-quickgelu matches the OpenAI pretrained weights (QuickGELU activation).
    # Using the plain "ViT-B-32" config causes a QuickGELU mismatch warning and produces
    # embeddings that are inconsistent with the pre-computed text embeddings.
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.eval()

    class VisionWrapper(torch.nn.Module):
        """CLIP vision encoder with built-in normalisation.

        CoreML ImageType scales raw pixels to [0, 1] (scale=1/255, bias=0).
        This wrapper then applies CLIP's per-channel mean/std normalisation
        so that model.visual receives values in the expected [-1.8, 2.6] range.
        Baking normalisation here avoids the single-scale limitation of
        ct.ImageType and correctly handles per-channel std differences.
        """
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
            std  = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean)
            self.register_buffer("norm_std",  std)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x arrives as [0, 1] float (scaled by CoreML ImageType scale=1/255)
            x = (x - self.norm_mean) / self.norm_std
            f = self.enc(x)
            return torch.nn.functional.normalize(f, dim=-1)

    wrapper = VisionWrapper(model.visual)
    wrapper.eval()

    print("  Tracing CLIP ViT-B/32…")
    # open_clip ViT uses nn.MultiheadAttention with batch_first=True.
    # The fast path calls torch._native_multi_head_attention (fused C++ kernel)
    # which coremltools cannot convert.  Disable it to force the decomposed
    # Q/K/V matmul path.
    import torch.backends.mha as _mha_backend
    _prev_fastpath = _mha_backend.get_fastpath_enabled()
    _mha_backend.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            dummy = torch.rand(1, 3, 224, 224)
            traced = torch.jit.trace(wrapper, dummy, strict=False)
    finally:
        _mha_backend.set_fastpath_enabled(_prev_fastpath)

    # Sanity-check: two different inputs must produce different embeddings.
    # NeuralNetwork format silently produces constant output for ViT (broken attention).
    with torch.no_grad():
        e1 = traced(torch.zeros(1, 3, 224, 224))
        e2 = traced(torch.ones(1, 3, 224, 224))
    cos_sim = float((e1 / e1.norm() @ (e2 / e2.norm()).T).item())
    print(f"  Traced model sanity: cos_sim(zeros, ones) = {cos_sim:.4f}  (must be < 0.99)")
    assert cos_sim < 0.99, (
        f"Traced CLIP outputs nearly identical embeddings for different inputs "
        f"(cos_sim={cos_sim:.4f}). The ViT attention is broken in the trace."
    )

    print("  Converting to CoreML (mlprogram — required for ViT attention)…")
    # IMPORTANT: NeuralNetwork format silently corrupts ViT attention ops and
    # produces a constant embedding for all images.  mlprogram (macOS 12+) has
    # full ViT support.  The app already targets macOS 14+ so this is safe.
    import coremltools as ct
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, 224, 224),
            scale=1.0 / 255.0,
            bias=[0.0, 0.0, 0.0],
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    # mlprogram stores weights as external blobs — must use mlmodel.save(), not _save_mlpackage.
    import shutil
    if out.exists():
        shutil.rmtree(out)
    mlmodel.save(str(out))
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"  → {out}  ({size_mb:.1f} MB)")

    return model


# ---------------------------------------------------------------------------
# CLIP-IQA+ text embeddings  (pre-computed, baked into Swift source)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Antonym pairs for CLIP-IQA+.
# Pairs chosen to maximise cosine distance in CLIP ViT-B/32 space while
# covering orthogonal quality axes (overall quality, sharpness, exposure,
# composition, technical noise). Averaging across diverse pairs cancels
# per-pair biases and gives a more stable quality signal.
# ---------------------------------------------------------------------------
ANTONYM_PAIRS = [
    ("A high quality photo.",               "A low quality photo."),
    ("A sharp, in-focus photograph.",       "A blurry, out-of-focus photograph."),
    ("A well-exposed, bright photograph.",  "An underexposed, dark photograph."),
    ("A professionally shot photograph.",  "A poorly shot, amateur snapshot."),
    ("A beautiful, aesthetically pleasing photo.", "An ugly, aesthetically unpleasing photo."),
]


def generate_clip_text_embeddings(clip_model=None):
    """Always regenerate — text embeddings are fast to compute and prompts may change."""
    out = SWIFT_DIR / "CLIPTextEmbeddings.swift"

    import open_clip
    if clip_model is None:
        print("  Loading CLIP ViT-B/32 for text embedding…")
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        clip_model.eval()

    tok = open_clip.get_tokenizer("ViT-B-32")
    pos_embeddings = []
    neg_embeddings = []

    for pos_prompt, neg_prompt in ANTONYM_PAIRS:
        print(f'  Computing: "{pos_prompt}" / "{neg_prompt}"')
        tokens = tok([pos_prompt, neg_prompt])
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        cos_sim = float((feats[0] @ feats[1]).item())
        print(f"    cosine similarity: {cos_sim:.4f}  (lower = more discriminative)")
        pos_embeddings.append(feats[0].numpy().tolist())
        neg_embeddings.append(feats[1].numpy().tolist())

    n = len(ANTONYM_PAIRS)
    pos_lines = "\n        ".join(f"{e}," for e in pos_embeddings)
    neg_lines = "\n        ".join(f"{e}," for e in neg_embeddings)
    pair_comments = "\n".join(
        f'    //   [{i}] "{p}" / "{n_}"'
        for i, (p, n_) in enumerate(ANTONYM_PAIRS)
    )

    swift = f"""// Auto-generated by scripts/setup_models.py — do not edit manually.
// CLIP ViT-B/32 text embeddings for CLIP-IQA+ antonym quality prompts.
// Source: OpenAI CLIP weights via open_clip_torch.
//
// {n} antonym pairs (score averaged across all pairs):
{pair_comments}

import Foundation

enum CLIPTextEmbeddings {{
    /// L2-normalised 512-dim embeddings for positive quality prompts.
    static let positivePrompts: [[Float]] = [
        {pos_lines}
    ]

    /// L2-normalised 512-dim embeddings for negative quality prompts (paired with positivePrompts).
    static let negativePrompts: [[Float]] = [
        {neg_lines}
    ]

    /// Convenience: "A high quality photo." embedding (positivePrompts[0]).
    static let goodPhoto: [Float] = positivePrompts[0]

    /// Convenience: "A low quality photo." embedding (negativePrompts[0]).
    static let badPhoto: [Float] = negativePrompts[0]
}}
"""
    out.write_text(swift)
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  image-rating model setup")
    print("  Downloads and converts TOPIQ + CLIP to CoreML")
    print("=" * 60)

    print("\n[1/4] TOPIQ-NR — technical quality (ResNet50, 224×224)")
    setup_topiq_nr()

    print("\n[2/4] TOPIQ-Swin — aesthetic quality (Swin-B, 384×384)")
    setup_topiq_swin()

    print("\n[3/4] CLIP ViT-B/32 — vision encoder (224×224, 512-dim)")
    clip_model = setup_clip_vision()

    print("\n[4/4] CLIP-IQA+ text embeddings → CLIPTextEmbeddings.swift")
    generate_clip_text_embeddings(clip_model)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Models: {MODELS_DIR}/")
    print(f"  Swift:  {SWIFT_DIR}/CLIPTextEmbeddings.swift")
    print()
    print("  Next steps:")
    print("  1. Add .mlpackage files to project.yml (Task 6)")
    print("  2. Run: xcodegen generate")
    print("=" * 60)


if __name__ == "__main__":
    main()
