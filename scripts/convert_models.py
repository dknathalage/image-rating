#!/usr/bin/env python3
"""
Convert CLIP ViT-B/32 and LAION Aesthetic Predictor to CoreML .mlpackage.

Outputs:
  models/clip.mlpackage       -- image -> score (0-1, CLIP similarity to "high quality photo")
  models/aesthetic.mlpackage  -- image -> score (1-10, LAION aesthetic predictor)

Both models accept 224x224 RGB pixel buffers (ct.colorlayout.RGB).
Note: coremltools >=8.0 removed ARGB from ct.colorlayout.  The Swift caller
should convert kCVPixelFormatType_32BGRA (native macOS format) to RGB order
before passing to the model (e.g. via vImage channel swap or CIFilter).
The model normalises [0,255] float RGB values internally.
"""

import os
import sys
import urllib.request
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from transformers import CLIPModel, CLIPTokenizer

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# CLIP normalisation constants (standard ImageNet-CLIP values)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# LAION aesthetic predictor weights (ViT-L/14 version)
AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_WEIGHTS_PATH = Path(__file__).parent / "aesthetic_weights.pth"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Normalise [0-255] float RGB tensor to CLIP input space."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(CLIP_STD ).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / 255.0 - self.mean) / self.std


def download_aesthetic_weights():
    if not AESTHETIC_WEIGHTS_PATH.exists():
        print("Downloading LAION aesthetic predictor weights (~50 MB)…")
        urllib.request.urlretrieve(AESTHETIC_WEIGHTS_URL, AESTHETIC_WEIGHTS_PATH)
    return AESTHETIC_WEIGHTS_PATH


def load_aesthetic_mlp(weight_path: Path) -> nn.Module:
    """Build the 5-layer MLP used by the improved aesthetic predictor."""
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    # Architecture: Linear(768,1024) ReLU Dropout Linear(1024,128) ReLU
    #               Dropout Linear(128,64) ReLU Dropout Linear(64,16)
    #               ReLU Linear(16,1)
    mlp = nn.Sequential(
        nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 16),    nn.ReLU(),
        nn.Linear(16, 1),
    )
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp


# ---------------------------------------------------------------------------
# Model 1: CLIP scorer (ViT-B/32)
# ---------------------------------------------------------------------------

class CLIPImageScorer(nn.Module):
    """
    Input : (1, 3, 224, 224) float32, values 0-255 RGB
    Output: (1,) float32, cosine-similarity to 'high quality photograph' scaled to [0,1]
    """
    def __init__(self, vision_model, projection, text_feat: torch.Tensor):
        super().__init__()
        self.norm = Normalizer()
        self.vision_model = vision_model
        self.projection   = projection
        self.register_buffer("text_feat", text_feat)  # (1, 512), L2-normalised

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.norm(image)
        # return_dict=False → tuple: (last_hidden_state, pooler_output)
        outputs = self.vision_model(pixel_values=x, return_dict=False)
        pooled = outputs[1]                                       # (1, 512)
        img_feat = self.projection(pooled)                        # (1, 512)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = torch.mm(img_feat, self.text_feat.T).squeeze(0)    # scalar
        score = ((sim + 1.0) / 2.0).reshape(1)                   # [0, 1]
        return score


def build_clip_scorer(clip_model, tokenizer) -> CLIPImageScorer:
    with torch.no_grad():
        tokens = tokenizer(
            ["a high quality, aesthetically pleasing photograph"],
            return_tensors="pt", padding=True
        )
        out = clip_model.get_text_features(**tokens)
        # transformers >=5.x returns BaseModelOutputWithPooling; pooler_output
        # holds the projected text embedding.  Older versions return a Tensor.
        text_feat = out.pooler_output if hasattr(out, "pooler_output") else out
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return CLIPImageScorer(
        clip_model.vision_model,
        clip_model.visual_projection,
        text_feat.detach(),
    )


# ---------------------------------------------------------------------------
# Model 2: Aesthetic scorer (ViT-L/14 + LAION MLP)
# ---------------------------------------------------------------------------

class AestheticScorer(nn.Module):
    """
    Input : (1, 3, 224, 224) float32, values 0-255 RGB
    Output: (1,) float32, aesthetic score ~[1, 10]
    """
    def __init__(self, vision_model, projection, mlp: nn.Module):
        super().__init__()
        self.norm       = Normalizer()
        self.vision_model = vision_model
        self.projection   = projection
        self.mlp          = mlp

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.norm(image)
        outputs  = self.vision_model(pixel_values=x, return_dict=False)
        pooled   = outputs[1]
        img_feat = self.projection(pooled)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score    = self.mlp(img_feat).reshape(1)
        return score


# ---------------------------------------------------------------------------
# CoreML conversion helper
# ---------------------------------------------------------------------------

def to_coreml(traced_model, output_name: str, save_path: Path):
    """Convert a traced PyTorch model to a CoreML .mlpackage.

    color_layout=RGB means CoreML expects a 3-channel RGB CVPixelBuffer.
    The Swift caller should convert kCVPixelFormatType_32BGRA (native macOS
    camera/CIImage format) to kCVPixelFormatType_24RGB before passing in,
    or use a CIFilter / vImage to swap channels.  The model itself handles
    [0,255] normalisation internally via the Normalizer layer.

    Note: coremltools >=8.0 removed ARGB from ct.colorlayout; use RGB.
    """
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, 224, 224),              # (batch, channels, H, W)
            color_layout=ct.colorlayout.RGB,     # coremltools 8+: no ARGB support
        )],
        outputs=[ct.TensorType(name=output_name)],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
    )
    mlmodel.save(str(save_path))
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Loading CLIP ViT-B/32 (for clip model) ===")
    clip_b32   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_b32.eval()

    # Enable torchscript mode to avoid return_dict issues during tracing
    clip_b32.vision_model.config.torchscript = True

    # --- CLIP scorer ---
    print("\n=== Building CLIP scorer ===")
    clip_scorer = build_clip_scorer(clip_b32, tokenizer)
    clip_scorer.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced_clip = torch.jit.trace(clip_scorer, dummy, strict=False)
        # Sanity check: output must be a scalar in [0,1]
        out = traced_clip(dummy)
        assert out.shape == (1,), f"clip scorer bad shape: {out.shape}"
        print(f"  CLIP scorer test output: {out.item():.4f}  (expected ~0.5 for zeros)")

    print("\n=== Converting CLIP scorer to CoreML ===")
    to_coreml(traced_clip, "score", MODELS_DIR / "clip.mlpackage")
    del clip_scorer, traced_clip, clip_b32

    # --- Aesthetic scorer ---
    print("\n=== Loading CLIP ViT-L/14 (for aesthetic model) ===")
    clip_l14 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_l14.eval()

    # Enable torchscript mode to avoid return_dict issues during tracing
    clip_l14.vision_model.config.torchscript = True

    print("=== Loading LAION aesthetic predictor weights ===")
    weights_path = download_aesthetic_weights()
    mlp = load_aesthetic_mlp(weights_path)

    aesthetic_scorer = AestheticScorer(
        clip_l14.vision_model,
        clip_l14.visual_projection,
        mlp,
    )
    aesthetic_scorer.eval()
    with torch.no_grad():
        traced_aes = torch.jit.trace(aesthetic_scorer, dummy, strict=False)
        out = traced_aes(dummy)
        assert out.shape == (1,), f"aesthetic scorer bad shape: {out.shape}"
        print(f"  Aesthetic scorer test output: {out.item():.4f}  (expected ~4-7 for real photos)")

    print("\n=== Converting aesthetic scorer to CoreML ===")
    to_coreml(traced_aes, "score", MODELS_DIR / "aesthetic.mlpackage")

    print("\n=== Done ===")
    print(f"Models written to: {MODELS_DIR.resolve()}")
    print("Next: run scripts/release_models.sh to upload to GitHub Releases.")


if __name__ == "__main__":
    main()
