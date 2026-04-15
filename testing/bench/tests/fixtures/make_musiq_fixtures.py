"""Generate deterministic reference tensors for Swift preprocessor parity tests.

Emits binary float32 little-endian files (and .txt shape/label sidecars) into
ImageRaterTests/Fixtures/musiq_reference/. Committed to repo so Swift tests run
without torch."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from pyiqa.data.multiscale_trans_util import (
    extract_image_patches,
    get_hashed_spatial_pos_emb_index,
    resize_preserve_aspect_ratio,
    _extract_patches_and_positions_from_image,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "ImageRaterTests" / "Fixtures" / "musiq_reference"

PATCH_SIZE = 32
HASH_GRID = 10
MUSIQ_SCALES = [224, 384]
NUM_CHANNELS = 3
ORIG_RES_MAX_SEQ = 512


def write_tensor(path: Path, t: torch.Tensor) -> None:
    """Write float32 raw + shape sidecar."""
    arr = t.contiguous().detach().cpu().float().numpy()
    path.write_bytes(arr.tobytes())
    (path.with_suffix(".shape")).write_text(",".join(str(d) for d in arr.shape))


def make_image(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """Deterministic [1, 3, h, w] float tensor in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixture 1: resize 1200x800 longer=224 -> (149, 224)
    img_1200x800 = make_image(1200, 800, seed=1)
    write_tensor(OUT_DIR / "img_1200x800.f32", img_1200x800)
    r224, rh, rw = resize_preserve_aspect_ratio(img_1200x800, 1200, 800, 224)
    write_tensor(OUT_DIR / "resize_1200x800_224.f32", r224)
    (OUT_DIR / "resize_1200x800_224.dims").write_text(f"{rh},{rw}")

    # Fixture 2: resize 800x1200 longer=384 -> (256, 384)
    img_800x1200 = make_image(800, 1200, seed=2)
    write_tensor(OUT_DIR / "img_800x1200.f32", img_800x1200)
    r384, rh, rw = resize_preserve_aspect_ratio(img_800x1200, 800, 1200, 384)
    write_tensor(OUT_DIR / "resize_800x1200_384.f32", r384)
    (OUT_DIR / "resize_800x1200_384.dims").write_text(f"{rh},{rw}")

    # Fixture 3: unfold 3x64x64 ranked-values input -> 4 patches of 3072.
    g = torch.Generator().manual_seed(42)
    img_64 = torch.rand(1, NUM_CHANNELS, 64, 64, generator=g)
    write_tensor(OUT_DIR / "img_64x64.f32", img_64)
    patches = extract_image_patches(img_64, PATCH_SIZE, PATCH_SIZE).transpose(1, 2)  # [1, 4, 3072]
    write_tensor(OUT_DIR / "unfold_64x64.f32", patches)

    # Fixture 4: hash spatial positions for (count_h=7, count_w=5, grid=10)
    hsp_7x5 = get_hashed_spatial_pos_emb_index(HASH_GRID, 7, 5)  # [1, 35]
    write_tensor(OUT_DIR / "hsp_7x5.f32", hsp_7x5)

    # Fixture 5: full multiscale patch tensor for 500x400 input, scales [224, 384]
    img_multi = make_image(500, 400, seed=3)
    write_tensor(OUT_DIR / "img_500x400.f32", img_multi)
    outs = []
    for scale_id, longer in enumerate(MUSIQ_SCALES):
        resized, rh, rw = resize_preserve_aspect_ratio(img_multi, 500, 400, longer)
        max_seq_len = int(np.ceil(longer / PATCH_SIZE) ** 2)
        out = _extract_patches_and_positions_from_image(
            resized, PATCH_SIZE, PATCH_SIZE, HASH_GRID, 1, rh, rw, NUM_CHANNELS, scale_id, max_seq_len,
        )
        outs.append(out)
    outs.append(_extract_patches_and_positions_from_image(
        img_multi, PATCH_SIZE, PATCH_SIZE, HASH_GRID, 1,
        500, 400, NUM_CHANNELS, 2, ORIG_RES_MAX_SEQ,
    ))
    full = torch.cat(outs, dim=-1).transpose(1, 2)  # [1, 705, 3075]
    write_tensor(OUT_DIR / "patch_tensor_500x400.f32", full)

    # MUSIQ normalization convention: (pix - 0.5) * 2. Emit both variants
    # so Swift can test either; preprocessor uses normalized input.
    write_tensor(OUT_DIR / "img_500x400_normalized.f32", (img_multi - 0.5) * 2)

    print(f"Wrote fixtures to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
