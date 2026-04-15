"""MUSIQ-AVA aesthetic scorer via pyiqa (PyTorch).

Drop-in replacement for FocalScorer's TOPIQ-aesthetic output, per the plan's
aesthetic model swap path. Scores an image directory; caches results keyed
by content hash so tuning runs don't re-score.

Uses torch.no_grad + MPS cache clear per image — without these, pyiqa
accumulates autograd graph / MTL buffers and bogs down after ~100 images.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .score import content_hash_dir


def score_musiq_with_cache(
    image_dir: Path,
    cache_dir: Path,
    model: str = "musiq-ava",
) -> pd.DataFrame:
    """Score all .jpg in image_dir with pyiqa MUSIQ-AVA; cache by content hash.

    Returns DataFrame with columns [filename, musiqAesthetic].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = content_hash_dir(image_dir)
    out_path = cache_dir / f"musiq_{model.replace('-','_')}_{h}.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    # Lazy imports — pyiqa / torch are heavy and optional.
    import torch
    import pyiqa
    from tqdm import tqdm

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    metric = pyiqa.create_metric(model, device=device, as_loss=False)
    metric.eval()

    files = sorted(image_dir.glob("*.jpg"))
    rows = []
    with torch.no_grad():
        for f in tqdm(files, ncols=70, mininterval=1.0, desc=model):
            try:
                s = float(metric(str(f)).detach().cpu().item())
            except Exception:
                s = float("nan")
            rows.append({"filename": f.name, "musiqAesthetic": s})
            if device == "mps":
                torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df
