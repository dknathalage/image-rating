"""Python reference vs Swift/CoreML parity helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr


def pyiqa_scores(image_dir: Path, limit: int = 100) -> pd.DataFrame:
    """Run pyiqa MUSIQ-AVA on first `limit` jpgs, return [filename, pyiqa_score].

    Reference config matches Swift/CoreML: 2 scales (224, 384), seq_len=193.
    pyiqa's default enables a 3rd scale (original-res patches via
    `max_seq_len_from_original_res=-1`); we disable it so the reference
    matches what CoreML was exported with.
    """
    import pyiqa
    import torch

    metric = pyiqa.create_metric("musiq-ava", device="cpu", as_loss=False)
    metric.eval()
    metric.net.data_preprocess_opts["max_seq_len_from_original_res"] = None
    files = sorted(image_dir.glob("*.jpg"))[:limit]
    rows = []
    with torch.no_grad():
        for f in files:
            s = float(metric(str(f)).detach().cpu().item())
            rows.append({"filename": f.name, "pyiqa_score": s})
    return pd.DataFrame(rows)


def compare(py_df: pd.DataFrame, sw_df: pd.DataFrame) -> dict:
    """Returns spearman, max/mean abs delta on joined frame."""
    m = py_df.merge(sw_df, on="filename", how="inner")
    rho, _ = spearmanr(m["pyiqa_score"], m["coreml_score"])
    delta = (m["pyiqa_score"] - m["coreml_score"]).abs()
    return {
        "n": len(m),
        "spearman": float(rho),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
    }
