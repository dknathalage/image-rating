"""AVA dataset loader: parse labels, compute per-image MOS, stratified sampling, downloader."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import time


AVA_COL_NAMES = (
    ["index", "image_id"]
    + [f"count_{i}" for i in range(1, 11)]
    + ["sem_tag_1", "sem_tag_2", "challenge_id"]
)


def parse_ava_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=AVA_COL_NAMES)
    return df


def compute_mos(df: pd.DataFrame) -> pd.DataFrame:
    count_cols = [f"count_{i}" for i in range(1, 11)]
    counts = df[count_cols].values.astype(float)
    total = counts.sum(axis=1)
    weighted = (counts * np.arange(1, 11)).sum(axis=1)
    df = df.copy()
    df["mos"] = np.where(total > 0, weighted / total, np.nan)
    df = df.dropna(subset=["mos"]).reset_index(drop=True)
    return df


def mos_to_stars(df: pd.DataFrame) -> pd.DataFrame:
    """Quintile rank of MOS → 1..5 stars."""
    df = df.copy()
    ranks = df["mos"].rank(method="first")
    pct = (ranks - 1) / len(df)
    bins = np.array([0.2, 0.4, 0.6, 0.8])
    df["gt_stars"] = np.searchsorted(bins, pct, side="right") + 1
    df["gt_stars"] = df["gt_stars"].clip(1, 5).astype(int)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample n rows with approximately uniform stars. Requires gt_stars column."""
    per_star = max(1, n // 5)
    rng = np.random.default_rng(seed)
    parts = []
    for star in range(1, 6):
        pool = df[df.gt_stars == star]
        k = min(per_star, len(pool))
        if k == 0:
            continue
        parts.append(pool.sample(n=k, random_state=rng.integers(0, 2**32)))
    sampled = pd.concat(parts).reset_index(drop=True)
    if len(sampled) < n:
        extras = df.drop(sampled.index, errors="ignore").sample(
            n=n - len(sampled), random_state=seed
        )
        sampled = pd.concat([sampled, extras]).reset_index(drop=True)
    return sampled.head(n).reset_index(drop=True)


def _dp_challenge_url(image_id: int) -> str:
    return f"https://images.dpchallenge.com/images_challenge/0-999/{image_id}.jpg"


def download_images(
    df: pd.DataFrame,
    out_dir: Path,
    sleep: float = 0.2,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download images to out_dir/<image_id>.jpg. Returns df with local_path column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    headers = {"User-Agent": "focal-bench/1.0"}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="download"):
        image_id = int(row["image_id"])
        dest = out_dir / f"{image_id}.jpg"
        if not dest.exists():
            try:
                resp = requests.get(_dp_challenge_url(image_id), timeout=timeout, headers=headers)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    dest.write_bytes(resp.content)
                    time.sleep(sleep)
                else:
                    continue
            except Exception:
                continue
        if dest.exists():
            rows.append({**row.to_dict(), "local_path": str(dest)})
    return pd.DataFrame(rows)
