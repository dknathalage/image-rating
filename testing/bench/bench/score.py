"""Run Swift FocalScorer CLI against an image directory and cache JSON."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import subprocess
import pandas as pd


def content_hash_dir(path: Path) -> str:
    """Deterministic hash of all file contents in path (sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_scorer(scorer_bin: Path, image_dir: Path, output_json: Path) -> None:
    """Invoke FocalScorer CLI. Raises RuntimeError on failure."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(scorer_bin), str(image_dir), str(output_json)],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FocalScorer failed: {result.stderr.decode(errors='ignore')}")


def load_scores_json(path: Path) -> pd.DataFrame:
    blob = json.loads(path.read_text())
    return pd.DataFrame(blob["images"])


def score_with_cache(
    scorer_bin: Path,
    image_dir: Path,
    cache_dir: Path,
) -> pd.DataFrame:
    """Score image_dir via FocalScorer, caching JSON by content hash."""
    digest = content_hash_dir(image_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"scores_{digest}.json"
    if not cached.exists():
        run_scorer(scorer_bin, image_dir, cached)
    return load_scores_json(cached)
