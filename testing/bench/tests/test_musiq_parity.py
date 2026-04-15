"""CoreML MUSIQ vs pyiqa reference — merge gate."""
from __future__ import annotations
import glob
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from bench.parity import compare, pyiqa_scores

BENCH = Path(__file__).resolve().parents[1]
IMAGES = BENCH / "data" / "ava" / "images"
CACHE = BENCH / ".cache"
TMP_OUT = CACHE / "parity_coreml.json"
SAMPLE = 100


def _locate_scorer_bin() -> str:
    home = os.path.expanduser("~")
    cand = glob.glob(
        f"{home}/Library/Developer/Xcode/DerivedData/**/Build/Products/Release/FocalScorer",
        recursive=True,
    )
    cand = [c for c in cand if os.access(c, os.X_OK)]
    cand.sort(key=os.path.getmtime, reverse=True)
    if not cand:
        pytest.skip("FocalScorer Release binary not built")
    return cand[0]


def test_compare_returns_expected_keys():
    py = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "pyiqa_score": [5.2, 6.1]})
    sw = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "coreml_score": [5.25, 6.05]})
    result = compare(py, sw)
    assert set(result.keys()) == {"n", "spearman", "max_abs_delta", "mean_abs_delta"}
    assert result["n"] == 2
    assert result["max_abs_delta"] < 0.1


@pytest.mark.skipif(not IMAGES.exists(), reason="AVA sample missing")
def test_coreml_parity_ava_100():
    bin_ = _locate_scorer_bin()
    tmp_dir = CACHE / "parity_sample"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGES.glob("*.jpg"))[:SAMPLE]
    if len(files) < SAMPLE:
        pytest.skip(f"need {SAMPLE} AVA images, have {len(files)}")
    for f in files:
        dst = tmp_dir / f.name
        if not dst.exists():
            dst.symlink_to(f)
    TMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([bin_, str(tmp_dir), str(TMP_OUT)])

    blob = json.loads(TMP_OUT.read_text())
    sw_df = pd.DataFrame(blob["images"]).rename(columns={"musiqAesthetic": "coreml_score"})

    py_df = pyiqa_scores(tmp_dir, limit=SAMPLE)
    report = compare(py_df, sw_df)

    print(f"\nParity report: {report}")
    assert report["n"] >= SAMPLE - 5, f"joined rows too few: {report}"
    assert report["spearman"] >= 0.97, f"Spearman {report['spearman']:.4f} < 0.97"
    assert report["max_abs_delta"] <= 0.10, f"max |Δ| {report['max_abs_delta']:.4f} > 0.10"
    assert report["mean_abs_delta"] <= 0.03, f"mean |Δ| {report['mean_abs_delta']:.4f} > 0.03"
