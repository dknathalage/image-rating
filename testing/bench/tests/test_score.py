import json
import subprocess
from pathlib import Path
import pandas as pd
import pytest
from bench.score import content_hash_dir, load_scores_json, run_scorer

def test_content_hash_dir_stable(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"A")
    (tmp_path / "b.jpg").write_bytes(b"B")
    h1 = content_hash_dir(tmp_path)
    h2 = content_hash_dir(tmp_path)
    assert h1 == h2
    (tmp_path / "c.jpg").write_bytes(b"C")
    h3 = content_hash_dir(tmp_path)
    assert h1 != h3

def test_load_scores_json(tmp_path):
    blob = {
        "generatedAt": "2026-01-01T00:00:00Z",
        "modelVersion": "v1",
        "images": [
            {"filename": "a.jpg", "topiqTechnical": 0.5, "topiqAesthetic": 0.7, "clipEmbedding": [0.1, 0.2]},
            {"filename": "b.jpg", "topiqTechnical": 0.3, "topiqAesthetic": 0.6, "clipEmbedding": [0.3, 0.4]},
        ],
    }
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(blob))
    df = load_scores_json(p)
    assert list(df.columns) >= ["filename", "topiqTechnical", "topiqAesthetic", "clipEmbedding"]
    assert len(df) == 2

def test_run_scorer_invokes_binary(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        out_path = Path(cmd[2])
        out_path.write_text(json.dumps({"generatedAt":"","modelVersion":"","images":[]}))
        class R: returncode = 0; stdout = b""; stderr = b""
        return R()
    monkeypatch.setattr(subprocess, "run", fake_run)
    run_scorer(Path("/fake/bin"), tmp_path, tmp_path / "out.json")
    assert len(calls) == 1
    assert calls[0][0] == "/fake/bin"
