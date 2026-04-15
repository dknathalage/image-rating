from pathlib import Path
import json
from bench.leaderboard import regenerate_leaderboard


def test_leaderboard_sorted(tmp_path):
    results_dir = tmp_path / "results"
    for ver, spearman in [("v0.1.0@aaa1111", 0.55), ("v0.2.0@bbb2222", 0.67), ("v0.3.0@ccc3333", 0.60)]:
        d = results_dir / ver
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"val": {"spearman": spearman, "mae": 0.8, "off_by_one": 0.9, "exact_match": 0.5}}))
        (d / "params.json").write_text(json.dumps({"ensemble": ["tech","aes","clip"], "notes": f"ver {ver}", "date": "2026-04-15"}))
    md_path = tmp_path / "LEADERBOARD.md"
    regenerate_leaderboard(results_dir, md_path)
    text = md_path.read_text()
    v2_pos = text.index("v0.2.0")
    v3_pos = text.index("v0.3.0")
    v1_pos = text.index("v0.1.0")
    assert v2_pos < v3_pos < v1_pos


def test_leaderboard_skips_incomplete(tmp_path):
    results_dir = tmp_path / "results"
    # Missing params.json — should skip silently.
    d = results_dir / "v0.1.0@aaa"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"val": {"spearman": 0.5, "mae": 0.8, "off_by_one": 0.9}}))
    md_path = tmp_path / "LEADERBOARD.md"
    regenerate_leaderboard(results_dir, md_path)
    text = md_path.read_text()
    assert "v0.1.0" not in text


def test_leaderboard_empty_results(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    md_path = tmp_path / "LEADERBOARD.md"
    regenerate_leaderboard(results_dir, md_path)
    text = md_path.read_text()
    assert "# Rating Ensemble Leaderboard" in text
    # Table header present even with no rows.
    assert "| Version |" in text
