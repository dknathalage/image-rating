"""Tests for run.py orchestration CLI."""
from __future__ import annotations
import json
import stat
import subprocess
import sys
from pathlib import Path
import pytest

import run


def test_locate_scorer_bin_respects_env_var(tmp_path, monkeypatch):
    fake = tmp_path / "FocalScorer"
    fake.write_text("#!/bin/sh\necho fake\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("FOCAL_SCORER_BIN", str(fake))
    found = run.locate_scorer_bin()
    assert found == fake


def test_locate_scorer_bin_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FOCAL_SCORER_BIN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(FileNotFoundError) as exc_info:
        run.locate_scorer_bin()
    msg = str(exc_info.value)
    assert "xcodebuild" in msg or "FocalScorer" in msg


def test_git_sha_short_returns_nonempty_in_repo():
    sha = run.git_sha_short()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_git_sha_short_returns_nogit_when_git_missing(monkeypatch):
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(subprocess, "check_output", raise_fnf)
    assert run.git_sha_short() == "nogit"


def test_load_save_params_roundtrip(tmp_path):
    payload = {
        "version": "v0.4.0",
        "date": "2026-04-15",
        "model": "musiq-ava",
        "notes": "test",
        "thresholds": [4.5, 5.2, 5.6, 6.1],
    }
    p = tmp_path / "params.json"
    run.save_params(p, payload)
    loaded = run.load_params(p)
    assert loaded == payload


def test_params_current_json_is_v040():
    payload = json.loads((Path(__file__).parents[1] / "params.current.json").read_text())
    assert payload["version"] == "v0.4.0"
    assert payload["model"] == "musiq-ava"
    assert len(payload["thresholds"]) == 4
    assert all(isinstance(x, (int, float)) for x in payload["thresholds"])
    assert "params" not in payload
    assert "bucket_edges" not in payload


def test_argparse_drops_ablate_subcommand():
    import argparse  # noqa: F401
    p = run._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["ablate"])


def test_main_dispatches_leaderboard(monkeypatch):
    called = {}

    def fake_leaderboard(args):
        called["hit"] = True

    monkeypatch.setattr(run, "cmd_leaderboard", fake_leaderboard)
    monkeypatch.setattr(sys, "argv", ["run.py", "leaderboard"])
    run.main()
    assert called.get("hit") is True


@pytest.mark.parametrize("subcmd,extra", [
    ("download", []),
    ("score", []),
    ("eval", []),
    ("optimize", []),
    ("leaderboard", []),
])
def test_main_argparse_accepts_all_subcommands(monkeypatch, subcmd, extra):
    """Verify each subcommand parses without error; handlers are no-oped."""
    for name in ("cmd_download", "cmd_score", "cmd_eval", "cmd_optimize", "cmd_leaderboard"):
        monkeypatch.setattr(run, name, lambda args: None)
    monkeypatch.setattr(sys, "argv", ["run.py", subcmd, *extra])
    run.main()
