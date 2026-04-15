"""Tests for scripts/gen_defaults.py — Swift constants bridge."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "gen_defaults.py"
PARAMS = REPO_ROOT / "testing" / "bench" / "params.current.json"
OUT = REPO_ROOT / "ImageRater" / "App" / "FocalSettings+Generated.swift"


def test_gen_defaults_emits_expected_numeric_values():
    """Generator reads params.current.json and emits Swift file with numeric thresholds."""
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert PARAMS.exists(), f"missing {PARAMS}"

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, check=True,
    )
    assert "wrote" in result.stdout

    assert OUT.exists(), f"generator did not write {OUT}"
    text = OUT.read_text()

    payload = json.loads(PARAMS.read_text())
    thresholds = payload["thresholds"]

    assert "extension FocalSettings" in text
    assert "import Foundation" in text
    assert text.count("{") == text.count("}"), "braces unbalanced"

    assert f'static let generatedVersion: String          = "{payload["version"]}"' in text
    assert f'static let generatedModel: String            = "{payload["model"]}"' in text

    def _has_float(name: str, value: float) -> bool:
        return f"static let {name}: Float" in text and repr(float(value)) in text

    assert _has_float("generatedMUSIQThreshold1", thresholds[0])
    assert _has_float("generatedMUSIQThreshold2", thresholds[1])
    assert _has_float("generatedMUSIQThreshold3", thresholds[2])
    assert _has_float("generatedMUSIQThreshold4", thresholds[3])
