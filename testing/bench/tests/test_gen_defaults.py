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
    """Generator reads params.current.json and emits Swift file with numeric values."""
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
    p = payload["params"]
    edges = p["bucket_edges"]

    # Structural sanity
    assert "extension FocalSettings" in text
    assert "import Foundation" in text
    assert text.count("{") == text.count("}"), "braces unbalanced"

    # Version
    assert f'static let generatedVersion: String               = "{payload["version"]}"' in text

    # Numeric constants — check the rendered Double literals
    def _has_double(name: str, value: float) -> bool:
        return f"static let {name}: Double" in text and repr(float(value)) in text

    assert _has_double("generatedWeightTechnical", p["w_tech"])
    assert _has_double("generatedWeightAesthetic", p["w_aes"])
    assert _has_double("generatedWeightClip", p["w_clip"])
    assert _has_double("generatedCullStrictness", p["strictness"])
    assert _has_double("generatedBucketEdge1", edges[0])
    assert _has_double("generatedBucketEdge2", edges[1])
    assert _has_double("generatedBucketEdge3", edges[2])
    assert _has_double("generatedBucketEdge4", edges[3])
    assert _has_double("generatedClipLogitScale", p["clip_logit_scale"])
