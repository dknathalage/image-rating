#!/usr/bin/env python3
"""Generate Swift constants from testing/bench/params.current.json.

Run manually or via xcodegen pre-build phase. Overwrites
ImageRater/App/FocalSettings+Generated.swift.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARAMS_PATH = ROOT / "testing" / "bench" / "params.current.json"
OUT_PATH    = ROOT / "ImageRater" / "App" / "FocalSettings+Generated.swift"

EXPECTED_VERSION = "v0.4.0"
EXPECTED_MODEL   = "musiq-ava"


TEMPLATE = """// Auto-generated from testing/bench/params.current.json. DO NOT EDIT.
// Regenerate via `python3 scripts/gen_defaults.py`.
import Foundation

extension FocalSettings {{
    static let generatedVersion: String          = "{version}"
    static let generatedModel: String            = "{model}"
    static let generatedMUSIQThreshold1: Float   = {t1}
    static let generatedMUSIQThreshold2: Float   = {t2}
    static let generatedMUSIQThreshold3: Float   = {t3}
    static let generatedMUSIQThreshold4: Float   = {t4}
}}
"""


def _fmt(value: float) -> str:
    """Render number as Swift Float literal (always contains decimal point)."""
    s = repr(float(value))
    return s


def main() -> None:
    payload = json.loads(PARAMS_PATH.read_text())
    version = payload.get("version")
    model = payload.get("model")
    thresholds = payload.get("thresholds")

    if version != EXPECTED_VERSION:
        sys.exit(f"version mismatch: expected {EXPECTED_VERSION!r}, got {version!r}")
    if model != EXPECTED_MODEL:
        sys.exit(f"model mismatch: expected {EXPECTED_MODEL!r}, got {model!r}")
    if not isinstance(thresholds, list) or len(thresholds) != 4:
        sys.exit(f"thresholds must be list of 4 numbers, got {thresholds!r}")
    for t in thresholds:
        if not isinstance(t, (int, float)):
            sys.exit(f"threshold not numeric: {t!r}")
    if thresholds != sorted(thresholds) or len(set(thresholds)) != 4:
        sys.exit(f"thresholds must be strictly increasing, got {thresholds}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(TEMPLATE.format(
        version=version,
        model=model,
        t1=_fmt(thresholds[0]),
        t2=_fmt(thresholds[1]),
        t3=_fmt(thresholds[2]),
        t4=_fmt(thresholds[3]),
    ))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
