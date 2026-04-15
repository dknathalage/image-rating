#!/usr/bin/env python3
"""Generate Swift constants from testing/bench/params.current.json.

Run manually or via xcodegen pre-build phase. Overwrites
ImageRater/App/FocalSettings+Generated.swift.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARAMS_PATH = ROOT / "testing" / "bench" / "params.current.json"
OUT_PATH    = ROOT / "ImageRater" / "App" / "FocalSettings+Generated.swift"


TEMPLATE = """// Auto-generated from testing/bench/params.current.json. DO NOT EDIT.
// Regenerate via `python3 scripts/gen_defaults.py`.
import Foundation

extension FocalSettings {{
    static let generatedVersion: String               = "{version}"
    static let generatedWeightTechnical: Double       = {wTech}
    static let generatedWeightAesthetic: Double       = {wAes}
    static let generatedWeightClip: Double            = {wClip}
    static let generatedCullStrictness: Double        = {strictness}
    static let generatedBucketEdge1: Double           = {e1}
    static let generatedBucketEdge2: Double           = {e2}
    static let generatedBucketEdge3: Double           = {e3}
    static let generatedBucketEdge4: Double           = {e4}
    static let generatedClipLogitScale: Double        = {clipLogit}
}}
"""


def _fmt(value: float) -> str:
    """Render a number as a Swift Double literal (always contains a decimal point)."""
    s = repr(float(value))
    return s


def main() -> None:
    payload = json.loads(PARAMS_PATH.read_text())
    p = payload["params"]
    e = p["bucket_edges"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(TEMPLATE.format(
        version=payload["version"],
        wTech=_fmt(p["w_tech"]),
        wAes=_fmt(p["w_aes"]),
        wClip=_fmt(p["w_clip"]),
        strictness=_fmt(p["strictness"]),
        e1=_fmt(e[0]), e2=_fmt(e[1]), e3=_fmt(e[2]), e4=_fmt(e[3]),
        clipLogit=_fmt(p["clip_logit_scale"]),
    ))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
