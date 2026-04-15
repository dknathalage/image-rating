"""Converter determinism test — same seed, identical weight blob."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONVERT = ROOT / "scripts" / "convert_musiq.py"
OUT = ROOT / "ImageRater" / "MLModels" / "musiq-ava.mlpackage"
WEIGHTS = OUT / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"

# python3.12 has pre-built coremltools wheels with native libs (libmilstoragepython.so).
# python3 on this system maps to 3.14 which installs from source without those libs.
PYTHON = "python3.12"


def _weight_hash(path: Path) -> str:
    """SHA-256 of the weight blob — this is the functionally meaningful artifact.

    The MIL proto (model.mlmodel) has non-deterministic map-field serialization
    ordering in protobuf-python, so its byte hash varies even when the model is
    identical.  The weight.bin is fully deterministic given fixed seeds.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_convert_musiq_deterministic(tmp_path):
    subprocess.check_call([PYTHON, str(CONVERT)])
    h1 = _weight_hash(WEIGHTS)
    subprocess.check_call([PYTHON, str(CONVERT)])
    h2 = _weight_hash(WEIGHTS)
    assert h1 == h2, f"Converter non-deterministic: {h1[:8]} != {h2[:8]}"
