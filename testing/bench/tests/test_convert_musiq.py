"""Converter determinism test — same seed, identical weight blob."""
from __future__ import annotations
import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONVERT = ROOT / "scripts" / "convert_musiq.py"
OUT = ROOT / "ImageRater" / "MLModels" / "musiq-ava.mlpackage"
WEIGHTS = OUT / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"

# coremltools distributes pre-built wheels (with native libs libcoremlpython
# and libmilstoragepython) only for Python versions up to 3.12.  On newer
# interpreters pip falls back to the source tarball and the native extensions
# are not built, which makes `MLModel.save` for mlprogram fail with
# "BlobWriter not loaded".  Default to python3.12; allow override via env.
PYTHON = os.environ.get("MUSIQ_PYTHON", "python3.12")


def _weight_hash(path: Path) -> str:
    """SHA-256 of the weight blob — the functionally meaningful artifact.

    The MIL proto (model.mlmodel) has non-deterministic map-field
    serialization ordering in protobuf-python, so its byte hash varies even
    when the model is identical.  weight.bin is fully deterministic given
    fixed seeds, so hashing it gives a stable bit-for-bit determinism check.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_convert_musiq_deterministic():
    subprocess.check_call([PYTHON, str(CONVERT)])
    h1 = _weight_hash(WEIGHTS)
    subprocess.check_call([PYTHON, str(CONVERT)])
    h2 = _weight_hash(WEIGHTS)
    assert h1 == h2, f"Converter non-deterministic: {h1[:8]} != {h2[:8]}"
