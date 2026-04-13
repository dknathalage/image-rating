#!/usr/bin/env python3
"""
Convert NIMA aesthetic and technical quality models to CoreML .mlpackage.

Outputs:
  models/nima-aesthetic.mlpackage  -- image -> score (1-10, AVA aesthetic quality)
  models/nima-technical.mlpackage  -- image -> score (1-10, AVA technical quality)

Weights sourced from:
  https://github.com/idealo/image-quality-assessment
  (MobileNet aesthetic: weights_mobilenet_aesthetic_0.07.hdf5)
  (MobileNet technical: weights_mobilenet_technical_0.11.hdf5)

Both models accept 224x224 RGB images and output a scalar mean opinion score [1-10].
"""

import urllib.request
import sys
from pathlib import Path

import numpy as np
import coremltools as ct

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

WEIGHTS_DIR = Path(__file__).parent / "nima_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

AESTHETIC_WEIGHTS_URL = (
    "https://github.com/idealo/image-quality-assessment/raw/master/"
    "models/MobileNet/aesthetic/weights_mobilenet_aesthetic_0.07.hdf5"
)
TECHNICAL_WEIGHTS_URL = (
    "https://github.com/idealo/image-quality-assessment/raw/master/"
    "models/MobileNet/technical/weights_mobilenet_technical_0.11.hdf5"
)

AESTHETIC_WEIGHTS_PATH = WEIGHTS_DIR / "weights_mobilenet_aesthetic_0.07.hdf5"
TECHNICAL_WEIGHTS_PATH = WEIGHTS_DIR / "weights_mobilenet_technical_0.11.hdf5"


def download_weights():
    for url, path, label in [
        (AESTHETIC_WEIGHTS_URL, AESTHETIC_WEIGHTS_PATH, "aesthetic"),
        (TECHNICAL_WEIGHTS_URL, TECHNICAL_WEIGHTS_PATH, "technical"),
    ]:
        if path.exists():
            print(f"  {label} weights already present: {path.name}")
            continue
        print(f"  Downloading NIMA {label} weights…")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  Downloaded: {path.name}")
        except Exception as e:
            print(f"  ERROR: could not download {label} weights: {e}")
            print(f"  Please manually download from:")
            print(f"    {url}")
            print(f"  and place at: {path}")
            sys.exit(1)


def build_nima_model(weights_path: Path):
    """
    Build NIMA MobileNet model from Keras weights.

    Architecture mirrors idealo/image-quality-assessment:
      MobileNet (no top, avg pooling) -> Dropout(0.75) -> Dense(10, softmax)
    Input: (224, 224, 3) float32 [0-255] RGB
    Output: scalar mean opinion score [1-10]
    """
    import tensorflow as tf
    import tf_keras as keras

    base = keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="avg",
        weights=None,
    )
    x = keras.layers.Dropout(0.75)(base.output)
    x = keras.layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=base.input, outputs=x)
    model.load_weights(str(weights_path))
    model.trainable = False

    # Wrap: accept [0-255] RGB, output mean opinion score [1-10]
    inp = keras.Input(shape=(224, 224, 3), name="image")
    # Normalise to MobileNet's expected range: [0, 1] then (x - 0.5) / 0.5
    x_norm = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inp)
    dist = model(x_norm)  # (batch, 10) softmax distribution
    bins = tf.constant(
        [[float(i) for i in range(1, 11)]], dtype=tf.float32
    )  # (1, 10)
    score = tf.reduce_sum(dist * bins, axis=1, keepdims=True)  # (batch, 1)
    wrapped = keras.Model(inputs=inp, outputs=score, name="nima_scorer")
    return wrapped


def convert_nima(keras_model, output_name: str, save_path: Path):
    import tempfile, os
    import tensorflow as tf

    print(f"  Converting to CoreML: {save_path.name}")

    # Save as TF SavedModel so coremltools can load it
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = os.path.join(tmpdir, "saved_model")
        tf.saved_model.save(keras_model, saved_model_path)

        mlmodel = ct.convert(
            saved_model_path,
            source="tensorflow",
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=(1, 224, 224, 3),
                    color_layout=ct.colorlayout.RGB,
                )
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram",
        )

    # Rename TF's auto-named output ('Identity') to 'score' for consistency with app code.
    try:
        spec = mlmodel.get_spec()
        ct.utils.rename_feature(spec, "Identity", "score", rename_outputs=True)
        mlmodel = ct.models.MLModel(spec)
        print(f"  Renamed output: Identity → score")
    except Exception as e:
        print(f"  Warning: could not rename output feature: {e}")

    mlmodel.save(str(save_path))
    print(f"  Saved: {save_path}")


def verify_model(save_path: Path):
    """Quick sanity check: score a black 224x224 image, expect 1-10 range."""
    import coremltools as ct
    import numpy as np
    from PIL import Image

    model = ct.models.MLModel(str(save_path))
    black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = model.predict({"image": black})
    # Output key is auto-named 'Identity' by TF SavedModel conversion
    score_key = list(result.keys())[0]
    score = float(np.array(result[score_key]).flatten()[0])
    assert 1.0 <= score <= 10.0, f"Score out of expected range: {score}"
    print(f"  Sanity check passed: black image → {score:.3f}")


def main():
    print("=== Downloading NIMA weights ===")
    download_weights()

    for label, weights_path, out_name in [
        ("aesthetic", AESTHETIC_WEIGHTS_PATH, "nima-aesthetic"),
        ("technical", TECHNICAL_WEIGHTS_PATH, "nima-technical"),
    ]:
        out_path = MODELS_DIR / f"{out_name}.mlpackage"
        if out_path.exists():
            print(f"\n=== Skipping {out_name} (already exists) ===")
            continue

        print(f"\n=== Building NIMA {label} model ===")
        keras_model = build_nima_model(weights_path)

        print(f"\n=== Converting NIMA {label} to CoreML ===")
        convert_nima(keras_model, "score", out_path)

        print(f"\n=== Verifying {out_name} ===")
        verify_model(out_path)

    print("\n=== Done ===")
    print(f"Models written to: {MODELS_DIR.resolve()}")
    print("Copy .mlpackage files to Application Support/ImageRater/models/")
    print("renaming each to <name>-local.mlpackage for local development.")


if __name__ == "__main__":
    main()
