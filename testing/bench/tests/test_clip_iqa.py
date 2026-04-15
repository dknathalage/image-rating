import json
import numpy as np
import pytest
from pathlib import Path
from bench.clip_iqa import clip_iqa_score, load_prompt_embeddings


def test_clip_iqa_symmetric_prompts_returns_half():
    emb = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    prompts = {
        "positive": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        "negative": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }
    s = clip_iqa_score(emb, prompts, logit_scale=100.0)
    assert 0.49 < s < 0.51


def test_clip_iqa_positive_favored():
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    prompts = {
        "positive": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "negative": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }
    s = clip_iqa_score(emb, prompts, logit_scale=100.0)
    assert s > 0.9


def test_clip_iqa_zero_embedding_returns_half():
    emb = np.zeros(4, dtype=np.float32)
    prompts = {
        "positive": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "negative": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }
    assert clip_iqa_score(emb, prompts) == 0.5


def test_clip_iqa_dim_mismatch_raises():
    emb = np.ones(3, dtype=np.float32)
    prompts = {
        "positive": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "negative": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="dim"):
        clip_iqa_score(emb, prompts)


def test_load_prompt_embeddings_file():
    path = Path(__file__).resolve().parent.parent / "bench" / "prompt_embeddings.json"
    if not path.exists():
        pytest.skip("prompt_embeddings.json not exported yet")
    prompts = load_prompt_embeddings(path)
    assert prompts["positive"].shape == prompts["negative"].shape
    assert prompts["positive"].shape[1] == 512  # CLIP ViT-B/32 dim
