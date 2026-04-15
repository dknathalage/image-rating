"""Python port of RatingPipeline.clipIQAScore."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def load_prompt_embeddings(path: Path) -> dict:
    blob = json.loads(path.read_text())
    return {
        "positive": np.asarray(blob["positive"], dtype=np.float32),
        "negative": np.asarray(blob["negative"], dtype=np.float32),
    }


def clip_iqa_score(embedding: np.ndarray, prompts: dict, logit_scale: float = 100.0) -> float:
    """CLIP-IQA+: mean softmax P(positive prompt) across antonym pairs.

    Mirrors Swift `RatingPipeline.clipIQAScore`. Defensive L2-normalises the image
    embedding; prompts are assumed pre-normalised.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm < 1e-6:
        return 0.5
    if not (0.999 < norm < 1.001):
        emb = emb / norm
    pos = np.asarray(prompts["positive"], dtype=np.float32)
    neg = np.asarray(prompts["negative"], dtype=np.float32)
    if pos.shape != neg.shape:
        raise ValueError(f"prompt shape mismatch: {pos.shape} vs {neg.shape}")
    if pos.shape[1] != emb.shape[0]:
        raise ValueError(
            f"embedding dim {emb.shape[0]} != prompt dim {pos.shape[1]}"
        )
    dot_pos = logit_scale * (emb[None, :] * pos).sum(axis=1)
    dot_neg = logit_scale * (emb[None, :] * neg).sum(axis=1)
    max_v = np.maximum(dot_pos, dot_neg)
    e_pos = np.exp(dot_pos - max_v)
    e_neg = np.exp(dot_neg - max_v)
    probs = e_pos / (e_pos + e_neg)
    return float(probs.mean())
