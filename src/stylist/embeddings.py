"""Embedding helpers. Embeddings are L2-normalized float vectors (see docs/DATA.md)."""

from __future__ import annotations

import numpy as np


def as_matrix(items: list[dict]) -> np.ndarray:
    """Stack item/person ``embedding`` fields into an (N, D) float array."""
    return np.asarray([it["embedding"] for it in items], dtype=np.float64)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of ``a`` against every row of ``b`` -> (len(a), len(b))."""
    return l2_normalize(a) @ l2_normalize(b).T
