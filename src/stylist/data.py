"""
Data loaders. Assemble the on-disk data into the dict shapes the ``recommend`` contract uses.
These are DONE for you — build your model on top.

Files (see docs/DATA.md):
    data/catalog.json                  catalog item attributes (ordered)
    data/catalog_embeddings.npy        (N_items, D) row i = catalog[i]
    data/personas/profiles.json        SEED personas: attrs + photo (NO labels)
    data/personas/photo_embeddings.npy (N_seed, D) row i = profiles[i]
    data/sales.json                    BIASED engagement log per seed persona (your main signal)
    data/personas/gold_labels.json     tiny CLEAN stylist-curated set (~10 personas; validation anchor)
    data/personas/photos/*.png         seed persona images (exploration only)
    data/catalog_images/*.png          product thumbnails (exploration only)

There are NO clean labels for most personas and NONE for the held-out set — inferring which items
suit a customer from the engagement log is the task.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    """Catalog items, each with its ``embedding`` (list[float]) attached."""
    items = json.loads((DATA_DIR / "catalog.json").read_text())
    emb = np.load(DATA_DIR / "catalog_embeddings.npy")
    for i, it in enumerate(items):
        it["embedding"] = emb[i].astype(float).tolist()
    return items


@lru_cache(maxsize=1)
def load_seed_personas() -> list[dict]:
    """Seed personas, each with ``id``, ``attrs``, ``photo``, and ``embedding`` attached.

    These have NO clean labels — use the engagement log (load_sales) and the gold anchor
    (load_gold_labels) to figure out what suits them.
    """
    profiles = json.loads((DATA_DIR / "personas" / "profiles.json").read_text())
    emb = np.load(DATA_DIR / "personas" / "photo_embeddings.npy")
    for i, p in enumerate(profiles):
        p["embedding"] = emb[i].astype(float).tolist()
    return profiles


@lru_cache(maxsize=1)
def load_sales() -> dict[str, dict[str, int]]:
    """Biased engagement log: persona_id -> {item_id: interaction_count}.

    This is your primary (noisy) training signal. It over-weights popular/featured items, so
    ranking by raw engagement is not the same as ranking by fit.
    """
    rows = json.loads((DATA_DIR / "sales.json").read_text())
    return {r["persona_id"]: {p["item_id"]: p["count"] for p in r["purchases"]} for r in rows}


@lru_cache(maxsize=1)
def load_gold_labels() -> dict[str, list[str]]:
    """Small clean stylist-curated anchor: persona_id -> [item_ids] (best first).

    Too few to train on alone — use it to validate/calibrate how you read the engagement log.
    """
    return json.loads((DATA_DIR / "personas" / "gold_labels.json").read_text())
