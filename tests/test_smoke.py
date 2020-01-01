"""Data integrity smoke tests."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stylist.data import (  # noqa: E402
    DATA_DIR, load_catalog, load_gold_labels, load_sales, load_seed_personas,
)


def test_catalog_shapes_align():
    catalog = load_catalog()
    emb = np.load(DATA_DIR / "catalog_embeddings.npy")
    assert len(catalog) == emb.shape[0]
    assert len(catalog[0]["embedding"]) == emb.shape[1]


def test_personas_have_embeddings_but_no_labels():
    personas = load_seed_personas()
    emb = np.load(DATA_DIR / "personas" / "photo_embeddings.npy")
    assert len(personas) == emb.shape[0]
    for p in personas:
        assert len(p["embedding"]) == emb.shape[1]
        assert {"size", "budget_usd", "style_archetype"} <= set(p["attrs"])
        assert "on_style_item_ids" not in p, "seed personas must NOT carry clean labels"


def test_sales_and_gold_reference_real_items():
    catalog_ids = {it["id"] for it in load_catalog()}
    persona_ids = {p["id"] for p in load_seed_personas()}
    sales = load_sales()
    assert set(sales) <= persona_ids
    for engagement in sales.values():
        assert set(engagement) <= catalog_ids
    gold = load_gold_labels()
    assert 0 < len(gold) < len(persona_ids), "gold anchor should be small"
    for items in gold.values():
        assert set(items) <= catalog_ids


def test_embeddings_same_dim():
    cat = np.load(DATA_DIR / "catalog_embeddings.npy")
    per = np.load(DATA_DIR / "personas" / "photo_embeddings.npy")
    assert cat.shape[1] == per.shape[1], "person and product embeddings must share a space dim"
