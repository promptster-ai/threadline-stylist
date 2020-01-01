"""
The recommendation contract.

This is the single seam your work is graded against. The held-out evaluator imports
``recommend`` from ``stylist.recommender`` and calls it exactly as described below. Keep
this signature stable; put your implementation in ``stylist/recommender.py``.

Inputs
------
person : dict
    {
      "id":        str,
      "embedding": list[float],   # frozen appearance embedding (see docs/DATA.md)
      "attrs": {
          "skin_tone": str, "hair": str, "body_type": str, "height_cm": int,
          "style_archetype": str, "occasion": str,
          "budget_usd": int,      # hard constraint: do not recommend pricier items
          "size": str,            # hard constraint: item must stock this size
      },
    }

catalog : list[dict]
    Each item:
    {
      "id": str, "category": str, "color": str, "pattern": str, "fit": str,
      "season": str, "price": float, "sizes": list[str],
      "embedding": list[float],   # frozen product embedding, same dimensionality as person
    }
    NOTE: the evaluator passes its OWN authoritative catalog (same ids/embeddings as
    data/). Treat ``catalog`` as the source of truth at call time — do not hard-code ids.

k : int
    Number of items to return (default 10).

Returns
-------
list[str]
    Item ids, ranked best-first, length >= k. The evaluator scores the top-k.

Hard constraints (property gate — violating these fails regardless of ranking quality):
    * every returned id exists in ``catalog``
    * no duplicate ids
    * at least k ids returned
    * every top-k item stocks ``person["attrs"]["size"]`` and costs <= ``budget_usd``
    * the top-k spans at least 3 categories (outfit coherence — not 10 of one thing)
    * deterministic: identical inputs -> identical output

What "good" means (quality bar — see docs/EVALUATION.md):
    Items that genuinely SUIT the person, measured as precision@k and NDCG@k against a held-out,
    hand-curated relevance set. Two things make this non-trivial (docs/DATA.md):
      * appearance-space and product-space are DIFFERENT spaces, so naive cosine between a person
        and items does not work — you must learn the mapping from data; and
      * your training signal is a BIASED engagement log (data/sales.json) plus a tiny clean gold
        anchor (data/personas/gold_labels.json) — popularity is not fit.
"""

from __future__ import annotations

# Embedding dimensionality is discovered from the data at runtime; never hard-code it.
RETURN_K = 10
