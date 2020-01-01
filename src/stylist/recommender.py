"""
YOUR WORK GOES HERE.

This starter ``recommend`` is intentionally weak: it ranks items by total engagement across the
sales log (i.e. global popularity), filtered to what the person can wear. It RESPECTS the hard
constraints (so it passes the property gate) but it ignores who the person is and inherits the
log's popularity bias, so its precision/NDCG are far below the bar in docs/EVALUATION.md.

Your job: recommend items that genuinely suit THIS person. You have:
  * appearance embeddings for each persona (load_seed_personas) and product embeddings (load_catalog)
    — but they live in DIFFERENT spaces (docs/DATA.md), so naive cosine between them does not work;
    you must learn the appearance->product mapping from data;
  * a BIASED engagement log (load_sales) — your main but noisy signal; popularity != fit;
  * a tiny CLEAN gold anchor (load_gold_labels) — too small to train on, good for validation.

How you turn that into good, coherent (multi-category) outfit recommendations is up to you.
You may use stylist.data, stylist.catalog, stylist.embeddings, numpy, scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from .catalog import is_eligible
from .data import load_sales


@lru_cache(maxsize=1)
def _popularity_ranking() -> list[str]:
    counts: Counter[str] = Counter()
    for engagement in load_sales().values():
        for item_id, n in engagement.items():
            counts[item_id] += n
    return [item_id for item_id, _ in counts.most_common()]


def recommend(person: dict, catalog: list[dict], k: int = 10) -> list[str]:
    by_id = {it["id"]: it for it in catalog}
    attrs = person["attrs"]

    picks = [iid for iid in _popularity_ranking()
             if iid in by_id and is_eligible(by_id[iid], attrs)]

    if len(picks) < k:  # backfill from the rest of the eligible catalog
        seen = set(picks)
        for it in catalog:
            if it["id"] not in seen and is_eligible(it, attrs):
                picks.append(it["id"])
                if len(picks) >= k:
                    break
    return picks[:k]
