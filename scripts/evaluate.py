#!/usr/bin/env python3
"""
Local self-evaluation harness.

Scores your ``recommend`` against the small CLEAN gold anchor and runs the property-gate checks
the grader uses (including outfit coherence). Run it often:

    python scripts/evaluate.py

IMPORTANT — the gold anchor is TINY (~10 personas) and your model may have used it, so this score
is OPTIMISTIC and high-variance. The real grade is on ~40 held-out personas you never see, drawn a
bit differently. Build your own validation (e.g. hold out part of the sales log) — that's the task.

Metric definitions (precision@k, NDCG@k) match the grader exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stylist.catalog import is_eligible  # noqa: E402
from stylist.data import load_catalog, load_gold_labels, load_seed_personas  # noqa: E402
from stylist.recommender import recommend  # noqa: E402

K = 10
MIN_CATEGORIES = 3


def precision_at_k(rec, relevant):
    rel = set(relevant)
    return sum(1 for r in rec[:K] if r in rel) / K


def ndcg_at_k(rec, grades):
    dcg = sum(grades.get(r, 0.0) / math.log2(i + 2) for i, r in enumerate(rec[:K]))
    ideal = sorted(grades.values(), reverse=True)[:K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    catalog = load_catalog()
    by_id = {it["id"]: it for it in catalog}
    gold = load_gold_labels()
    personas = {p["id"]: p for p in load_seed_personas()}

    precisions, ndcgs, violations = [], [], 0
    for pid, labels in gold.items():
        p = personas[pid]
        grades = {iid: float(len(labels) - r) for r, iid in enumerate(labels)}
        rec = recommend(p, catalog, K)
        topk = rec[:K]
        cats = {by_id[i]["category"] for i in topk if i in by_id}
        ok = (len(rec) >= K and len(set(topk)) == K
              and all(i in by_id for i in topk)
              and all(is_eligible(by_id[i], p["attrs"]) for i in topk if i in by_id)
              and len(cats) >= MIN_CATEGORIES)
        if not ok:
            violations += 1
        precisions.append(precision_at_k(rec, labels))
        ndcgs.append(ndcg_at_k(rec, grades))

    n = len(gold)
    print(f"gold personas evaluated : {n}  (OPTIMISTIC / high-variance)")
    print(f"property violations     : {violations}  (must be 0; includes outfit coherence)")
    print(f"precision@{K}            : {sum(precisions)/n:.3f}")
    print(f"NDCG@{K}                 : {sum(ndcgs)/n:.3f}")
    print("\nThe held-out bar is in docs/EVALUATION.md. Held-out << this — build real validation.")


if __name__ == "__main__":
    main()
