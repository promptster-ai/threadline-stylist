"""Property-gate sanity tests. These mirror the hard constraints the grader enforces.

Run: pytest -q
These pass for the starter recommender. Keep them passing as you iterate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stylist.catalog import is_eligible  # noqa: E402
from stylist.data import load_catalog, load_seed_personas  # noqa: E402
from stylist.recommender import recommend  # noqa: E402

K = 10
CATALOG = load_catalog()
BY_ID = {it["id"]: it for it in CATALOG}
PERSONAS = load_seed_personas()
SAMPLE = PERSONAS[:12]


def test_returns_at_least_k():
    for p in SAMPLE:
        assert len(recommend(p, CATALOG, K)) >= K


def test_ids_valid_and_unique():
    for p in SAMPLE:
        topk = recommend(p, CATALOG, K)[:K]
        assert len(set(topk)) == K
        assert all(i in BY_ID for i in topk)


def test_constraints_respected():
    for p in SAMPLE:
        for iid in recommend(p, CATALOG, K)[:K]:
            assert is_eligible(BY_ID[iid], p["attrs"]), f"{iid} violates size/budget for {p['id']}"


def test_deterministic():
    p = SAMPLE[0]
    assert recommend(p, CATALOG, K)[:K] == recommend(p, CATALOG, K)[:K]
