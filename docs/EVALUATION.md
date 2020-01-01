# Evaluation

Your `recommend` is scored automatically against a **held-out set of ~40 customers** that are not
in this repo and have no engagement log or labels here. `k = 10`.

Two layers: a **property gate** (hard pass/fail) and a **quality bar** (metric floor). To pass the
automated bar you must clear both.

## 1. Property gate — must pass

For every held-out customer, your top-10 must:

- contain only ids that exist in the catalog,
- contain no duplicates,
- return at least 10 items,
- contain only items that **stock the customer's size** and are **within their budget**,
- **span at least 3 categories** (outfit coherence — not 10 of one thing), and
- be **deterministic** (same input → same output).

`stylist.catalog.is_eligible` enforces size/budget; `tests/test_contract.py` checks these locally.

## 2. Quality bar — metric floor

Aggregated over all held-out customers:

| metric | definition | floor |
|--------|------------|-------|
| **precision@10** | fraction of your top-10 that are in the customer's curated on-style set | **≥ 0.45** |
| **NDCG@10** | rank-weighted overlap with the graded on-style set | **≥ 0.37** |

For reference: ranking by popularity / raw engagement, or training only on the tiny gold set, both
score well **under** these floors. An approach that reads the biased engagement log as real
preference signal *and* learns the appearance→product mapping clears them with margin. The floors
reject the lazy shortcuts, not thoughtful work.

## What you see vs. what we see

- **You** self-score with `python scripts/evaluate.py` — but that runs on the tiny gold anchor and
  is OPTIMISTIC and high-variance. Build your own validation (e.g. hold out part of the engagement
  log and reason about the popularity bias) — that's part of the task.
- **The held-out customers are drawn differently from your seed segments** (different style mix,
  tighter budgets — real new-customer "cold start"). A model that memorizes the seed segments, or
  tuning that trusts a seed-distribution validation score, will read far better locally than it
  scores for real. Favor approaches that learn a generalizable mapping and validate accordingly.
- **We** run the held-out grader after you submit. It reports precision@10, NDCG@10, the
  property-gate result, and a per-check pass/fail breakdown.

## Beyond the automated bar

Clearing the floor is necessary, not sufficient. We also read *how* you worked: how you explored the
data, how you handled the biased log, the reasoning behind your modeling choice, how you validated
generalization, and how you balanced ranking quality against the constraints. Capture that in
`NOTES.md` or your PR description.
