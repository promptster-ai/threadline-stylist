# Threadline AI Stylist

Internal recommender service for Threadline's "Style Match" feature: given a customer's look,
recommend catalog items that genuinely suit them.

This repo is a working skeleton — data loaders, a baseline recommender, a self-eval harness, and
tests. **Your job is to make the recommendations good.** Start with `docs/BRIEF.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # numpy, scikit-learn (+ pytest, pillow)
```

## Run

```bash
pytest -q                    # property-gate + data sanity tests (green out of the box)
python scripts/evaluate.py   # in-sample precision@10 / NDCG@10 for your recommender
```

## Where things are

```
docs/                  READ THESE FIRST
  BRIEF.md             the task + product context
  DATA.md              dataset + the embedding model (read carefully — it's subtle)
  ARCHITECTURE.md      conventions, the contract, train/serve options
  EVALUATION.md        exactly how you're graded + the bar to clear
src/stylist/
  contract.py          the recommend() contract you implement (the graded seam)
  recommender.py       <-- YOUR WORK GOES HERE (starter baseline lives here)
  data.py              load_catalog(), load_seed_personas(),
                       load_sales(), load_gold_labels()            [done for you]
  catalog.py           is_eligible(), eligible_items()             [done for you]
  embeddings.py        cosine(), l2_normalize(), as_matrix()       [done for you]
data/                  catalog + embeddings + images
  sales.json           biased engagement log (your main signal)
  personas/            seed personas (no labels) + photos
  personas/gold_labels.json   tiny clean validation anchor
scripts/evaluate.py    self-eval harness (extend it)
tests/                 property-gate + smoke tests
```

## The one-line version

Customers give us a photo. We turn it into an appearance embedding. You recommend catalog items
that suit them — respecting their size and budget — and we measure how well your picks match a
hand-curated "this suits them" set. The catch is in `docs/DATA.md`.
