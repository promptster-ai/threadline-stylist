# Brief — Style Match recommender

## Context

You've joined the Personalization team at **Threadline**, a mid-market apparel retailer. We're
launching **Style Match**: a customer uploads a photo, we recommend pieces from our catalog that
suit *them* — their coloring, build, and personal style — not just whatever is trending.

Upstream, a vision service has already turned each customer photo into an **appearance embedding**.
You do **not** work with raw images — you work with these embeddings plus structured attributes
(size, budget, occasion, style archetype, …). Every catalog item also has a **product embedding**.

What you have to learn from is **messy, like real life**:

- a large **engagement log** (`data/sales.json`) of how customers in each segment interacted with
  items — but it's **biased toward bestsellers and featured products**, not fit;
- a **tiny** clean, stylist-curated set (`data/personas/gold_labels.json`) — trustworthy but far
  too small to train on alone.

The previous prototype just returned the items whose embedding was closest to the customer's
embedding. It tested terribly. Part of your task is to understand why (see `docs/DATA.md`) and do
better.

## Your task

Implement `recommend(person, catalog, k)` in `src/stylist/recommender.py` so it returns the `k`
catalog items that best suit a given customer, as a coherent outfit.

You should:

1. **Understand the data.** Explore the engagement log, the gold anchor, and how appearance
   embeddings relate (or don't) to product embeddings. The starter `recommend` (popularity) and
   `scripts/evaluate.py` are your starting points.
2. **Turn the biased log into signal.** Decide how to infer "what suits this segment" from
   engagement that's skewed by popularity — and how to combine that with the scarce gold anchor.
3. **Learn the appearance→product mapping.** Naive cosine between the two embedding spaces fails;
   learn the relationship from data. Method is your call (learned projection, nearest-segments, a
   trained model, attribute features, or a blend) — justify it.
4. **Build a coherent outfit.** Respect the hard constraints (size, budget) and the outfit-coherence
   property (span ≥ 3 categories — don't return 10 tops).
5. **Validate yourself.** The gold set is tiny and optimistic. Decide how to estimate whether your
   model generalizes to unseen customers, and build that check. We care how you convince yourself
   you're right, not just the final number.

## Ground rules

- Work in this repo's style and structure. Keep `recommend`'s signature exactly as specified.
- Any approach is fair game as long as it's your own work and clears the bar in `EVALUATION.md`.
- You may add modules, a training step, and a model artifact. Keep it runnable from a clean checkout
  (`pip install -e ".[dev]"` then `pytest -q`) with only numpy + scikit-learn, no network.

## Definition of done

- `recommend` clears the precision@k and NDCG@k floor on held-out customers (see `EVALUATION.md`).
- The property gate passes (valid, in-budget, in-size, coherent, deterministic).
- A short note (`NOTES.md` or PR description) on your approach, how you handled the biased log, your
  validation, and what you'd do next.
