# Data

All data lives under `data/`. Loaders are in `src/stylist/data.py`.

## Catalog — `data/catalog.json` + `data/catalog_embeddings.npy`

~400 items. `catalog.json` is an ordered list; row `i` of `catalog_embeddings.npy` is the product
embedding for item `i`. `load_catalog()` attaches each embedding onto its item as `embedding`.

| field | meaning |
|-------|---------|
| `id` | stable item id, e.g. `TL-0042` |
| `category` | tops / bottoms / dresses / outerwear / shoes / accessories |
| `color`, `pattern`, `fit`, `season` | structured attributes |
| `price` | USD |
| `sizes` | sizes this item stocks |
| `embedding` | product embedding (L2-normalized) |

Thumbnails: `data/catalog_images/<id>.png` (exploration only).

## Seed personas — `data/personas/profiles.json` + `data/personas/photo_embeddings.npy`

160 customers. Row `i` of `photo_embeddings.npy` is persona `i`'s appearance embedding;
`load_seed_personas()` attaches it as `embedding`. **These personas have NO clean labels.**

| field | meaning |
|-------|---------|
| `id`, `name` | persona id |
| `attrs.skin_tone`, `.hair`, `.body_type`, `.height_cm` | appearance attributes |
| `attrs.style_archetype`, `.occasion` | style context |
| `attrs.size`, `attrs.budget_usd` | **hard constraints** for recommendations |
| `photo` | path under `data/personas/photos/` (exploration only) |
| `embedding` | appearance embedding (same dimensionality as product embeddings) |

## Engagement log — `data/sales.json`  (your primary signal)

For each seed persona, an engagement log: how often customers in that segment interacted with
each item — `{persona_id, purchases: [{item_id, count}, …]}`. `load_sales()` returns
`persona_id -> {item_id: count}`.

**This signal is biased.** Engagement is driven partly by genuine fit and partly by
**popularity and merchandising** — bestsellers and featured items rack up interactions regardless
of who they suit. So *ranking items by raw engagement is not the same as ranking by fit*, and a
popularity recommender scores poorly. Turning this noisy log into a usable "what suits this
segment" signal is the core of the task (e.g. infer per-segment preferences, correct for
popularity, pool across similar segments — your call).

## Gold anchor — `data/personas/gold_labels.json`  (validation)

A **small** clean, stylist-curated set: `persona_id -> [item_ids]` (best first) for ~10 personas.
`load_gold_labels()` returns it. It is **too small to train on alone** — use it to sanity-check how
well you're reading the engagement log, or to anchor a model.

## The important part: appearance space ≠ product space

Appearance embeddings and product embeddings share dimensionality and come from the same vision
service, so it is tempting to score items by `cosine(person.embedding, item.embedding)`. **This does
not work** — appearance embeddings describe a *person*, product embeddings describe a *garment*, and
the stylist's judgment of what flatters whom is a **learned transformation**, not an identity. You
recover that transformation from data: the engagement log (noisy) plus the gold anchor (clean,
scarce). Structured attributes (`style_archetype`, `occasion`) carry signal too.

## Outfit coherence

Recommendations are an outfit, not 10 of one thing. The grader requires your top-k to span at least
**3 categories** (see `docs/EVALUATION.md`). A pure precision-optimizer that returns all tops will
fail this property, so plan to balance ranking quality with category spread.

## Embedding conventions

- L2-normalized float vectors; don't hard-code the dimensionality (read it from the arrays).
- `src/stylist/embeddings.py` has `cosine`, `l2_normalize`, `as_matrix`.
