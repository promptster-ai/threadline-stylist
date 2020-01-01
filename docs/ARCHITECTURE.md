# Architecture & conventions

## The contract is the seam

Everything is graded through one function: `recommend(person, catalog, k)` in
`src/stylist/recommender.py`. The full spec — input shapes, return shape, hard constraints — is in
`src/stylist/contract.py`. The grader imports `stylist.recommender.recommend` and calls it
directly. Keep the signature stable; everything else inside `src/stylist/` is yours to restructure.

## Train vs. serve — your choice

`recommend` must work from a clean checkout. Two patterns both work:

- **Fit-on-first-call.** Learn your mapping from the engagement log + gold anchor the first time
  `recommend` runs and cache it (e.g. `functools.lru_cache` or a module-level singleton). Simplest;
  fine here — fitting on ~160 segments is milliseconds.
- **Precompute an artifact.** Add a `scripts/train.py` that fits a model and writes an artifact
  (e.g. `artifacts/model.npz`); `recommend` loads it lazily. Cleaner separation; commit the
  artifact so a clean checkout still works.

Either way, **the repo must run end-to-end without network access** and without raw images.

## Libraries

`numpy` and `scikit-learn` are available in the grading environment. Do **not** rely on heavy or
GPU-only libraries (e.g. torch/tensorflow) or anything requiring a download at grade time — the
grader installs only `numpy` + `scikit-learn` and runs CPU-only with a tight time budget.

## House conventions

- `src/` layout, package is `stylist`. Tests under `tests/`, run with `pytest -q`.
- Type hints on public functions; small, named helpers over clever one-liners.
- Reuse the provided helpers (`stylist.data`, `stylist.catalog`, `stylist.embeddings`) rather than
  re-loading files ad hoc.
- Respect the hard constraints in code, not as an afterthought — `stylist.catalog.is_eligible` is
  there for exactly this.
- Keep `recommend` deterministic. If you use anything stochastic, seed it.

## Suggested shape (not prescriptive)

```
src/stylist/recommender.py   recommend() + your model
scripts/train.py             (optional) fit + persist an artifact
artifacts/                   (optional) committed model artifact
NOTES.md                     your approach, validation, and next steps
```
