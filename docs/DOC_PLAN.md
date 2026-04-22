# Documentation Rewrite Plan

## 1. User Goals

The rewritten docs should let a new user answer these questions quickly:

1. I have a binary treatment. Which estimator should I use, and what should `t` look like?
2. I have a continuous treatment. How do I estimate a dose-response curve?
3. I have a categorical treatment with more than two levels. How do I estimate average outcomes per level?
4. I want one full end-to-end example with known ground truth so I can understand the library before using it on real data.

From the user perspective, the current gaps are:

- The docs only expose one example path.
- The main walkthrough uses stale imports and stale method names.
- There is no obvious decision path by treatment type.
- It is not clear that boolean treatments should be treated as categorical in `skcausal`.
- The examples do not consistently teach the expected shapes and schemas for `X`, `t`, and `y`.

## 2. Success Criteria

The rewrite is complete when:

- A new user can choose the right guide from the homepage in one or two clicks.
- Each guide contains a minimal working example that runs against the current API.
- Binary, continuous, and categorical treatments each have a dedicated example page.
- The full `SyntheticDataset2` walkthrough remains available at the current `dose_response_curve.qmd` path.
- All example code uses current imports, current dataset methods, and current plotting logic.

## 3. Proposed Information Architecture

The docs should be organized around the user's treatment type, not around internal modules.

Planned page structure:

- `docs/index.qmd`
- `docs/examples/index.qmd`
- `docs/examples/binary_treatments.qmd`
- `docs/examples/continuous_treatments.qmd`
- `docs/examples/categorical_treatments.qmd`
- `docs/examples/dose_response_curve.qmd`

Navigation changes:

- Update `docs/index.qmd` to present four user-facing entry points:
  - Binary treatments
  - Continuous treatments
  - Categorical treatments
  - Full `SyntheticDataset2` walkthrough
- Keep the density guide as a separate entry point.
- Update `docs/examples/index.qmd` into an examples hub with a short "Choose the right guide" section.
- Update `docs/_quarto.yml` so the navbar has an `Examples` menu instead of a single `Example` link.
- Keep `docs/examples/dose_response_curve.qmd` in place so existing links do not break.

## 4. Recommended User Flow

The homepage and examples hub should guide users like this:

1. If your treatment is `True`/`False`, go to the binary treatments guide.
2. If your treatment is a dosage, score, or other numeric exposure, go to the continuous treatments guide.
3. If your treatment is a label such as `control`, `treated`, or `placebo`, go to the categorical treatments guide.
4. If you want the detailed benchmark-style walkthrough with ground truth, go to the full `SyntheticDataset2` guide.

## 5. Page-by-Page Plan

### A. Binary Treatments Guide

Primary user question:

"I have treated vs control data. What is the shortest correct example I can copy?"

Recommended dataset:

- `SyntheticDataset2Discrete`

Key teaching points:

- In `skcausal`, boolean treatments are handled through the categorical-treatment estimators.
- The treatment query passed to `predict` should be an explicit treatment DataFrame.
- Users should understand the difference between direct, weighting, and doubly robust estimators without reading theory first.

Planned outline:

1. State that binary is a special case of categorical treatment.
2. Generate data with `SyntheticDataset2Discrete`.
3. Show `X`, `t`, and `y` and explain their expected structure.
4. Fit `CategoricalDirectMethod` as the simplest baseline.
5. Fit `CategoricalInversePropensityWeighting`.
6. Fit `CategoricalDoublyRobust`.
7. Query average potential outcomes for `False` and `True`.
8. End with a short interpretation section: average potential outcome, uplift, and when to prefer DR over simpler methods.

Implementation note:

- Prefer `PermutationWeighting` in examples whenever a stabilized density ratio is appropriate.

### B. Continuous Treatments Guide

Primary user question:

"I have a numeric treatment. How do I estimate an average dose-response curve with the current API?"

Recommended dataset:

- `SimpleLinearDataset(t_types=("continuous",))` for the quickstart example

Reasoning:

- This keeps the quickstart short and easy to read.
- The more realistic `SyntheticDataset2` example can stay focused in the full walkthrough.

Key teaching points:

- Continuous treatment examples should be shown on a treatment grid, not only on observed treatment values.
- Users need one minimal example first, then one density-aware example.

Planned outline:

1. Generate a simple continuous-treatment dataset.
2. Show `X`, `t`, and `y`.
3. Create a treatment grid.
4. Fit `DirectRegressor` as the simplest covariate-aware baseline.
5. Fit `GPS` as the first density-based method.
6. Plot both estimated curves.
7. Link to the full `SyntheticDataset2` walkthrough for a richer benchmark and ground-truth comparison.

### C. Categorical Treatments Guide

Primary user question:

"I have more than two treatment levels. How do I estimate average outcomes for each level?"

Recommended example setup:

- A small in-page synthetic multiclass example using a `polars` Enum treatment such as `control`, `treated`, and `placebo`

Reasoning:

- There is no built-in multiclass synthetic dataset today.
- For the first rewrite pass, creating the example inline is lower scope than adding a new dataset class.

Key teaching points:

- Each observed treatment level is treated as a valid query level.
- Predictions are average potential outcomes for requested levels.
- Only observed treatment levels should be requested.

Planned outline:

1. Build a small multiclass synthetic dataset directly in the page.
2. Make the treatment column explicitly categorical or enum-valued.
3. Fit `CategoricalDirectMethod`.
4. Fit `CategoricalInversePropensityWeighting`.
5. Fit `CategoricalDoublyRobust`.
6. Query predictions for each observed level.
7. Compare estimated average outcomes across levels.
8. End with a note that binary treatment is the same API family with only two levels.

### D. Full `SyntheticDataset2` Walkthrough

Primary user question:

"Show me one end-to-end example that explains the library in practice and compares estimators against known truth."

File to rewrite:

- `docs/examples/dose_response_curve.qmd`

Keep this path stable because it is already linked from the homepage and navbar.

Planned outline:

1. Introduce `SyntheticDataset2` and why it is useful for benchmarking.
2. Generate data with the current dataset API.
3. Inspect `X`, `t`, and `y`.
4. Build a smooth treatment grid with `dataset.get_grid(...)`.
5. Compute the true ADRF with `dataset.predict(X, treatment_grid)`.
6. Visualize the observed data and the true curve.
7. Fit `DirectNoCovariates` as the naive baseline.
8. Fit `DirectRegressor` as the first covariate-aware baseline.
9. Fit `GPS`.
10. Fit `DoublyRobustPseudoOutcome`.
11. Plot all estimated curves against the true ADRF.
12. End with a short estimator-selection guide for users.

Implementation note:

- `DoublyRobustPseudoOutcome` requires a stabilized density estimator. Prefer `PermutationWeighting` for the walkthrough unless a page specifically needs a wrapped conditional density estimator.

## 6. Migration Rules for the Rewrite

The old `dose_response_curve.qmd` page is outdated. The rewrite should explicitly fix these issues:

- Replace stale internal imports with imports from current public modules that exist in the repo.
- Replace `prepare(n=..., split_seed=..., preparation_seed=...)` with the current dataset API.
- Replace the nonexistent `get_adrf(...)` call with `dataset.predict(X, treatment_grid)`.
- Replace plots built from sorted observed treatments with plots over an explicit treatment grid when the goal is to show a smooth response curve.
- Remove references to modules that no longer exist, such as the old continuous estimator import paths.

## 7. Writing Conventions Across All Guides

Every page should follow the same user-first pattern:

1. Start with "When to use this guide".
2. Show a minimal working example early.
3. Explain the expected structure of `X`, `t`, and `y`.
4. Use current, copy-pastable imports only.
5. Use `polars` data structures consistently, since that is what the datasets currently expose.
6. Add one plot or compact table that lets the user see the result immediately.
7. End with "What to try next" and link to the next most relevant page.

Additional content rules:

- Prefer current public APIs over deep internal imports.
- Avoid long theory blocks before the first working example.
- Explain estimator tradeoffs in plain language after the code works.
- Keep the first version of each page minimal; optional variants can be added later.

## 8. Implementation Order

Recommended execution order:

1. Rewrite `docs/examples/dose_response_curve.qmd` around the current `SyntheticDataset2` API while preserving its path.
2. Add `docs/examples/binary_treatments.qmd`.
3. Add `docs/examples/continuous_treatments.qmd`.
4. Add `docs/examples/categorical_treatments.qmd`.
5. Rewrite `docs/examples/index.qmd` into a real examples hub.
6. Update `docs/index.qmd` so the landing page routes users by treatment type.
7. Update `docs/_quarto.yml` navigation.
8. Build the docs and fix any broken imports, execution failures, or stale links.

## 9. Out-of-Scope for the First Pass

To keep the rewrite focused, these items should stay out of scope unless needed later:

- Adding a brand-new multiclass dataset class just for documentation.
- Writing API reference pages for every estimator before the example guides are fixed.
- Expanding the density documentation unless a causal example needs a missing explanation.

## 10. Final Deliverable Shape

After the rewrite, the docs should feel like this from the user's perspective:

- I can identify my treatment type immediately.
- I can copy one small example that runs.
- I can find one deeper walkthrough that explains how the pieces fit together.
- I do not need to reverse-engineer stale imports or guess which estimator family applies to my data.