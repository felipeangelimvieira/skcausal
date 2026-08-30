# Task 2 implementer report

## Files changed

- `src/skcausal/datasets/semi_synthetic.py`
- `tests/datasets/test_semi_synthetic_helpers.py`
- `.superpowers/sdd/2026-08-30-semi-synthetic-dgp/task-2-report.md`

## Design decisions

- The frozen score map uses a fitted numeric median-imputation/scaling pipeline
  and categorical missing-value/one-hot pipeline, while rejecting numeric data
  without finite observations.
- It freezes up to 16 encoded-feature products and creates sparse (at most five
  terms per output) projections for the four requested causal roles.
- The baseline surface normalizes total baseline risk and its confounding-only
  contribution independently, retaining the latter as the targeted-selection
  signal.

## RED/GREEN evidence

- RED: `uv run pytest tests/datasets/test_semi_synthetic_helpers.py -q` exited
  2 during collection with `ImportError: cannot import name
  '_fit_baseline_surface' from 'skcausal.datasets.semi_synthetic'`.
- GREEN: after implementation, the same command reported `3 passed in 2.24s`.

## Verification

- `uv run pytest tests/datasets/test_semi_synthetic_helpers.py
  tests/datasets/test_semi_synthetic_base.py -q` — `9 passed in 1.14s`.
- `uv run ruff check src/skcausal/datasets/semi_synthetic.py
  tests/datasets/test_semi_synthetic_helpers.py` — all checks passed.
- `uv run ruff format --check src/skcausal/datasets/semi_synthetic.py
  tests/datasets/test_semi_synthetic_helpers.py` — 2 files already formatted.
- `uv run pytest -q` — `726 passed, 419 warnings in 4.25s`.
- `git diff --check` — no whitespace errors.

## Concerns

None. The full-suite warnings are pre-existing deprecation and estimator
warnings; this task adds no warnings to the focused helper or base tests.
