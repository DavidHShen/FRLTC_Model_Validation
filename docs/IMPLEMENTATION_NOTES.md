# Implementation notes

## Purpose

`FRLTC_Model_Validation.py` is a compact scientific-computing demonstration of Factor-Residual Local Turning Calculus (FR-LTC). It is designed for model-validation diagnostics and usage-control gating. It is not a trading strategy.

## Core pipeline

1. Generate or receive an observed local matrix `R_W`.
2. Fit a teacher baseline by row-wise ordinary least squares.
3. Form the teacher residual matrix `E_W`.
4. Extract low-rank residual modes by SVD.
5. Fit the first residual-mode score with a scalar oscillatory-envelope form.
6. Compute analytic first, second, and third derivative diagnostics.
7. Audit numerical identities and finite-difference derivative checks.
8. Apply a rule-based accept / downgrade / fallback / reject gate.

## Main implementation objects

- `OEFit`: fitted scalar oscillatory-envelope parameters.
- `NumericalTolerance`: audit tolerances for algebraic, projection, orthogonality, and derivative checks.
- `AuditRow`: one row of the numerical audit report.
- `run_demo`: end-to-end reproducible demonstration runner.

## Diagnostics generated

- Gate metrics: residual energy, singular gap, condition number, O-E fit loss, subspace stability, stress sensitivity, rank instability, and gate label.
- Singular values: base and severe-stress singular spectra.
- Mode fit: residual-mode score, fitted O-E path, and derivative diagnostics.
- Parameter stability: compact comparison across base, neighboring-window, mild-noise, and moderate-perturbation cases.
- Numerical audit: residual bookkeeping, OLS normal equation, SVD reconstruction, orthogonality, rank-tail error, and finite-difference derivative checks.

## Default gate interpretation

The default deterministic example is constructed to show a stable base residual mode and a severe boundary-stress misuse case:

- `base_window`: accepted for limited diagnostic use.
- `stress_boundary_test`: sent to fallback.

The gate thresholds are demonstration thresholds and should be recalibrated before any real-data use.
