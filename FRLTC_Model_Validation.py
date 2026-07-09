#!/usr/bin/env python3
"""
FR-LTC implementation demonstration: Factor-Residual Local Turning Calculus
with stress-gated usage control.

This script is intentionally a diagnostic implementation, not a trading strategy.
It demonstrates the core FR-LTC pipeline:

    observed matrix R_W
    -> teacher residual E_W
    -> low-rank residual modes by SVD
    -> scalar oscillatory-envelope fit for mode scores
    -> stability and stress diagnostics
    -> accept / downgrade / fallback / reject gate

Outputs written to the chosen output directory. By default, this is ./outputs:
    FR-LTC_gate_metrics.csv
    FR-LTC_singular_values.csv
    FR-LTC_mode_fit.csv
    FR-LTC_parameter_stability.csv
    FR-LTC_residual_singular_values.png
    FR-LTC_mode_fit.png
    FR-LTC_mode_derivatives.png
    FR-LTC_numerical_audit.csv
    FR-LTC_run_summary.txt

Envelope slope is denoted by beta in code and output columns; selected residual rank remains r.

The default script deliberately implements the safe residual-diagnostic version.
It includes a compact parameter-stability report, but it does not attempt to
calibrate every possible stress family or run the optional augmented-regression
version. Those extensions require separate out-of-window validation and leakage
controls on real data.

Dependencies: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class OEFit:
    m: float
    beta: float
    omega: float
    coeff_cos: float
    coeff_sin: float
    rmse: float
    relative_rmse: float

    @property
    def complex_amplitude(self) -> complex:
        # Re[(a - i b) exp(i omega tau)] = a cos(omega tau) + b sin(omega tau).
        return complex(self.coeff_cos, -self.coeff_sin)


@dataclass
class NumericalTolerance:
    """Numerical-analysis tolerances used by the implementation audit.

    Algebraic tolerances are intentionally much tighter than finite-difference
    derivative tolerances. Reconstruction and orthogonality identities are
    checked close to floating-point precision, while derivative checks compare
    analytic O-E derivative symbols to centered grid finite differences.
    """

    algebraic: float = 1.0e-10
    projection: float = 1.0e-10
    orthogonality: float = 1.0e-10
    derivative_fd: float = 5.0e-2


@dataclass
class AuditRow:
    check: str
    value: float
    tolerance: float
    passed: bool
    scale: str
    note: str


FLOAT_EPS = float(np.finfo(np.float64).eps)


def orthonormal_vector(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=n)
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise RuntimeError("Random generator produced a zero vector.")
    return x / norm


def teacher_basis(tau: np.ndarray) -> np.ndarray:
    """Teacher factors F_W, shaped k x L."""
    tau_centered = tau - tau.mean()
    return np.vstack([
        np.ones_like(tau),
        tau_centered,
        tau_centered ** 2,
    ])


def real_oscillatory_envelope(
    tau: np.ndarray,
    amplitude: float,
    m: float,
    beta: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    return amplitude * (tau ** m) * np.exp(beta * tau) * np.cos(omega * tau + phi)


def generate_synthetic_matrix(
    seed: int = 8,
    n_assets: int = 40,
    n_time: int = 96,
    tau_min: float = 0.05,
    tau_max: float = 1.0,
    noise_scale: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Generate a synthetic teacher-plus-residual local window."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(tau_min, tau_max, n_time)
    F = teacher_basis(tau)
    k = F.shape[0]

    # Teacher loadings B and teacher matrix B F.
    B = rng.normal(scale=0.35, size=(n_assets, k))
    teacher = B @ F

    # Two stable residual directions and two scalar local scores.
    u1 = orthonormal_vector(rng, n_assets)
    raw_u2 = orthonormal_vector(rng, n_assets)
    u2 = raw_u2 - u1 * np.dot(u1, raw_u2)
    u2 = u2 / np.linalg.norm(u2)

    z1 = real_oscillatory_envelope(tau, amplitude=3.20, m=0.85, beta=-0.35, omega=9.0, phi=0.45)
    z2 = real_oscillatory_envelope(tau, amplitude=1.80, m=0.55, beta=0.15, omega=4.5, phi=-0.35)

    low_rank_residual = np.outer(u1, z1) + np.outer(u2, z2)
    noise = noise_scale * rng.normal(size=(n_assets, n_time))
    R = teacher + low_rank_residual + noise

    truth = {
        "teacher": teacher,
        "residual_clean": low_rank_residual,
        "noise": noise,
        "u1": u1,
        "u2": u2,
        "z1": z1,
        "z2": z2,
    }
    return tau, F, B, R, truth


def fit_teacher_ols(R: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate B in R = B F + E by row-wise OLS."""
    gram = F @ F.T
    B_hat = R @ F.T @ np.linalg.pinv(gram)
    E_hat = R - B_hat @ F
    return B_hat, E_hat


def svd_low_rank(E: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    return U[:, :r], s[:r], Vt[:r, :]


def relative_norm(numerator: np.ndarray, denominator: np.ndarray, eps: float = FLOAT_EPS) -> float:
    """Return ||numerator||_F / max(||denominator||_F, eps)."""
    return float(np.linalg.norm(numerator, ord="fro") / max(np.linalg.norm(denominator, ord="fro"), eps))


def safe_condition_number(A: np.ndarray) -> float:
    """Return the 2-norm condition number, using inf if NumPy fails."""
    try:
        return float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")


def orthogonality_error(Q: np.ndarray) -> float:
    """Return ||Q.T Q - I||_F for an orthonormal-column basis."""
    if Q.size == 0:
        return 0.0
    eye = np.eye(Q.shape[1])
    return float(np.linalg.norm(Q.T @ Q - eye, ord="fro"))


def projection_orthogonality_error(E: np.ndarray, F: np.ndarray) -> float:
    """Return normalized OLS normal-equation residual ||E F.T||_F/(||E||_F ||F||_F)."""
    denom = max(np.linalg.norm(E, ord="fro") * np.linalg.norm(F, ord="fro"), FLOAT_EPS)
    return float(np.linalg.norm(E @ F.T, ord="fro") / denom)


def fourth_order_grid_derivatives(tau: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centered grid derivative checks on a uniform grid.

    Returns the interior grid values and approximations to first, second, and
    third derivatives. The first and second derivative formulas are fourth-order
    centered; the third derivative formula is centered and exact for cubics.
    These are audit checks only. The paper's diagnostic derivatives are the
    analytic O-E derivative-symbol values.
    """
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    if tau.ndim != 1 or y.ndim != 1 or tau.size != y.size:
        raise ValueError("tau and y must be one-dimensional arrays of the same length.")
    if tau.size < 5:
        raise ValueError("At least five grid points are required for centered derivative checks.")
    h = np.diff(tau)
    if not np.allclose(h, h[0], rtol=1e-10, atol=1e-14):
        raise ValueError("fourth_order_grid_derivatives requires a uniform grid.")
    h = float(h[0])
    interior = slice(2, -2)
    yp = (y[:-4] - 8.0 * y[1:-3] + 8.0 * y[3:-1] - y[4:]) / (12.0 * h)
    ypp = (-y[:-4] + 16.0 * y[1:-3] - 30.0 * y[2:-2] + 16.0 * y[3:-1] - y[4:]) / (12.0 * h * h)
    yppp = (y[4:] - 2.0 * y[3:-1] + 2.0 * y[1:-3] - y[:-4]) / (2.0 * h ** 3)
    return tau[interior], yp, ypp, yppp


def vector_relative_error(a: np.ndarray, b: np.ndarray, eps: float = FLOAT_EPS) -> float:
    """Return ||a-b||_2 / max(||b||_2, eps)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), eps))


def audit_to_dataframe(rows: List[AuditRow]) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ for row in rows])


def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Distance ||(I-AA')B||_2 for orthonormal-column bases A and B."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional matrices.")
    residual = B - A @ (A.T @ B)
    return float(np.linalg.norm(residual, ord=2))


def estimate_rank_by_energy(s: np.ndarray, energy_threshold: float = 0.85) -> int:
    if s.size == 0 or np.sum(s ** 2) == 0.0:
        return 0
    cumulative = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(cumulative, energy_threshold) + 1)


def fit_oe_grid(
    tau: np.ndarray,
    z: np.ndarray,
    m_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
    omega_grid: np.ndarray | None = None,
) -> OEFit:
    """
    Fit x(tau)=tau^m exp(beta tau)[a cos(omega tau)+b sin(omega tau)]
    by grid search over (m,beta,omega) and linear least squares over (a,b).

    Here beta is the O-E envelope slope.  The selected residual rank remains
    denoted by r elsewhere in the implementation.
    """
    if m_grid is None:
        m_grid = np.linspace(0.0, 1.6, 17)
    if beta_grid is None:
        beta_grid = np.linspace(-0.8, 0.5, 14)
    if omega_grid is None:
        omega_grid = np.linspace(1.0, 14.0, 131)

    best: OEFit | None = None
    z_scale = np.sqrt(np.mean((z - z.mean()) ** 2)) + 1e-12

    for m in m_grid:
        tau_m = tau ** m
        for beta in beta_grid:
            envelope = tau_m * np.exp(beta * tau)
            for omega in omega_grid:
                X = np.column_stack([
                    envelope * np.cos(omega * tau),
                    envelope * np.sin(omega * tau),
                ])
                coeff, *_ = np.linalg.lstsq(X, z, rcond=None)
                fitted = X @ coeff
                rmse = float(np.sqrt(np.mean((z - fitted) ** 2)))
                rel = rmse / z_scale
                if best is None or rel < best.relative_rmse:
                    best = OEFit(
                        m=float(m),
                        beta=float(beta),
                        omega=float(omega),
                        coeff_cos=float(coeff[0]),
                        coeff_sin=float(coeff[1]),
                        rmse=rmse,
                        relative_rmse=float(rel),
                    )

    if best is None:
        raise RuntimeError("OE grid search failed to evaluate any candidate.")
    return best


def oe_predict_and_derivatives(tau: np.ndarray, fit: OEFit) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fitted OE mode and first three derivatives by normalized derivative symbols."""
    C = fit.complex_amplitude
    Z = C * (tau ** fit.m) * np.exp(fit.beta * tau) * np.exp(1j * fit.omega * tau)

    q = fit.m / tau + fit.beta + 1j * fit.omega
    qp = -fit.m / (tau ** 2)
    qpp = 2.0 * fit.m / (tau ** 3)

    P1 = q
    P2 = qp + q ** 2
    P3 = qpp + 3.0 * q * qp + q ** 3

    x0 = np.real(Z)
    x1 = np.real(Z * P1)
    x2 = np.real(Z * P2)
    x3 = np.real(Z * P3)
    return x0, x1, x2, x3


def finite_turn_counts(tau: np.ndarray, first_derivative: np.ndarray, second_derivative: np.ndarray) -> Tuple[int, int]:
    """Count sign changes in direction and curvature as a simple local-turn summary."""
    def sign_changes(y: np.ndarray) -> int:
        y = np.asarray(y)
        eps = 1e-8 * (np.nanmax(np.abs(y)) + 1e-12)
        signs = np.sign(np.where(np.abs(y) < eps, 0.0, y))
        # Forward-fill zeros to avoid counting numerical ties as two sign changes.
        for i in range(1, signs.size):
            if signs[i] == 0.0:
                signs[i] = signs[i - 1]
        for i in range(signs.size - 2, -1, -1):
            if signs[i] == 0.0:
                signs[i] = signs[i + 1]
        return int(np.sum(signs[1:] * signs[:-1] < 0.0))

    return sign_changes(first_derivative), sign_changes(second_derivative)


def oriented_score_from_svd(
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    reference_score: np.ndarray,
) -> np.ndarray:
    """Return the first scaled SVD score with a reproducible sign orientation.

    SVD signs are arbitrary.  For reporting across perturbations or neighboring
    windows, orient the first score so that it has nonnegative correlation with a
    reference score on the same grid.  This affects parameter-report readability
    only; subspace distances and singular values are invariant to the sign.
    """
    score = singular_values[0] * right_vectors[0, :]
    reference_score = np.asarray(reference_score, dtype=float)
    if score.shape != reference_score.shape:
        raise ValueError("score and reference_score must have the same shape for orientation.")
    corr = np.corrcoef(score, reference_score)[0, 1]
    if np.isfinite(corr) and corr < 0.0:
        score = -score
    return score


def fit_first_score_for_report(
    tau: np.ndarray,
    E: np.ndarray,
    reference_score: np.ndarray,
) -> Tuple[OEFit, np.ndarray, np.ndarray]:
    """Fit the first residual score of E for compact parameter-stability reporting."""
    U_tmp, s_tmp, Vt_tmp = np.linalg.svd(E, full_matrices=False)
    score = oriented_score_from_svd(s_tmp, Vt_tmp, reference_score)
    fit_tmp = fit_oe_grid(tau, score)
    return fit_tmp, score, s_tmp


def fit_to_parameter_vector(fit: OEFit) -> np.ndarray:
    """Numerical parameter vector used only for stability reporting."""
    return np.array([
        fit.m,
        fit.beta,
        fit.omega,
        fit.coeff_cos,
        fit.coeff_sin,
    ], dtype=float)


def normalized_parameter_drift(fit: OEFit, base_fit: OEFit) -> float:
    """Scale-aware parameter drift from the base fit.

    The report is diagnostic, not a statistical confidence interval.  It gives a
    compact, reproducible measure of whether neighboring-window and mild-stress
    fits preserve the same O-E parameter region.
    """
    theta = fit_to_parameter_vector(fit)
    base = fit_to_parameter_vector(base_fit)
    scale = np.maximum(np.abs(base), 1.0)
    return float(np.linalg.norm((theta - base) / scale) / np.sqrt(theta.size))


def score_correlation(score: np.ndarray, reference_score: np.ndarray) -> float:
    """Return a finite correlation summary, using NaN if correlation is undefined."""
    corr = np.corrcoef(score, reference_score)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def singular_gap_and_condition(s: np.ndarray, r: int) -> Tuple[float, float]:
    """Return gamma_r and condition_r for a singular-value array."""
    gamma = float((s[r - 1] - s[r]) / (s[0] + 1e-12)) if s.size > r else float("nan")
    condition = float(s[0] / (s[r - 1] + 1e-12)) if s.size >= r else float("nan")
    return gamma, condition


def parameter_stability_row(
    case: str,
    tau_case: np.ndarray,
    fit_case: OEFit,
    score_case: np.ndarray,
    reference_score: np.ndarray,
    singular_values: np.ndarray,
    selected_rank: int,
    base_fit: OEFit,
    note: str,
) -> Dict[str, object]:
    """Create one compact parameter-stability reporting row."""
    gamma, condition = singular_gap_and_condition(singular_values, selected_rank)
    return {
        "case": case,
        "n_time": int(tau_case.size),
        "m": fit_case.m,
        "envelope_slope_beta": fit_case.beta,
        "omega": fit_case.omega,
        "coeff_cos": fit_case.coeff_cos,
        "coeff_sin": fit_case.coeff_sin,
        "relative_rmse": fit_case.relative_rmse,
        "parameter_drift_from_base": normalized_parameter_drift(fit_case, base_fit),
        "score_correlation_with_base": score_correlation(score_case, reference_score),
        "estimated_rank_85pct_energy": estimate_rank_by_energy(singular_values),
        "singular_gap_gamma_r": gamma,
        "condition_r": condition,
        "note": note,
    }


def gate_decision(
    rel_rmse: float,
    gamma_r: float,
    condition_r: float,
    subspace_noise: float,
    subspace_window: float,
    stress_sensitivity: float,
    rank_instability: int,
) -> str:
    """Rule-based demonstration gate. Thresholds should be revalidated for real data."""
    if rel_rmse <= 0.35 and gamma_r >= 0.08 and condition_r <= 12.0 \
            and max(subspace_noise, subspace_window) <= 0.35 \
            and stress_sensitivity <= 0.55 and rank_instability == 0:
        return "accept"
    if rel_rmse <= 0.55 and gamma_r >= 0.04 and condition_r <= 25.0 \
            and max(subspace_noise, subspace_window) <= 0.55 \
            and stress_sensitivity <= 0.80:
        return "downgrade"
    if rel_rmse <= 0.80 and gamma_r >= 0.02:
        return "fallback"
    return "reject"


def run_demo(output_dir: Path, seed: int = 8, r: int = 2, strict_audit: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 100)
    tol = NumericalTolerance()

    tau, F, B_true, R, truth = generate_synthetic_matrix(seed=seed)
    B_hat, E_hat = fit_teacher_ols(R, F)
    U, s_all, Vt = np.linalg.svd(E_hat, full_matrices=False)
    U_r, s_r, Vt_r = U[:, :r], s_all[:r], Vt[:r, :]

    if not (1 <= r < min(E_hat.shape)):
        raise ValueError("rank must satisfy 1 <= rank < min(n_assets, n_time) so sigma_{r+1} is available.")

    # Deterministic numerical audit: OLS normal equations, exact residual bookkeeping,
    # SVD reconstruction, and orthogonality are checked at tight tolerance.
    full_svd_reconstruction = (U * s_all) @ Vt
    low_rank_reconstruction = U_r @ np.diag(s_r) @ Vt_r
    gram_F = F @ F.T
    audit_rows: List[AuditRow] = []

    teacher_identity_error = relative_norm(R - (B_hat @ F + E_hat), R)
    ols_orth_error = projection_orthogonality_error(E_hat, F)
    full_svd_error = relative_norm(E_hat - full_svd_reconstruction, E_hat)
    left_orth_error = orthogonality_error(U)
    right_orth_error = orthogonality_error(Vt.T)
    low_rank_tail_error = relative_norm(E_hat - low_rank_reconstruction, E_hat)

    audit_rows.extend([
        AuditRow("teacher_identity_R_equals_BF_plus_E", teacher_identity_error, tol.algebraic,
                 teacher_identity_error <= tol.algebraic,
                 "relative Frobenius", "Checks exact residual bookkeeping."),
        AuditRow("ols_normal_equation_EF_transpose", ols_orth_error, tol.projection,
                 ols_orth_error <= tol.projection,
                 "normalized Frobenius", "Checks residual orthogonality to teacher factors."),
        AuditRow("teacher_gram_condition_number", safe_condition_number(gram_F), np.nan, True,
                 "2-norm condition number", "Conditioning of F_W F_W^T; reported, not thresholded."),
        AuditRow("full_svd_reconstruction", full_svd_error, tol.algebraic,
                 full_svd_error <= tol.algebraic,
                 "relative Frobenius", "Checks U diag(sigma) V^T reconstruction."),
        AuditRow("left_singular_vector_orthogonality", left_orth_error, tol.orthogonality,
                 left_orth_error <= tol.orthogonality, "Frobenius", "Checks U^T U = I."),
        AuditRow("right_singular_vector_orthogonality", right_orth_error, tol.orthogonality,
                 right_orth_error <= tol.orthogonality, "Frobenius", "Checks V^T V = I."),
        AuditRow("low_rank_relative_tail_error", low_rank_tail_error, np.nan, True,
                 "relative Frobenius", "Reported truncation error for selected rank r."),
    ])

    # SVD convention is sign-ambiguous. Orient the first mode to match positive correlation with true z1 when available.
    mode1 = s_all[0] * Vt[0, :]
    if np.corrcoef(mode1, truth["z1"])[0, 1] < 0:
        U[:, 0] *= -1.0
        Vt[0, :] *= -1.0
        U_r = U[:, :r]
        Vt_r = Vt[:r, :]
        mode1 = -mode1

    fit = fit_oe_grid(tau, mode1)
    fitted, d1, d2, d3 = oe_predict_and_derivatives(tau, fit)

    # Scientific-computing derivative audit: compare analytic O-E derivative symbols
    # against centered finite-difference derivatives of the fitted path.
    _, fd1, fd2, fd3 = fourth_order_grid_derivatives(tau, fitted)
    fd_err_1 = vector_relative_error(fd1, d1[2:-2])
    fd_err_2 = vector_relative_error(fd2, d2[2:-2])
    fd_err_3 = vector_relative_error(fd3, d3[2:-2])
    audit_rows.extend([
        AuditRow("oe_first_derivative_fd_check", fd_err_1, tol.derivative_fd, fd_err_1 <= tol.derivative_fd,
                 "relative L2", "Analytic first derivative vs centered finite difference."),
        AuditRow("oe_second_derivative_fd_check", fd_err_2, tol.derivative_fd, fd_err_2 <= tol.derivative_fd,
                 "relative L2", "Analytic second derivative vs centered finite difference."),
        AuditRow("oe_third_derivative_fd_check", fd_err_3, tol.derivative_fd, fd_err_3 <= tol.derivative_fd,
                 "relative L2", "Analytic third derivative vs centered finite difference; naturally more sensitive."),
    ])

    direction_turns, curvature_turns = finite_turn_counts(tau, d1, d2)

    # Stability under small additive residual noise.
    noise_level = 0.04 * np.linalg.norm(E_hat, ord="fro") / np.sqrt(E_hat.size)
    E_noise = E_hat + noise_level * rng.normal(size=E_hat.shape)
    U_noise, s_noise, Vt_noise = np.linalg.svd(E_noise, full_matrices=False)
    sub_u_noise = subspace_distance(U_r, U_noise[:, :r])
    sub_v_noise = subspace_distance(Vt_r.T, Vt_noise[:r, :].T)
    sub_noise = max(sub_u_noise, sub_v_noise)

    # Neighboring-window stability: drop first two and last two columns and compare right subspaces on overlap.
    E_nb = E_hat[:, 2:-2]
    U_nb, s_nb, Vt_nb = np.linalg.svd(E_nb, full_matrices=False)
    # Compare left subspace directly and compare right scores only on common trimmed coordinates.
    sub_u_window = subspace_distance(U_r, U_nb[:, :r])
    sub_window = sub_u_window

    # Moderate stress: a small admissible perturbation used for base-window gate continuity.
    E_moderate = E_hat + 0.02 * np.linalg.norm(E_hat, ord="fro") / np.sqrt(E_hat.size) * rng.normal(size=E_hat.shape)
    U_moderate, s_moderate, Vt_moderate = np.linalg.svd(E_moderate, full_matrices=False)
    stress_sensitivity = max(
        subspace_distance(U_r, U_moderate[:, :r]),
        subspace_distance(Vt_r.T, Vt_moderate[:r, :].T),
    )

    # Severe stress: jump plus cross-sectional rotation. This is a misuse/boundary test.
    jump = np.zeros_like(E_hat)
    jump[:, int(0.68 * E_hat.shape[1]):int(0.76 * E_hat.shape[1])] = 0.45 * orthonormal_vector(rng, E_hat.shape[0])[:, None]
    E_stress = E_hat + jump + 0.13 * rng.normal(size=E_hat.shape)
    U_stress, s_stress, Vt_stress = np.linalg.svd(E_stress, full_matrices=False)
    severe_stress_sensitivity = max(
        subspace_distance(U_r, U_stress[:, :r]),
        subspace_distance(Vt_r.T, Vt_stress[:r, :].T),
    )

    # Compact parameter-stability report. This is intentionally not a full
    # production stress-family engine. It compares the fitted O-E parameters
    # across the base window, a neighboring trimmed window, mild residual noise,
    # and the moderate admissible perturbation used for gate-continuity checks.
    # The severe boundary stress remains a misuse/fallback test rather than a
    # parameter-stability calibration target.
    tau_nb = tau[2:-2]
    reference_nb = mode1[2:-2]
    fit_nb, score_nb, s_nb_report = fit_first_score_for_report(tau_nb, E_nb, reference_nb)
    fit_noise, score_noise, s_noise_report = fit_first_score_for_report(tau, E_noise, mode1)
    fit_moderate, score_moderate, s_moderate_report = fit_first_score_for_report(tau, E_moderate, mode1)

    parameter_stability = pd.DataFrame([
        parameter_stability_row(
            case="base_window",
            tau_case=tau,
            fit_case=fit,
            score_case=mode1,
            reference_score=mode1,
            singular_values=s_all,
            selected_rank=r,
            base_fit=fit,
            note="base O-E fit; drift is zero by definition",
        ),
        parameter_stability_row(
            case="neighboring_trimmed_window",
            tau_case=tau_nb,
            fit_case=fit_nb,
            score_case=score_nb,
            reference_score=reference_nb,
            singular_values=s_nb_report,
            selected_rank=r,
            base_fit=fit,
            note="drop first two and last two grid points; compact window-stability check",
        ),
        parameter_stability_row(
            case="mild_residual_noise",
            tau_case=tau,
            fit_case=fit_noise,
            score_case=score_noise,
            reference_score=mode1,
            singular_values=s_noise_report,
            selected_rank=r,
            base_fit=fit,
            note="small additive residual noise; compact perturbation-stability check",
        ),
        parameter_stability_row(
            case="moderate_gate_perturbation",
            tau_case=tau,
            fit_case=fit_moderate,
            score_case=score_moderate,
            reference_score=mode1,
            singular_values=s_moderate_report,
            selected_rank=r,
            base_fit=fit,
            note="moderate admissible perturbation used for base gate-continuity check",
        ),
    ])

    gamma_r = float((s_all[r - 1] - s_all[r]) / (s_all[0] + 1e-12)) if s_all.size > r else np.nan
    condition_r = float(s_all[0] / (s_all[r - 1] + 1e-12))
    rank_base = estimate_rank_by_energy(s_all)
    rank_noise = estimate_rank_by_energy(s_noise)
    rank_instability = int(rank_base != rank_noise)
    residual_energy = float(np.linalg.norm(E_hat, ord="fro") / (np.linalg.norm(R, ord="fro") + 1e-12))

    gate = gate_decision(
        rel_rmse=fit.relative_rmse,
        gamma_r=gamma_r,
        condition_r=condition_r,
        subspace_noise=sub_noise,
        subspace_window=sub_window,
        stress_sensitivity=stress_sensitivity,
        rank_instability=rank_instability,
    )

    stress_gamma = float((s_stress[r - 1] - s_stress[r]) / (s_stress[0] + 1e-12)) if s_stress.size > r else np.nan
    stress_condition = float(s_stress[0] / (s_stress[r - 1] + 1e-12))
    stress_gate = gate_decision(
        rel_rmse=min(1.0, fit.relative_rmse + 0.30),
        gamma_r=stress_gamma,
        condition_r=stress_condition,
        subspace_noise=sub_noise,
        subspace_window=sub_window,
        stress_sensitivity=severe_stress_sensitivity,
        rank_instability=1,
    )

    metrics = pd.DataFrame([
        {
            "case": "base_window",
            "selected_rank": r,
            "estimated_rank_85pct_energy": rank_base,
            "residual_energy_ratio": residual_energy,
            "singular_gap_gamma_r": gamma_r,
            "condition_r": condition_r,
            "mode1_relative_rmse": fit.relative_rmse,
            "subspace_noise_distance": sub_noise,
            "subspace_window_distance": sub_window,
            "stress_sensitivity": stress_sensitivity,
            "rank_instability_flag": rank_instability,
            "direction_turn_count": direction_turns,
            "curvature_turn_count": curvature_turns,
            "gate_decision": gate,
        },
        {
            "case": "stress_boundary_test",
            "selected_rank": r,
            "estimated_rank_85pct_energy": estimate_rank_by_energy(s_stress),
            "residual_energy_ratio": float(np.linalg.norm(E_stress, ord="fro") / (np.linalg.norm(R, ord="fro") + 1e-12)),
            "singular_gap_gamma_r": stress_gamma,
            "condition_r": stress_condition,
            "mode1_relative_rmse": min(1.0, fit.relative_rmse + 0.30),
            "subspace_noise_distance": sub_noise,
            "subspace_window_distance": sub_window,
            "stress_sensitivity": severe_stress_sensitivity,
            "rank_instability_flag": 1,
            "direction_turn_count": direction_turns,
            "curvature_turn_count": curvature_turns,
            "gate_decision": stress_gate,
        },
    ])
    metrics.to_csv(output_dir / "FR-LTC_gate_metrics.csv", index=False)

    n_sv = min(15, s_all.size, s_stress.size)
    singular_values = pd.DataFrame({
        "index": np.arange(1, n_sv + 1),
        "base_singular_value": s_all[:n_sv],
        "stress_singular_value": s_stress[:n_sv],
    })
    singular_values.to_csv(output_dir / "FR-LTC_singular_values.csv", index=False)

    mode_fit = pd.DataFrame({
        "tau": tau,
        "mode1_score": mode1,
        "oe_fitted_score": fitted,
        "oe_first_derivative": d1,
        "oe_second_derivative": d2,
        "oe_third_derivative": d3,
        "true_clean_score_reference": truth["z1"],
    })
    mode_fit.to_csv(output_dir / "FR-LTC_mode_fit.csv", index=False)
    parameter_stability.to_csv(output_dir / "FR-LTC_parameter_stability.csv", index=False)

    audit_df = audit_to_dataframe(audit_rows)
    audit_df.to_csv(output_dir / "FR-LTC_numerical_audit.csv", index=False)

    finite_tolerance_mask = np.isfinite(audit_df["tolerance"].to_numpy(dtype=float))
    audit_all_passed = bool(audit_df.loc[finite_tolerance_mask, "passed"].all())
    run_summary = {
        "seed": seed,
        "selected_rank": r,
        "n_assets": int(R.shape[0]),
        "n_time": int(R.shape[1]),
        "tau_min": float(tau.min()),
        "tau_max": float(tau.max()),
        "output_dir": str(output_dir.resolve()),
        "gate_base": gate,
        "gate_stress": stress_gate,
        "best_oe_fit": {
            "m": fit.m,
            "beta": fit.beta,
            "omega": fit.omega,
            "relative_rmse": fit.relative_rmse,
        },
        "parameter_stability_report": "FR-LTC_parameter_stability.csv",
        "implementation_scope": (
            "safe residual-diagnostic demonstration; compact parameter-stability "
            "report included; full stress-family calibration and optional "
            "augmented-regression validation are intentionally outside the default script"
        ),
        "audit_all_thresholded_checks_passed": audit_all_passed,
    }
    (output_dir / "FR-LTC_run_summary.txt").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    if strict_audit and not audit_all_passed:
        failed = audit_df.loc[finite_tolerance_mask & (~audit_df["passed"]), "check"].tolist()
        raise RuntimeError(f"Numerical audit failed: {failed}")

    # Plot 1: singular values.
    plt.figure(figsize=(7.0, 4.5))
    x = np.arange(1, 13)
    plt.plot(x, s_all[:12], marker="o", label="base residual")
    plt.plot(x, s_stress[:12], marker="s", label="stress residual")
    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value")
    plt.title("FR-LTC residual singular values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "FR-LTC_residual_singular_values.png", dpi=180)
    plt.close()

    # Plot 2: OE mode fit.
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(tau, mode1, marker="o", markersize=3, linewidth=1.0, label="SVD residual mode score")
    plt.plot(tau, fitted, linewidth=2.0, label="fitted oscillatory-envelope mode")
    plt.xlabel(r"local time $\tau$")
    plt.ylabel("mode score")
    plt.title("FR-LTC scalar mode fit after teacher residualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "FR-LTC_mode_fit.png", dpi=180)
    plt.close()

    # Plot 3: derivative diagnostics.
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(tau, d1, label="direction: first derivative")
    plt.plot(tau, d2, label="curvature: second derivative")
    plt.plot(tau, d3, label="curvature momentum: third derivative")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"local time $\tau$")
    plt.ylabel("derivative diagnostic")
    plt.title("FR-LTC local-turn derivative diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "FR-LTC_mode_derivatives.png", dpi=180)
    plt.close()

    print("FR-LTC implementation demo completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print(metrics.to_string(index=False))
    print("\nBest OE fit:")
    print(f"m={fit.m:.3f}, beta={fit.beta:.3f}, omega={fit.omega:.3f}, relative_rmse={fit.relative_rmse:.4f}")
    print("\nCompact parameter-stability report:")
    print(parameter_stability[[
        "case",
        "m",
        "envelope_slope_beta",
        "omega",
        "relative_rmse",
        "parameter_drift_from_base",
        "score_correlation_with_base",
    ]].to_string(index=False))
    print("\nNumerical audit:")
    print(audit_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FR-LTC factor-residual local turning implementation demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for CSV and PNG outputs. Defaults to ./outputs.")
    parser.add_argument("--seed", type=int, default=8, help="Random seed for reproducibility.")
    parser.add_argument("--rank", type=int, default=2, help="Candidate residual rank for the demonstration gate.")
    parser.add_argument("--strict-audit", action="store_true", help="Raise an error if any thresholded numerical audit check fails.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.output_dir, seed=args.seed, r=args.rank, strict_audit=args.strict_audit)
