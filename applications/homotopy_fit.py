"""Rao--Blackwellised diffusion training: homotopy on the exact target path.

T1  The semigroup identity E[phi_k(x_t) | x_0] = e^{-kt} phi_k(x_0) makes
    e^{-kt} R_hat_k(0) the Rao--Blackwellisation of the noised-sample
    coefficient estimator: measure the error of both against the exact
    slice coefficients.

T2  Training as deterministic homotopy: fit along the known path
    R(s) = diag(e^{-ks}) R_hat(0) from s large to 0 with a
    predictor--corrector, degree chosen once by held-out likelihood on
    independent validation data.  Compare with the cold continued fit.

T3  Path smoothness: the largest coefficient step along the homotopy.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
)
from nefqvf import Normal, NormalParams

BASELINE = NormalParams(0.0, 1.0)
GRID = np.linspace(-10.0, 10.0, 801)


def sample_mixture(d: float, size: int, rng: Any) -> np.ndarray:
    signs = np.where(rng.random(size) < 0.5, 1.0, -1.0)
    return signs * d + rng.standard_normal(size)


def exact_coefficients(d: float, t: float, k_max: int) -> np.ndarray:
    m = d * np.exp(-t)
    r = np.zeros(k_max + 1)
    for k in range(0, k_max + 1, 2):
        r[k] = m**k / math.sqrt(math.factorial(k))
    return r


def truth_law(d: float, t: float) -> np.ndarray:
    m = d * np.exp(-t)
    return (
        0.5
        * (np.exp(-0.5 * (GRID - m) ** 2) + np.exp(-0.5 * (GRID + m) ** 2))
        / np.sqrt(2.0 * np.pi)
    )


def fitted_law(c: np.ndarray, k: int) -> np.ndarray:
    basis = np.asarray(Normal.basis(GRID, k, BASELINE), dtype=float)
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    law = ref * np.abs(basis @ c) ** 2
    return law / np.trapezoid(law, GRID)


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(a - b), GRID))


def empirical(x: np.ndarray, k_max: int) -> np.ndarray:
    basis = np.asarray(Normal.basis(x, k_max, BASELINE), dtype=float)
    return basis.mean(axis=0)


# ------------------------------------------------------------------ T1 --
def test_rao_blackwell(d: float, seed: int, k_max: int = 12, redraws: int = 200):
    rng = np.random.default_rng(seed)
    x0 = sample_mixture(d, 5000, rng)
    r0 = empirical(x0, 2 * k_max)
    print("T1  RMSE of slice-coefficient estimators vs exact (modes 2,4,6,8):")
    for t in (0.25, 0.5, 1.0):
        exact = exact_coefficients(d, t, 2 * k_max)
        deterministic = np.exp(-np.arange(2 * k_max + 1) * t) * r0
        noised = np.stack(
            [
                empirical(
                    np.exp(-t) * x0
                    + np.sqrt(1 - np.exp(-2 * t)) * rng.standard_normal(len(x0)),
                    2 * k_max,
                )
                for _ in range(redraws)
            ]
        )
        for k in (2, 4, 6, 8):
            rmse_noised = float(np.sqrt(np.mean((noised[:, k] - exact[k]) ** 2)))
            rmse_det = abs(deterministic[k] - exact[k])
            print(
                f"      t {t:4.2f} mode {k}:  noised {rmse_noised:9.5f}"
                f"   deterministic {rmse_det:9.5f}"
                f"   ratio {rmse_noised / max(rmse_det, 1e-12):6.2f}"
            )


# ------------------------------------------------------------------ T2 --
def homotopy_fit(
    r0_train: np.ndarray, k: int, n_steps: int = 16
) -> tuple[np.ndarray, float]:
    """Predictor--corrector along R(s) = diag(e^{-ks}) R0, s: 3 -> 0."""

    phi = fitting_matrices(Normal, BASELINE, k)
    s_path = -np.log(np.linspace(np.exp(-3.0), 1.0, n_steps))
    c = np.zeros(k + 1, dtype=complex)
    c[0] = 1.0
    s_prev = s_path[0]
    max_step = 0.0
    for s in s_path:
        target = np.exp(-np.arange(2 * k + 1) * s) * r0_train[: 2 * k + 1]
        predictor = c * np.exp(np.arange(k + 1) * (s_prev - s))
        predictor = predictor / np.linalg.norm(predictor)
        fit = fit_complex_amplitude(phi, target, initial=predictor)
        step = float(np.linalg.norm(np.abs(fit["coefficients"]) - np.abs(c)))
        max_step = max(max_step, step)
        c = fit["coefficients"]
        s_prev = float(s)
    return c, max_step


def held_out_nll(c: np.ndarray, k: int, x_val: np.ndarray) -> float:
    basis = np.asarray(Normal.basis(x_val, k, BASELINE), dtype=float)
    h2 = np.abs(basis @ c) ** 2
    return -float(np.mean(np.log(h2 + 1e-300)))


def test_homotopy(d: float, seeds: int, k_max: int = 12):
    print("T2/T3  homotopy vs cold fit (degree by held-out NLL on validation):")
    tv_h, tv_c, steps = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        x = sample_mixture(d, 5000, rng)
        x_train, x_val = x[:4000], x[4000:]
        r0 = empirical(x_train, 2 * k_max)
        floors = np.asarray(
            Normal.basis(x_train, 2 * k_max, BASELINE), dtype=float
        ).std(axis=0) / np.sqrt(len(x_train))
        screen = max(
            [k for k in range(1, k_max + 1) if abs(r0[k]) > 2.0 * floors[k]],
            default=2,
        )
        candidates = sorted({k for k in (2, 4, 6, 8, 10, 12) if k <= screen} | {screen})
        best = None
        for k in candidates:
            c, max_step = homotopy_fit(r0, k)
            nll = held_out_nll(c, k, x_val)
            if best is None or nll < best[2]:
                best = (c, k, nll, max_step)
        c_star, k_star, _, max_step = best
        tv_homotopy = total_variation(truth_law(d, 0.0), fitted_law(c_star, k_star))

        cold = continued_complex_fit(
            fitting_matrices(Normal, BASELINE, k_star), r0[: 2 * k_star + 1]
        )["complex"]
        tv_cold = total_variation(
            truth_law(d, 0.0), fitted_law(cold["coefficients"], k_star)
        )
        tv_h.append(tv_homotopy)
        tv_c.append(tv_cold)
        steps.append(max_step)
        print(
            f"  seed {seed}:  K* {k_star:2d}  homotopy TV {tv_homotopy:8.4f}"
            f"   cold TV {tv_cold:8.4f}   max path step {max_step:6.3f}"
        )
    print(
        f"T2  mean TV:  homotopy {np.mean(tv_h):.4f} +- {np.std(tv_h):.4f}"
        f"   cold {np.mean(tv_c):.4f} +- {np.std(tv_c):.4f}"
    )
    print(f"T3  max coefficient step along the path: {max(steps):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=2.2)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    test_rao_blackwell(args.d, seed=0)
    test_homotopy(args.d, seeds=args.seeds)


if __name__ == "__main__":
    main()
