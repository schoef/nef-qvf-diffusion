"""The deterministic target path at two sites.

Pair moments obey E[phi_j(X_t) phi_k(Y_t) | X_0, Y_0]
= e^{-(j+k)t} phi_j(X_0) phi_k(Y_0), so the decayed data statistics
dominate the noised-slice estimators channel by channel, and the whole
slice tower of pair targets is a deterministic image of one pass over
the data.  The study measures, on the two-site hidden-Markov pair:

  T1  RMSE of noised vs deterministic pair-coefficient estimators;
  T2  homotopy chain on the deterministic path vs the noised-slice warm
      chain and the cold fit: pair TV at t = 0 and latent factorisation;
  T3  smoothness of the coefficient path.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
)
from applications.two_site_diffusion import (
    build_pair_target,
    empirical_pair_coefficients,
    exact_pair_coefficients,
    factorise_pair_moments,
    pair_fitting_matrices,
    pair_total_variation,
)
from nefqvf import Poisson  # noqa: F401  (family fixed by the target)

GRID = np.arange(0, 60)


def noised(sample: np.ndarray, t: float, target: Any, rng: Any) -> np.ndarray:
    family, baseline = target.family, target.baseline
    return np.column_stack(
        [
            family.one_shot_sample(sample[:, i], t, params=baseline, rng=rng)
            for i in range(2)
        ]
    )


# ------------------------------------------------------------------ T1 --
def test_rao_blackwell(target, sample, k_max: int, rng: Any, redraws: int = 60):
    family, baseline = target.family, target.baseline
    r0 = empirical_pair_coefficients(family, baseline, sample, k_max)
    print("T1  RMSE of pair-coefficient estimators vs exact:")
    for t in (0.25, 0.5, 1.0):
        exact = exact_pair_coefficients(target, t, k_max)
        decay = np.exp(-(np.add.outer(np.arange(k_max + 1), np.arange(k_max + 1))) * t)
        deterministic = decay * r0
        draws = np.stack(
            [
                empirical_pair_coefficients(
                    family, baseline, noised(sample, t, target, rng), k_max
                )
                for _ in range(redraws)
            ]
        )
        for j, k in ((1, 1), (2, 2), (3, 3)):
            rmse_noised = float(np.sqrt(np.mean((draws[:, j, k] - exact[j, k]) ** 2)))
            rmse_det = abs(deterministic[j, k] - exact[j, k])
            print(
                f"      t {t:4.2f} mode ({j},{k}):  noised {rmse_noised:9.6f}"
                f"   deterministic {rmse_det:9.6f}"
                f"   ratio {rmse_noised / max(rmse_det, 1e-12):6.2f}"
            )


# ------------------------------------------------------------------ T2 --
def homotopy_chain(
    r0: np.ndarray, stack, degrees, n_steps: int = 14, coeff_degrees=None
):
    if coeff_degrees is None:
        coeff_degrees = degrees_of_coefficients(stack.shape[1])
    s_path = -np.log(np.linspace(np.exp(-3.0), 1.0, n_steps))
    c = np.zeros(stack.shape[1], dtype=complex)
    c[0] = 1.0
    s_prev, max_step = s_path[0], 0.0
    for s in s_path:
        target_vec = np.exp(-degrees * s) * r0.reshape(-1)
        predictor = c * np.exp((s_prev - s) * coeff_degrees)
        predictor = predictor / np.linalg.norm(predictor)
        fit = fit_complex_amplitude(stack, target_vec, initial=predictor)
        max_step = max(
            max_step, float(np.linalg.norm(np.abs(fit["coefficients"]) - np.abs(c)))
        )
        c = fit["coefficients"]
        s_prev = float(s)
    return c, max_step


def degrees_of_coefficients(size: int) -> np.ndarray:
    k = int(np.sqrt(size)) - 1
    j, m = np.divmod(np.arange(size), k + 1)
    return (j + m).astype(float)


# ------------------------------------------------------------------ T4 --
def held_out_nll(c_full: np.ndarray, family, baseline, k_max: int, x_val) -> float:
    basis1 = np.asarray(family.basis(x_val[:, 0], k_max, baseline), dtype=float)
    basis2 = np.asarray(family.basis(x_val[:, 1], k_max, baseline), dtype=float)
    h = np.einsum("ij,jk,ik->i", basis1, c_full.reshape(k_max + 1, k_max + 1), basis2)
    return -float(np.mean(np.log(np.abs(h) ** 2 + 1e-300)))


def test_capacity(name: str, separation: float, epsilon: float, seeds: int, k_max: int):
    """Total pair degree j + k <= D chosen by held-out likelihood, for the
    homotopy chain and the cold continued fit alike."""

    print("T4  capacity by held-out NLL (total degree j + k <= D):")
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        target = build_pair_target(name, separation, epsilon)
        family, baseline = target.family, target.baseline
        sample = np.asarray(target.sample(30_000, rng))
        train, val = sample[:24_000], sample[24_000:]
        r0_vec = empirical_pair_coefficients(
            family, baseline, train, 2 * k_max
        ).reshape(-1)
        phi = fitting_matrices(family, baseline, k_max)
        stack, degrees = pair_fitting_matrices(phi)
        coeff_degrees = degrees_of_coefficients(stack.shape[1])

        def scatter(c, idx):
            full = np.zeros(stack.shape[1], dtype=complex)
            full[idx] = c
            return full

        best = {}
        for label in ("homotopy", "cold"):
            chosen = None
            for cap in (2, 3, 4, 6, 9, 12):
                idx = np.where(coeff_degrees <= cap)[0]
                stack_r = stack[:, idx][:, :, idx]
                if label == "homotopy":
                    c_r, _ = homotopy_chain(
                        r0_vec, stack_r, degrees, coeff_degrees=coeff_degrees[idx]
                    )
                else:
                    c_r = continued_complex_fit(stack_r, r0_vec)["complex"][
                        "coefficients"
                    ]
                c_full = scatter(c_r, idx)
                nll = held_out_nll(c_full, family, baseline, k_max, val)
                if chosen is None or nll < chosen[1]:
                    chosen = (c_full, nll, cap)
            best[label] = chosen
        tv = {
            label: pair_total_variation(
                target, 0.0, best[label][0].reshape(k_max + 1, k_max + 1), GRID
            )
            for label in best
        }
        rows.append((tv["homotopy"], tv["cold"]))
        print(
            f"  seed {seed}:  homotopy TV {tv['homotopy']:8.4f} (D={best['homotopy'][2]})"
            f"   cold TV {tv['cold']:8.4f} (D={best['cold'][2]})"
        )
    arr = np.array(rows)
    print(
        f"T4  mean TV:  homotopy {arr[:, 0].mean():.4f}   cold {arr[:, 1].mean():.4f}"
    )


def noised_chain(target, sample, stack, k_max: int, rng: Any, n_steps: int = 14):
    family, baseline = target.family, target.baseline
    s_path = -np.log(np.linspace(np.exp(-3.0), 1.0, n_steps))
    c = np.zeros(stack.shape[1], dtype=complex)
    c[0] = 1.0
    for s in s_path:
        xt = noised(sample, float(s), target, rng) if s > 0 else sample
        target_vec = empirical_pair_coefficients(
            family, baseline, xt, 2 * k_max
        ).reshape(-1)
        fit = fit_complex_amplitude(stack, target_vec, initial=c)
        c = fit["coefficients"]
    return c


def test_chains(name: str, separation: float, epsilon: float, seeds: int, k_max: int):
    print("T2/T3  chains at two sites:")
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        target = build_pair_target(name, separation, epsilon)
        family, baseline = target.family, target.baseline
        sample = np.asarray(target.sample(30_000, rng))
        r0 = empirical_pair_coefficients(family, baseline, sample, 2 * k_max)
        phi = fitting_matrices(family, baseline, k_max)
        stack, degrees = pair_fitting_matrices(phi)

        c_h, max_step = homotopy_chain(r0, stack, degrees)
        c_n = noised_chain(target, sample, stack, k_max, rng)
        cold = continued_complex_fit(stack, r0.reshape(-1))["complex"]["coefficients"]

        def score(c):
            matrix = c.reshape(k_max + 1, k_max + 1)
            tv = pair_total_variation(target, 0.0, matrix, GRID)
            from applications.amplitude_fit_complex import ratio_coefficients_complex

            fitted = ratio_coefficients_complex(c, stack)
            factor = factorise_pair_moments(
                fitted.reshape(2 * k_max + 1, 2 * k_max + 1), family, baseline, 1.2
            )
            return tv, factor["shifts"], factor["epsilon"]

        tv_h, shifts_h, eps_h = score(c_h)
        tv_n, _, _ = score(c_n)
        tv_c, _, _ = score(cold)
        rows.append((tv_h, tv_n, tv_c))
        print(
            f"  seed {seed}:  homotopy {tv_h:8.4f}  noised chain {tv_n:8.4f}"
            f"  cold {tv_c:8.4f}   shifts {shifts_h[0]:+.3f}/{shifts_h[1]:+.3f}"
            f"  eps {eps_h:.3f}   max step {max_step:.3f}"
        )
    arr = np.array(rows)
    print(
        f"T2  mean TV:  homotopy {arr[:, 0].mean():.4f}"
        f"   noised chain {arr[:, 1].mean():.4f}   cold {arr[:, 2].mean():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--separation", type=float, default=0.57)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--skip-t1", action="store_true")
    parser.add_argument("--skip-chains", action="store_true")
    args = parser.parse_args()

    if not args.skip_t1:
        rng = np.random.default_rng(0)
        target = build_pair_target(args.family, args.separation, args.epsilon)
        sample = np.asarray(target.sample(30_000, rng))
        test_rao_blackwell(target, sample, args.kmax, rng)
    if not args.skip_chains:
        test_chains(args.family, args.separation, args.epsilon, args.seeds, args.kmax)
    test_capacity(args.family, args.separation, args.epsilon, args.seeds, args.kmax)


if __name__ == "__main__":
    main()
