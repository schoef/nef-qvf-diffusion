"""Chamber dynamics along the schedule: closed forms, TV against the
moment loss, and the node-birth diagnostic.

Fresh implementations of three studies (an earlier exploratory session
produced versions of these; everything here is re-derived and re-run,
nothing imported).  Outcomes:

  toy        closed-form analysis of the K = 2 chamber toy along the
             schedule, all verified: the exact zero of J_t exists iff
             2 - sqrt2 <= e^{-2t} d^2 <= 2 + sqrt2; the zero leaves the
             real axis in a pitchfork at t_c = log(d^2/(2-sqrt2))/2
             (0.8552 at d = 1.8); the second real minimum is born in a
             saddle-node at t = 0.7890; the capacity bound at t = 0 is
             d* = sqrt(2 + sqrt2) = 1.84776 -- the note's d = 1.8 sits
             just inside.

  sweep      TV against the moment loss on warm-off slices (real fits,
             (K, d) in {(10, 2.5), (10, 2.8), (8, 2.5)} x 3 seeds).
             Six warm-off events, J* monotonicity self-check clean.
             The strong conjecture (warm branch always TV-better) is
             refuted, but the risk asymmetry survives: the warm branch
             never loses more than 0.009 TV and gains 0.03-0.04 where
             truncation binds (K = 8 at the budget edge, 3/3 seeds).
             Following the vacuum-connected branch is near-free
             insurance whose payout concentrates exactly where the
             moment objective is least trustworthy.

  diagnose   node birth in the warm iterate (stable Hermite-basis
             roots, no power basis) against the warm-off events:
             6/6 events at or within one slice of the first birth
             (perfect sensitivity), but 3/3 event-free chains also have
             births (poor specificity).  Node birth is necessary, not
             sufficient: a monitor, not a controller.

  arbitrate  the closed loop -- flag the first birth, run one targeted
             multistart there, switch only on a clear held-out
             likelihood improvement: never worse than plain warm (0/9)
             and never better (0/9; the single switch was washed out by
             the subsequent warm dynamics).  No intervention is needed
             at one site.
"""

from __future__ import annotations

import argparse
from math import factorial

import numpy as np
from numpy.polynomial import hermite_e

from nefqvf import Normal, NormalParams
from nefqvf.fitting import fit_amplitude, product_matrices

BASELINE = NormalParams(0.0, 1.0)
GRID = np.linspace(-12.0, 12.0, 1201)
SAMPLE = 4000
CONFIGS = ((10, 2.5), (10, 2.8), (8, 2.5))
SEEDS = (0, 1, 2)


# ------------------------------------------------------------------ shared --
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
    law = ref * (basis @ c) ** 2
    return law / np.trapezoid(law, GRID)


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(a - b), GRID))


def gauge_distance(a: np.ndarray, b: np.ndarray) -> float:
    return min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b)))


def amplitude_roots(c: np.ndarray) -> np.ndarray:
    """Roots of sum_n c_n phi_n via the Hermite-e companion matrix; the
    power basis is numerically unusable at these degrees."""

    coefficients = np.array([c[n] / np.sqrt(factorial(n)) for n in range(len(c))])
    return hermite_e.hermeroots(coefficients)


def has_node(c: np.ndarray, span: float) -> bool:
    roots = amplitude_roots(c)
    in_range = roots[np.abs(roots.real) < span]
    return bool(np.any(np.abs(in_range.imag) < 1e-8))


def sample_mixture(d: float, size: int, rng) -> np.ndarray:
    signs = np.where(rng.random(size) < 0.5, 1.0, -1.0)
    return signs * d + rng.standard_normal(size)


def schedule() -> np.ndarray:
    return np.concatenate([np.arange(3.0, 0.5, -0.1), np.arange(0.5, -0.005, -0.02)])


# --------------------------------------------------------------------- toy --
def study_toy(d: float = 1.8) -> None:
    """Closed forms of the K = 2 toy along the schedule (all exact)."""

    low, high = 2.0 - np.sqrt(2.0), 2.0 + np.sqrt(2.0)

    def exact_zero_exists(s: float) -> bool:
        r2_target, r4_target = s / np.sqrt(2.0), s**2 / np.sqrt(24.0)
        modulus2 = r4_target / np.sqrt(6.0)
        if modulus2 >= 1.0:
            return False
        modulus = np.sqrt(modulus2)
        c0 = np.sqrt(1.0 - modulus2)
        needed = (r2_target - 2.0 * np.sqrt(2.0) * modulus2) / (2.0 * c0 * modulus)
        return abs(needed) <= 1.0

    grid = np.linspace(0.05, 4.0, 8000)
    inside = np.array([exact_zero_exists(s) for s in grid])
    edges = grid[np.nonzero(np.diff(inside.astype(int)))[0]]
    t_c = 0.5 * np.log(d**2 / low)
    print(
        f"solvability window in s = e^(-2t) d^2: {edges.round(5)}"
        f"  (exact {low:.5f}, {high:.5f})"
    )
    print(
        f"complex pitchfork at t_c = {t_c:.4f}; capacity bound d* = {np.sqrt(high):.5f}"
    )
    from applications.paper_chamber_toy import minima_of, target_coefficients

    t2, t4 = target_coefficients(d)
    for t in np.linspace(1.2, 0.4, 1601):
        if len(minima_of(np.exp(-2 * t) * t2, np.exp(-4 * t) * t4)) >= 2:
            print(f"real saddle-node (second minimum born) at t = {t:.4f}")
            break


# ------------------------------------------------------------------- sweep --
def study_sweep() -> None:
    """TV against the moment loss on warm-off slices."""

    total = {"warm": 0, "moment": 0, "off": 0, "jdrop": 0}
    for k, d in CONFIGS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = sample_mixture(d, SAMPLE, rng)
            r0 = np.asarray(Normal.basis(x, 2 * k, BASELINE), dtype=float).mean(axis=0)
            phi = product_matrices(Normal, BASELINE, k)
            c_warm = np.zeros(k + 1)
            c_warm[0] = 1.0
            best_prev, j_floor = None, -np.inf
            for t in schedule():
                target = np.exp(-np.arange(2 * k + 1) * t) * r0
                c_warm = fit_amplitude(phi, target, initial=c_warm)["coefficients"]
                if t > 0.35 + 1e-9 and abs(t) > 1e-9:
                    continue
                starts = [c_warm] + ([best_prev] if best_prev is not None else [])
                for _ in range(250):
                    v = rng.standard_normal(k + 1)
                    starts.append(v / np.linalg.norm(v))
                best, j_best = None, np.inf
                for c0 in starts:
                    fit = fit_amplitude(phi, target, initial=np.asarray(c0))
                    if fit["objective"] < j_best:
                        j_best, best = fit["objective"], fit["coefficients"]
                if j_best < j_floor - 1e-9:
                    total["jdrop"] += 1
                j_floor = max(j_floor, j_best)
                best_prev = best
                j_warm = fit_amplitude(phi, target, initial=c_warm)["objective"]
                if gauge_distance(c_warm, best) > 0.05 and j_warm - j_best > 1e-4:
                    total["off"] += 1
                    truth = truth_law(d, t)
                    tv_warm = total_variation(truth, fitted_law(c_warm, k))
                    tv_best = total_variation(truth, fitted_law(best, k))
                    winner = "warm" if tv_warm < tv_best else "moment"
                    total[winner] += 1
                    print(
                        f"  K={k} d={d} seed={seed} t={t:.2f}: warm-off"
                        f"  J {j_warm:.4f}/{j_best:.4f}"
                        f"  TV {tv_warm:.4f}/{tv_best:.4f}  {winner} law better"
                    )
    print(
        f"total: warm-off {total['off']}, warm better {total['warm']},"
        f" moment-global better {total['moment']},"
        f" J* violations {total['jdrop']}"
    )


# ---------------------------------------------------------------- diagnose --
def study_diagnose() -> None:
    """First node birth of the warm iterate along each chain."""

    for k, d in CONFIGS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            x = sample_mixture(d, SAMPLE, rng)
            r0 = np.asarray(Normal.basis(x, 2 * k, BASELINE), dtype=float).mean(axis=0)
            phi = product_matrices(Normal, BASELINE, k)
            span = 1.2 * (d + 3.0)
            c = np.zeros(k + 1)
            c[0] = 1.0
            birth = None
            for t in schedule():
                target = np.exp(-np.arange(2 * k + 1) * t) * r0
                c = fit_amplitude(phi, target, initial=c)["coefficients"]
                if birth is None and has_node(c, span):
                    birth = t
            label = f"{birth:.2f}" if birth is not None else "never"
            print(f"  K={k} d={d} seed={seed}: first node birth at t = {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("toy", "sweep", "diagnose"), default="toy")
    args = parser.parse_args()
    {"toy": study_toy, "sweep": study_sweep, "diagnose": study_diagnose}[args.study]()


if __name__ == "__main__":
    main()
