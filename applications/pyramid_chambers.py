"""P-E: the pyramid against the chambers.

A deliberately rugged one-site problem: Normal baseline, wide bimodal
mixture, fixed high capacity K = 12, N = 5000.  Four fits of the same
data per seed:

  cold        fit_complex_amplitude from the vacuum at full K;
  continued   the degree continuation (capacity homotopy);
  homotopy    the full-capacity time homotopy (14 slices, predictor-
              corrector on the deterministic path);
  pyramid     the time homotopy with the active set restricted to the
              corner n <= log(1/theta)/t -- each coefficient enters in
              its linear window, coupling switches on progressively.

Registered predictions:

  P-E1  the cold fit strands on a nonzero fraction of seeds (TV at
        least three times the seed's best); the pyramid reaches the
        good optimum on every seed -- the comparison lives in the worst
        case, not the mean;
  P-E2  pyramid and full-capacity homotopy agree in quality, the
        pyramid at lower cost (its early problems are small);
  P-E3  pyramid coefficients are born near their final values (the
        linear-window mechanism), as in the dynamic-degree study.
"""

from __future__ import annotations

import argparse
import time
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
K = 12
STEPS = 14
THETA = 0.05


def sample_mixture(d: float, size: int, rng: Any) -> np.ndarray:
    signs = np.where(rng.random(size) < 0.5, 1.0, -1.0)
    return signs * d + rng.standard_normal(size)


def truth_law(d: float) -> np.ndarray:
    return (
        0.5
        * (np.exp(-0.5 * (GRID - d) ** 2) + np.exp(-0.5 * (GRID + d) ** 2))
        / np.sqrt(2.0 * np.pi)
    )


def fitted_law(c: np.ndarray) -> np.ndarray:
    basis = np.asarray(Normal.basis(GRID, K, BASELINE), dtype=float)
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    law = ref * np.abs(basis @ c) ** 2
    return law / np.trapezoid(law, GRID)


def total_variation(truth: np.ndarray, law: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(truth - law), GRID))


def s_path() -> np.ndarray:
    return -np.log(np.linspace(np.exp(-3.0), 1.0, STEPS))


def cold_fit(phi, r0):
    return fit_complex_amplitude(phi, r0)["coefficients"]


def continued_fit(phi, r0):
    return continued_complex_fit(phi, r0)["complex"]["coefficients"]


def homotopy_fit(phi, r0):
    c = np.zeros(K + 1, dtype=complex)
    c[0] = 1.0
    s_prev = s_path()[0]
    for s in s_path():
        target = np.exp(-np.arange(2 * K + 1) * s) * r0
        predictor = c * np.exp(np.arange(K + 1) * (s_prev - s))
        predictor = predictor / np.linalg.norm(predictor)
        c = fit_complex_amplitude(phi, target, initial=predictor)["coefficients"]
        s_prev = float(s)
    return c


def pyramid_fit(phi, r0):
    c_active = None
    active_previous = None
    births: dict[int, float] = {}
    for s in s_path():
        cap = np.log(1.0 / THETA) / max(s, 1e-9)
        active = np.arange(K + 1)[np.arange(K + 1) <= cap]
        if len(active) < 2:
            active = np.arange(2)
        start = np.zeros(len(active), dtype=complex)
        start[0] = 1.0
        if c_active is not None:
            start[: len(active_previous)] = c_active
            start = start / np.linalg.norm(start)
        target = np.exp(-np.arange(2 * K + 1) * s) * r0
        phi_active = phi[:, active][:, :, active]
        c_active = fit_complex_amplitude(phi_active, target, initial=start)[
            "coefficients"
        ]
        for n in active:
            if n not in births and s > 0:
                births[n] = abs(c_active[n]) * np.exp(n * s)
        active_previous = active
    c = np.zeros(K + 1, dtype=complex)
    c[active_previous] = c_active
    return c, births


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=2.2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    truth = truth_law(args.d)
    phi = fitting_matrices(Normal, BASELINE, K)
    results = {name: [] for name in ("cold", "continued", "homotopy", "pyramid")}
    costs = {name: 0.0 for name in results}
    birth_ratios = []
    print(f"bimodal d = {args.d}, K = {K}, N = {args.n}, {args.seeds} seeds")
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        x = sample_mixture(args.d, args.n, rng)
        r0 = np.asarray(Normal.basis(x, 2 * K, BASELINE), dtype=float).mean(axis=0)
        row = {}
        for name, fitter in (
            ("cold", cold_fit),
            ("continued", continued_fit),
            ("homotopy", homotopy_fit),
        ):
            begin = time.perf_counter()
            c = fitter(phi, r0)
            costs[name] += time.perf_counter() - begin
            row[name] = total_variation(truth, fitted_law(c))
        begin = time.perf_counter()
        c, births = pyramid_fit(phi, r0)
        costs["pyramid"] += time.perf_counter() - begin
        row["pyramid"] = total_variation(truth, fitted_law(c))
        for n, value in births.items():
            final = abs(c[n])
            if final > 1e-3:
                birth_ratios.append(value / final)
        for name, value in row.items():
            results[name].append(value)
        print(
            f"  seed {seed:2d}:  "
            + "  ".join(f"{name} {row[name]:7.4f}" for name in results)
        )

    best = np.minimum.reduce([np.array(results[n]) for n in results])
    print("\nsummary (TV at t = 0):")
    for name in results:
        values = np.array(results[name])
        stranded = int(np.sum(values > 3.0 * best))
        print(
            f"  {name:10s}  mean {values.mean():7.4f}   worst {values.max():7.4f}"
            f"   stranded {stranded}/{len(values)}"
            f"   cost {costs[name]:6.1f}s"
        )
    if birth_ratios:
        print(
            f"P-E3  birth/final coefficient ratio: median"
            f" {np.median(birth_ratios):.3f}"
            f" (IQR {np.percentile(birth_ratios, 25):.3f}"
            f"-{np.percentile(birth_ratios, 75):.3f})"
        )


if __name__ == "__main__":
    main()
