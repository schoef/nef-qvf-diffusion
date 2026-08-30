"""Dynamic degree budget: learn from t = infinity backwards.

At t = infinity everything is dead except the baseline; as t decreases,
modes cross their sampling floors one at a time and are admitted to the
fit.  Each coefficient is therefore born in its linearised window
(R_k = 2 c_k + O(|c|^2) near the reference), warm-started thereafter,
and optionally frozen once mature.  The study measures, on a bimodal
one-site target:

  P1  final TV at t = 0 vs the cold fixed-degree fit, across seeds;
  P2  admission-time coefficient vs its final value;
  P3  measured admission slices vs the a-priori SNR crossings;
  P4  freezing mature modes vs full warm refits.
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
from nefqvf import Normal, NormalParams

BASELINE = NormalParams(0.0, 1.0)
GRID = np.linspace(-10.0, 10.0, 801)


def sample_mixture(d: float, size: int, rng: Any) -> np.ndarray:
    signs = np.where(rng.random(size) < 0.5, 1.0, -1.0)
    return signs * d + rng.standard_normal(size)


def slice_sample(x0: np.ndarray, t: float, rng: Any) -> np.ndarray:
    return np.exp(-t) * x0 + np.sqrt(1.0 - np.exp(-2.0 * t)) * rng.standard_normal(
        len(x0)
    )


def slice_truth(d: float, t: float) -> np.ndarray:
    m = d * np.exp(-t)
    return (
        0.5
        * (np.exp(-0.5 * (GRID - m) ** 2) + np.exp(-0.5 * (GRID + m) ** 2))
        / np.sqrt(2.0 * np.pi)
    )


def fitted_law(c: np.ndarray, k: int) -> np.ndarray:
    basis = np.asarray(Normal.basis(GRID, k, BASELINE), dtype=float)
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    h = basis @ c
    law = ref * np.abs(h) ** 2
    return law / np.trapezoid(law, GRID)


def total_variation(truth: np.ndarray, law: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(truth - law), GRID))


def coefficients_and_floors(x: np.ndarray, k_max: int):
    basis = np.asarray(Normal.basis(x, 2 * k_max, BASELINE), dtype=float)
    r_hat = basis.mean(axis=0)
    floors = basis.std(axis=0) / np.sqrt(len(x))
    return r_hat, floors


def frozen_fit(
    phi: np.ndarray, target: np.ndarray, c0: np.ndarray, free: np.ndarray
) -> np.ndarray:
    """LM on the unnormalised residuals (R_0 - 1 included), moving only
    the coordinates marked free; the frozen block keeps its values."""

    c = c0.astype(complex).copy()
    idx = np.where(free)[0]
    mu = 1e-3

    def residual(cv):
        return np.einsum("m,kmn,n->k", np.conj(cv), phi, cv).real - target

    value = 0.5 * float(residual(c) @ residual(c))
    for _ in range(120):
        r = residual(c)
        # complex Jacobian via real/imag split on the free block
        a, b = c.real, c.imag
        ja = 2.0 * np.einsum("kmn,n->km", phi, a)
        jb = 2.0 * np.einsum("kmn,n->km", phi, b)
        jac = np.concatenate([ja[:, idx], jb[:, idx]], axis=1)
        grad = jac.T @ r
        if np.linalg.norm(grad) < 1e-12:
            break
        hess = jac.T @ jac
        accepted = False
        for _ in range(30):
            step = np.linalg.solve(hess + mu * np.eye(hess.shape[0]), -grad)
            trial = c.copy()
            trial[idx] = trial[idx] + step[: len(idx)] + 1j * step[len(idx) :]
            tr = residual(trial)
            tval = 0.5 * float(tr @ tr)
            if tval < value:
                c, value = trial, tval
                mu = max(mu * 0.3, 1e-14)
                accepted = True
                break
            mu *= 3.0
        if not accepted:
            break
    return c / np.linalg.norm(c)


def run_chain(
    d: float,
    seed: int,
    *,
    k_max: int = 12,
    n_draws: int = 5000,
    n_slices: int = 16,
    z: float = 2.5,
    freeze: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x0 = sample_mixture(d, n_draws, rng)
    times = -np.log(np.linspace(np.exp(-3.0), 1.0, n_slices))

    c = np.array([1.0 + 0.0j])
    k_now = 0
    admission: dict[int, dict[str, Any]] = {}
    rows = []
    previous_supported: set = set()
    t_prev = times[0]
    for t in times:
        dt = float(t_prev - t)
        t_prev = float(t)
        xt = slice_sample(x0, float(t), rng) if t > 0 else x0
        r_hat, floors = coefficients_and_floors(xt, k_max)
        supported = {k for k in range(1, k_max + 1) if abs(r_hat[k]) > z * floors[k]}
        # persistence: admit a mode only if it was also supported last slice
        persistent = supported & previous_supported
        previous_supported = supported
        k_new = max(max(persistent, default=1), k_now)  # monotone
        phi = fitting_matrices(Normal, BASELINE, k_new)
        target = r_hat[: 2 * k_new + 1]

        init = np.zeros(k_new + 1, dtype=complex)
        growth = np.exp(np.arange(len(c)) * dt)
        init[: len(c)] = c * growth
        init = init / np.linalg.norm(init)
        if init[0] == 0:
            init[0] = 1.0
        newly = [k for k in range(k_now + 1, k_new + 1)]
        if freeze and k_now > 0:
            free = np.zeros(k_new + 1, dtype=bool)
            for k in newly:
                free[k] = True
            free[0] = True  # let the vacuum breathe for normalisation
            c = frozen_fit(phi, target, init / np.linalg.norm(init), free)
        else:
            fit = fit_complex_amplitude(phi, target, initial=init)
            c = fit["coefficients"]
        for k in newly:
            admission[k] = {
                "t": float(t),
                "c_at_birth": abs(c[k]) * np.exp(k * float(t)),
            }
        k_now = k_new
        tv = total_variation(slice_truth(d, float(t)), fitted_law(c, k_now))
        rows.append({"t": float(t), "k": k_now, "tv": tv})
    for k, entry in admission.items():
        entry["c_final"] = abs(c[k]) if k < len(c) else 0.0
    return {"rows": rows, "admission": admission, "c": c, "k": k_now}


def cold_fit(d: float, seed: int, *, k_max: int = 12, n_draws: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    x0 = sample_mixture(d, n_draws, rng)
    r_hat, _ = coefficients_and_floors(x0, k_max)
    phi = fitting_matrices(Normal, BASELINE, k_max)
    fit = continued_complex_fit(phi, r_hat)["complex"]
    return total_variation(slice_truth(d, 0.0), fitted_law(fit["coefficients"], k_max))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=2.2)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--kmax", type=int, default=12)
    args = parser.parse_args()

    print(f"bimodal target d = {args.d}, K_max = {args.kmax}")
    tv_cold, tv_warm, tv_frozen = [], [], []
    birth_ratios, admissions = [], {}
    for seed in range(args.seeds):
        cold = cold_fit(args.d, seed, k_max=args.kmax)
        warm = run_chain(args.d, seed, k_max=args.kmax, freeze=False)
        froz = run_chain(args.d, seed, k_max=args.kmax, freeze=True)
        tv_cold.append(cold)
        tv_warm.append(warm["rows"][-1]["tv"])
        tv_frozen.append(froz["rows"][-1]["tv"])
        for k, e in warm["admission"].items():
            if e["c_final"] > 1e-3:
                birth_ratios.append(e["c_at_birth"] / e["c_final"])
            admissions.setdefault(k, []).append(e["t"])
        print(
            f"  seed {seed}:  cold {cold:8.4f}   dynamic-K {warm['rows'][-1]['tv']:8.4f}"
            f"   frozen {froz['rows'][-1]['tv']:8.4f}   K path "
            + "->".join(str(r["k"]) for r in warm["rows"])
        )
    print(
        f"P1  TV at t=0:  cold {np.mean(tv_cold):.4f} +- {np.std(tv_cold):.4f}"
        f"   dynamic {np.mean(tv_warm):.4f} +- {np.std(tv_warm):.4f}"
    )
    print(
        f"P2  birth/final coefficient ratio: median "
        f"{np.median(birth_ratios):.3f} (IQR {np.percentile(birth_ratios, 25):.3f}"
        f"-{np.percentile(birth_ratios, 75):.3f})"
    )
    print("P3  admission slices (mean t per mode):")
    for k in sorted(admissions):
        print(f"      mode {k:2d}: t = {np.mean(admissions[k]):.3f}")
    print(
        f"P4  frozen vs full-refit TV: {np.mean(tv_frozen):.4f} vs {np.mean(tv_warm):.4f}"
    )


if __name__ == "__main__":
    main()
