"""Rung 3 of the benchmark ladder: combinatorial chamber growth at one site.

The question: when the number of spurious basins of the amplitude fit
grows combinatorially with the number of mixture components, does the
single warm path of the diffused amplitude fit still find the global
basin at a cost that does not?

Setup (registered before running).  Equal-weight mixture of m
unit-variance normal components at equally spaced means, spacing
s = 3.5 component standard deviations, m = 2..6.  The sample
(N = 20000) is standardized by the population mean and standard
deviation, so the components become narrow bumps in the N(0,1)
baseline frame; the slice laws remain exact (normal components stay
normal under the semigroup, means contract as e^{-t}, variances relax
to one).  Degree K(m) = 2m + 10, three seeds, 400 random cold starts,
schedule t: 3 -> 0 as in the other chamber studies, diagonal weights
= inverse sampling variance of the coefficient estimators at t = 0,
held fixed along the path.

Predictions registered before running:

  P1  the number of distinct basins B(m) of the cold weighted fit at
      t = 0 grows super-linearly in m, and the cold success
      probability (a random start lands in the global basin) decays
      at least geometrically in m;
  P2  the warm path (one continuation from the vacuum, finished by
      the likelihood polish) lands in the global basin for every m,
      at a per-m cost that grows with K but not with B(m);
  P3  node births along the warm path are monotone in decreasing t
      (nodes appear, never disappear), ending at the terminal fit's
      node count;
  P4  if P2 fails anywhere, it fails at the largest m, where the
      walls crowd;
  P5  (added after an m = 2 smoke test, before the full run) the
      likelihood polish alone rescues shallow near-basins but fails
      from the junk basins random starts find, with failures
      appearing by m = 4.

Outcomes (run of 2026-09-02):

  P1  CONFIRMED.  Basins found: ~45 (m = 2), ~180 (m = 3), ~280
      (m = 4..6, saturating 400 starts); cold success collapses from
      ~0.10 at m = 2..3 to 0.003..0.025 at m = 4..6.
  P2  REFUTED at the moment stage: at capacity-adequate degree
      (m = 4, K = 34) the warm quartic endpoint strands in a wrong
      basin (J 0.06 against the multistart best 4e-4, TV 0.16
      against 0.013, gauge distance 1.19); the pyramid variant
      strands the same way.  The pipeline endpoint is nevertheless
      correct everywhere because the polish rescues it.
  P3  UNTESTED (vacuous): at spacing 3.5 the standardized law stays
      well above zero between bumps and the warm iterate never
      develops a real node.  A wide-spacing variant is needed.
  P4  CONFIRMED in direction: the strand appears at m >= 4.
  P5  REFUTED decisively: the polish rescued every probed basin --
      360/360 across all runs, including the 12 worst basins per
      configuration.  At one site the complex likelihood behaves
      as if it had a single basin, operationally and not only
      formally (the nodal walls have real codimension two).

Verdict: at one site the chamber problem of the moment objective is
real (P1) and the diffusion continuation neither solves it at large
m (P2) nor is needed for it (P5) -- complex likelihood descent from
any start reaches the global optimum.  The continuation's
reachability value must therefore be defended where the polish is
not global: the multisite sweeps, whose likelihood updates are
local per bond, and moment-only settings.
"""

from __future__ import annotations

import argparse
import json
import time
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import hermite_e

from applications.amplitude_fit_complex import fit_complex_amplitude, fitting_matrices
from nefqvf import Normal, NormalParams

BASELINE = NormalParams(0.0, 1.0)
GRID = np.linspace(-6.0, 6.0, 2401)
SPACING = 3.5
M_VALUES = (2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
SAMPLE = 20000
STARTS = 400
NODE_SPAN = 3.0
ARTIFACTS = Path("artifacts") / "mixture-chambers"


# ------------------------------------------------------------------- truth --
def component_frame(m: int) -> tuple[np.ndarray, float]:
    """Standardized component means and component standard deviation."""

    means = SPACING * (np.arange(m) - (m - 1) / 2.0)
    sigma = np.sqrt(1.0 + float(np.mean(means**2)))
    return means / sigma, 1.0 / sigma


def truth_law(m: int, t: float) -> np.ndarray:
    """Exact slice law in the standardized frame at semigroup time t."""

    means, sd = component_frame(m)
    mu = np.exp(-t) * means
    var = 1.0 + np.exp(-2.0 * t) * (sd**2 - 1.0)
    law = np.zeros_like(GRID)
    for mean in mu:
        law += np.exp(-0.5 * (GRID - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
    return law / m


def sample_mixture(m: int, size: int, rng) -> np.ndarray:
    means, sd = component_frame(m)
    picks = rng.integers(0, m, size)
    return means[picks] + sd * rng.standard_normal(size)


# --------------------------------------------------------------------- fit --
def fitted_law(c: np.ndarray, k: int) -> np.ndarray:
    basis = np.asarray(Normal.basis(GRID, k, BASELINE), dtype=float)
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    law = ref * np.abs(basis @ c) ** 2
    return law / np.trapezoid(law, GRID)


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.trapezoid(np.abs(a - b), GRID))


def gauge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance on the sphere modulo global phase and conjugation."""

    direct = np.sqrt(max(0.0, 2.0 - 2.0 * abs(np.vdot(a, b))))
    mirror = np.sqrt(max(0.0, 2.0 - 2.0 * abs(np.vdot(a, np.conj(b)))))
    return min(float(direct), float(mirror))


def channel_weight(prods: np.ndarray) -> np.ndarray:
    """Diagonal inverse sampling-variance weights, excluding the R_0 row."""

    sd = prods.std(axis=0) / np.sqrt(len(prods))
    inverse = 1.0 / np.maximum(sd, sd[sd > 0].min()) ** 2
    return np.diag((inverse / inverse.max())[1:])


def real_node_count(c: np.ndarray) -> int:
    coefficients = np.array(
        [c[n] / np.sqrt(float(factorial(n))) for n in range(len(c))]
    )
    roots = hermite_e.hermeroots(coefficients)
    keep = np.abs(roots.real) < NODE_SPAN
    return int(np.sum(np.abs(roots[keep].imag) < 1e-6))


def schedule() -> np.ndarray:
    return np.concatenate([np.arange(3.0, 0.5, -0.1), np.arange(0.5, -0.005, -0.02)])


def likelihood_polish(c0: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """L-BFGS descent of the exact one-site likelihood from a complex start."""

    from scipy.optimize import minimize

    basis = np.asarray(Normal.basis(x, k, BASELINE), dtype=float)
    d = k + 1

    def objective(v):
        c = v[:d] + 1j * v[d:]
        n2 = float(np.sum(v * v))
        z = basis @ c
        a2 = np.abs(z) ** 2 + 1e-300
        value = -float(np.mean(np.log(a2))) + np.log(n2)
        grad_re = -2.0 / len(z) * (basis.T @ (z.real / a2))
        grad_im = -2.0 / len(z) * (basis.T @ (z.imag / a2))
        return value, np.concatenate([grad_re, grad_im]) + 2.0 * v / n2

    start = np.concatenate([np.asarray(c0, dtype=complex).real, np.asarray(c0).imag])
    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 600, "ftol": 1e-13},
    )
    polished = result.x[:d] + 1j * result.x[d:]
    return polished / np.linalg.norm(polished)


def held_out_nll(c: np.ndarray, x: np.ndarray, k: int) -> float:
    basis = np.asarray(Normal.basis(x, k, BASELINE), dtype=float)
    ref = np.log(np.asarray(Normal.prob(x, BASELINE), dtype=float))
    return -float(np.mean(ref + np.log(np.abs(basis @ c) ** 2 + 1e-300)))


# ------------------------------------------------------------------ studies --
def warm_path(phi, r0, weight, k):
    """The diffused amplitude fit: one continuation from the vacuum."""

    c = np.zeros(k + 1, dtype=complex)
    c[0] = 1.0
    node_trace = []
    for t in schedule():
        target = np.exp(-np.arange(2 * k + 1) * t) * r0
        c = fit_complex_amplitude(phi, target, initial=c, weight=weight)[
            "coefficients"
        ]
        node_trace.append((float(t), real_node_count(c)))
    return c, node_trace


def run(m_values=M_VALUES, seeds=SEEDS, starts=STARTS, store: Path | None = None) -> dict:
    results = {}
    for m in m_values:
        k = 2 * m + 10
        phi = fitting_matrices(Normal, BASELINE, k)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            x = sample_mixture(m, SAMPLE, rng)
            x_test = sample_mixture(m, SAMPLE, rng)
            prods = np.asarray(Normal.basis(x, 2 * k, BASELINE), dtype=float)
            r0 = prods.mean(axis=0)
            weight = channel_weight(prods)
            truth = truth_law(m, 0.0)

            # cold multistart landscape at t = 0
            begin = time.perf_counter()
            basins: list[dict] = []
            memberships = []
            for _ in range(starts):
                v = rng.standard_normal(k + 1) + 1j * rng.standard_normal(k + 1)
                fit = fit_complex_amplitude(phi, r0, initial=v, weight=weight)
                c, j = fit["coefficients"], fit["objective"]
                for index, basin in enumerate(basins):
                    if gauge_distance(c, basin["c"]) < 0.05:
                        memberships.append(index)
                        if j < basin["objective"]:
                            basin.update(c=c, objective=j)
                        break
                else:
                    memberships.append(len(basins))
                    basins.append(
                        {"c": c, "objective": j, "tv": total_variation(truth, fitted_law(c, k))}
                    )
            cold_cost = time.perf_counter() - begin
            for basin in basins:
                basin["tv"] = total_variation(truth, fitted_law(basin["c"], k))
            counts = np.bincount(memberships, minlength=len(basins))
            best_index = int(np.argmin([b["objective"] for b in basins]))
            modal_index = int(np.argmax(counts))
            success = float(counts[best_index]) / starts

            # the warm path
            begin = time.perf_counter()
            c_warm, node_trace = warm_path(phi, r0, weight, k)
            warm_cost = time.perf_counter() - begin
            j_warm = fit_complex_amplitude(phi, r0, initial=c_warm, weight=weight)[
                "objective"
            ]
            stranded = (
                gauge_distance(c_warm, basins[best_index]["c"]) > 0.05
                and j_warm - basins[best_index]["objective"] > 1e-6
            )

            # the polish, from the warm end and from the stranded modal basin
            c_polished = likelihood_polish(c_warm, x, k)
            c_modal_polished = likelihood_polish(basins[modal_index]["c"], x, k)

            # rescue probes: does the polish alone recover the most-visited
            # cold basins?  If yes, the chambers of the quartic loss are
            # harmless at one site; if no, the warm path does real work.
            objectives = np.array([b["objective"] for b in basins])
            visited = list(np.argsort(counts)[::-1][:12])
            worst = list(np.argsort(objectives)[::-1][:12])
            rescue, rescue_worst = [], []
            for bucket, indices in ((rescue, visited), (rescue_worst, worst)):
                for index in indices:
                    c_r = likelihood_polish(basins[index]["c"], x, k)
                    bucket.append(
                        {
                            "share": float(counts[index]) / starts,
                            "objective": basins[index]["objective"],
                            "tv_raw": basins[index]["tv"],
                            "tv_polished": total_variation(truth, fitted_law(c_r, k)),
                        }
                    )

            births = [n for _, n in node_trace]
            monotone = all(b <= a for a, b in zip(births[1:], births))

            results[f"m{m}s{seed}"] = {
                "m": m,
                "seed": seed,
                "k": k,
                "basin_count": len(basins),
                "success": success,
                "modal_share": float(counts[modal_index]) / starts,
                "tv_best": basins[best_index]["tv"],
                "tv_modal": basins[modal_index]["tv"],
                "tv_median_start": float(
                    np.median([basins[i]["tv"] for i in memberships])
                ),
                "tv_warm": total_variation(truth, fitted_law(c_warm, k)),
                "tv_warm_polished": total_variation(truth, fitted_law(c_polished, k)),
                "tv_modal_polished": total_variation(
                    truth, fitted_law(c_modal_polished, k)
                ),
                "nll_warm_polished": held_out_nll(c_polished, x_test, k),
                "nll_modal_polished": held_out_nll(c_modal_polished, x_test, k),
                "stranded": bool(stranded),
                "rescue": rescue,
                "rescue_worst": rescue_worst,
                "basin_objectives": sorted(b["objective"] for b in basins),
                "monotone_births": bool(monotone),
                "terminal_nodes": births[-1],
                "node_trace": node_trace,
                "cold_cost": cold_cost,
                "warm_cost": warm_cost,
                "c_warm": [c_warm.real.tolist(), c_warm.imag.tolist()],
                "c_modal": [
                    basins[modal_index]["c"].real.tolist(),
                    basins[modal_index]["c"].imag.tolist(),
                ],
            }
            row = results[f"m{m}s{seed}"]
            print(
                f"m={m} K={k} seed={seed}: basins {row['basin_count']:3d}"
                f"  success {row['success']:.3f}"
                f"  TV best/modal/warm {row['tv_best']:.4f}/{row['tv_modal']:.4f}/"
                f"{row['tv_warm']:.4f}"
                f"  stranded {row['stranded']}  births monotone {row['monotone_births']}"
                f"  polished warm/modal {row['tv_warm_polished']:.4f}/"
                f"{row['tv_modal_polished']:.4f}"
                f"  rescue fails {sum(1 for r in rescue if r['tv_polished'] > row['tv_warm_polished'] + 0.01)}/12"
                f" worst {sum(1 for r in rescue_worst if r['tv_polished'] > row['tv_warm_polished'] + 0.01)}/12"
                f"  cost cold/warm {cold_cost:.1f}s/{warm_cost:.1f}s",
                flush=True,
            )
            if store is not None:
                store.parent.mkdir(parents=True, exist_ok=True)
                store.write_text(json.dumps(results))
    return results


def pyramid_path(phi, r0, weight, k, theta: float = 0.05):
    """The thresholded-activation variant: mode n joins once e^{-nt} > theta."""

    c_active, active_prev = None, None
    for t in schedule():
        cap = np.log(1.0 / theta) / max(t, 1e-9)
        active = np.arange(k + 1)[np.arange(k + 1) <= cap]
        if len(active) < 2:
            active = np.arange(2)
        start = np.zeros(len(active), dtype=complex)
        start[0] = 1.0
        if c_active is not None:
            start[: len(active_prev)] = c_active
            start = start / np.linalg.norm(start)
        target = np.exp(-np.arange(2 * k + 1) * t) * r0
        phi_active = phi[:, active][:, :, active]
        c_active = fit_complex_amplitude(phi_active, target, initial=start, weight=weight)[
            "coefficients"
        ]
        active_prev = active
    c = np.zeros(k + 1, dtype=complex)
    c[active_prev] = c_active
    return c


def capacity_scan(store: Path | None = None) -> dict:
    """Amendment after the main run: at the registered K(m) = 2m + 10 the
    capacity floor masks the landscape, so rerun m = 4..6 at larger degrees
    with the full comparison -- warm path, pyramid, multistart control
    (100 starts), and the polish from the warm end."""

    ladders = {4: (18, 26, 34), 5: (20, 30, 40), 6: (22, 34, 46)}
    results = {}
    for m, ks in ladders.items():
        for k in ks:
            rng = np.random.default_rng(0)
            x = sample_mixture(m, SAMPLE, rng)
            truth = truth_law(m, 0.0)
            phi = fitting_matrices(Normal, BASELINE, k)
            prods = np.asarray(Normal.basis(x, 2 * k, BASELINE), dtype=float)
            r0 = prods.mean(axis=0)
            weight = channel_weight(prods)

            begin = time.perf_counter()
            c_warm, _ = warm_path(phi, r0, weight, k)
            warm_cost = time.perf_counter() - begin
            c_pyramid = pyramid_path(phi, r0, weight, k)
            c_polished = likelihood_polish(c_warm, x, k)

            best, j_best = None, np.inf
            begin = time.perf_counter()
            for _ in range(100):
                v = rng.standard_normal(k + 1) + 1j * rng.standard_normal(k + 1)
                fit = fit_complex_amplitude(phi, r0, initial=v, weight=weight)
                if fit["objective"] < j_best:
                    j_best, best = fit["objective"], fit["coefficients"]
            multi_cost = time.perf_counter() - begin

            j_warm = fit_complex_amplitude(phi, r0, initial=c_warm, weight=weight)[
                "objective"
            ]
            j_pyramid = fit_complex_amplitude(phi, r0, initial=c_pyramid, weight=weight)[
                "objective"
            ]
            row = {
                "m": m,
                "k": k,
                "j_warm": j_warm,
                "j_pyramid": j_pyramid,
                "j_best": j_best,
                "tv_warm": total_variation(truth, fitted_law(c_warm, k)),
                "tv_pyramid": total_variation(truth, fitted_law(c_pyramid, k)),
                "tv_best": total_variation(truth, fitted_law(best, k)),
                "tv_warm_polished": total_variation(truth, fitted_law(c_polished, k)),
                "warm_cost": warm_cost,
                "multi_cost": multi_cost,
                "c_warm": [c_warm.real.tolist(), c_warm.imag.tolist()],
                "c_polished": [c_polished.real.tolist(), c_polished.imag.tolist()],
            }
            results[f"m{m}k{k}"] = row
            print(
                f"m={m} K={k:2d}: J warm/pyramid/best"
                f" {j_warm:.3g}/{j_pyramid:.3g}/{j_best:.3g}"
                f"  TV warm/pyramid/best/polished {row['tv_warm']:.4f}/"
                f"{row['tv_pyramid']:.4f}/{row['tv_best']:.4f}/"
                f"{row['tv_warm_polished']:.4f}"
                f"  cost warm/multi {warm_cost:.0f}s/{multi_cost:.0f}s",
                flush=True,
            )
            if store is not None:
                store.parent.mkdir(parents=True, exist_ok=True)
                store.write_text(json.dumps(results))
    return results


# ------------------------------------------------------------------- figure --
def make_figure(results: dict) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.4))
    ms = sorted({row["m"] for row in results.values()})

    def per_m(key):
        return [
            [row[key] for row in results.values() if row["m"] == m] for m in ms
        ]

    ax = axes[0]
    basin_runs = per_m("basin_count")
    success_runs = per_m("success")
    ax.plot(ms, [np.mean(v) for v in basin_runs], "o-", color="tab:blue", label="basins found")
    ax.set_ylabel("distinct basins", color="tab:blue")
    ax.set_xlabel("components $m$")
    twin = ax.twinx()
    twin.semilogy(
        ms, [np.mean(v) for v in success_runs], "s-", color="tab:red", label="cold success"
    )
    twin.set_ylabel("cold success probability", color="tab:red")
    ax.set_title("landscape growth")

    ax = axes[1]
    for key, label, marker in (
        ("tv_median_start", "cold, median start", "v"),
        ("tv_modal", "cold, modal basin", "s"),
        ("tv_best", "multistart best", "d"),
        ("tv_warm", "warm path", "o"),
    ):
        ax.semilogy(ms, [np.mean(v) for v in per_m(key)], marker + "-", label=label)
    ax.set_xlabel("components $m$")
    ax.set_ylabel("TV to truth at $t=0$")
    ax.legend(fontsize=7)
    ax.set_title("who reaches the global basin")

    capacity_store = ARTIFACTS / "capacity.json"
    capacity = (
        json.loads(capacity_store.read_text()) if capacity_store.exists() else {}
    )

    ax = axes[2]
    rows = sorted(
        (r for r in capacity.values() if r["m"] == 4), key=lambda r: r["k"]
    )
    if rows:
        ks = [r["k"] for r in rows]
        for key, label, marker, color in (
            ("tv_warm", "warm path", "o", "tab:red"),
            ("tv_pyramid", "pyramid", "^", "tab:orange"),
            ("tv_best", "multistart best", "d", "tab:blue"),
            ("tv_warm_polished", "warm + polish", "s", "tab:green"),
        ):
            ax.semilogy(ks, [r[key] for r in rows], marker + "-", color=color, label=label)
        ax.set_xlabel("degree $K$")
        ax.set_ylabel("TV to truth at $t=0$")
        ax.legend(fontsize=7)
        ax.set_title("the strand at $m=4$")

    ax = axes[3]
    row = capacity.get("m4k34")
    if row is not None:
        k = row["k"]
        truth = truth_law(4, 0.0)
        c_warm = np.array(row["c_warm"][0]) + 1j * np.array(row["c_warm"][1])
        c_pol = np.array(row["c_polished"][0]) + 1j * np.array(row["c_polished"][1])
        ax.plot(GRID, truth, color="0.3", lw=2.0, label="truth")
        ax.plot(
            GRID, fitted_law(c_warm, k), color="tab:red", lw=1.0, ls="--",
            label="warm path (stranded)",
        )
        ax.plot(
            GRID, fitted_law(c_pol, k), color="tab:green", lw=1.2, label="warm + polish"
        )
        ax.set_xlim(-3.2, 3.2)
        ax.set_xlabel("$x$")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax.set_title("example at $m=4$, $K=34$")

    figure.tight_layout()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(ARTIFACTS / "mixture-chambers.pdf")
    figure.savefig(ARTIFACTS / "mixture-chambers.png", dpi=140)
    print(f"figure: {ARTIFACTS / 'mixture-chambers.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", default="all", choices=("run", "figure", "capacity", "all")
    )
    args = parser.parse_args()
    store = ARTIFACTS / "results.json"
    if args.study in ("run", "all"):
        run(store=store)
    if args.study == "capacity":
        capacity_scan(store=ARTIFACTS / "capacity.json")
    if args.study in ("figure", "all"):
        results = json.loads(store.read_text())
        make_figure(results)


if __name__ == "__main__":
    main()
