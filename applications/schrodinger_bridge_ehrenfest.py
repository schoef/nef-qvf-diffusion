"""Schroedinger bridge on the Ehrenfest lattice: amplitude Sinkhorn, P1-P3.

Reference: Binomial(N=20, mean 10) with its Ehrenfest semigroup, exactly
diagonal in the orthonormal Krawtchouk basis, P_t phi_n = exp(-nt) phi_n.
Endpoints: two deliberately non-NEF laws p = pi |psi|^2 from complex
Krawtchouk amplitudes, each with one planted conjugate root pair whose
depth delta is the control knob -- p_0's zero ON the lattice (x = 7),
p_T's zero at a half-integer (x = 13.5).  Horizon T = 1 (kernel strictly
positive).  Density Sinkhorn f <- r_0/(P_T g), g <- r_T/(P_T f) is exact
on 21 states; amplitude Sinkhorn replaces P_T by S_T H = sqrt(P_T H^2).

Predictions registered before running (discussion of 2026-09-03):

  P1  (harness) exact density and amplitude Sinkhorn agree iterate by
      iterate and at the fixed point to 1e-12 relative; the bridge
      amplitude obeys h_t^2 = r_t on a t-grid to 1e-12; both endpoint
      residuals reach 1e-12.  Validates code; the equivalence is a
      theorem.
  P2  (as registered in the note discussion) at fixed accuracy the
      required degree of the rank-1 (real-root) potential diverges as
      delta -> 0 while the rank-2 (complex-root) degree stays flat --
      one degree per near-zero.
  R2  (refinement, registered after design review, BEFORE any run) on
      discrete support the signed real root evades the kink: a real
      polynomial may change sign near the planted zero (f = c^2 stays
      nonnegative), so the rank-1 representability error should FALL
      with delta (like delta^2 at the on-lattice zero, essentially
      exactly at the half-integer zero), and the true rank-1 deficit
      should appear as (a) chambered optimization -- multistart
      dependence and warm-start stranding -- and (b) at worst an
      algebraic-vs-exponential gap at adversarial intermediate depth
      delta ~ 1/K.  P2 and R2 are scored against each other on the
      delta x K error surface and on best-of-multistart against
      single-warm-start errors.
  P3  truncated rank-2 Sinkhorn -- each half-step a weighted Born fit
      at degree K (fit |F|^2 to r_0 with multiplicative weight P_T g;
      no pointwise division or root ever taken) -- converges
      geometrically to endpoint-residual floors within a factor ~3 of
      the offline truncation error of the exact potentials at the same
      degree, with no drift over late rounds; the intermediate bridge
      marginal is accurate at the floor scale.  The rank-1 contrast
      run is expected less stable (chamber hopping under the moving
      Sinkhorn targets).
  P4  (horizon, registered in the session restatement) representational
      cost blows up as T -> 0 and the cost curve has a valley at
      intermediate T.
  R4  (refinement, registered before running) the valley is expected
      only in the blow-up sense: the minimal rank-2 degree for 1e-6
      accuracy and the Sinkhorn iteration count should both fall
      monotonically with T, the degree approaching the endpoint
      amplitude's own degree (two) at large T -- no interior minimum
      in representational cost alone.

Outcome (run 2026-09-03, seed 0, artifacts/schrodinger-bridge/):
P1 passes at machine precision (iterate and fixed-point deviations
~1e-15, h_t^2 = r_t to 4e-16, bridge mass exact; 14 Sinkhorn
iterations at T = 1).  P2 as registered FAILS and R2 is confirmed:
the rank-1 error is flat-to-falling in delta -- the signed root
evades the kink (at delta = 0.01, K = 10 it reaches 2.5e-6) -- and
the true rank-1 deficit is the predicted pair: algebraic against
exponential decay in K (at K = 10, ~3e-6 against ~5e-9) plus
chambered optimization, visible as NON-monotone rank-1 error in the
nested degree ladder (3e-6 -> 4.5e-4 from K = 10 to 12 in several
cells: 16 multistarts stop finding the best chamber), while rank-2
descends smoothly to ~1e-10, delta-independent.  P3 confirmed for
rank 2: geometric convergence in ~4 rounds to a floor of 1.8e-6,
within 2.4x of the offline truncation error, zero late drift, bridge
midpoint error 1.2e-9.  The rank-1 contrast run is floor-limited
(1.6e-4; bridge midpoint error 2e-5, four orders worse than rank 2)
but STABLE at this configuration -- the predicted chamber-hopping
instability did not materialise at K = 8, delta = 0.03 with a
projection restart available each round.

Horizon outcome (run 2026-09-03, seed 0): P4's valley is refuted and
R4 confirmed on both axes -- the minimal rank-2 degree for 1e-6 falls
monotonically with T (13 at T = 0.1 down to 3 at T = 4, approaching
the endpoint amplitude's own degree) with no interior minimum, and
the Sinkhorn iteration count blows up sharply below T ~ 0.4 (2037
iterations at T = 0.4; the 20000 cap is hit at T <= 0.25, so those
degrees are measured on not-yet-converged potentials -- the monotone
trend stands with that caveat).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from nefqvf import Binomial, BinomialParams

N = 20
PARAMS = BinomialParams(mean=10.0, N=N)
X = np.arange(N + 1)
PI = np.asarray(Binomial.prob(X, PARAMS), dtype=float)
BASIS = np.asarray(Binomial.basis(X, N, PARAMS), dtype=float)  # (x, n)
EIGEN = np.arange(N + 1).astype(float)
HORIZON = 1.0
DELTAS = (1.0, 0.3, 0.1, 0.03, 0.01)
DEGREES = (2, 4, 6, 8, 10, 12, 14)
STARTS = 16
ARTIFACTS = Path("artifacts") / "schrodinger-bridge"


def kernel(t: float) -> np.ndarray:
    """Exact Ehrenfest transition matrix over time t, rows x, columns y.

    Entries at the 1e-15 scale can come out negative at small t; they are
    clipped and the rows renormalised.
    """

    matrix = (BASIS * np.exp(-EIGEN * t)[None, :]) @ (BASIS * PI[:, None]).T
    matrix = np.maximum(matrix, 0.0)
    return matrix / matrix.sum(axis=1, keepdims=True)


def endpoint_ratios(delta: float) -> tuple[np.ndarray, np.ndarray]:
    psi0 = (X - 7.0 - 1j * delta) * (1.0 + 0.05 * (X - 12.0))
    psit = (X - 13.5 - 1j * delta) * (1.0 - 0.04 * (X - 8.0))
    r0 = np.abs(psi0) ** 2 / np.sum(PI * np.abs(psi0) ** 2)
    rt = np.abs(psit) ** 2 / np.sum(PI * np.abs(psit) ** 2)
    return r0, rt


def rel_l2(values: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(PI * (values - reference) ** 2))
        / np.sqrt(np.sum(PI * reference**2))
    )


# ---------------------------------------------------------------- sinkhorn --
def density_sinkhorn(r0, rt, kt, iterations=2000, tol=1e-14, record=0):
    g = np.ones(N + 1)
    trail = []
    for index in range(iterations):
        f = r0 / (kt @ g)
        g_new = rt / (kt @ f)
        if index < record:
            trail.append((f.copy(), g_new.copy()))
        change = np.max(np.abs(g_new - g) / np.maximum(g, 1e-300))
        g = g_new
        if change < tol:
            break
    return r0 / (kt @ g), g, index + 1, trail


def amplitude_sinkhorn(h0, ht, kt, iterations=2000, tol=1e-14, record=0):
    def half_root(values):
        return np.sqrt(kt @ values**2)

    big_g = np.ones(N + 1)
    trail = []
    for index in range(iterations):
        big_f = h0 / half_root(big_g)
        g_new = ht / half_root(big_f)
        if index < record:
            trail.append((big_f.copy(), g_new.copy()))
        change = np.max(np.abs(g_new - big_g) / np.maximum(big_g, 1e-300))
        big_g = g_new
        if change < tol:
            break
    return h0 / half_root(big_g), big_g, index + 1, trail


# -------------------------------------------------------------------- fits --
def fit_potential(target, degree, rank, rng, weight=None, warm=None, starts=STARTS):
    """Fit |c|^2 * weight ~ target with c of the given degree and rank.

    weight=None fits |c|^2 ~ target directly.  Returns (best rel error,
    projection-start rel error, best theta, number of distinct optima).
    """

    columns = BASIS[:, : degree + 1]
    w = np.ones(N + 1) if weight is None else weight
    scale = np.sqrt(PI)

    def unpack(theta):
        if rank == 2:
            return columns @ (theta[: degree + 1] + 1j * theta[degree + 1 :])
        return columns @ theta

    def residual(theta):
        return scale * (np.abs(unpack(theta)) ** 2 * w - target)

    proj = ((BASIS * PI[:, None]).T @ np.sqrt(np.maximum(target / w, 0.0)))[
        : degree + 1
    ]
    inits = []
    if warm is not None:
        inits.append(np.asarray(warm, dtype=float))
    base = proj if rank == 1 else np.concatenate([proj, 1e-3 * rng.standard_normal(degree + 1)])
    inits.append(base)
    size = degree + 1 if rank == 1 else 2 * (degree + 1)
    spread = np.linalg.norm(proj)
    while len(inits) < starts + (warm is not None):
        inits.append(0.7 * spread * rng.standard_normal(size) / np.sqrt(size))

    norm = np.sqrt(np.sum(PI * target**2))
    solutions = []
    method = "lm" if N + 1 >= (degree + 1) * rank else "trf"
    for theta0 in inits:
        result = least_squares(residual, theta0, method=method, max_nfev=6000)
        solutions.append((np.sqrt(2.0 * result.cost) / norm, result.x))
    errors = np.array([s[0] for s in solutions])
    best = int(np.argmin(errors))
    start_error = errors[1] if warm is not None else errors[0]
    distinct = len(np.unique(np.round(np.log10(np.maximum(errors, 1e-16)), 1)))
    return float(errors[best]), float(start_error), solutions[best][1], distinct


# ------------------------------------------------------------------ studies --
def study_exact() -> dict:
    kt = kernel(HORIZON)
    report = {}
    for delta in (0.3, 0.03):
        r0, rt = endpoint_ratios(delta)
        f, g, n_it, trail_d = density_sinkhorn(r0, rt, kt, record=5)
        big_f, big_g, _, trail_a = amplitude_sinkhorn(
            np.sqrt(r0), np.sqrt(rt), kt, record=5
        )
        iterate_dev = max(
            max(
                np.max(np.abs(fa**2 - fd) / np.max(fd)),
                np.max(np.abs(ga**2 - gd) / np.max(gd)),
            )
            for (fd, gd), (fa, ga) in zip(trail_d, trail_a)
        )
        fixed_dev = max(
            np.max(np.abs(big_f**2 - f) / np.max(f)),
            np.max(np.abs(big_g**2 - g) / np.max(g)),
        )
        bridge_dev, mass_dev = 0.0, 0.0
        for t in (0.2, 0.5, 0.8):
            r_mid = (kernel(t) @ f) * (kernel(HORIZON - t) @ g)
            h_mid = np.sqrt(kernel(t) @ big_f**2) * np.sqrt(
                kernel(HORIZON - t) @ big_g**2
            )
            bridge_dev = max(bridge_dev, np.max(np.abs(h_mid**2 - r_mid) / np.max(r_mid)))
            mass_dev = max(mass_dev, abs(np.sum(PI * r_mid) - 1.0))
        residual0 = rel_l2(f * (kt @ g), r0)
        residual_t = rel_l2(g * (kt @ f), rt)
        report[delta] = {
            "iterations": n_it,
            "iterate_dev": iterate_dev,
            "fixed_dev": fixed_dev,
            "bridge_dev": bridge_dev,
            "mass_dev": mass_dev,
            "endpoint_residuals": (residual0, residual_t),
        }
        print(
            f"delta={delta}: {n_it} iterations, iterate dev {iterate_dev:.2e}, "
            f"fixed-point dev {fixed_dev:.2e}, bridge dev {bridge_dev:.2e}, "
            f"mass dev {mass_dev:.2e}, residuals {residual0:.2e}/{residual_t:.2e}",
            flush=True,
        )
    return report


def study_truncation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    kt = kernel(HORIZON)
    report = {}
    for delta in DELTAS:
        r0, rt = endpoint_ratios(delta)
        f, g, _, _ = density_sinkhorn(r0, rt, kt)
        row = {}
        for name, target in (("f", f), ("g", g)):
            for rank in (1, 2):
                errors, starts_only, chambers = [], [], []
                for degree in DEGREES:
                    best, start, _, distinct = fit_potential(
                        target, degree, rank, rng
                    )
                    errors.append(best)
                    starts_only.append(start)
                    chambers.append(distinct)
                row[f"{name}_rank{rank}"] = errors
                row[f"{name}_rank{rank}_projstart"] = starts_only
                row[f"{name}_rank{rank}_optima"] = chambers
        report[delta] = row
        for name in ("f", "g"):
            r1 = " ".join(f"{e:.1e}" for e in row[f"{name}_rank1"])
            r2 = " ".join(f"{e:.1e}" for e in row[f"{name}_rank2"])
            print(f"delta={delta} {name}: rank1 {r1}", flush=True)
            print(f"delta={delta} {name}: rank2 {r2}", flush=True)
    return report


def study_sinkhorn(seed: int, delta=0.03, degree=8, rounds=40) -> dict:
    rng = np.random.default_rng(seed)
    kt = kernel(HORIZON)
    r0, rt = endpoint_ratios(delta)
    f_exact, g_exact, _, _ = density_sinkhorn(r0, rt, kt)
    offline = {
        rank: max(
            fit_potential(f_exact, degree, rank, rng)[0],
            fit_potential(g_exact, degree, rank, rng)[0],
        )
        for rank in (1, 2)
    }
    r_mid_exact = (kernel(0.5) @ f_exact) * (kernel(HORIZON - 0.5) @ g_exact)

    report = {"offline": offline}
    for rank in (2, 1):
        theta_f, theta_g = None, None
        g_values = np.ones(N + 1)
        residuals = []
        for round_index in range(rounds):
            weight = kt @ g_values
            _, _, theta_f, _ = (
                fit_potential(r0, degree, rank, rng, weight=weight, warm=theta_f)
                if round_index == 0
                else fit_potential(
                    r0, degree, rank, rng, weight=weight, warm=theta_f, starts=0
                )
            )
            columns = BASIS[:, : degree + 1]
            f_values = (
                np.abs(columns @ (theta_f[: degree + 1] + 1j * theta_f[degree + 1 :]))
                ** 2
                if rank == 2
                else (columns @ theta_f) ** 2
            )
            weight = kt @ f_values
            _, _, theta_g, _ = (
                fit_potential(rt, degree, rank, rng, weight=weight, warm=theta_g)
                if round_index == 0
                else fit_potential(
                    rt, degree, rank, rng, weight=weight, warm=theta_g, starts=0
                )
            )
            g_values = (
                np.abs(columns @ (theta_g[: degree + 1] + 1j * theta_g[degree + 1 :]))
                ** 2
                if rank == 2
                else (columns @ theta_g) ** 2
            )
            residuals.append(
                (
                    rel_l2(f_values * (kt @ g_values), r0),
                    rel_l2(g_values * (kt @ f_values), rt),
                )
            )
        r_mid = (kernel(0.5) @ f_values) * (kernel(HORIZON - 0.5) @ g_values)
        tail = np.array(residuals[-20:])
        drift = float(np.log(tail[-1].max() / tail[0].max()))
        report[f"rank{rank}"] = {
            "residuals": residuals,
            "floor": float(np.max(tail[-1])),
            "drift_log": drift,
            "bridge_mid_error": rel_l2(r_mid, r_mid_exact),
        }
        shown = " ".join(f"{max(a, b):.1e}" for a, b in residuals[:: max(rounds // 10, 1)])
        print(
            f"rank {rank}: residual trail {shown}; floor {report[f'rank{rank}']['floor']:.2e} "
            f"(offline {offline[rank]:.2e}), late drift {drift:+.3f}, "
            f"bridge mid error {report[f'rank{rank}']['bridge_mid_error']:.2e}",
            flush=True,
        )
    return report


def study_horizon(seed: int, delta=0.1, tolerance=1e-6) -> dict:
    rng = np.random.default_rng(seed)
    report = {}
    for horizon in (0.1, 0.15, 0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0):
        kt = kernel(horizon)
        r0, rt = endpoint_ratios(delta)
        f, g, n_it, _ = density_sinkhorn(r0, rt, kt, iterations=20000)
        degrees_needed = []
        for target in (f, g):
            needed = None
            for degree in range(2, 17):
                best, _, _, _ = fit_potential(target, degree, 2, rng, starts=8)
                if best <= tolerance:
                    needed = degree
                    break
            degrees_needed.append(needed if needed is not None else ">16")
        report[horizon] = {"iterations": n_it, "degrees": degrees_needed}
        print(
            f"T={horizon}: {n_it} iterations, minimal rank-2 degree for "
            f"{tolerance:.0e}: f {degrees_needed[0]}, g {degrees_needed[1]}",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        default="all",
        choices=("exact", "truncation", "sinkhorn", "horizon", "all"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    store_path = ARTIFACTS / "results.json"
    results = json.loads(store_path.read_text()) if store_path.exists() else {}
    if args.study in ("exact", "all"):
        print("== P1: exact equivalence ==", flush=True)
        results["exact"] = study_exact()
    if args.study in ("truncation", "all"):
        print("== P2/R2: truncation, delta x degree ==", flush=True)
        results["truncation"] = study_truncation(args.seed)
    if args.study in ("sinkhorn", "all"):
        print("== P3: truncated Sinkhorn ==", flush=True)
        results["sinkhorn"] = study_sinkhorn(args.seed)
    if args.study in ("horizon", "all"):
        print("== P4/R4: horizon scan ==", flush=True)
        results["horizon"] = study_horizon(args.seed)
    store = ARTIFACTS / "results.json"
    store.write_text(json.dumps(results, default=float))
    print(f"stored: {store}")


if __name__ == "__main__":
    main()
