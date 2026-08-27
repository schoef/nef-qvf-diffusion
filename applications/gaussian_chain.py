"""The exactly solvable instance: a Gaussian chain under the reference flow.

The target is the stationary AR(1) Gaussian chain with unit marginals
and lag-one correlation ``rho0``.  Sitewise Ornstein--Uhlenbeck noising
acts on the covariance as

    Sigma(t) = exp(-2t) Sigma_0 + (1 - exp(-2t)) * Identity,

so the marginals stay standard normal and every correlation decays as
``exp(-2t)``.  Three closed-form statements follow and are verified
here against quadrature and against the fitting pipeline:

1.  Across any cut the cross-covariance of the chain is rank one at
    every t, leaving a single canonical correlation ``r_cut(t)``.
2.  The slice amplitude's Schmidt spectrum at every cut is geometric,
    ``kappa(r_cut(t))^n`` with ``kappa(r) = r / (1 + sqrt(1 - r^2))``:
    the entanglement melts along the schedule at a computable rate.
3.  Per slice the fixed-frame fit has a computable expressivity floor,
    the total variation of the degree-truncated exact amplitude.  Warm
    starts down the schedule close the optimisation gap to that floor;
    only degree or a moved frame goes below it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
)
from applications.pair_schmidt import exact_amplitude, schmidt_parameter
from applications.two_site_diffusion import (
    empirical_pair_coefficients,
    pair_fitting_matrices,
)
from nefqvf import Normal, NormalParams

DEFAULT_RHO = 0.95
DEFAULT_DEGREE = 8
DEFAULT_DRAWS = 100_000
DEFAULT_SEED = 5
FIGURE_SUBDIRECTORY = "gaussian_chain"

BASELINE = NormalParams(0.0, 1.0)


# ------------------------------------------------------------------ exact --
def chain_covariance(rho0: float, d: int, t: float) -> np.ndarray:
    """``Sigma(t)`` for the noised AR(1) chain."""

    lags = np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    sigma0 = rho0**lags
    return np.exp(-2.0 * t) * sigma0 + (1.0 - np.exp(-2.0 * t)) * np.eye(d)


def canonical_correlation(sigma: np.ndarray, cut: int) -> float:
    """The largest canonical correlation between the two blocks at a cut."""

    cross = sigma[:cut, cut:]
    a = np.linalg.solve(sigma[:cut, :cut], cross)
    b = np.linalg.solve(sigma[cut:, cut:], cross.T)
    eigenvalues = np.linalg.eigvals(a @ b)
    return float(np.sqrt(np.max(np.real(eigenvalues))))


def cross_block_rank(sigma: np.ndarray, cut: int, tolerance: float = 1e-10) -> int:
    singular = np.linalg.svd(sigma[:cut, cut:], compute_uv=False)
    return int(np.sum(singular > tolerance * singular[0]))


def exact_chain_amplitude(
    sigma: np.ndarray, degree: int, nodes: int = 60
) -> np.ndarray:
    """Coefficient tensor of sqrt(joint / product) by Gauss--Hermite."""

    d = sigma.shape[0]
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    basis = np.asarray(Normal.basis(x, degree, BASELINE), dtype=float)
    grids = np.meshgrid(*([x] * d), indexing="ij")
    points = np.stack([g.reshape(-1) for g in grids], axis=1)
    precision = np.linalg.inv(sigma)
    log_joint = -0.5 * np.einsum("si,ij,sj->s", points, precision, points)
    log_joint -= 0.5 * np.linalg.slogdet(sigma)[1]
    log_product = -0.5 * np.einsum("si,si->s", points, points)
    ratio = np.exp(0.5 * (log_joint - log_product)).reshape([nodes] * d)
    weighted = np.asarray(w[:, None] * basis)
    tensor = ratio
    for _ in range(d):
        tensor = np.tensordot(tensor, weighted, axes=([0], [0]))
    return tensor


def schmidt_spectrum(tensor: np.ndarray, cut: int) -> np.ndarray:
    shape = tensor.shape
    matrix = tensor.reshape(int(np.prod(shape[:cut])), -1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return singular / singular[0]


# ------------------------------------------------------------------ floor --
def pair_expressivity_floor(rho: float, degree: int, grid: np.ndarray) -> float:
    """TV of the pair law of the degree-truncated exact amplitude."""

    c = exact_amplitude(rho, degree)
    c = c / np.linalg.norm(c)
    basis = np.asarray(Normal.basis(grid, degree, BASELINE), dtype=float)
    field = basis @ c @ basis.T
    reference = np.asarray(Normal.prob(grid, BASELINE), dtype=float)
    law = np.outer(reference, reference) * field**2
    x, y = np.meshgrid(grid, grid, indexing="ij")
    truth = np.exp(-0.5 * (x * x - 2 * rho * x * y + y * y) / (1 - rho * rho)) / (
        2 * np.pi * np.sqrt(1 - rho * rho)
    )
    cell = np.gradient(grid)
    return 0.5 * float(np.sum(np.abs(truth - law) * np.outer(cell, cell)))


def pair_tv(rho: float, coefficients: np.ndarray, degree: int, grid) -> float:
    basis = np.asarray(Normal.basis(grid, degree, BASELINE), dtype=float)
    field = basis @ coefficients.reshape(degree + 1, degree + 1) @ basis.T
    reference = np.asarray(Normal.prob(grid, BASELINE), dtype=float)
    law = np.outer(reference, reference) * np.abs(field) ** 2
    x, y = np.meshgrid(grid, grid, indexing="ij")
    truth = np.exp(-0.5 * (x * x - 2 * rho * x * y + y * y) / (1 - rho * rho)) / (
        2 * np.pi * np.sqrt(1 - rho * rho)
    )
    cell = np.gradient(grid)
    return 0.5 * float(np.sum(np.abs(truth - law) * np.outer(cell, cell)))


# ------------------------------------------------------------------ study --
def verify_melting(
    rho0: float, d: int = 4, degree: int = 14, times=(0.0, 0.3, 0.8)
) -> list[dict[str, Any]]:
    """Check rank-one cuts and the geometric Schmidt law on the exact tensor."""

    rows = []
    for t in times:
        sigma = chain_covariance(rho0, d, t)
        tensor = exact_chain_amplitude(sigma, degree, nodes=60)
        for cut in range(1, d):
            r = canonical_correlation(sigma, cut)
            kappa = schmidt_parameter(r)
            spectrum = schmidt_spectrum(tensor, cut)[: degree + 1]
            predicted = kappa ** np.arange(len(spectrum))
            head = 4
            error = float(np.max(np.abs(spectrum[:head] - predicted[:head])))
            rows.append(
                {
                    "t": t,
                    "cut": cut,
                    "rank": cross_block_rank(sigma, cut),
                    "r": r,
                    "kappa": kappa,
                    "measured": float(spectrum[1]),
                    "spectrum_error": error,
                }
            )
    return rows


def run_schedule(
    rho0: float,
    *,
    degree: int = DEFAULT_DEGREE,
    draws: int = DEFAULT_DRAWS,
    n_slices: int = 13,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Warm against cold fixed-frame pair fits down the schedule."""

    rng = np.random.default_rng(seed)
    phi = fitting_matrices(Normal, BASELINE, degree)
    k_max = phi.shape[0] - 1
    stack, _ = pair_fitting_matrices(phi)
    grid = np.linspace(-6.0, 6.0, 401)

    # uniform in rho, so consecutive warm starts stay close where it is hard
    rhos = np.linspace(np.exp(-4.0) * rho0, rho0, n_slices)
    times = -0.5 * np.log(rhos / rho0)
    rows = []
    previous = None
    for t in times:
        rho_t = float(np.exp(-2.0 * t) * rho0)
        cov = np.array([[1.0, rho_t], [rho_t, 1.0]])
        sample = rng.multivariate_normal(np.zeros(2), cov, size=draws)
        empirical = empirical_pair_coefficients(Normal, BASELINE, sample, k_max)
        target = empirical.reshape(-1)

        continued = continued_complex_fit(stack, target)["complex"]
        local_cold = fit_complex_amplitude(stack, target)
        if previous is None:
            warm_c = continued["coefficients"]
        else:
            warm_c = fit_complex_amplitude(stack, target, initial=previous)[
                "coefficients"
            ]
        previous = warm_c

        rows.append(
            {
                "t": float(t),
                "rho": rho_t,
                "floor": pair_expressivity_floor(rho_t, degree, grid),
                "continued_tv": pair_tv(rho_t, continued["coefficients"], degree, grid),
                "warm_tv": pair_tv(rho_t, warm_c, degree, grid),
                "cold_tv": pair_tv(rho_t, local_cold["coefficients"], degree, grid),
            }
        )
        print(
            f"  t {t:5.2f}  rho {rho_t:6.4f}  floor {rows[-1]['floor']:9.3e}  "
            f"continued TV {rows[-1]['continued_tv']:9.3e}  "
            f"warm-local TV {rows[-1]['warm_tv']:9.3e}  "
            f"cold-local TV {rows[-1]['cold_tv']:9.3e}"
        )
    return {"rho0": rho0, "degree": degree, "rows": rows}


def plot_study(
    melting: list[dict[str, Any]],
    schedule: dict[str, Any],
    *,
    output_dir: Any = None,
) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    rho0, degree = schedule["rho0"], schedule["degree"]
    t_grid = np.linspace(0.0, 2.0, 201)
    kappa_link = [schmidt_parameter(np.exp(-2 * t) * rho0) for t in t_grid]
    kappa_cut = [
        schmidt_parameter(canonical_correlation(chain_covariance(rho0, 4, t), 2))
        for t in t_grid
    ]
    axes[0].plot(
        t_grid, kappa_cut, color="0.3", label=r"$\kappa(r_{\rm cut}(t))$, $d=4$"
    )
    axes[0].plot(
        t_grid,
        kappa_link,
        ls="--",
        lw=1.0,
        color="0.6",
        label=r"$\kappa(e^{-2t}\rho_0)$, single link",
    )
    for row in melting:
        if row["cut"] == 2:
            axes[0].plot(
                row["t"], row["measured"], "o", color="#2a6f97", ms=6, mfc="none"
            )
    axes[0].plot([], [], "o", color="#2a6f97", mfc="none", label="exact tensor, $d=4$")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel(r"Schmidt parameter $\kappa$")
    axes[0].set_title("the entanglement melts")
    axes[0].legend(fontsize=9, frameon=False)

    rows = schedule["rows"]
    t = [row["t"] for row in rows]
    axes[1].semilogy(
        t,
        [row["continued_tv"] for row in rows],
        "o-",
        color="#2a6f97",
        label="continuation fit",
    )
    axes[1].semilogy(
        t,
        [row["warm_tv"] for row in rows],
        "s-",
        color="#7fa653",
        label="local fit, warm chain",
    )
    axes[1].semilogy(
        t,
        [row["cold_tv"] for row in rows],
        "^-",
        color="#b0413e",
        label="local fit, cold",
    )
    axes[1].semilogy(
        t, [row["floor"] for row in rows], ":", color="0.3", label="expressivity floor"
    )
    sampling = rows[0]["continued_tv"]
    axes[1].axhline(sampling, color="0.75", lw=0.8)
    axes[1].text(
        1.95, sampling * 1.2, "sampling floor", fontsize=8, color="0.5", va="bottom"
    )
    axes[1].set_ylim(1e-3, 1.0)
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("pair total variation")
    axes[1].invert_xaxis()
    axes[1].set_title(rf"$\rho_0 = {rho0:g}$, $K = {degree}$")
    axes[1].legend(fontsize=9, frameon=False)
    figure.tight_layout()

    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gaussian-chain-rho{rho0:g}.png"
    figure.savefig(path, dpi=150)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    kappa0 = schmidt_parameter(args.rho)
    exact_pair = exact_amplitude(args.rho, 40)
    spectrum = np.linalg.svd(exact_pair, compute_uv=False)
    spectrum = spectrum / spectrum[0]
    head_error = float(np.max(np.abs(spectrum[:8] - kappa0 ** np.arange(8))))
    print(
        f"t = 0 link check at d = 2, degree 40: "
        f"max|sigma_n/sigma_0 - kappa^n| = {head_error:.2e}"
    )
    print("melting law on the exact d = 4 tensor:")
    melting = verify_melting(args.rho)
    for row in melting:
        print(
            f"  t {row['t']:4.2f}  cut {row['cut']}  cross rank {row['rank']}  "
            f"r {row['r']:6.4f}  kappa {row['kappa']:6.4f}  "
            f"spectrum error {row['spectrum_error']:.2e}"
        )
    print("schedule, warm vs cold vs floor:")
    schedule = run_schedule(
        args.rho, degree=args.degree, draws=args.draws, seed=args.seed
    )
    if args.plot:
        print(f"figure: {plot_study(melting, schedule, output_dir=args.output)}")


if __name__ == "__main__":
    main()
