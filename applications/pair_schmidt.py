"""The d = 2 mother model on a correlated Gaussian pair.

The joint law of two standard Gaussians at correlation ``rho`` has, by
Mehler's formula, the diagonal ratio tensor ``R_jk = rho^j delta_jk``
against the product baseline.  The amplitude is the square root of the
ratio, a two-mode squeezed kernel, whose Schmidt spectrum is geometric
with the smaller parameter ``kappa = rho / (1 + sqrt(1 - rho^2))``.

The study fits the full coefficient matrix ``c_nm`` from a sample with
the one-site complex fit on the Kronecker stack and checks, per rho:
the Mehler diagonal of the empirical coefficients, the geometric
Schmidt spectrum of the fitted amplitude against kappa, the parity
support of ``c_nm``, and the total variation of the fitted pair law.
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
    fitting_matrices,
)
from applications.two_site_diffusion import (
    empirical_pair_coefficients,
    pair_fitting_matrices,
)
from nefqvf import Normal, NormalParams

DEFAULT_DEGREE = 8
DEFAULT_DRAWS = 100_000
DEFAULT_SEED = 5
DEFAULT_RHOS = (0.3, 0.6, 0.9)
FIGURE_SUBDIRECTORY = "pair_schmidt"

BASELINE = NormalParams(0.0, 1.0)


def schmidt_parameter(rho: float) -> float:
    """The predicted geometric ratio of the amplitude Schmidt spectrum."""

    return rho / (1.0 + np.sqrt(1.0 - rho * rho))


def sample_pair(rho: float, size: int, rng: Any) -> np.ndarray:
    z1 = rng.standard_normal(size)
    z2 = rng.standard_normal(size)
    return np.column_stack((z1, rho * z1 + np.sqrt(1.0 - rho * rho) * z2))


def pair_density(rho: float, grid: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(grid, grid, indexing="ij")
    norm = 1.0 / (2.0 * np.pi * np.sqrt(1.0 - rho * rho))
    quad = (x * x - 2.0 * rho * x * y + y * y) / (1.0 - rho * rho)
    return norm * np.exp(-0.5 * quad)


def exact_amplitude(rho: float, degree: int) -> np.ndarray:
    """Project sqrt(joint / product) on the product OPS by quadrature."""

    nodes, weights = np.polynomial.hermite_e.hermegauss(160)
    weights = weights / np.sqrt(2.0 * np.pi)
    basis = np.asarray(Normal.basis(nodes, degree, BASELINE), dtype=float)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    log_ratio = -0.5 * np.log(1.0 - rho * rho) - (
        rho * rho * (x * x + y * y) - 2.0 * rho * x * y
    ) / (2.0 * (1.0 - rho * rho))
    h = np.exp(0.5 * log_ratio)
    return np.einsum("a,b,an,bm,ab->nm", weights, weights, basis, basis, h)


def fitted_pair_law(
    coefficients: np.ndarray, grid: np.ndarray, degree: int
) -> np.ndarray:
    basis = np.asarray(Normal.basis(grid, degree, BASELINE), dtype=float)
    field = basis @ coefficients.reshape(degree + 1, degree + 1) @ basis.T
    reference = np.asarray(Normal.prob(grid, BASELINE), dtype=float)
    return np.abs(field) ** 2 * np.outer(reference, reference)


def total_variation(truth: np.ndarray, law: np.ndarray, grid: np.ndarray) -> float:
    cell = np.gradient(grid)
    return 0.5 * float(np.sum(np.abs(truth - law) * np.outer(cell, cell)))


def parity_defect(matrix: np.ndarray) -> float:
    """Weight of the fitted amplitude on odd total parity."""

    n, m = np.meshgrid(*(np.arange(s) for s in matrix.shape), indexing="ij")
    odd = (n + m) % 2 == 1
    return float(np.sum(np.abs(matrix[odd]) ** 2) / np.sum(np.abs(matrix) ** 2))


def run_study(
    rho: float,
    *,
    degree: int = DEFAULT_DEGREE,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample = sample_pair(rho, draws, rng)

    phi = fitting_matrices(Normal, BASELINE, degree)
    k_max = phi.shape[0] - 1
    stack, _ = pair_fitting_matrices(phi)
    empirical = empirical_pair_coefficients(Normal, BASELINE, sample, k_max)

    diagonal = np.diagonal(empirical)
    mehler_error = float(
        np.max(np.abs(empirical - np.diag(rho ** np.arange(k_max + 1))))
    )

    fit = continued_complex_fit(stack, empirical.reshape(-1))["complex"]
    matrix = fit["coefficients"].reshape(degree + 1, degree + 1)
    singular = np.linalg.svd(matrix, compute_uv=False)

    exact = exact_amplitude(rho, degree)
    exact_singular = np.linalg.svd(exact, compute_uv=False)

    grid = np.linspace(-6.0, 6.0, 401)
    truth = pair_density(rho, grid)
    law = fitted_pair_law(fit["coefficients"], grid, degree)
    tv = total_variation(truth, law, grid)

    kappa = schmidt_parameter(rho)
    return {
        "rho": rho,
        "degree": degree,
        "kappa": kappa,
        "mehler_diagonal": diagonal,
        "mehler_error": mehler_error,
        "singular": singular / singular[0],
        "exact_singular": exact_singular / exact_singular[0],
        "predicted": kappa ** np.arange(degree + 1),
        "parity_defect": parity_defect(matrix),
        "total_variation": tv,
        "grid": grid,
        "truth": truth,
        "law": law,
        "coefficients": fit["coefficients"],
    }


def rotated_projection(
    coefficients: np.ndarray, degree: int, sign: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project the fitted law onto ``u = (x + sign * y) / sqrt(2)`` by a
    line integral over the conjugate coordinate on a rotated lattice."""

    u = np.linspace(-6.0, 6.0, 241)
    v = np.linspace(-8.5, 8.5, 341)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (uu + vv) / np.sqrt(2.0)
    y = sign * (uu - vv) / np.sqrt(2.0)
    matrix = coefficients.reshape(degree + 1, degree + 1)
    basis_x = np.asarray(Normal.basis(x.reshape(-1), degree, BASELINE), dtype=float)
    basis_y = np.asarray(Normal.basis(y.reshape(-1), degree, BASELINE), dtype=float)
    field = np.einsum("an,nm,am->a", basis_x, matrix, basis_y).reshape(x.shape)
    reference = np.exp(-0.5 * (x * x + y * y)) / (2.0 * np.pi)
    law = np.abs(field) ** 2 * reference
    return u, np.trapezoid(law, v, axis=1)


def plot_quality(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Law-level diagnostics: contours, difference, projections, conditionals."""

    rho = result["rho"]
    grid, truth, law = result["grid"], result["truth"], result["law"]
    cell = np.gradient(grid)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))

    # contour overlay
    axis = axes[0, 0]
    levels = np.max(truth) * np.array([0.01, 0.05, 0.2, 0.5, 0.9])
    axis.contour(grid, grid, truth.T, levels=levels, colors="0.3", linewidths=1.0)
    axis.contour(
        grid,
        grid,
        law.T,
        levels=levels,
        colors="#b0413e",
        linewidths=1.0,
        linestyles="--",
    )
    axis.set_xlim(-4.5, 4.5)
    axis.set_ylim(-4.5, 4.5)
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")
    axis.set_title("contours: truth (solid), fitted (dashed)")

    # signed difference
    axis = axes[0, 1]
    difference = law - truth
    scale = float(np.max(np.abs(difference)))
    image = axis.pcolormesh(
        grid, grid, difference.T, cmap="RdBu_r", vmin=-scale, vmax=scale
    )
    figure.colorbar(image, ax=axis, shrink=0.85)
    axis.set_xlim(-4.5, 4.5)
    axis.set_ylim(-4.5, 4.5)
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")
    axis.set_title(f"fitted $-$ truth   (TV = {result['total_variation']:.2e})")

    # 1D projections: marginal and the rotated axes
    axis = axes[1, 0]
    marginal = law @ cell
    axis.plot(grid, marginal, color="#b0413e", label="fitted $x_1$ marginal")
    normal = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    axis.plot(grid, normal, color="0.3", ls=":", lw=1.2, label="$N(0,1)$")
    for sign, style in ((1.0, "-"), (-1.0, "--")):
        centers, projected = rotated_projection(
            result["coefficients"], result["degree"], sign
        )
        variance = 1.0 + sign * rho
        exact = np.exp(-0.5 * centers**2 / variance) / np.sqrt(2.0 * np.pi * variance)
        tag = "+" if sign > 0 else "-"
        axis.plot(
            centers,
            projected,
            style,
            color="#2a6f97",
            label=rf"fitted $(x_1 {tag} x_2)/\sqrt{{2}}$",
        )
        axis.plot(centers, exact, ls=":", lw=1.2, color="0.3")
    axis.set_xlim(-5.0, 5.0)
    axis.set_xlabel("projected coordinate")
    axis.set_ylabel("density")
    axis.set_title(r"1D projections (dotted: exact $N(0, 1\pm\rho)$)")
    axis.legend(fontsize=8, frameon=False)

    # conditional slices
    axis = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0.0, 0.8, 3))
    for x1, color in zip((0.0, 1.0, 2.0), colors, strict=True):
        row = int(np.argmin(np.abs(grid - x1)))
        fitted = law[row] / np.trapezoid(law[row], grid)
        variance = 1.0 - rho * rho
        exact = np.exp(-0.5 * (grid - rho * x1) ** 2 / variance) / np.sqrt(
            2.0 * np.pi * variance
        )
        axis.plot(grid, fitted, color=color, label=rf"$x_1 = {x1:g}$")
        axis.plot(grid, exact, ls=":", lw=1.2, color="0.3")
    axis.set_xlim(-4.5, 4.5)
    axis.set_xlabel("$x_2$")
    axis.set_ylabel(r"$p(x_2 \mid x_1)$")
    axis.set_title("conditional slices (dotted: exact)")
    axis.legend(fontsize=8, frameon=False)

    figure.suptitle(
        rf"correlated Gaussian pair, $\rho = {rho:g}$, $K = {result['degree']}$"
    )
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pair-schmidt-quality-rho{rho:g}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_study(results: list[dict[str, Any]], *, output_dir: Any = None) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))
    for result, color in zip(results, colors, strict=True):
        n = np.arange(len(result["singular"]))
        label = rf"$\rho = {result['rho']:g}$"
        axes[0].semilogy(n, result["singular"], "o", color=color, label=label)
        axes[0].semilogy(n, result["predicted"], "-", color=color, lw=1.0)
        k = np.arange(len(result["mehler_diagonal"]))
        axes[1].semilogy(k, np.abs(result["mehler_diagonal"]), "s", color=color)
        axes[1].semilogy(k, result["rho"] ** k, "-", color=color, lw=1.0)
    axes[0].set_xlabel("Schmidt index $n$")
    axes[0].set_ylabel(r"$\sigma_n / \sigma_0$")
    axes[0].set_title(r"amplitude: fitted vs $\kappa^n$")
    axes[0].legend(fontsize=9, frameon=False)
    axes[1].set_xlabel("degree $k$")
    axes[1].set_ylabel(r"$\widehat R_{kk}$")
    axes[1].set_title(r"ratio diagonal vs $\rho^k$ (Mehler)")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pair-schmidt.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rhos", type=float, nargs="+", default=list(DEFAULT_RHOS))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    for rho in args.rhos:
        result = run_study(rho, degree=args.degree, draws=args.draws, seed=args.seed)
        results.append(result)
        kappa = result["kappa"]
        fitted_ratio = result["singular"][1]
        exact_ratio = result["exact_singular"][1]
        print(
            f"rho {rho:4.2f}  kappa {kappa:.4f}  "
            f"sigma_1/sigma_0 fitted {fitted_ratio:.4f} exact {exact_ratio:.4f}  "
            f"mehler max err {result['mehler_error']:.2e}  "
            f"parity defect {result['parity_defect']:.2e}  "
            f"TV {result['total_variation']:.3e}"
        )
        with np.printoptions(precision=4, suppress=False):
            print(f"          fitted spectrum {result['singular']}")
            print(f"          predicted       {result['predicted']}")
    if args.plot:
        print(f"figure: {plot_study(results, output_dir=args.output)}")
        for result in results:
            print(f"figure: {plot_quality(result, output_dir=args.output)}")


if __name__ == "__main__":
    main()
