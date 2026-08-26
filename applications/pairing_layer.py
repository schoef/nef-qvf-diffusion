"""The pairing layer: correlated references from one bilinear generator.

The generic pair creator ``A_1^dagger A_2^dagger`` preserves the diagonal
tower ``|n, n>``, so the unitary ``exp(xi (A_1^dagger A_2^dagger - A_1 A_2))``
restricted to that tower is a tridiagonal flow -- the baseline flow's
structure, one degree up.  Applied to the vacuum it creates an amplitude
``c_n`` on the diagonal, and the Born law

    p(x_1, x_2) = p_0(x_1) p_0(x_2) (sum_n c_n phi_n(x_1) phi_n(x_2))^2

is a valid correlated pair law for any family, with every mode of one
leg coherently tied to the same mode of the other.

For Normal legs this is the two-mode squeezed vacuum,
``c_n = tanh(xi)^n / cosh(xi)``, i.e. exactly the Mehler amplitude of
the correlated Gaussian.  For Poisson and Gamma legs it generates
one-parameter families of correlated lattice and half-line laws whose
ratio tensor the study inspects.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from nefqvf import Gamma, GammaParams, Normal, NormalParams, Poisson, PoissonParams

DEFAULT_DEGREE = 30
FIGURE_SUBDIRECTORY = "pairing_layer"

LEGS = {
    "normal": (Normal, NormalParams(0.0, 1.0)),
    "poisson": (Poisson, PoissonParams(4.0)),
    "gamma": (Gamma, GammaParams(3.0, 3.0)),
}


def ladder_weights(name: str, degree: int) -> np.ndarray:
    """The generic creator weights a_{n+1} / z'(0) on the orthonormal tower."""

    n = np.arange(degree)
    if name == "normal":
        return np.sqrt(n + 1.0)
    if name == "poisson":
        lam = LEGS["poisson"][1].mean
        return np.sqrt(lam * (n + 1.0))
    if name == "gamma":
        r = LEGS["gamma"][1].r
        return np.sqrt((n + 1.0) * (r + n))
    raise ValueError(name)


def xi_scaled(name: str, xi: float) -> float:
    """Scale xi so the leading rotation angle matches the Normal case."""

    return xi / float(ladder_weights(name, 1)[0] ** 2)


def pairing_amplitude(name: str, xi: float, degree: int) -> np.ndarray:
    """``exp(xi (P - P^T)) e_0`` on the diagonal tower ``|n, n>``."""

    weights = ladder_weights(name, degree)
    generator = np.zeros((degree + 1, degree + 1))
    rows = np.arange(1, degree + 1)
    generator[rows, rows - 1] = weights * weights
    generator -= generator.T
    vacuum = np.zeros(degree + 1)
    vacuum[0] = 1.0
    return expm(xi * generator) @ vacuum


def support_grid(name: str) -> np.ndarray:
    if name == "normal":
        return np.linspace(-8.0, 8.0, 481)
    if name == "poisson":
        return np.arange(0, 80).astype(float)
    return np.linspace(1e-4, 40.0, 2401)


def pair_law(name: str, c: np.ndarray, grid: np.ndarray) -> np.ndarray:
    family, baseline = LEGS[name]
    degree = len(c) - 1
    basis = np.asarray(family.basis(grid, degree, baseline), dtype=float)
    field = (basis * c) @ basis.T
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    return np.outer(reference, reference) * field**2


def cell(name: str, grid: np.ndarray) -> np.ndarray:
    return np.ones_like(grid) if name == "poisson" else np.gradient(grid)


def law_moments(name: str, law: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    w = np.outer(cell(name, grid), cell(name, grid))
    mass = float(np.sum(law * w))
    p = law * w / mass
    m1 = float(np.sum(grid[:, None] * p))
    m2 = float(np.sum(grid[None, :] * p))
    v1 = float(np.sum(grid[:, None] ** 2 * p)) - m1 * m1
    v2 = float(np.sum(grid[None, :] ** 2 * p)) - m2 * m2
    cross = float(np.sum(grid[:, None] * grid[None, :] * p)) - m1 * m2
    return {
        "mass": mass,
        "mean": m1,
        "variance": v1,
        "correlation": cross / np.sqrt(v1 * v2),
        "mean2": m2,
        "variance2": v2,
    }


def ratio_tensor(name: str, law: np.ndarray, grid: np.ndarray, k_max: int):
    """Coefficients ``R_jk = E[phi_j(x_1) phi_k(x_2)]`` of the pair law."""

    family, baseline = LEGS[name]
    basis = np.asarray(family.basis(grid, k_max, baseline), dtype=float)
    w = np.outer(cell(name, grid), cell(name, grid))
    return basis.T @ (law * w) @ basis


def study_family(name: str, xi: float, degree: int) -> dict[str, Any]:
    c = pairing_amplitude(name, xi, degree)
    tail = float(np.sum(c[-3:] ** 2))
    grid = support_grid(name)
    law = pair_law(name, c, grid)
    moments = law_moments(name, law, grid)
    ratio = ratio_tensor(name, law, grid, min(8, degree))
    off = ratio - np.diag(np.diagonal(ratio))
    diagonal = np.diagonal(ratio)
    return {
        "name": name,
        "xi": xi,
        "amplitude": c,
        "tail": tail,
        "grid": grid,
        "law": law,
        "moments": moments,
        "ratio_diagonal": diagonal,
        "off_diagonal_mass": float(np.max(np.abs(off))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--xi", type=float, default=0.6)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    xi, degree = args.xi, args.degree

    # 1. Normal legs: the two-mode squeezed vacuum, i.e. Mehler
    c = pairing_amplitude("normal", xi, degree)
    tmsv = np.tanh(xi) ** np.arange(degree + 1) / np.cosh(xi)
    rho_predicted = np.tanh(2.0 * xi)
    var_predicted = np.cosh(2.0 * xi)
    result = study_family("normal", xi, degree)
    grid = result["grid"]
    x, y = np.meshgrid(grid, grid, indexing="ij")
    # the pairing state is Gaussian with thermal marginals: a Bogoliubov
    # transformation of the vacuum -- covariance cosh(2 xi), correlation
    # tanh(2 xi); local squeezes would restore unit marginals.
    det = var_predicted**2 * (1.0 - rho_predicted**2)
    quad = (x * x - 2 * rho_predicted * x * y + y * y) / (
        var_predicted * (1.0 - rho_predicted**2)
    )
    exact = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
    w = np.outer(np.gradient(grid), np.gradient(grid))
    tv = 0.5 * float(np.sum(np.abs(result["law"] - exact) * w))
    m = result["moments"]
    print(f"normal  xi {xi:g}: max|c_n - tanh^n/cosh| = {np.max(np.abs(c - tmsv)):.2e}")
    print(
        f"        correlation {m['correlation']:.6f} (tanh 2xi = {rho_predicted:.6f})  "
        f"marginal variance {m['variance']:.6f} (cosh 2xi = {var_predicted:.6f})  "
        f"TV vs thermal Gaussian {tv:.2e}"
    )

    # 2. Poisson and Gamma legs: the new correlated references.  The
    # amplitude is diagonal by construction; the questions are what the
    # marginals become and how much correlation one parameter buys.
    from nefqvf import NegativeBinomial, NegativeBinomialParams

    for name in ("poisson", "gamma"):
        scaled = xi_scaled(name, xi)
        result = study_family(name, scaled, degree)
        m = result["moments"]
        family, baseline = LEGS[name]
        base_mean = float(family.mean(baseline))
        base_var = float(family.variance(baseline))
        grid = result["grid"]
        marginal = np.sum(result["law"] * cell(name, grid)[None, :], axis=1)

        # moment-matched member of the conjectured marginal family
        if name == "poisson":
            r_eff = m["mean"] ** 2 / (m["variance"] - m["mean"])
            candidate = np.asarray(
                NegativeBinomial.prob(grid, NegativeBinomialParams(m["mean"], r_eff)),
                dtype=float,
            )
            label = f"NB(mean, r = {r_eff:.3f})"
        else:
            r_eff = m["mean"] ** 2 / m["variance"]
            candidate = np.asarray(
                Gamma.prob(grid, GammaParams(m["mean"], r_eff)), dtype=float
            )
            label = f"Gamma(mean, r = {r_eff:.3f})"
        marginal_tv = 0.5 * float(
            np.sum(np.abs(marginal - candidate) * cell(name, grid))
        )
        print(
            f"{name:7s} xi_eff {scaled:.4f}: norm tail {result['tail']:.1e}  "
            f"mass {m['mass']:.6f}"
        )
        print(
            f"        marginal mean {m['mean']:.4f} (base {base_mean:g})  "
            f"variance {m['variance']:.4f} (base {base_var:g})  "
            f"correlation {m['correlation']:.4f}"
        )
        print(f"        marginal TV vs {label} = {marginal_tv:.3e}")

    if args.plot:
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
        for axis, name in zip(axes, ("normal", "poisson", "gamma"), strict=True):
            result = study_family(name, xi_scaled(name, xi), degree)
            grid = result["grid"]
            axis.pcolormesh(grid, grid, result["law"].T, cmap="viridis")
            rho = result["moments"]["correlation"]
            axis.set_title(rf"{name}, $\rho = {rho:.3f}$")
            axis.set_xlabel("$x_1$")
            if name == "normal":
                axis.set_xlim(-5, 5)
                axis.set_ylim(-5, 5)
            elif name == "poisson":
                axis.set_xlim(0, 25)
                axis.set_ylim(0, 25)
            else:
                axis.set_xlim(0, 20)
                axis.set_ylim(0, 20)
        axes[0].set_ylabel("$x_2$")
        figure.tight_layout()
        directory = (
            Path("artifacts") / FIGURE_SUBDIRECTORY
            if args.output is None
            else Path(args.output)
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"pairing-laws-xi{xi:g}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
