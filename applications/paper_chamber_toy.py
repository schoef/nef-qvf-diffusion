"""The analytic chamber toy: K = 2, Normal baseline, symmetric target.

Real even amplitude h = cos(a) phi_0 + sin(a) phi_2 on the circle.  With
orthonormal Hermite products (phi_2^2 = sqrt(6) phi_4 + 2 sqrt(2) phi_2
+ phi_0) the fitted coefficients are

    R_2(a) = sin(2a) + 2 sqrt(2) sin^2(a),
    R_4(a) = sqrt(6) sin^2(a),

and the coefficient loss against a symmetric target (R_1 = R_3 = 0) is
an explicit trigonometric polynomial with two minima: the true square
and the wrong square.  The wall -- the amplitude touching zero at the
origin, h(0) = cos(a) - sin(a)/sqrt(2) = 0 -- sits at
a = arctan(sqrt 2), and the exact likelihood erects a logarithmic
barrier there.  The complex extension c_2 -> r e^{i phi} turns the wall
into a point of real codimension two: the disk (Re c_2, Im c_2) shows
the valley connecting the two real minima.  Along diffusion time the
targets decay as e^{-2t} R_2, e^{-4t} R_4: at large t there is one
minimum, the spurious one is born in a saddle-node at finite t, and the
warm-started path never enters it.

Four panels: the loss and the likelihood barrier on the circle; the two
squares as laws; the complex disk; the bifurcation diagram in t.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import fitting_matrices
from nefqvf import Normal, NormalParams

BASELINE = NormalParams(0.0, 1.0)
FIGURE_SUBDIRECTORY = "chambers-toy"
GRID = np.linspace(-6.0, 6.0, 1201)


def target_coefficients(d: float):
    """Symmetric mixture (N(-d,1) + N(+d,1))/2: R_2 = d^2/sqrt2,
    R_4 = d^4/sqrt24."""

    return d**2 / np.sqrt(2.0), d**4 / np.sqrt(24.0)


def fitted_coefficients(alpha):
    r2 = np.sin(2 * alpha) + 2.0 * np.sqrt(2.0) * np.sin(alpha) ** 2
    r4 = np.sqrt(6.0) * np.sin(alpha) ** 2
    return r2, r4


def loss(alpha, t2, t4):
    r2, r4 = fitted_coefficients(alpha)
    return (r2 - t2) ** 2 + (r4 - t4) ** 2


def amplitude_on_grid(alpha):
    basis = np.asarray(Normal.basis(GRID, 2, BASELINE), dtype=float)
    return np.cos(alpha) * basis[:, 0] + np.sin(alpha) * basis[:, 2]


def law_on_grid(alpha):
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    law = ref * amplitude_on_grid(alpha) ** 2
    return law / np.trapezoid(law, GRID)


def verify_against_package():
    phi = fitting_matrices(Normal, BASELINE, 2)
    rng = np.random.default_rng(0)
    for alpha in rng.uniform(-np.pi, np.pi, 5):
        c = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
        r = np.einsum("m,kmn,n->k", c, phi, c)
        r2, r4 = fitted_coefficients(alpha)
        assert abs(r[2] - r2) < 1e-12 and abs(r[4] - r4) < 1e-12
        assert abs(r[1]) < 1e-12 and abs(r[3]) < 1e-12
    print("analytic R_2, R_4 verified against the product tensor")


def minima_of(t2, t4):
    alphas = np.linspace(-np.pi / 2, np.pi / 2, 20001)
    values = loss(alphas, t2, t4)
    interior = (
        np.nonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
    )
    return [(float(alphas[i]), float(values[i])) for i in interior]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=1.8)
    parser.add_argument("--nll-sample", type=int, default=400)
    args = parser.parse_args()
    d = args.d

    verify_against_package()
    t2, t4 = target_coefficients(d)
    wall = np.arctan(np.sqrt(2.0))
    minima = minima_of(t2, t4)
    print(
        f"d = {d}: minima at "
        + ", ".join(f"{a:+.4f} (J={v:.4f})" for a, v in minima)
        + f"; wall at {wall:+.4f}"
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    # ---- A: the loss on the circle, with the likelihood barrier
    axis = axes[0, 0]
    alphas = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    axis.plot(
        alphas,
        loss(alphas, t2, t4),
        color="#1b6ca8",
        lw=1.8,
        label=r"coefficient loss $\mathcal{J}(\alpha)$",
    )
    for a, _ in minima:
        axis.plot([a], [loss(np.array([a]), t2, t4)], "o", color="#1b6ca8", ms=7)
    axis.axvline(wall, color="0.4", ls=":", lw=1.2)
    axis.axvline(-np.pi / 2 + 1e-9, color="none")
    axis.set_xlabel(r"$\alpha$")
    axis.set_ylabel(r"$\mathcal{J}$", color="#1b6ca8")
    twin = axis.twinx()
    rng = np.random.default_rng(1)
    signs = np.where(rng.random(args.nll_sample) < 0.5, 1.0, -1.0)
    sample = signs * d + rng.standard_normal(args.nll_sample)
    basis = np.asarray(Normal.basis(sample, 2, BASELINE), dtype=float)
    nll = [
        -float(
            np.mean(
                np.log(
                    (np.cos(a) * basis[:, 0] + np.sin(a) * basis[:, 2]) ** 2 + 1e-300
                )
            )
        )
        for a in alphas
    ]
    twin.plot(alphas, nll, color="#b0413e", lw=1.2, ls=(0, (4, 2)), label="likelihood")
    twin.set_ylim(min(nll) - 0.2, min(nll) + 4.0)
    twin.set_ylabel("negative log-likelihood", color="#b0413e")
    axis.set_title("two minima, one wall: the chamber on the circle", fontsize=10)
    axis.text(
        wall, axis.get_ylim()[1] * 0.92, " wall: $h(0)=0$", fontsize=9, color="0.3"
    )

    # ---- B: the two squares as laws
    axis = axes[0, 1]
    truth = (np.exp(-0.5 * (GRID - d) ** 2) + np.exp(-0.5 * (GRID + d) ** 2)) / (
        2.0 * np.sqrt(2.0 * np.pi)
    )
    axis.plot(GRID, truth, color="0.15", lw=1.6, label="target")
    order = np.argsort([v for _, v in minima])
    styles = [("#b0413e", "moment optimum"), ("#1b6ca8", "wall-adjacent optimum")]
    for rank, index in enumerate(order[:2]):
        a, _ = minima[index]
        colour, label = styles[rank]
        law = law_on_grid(a)
        tv = 0.5 * float(np.trapezoid(np.abs(law - truth), GRID))
        axis.plot(
            GRID,
            law,
            ls=(0, (2.6, 1.6)),
            color=colour,
            lw=1.9,
            label=rf"{label}, $\alpha={a:+.2f}$, TV {tv:.2f}",
        )
    axis.set_xlabel("$x$")
    axis.set_ylabel("density")
    axis.set_title("what the chamber costs: the two squares", fontsize=10)
    axis.legend(frameon=False, fontsize=8)

    # ---- C: the complex disk
    axis = axes[1, 0]
    radius = np.linspace(0.0, 1.0, 240)
    phase = np.linspace(-np.pi, np.pi, 361)
    R, P = np.meshgrid(radius, phase, indexing="ij")
    # c = (sqrt(1-r^2), 0, r e^{i phi}); coefficients of |h|^2
    c0 = np.sqrt(1.0 - R**2)
    r2 = 2.0 * c0 * R * np.cos(P) + 2.0 * np.sqrt(2.0) * R**2
    r4 = np.sqrt(6.0) * R**2
    J = (r2 - t2) ** 2 + (r4 - t4) ** 2
    X, Y = R * np.cos(P), R * np.sin(P)
    contour = axis.contourf(X, Y, np.log10(J + 1e-6), levels=30, cmap="viridis")
    figure.colorbar(contour, ax=axis, fraction=0.046, label=r"$\log_{10}\mathcal{J}$")
    for a, _ in minima:
        axis.plot([np.sin(a)], [0.0], "o", color="white", ms=7, mec="0.2")
    axis.set_xlabel(r"$\mathrm{Re}\,c_2$")
    axis.set_ylabel(r"$\mathrm{Im}\,c_2$")
    axis.set_aspect("equal")
    axis.set_title("the complex disk: the wall has codimension two", fontsize=10)

    # ---- D: bifurcation along diffusion time, with the warm path
    axis = axes[1, 1]
    times = np.linspace(2.0, 0.0, 241)
    for t in times:
        for a, v in minima_of(np.exp(-2 * t) * t2, np.exp(-4 * t) * t4):
            axis.plot([t], [a], ".", color="0.55", ms=2.5)
    path_alpha, path = 0.0, []
    for t in times:
        fine = np.linspace(path_alpha - 0.2, path_alpha + 0.2, 4001)
        values = loss(fine, np.exp(-2 * t) * t2, np.exp(-4 * t) * t4)
        path_alpha = float(fine[np.argmin(values)])
        path.append(path_alpha)
    axis.plot(times, path, color="#1b6ca8", lw=2.2, label="warm-started path")
    axis.axhline(wall, color="0.4", ls=":", lw=1.0)
    axis.invert_xaxis()
    axis.set_xlabel("slice time $t$")
    axis.set_ylabel(r"minima $\alpha^\ast(t)$")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.set_title(
        "the diffusion path selects the wall-adjacent basin;\n"
        "the moment-global basin is born disconnected",
        fontsize=10,
    )

    figure.tight_layout()
    directory = Path("artifacts") / FIGURE_SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "chamber-toy.pdf")
    figure.savefig(directory / "chamber-toy.png", dpi=140)
    print(f"figure: {directory / 'chamber-toy.png'}")


if __name__ == "__main__":
    main()
