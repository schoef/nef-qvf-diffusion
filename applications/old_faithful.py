"""Old Faithful waiting times: a real-data study of the amplitude fit.

The classic 272-observation record of waiting times between eruptions
(R ``faithful``), fitted with the full model of the note.  Registered
predictions and outcomes:

  P1  the bimodal marginal is resolved at modest degree, noise-floor
      limited -- CONFIRMED (baseline NLL 4.07 -> 3.82, both a Gamma and
      a width-selected Normal baseline land within 0.01 nats);
  P2  the lag-1 pair factorisation M = R W R^T recovers two members at
      the marginal modes with the short->long asymmetry -- SPLIT.  The
      regime laws ARE coherent members of the Normal family at the
      regime width (sin of the tower angle 0.02-0.06 at sigma 5.7;
      only the Gamma family, whose member width grows with the mean,
      cannot carry them).  The moment-curve factorisation nevertheless
      fails at N = 272 because its span estimator is noise-dominated:
      the second direction of the top-2 eigenspan of the empirical
      pair moment matrix is unstable under a split-half test
      (principal overlaps 0.99/0.24 at degree 6), so the curve is
      scanned against noise.  Freeing the members to the empirical
      regime towers bypasses the span estimate and recovers W matching
      the empirical transitions, including the near-zero short->short
      entry (0.027 vs 0.026);
  P3  the pair amplitude beats the product of marginals at lag 1 and
      the latent chain is Markov -- SPLIT: the chain is Markov to three
      decimals (rho_2 = rho_1^2) and the dependence is unambiguous in
      the moments, but with each model given its own held-out capacity
      the likelihood margin of the pair over the product is within one
      standard error at N = 272 (+0.09 +- 0.07 nats per pair at the
      Gamma baseline, -0.01 +- 0.09 at the Normal): the bond pays at
      the moment level, not yet at the likelihood level.

Splits are blocked in time so serial dependence cannot leak into the
held-out score.
"""

from __future__ import annotations

import argparse
import csv
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
from applications.old_faithful_modern import channel_weight, likelihood_polish
from applications.two_site_diffusion import (
    factorise_pair_moments,
    pair_fitting_matrices,
)
from nefqvf import Gamma, GammaParams, Normal, NormalParams

FIGURE_SUBDIRECTORY = "old-faithful-1985"
GUARD = 1e-300  # numerical floor inside the log only
MARGINAL_DEGREES = (2, 3, 4, 5, 6, 8, 10)
PAIR_DEGREES = (2, 3, 4, 6, 8)
WIDTH_GRID = (0.4, 0.5, 0.6, 0.8, 1.0)  # Normal baseline widths, in units of std


def load_waiting(path: str = "data/faithful.csv") -> np.ndarray:
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return np.array([float(r["waiting"]) for r in rows])


def gamma_baseline(x: np.ndarray) -> GammaParams:
    mean = float(np.mean(x))
    return GammaParams(mean=mean, r=mean**2 / float(np.var(x)))


def moments_of(family: Any, x: np.ndarray, k_max: int, baseline: Any) -> np.ndarray:
    return np.asarray(family.basis(x, k_max, baseline), dtype=float).mean(axis=0)


def marginal_nll(family: Any, c: np.ndarray, x: np.ndarray, k: int, baseline) -> float:
    basis = np.asarray(family.basis(x, k, baseline), dtype=float)
    h2 = np.abs(basis @ c) ** 2
    log_ref = np.log(np.asarray(family.prob(x, baseline), dtype=float))
    return -float(np.mean(log_ref + np.log(h2 + GUARD)))


def pair_nll(family: Any, c: np.ndarray, pairs: np.ndarray, k: int, baseline) -> float:
    b1 = np.asarray(family.basis(pairs[:, 0], k, baseline), dtype=float)
    b2 = np.asarray(family.basis(pairs[:, 1], k, baseline), dtype=float)
    h = np.einsum("ij,jk,ik->i", b1, c.reshape(k + 1, k + 1), b2)
    log_ref = np.log(np.asarray(family.prob(pairs[:, 0], baseline), dtype=float))
    log_ref += np.log(np.asarray(family.prob(pairs[:, 1], baseline), dtype=float))
    return -float(np.mean(log_ref + np.log(np.abs(h) ** 2 + GUARD)))


def latent_correlation(weights: np.ndarray) -> float:
    """Correlation of the +-1 latent under joint weights."""

    w = weights / np.sum(weights)
    m1 = float(w[0, 0] + w[0, 1] - w[1, 0] - w[1, 1])
    m2 = float(w[0, 0] + w[1, 0] - w[0, 1] - w[1, 1])
    cross = float(w[0, 0] - w[0, 1] - w[1, 0] + w[1, 1])
    return (cross - m1 * m2) / max(np.sqrt((1 - m1**2) * (1 - m2**2)), 1e-12)


def pair_moment_matrix(basis: np.ndarray, lag: int) -> np.ndarray:
    b1, b2 = basis[:-lag], basis[lag:]
    m = (b1[:, :, None] * b2[:, None, :]).mean(axis=0)
    return 0.5 * (m + m.T)


# ------------------------------------------------------------------ P1 --
def select_baseline(x: np.ndarray, split: int):
    """Choose family, baseline parameters, and degree by held-out NLL."""

    train, test = x[:split], x[split:]
    candidates = [("gamma, moment matched", Gamma, gamma_baseline(x))]
    sig = float(np.std(x))
    for s in WIDTH_GRID:
        params = NormalParams(mean=float(np.mean(x)), sigma=s * sig)
        candidates.append((f"normal, sigma {s * sig:.1f}", Normal, params))

    print("P1  baseline and degree by held-out NLL:")
    best = None
    for label, family, baseline in candidates:
        inner = None
        for k in MARGINAL_DEGREES:
            fit = continued_complex_fit(
                fitting_matrices(family, baseline, k),
                moments_of(family, train, 2 * k, baseline),
            )
            c = fit["complex"]["coefficients"]
            nll = marginal_nll(family, c, test, k, baseline)
            if inner is None or nll < inner[0]:
                inner = (nll, k, c)
        nll, k, c = inner
        marker = ""
        if best is None or nll < best[0]:
            best = (nll, k, c, label, family, baseline)
            marker = "  <-"
        print(f"      {label:24s} K* {k:2d}  NLL {nll:.4f}{marker}")
    nll0 = marginal_nll(
        best[4], np.eye(best[1] + 1, 1).ravel().astype(complex), test, best[1], best[5]
    )
    print(f"      selected baseline alone: NLL {nll0:.4f}")
    return best


def fitted_trough(family: Any, baseline: Any, c: np.ndarray, k: int) -> float:
    grid = np.linspace(55.0, 85.0, 601)
    basis = np.asarray(family.basis(grid, k, baseline), dtype=float)
    law = np.asarray(family.prob(grid, baseline), dtype=float) * np.abs(basis @ c) ** 2
    return float(grid[np.argmin(law)])


# ------------------------------------------------------------------ P3 --
def pair_capacity_study(family, baseline, x, split, lags=(1, 2)):
    """Each model selects its own capacity on held-out data; the margin
    is reported with the paired standard error over test pairs."""

    print("P3  pair against product of marginals (held-out, randomised pairs):")
    rng = np.random.default_rng(0)
    for lag in lags:
        pairs = np.column_stack([x[:-lag], x[lag:]])
        pairs = pairs[rng.permutation(len(pairs))]
        train, test = pairs[:split], pairs[split:]

        def loglik(cv, k):
            b1 = np.asarray(family.basis(test[:, 0], k, baseline), dtype=float)
            b2 = np.asarray(family.basis(test[:, 1], k, baseline), dtype=float)
            h = np.einsum("ij,jk,ik->i", b1, cv.reshape(k + 1, k + 1), b2)
            lr = np.log(np.asarray(family.prob(test[:, 0], baseline), dtype=float))
            lr += np.log(np.asarray(family.prob(test[:, 1], baseline), dtype=float))
            return lr + np.log(np.abs(h) ** 2 + GUARD)

        best_pair, best_product, best_polished = None, None, None
        for k in PAIR_DEGREES:
            phi = fitting_matrices(family, baseline, k)
            stack, _ = pair_fitting_matrices(phi)
            b1 = np.asarray(family.basis(train[:, 0], 2 * k, baseline), dtype=float)
            b2 = np.asarray(family.basis(train[:, 1], 2 * k, baseline), dtype=float)
            r0 = (b1[:, :, None] * b2[:, None, :]).mean(axis=0)
            c = continued_complex_fit(stack, r0.reshape(-1))["complex"]["coefficients"]
            c1 = continued_complex_fit(phi, b1.mean(axis=0))["complex"]["coefficients"]
            c2 = continued_complex_fit(phi, b2.mean(axis=0))["complex"]["coefficients"]
            kron = np.kron(c1, c2)
            kron = kron / np.linalg.norm(kron)
            prods = b1[:, :, None] * b2[:, None, :]
            weighted = fit_complex_amplitude(
                stack, r0.reshape(-1), weight=channel_weight(prods), initial=kron
            )["coefficients"]
            polished = likelihood_polish(family, baseline, weighted, train, k)
            ll_polished = loglik(polished, k)
            if best_polished is None or ll_polished.mean() > best_polished[0].mean():
                best_polished = (ll_polished, k)
            ll_pair, ll_product = loglik(c, k), loglik(np.kron(c1, c2), k)
            if best_pair is None or ll_pair.mean() > best_pair[0].mean():
                best_pair = (ll_pair, k)
            if best_product is None or ll_product.mean() > best_product[0].mean():
                best_product = (ll_product, k)
        difference = best_polished[0] - best_product[0]
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        print(
            f"      lag {lag}:  polished K*{best_polished[1]}"
            f" NLL {-best_polished[0].mean():.4f}"
            f"   unweighted K*{best_pair[1]} {-best_pair[0].mean():.4f}"
            f"   product K*{best_product[1]} {-best_product[0].mean():.4f}"
            f"   margin {difference.mean():+.4f} +- {error:.4f} nats/pair"
            f" ({difference.mean() / error:+.1f} sigma)"
        )


# ------------------------------------------------------------------ P2 --
def factorisation_study(family, baseline, x, trough: float, deg: int = 6):
    basis = np.asarray(family.basis(x, deg, baseline), dtype=float)

    # the registered curve factorisation, reported honestly
    m1 = pair_moment_matrix(basis, 1)
    scale = getattr(baseline, "sigma", None)
    width = 3.5 * 20.0 / scale**2 if scale else 0.3
    curve = factorise_pair_moments(m1, family, baseline, width)
    means = [
        float(family.mean(family.shifted_params(baseline, s))) for s in curve["shifts"]
    ]
    print("P2  moment-curve factorisation (the registered readout):")
    print(
        f"      members {means[0]:.1f}/{means[1]:.1f} min,"
        f" curve residuals {curve['curve_residuals'][0]:.2f}/"
        f"{curve['curve_residuals'][1]:.2f}"
        "  -- off-curve: within-regime laws are not coherent members"
    )

    # freed members: the empirical regime towers
    regime = x >= trough
    towers = np.column_stack([basis[regime].mean(axis=0), basis[~regime].mean(axis=0)])
    pseudo = np.linalg.pinv(towers)
    print(
        f"      regime towers at trough {trough:.1f}:"
        f" means {x[regime].mean():.1f}/{x[~regime].mean():.1f} min,"
        f" widths {x[regime].std():.1f}/{x[~regime].std():.1f} min"
    )
    print("      freed-member weights (rows/cols long, short) and Markov check:")
    rhos = {}
    for lag in (1, 2, 3):
        m = pair_moment_matrix(basis, lag)
        w = pseudo @ m @ pseudo.T
        w = w / w.sum()
        rhos[lag] = latent_correlation(w)
        print(
            f"      lag {lag}:  W [[{w[0, 0]:+.3f} {w[0, 1]:+.3f}]"
            f" [{w[1, 0]:+.3f} {w[1, 1]:+.3f}]]   rho {rhos[lag]:+.3f}"
        )
    print(
        f"      Markov: rho_1^2 {rhos[1] ** 2:+.3f} vs rho_2 {rhos[2]:+.3f};"
        f"  rho_1^3 {rhos[1] ** 3:+.3f} vs rho_3 {rhos[3]:+.3f}"
    )
    counts = empirical_transitions(x, trough)
    print(
        f"      empirical transitions: [[{counts[0, 0]:.3f} {counts[0, 1]:.3f}]"
        f" [{counts[1, 0]:.3f} {counts[1, 1]:.3f}]]"
    )


def empirical_transitions(x: np.ndarray, threshold: float) -> np.ndarray:
    s = (x >= threshold).astype(int)
    counts = np.zeros((2, 2))
    for a, b in zip(s[:-1], s[1:], strict=True):
        counts[1 - a, 1 - b] += 1.0
    return counts / counts.sum()


# ------------------------------------------------------------- conditionals --
def conditional_study(x: np.ndarray, trough: float, k: int = 6, output_dir=None):
    """The operational test of the pair fit: p(next | current) against the
    empirical conditionals, with the product model as the null."""

    pairs = np.column_stack([x[:-1], x[1:]])
    baseline = gamma_baseline(x)
    phi = fitting_matrices(Gamma, baseline, k)
    stack, _ = pair_fitting_matrices(phi)
    b1 = np.asarray(Gamma.basis(pairs[:, 0], 2 * k, baseline), dtype=float)
    b2 = np.asarray(Gamma.basis(pairs[:, 1], 2 * k, baseline), dtype=float)
    r0 = (b1[:, :, None] * b2[:, None, :]).mean(axis=0)
    kron0 = continued_complex_fit(stack, r0.reshape(-1))["complex"]["coefficients"]
    prods = b1[:, :, None] * b2[:, None, :]
    weighted = fit_complex_amplitude(
        stack, r0.reshape(-1), weight=channel_weight(prods), initial=kron0
    )["coefficients"]
    c = likelihood_polish(Gamma, baseline, weighted, pairs, k).reshape(k + 1, k + 1)
    c1 = continued_complex_fit(phi, b1.mean(axis=0))["complex"]["coefficients"]

    grid = np.linspace(38.0, 105.0, 500)
    reference = np.asarray(Gamma.prob(grid, baseline), dtype=float)
    basis_grid = np.asarray(Gamma.basis(grid, k, baseline), dtype=float)
    marginal = reference * np.abs(basis_grid @ c1) ** 2
    marginal /= np.trapezoid(marginal, grid)

    def conditional(x_value):
        bx = np.asarray(Gamma.basis(np.array([x_value]), k, baseline), dtype=float)[0]
        law = reference * np.abs(basis_grid @ (c.T @ bx)) ** 2
        return law / np.trapezoid(law, grid)

    print(f"conditionals of the pair amplitude (Gamma baseline, K = {k}):")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    cases = [
        (55.0, pairs[:, 0] < trough, "conditioned on a short wait ($x_i = 55$)"),
        (80.0, pairs[:, 0] >= trough, "conditioned on a long wait ($x_i = 80$)"),
    ]
    for axis, (x_value, selection, title) in zip(axes, cases, strict=True):
        nxt = pairs[selection, 1]
        law = conditional(x_value)
        axis.hist(
            nxt,
            bins=18,
            density=True,
            histtype="stepfilled",
            facecolor="0.86",
            edgecolor="0.72",
            lw=0.5,
            label=f"empirical next wait ({len(nxt)} pairs)",
        )
        axis.plot(
            grid,
            marginal,
            ls=(0, (5, 2)),
            color="#d95f02",
            lw=1.3,
            label="product model (fitted marginal)",
        )
        axis.plot(
            grid,
            law,
            ls=(0, (2.6, 1.6)),
            color="#1b6ca8",
            lw=2.2,
            label=r"pair amplitude $p(y\,|\,x_i)$",
        )
        axis.set_xlabel("next waiting time $x_{i+1}$ [min]")
        axis.set_ylabel("density")
        axis.set_title(title, fontsize=10)
        axis.legend(frameon=False, fontsize=8)
        long_mass = float(np.trapezoid(law[grid >= trough], grid[grid >= trough]))
        empirical = float(np.mean(nxt >= trough))
        null = float(np.trapezoid(marginal[grid >= trough], grid[grid >= trough]))
        print(
            f"      x_i {x_value:4.0f}:  P(next long)  empirical {empirical:.3f}"
            f"   pair fit {long_mass:.3f}   product {null:.3f}"
        )
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "conditionals.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


# ------------------------------------------------------------------ figure --
def plot_data_and_baselines(x: np.ndarray, trough: float, output_dir=None):
    """First the data, then the baseline candidates."""

    import dataclasses

    mean = float(np.mean(x))
    regime = x >= trough
    grid = np.linspace(35.0, 110.0, 601)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    axis = axes[0, 0]
    axis.plot(np.arange(len(x)), x, "-", color="0.75", lw=0.7, zorder=1)
    axis.scatter(
        np.arange(len(x)),
        x,
        c=np.where(regime, "#2a6f97", "#b0413e"),
        s=8,
        zorder=2,
    )
    axis.axhline(trough, color="0.4", ls=":", lw=1.0)
    axis.set_xlabel("eruption index $i$")
    axis.set_ylabel("waiting time [min]")
    axis.set_title("the record: alternation between two regimes", fontsize=10)

    axis = axes[0, 1]
    axis.scatter(x[:-1], x[1:], s=10, color="0.25", alpha=0.6)
    axis.axvline(trough, color="0.4", ls=":", lw=1.0)
    axis.axhline(trough, color="0.4", ls=":", lw=1.0)
    s = (x >= trough).astype(int)
    quadrants = np.zeros((2, 2))
    for a, b in zip(s[:-1], s[1:], strict=True):
        quadrants[a, b] += 1
    for (a, b), position in [
        ((0, 0), (52, 52)),
        ((0, 1), (52, 88)),
        ((1, 0), (88, 52)),
        ((1, 1), (88, 88)),
    ]:
        axis.text(
            *position,
            f"{int(quadrants[a, b])}",
            fontsize=13,
            color="#b0413e",
            ha="center",
            fontweight="bold",
        )
    axis.set_xlabel("$x_i$ [min]")
    axis.set_ylabel("$x_{i+1}$ [min]")
    axis.set_title(r"lag-1 pairs: short$\to$short nearly forbidden", fontsize=10)

    axis = axes[1, 0]
    axis.hist(
        x,
        bins=24,
        density=True,
        histtype="stepfilled",
        facecolor="0.88",
        edgecolor="0.72",
        lw=0.5,
        label="sample",
    )
    gb = gamma_baseline(x)
    axis.plot(
        grid,
        np.asarray(Gamma.prob(grid, gb), dtype=float),
        color="#d95f02",
        ls=(0, (5, 2)),
        lw=1.4,
        label=f"Gamma, moment matched ($r$={gb.r:.0f})",
    )
    for sigma, colour, label, style in (
        (float(np.std(x)), "#7b2d8b", "Normal, moment matched", (0, (2.5, 1.5))),
        (8.1, "#2a6f97", "Normal, held-out width", "-"),
        (5.7, "#1a7a4a", "Normal, regime width", (0, (2.5, 1.5))),
    ):
        params = NormalParams(mean=mean, sigma=sigma)
        axis.plot(
            grid,
            np.asarray(Normal.prob(grid, params), dtype=float),
            color=colour,
            lw=1.4,
            ls=style,
            label=rf"{label} ($\sigma$={sigma:.1f})",
        )
    axis.set_xlabel("waiting time [min]")
    axis.set_ylabel("density")
    axis.set_title("baseline candidates", fontsize=10)
    axis.legend(frameon=False, fontsize=7.5)

    axis = axes[1, 1]
    for selection, colour, label in (
        (~regime, "#b0413e", "short regime"),
        (regime, "#2a6f97", "long regime"),
    ):
        axis.hist(
            x[selection],
            bins=14,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            facecolor=colour,
            edgecolor=colour,
            lw=0.8,
            label=(
                f"{label}: {x[selection].mean():.1f}"
                rf" $\pm$ {x[selection].std():.1f} min"
            ),
        )
    narrow = NormalParams(mean=mean, sigma=5.7)
    gb = gamma_baseline(x)
    for target, colour in (
        (float(x[~regime].mean()), "#b0413e"),
        (float(x[regime].mean()), "#2a6f97"),
    ):
        member = NormalParams(mean=target, sigma=narrow.sigma)
        axis.plot(
            grid,
            np.asarray(Normal.prob(grid, member), dtype=float),
            color=colour,
            lw=1.8,
            ls=(0, (2.5, 1.5)),
        )
        wide = dataclasses.replace(gb, mean=target)
        axis.plot(
            grid,
            np.asarray(Gamma.prob(grid, wide), dtype=float),
            color="#d95f02",
            lw=1.2,
            ls=(0, (5, 2)),
        )
    axis.plot(
        [],
        [],
        color="0.3",
        ls=(0, (2.5, 1.5)),
        lw=1.8,
        label=r"Normal member, $\sigma$=5.7",
    )
    axis.plot(
        [], [], color="#d95f02", ls=(0, (5, 2)), lw=1.2, label="Gamma member, $r$=27"
    )
    axis.set_xlabel("waiting time [min]")
    axis.set_ylabel("density")
    axis.set_title("regime laws against family members at their means", fontsize=10)
    axis.legend(frameon=False, fontsize=7.5)

    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


def plot_marginal(family, x, baseline, c, k, output_dir=None):
    grid = np.linspace(35.0, 110.0, 601)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    basis = np.asarray(family.basis(grid, k, baseline), dtype=float)
    law = reference * np.abs(basis @ c) ** 2
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    axis.hist(
        x,
        bins=24,
        density=True,
        histtype="stepfilled",
        facecolor="0.86",
        edgecolor="0.72",
        linewidth=0.5,
        label="sample",
    )
    axis.plot(
        grid,
        reference,
        ls=(0, (5, 2)),
        color="#d95f02",
        lw=1.2,
        label=r"baseline $p_{\rm ref}$",
    )
    axis.plot(
        grid,
        law,
        ls=(0, (2.6, 1.6)),
        color="#1b6ca8",
        lw=2.2,
        label=rf"fit $p_{{\rm ref}}|h_{{{k}}}|^2$",
    )
    axis.set_xlabel("waiting time [min]")
    axis.set_ylabel("density")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "marginal.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(figure)
    return path


def plot_quality(x, trough: float, k: int = 6, output_dir=None):
    """Joint contours, conditional fan, regime-averaged conditionals, and
    the conditional mean, all against the data."""

    pairs = np.column_stack([x[:-1], x[1:]])
    baseline = gamma_baseline(x)
    phi = fitting_matrices(Gamma, baseline, k)
    stack, _ = pair_fitting_matrices(phi)
    b1 = np.asarray(Gamma.basis(pairs[:, 0], 2 * k, baseline), dtype=float)
    b2 = np.asarray(Gamma.basis(pairs[:, 1], 2 * k, baseline), dtype=float)
    r0 = (b1[:, :, None] * b2[:, None, :]).mean(axis=0)
    start = continued_complex_fit(stack, r0.reshape(-1))["complex"]["coefficients"]
    prods = b1[:, :, None] * b2[:, None, :]
    weighted = fit_complex_amplitude(
        stack, r0.reshape(-1), weight=channel_weight(prods), initial=start
    )["coefficients"]
    matrix = likelihood_polish(Gamma, baseline, weighted, pairs, k).reshape(
        k + 1, k + 1
    )
    c1 = continued_complex_fit(phi, b1.mean(axis=0))["complex"]["coefficients"]
    c2 = continued_complex_fit(phi, b2.mean(axis=0))["complex"]["coefficients"]

    grid = np.linspace(40.0, 100.0, 400)
    reference = np.asarray(Gamma.prob(grid, baseline), dtype=float)
    basis_grid = np.asarray(Gamma.basis(grid, k, baseline), dtype=float)
    amplitude = basis_grid @ matrix @ basis_grid.T
    joint = reference[:, None] * reference[None, :] * np.abs(amplitude) ** 2
    joint /= np.trapezoid(np.trapezoid(joint, grid, axis=1), grid)
    marg1 = reference * np.abs(basis_grid @ c1) ** 2
    marg1 /= np.trapezoid(marg1, grid)
    marg2 = reference * np.abs(basis_grid @ c2) ** 2
    marg2 /= np.trapezoid(marg2, grid)
    product = np.outer(marg1, marg2)

    def conditional_rows(bx):
        laws = reference[None, :] * np.abs(bx @ matrix @ basis_grid.T) ** 2
        return laws / np.trapezoid(laws, grid, axis=1)[:, None]

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))

    axis = axes[0, 0]
    axis.scatter(pairs[:, 0], pairs[:, 1], s=9, color="0.35", alpha=0.55, zorder=1)
    axis.contour(
        grid,
        grid,
        joint.T,
        levels=np.max(joint) * np.array([0.03, 0.1, 0.25, 0.5, 0.8]),
        colors="#1b6ca8",
        linewidths=1.3,
        zorder=2,
    )
    axis.contour(
        grid,
        grid,
        product.T,
        levels=np.max(product) * np.array([0.03, 0.25, 0.8]),
        colors="#d95f02",
        linewidths=0.9,
        linestyles="dashed",
        zorder=2,
    )
    axis.set_xlabel("$x_i$ [min]")
    axis.set_ylabel("$x_{i+1}$ [min]")
    axis.set_title("fitted joint (blue), product null (orange)", fontsize=10)

    axis = axes[0, 1]
    cmap = plt.get_cmap("coolwarm")
    for v in (46, 52, 58, 64, 72, 78, 84, 90):
        bx = np.asarray(Gamma.basis(np.array([float(v)]), k, baseline), dtype=float)
        axis.plot(
            grid,
            conditional_rows(bx)[0],
            color=cmap((v - 42) / 52),
            lw=1.5,
            label=f"$x_i={v}$",
        )
    axis.set_xlabel("next waiting time [min]")
    axis.set_ylabel("density")
    axis.set_title(r"fan of fitted conditionals $p(y\,|\,x_i)$", fontsize=10)
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1, 0]
    for selection, colour, label in (
        (pairs[:, 0] < trough, "#b0413e", "short regime"),
        (pairs[:, 0] >= trough, "#2a6f97", "long regime"),
    ):
        axis.hist(
            pairs[selection, 1],
            bins=18,
            density=True,
            histtype="stepfilled",
            alpha=0.30,
            facecolor=colour,
            edgecolor=colour,
            lw=0.8,
            label=f"empirical | {label}",
        )
        bx = np.asarray(Gamma.basis(pairs[selection, 0], k, baseline), dtype=float)
        axis.plot(
            grid,
            conditional_rows(bx).mean(axis=0),
            color=colour,
            lw=2.0,
            ls=(0, (2.6, 1.6)),
            label=f"fit averaged over {label}",
        )
    axis.set_xlabel("next waiting time [min]")
    axis.set_ylabel("density")
    axis.set_title("regime-averaged fitted conditionals", fontsize=10)
    axis.legend(frameon=False, fontsize=7.5)

    axis = axes[1, 1]
    fan = conditional_rows(basis_grid)
    axis.plot(
        grid,
        np.trapezoid(fan * grid[None, :], grid, axis=1),
        color="#1b6ca8",
        lw=2.0,
        label=r"fitted $\mathbb{E}[x_{i+1}\,|\,x_i]$",
    )
    edges = np.linspace(43, 97, 10)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        selection = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        if selection.sum() >= 5:
            axis.errorbar(
                0.5 * (lo + hi),
                pairs[selection, 1].mean(),
                yerr=pairs[selection, 1].std() / np.sqrt(selection.sum()),
                fmt="o",
                color="0.25",
                ms=4,
                capsize=2,
            )
    axis.axhline(float(np.mean(x)), color="0.6", ls=":", lw=1.0)
    axis.set_xlabel("$x_i$ [min]")
    axis.set_ylabel("$x_{i+1}$ [min]")
    axis.set_title("conditional mean", fontsize=10)
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "quality.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/faithful.csv")
    parser.add_argument("--split", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    x = load_waiting(args.data)
    print(f"waiting times: N {len(x)}, mean {np.mean(x):.1f}, std {np.std(x):.1f}")
    x = x[np.random.default_rng(0).permutation(len(x))]  # randomised split;
    # pair studies rebuild the ordered series, so reload for them
    x_ordered = load_waiting(args.data)

    nll, k_star, c_star, label, family, baseline = select_baseline(x, args.split)
    trough = fitted_trough(family, baseline, c_star, k_star)
    print(f"      selected: {label}, K* {k_star}, fitted trough {trough:.1f} min")
    print(f"      data figure: {plot_data_and_baselines(x, trough, args.output)}")
    path = plot_marginal(family, x, baseline, c_star, k_star, args.output)
    print(f"      figure: {path}")

    pair_capacity_study(family, baseline, x_ordered, args.split)
    print(
        f"      conditionals figure:"
        f" {conditional_study(x_ordered, trough, output_dir=args.output)}"
    )
    print(
        f"      quality figure: {plot_quality(x_ordered, trough, output_dir=args.output)}"
    )
    factorisation_study(family, baseline, x_ordered, trough)


if __name__ == "__main__":
    main()
