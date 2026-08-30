"""Old Faithful, the modern record: 30,006 intervals from GeyserTimes.

Eruption entries 2016--2024 fetched from the GeyserTimes API
(https://www.geysertimes.org/api/v5/entries/START/END/Old+Faithful,
quarterly chunks; the raw entries are in data/geysertimes_oldfaithful.csv).
Cleaning: primary entries only, exact or near-start times, no approximate
or questionable reports, intervals restricted to [40, 150] minutes so
data gaps never masquerade as intervals.  Pairs require two consecutive
valid intervals.  The intervals are minute-quantised with variance 98
against mean 95 (Fano factor 1.03), so the Poisson baseline on the
minute lattice is nearly moment-matched.

Registered predictions and outcomes:

  P1m  the marginal resolves the 5% short mode; the lattice Poisson
       baseline is competitive with the continuous families --
       CONFIRMED (Poisson within 0.005 nats of the best baseline);
  P2m  the pair amplitude beats the product decisively -- CONFIRMED
       ONLY for the weighted coefficient objective (+0.043 +- 0.006
       nats/pair, +7.7 sigma, weighted pair K=8 vs product K=6).  The
       UNWEIGHTED objective fails catastrophically (-67 to -76 sigma
       at every baseline, cold and warm starts reaching the same
       optimum): on data spanning +-5 sigma of the baseline the
       high-degree moment channels are tail-dominated, and lowering
       the unweighted moment energy raises the likelihood.  Inverse-
       variance weighting of the channels is what reconciles the
       coefficient objective with the likelihood;
  P3m  the fitted conditionals reproduce the ~30x short->short
       suppression and the after-short excess -- CONFIRMED
       (P(short|short) 0.0007 vs empirical 0.0000 at 5% base rate;
       E[next|short] 102.8 vs 104.1 min; the conditional mean tracks
       the binned data across 60-115 min).

Splits are blocked in time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
)
from applications.two_site_diffusion import pair_fitting_matrices
from nefqvf import Gamma, GammaParams, Normal, NormalParams, Poisson, PoissonParams

FIGURE_SUBDIRECTORY = "old-faithful"
SHORT_THRESHOLD = 75.0
PAIR_DEGREES = (4, 6, 8)


def clean_intervals(path: str = "data/geysertimes_oldfaithful.csv"):
    """Intervals and consecutive interval pairs from the raw entries."""

    rows = list(csv.DictReader(open(path)))
    good = [
        r
        for r in rows
        if r["primaryID"] == r["eruptionID"]
        and r["q"] == "0"
        and r["A"] == "0"
        and (r["exact"] == "1" or r["ns"] == "1")
    ]
    times = np.array(sorted(int(r["time"]) for r in good), dtype=float)
    gaps = np.diff(times) / 60.0
    valid = (gaps >= 40.0) & (gaps <= 150.0)
    w = gaps[valid]
    index = np.where(valid[:-1] & valid[1:])[0]
    pairs = np.column_stack([gaps[index], gaps[index + 1]])
    return w, pairs


def channel_weight(prods: np.ndarray) -> np.ndarray:
    """Inverse sampling-variance weights for the moment channels,
    excluding the R_0 row the fitter removes."""

    sd = prods.reshape(len(prods), -1).std(axis=0) / np.sqrt(len(prods))
    inverse = 1.0 / np.maximum(sd, sd[sd > 0].min()) ** 2
    return np.diag((inverse / inverse.max())[1:])


def pair_loglik(family, baseline, cv, pairs, k):
    b1 = np.asarray(family.basis(pairs[:, 0], k, baseline), dtype=float)
    b2 = np.asarray(family.basis(pairs[:, 1], k, baseline), dtype=float)
    h = np.einsum("ij,jk,ik->i", b1, cv.reshape(k + 1, k + 1), b2)
    lr = np.log(np.asarray(family.prob(pairs[:, 0], baseline), dtype=float))
    lr += np.log(np.asarray(family.prob(pairs[:, 1], baseline), dtype=float))
    return lr + np.log(np.abs(h) ** 2 + 1e-300)


# ------------------------------------------------------------------ P1m --
def marginal_study(w: np.ndarray, split: int):
    train, test = w[:split], w[split:]
    mean = float(np.mean(w))
    candidates = [
        (
            "gamma moment-matched",
            Gamma,
            GammaParams(mean=mean, r=mean**2 / float(np.var(w))),
        ),
        ("poisson lattice", Poisson, PoissonParams(mean=mean)),
        ("normal sigma 8", Normal, NormalParams(mean=mean, sigma=8.0)),
        (
            "normal moment-matched",
            Normal,
            NormalParams(mean=mean, sigma=float(np.std(w))),
        ),
    ]
    print("P1m  marginal baselines (held-out NLL):")
    for label, family, baseline in candidates:
        xs_train = np.round(train) if family is Poisson else train
        xs_test = np.round(test) if family is Poisson else test
        best = None
        for k in (4, 6, 8, 10, 12):
            moments = np.asarray(
                family.basis(xs_train, 2 * k, baseline), dtype=float
            ).mean(axis=0)
            c = continued_complex_fit(fitting_matrices(family, baseline, k), moments)[
                "complex"
            ]["coefficients"]
            basis = np.asarray(family.basis(xs_test, k, baseline), dtype=float)
            lr = np.log(np.asarray(family.prob(xs_test, baseline), dtype=float))
            nll = -float(np.mean(lr + np.log(np.abs(basis @ c) ** 2 + 1e-300)))
            if best is None or nll < best[0]:
                best = (nll, k)
        print(f"      {label:24s} K* {best[1]:2d}  NLL {best[0]:.4f}")


# ------------------------------------------------------------------ P2m --
def pair_study(w: np.ndarray, pairs: np.ndarray, split: int):
    """Weighted pair amplitude against the product of marginals; the
    unweighted fit is reported as the control that motivates the weights."""

    family, baseline = Poisson, PoissonParams(mean=float(np.mean(w)))
    train, test = np.round(pairs[:split]), np.round(pairs[split:])
    print("P2m  pair vs product at the Poisson lattice baseline:")
    best_weighted = best_unweighted = best_product = None
    for k in PAIR_DEGREES:
        phi = fitting_matrices(family, baseline, k)
        stack, _ = pair_fitting_matrices(phi)
        b1 = np.asarray(family.basis(train[:, 0], 2 * k, baseline), dtype=float)
        b2 = np.asarray(family.basis(train[:, 1], 2 * k, baseline), dtype=float)
        prods = b1[:, :, None] * b2[:, None, :]
        r0 = prods.mean(axis=0).reshape(-1)
        c1 = continued_complex_fit(phi, b1.mean(axis=0))["complex"]["coefficients"]
        c2 = continued_complex_fit(phi, b2.mean(axis=0))["complex"]["coefficients"]
        kron = np.kron(c1, c2)
        kron = kron / np.linalg.norm(kron)
        unweighted = continued_complex_fit(stack, r0)["complex"]["coefficients"]
        weighted = fit_complex_amplitude(
            stack, r0, weight=channel_weight(prods), initial=kron
        )["coefficients"]
        rows = {
            "product": pair_loglik(family, baseline, kron, test, k),
            "unweighted": pair_loglik(family, baseline, unweighted, test, k),
            "weighted": pair_loglik(family, baseline, weighted, test, k),
        }
        print(
            f"      K {k}:  product {-rows['product'].mean():.4f}"
            f"   unweighted pair {-rows['unweighted'].mean():.4f}"
            f"   weighted pair {-rows['weighted'].mean():.4f}"
        )
        if best_product is None or rows["product"].mean() > best_product[0].mean():
            best_product = (rows["product"], k)
        if (
            best_unweighted is None
            or rows["unweighted"].mean() > best_unweighted[0].mean()
        ):
            best_unweighted = (rows["unweighted"], k)
        if best_weighted is None or rows["weighted"].mean() > best_weighted[0].mean():
            best_weighted = (rows["weighted"], k, weighted)
    for label, best in (("weighted", best_weighted), ("unweighted", best_unweighted)):
        diff = best[0] - best_product[0]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(
            f"      {label} pair K*{best[1]} vs product K*{best_product[1]}:"
            f" margin {diff.mean():+.4f} +- {se:.4f}  ({diff.mean() / se:+.1f} sigma)"
        )
    return best_weighted[2], best_weighted[1]


# ------------------------------------------------------------------ P3m --
def conditional_study(w, pairs, c, k, output_dir=None):
    family, baseline = Poisson, PoissonParams(mean=float(np.mean(w)))
    matrix = c.reshape(k + 1, k + 1)
    grid = np.arange(40, 151)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    basis_grid = np.asarray(family.basis(grid, k, baseline), dtype=float)

    def conditional(x_value):
        bx = np.asarray(family.basis(np.array([x_value]), k, baseline), dtype=float)[0]
        law = reference * np.abs(basis_grid @ (matrix.T @ bx)) ** 2
        return law / law.sum()

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    print("P3m  conditionals of the weighted pair fit:")
    for axis, (x_value, window) in zip(axes[:2], ((65.0, 3), (95.0, 2)), strict=False):
        law = conditional(x_value)
        selection = np.abs(pairs[:, 0] - x_value) <= window
        nxt = pairs[selection, 1]
        axis.hist(
            nxt,
            bins=np.arange(40, 152, 3),
            density=True,
            histtype="stepfilled",
            facecolor="0.86",
            edgecolor="0.72",
            lw=0.5,
            label=rf"empirical, $|x_i-{x_value:.0f}|\leq{window}$ ({selection.sum()})",
        )
        axis.plot(
            grid,
            law,
            color="#1b6ca8",
            lw=2.0,
            ls=(0, (2.6, 1.6)),
            label=r"weighted pair fit $p(y\,|\,x_i)$",
        )
        axis.set_xlabel("next interval [min]")
        axis.set_ylabel("density")
        axis.set_title(f"conditioned on $x_i = {x_value:.0f}$", fontsize=10)
        axis.legend(frameon=False, fontsize=8)
        p_short = float(law[grid < SHORT_THRESHOLD].sum())
        print(
            f"      x_i={x_value:.0f}: P(next short) fit {p_short:.4f}"
            f" emp {float(np.mean(nxt < SHORT_THRESHOLD)):.4f}"
            f"   E[next] fit {float((grid * law).sum()):.1f} emp {nxt.mean():.1f}"
        )

    axis = axes[2]
    means = [float((grid * conditional(float(v))).sum()) for v in grid]
    axis.plot(
        grid,
        means,
        color="#1b6ca8",
        lw=2.0,
        label=r"fitted $\mathbb{E}[w_{i+1}\,|\,w_i]$",
    )
    edges = np.arange(50, 130, 5)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        selection = (pairs[:, 0] >= lo) & (pairs[:, 0] < hi)
        if selection.sum() >= 30:
            axis.errorbar(
                0.5 * (lo + hi),
                pairs[selection, 1].mean(),
                yerr=pairs[selection, 1].std() / np.sqrt(selection.sum()),
                fmt="o",
                color="0.25",
                ms=4,
                capsize=2,
            )
    axis.set_xlim(50, 130)
    axis.set_xlabel("$w_i$ [min]")
    axis.set_ylabel("$w_{i+1}$ [min]")
    axis.set_title("conditional mean", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "geysertimes-conditionals.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


def plot_data(w, pairs, output_dir=None):
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].hist(
        w,
        bins=110,
        density=True,
        histtype="stepfilled",
        facecolor="0.85",
        edgecolor="0.7",
        lw=0.4,
    )
    axes[0].set_xlabel("interval [min]")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"GeyserTimes 2016-2024: {len(w)} intervals", fontsize=10)
    axes[1].hist(
        w,
        bins=110,
        density=True,
        histtype="stepfilled",
        facecolor="0.85",
        edgecolor="0.7",
        lw=0.4,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("interval [min]")
    axes[1].set_ylabel("density (log)")
    axes[1].set_title("same, log scale: the short mode survives", fontsize=10)
    axes[2].scatter(pairs[:, 0], pairs[:, 1], s=2, color="0.3", alpha=0.15)
    axes[2].set_xlabel("$w_i$ [min]")
    axes[2].set_ylabel("$w_{i+1}$ [min]")
    axes[2].set_title(f"lag-1 pairs ({len(pairs)})", fontsize=10)
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "geysertimes-data.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/geysertimes_oldfaithful.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    w, pairs = clean_intervals(args.data)
    short = w < SHORT_THRESHOLD
    print(
        f"intervals {len(w)}, mean {w.mean():.1f}, std {w.std():.1f},"
        f" short mode {100 * short.mean():.1f}%,"
        f" pairs {len(pairs)}"
    )
    print(f"data figure: {plot_data(w, pairs, args.output)}")
    split_w = int(2 * len(w) / 3)
    split_p = int(2 * len(pairs) / 3)
    marginal_study(np.round(w), split_w)
    c, k = pair_study(w, pairs, split_p)
    print(
        f"conditionals figure: {conditional_study(w, np.round(pairs), c, k, args.output)}"
    )


if __name__ == "__main__":
    main()
