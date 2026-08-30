"""Old Faithful as a diffusion model: the slice tower, the reverse
sampler, and the pyramid.

The modern record (30,006 intervals) fitted not once at t = 0 but as a
tower of slices along the reference process, then sampled by the
reverse Doob chain.  Registered predictions:

  P-A  (pipeline) the reverse chain, run from the product baseline down
       the schedule with the fitted slice tilts, generates pairs whose
       marginal, conditionals, and transition quadrants match the
       held-out record within sampling error;
  P-B  (stratification) the Schmidt spectrum of the fitted slice
       coefficient matrices melts monotonically along the schedule, and
       the latent (second) channel dies at the predicted time
       t* = log(R_11(0)/2SE)/2 where the decayed cross channel crosses
       its sampling floor;
  P-C' (telescoping equivalence) because the deterministic targets are
       semigroup-consistent, the reverse-chain law coincides with the
       direct t = 0 fit: the distance of chain samples to the terminal
       fitted law equals the distance of exact samples from that law;
  P-D  (pyramid) fitting each slice on the active corner
       {(j,k): exp(-(j+k)t) >= theta} only -- the pyramidal fit --
       reaches the same terminal quality as the direct weighted +
       polished fit at comparable cost (parity expected: the target is
       benign; the pyramid's case for superiority is chambered targets).

Protocol as in applications/old_faithful_modern.py: Poisson lattice
baseline, K = 6, random pair split, weighted coefficient fits with a
likelihood polish at the terminal slice.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
)
from applications.old_faithful_modern import (
    SHORT_THRESHOLD,
    channel_weight,
    clean_intervals,
    likelihood_polish,
)
from applications.two_site_diffusion import pair_fitting_matrices
from nefqvf import Poisson, PoissonParams

FIGURE_SUBDIRECTORY = "old-faithful-diffusion"
K = 6
SLICES = 12
T_MAX = 3.0
PROPOSALS = 32
THETA = 0.05  # pyramid activation threshold on exp(-(j+k)t)


def schedule() -> np.ndarray:
    """Slice times from T_MAX down to 0, dense near 0."""

    return -np.log(np.linspace(np.exp(-T_MAX), 1.0, SLICES))


def degrees_matrix(k: int) -> np.ndarray:
    j, m = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
    return (j + m).astype(float)


def pair_moments(family, baseline, pairs, k_max):
    b1 = np.asarray(family.basis(pairs[:, 0], k_max, baseline), dtype=float)
    b2 = np.asarray(family.basis(pairs[:, 1], k_max, baseline), dtype=float)
    return b1[:, :, None] * b2[:, None, :]


# ------------------------------------------------------------- the tower --
def fit_tower(family, baseline, train, k: int):
    """Weighted slice fits on the deterministic target path, warm-started
    down the schedule; the terminal slice gets the likelihood polish."""

    phi = fitting_matrices(family, baseline, k)
    stack, target_degrees = pair_fitting_matrices(phi)
    prods = pair_moments(family, baseline, train, 2 * k)
    r0 = prods.mean(axis=0).reshape(-1)
    base_weight = channel_weight(prods)

    m1 = np.asarray(family.basis(train[:, 0], 2 * k, baseline), dtype=float).mean(
        axis=0
    )
    m2 = np.asarray(family.basis(train[:, 1], 2 * k, baseline), dtype=float).mean(
        axis=0
    )
    c1 = continued_complex_fit(phi, m1)["complex"]["coefficients"]
    c2 = continued_complex_fit(phi, m2)["complex"]["coefficients"]
    c = np.kron(c1, c2)
    c = c / np.linalg.norm(c)

    tower = {}
    for t in schedule()[::-1][::-1]:  # from T_MAX down to 0
        decay = np.exp(-target_degrees * t)
        target = decay * r0
        # GLS weights of the decayed targets: variances scale with decay^2.
        # Channels decayed below the cap carry no numerical information at
        # this slice and are capped -- exact GLS is itself the pyramid.
        scale = np.diag(1.0 / np.maximum(decay[1:], 1e-3) ** 2)
        weight = scale @ base_weight @ scale
        weight = weight / weight.max()
        c = fit_complex_amplitude(stack, target, weight=weight, initial=c)[
            "coefficients"
        ]
        tower[float(t)] = c.copy()
    # terminal polish on the raw data
    polished = likelihood_polish(family, baseline, tower[0.0], train, k)
    tower[0.0] = polished
    return tower


# ---------------------------------------------------------------- P-B --
def stratification(tower, train, family, baseline, k):
    print("P-B  Schmidt spectrum of the slice coefficient matrices:")
    times = sorted(tower, reverse=True)
    sigma2_terminal = None
    for t in times:
        s = np.linalg.svd(tower[t].reshape(k + 1, k + 1), compute_uv=False)
        s = s / s[0]
        if t == 0.0:
            sigma2_terminal = s[1]
        print(
            f"      t {t:5.2f}:  sigma2/sigma1 {s[1]:8.5f}   sigma3/sigma1 {s[2]:8.5f}"
        )
    prods = pair_moments(family, baseline, train, 2 * k)
    r11 = prods[:, 1, 1]
    signal = abs(float(r11.mean()))
    floor = 2.0 * float(r11.std()) / np.sqrt(len(r11))
    t_star = 0.5 * np.log(signal / floor)
    print(
        f"      predicted death of the latent channel: t* = {t_star:.2f}"
        f"  (R_11 {signal:.4f}, 2SE {floor:.4f});"
        f" terminal sigma2/sigma1 {sigma2_terminal:.4f}"
    )
    return float(t_star)


# ------------------------------------------------------------- sampling --
def model_pmf(c, family, baseline, k, grid):
    ref = np.asarray(family.prob(grid, baseline), dtype=float)
    basis = np.asarray(family.basis(grid, k, baseline), dtype=float)
    amplitude = basis @ c.reshape(k + 1, k + 1) @ basis.T
    pmf = ref[:, None] * ref[None, :] * np.abs(amplitude) ** 2
    return pmf / pmf.sum()


def sample_pmf(pmf, grid, size, rng):
    flat = rng.choice(pmf.size, size=size, p=pmf.reshape(-1))
    i, j = np.divmod(flat, pmf.shape[1])
    return np.column_stack([grid[i], grid[j]])


def reverse_chain(tower, family, baseline, k, size, rng):
    """Reverse Doob sampling: propose with the one-shot kernel per site,
    reweight by the next slice's tilt, resample."""

    times = sorted(tower, reverse=True)
    x = np.column_stack(
        [
            np.asarray(family.sample(baseline, size, rng=rng)),
            np.asarray(family.sample(baseline, size, rng=rng)),
        ]
    ).astype(float)
    for t_from, t_to in zip(times[:-1], times[1:], strict=True):
        delta = t_from - t_to
        matrix = tower[t_to].reshape(k + 1, k + 1)
        proposals = np.empty((size, PROPOSALS, 2))
        for p in range(PROPOSALS):
            for site in (0, 1):
                proposals[:, p, site] = family.one_shot_sample(
                    x[:, site], delta, params=baseline, rng=rng
                )
        flat = proposals.reshape(-1, 2)
        b1 = np.asarray(family.basis(flat[:, 0], k, baseline), dtype=float)
        b2 = np.asarray(family.basis(flat[:, 1], k, baseline), dtype=float)
        h = np.einsum("ij,jk,ik->i", b1, matrix, b2)
        weights = (np.abs(h) ** 2 + 1e-12).reshape(size, PROPOSALS)
        weights = weights / weights.sum(axis=1, keepdims=True)
        pick = (weights.cumsum(axis=1) > rng.random((size, 1))).argmax(axis=1)
        x = proposals[np.arange(size), pick]
    return x


def quadrants(pairs):
    s = (pairs >= SHORT_THRESHOLD).astype(int)
    counts = np.zeros((2, 2))
    for a, b in zip(s[:, 0], s[:, 1], strict=True):
        counts[1 - a, 1 - b] += 1.0
    return counts / counts.sum()


def lattice_tv(pairs_a, pairs_b, grid):
    lo = grid[0]
    n = len(grid)

    def hist(p):
        h = np.zeros((n, n))
        i = np.clip(np.round(p[:, 0]).astype(int) - lo, 0, n - 1)
        j = np.clip(np.round(p[:, 1]).astype(int) - lo, 0, n - 1)
        np.add.at(h, (i, j), 1.0)
        return h / h.sum()

    return 0.5 * float(np.abs(hist(pairs_a) - hist(pairs_b)).sum())


def generation_study(tower, family, baseline, k, test, rng):
    grid = np.arange(40, 151)
    size = 30_000
    generated = reverse_chain(tower, family, baseline, k, size, rng)
    pmf = model_pmf(tower[0.0], family, baseline, k, grid)
    exact = sample_pmf(pmf, grid, size, rng)

    def tv_to_model(p):
        lo = grid[0]
        n = len(grid)
        h = np.zeros((n, n))
        i = np.clip(np.round(p[:, 0]).astype(int) - lo, 0, n - 1)
        j = np.clip(np.round(p[:, 1]).astype(int) - lo, 0, n - 1)
        np.add.at(h, (i, j), 1.0)
        return 0.5 * float(np.abs(h / h.sum() - pmf).sum())

    print("P-A/P-C'  reverse-chain generation:")
    print(
        f"      TV(chain, terminal fit) {tv_to_model(generated):.4f}"
        f"   TV(exact model samples, terminal fit) {tv_to_model(exact):.4f}"
        "   (P-C': these should coincide)"
    )
    w_gen, w_test = quadrants(generated), quadrants(np.round(test))
    print(
        f"      quadrants generated [[{w_gen[0, 0]:.3f} {w_gen[0, 1]:.3f}]"
        f" [{w_gen[1, 0]:.3f} {w_gen[1, 1]:.3f}]]"
        f"   held-out [[{w_test[0, 0]:.3f} {w_test[0, 1]:.3f}]"
        f" [{w_test[1, 0]:.3f} {w_test[1, 1]:.3f}]]"
    )
    for x_val, window in ((65.0, 3), (95.0, 2)):
        sel_g = np.abs(generated[:, 0] - x_val) <= window
        sel_t = np.abs(test[:, 0] - x_val) <= window
        print(
            f"      E[next | {x_val:.0f}]  generated"
            f" {generated[sel_g, 1].mean():6.1f}"
            f"   held-out {test[sel_t, 1].mean():6.1f}"
        )
    print(
        f"      TV(generated, held-out pairs) {lattice_tv(generated, np.round(test), grid):.4f}"
        f"   TV(exact model samples, held-out) {lattice_tv(exact, np.round(test), grid):.4f}"
    )
    return generated


# ---------------------------------------------------------------- P-D --
def pyramid_fit(family, baseline, train, k: int):
    """Fit down the schedule with only the active corner
    {(j,k): exp(-(j+k)t) >= THETA} free, growing as t decreases."""

    phi = fitting_matrices(family, baseline, k)
    stack, target_degrees = pair_fitting_matrices(phi)
    prods = pair_moments(family, baseline, train, 2 * k)
    r0 = prods.mean(axis=0).reshape(-1)
    base_weight = channel_weight(prods)
    coeff_degrees = degrees_matrix(k).reshape(-1)

    c_active = None
    active_previous = None
    for t in sorted(schedule(), reverse=True):
        cap = np.log(1.0 / THETA) / max(t, 1e-9)
        active = np.where(coeff_degrees <= cap)[0]
        if len(active) < 2:
            active = np.where(coeff_degrees <= 1)[0]
        stack_r = stack[:, active][:, :, active]
        start = np.zeros(len(active), dtype=complex)
        start[0] = 1.0
        if c_active is not None:
            lookup = {index: i for i, index in enumerate(active_previous)}
            for i, index in enumerate(active):
                if index in lookup:
                    start[i] = c_active[lookup[index]]
            start = start / np.linalg.norm(start)
        decay = np.exp(-target_degrees * t)
        target = decay * r0
        scale = np.diag(1.0 / np.maximum(decay[1:], 1e-3) ** 2)
        weight = scale @ base_weight @ scale
        weight = weight / weight.max()
        c_active = fit_complex_amplitude(stack_r, target, weight=weight, initial=start)[
            "coefficients"
        ]
        active_previous = active
    c_full = np.zeros((k + 1) ** 2, dtype=complex)
    c_full[active_previous] = c_active
    return likelihood_polish(family, baseline, c_full, train, k)


def held_out_nll(family, baseline, c, pairs, k):
    b1 = np.asarray(family.basis(pairs[:, 0], k, baseline), dtype=float)
    b2 = np.asarray(family.basis(pairs[:, 1], k, baseline), dtype=float)
    h = np.einsum("ij,jk,ik->i", b1, c.reshape(k + 1, k + 1), b2)
    lr = np.log(np.asarray(family.prob(pairs[:, 0], baseline), dtype=float))
    lr += np.log(np.asarray(family.prob(pairs[:, 1], baseline), dtype=float))
    return -float(np.mean(lr + np.log(np.abs(h) ** 2 + 1e-300)))


# ---------------------------------------------------------------- figures --
def output_directory(output_dir=None):
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_tower(tower, t_star, k, output_dir=None):
    """The stratification page: Schmidt melting with the predicted birth,
    and the coefficient pyramid filling in along the schedule."""

    times = np.array(sorted(tower, reverse=True))
    spectra = np.array(
        [np.linalg.svd(tower[t].reshape(k + 1, k + 1), compute_uv=False) for t in times]
    )
    spectra = spectra / spectra[:, :1]

    figure = plt.figure(figsize=(11.0, 6.6))
    grid_spec = figure.add_gridspec(2, 4, height_ratios=[1.15, 1.0])

    axis = figure.add_subplot(grid_spec[0, :])
    axis.semilogy(
        times,
        spectra[:, 1],
        "o-",
        color="#1b6ca8",
        label=r"$\sigma_2/\sigma_1$ (latent channel)",
    )
    axis.semilogy(
        times, spectra[:, 2], "s-", color="#b0413e", label=r"$\sigma_3/\sigma_1$"
    )
    axis.axvline(t_star, color="0.4", ls=":", lw=1.2)
    axis.text(
        t_star - 0.03,
        2e-3,
        r"predicted birth $t^\ast$",
        fontsize=9,
        color="0.3",
        rotation=90,
        va="bottom",
        ha="right",
    )
    axis.invert_xaxis()
    axis.set_xlabel("slice time $t$")
    axis.set_ylabel("Schmidt value / leading")
    axis.set_title(
        "entanglement stratification: channels are born on schedule",
        fontsize=10,
    )
    axis.legend(frameon=False, fontsize=9)

    show = [times[0], 1.17, 0.42, 0.0]
    for column, t_show in enumerate(show):
        key = min(tower, key=lambda u: abs(u - t_show))
        axis = figure.add_subplot(grid_spec[1, column])
        magnitude = np.abs(tower[key].reshape(k + 1, k + 1))
        image = axis.imshow(
            np.log10(magnitude / magnitude.max() + 1e-8),
            origin="lower",
            cmap="viridis",
            vmin=-6,
            vmax=0,
        )
        axis.set_title(rf"$|c_{{jk}}|$ at $t={key:.2f}$", fontsize=9)
        axis.set_xlabel("$k$")
        if column == 0:
            axis.set_ylabel("$j$")
        if column == len(show) - 1:
            figure.colorbar(
                image, ax=axis, fraction=0.046, label=r"$\log_{10}$ relative magnitude"
            )
    figure.tight_layout()
    directory = output_directory(output_dir)
    figure.savefig(directory / "tower.pdf")
    figure.savefig(directory / "tower.png", dpi=140)
    plt.close(figure)
    return directory / "tower.pdf"


def plot_generation(generated, test, pmf, grid, output_dir=None):
    """The generation page: chain samples against the terminal fit, and
    generated against held-out statistics."""

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axis = axes[0]
    edges = np.arange(grid[0] - 0.5, grid[-1] + 1.5, 1.0)
    histogram, _, _ = np.histogram2d(
        generated[:, 0], generated[:, 1], bins=[edges, edges]
    )
    axis.imshow(
        np.log10(histogram.T + 1.0),
        origin="lower",
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        cmap="Greys",
        aspect="equal",
    )
    axis.contour(
        grid,
        grid,
        pmf.T,
        levels=np.max(pmf) * np.array([1e-4, 1e-3, 0.01, 0.1, 0.5]),
        colors="#1b6ca8",
        linewidths=1.0,
    )
    axis.set_xlabel("$w_i$ [min]")
    axis.set_ylabel("$w_{i+1}$ [min]")
    axis.set_title("reverse-chain samples, terminal fit as contours", fontsize=10)

    axis = axes[1]
    bins = np.arange(39.5, 151.5, 1.0)
    axis.hist(
        test[:, 1],
        bins=bins,
        density=True,
        histtype="stepfilled",
        facecolor="0.85",
        edgecolor="0.7",
        lw=0.4,
        label="held-out record",
    )
    axis.hist(
        generated[:, 1],
        bins=bins,
        density=True,
        histtype="step",
        edgecolor="#1b6ca8",
        lw=1.6,
        label="reverse chain",
    )
    axis.set_yscale("log")
    axis.set_ylim(1e-6, 0.1)
    axis.set_xlabel("interval [min]")
    axis.set_ylabel("density (log)")
    axis.set_title("generated against held-out marginal", fontsize=10)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    centres, gen_means, gen_errors, test_means, test_errors = [], [], [], [], []
    for lo in np.arange(55, 125, 5):
        centre = lo + 2.5
        sel_g = (generated[:, 0] >= lo) & (generated[:, 0] < lo + 5)
        sel_t = (test[:, 0] >= lo) & (test[:, 0] < lo + 5)
        if sel_g.sum() >= 30 and sel_t.sum() >= 30:
            centres.append(centre)
            gen_means.append(generated[sel_g, 1].mean())
            gen_errors.append(generated[sel_g, 1].std() / np.sqrt(sel_g.sum()))
            test_means.append(test[sel_t, 1].mean())
            test_errors.append(test[sel_t, 1].std() / np.sqrt(sel_t.sum()))
    axis.errorbar(
        centres,
        test_means,
        yerr=test_errors,
        fmt="o",
        color="0.25",
        ms=4,
        capsize=2,
        label="held-out record",
    )
    axis.errorbar(
        centres,
        gen_means,
        yerr=gen_errors,
        fmt="s",
        color="#1b6ca8",
        ms=4,
        capsize=2,
        label="reverse chain",
    )
    axis.set_xlabel("$w_i$ [min]")
    axis.set_ylabel("$\\mathbb{E}[w_{i+1}\\,|\\,w_i]$ [min]")
    axis.set_title("conditional means, generated against data", fontsize=10)
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    directory = output_directory(output_dir)
    figure.savefig(directory / "generation.pdf")
    figure.savefig(directory / "generation.png", dpi=140)
    plt.close(figure)
    return directory / "generation.pdf"


# ------------------------------------------------------------------ main --
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/geysertimes_oldfaithful.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    w, pairs = clean_intervals(args.data)
    rng = np.random.default_rng(args.seed)
    pairs = pairs[rng.permutation(len(pairs))]
    split = int(2 * len(pairs) / 3)
    train, test = np.round(pairs[:split]), np.round(pairs[split:])
    family, baseline = Poisson, PoissonParams(mean=float(np.mean(w)))
    print(
        f"tower: {SLICES} slices on [0, {T_MAX}], K {K},"
        f" train pairs {len(train)}, test pairs {len(test)}"
    )

    tower = fit_tower(family, baseline, train, K)
    t_star = stratification(tower, train, family, baseline, K)
    generated = generation_study(tower, family, baseline, K, test, rng)
    grid = np.arange(40, 151)
    pmf = model_pmf(tower[0.0], family, baseline, K, grid)
    print(f"tower figure: {plot_tower(tower, t_star, K)}")
    print(f"generation figure: {plot_generation(generated, test, pmf, grid)}")

    direct = tower[0.0]
    pyramid = pyramid_fit(family, baseline, train, K)
    print("P-D  pyramid against the direct fit (held-out NLL):")
    print(
        f"      direct {held_out_nll(family, baseline, direct, test, K):.4f}"
        f"   pyramid {held_out_nll(family, baseline, pyramid, test, K):.4f}"
    )


if __name__ == "__main__":
    main()
