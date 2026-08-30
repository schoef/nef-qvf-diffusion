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
    stratification(tower, train, family, baseline, K)
    generation_study(tower, family, baseline, K, test, rng)

    direct = tower[0.0]
    pyramid = pyramid_fit(family, baseline, train, K)
    print("P-D  pyramid against the direct fit (held-out NLL):")
    print(
        f"      direct {held_out_nll(family, baseline, direct, test, K):.4f}"
        f"   pyramid {held_out_nll(family, baseline, pyramid, test, K):.4f}"
    )


if __name__ == "__main__":
    main()
