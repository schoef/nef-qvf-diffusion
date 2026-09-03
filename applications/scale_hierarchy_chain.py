"""Rung 6b: the scale-hierarchy chain -- the schedule's remaining defense.

The law: a two-state Markov chain (flip probability phi) emits, at every
site, a mixture of two normals,

    x_i | s_i  ~  1/2 N(s_i a + b, sigma^2) + 1/2 N(s_i a - b, sigma^2),

so the coarse split +-a carries the chain (low degree, bond rank two)
and the sub-split +-b is an independent coin per site: site-local
texture at higher degree, zero entanglement, and -- since e^{-t}
erases the fine scale first -- exactly the structure a schedule
diffuses away before it decides anything.  Slice laws are exact (four
normal components, means contracted, variance relaxed), and the exact
likelihood is a forward pass with mixture emissions.

The race, at fixed degree and chi = 2: the warm tower (deterministic
decayed pair targets down the schedule, per the boxed diffused
amplitude fit) against the cold t = 0 moment sweep, each finished by
identical local likelihood sweeps, scored by the held-out NLL gap per
site against the exact oracle.  Grid: texture loudness b/sigma in
{0, 1, 2} (b = 0 is the rung-6-style control) x flip probability in
{0.1, 0.3} (strong/weak skeleton).

Predictions registered before running (from the discussion of
2026-09-03):

  U   (parity) the cold sweep plus likelihood matches the warm tower
      in every cell; any failure rate is flat in d.  Rationale: the
      moments determine the bond structure and need no schedule; the
      pyramid null of rung 3 transfers.
  A   (adversarial) the cold failure rate grows with b/sigma at fixed
      coupling and with weakening coupling at fixed texture; the
      mechanism is bond-subspace lock-in -- fitting error masquerades
      as entanglement and the SVD truncation spends chi on it -- so
      the cold fit's bond subspaces diverge from the warm fit's in
      the first sweeps and extra sweeps plateau rather than converge.

Win-levels for the schedule, registered with the predictions: reduced
CPU at parity is a weak win; better degree/bond allocation (subspace
diagnostic) a strong win; convergence where the cold fit fails
conquers new territory.

Outcome (run 2026-09-03, d = 8, seed 0, artifacts/scale-hierarchy/):
U holds -- after the identical likelihood sweeps the cold fit matches
or beats the warm tower in 5/6 cells (warm leads only the quiet
control, by 0.007 nats/site); cold failures 0/6.  A fails, with a
twist: the min bond-subspace overlap does fall monotonically with
texture and with weakening coupling (flip 0.3 row: 0.73 -> 0.33 ->
0.15), but never costs likelihood -- at flip 0.3, b/sigma 2 the
subspaces are nearly orthogonal (overlap 0.15) at *identical* final
gaps (0.15015 = 0.15015), so low overlap is bond-subspace
non-identification at equal likelihood, not lock-in.  No win-level
reached: the warm tower costs ~850 s/cell against the cold sweep's
~350 s, and the loud-texture cells strand both fits together at
~0.15 nats/site -- a shared degree-7/chi-2 expressivity ceiling, not
a basin problem.

Plateau and gauge check (same day, rounds 3 -> 6 -> 9 and a second
seed on the decisive cells): every gap is converged to five digits by
round three, so the residuals are endpoints, not transients.  The
second seed FLIPS the winner at flip 0.3, b/sigma 1 (warm +0.031
against cold +0.042, reversing seed 0), so the honest form of U is:
both pipelines are basin lotteries with ~0.01-0.02 nats/site of
plateaued scatter and no systematic winner -- the schedule confers no
reliability, it buys a different ticket.  Modding conjugation out of
the overlap resolves the mechanism: at b/sigma 2 the gauge-fixed
overlap is 1.000 (warm and cold converge to the same fit at the
shared floor), while at b/sigma 1 it stays 0.47 with unequal
plateaued gaps -- genuinely different metastable basins, afflicting
both pipelines equally.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from applications.amplitude_fit_complex import fit_complex_amplitude, fitting_matrices
from applications.hmm_likelihood_sweep import likelihood_sweeps
from applications.mps_amplitude import (
    bond_fit_stack,
    bond_spectrum,
    merge_bond,
    right_canonicalise,
    split_bond,
    vacuum_state,
)
from applications.two_site_diffusion import (
    build_pair_schedule,
    empirical_pair_coefficients,
)
from nefqvf import Normal, NormalParams

BASELINE = NormalParams(0.0, 1.0)
SITES = 8
DEGREE = 7
CHI = 2
DRAWS = 40_000
A_RAW, SIGMA_RAW = 1.0, 0.4
B_VALUES = (0.0, 0.4, 0.8)
FLIP_VALUES = (0.1, 0.3)
SLICES = 8
LIK_ROUNDS = 3
MAX_LM = 80
ARTIFACTS = Path("artifacts") / "scale-hierarchy"


# ------------------------------------------------------------------- target --
class HierarchyTarget:
    """The scale-hierarchy chain, standardized to the N(0,1) frame."""

    def __init__(self, d: int, flip: float, b_raw: float):
        scale = np.sqrt(A_RAW**2 + b_raw**2 + SIGMA_RAW**2)
        self.d = d
        self.flip = flip
        self.a = A_RAW / scale
        self.b = b_raw / scale
        self.sigma = SIGMA_RAW / scale

    def sample(self, size: int, rng) -> np.ndarray:
        states = np.empty((size, self.d))
        states[:, 0] = np.where(rng.random(size) < 0.5, 1.0, -1.0)
        for site in range(1, self.d):
            flips = rng.random(size) < self.flip
            states[:, site] = np.where(flips, -states[:, site - 1], states[:, site - 1])
        coins = np.where(rng.random((size, self.d)) < 0.5, 1.0, -1.0)
        return (
            states * self.a
            + coins * self.b
            + self.sigma * rng.standard_normal((size, self.d))
        )

    def slice_params(self, t: float) -> tuple[float, float, float]:
        decay = np.exp(-t)
        variance = 1.0 + np.exp(-2.0 * t) * (self.sigma**2 - 1.0)
        return decay * self.a, decay * self.b, variance

    def truth_nll(self, x: np.ndarray, t: float) -> float:
        """Exact forward pass with mixture emissions, per sample."""

        a_t, b_t, var = self.slice_params(t)

        def emission(values: np.ndarray, s: float) -> np.ndarray:
            out = 0.0
            for sub in (b_t, -b_t):
                out = out + 0.5 * np.exp(
                    -0.5 * (values - s * a_t - sub) ** 2 / var
                ) / np.sqrt(2.0 * np.pi * var)
            return out

        stay, move = 1.0 - self.flip, self.flip
        alpha = np.stack(
            [0.5 * emission(x[:, 0], 1.0), 0.5 * emission(x[:, 0], -1.0)], axis=1
        )
        log_norm = np.zeros(len(x))
        for site in range(1, self.d):
            scale = alpha.sum(axis=1, keepdims=True)
            log_norm += np.log(scale[:, 0] + 1e-300)
            alpha = alpha / np.maximum(scale, 1e-300)
            propagated = np.stack(
                [
                    stay * alpha[:, 0] + move * alpha[:, 1],
                    move * alpha[:, 0] + stay * alpha[:, 1],
                ],
                axis=1,
            )
            emitted = np.stack(
                [emission(x[:, site], 1.0), emission(x[:, site], -1.0)], axis=1
            )
            alpha = propagated * emitted
        log_norm += np.log(alpha.sum(axis=1) + 1e-300)
        return -float(np.mean(log_norm))


def born_nll(tensors: list[np.ndarray], x: np.ndarray) -> float:
    degree = tensors[0].shape[1] - 1
    size = len(x)
    state = np.ones((size, 1), dtype=complex)
    log_reference = np.zeros(size)
    for site, tensor in enumerate(tensors):
        basis = np.asarray(Normal.basis(x[:, site], degree, BASELINE), dtype=float)
        state = np.einsum("sl,sk,lkr->sr", state, basis, tensor, optimize=True)
        log_reference += np.log(np.asarray(Normal.prob(x[:, site], BASELINE), dtype=float))
    return -float(np.mean(log_reference + np.log(np.abs(state[:, 0]) ** 2 + 1e-300)))


# ------------------------------------------------------------- moment sweeps --
def pair_statistics(x: np.ndarray, k_max: int):
    """Per-bond empirical pair coefficients and inverse-variance weights."""

    targets, weights = [], []
    for bond in range(x.shape[1] - 1):
        pairs = x[:, bond : bond + 2]
        matrix = empirical_pair_coefficients(Normal, BASELINE, pairs, k_max)
        b1 = np.asarray(Normal.basis(pairs[:, 0], k_max, BASELINE), dtype=float)
        b2 = np.asarray(Normal.basis(pairs[:, 1], k_max, BASELINE), dtype=float)
        prods = np.einsum("sj,sk->sjk", b1, b2, optimize=True).reshape(len(pairs), -1)
        sd = prods.std(axis=0) / np.sqrt(len(pairs))
        inverse = 1.0 / np.maximum(sd, sd[sd > 0].min()) ** 2
        targets.append(matrix.reshape(-1))
        weights.append(np.diag((inverse / inverse.max())[1:]))
    return targets, weights


def moment_sweep_round(state, targets, weights, phi, chi, stacks, iterations=MAX_LM):
    """One right-then-left pass of warm-started bond moment fits."""

    d = len(state)
    for bond, center_right in [(b, True) for b in range(d - 1)] + [
        (b, False) for b in range(d - 2, -1, -1)
    ]:
        merged = merge_bond(state, bond)
        shape = merged.shape
        if shape not in stacks:
            stacks[shape] = bond_fit_stack(phi, shape[0], shape[3])
        initial = merged.reshape(-1)
        initial = initial / np.linalg.norm(initial)
        fit = fit_complex_amplitude(
            stacks[shape],
            targets[bond],
            initial=initial,
            weight=weights[bond],
            max_iterations=iterations,
        )
        first, second, _ = split_bond(
            fit["coefficients"].reshape(shape), chi, center_right=center_right
        )
        state[bond], state[bond + 1] = first, second
    return state


def warm_tower(targets0, weights, degrees, phi, d, chi, schedule):
    state = vacuum_state(d, DEGREE)
    stacks: dict = {}
    for index in range(1, len(schedule)):
        t = float(schedule[index])
        decay = np.exp(-degrees * t)
        targets = [decay * tgt for tgt in targets0]
        state = right_canonicalise(state)
        rounds = 2 if abs(t) < 1e-12 else 1
        for _ in range(rounds):
            state = moment_sweep_round(state, targets, weights, phi, chi, stacks)
    return state


def cold_fit(targets0, weights, phi, d, chi, rounds=4):
    state = right_canonicalise(vacuum_state(d, DEGREE))
    stacks: dict = {}
    for _ in range(rounds):
        state = moment_sweep_round(state, targets0, weights, phi, chi, stacks)
    return state


# -------------------------------------------------------------- diagnostics --
def bond_subspace(state, bond, chi):
    merged = merge_bond(right_canonicalise(state), bond)
    chi_l, n1, n2, chi_r = merged.shape
    left, _, _ = np.linalg.svd(merged.reshape(chi_l * n1, n2 * chi_r), full_matrices=False)
    return left[:, :chi]


def subspace_overlap(state_a, state_b, chi):
    """Smallest principal-angle cosine between bond subspaces, per bond."""

    overlaps = []
    for bond in range(len(state_a) - 1):
        ua = bond_subspace(state_a, bond, chi)
        ub = bond_subspace(state_b, bond, chi)
        values = np.linalg.svd(ua.conj().T @ ub, compute_uv=False)
        overlaps.append(float(values.min()))
    return overlaps


# -------------------------------------------------------------------- cells --
def run_cell(flip: float, b_raw: float, d: int, seed: int, store: Path) -> dict:
    rng = np.random.default_rng(seed)
    target = HierarchyTarget(d, flip, b_raw)
    pool = target.sample(DRAWS, rng)
    held, train = pool[: DRAWS // 5], pool[DRAWS // 5 :]
    truth = target.truth_nll(held, 0.0)

    phi = fitting_matrices(Normal, BASELINE, DEGREE)
    k_max = phi.shape[0] - 1
    j_index, k_index = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    degrees = (j_index + k_index).astype(float)
    targets0, weights = pair_statistics(train, k_max)
    middle = (d - 2) // 2
    schedule = build_pair_schedule(targets0[middle], degrees, SLICES)

    def gap(state):
        return (born_nll(state, held) - truth) / d

    begin = time.perf_counter()
    warm = warm_tower(targets0, weights, degrees, phi, d, CHI, schedule)
    t_warm = time.perf_counter() - begin
    begin = time.perf_counter()
    cold = cold_fit(targets0, weights, phi, d, CHI)
    t_cold = time.perf_counter() - begin

    overlap_moment = subspace_overlap(warm, cold, CHI)

    begin = time.perf_counter()
    warm_l = likelihood_sweeps(warm, Normal, BASELINE, train, CHI, rounds=LIK_ROUNDS)
    t_warm_l = time.perf_counter() - begin
    begin = time.perf_counter()
    cold_l = likelihood_sweeps(cold, Normal, BASELINE, train, CHI, rounds=LIK_ROUNDS)
    t_cold_l = time.perf_counter() - begin

    row = {
        "flip": flip,
        "b_raw": b_raw,
        "d": d,
        "seed": seed,
        "gap_warm": gap(warm),
        "gap_cold": gap(cold),
        "gap_warm_lik": gap(warm_l),
        "gap_cold_lik": gap(cold_l),
        "overlap_moment": overlap_moment,
        "overlap_final": subspace_overlap(warm_l, cold_l, CHI),
        "spectrum_warm": bond_spectrum(warm_l, middle).tolist()[:4],
        "spectrum_cold": bond_spectrum(cold_l, middle).tolist()[:4],
        "cost": {
            "warm": t_warm,
            "cold": t_cold,
            "warm_lik": t_warm_l,
            "cold_lik": t_cold_l,
        },
    }
    print(
        f"flip={flip} b/sigma={b_raw / SIGMA_RAW:.0f} d={d} seed={seed}:"
        f"  moment gap warm/cold {row['gap_warm']:+.4f}/{row['gap_cold']:+.4f}"
        f"  +lik {row['gap_warm_lik']:+.5f}/{row['gap_cold_lik']:+.5f}"
        f"  min overlap {min(overlap_moment):.3f}"
        f"  cost {t_warm:.0f}/{t_cold:.0f}+{t_warm_l:.0f}/{t_cold_l:.0f}s",
        flush=True,
    )
    return row


def study_spectra() -> None:
    """Verify the degree hierarchy exists before racing anything."""

    grid = np.linspace(-6.0, 6.0, 4001)
    for b_raw in B_VALUES:
        target = HierarchyTarget(SITES, 0.1, b_raw)
        law = np.zeros_like(grid)
        for s in (1.0, -1.0):
            for sub in (target.b, -target.b):
                law += 0.25 * np.exp(
                    -0.5 * (grid - s * target.a - sub) ** 2 / target.sigma**2
                ) / np.sqrt(2.0 * np.pi * target.sigma**2)
        basis = np.asarray(Normal.basis(grid, 14, BASELINE), dtype=float)
        moments = np.trapezoid(law[:, None] * basis, grid, axis=0)
        print(
            f"b/sigma={b_raw / SIGMA_RAW:.0f}: |R_k| = "
            + " ".join(f"{abs(m):.3f}" for m in moments[1:11])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="race", choices=("spectra", "race"))
    parser.add_argument("--sites", type=int, default=SITES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.study == "spectra":
        study_spectra()
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    store = ARTIFACTS / "results.json"
    results = json.loads(store.read_text()) if store.exists() else {}
    for flip in FLIP_VALUES:
        for b_raw in B_VALUES:
            key = f"f{flip}b{b_raw}d{args.sites}s{args.seed}"
            if key in results:
                continue
            results[key] = run_cell(flip, b_raw, args.sites, args.seed, store)
            store.write_text(json.dumps(results))
    print(f"stored: {store}")


if __name__ == "__main__":
    main()
