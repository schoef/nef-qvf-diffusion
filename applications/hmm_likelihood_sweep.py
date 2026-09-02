"""Rung 6 of the benchmark ladder: the local likelihood sweep at d sites.

The outstanding implementation step recorded in the note: pair moments
under-determine the chain law at d >= 3 (a held-out gap of roughly
0.03 nats/site at d = 8 that more sweeps and higher degree do not
close), and the cure should be a sweep realisation of the exact
likelihood by local updates.  This module implements it.  In mixed
canonical form the amplitude is linear in the merged bond tensor,

    psi_i = <E_i, B>,   E_i = l_i (x) phi(x_b^i) (x) phi(x_{b+1}^i) (x) r_i,

with per-sample environment vectors l_i, r_i from the isometric halves
of the chain, and the Born normalisation is ||B||^2.  The local
likelihood update is therefore the one-site complex polish with a
complex design matrix; each update is followed by the SVD split back
to bond dimension chi, and the sweep passes right then left.

The rung-6 race at the note's d = 8 configuration (two-state hidden
Markov chain, separation 0.57, epsilon 0.05, K = 5, chi = 2, 60000
draws), scored by the held-out NLL gap per site against the exact
transfer-matrix likelihood:

  a0  the pair-moment tower alone (the recorded baseline);
  a   tower + local likelihood sweeps at t = 0 (the full pipeline);
  b   random MPS init + likelihood sweeps only;
  c   cold t = 0 moment sweep (no schedule) + likelihood sweeps.

Predictions registered before running:

  R6-P1  a0 reproduces a residual gap of 0.02..0.05 nats/site;
  R6-P2  a closes the gap below 0.005 nats/site;
  R6-P3  local likelihood sweeps cannot jump global basins: a majority
         of random inits (b) end >= 0.01 nats/site worse than a;
  R6-P4  c strands relative to a in a nontrivial fraction of seeds;
  R6-P5  the warm fit recovers chi = 2 at every bond (third Schmidt
         value below 1e-2 of the first).

Outcomes (run of 2026-09-02, d = 8, 60000 draws):

  R6-P1  CONFIRMED in phenomenon, overpredicted in size: the tower
         alone leaves +0.0153 nats/site (registered band 0.02..0.05).
  R6-P2  CONFIRMED decisively: three rounds of likelihood sweeps take
         the gap to +0.00015 nats/site, a hundredfold closure at 49 s.
         The note's outstanding implementation step is closed.
  R6-P3  CONFIRMED in substance, registered threshold missed: all five
         random inits strand (gaps 0.0062..0.0426, 40..280x worse
         than a), but only 2/5 exceed the registered 0.01 margin.
         Local likelihood descent cannot jump global basins at d = 8;
         the one-site universal rescue does not survive locality.
  R6-P4  REFUTED: a single cold t = 0 moment sweep initialises the
         likelihood stage as well as the full tower (0.0001..0.0009
         nats/site, 3/3 seeds).  At this separation the schedule adds
         nothing; the moment sweep alone finds the basin.
  R6-P5  PENDING: under the chi = 2 cap the spectra are vacuous; the
         chi = 4 rank-recovery run has not completed.

Division of labour, measured across rungs 3 and 6: the moment stage
supplies the basin, the likelihood stage the statistical efficiency,
and the diffusion schedule so far neither -- at benign geometry.  Its
remaining claim is a frontier claim (rung 6b): laws with a scale
hierarchy in the degree, where site-local high-degree texture
obscures a low-degree entangled skeleton at t = 0 and is diffused
away first along the schedule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from applications.amplitude_fit_complex import fitting_matrices
from applications.d_site_diffusion import (
    build_chain_target,
    fit_chain_schedule,
    model_nll,
)
from applications.mps_amplitude import (
    bond_spectrum,
    merge_bond,
    move_center_right,
    right_canonicalise,
    split_bond,
    vacuum_state,
)
from applications.two_site_diffusion import (
    build_pair_schedule,
    empirical_pair_coefficients,
)

SITES = 8
DEGREE = 5
CHI = 2
DRAWS = 60_000
SEPARATION = 0.57
EPSILON = 0.05
SLICES = 6
TAU = 0.5
SWEEPS = 2
LIK_SWEEPS = 3
SEED = 5
ARTIFACTS = Path("artifacts") / "hmm-likelihood-sweep"


# ------------------------------------------------------------- environments --
def site_basis(family, baseline, sample: np.ndarray, degree: int) -> list[np.ndarray]:
    """Per-site basis values, each of shape (size, degree + 1)."""

    return [
        np.asarray(family.basis(sample[:, site], degree, baseline), dtype=float)
        for site in range(sample.shape[1])
    ]


def left_environments(
    tensors: list[np.ndarray], basis: list[np.ndarray], bond: int
) -> np.ndarray:
    """Per-sample left environment at ``bond`` (sites < bond contracted)."""

    size = basis[0].shape[0]
    state = np.ones((size, 1), dtype=complex)
    for site in range(bond):
        state = np.einsum("sl,sk,lkr->sr", state, basis[site], tensors[site])
    return state


def right_environments(
    tensors: list[np.ndarray], basis: list[np.ndarray], bond: int
) -> np.ndarray:
    """Per-sample right environment at ``bond`` (sites > bond + 1 contracted)."""

    size = basis[0].shape[0]
    state = np.ones((size, 1), dtype=complex)
    for site in range(len(tensors) - 1, bond + 1, -1):
        state = np.einsum("lkr,sk,sr->sl", tensors[site], basis[site], state)
    return state


# ------------------------------------------------------- the local lik. update --
def bond_likelihood_polish(
    merged: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    basis_1: np.ndarray,
    basis_2: np.ndarray,
) -> np.ndarray:
    """L-BFGS descent of the exact likelihood in the merged bond tensor.

    The design matrix is complex; with c = u + iv and g = E^H (z / |z|^2)
    the gradient of -mean log |E c|^2 is (-2/N)(Re g, -Im g), and the
    normalisation term log ||c||^2 adds 2 (u, v) / ||c||^2.
    """
    from scipy.optimize import minimize

    shape = merged.shape
    design = np.einsum(
        "sl,sj,sk,sr->sljkr", left, basis_1, basis_2, right, optimize=True
    ).reshape(left.shape[0], -1)
    dim = design.shape[1]
    size = design.shape[0]

    def objective(v):
        c = v[:dim] + 1j * v[dim:]
        n2 = float(np.sum(v * v))
        z = design @ c
        a2 = np.abs(z) ** 2 + 1e-300
        value = -float(np.mean(np.log(a2))) + np.log(n2)
        g = design.conj().T @ (z / a2)
        gradient = np.concatenate([-2.0 * g.real, 2.0 * g.imag]) / size
        return value, gradient + 2.0 * v / n2

    flat = merged.reshape(-1)
    flat = flat / np.linalg.norm(flat)
    start = np.concatenate([flat.real, flat.imag])
    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-12},
    )
    polished = result.x[:dim] + 1j * result.x[dim:]
    return (polished / np.linalg.norm(polished)).reshape(shape)


def likelihood_sweeps(
    tensors: list[np.ndarray],
    family,
    baseline,
    sample: np.ndarray,
    chi: int,
    rounds: int = LIK_SWEEPS,
) -> list[np.ndarray]:
    """Right-then-left sweeps of local likelihood updates at fixed chi."""

    degree = tensors[0].shape[1] - 1
    basis = site_basis(family, baseline, sample, degree)
    d = len(tensors)
    state = right_canonicalise([t.copy() for t in tensors])
    for _ in range(rounds):
        for bond, center_right in [(b, True) for b in range(d - 1)] + [
            (b, False) for b in range(d - 2, -1, -1)
        ]:
            merged = merge_bond(state, bond)
            left = left_environments(state, basis, bond)
            right = right_environments(state, basis, bond)
            polished = bond_likelihood_polish(
                merged, left, right, basis[bond], basis[bond + 1]
            )
            first, second, _ = split_bond(polished, chi, center_right=center_right)
            state[bond], state[bond + 1] = first, second
    return state


# ------------------------------------------------------------------ variants --
def random_state(d: int, degree: int, chi: int, rng) -> list[np.ndarray]:
    tensors = []
    for site in range(d):
        chi_l = 1 if site == 0 else chi
        chi_r = 1 if site == d - 1 else chi
        tensor = rng.standard_normal((chi_l, degree + 1, chi_r)) + 1j * rng.standard_normal(
            (chi_l, degree + 1, chi_r)
        )
        tensors.append(tensor)
    state = right_canonicalise(tensors)
    return state


def cold_moment_state(
    target, train: np.ndarray, degree: int, chi: int, rng
) -> list[np.ndarray]:
    """A single t = 0 moment fit from the vacuum: the no-schedule pipeline."""

    schedule = np.array([0.0, 0.0])
    result = fit_chain_schedule(
        target, schedule, degree, chi, len(train), 0.0, SWEEPS, rng
    )
    return result["slices"][-1]["tensors"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=SITES)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--random-inits", type=int, default=5)
    args = parser.parse_args()
    d, draws = args.sites, args.draws

    rng = np.random.default_rng(SEED)
    target = build_chain_target("normal", d, SEPARATION, EPSILON)
    family, baseline = target.family, target.baseline
    phi = fitting_matrices(family, baseline, DEGREE)
    k_max = phi.shape[0] - 1
    j_index, k_index = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    degrees = (j_index + k_index).astype(float)

    probe = np.asarray(target.sample(min(draws, 20_000), rng))
    middle = (d - 2) // 2
    probe_matrix = empirical_pair_coefficients(
        family, baseline, probe[:, middle : middle + 2], k_max
    )
    schedule = build_pair_schedule(probe_matrix.reshape(-1), degrees, SLICES)

    # the tower (variant a0); fit_chain_schedule draws its own pool from rng
    begin = time.perf_counter()
    tower = fit_chain_schedule(
        target, schedule, DEGREE, CHI, draws, TAU, SWEEPS, np.random.default_rng(SEED)
    )
    tower_cost = time.perf_counter() - begin
    train, held = tower["train"], tower["held"]
    truth_nll = target.truth_nll(held, 0.0)
    tensors_tower = tower["slices"][-1]["tensors"]

    def gap(tensors) -> float:
        return (model_nll(tensors, target, held) - truth_nll) / d

    results: dict[str, Any] = {"truth_nll": truth_nll, "d": d}
    results["a0_tower"] = {"gap": gap(tensors_tower), "cost": tower_cost}
    print(f"a0 tower alone: gap {results['a0_tower']['gap']:+.5f} nats/site", flush=True)

    begin = time.perf_counter()
    tensors_a = likelihood_sweeps(tensors_tower, family, baseline, train, CHI)
    results["a_tower_lik"] = {
        "gap": gap(tensors_a),
        "cost": time.perf_counter() - begin,
        "spectra": [bond_spectrum(tensors_a, b).tolist() for b in range(d - 1)],
    }
    print(
        f"a  tower + lik sweeps: gap {results['a_tower_lik']['gap']:+.5f}"
        f"  ({results['a_tower_lik']['cost']:.0f}s)",
        flush=True,
    )

    gaps_b = []
    for init in range(args.random_inits):
        state = random_state(d, DEGREE, CHI, np.random.default_rng(100 + init))
        state = likelihood_sweeps(state, family, baseline, train, CHI)
        gaps_b.append(gap(state))
        print(f"b  random init {init}: gap {gaps_b[-1]:+.5f}", flush=True)
    results["b_random_lik"] = {"gaps": gaps_b}

    gaps_c = []
    for seed in range(3):
        state = cold_moment_state(
            target, train, DEGREE, CHI, np.random.default_rng(200 + seed)
        )
        state = likelihood_sweeps(state, family, baseline, train, CHI)
        gaps_c.append(gap(state))
        print(f"c  cold moment seed {seed}: gap {gaps_c[-1]:+.5f}", flush=True)
    results["c_cold_moment_lik"] = {"gaps": gaps_c}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results.json").write_text(json.dumps(results))
    print(f"stored: {ARTIFACTS / 'results.json'}")


if __name__ == "__main__":
    main()
