"""R6-P5 in full: the complete pipeline at chi = 4 on the two-state chain.

The rank-recovery question: the two-state hidden Markov chain has exact
bond rank two.  Run the whole pipeline -- moment tower down the schedule,
then local likelihood sweeps -- at chi = 4 and ask whether the third and
fourth Schmidt values collapse at every bond while the held-out gap
matches the chi = 2 result.  Expensive (the merged bond tensors have
dimension 4 x 36 x 4); intended for an overnight run.  Progress prints
per stage; results land in artifacts/hmm-likelihood-sweep/chi4.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

from applications.amplitude_fit_complex import fitting_matrices
from applications.d_site_diffusion import (
    build_chain_target,
    fit_chain_schedule,
    model_nll,
)
from applications.hmm_likelihood_sweep import (
    ARTIFACTS,
    DEGREE,
    DRAWS,
    EPSILON,
    SEED,
    SEPARATION,
    SITES,
    SLICES,
    SWEEPS,
    TAU,
    likelihood_sweeps,
)
from applications.mps_amplitude import bond_spectrum
from applications.two_site_diffusion import (
    build_pair_schedule,
    empirical_pair_coefficients,
)

CHI = 4


def main() -> None:
    d = SITES
    rng = np.random.default_rng(SEED)
    target = build_chain_target("normal", d, SEPARATION, EPSILON)
    family, baseline = target.family, target.baseline
    phi = fitting_matrices(family, baseline, DEGREE)
    k_max = phi.shape[0] - 1
    j_index, k_index = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    degrees = (j_index + k_index).astype(float)

    probe = np.asarray(target.sample(20_000, rng))
    middle = (d - 2) // 2
    probe_matrix = empirical_pair_coefficients(
        family, baseline, probe[:, middle : middle + 2], k_max
    )
    schedule = build_pair_schedule(probe_matrix.reshape(-1), degrees, SLICES)
    print(f"chi={CHI}, d={d}, {DRAWS} draws, {len(schedule) - 1} slices", flush=True)

    begin = time.perf_counter()
    tower = fit_chain_schedule(
        target, schedule, DEGREE, CHI, DRAWS, TAU, SWEEPS, np.random.default_rng(SEED)
    )
    print(f"tower done ({time.perf_counter() - begin:.0f}s)", flush=True)
    train, held = tower["train"], tower["held"]
    truth_nll = target.truth_nll(held, 0.0)
    tensors = tower["slices"][-1]["tensors"]
    gap_tower = (model_nll(tensors, target, held) - truth_nll) / d
    print(f"tower gap: {gap_tower:+.5f} nats/site", flush=True)

    begin = time.perf_counter()
    state = likelihood_sweeps(tensors, family, baseline, train, CHI)
    print(f"likelihood sweeps done ({time.perf_counter() - begin:.0f}s)", flush=True)
    gap = (model_nll(state, target, held) - truth_nll) / d
    spectra = [bond_spectrum(state, b).tolist() for b in range(d - 1)]
    print(f"final gap: {gap:+.5f} nats/site", flush=True)
    for bond, values in enumerate(spectra):
        print(
            f"bond {bond}: " + " ".join(f"{v:.2e}" for v in values[:4]), flush=True
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "chi4.json").write_text(
        json.dumps({"gap_tower": gap_tower, "gap": gap, "spectra": spectra})
    )
    print(f"stored: {ARTIFACTS / 'chi4.json'}")


if __name__ == "__main__":
    main()
