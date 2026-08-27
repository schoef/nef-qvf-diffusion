"""The exactly solvable Gaussian chain."""

import numpy as np

from applications.gaussian_chain import (
    canonical_correlation,
    chain_covariance,
    cross_block_rank,
    exact_chain_amplitude,
    pair_expressivity_floor,
    schmidt_spectrum,
)
from applications.pair_schmidt import schmidt_parameter


def test_covariance_flow_preserves_marginals():
    sigma = chain_covariance(0.9, 5, 0.7)
    assert np.allclose(np.diag(sigma), 1.0)
    assert abs(sigma[0, 1] - np.exp(-1.4) * 0.9) < 1e-12


def test_cross_blocks_are_rank_one_at_all_times():
    for t in (0.0, 0.4, 1.2):
        sigma = chain_covariance(0.85, 6, t)
        for cut in range(1, 6):
            assert cross_block_rank(sigma, cut) == 1


def test_melting_law_on_the_exact_tensor():
    sigma = chain_covariance(0.9, 4, 0.4)
    tensor = exact_chain_amplitude(sigma, 8, nodes=48)
    for cut in (1, 2, 3):
        r = canonical_correlation(sigma, cut)
        kappa = schmidt_parameter(r)
        spectrum = schmidt_spectrum(tensor, cut)
        assert np.max(np.abs(spectrum[:4] - kappa ** np.arange(4))) < 1e-6


def test_expressivity_floor_grows_toward_the_data_slice():
    grid = np.linspace(-6.0, 6.0, 401)
    floors = [pair_expressivity_floor(rho, 8, grid) for rho in (0.3, 0.6, 0.9)]
    assert floors[0] < floors[1] < floors[2]
    assert floors[0] < 1e-6
    assert floors[2] > 1e-2
