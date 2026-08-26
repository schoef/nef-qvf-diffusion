"""The pairing layer: one bilinear generator per pair of legs."""

import numpy as np

from applications.pairing_layer import (
    law_moments,
    pair_law,
    pairing_amplitude,
    study_family,
    support_grid,
    xi_scaled,
)


def test_normal_pairing_is_the_two_mode_squeezed_vacuum():
    xi, degree = 0.5, 30
    c = pairing_amplitude("normal", xi, degree)
    tmsv = np.tanh(xi) ** np.arange(degree + 1) / np.cosh(xi)
    assert np.max(np.abs(c - tmsv)) < 1e-10


def test_normal_pairing_law_is_the_thermal_gaussian():
    xi, degree = 0.5, 30
    c = pairing_amplitude("normal", xi, degree)
    grid = support_grid("normal")
    moments = law_moments("normal", pair_law("normal", c, grid), grid)
    assert abs(moments["correlation"] - np.tanh(2 * xi)) < 1e-9
    assert abs(moments["variance"] - np.cosh(2 * xi)) < 1e-6


def test_lattice_and_halfline_pairings_are_valid_and_correlated():
    for name in ("poisson", "gamma"):
        result = study_family(name, xi_scaled(name, 0.5), 40)
        moments = result["moments"]
        assert abs(moments["mass"] - 1.0) < 0.01
        assert moments["correlation"] > 0.6
        assert np.all(result["law"] >= 0.0)
