"""The d = 2 mother model on the correlated Gaussian: Mehler and Schmidt."""

import numpy as np

from applications.pair_schmidt import (
    exact_amplitude,
    run_study,
    schmidt_parameter,
)


def test_exact_amplitude_spectrum_is_geometric():
    """The quadrature amplitude's Schmidt spectrum is kappa^n with
    kappa = rho / (1 + sqrt(1 - rho^2))."""

    for rho in (0.3, 0.6, 0.9):
        kappa = schmidt_parameter(rho)
        singular = np.linalg.svd(exact_amplitude(rho, 8), compute_uv=False)
        ratios = singular / singular[0]
        # truncation nibbles at the last entries; check the resolvable head
        head = 5 if rho < 0.8 else 3
        assert np.allclose(ratios[:head], kappa ** np.arange(head), rtol=2e-2)


def test_small_rho_amplitude_correlation_is_half():
    """h = sqrt(1 + rho phi_1 phi_1 + ...) gives c_11 -> rho / 2."""

    exact = exact_amplitude(0.05, 4)
    assert abs(exact[1, 1] / exact[0, 0] - 0.025) < 1e-3


def test_fitted_study_matches_predictions():
    result = run_study(0.6, degree=6, draws=60_000, seed=3)
    kappa = result["kappa"]
    assert abs(kappa - 1.0 / 3.0) < 1e-12
    assert abs(result["singular"][1] - kappa) < 0.02
    assert result["parity_defect"] < 1e-2
    assert result["total_variation"] < 0.02
    # Mehler diagonal at low degree beats the sampling floor
    diagonal = result["mehler_diagonal"]
    assert np.allclose(diagonal[:4], 0.6 ** np.arange(4), atol=0.02)
