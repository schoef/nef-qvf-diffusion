"""The neural conditioner with the member-frame amplitude head."""

import numpy as np

from applications.neural_frame_pair import (
    gradient_check,
    head_density,
    run_study,
)


def test_head_gradient_is_exact():
    assert gradient_check(2) < 1e-6
    assert gradient_check(4, seed=1) < 1e-6


def test_head_density_normalises():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=2 + 2 * 4)
    grid = np.linspace(-12.0, 12.0, 4001)
    density = head_density(grid, raw, 3)
    assert abs(np.trapezoid(density, grid) - 1.0) < 1e-6


def test_high_correlation_is_learned():
    """At rho = 0.95 the fixed frame at K = 8 fails (TV ~ 0.24); the
    neural frame with a degree-2 core reaches a few 1e-2."""

    result = run_study(0.95, degree=2, draws=30_000, steps=4_000, seed=7)
    assert result["total_variation"] < 0.08
    assert abs(result["width"] - np.sqrt(1.0 - 0.95**2)) < 0.05
