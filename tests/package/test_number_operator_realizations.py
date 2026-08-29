"""Coordinate realizations of the number operator and the lowering ladders."""

import numpy as np

from nefqvf import (
    Binomial,
    BinomialParams,
    Gamma,
    GammaParams,
    NegativeBinomial,
    NegativeBinomialParams,
    Normal,
    NormalParams,
    Poisson,
    PoissonParams,
)

K = 8


def matrix_elements(family, params, grid, image, k_max, lattice=True):
    """E[phi_m (O phi)_n] for an operator given by its image on the basis."""

    p = np.asarray(family.prob(grid, params), dtype=float)
    basis = np.asarray(family.basis(grid, k_max, params), dtype=float)
    weights = p if lattice else p * np.gradient(grid)
    return basis.T @ (weights[:, None] * image)


def test_charlier_number_operator_is_the_difference_equation():
    lam = 4.0
    params = PoissonParams(lam)
    x = np.arange(0, 200)
    b = np.asarray(Poisson.basis(x, K, params), dtype=float)
    up = np.asarray(Poisson.basis(x + 1, K, params), dtype=float)
    down = np.vstack([np.zeros(K + 1), b[:-1]])  # phi(x-1), zero row at x=0
    image = x[:, None] * (b - down) - lam * (up - b)  # x nabla - lam Delta
    m = matrix_elements(Poisson, params, x, image, K)
    assert np.allclose(m, np.diag(np.arange(K + 1, dtype=float)), atol=1e-9)


def test_krawtchouk_number_operator_and_lowering():
    p, n_bin = 0.3, 12
    params = BinomialParams(p * n_bin, n_bin)
    q = 1 - p
    x = np.arange(0, n_bin + 1)
    k_max = 6
    b = np.asarray(Binomial.basis(x, k_max, params), dtype=float)
    up = np.vstack([b[1:], np.zeros(k_max + 1)])  # phi(x+1), zero beyond N
    down = np.vstack([np.zeros(k_max + 1), b[:-1]])
    image = -(n_bin - x)[:, None] * p * (up - b) + (x[:, None] * q) * (b - down)
    m = matrix_elements(Binomial, params, x, image, k_max)
    assert np.allclose(m, np.diag(np.arange(k_max + 1, dtype=float)), atol=1e-9)

    lower = np.sqrt(p * q) * ((n_bin - x)[:, None] * (up - b) + x[:, None] * (b - down))
    m = matrix_elements(Binomial, params, x, lower, k_max)
    target = np.zeros_like(m)
    for n in range(1, k_max + 1):
        target[n - 1, n] = np.sqrt((n_bin - n + 1) * n)
    assert np.allclose(m, target, atol=1e-9)


def test_meixner_number_operator_and_lowering():
    r, c = 3.0, 0.4
    params = NegativeBinomialParams(r * c / (1 - c), r)
    x = np.arange(0, 400)
    b = np.asarray(NegativeBinomial.basis(x, K, params), dtype=float)
    up = np.asarray(NegativeBinomial.basis(x + 1, K, params), dtype=float)
    down = np.vstack([np.zeros(K + 1), b[:-1]])
    image = -(c * (x + r)[:, None] * (up - b) - x[:, None] * (b - down)) / (1 - c)
    m = matrix_elements(NegativeBinomial, params, x, image, K)
    assert np.allclose(m, np.diag(np.arange(K + 1, dtype=float)), atol=1e-8)

    lower = (np.sqrt(c) / (1 - c)) * (
        (x + r)[:, None] * (up - b) - x[:, None] * (b - down)
    )
    m = matrix_elements(NegativeBinomial, params, x, lower, K)
    target = np.zeros_like(m)
    for n in range(1, K + 1):
        target[n - 1, n] = np.sqrt(n * (n + r - 1))
    assert np.allclose(m, target, atol=1e-8)


def test_hermite_and_laguerre_number_operators():
    grid = np.linspace(-10, 10, 40001)
    params = NormalParams(0.0, 1.0)
    b = np.asarray(Normal.basis(grid, K, params), dtype=float)
    d1 = np.gradient(b, grid, axis=0)
    d2 = np.gradient(d1, grid, axis=0)
    image = grid[:, None] * d1 - d2
    m = matrix_elements(Normal, params, grid, image, K, lattice=False)
    assert np.allclose(m, np.diag(np.arange(K + 1, dtype=float)), atol=1e-4)

    r, theta = 3.0, 1.0
    y = np.linspace(1e-6, 80, 120001)
    params = GammaParams(r * theta, r)
    b = np.asarray(Gamma.basis(y, K, params), dtype=float)
    d1 = np.gradient(b, y, axis=0)
    d2 = np.gradient(d1, y, axis=0)
    image = -(y[:, None] * d2 + (r - y)[:, None] * d1)
    m = matrix_elements(Gamma, params, y, image, K, lattice=False)
    assert np.allclose(m, np.diag(np.arange(K + 1, dtype=float)), atol=1e-3)


def test_meixner_pollaczek_number_operator_by_complex_shift():
    """GHS at eta = 0 (phi = pi/2): generate the basis by the Jacobi
    recurrence, which accepts complex arguments, and check the imaginary
    shift form of the number operator by quadrature."""

    r = 2.0
    phi_angle = np.pi / 2.0

    def basis(z, k_max):
        z = np.asarray(z, dtype=complex)
        a = lambda n: np.sqrt(n * (n + 2 * r - 1)) / (2 * np.sin(phi_angle))
        bdiag = lambda n: -(n + r) / np.tan(phi_angle)
        out = [np.ones_like(z), (z - bdiag(0)) / a(1)]
        for n in range(1, k_max):
            out.append(((z - bdiag(n)) * out[n] - a(n) * out[n - 1]) / a(n + 1))
        return np.stack(out, axis=1)

    x = np.linspace(-40, 40, 400001)
    # GHS density at eta = 0: proportional to |Gamma(r + ix)|^2
    from scipy.special import loggamma

    log_density = 2 * np.real(loggamma(r + 1j * x))
    density = np.exp(log_density - log_density.max())
    density /= np.trapezoid(density, x)

    k_max = 6
    b0 = basis(x, k_max)
    bp = basis(x + 1j, k_max)
    bm = basis(x - 1j, k_max)
    pref = np.exp(1j * phi_angle) * (x + 1j * r)
    image = -(pref[:, None] * (bp - b0) + np.conj(pref)[:, None] * (bm - b0)) / (
        2 * np.sin(phi_angle)
    )
    weights = density * np.gradient(x)
    m = np.real(b0).T @ (weights[:, None] * image)
    assert np.allclose(
        np.real(m), np.diag(np.arange(k_max + 1, dtype=float)), atol=1e-6
    )
    assert np.max(np.abs(np.imag(m))) < 1e-8
