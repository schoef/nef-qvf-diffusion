"""One creator, two lowering partners: [D, Adag] = 1 and Adag D = nhat."""

import math

import numpy as np

LAM, R, THETA, P, N_BIN, C = 4.0, 3.0, 2.0, 0.3, 12, 0.4
PHI = 1.1  # GHS angle in (0, pi)


def family_data(name, levels):
    """Half-shift coefficients gamma_n and creator weights a_{n+1}/z'(0)."""

    n = np.arange(levels + 1, dtype=float)
    if name == "normal":
        gam = np.array([1 / math.sqrt(math.factorial(k)) for k in range(levels + 1)])
        w = np.sqrt(n[:-1] + 1)
    elif name == "poisson":
        gam = np.array(
            [LAM ** (k / 2) / math.sqrt(math.factorial(k)) for k in range(levels + 1)]
        )
        w = np.sqrt(LAM * (n[:-1] + 1))
    elif name == "gamma":
        gam = np.array(
            [
                math.sqrt(math.gamma(R + k) / math.gamma(R) / math.factorial(k))
                for k in range(levels + 1)
            ]
        )
        w = np.sqrt((n[:-1] + 1) * (R + n[:-1]))
    elif name == "binomial":
        q = 1 - P
        gam = np.array(
            [math.sqrt(math.comb(N_BIN, k) * (q / P) ** k) for k in range(levels + 1)]
        )
        w = np.sqrt((q / P) * (n[:-1] + 1) * (N_BIN - n[:-1]))
    elif name == "negative-binomial":
        gam = np.array(
            [
                math.sqrt(
                    math.gamma(R + k) / math.gamma(R) / math.factorial(k) * C ** (-k)
                )
                for k in range(levels + 1)
            ]
        )
        w = np.sqrt((1 / C) * (n[:-1] + 1) * (n[:-1] + R))
    elif name == "ghs":
        gam = np.array(
            [
                math.sqrt(math.gamma(2 * R + k) / math.gamma(2 * R) / math.factorial(k))
                for k in range(levels + 1)
            ]
        )
        w = np.sqrt((n[:-1] + 1) * (n[:-1] + 2 * R))
    else:
        raise ValueError(name)
    return gam, w


def ladder_matrices(name, levels):
    gam, w = family_data(name, levels)
    delta = np.zeros((levels + 1, levels + 1))
    creator = np.zeros((levels + 1, levels + 1))
    for k in range(1, levels + 1):
        delta[k - 1, k] = gam[k - 1] / gam[k]
    for k in range(levels):
        creator[k + 1, k] = w[k]
    return delta, creator


def test_weight_identity_ties_the_two_ladders():
    """(n+1) gamma_{n+1} / gamma_n equals the creator weight, all families."""

    for name in ("normal", "poisson", "gamma", "binomial", "negative-binomial", "ghs"):
        levels = N_BIN if name == "binomial" else 14
        gam, w = family_data(name, levels)
        n = np.arange(levels, dtype=float)
        assert np.allclose((n + 1) * gam[1:] / gam[:-1], w, rtol=1e-12), name


def test_creator_delta_products_are_the_gradings():
    """Adag D = nhat, D Adag = nhat + 1, [D, Adag] = 1 on the tower."""

    for name in ("normal", "poisson", "gamma", "binomial", "negative-binomial", "ghs"):
        levels = N_BIN if name == "binomial" else 14
        delta, creator = ladder_matrices(name, levels)
        interior = levels if name != "binomial" else N_BIN
        nhat = (creator @ delta)[:interior, :interior]
        assert np.allclose(nhat, np.diag(np.arange(interior, dtype=float))), name
        comm = (delta @ creator - creator @ delta)[:interior, :interior]
        assert np.allclose(comm, np.eye(interior)), name


def test_kernel_is_the_normal_ordered_exponential():
    """exp(z Adag) applied to e_0 reproduces gamma_n z^n coefficient-wise."""

    for name in ("poisson", "gamma"):
        gam, _ = family_data(name, 14)
        _, creator = ladder_matrices(name, 14)
        z = 0.37
        from scipy.linalg import expm

        coefficients = expm(z * creator)[:, 0]
        assert np.allclose(coefficients[:12], gam[:12] * z ** np.arange(12), atol=1e-12)
