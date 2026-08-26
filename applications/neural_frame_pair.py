"""A neural conditioner with the analytic member-frame amplitude head.

The pair law is factored autoregressively.  Each leg is the
one-dimensional Born model in a transported frame,

    p(x_i | context) = N(x_i; theta, 1) |h_c(x_i - theta)|^2 / |c|^2 ,

where for leg one (theta, c) are free parameters and for leg two they
are the outputs of a small MLP evaluated on x_1.  Everything analytic
about the head is kept: the conditional is exactly normalised for any
network output, so training is plain maximum likelihood -- no partition
function, no Monte Carlo bias.  Gradients are manual; the only special
ingredient is the ladder identity phi_n' = sqrt(n) phi_{n-1}.

The study trains on the correlated Gaussian pair at high correlations,
where the fixed-frame fit collapses (the ridge outruns the degree
budget), and scores the learned joint law in total variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.pair_schmidt import pair_density, sample_pair, total_variation
from nefqvf import Normal, NormalParams

DEFAULT_CORE_DEGREE = 2
DEFAULT_WIDTH = 32
DEFAULT_DRAWS = 50_000
DEFAULT_STEPS = 8_000
DEFAULT_BATCH = 512
DEFAULT_SEED = 5
DEFAULT_RHOS = (0.6, 0.9, 0.95, 0.99)
FIGURE_SUBDIRECTORY = "neural_frame_pair"

BASELINE = NormalParams(0.0, 1.0)


# ------------------------------------------------------------------ the head --
def head_log_likelihood(x: np.ndarray, raw: np.ndarray, degree: int):
    """Log-likelihood of the member-frame head and its gradient.

    ``raw`` has columns (theta, log s, a_0..a_K, b_0..b_K) per sample:
    a location transport, the Normal squeeze, and the shape core.
    Returns the per-sample log-likelihood and its gradient w.r.t. raw.
    """

    theta = raw[:, 0]
    log_s = raw[:, 1]
    s = np.exp(log_s)
    a = raw[:, 2 : degree + 3]
    b = raw[:, degree + 3 :]
    u = (x - theta) / s
    basis = np.asarray(Normal.basis(u, degree, BASELINE), dtype=float)
    c = a + 1j * b
    h = np.einsum("sn,sn->s", basis, c)
    norm2 = np.sum(a * a + b * b, axis=1)
    h2 = np.abs(h) ** 2 + 1e-300

    loglik = (
        -0.5 * u * u - log_s - 0.5 * np.log(2.0 * np.pi) + np.log(h2) - np.log(norm2)
    )

    # dh/du through the ladder: phi_n' = sqrt(n) phi_{n-1}
    roots = np.sqrt(np.arange(1, degree + 1))
    dh_du = np.einsum("sn,sn->s", basis[:, :-1], c[:, 1:] * roots)
    dll_du = -u + 2.0 * np.real(np.conj(h) * dh_du) / h2

    grad = np.empty_like(raw)
    grad[:, 0] = -dll_du / s
    grad[:, 1] = -dll_du * u - 1.0
    grad[:, 2 : degree + 3] = (
        2.0 * np.real(np.conj(h)[:, None] * basis) / h2[:, None]
        - 2.0 * a / norm2[:, None]
    )
    grad[:, degree + 3 :] = (
        -2.0 * np.imag(np.conj(h)[:, None] * basis) / h2[:, None]
        - 2.0 * b / norm2[:, None]
    )
    return loglik, grad


def head_density(grid: np.ndarray, raw_row: np.ndarray, degree: int) -> np.ndarray:
    """The head's density on a grid for one output row."""

    theta, s = raw_row[0], np.exp(raw_row[1])
    c = raw_row[2 : degree + 3] + 1j * raw_row[degree + 3 :]
    u = (grid - theta) / s
    basis = np.asarray(Normal.basis(u, degree, BASELINE), dtype=float)
    h2 = np.abs(basis @ c) ** 2
    reference = np.exp(-0.5 * u * u) / (s * np.sqrt(2.0 * np.pi))
    return reference * h2 / np.sum(np.abs(c) ** 2)


# ------------------------------------------------------------------ the MLP --
class Conditioner:
    """Two-hidden-layer tanh MLP: context -> (theta, a, b)."""

    def __init__(self, out_dim: int, width: int, rng: Any):
        scale = 1.0 / np.sqrt(width)
        self.w1 = rng.normal(scale=1.0, size=(1, width))
        self.b1 = np.zeros(width)
        self.w2 = rng.normal(scale=scale, size=(width, width))
        self.b2 = np.zeros(width)
        self.w3 = np.zeros((width, out_dim))
        self.b3 = np.zeros(out_dim)
        # start near the baseline, complex so the amplitude has no nodes
        self.b3[2] = 1.0
        self.b3[out_dim // 2 + 1] = 0.3

    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def forward(self, x: np.ndarray):
        z1 = x[:, None] @ self.w1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.tanh(z2)
        out = h2 @ self.w3 + self.b3
        return out, (x, h1, h2)

    def backward(self, cache, grad_out: np.ndarray) -> list[np.ndarray]:
        x, h1, h2 = cache
        gw3 = h2.T @ grad_out
        gb3 = grad_out.sum(axis=0)
        gh2 = grad_out @ self.w3.T
        gz2 = gh2 * (1.0 - h2 * h2)
        gw2 = h1.T @ gz2
        gb2 = gz2.sum(axis=0)
        gh1 = gz2 @ self.w2.T
        gz1 = gh1 * (1.0 - h1 * h1)
        gw1 = x[:, None].T @ gz1
        gb1 = gz1.sum(axis=0)
        return [gw1, gb1, gw2, gb2, gw3, gb3]


class Adam:
    def __init__(self, parameters: list[np.ndarray], rate: float = 2e-3):
        self.parameters = parameters
        self.rate = rate
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(self.parameters, gradients, self.m, self.v, strict=True):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            m_hat = m / (1.0 - 0.9**self.t)
            v_hat = v / (1.0 - 0.999**self.t)
            p -= self.rate * m_hat / (np.sqrt(v_hat) + 1e-8)


# ------------------------------------------------------------------ training --
def train_pair(
    sample: np.ndarray,
    *,
    degree: int = DEFAULT_CORE_DEGREE,
    width: int = DEFAULT_WIDTH,
    steps: int = DEFAULT_STEPS,
    batch: int = DEFAULT_BATCH,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Maximum-likelihood training of the two-leg autoregressive model."""

    rng = np.random.default_rng(seed)
    out_dim = 2 + 2 * (degree + 1)

    leg1 = np.zeros(out_dim)
    leg1[2] = 1.0
    leg1[out_dim // 2 + 1] = 0.3
    conditioner = Conditioner(out_dim, width, rng)
    optimizer = Adam([leg1, *conditioner.parameters()])

    history = []
    for step in range(steps):
        rows = rng.integers(0, len(sample), size=batch)
        x1, x2 = sample[rows, 0], sample[rows, 1]

        ll1, g1 = head_log_likelihood(x1, np.tile(leg1, (batch, 1)), degree)
        raw2, cache = conditioner.forward(x1)
        ll2, g2 = head_log_likelihood(x2, raw2, degree)

        grad_leg1 = -g1.mean(axis=0)
        grads_mlp = conditioner.backward(cache, -g2 / batch)
        gradients = [grad_leg1, *grads_mlp]
        total = np.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        if total > 10.0:
            gradients = [g * (10.0 / total) for g in gradients]
        optimizer.step(gradients)
        if step % 200 == 0 or step == steps - 1:
            history.append({"step": step, "nll": -float(np.mean(ll1 + ll2))})
    return {"leg1": leg1, "conditioner": conditioner, "history": history}


def model_pair_law(model: dict[str, Any], grid: np.ndarray, degree: int) -> np.ndarray:
    """The learned joint density on ``grid x grid``."""

    marginal = head_density(grid, model["leg1"], degree)
    raw2, _ = model["conditioner"].forward(grid)
    conditional = np.stack(
        [head_density(grid, raw2[i], degree) for i in range(len(grid))]
    )
    return marginal[:, None] * conditional


def gradient_check(degree: int = 2, seed: int = 0) -> float:
    """Central-difference check of the head gradient; returns the max error."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=5)
    raw = rng.normal(size=(5, 2 + 2 * (degree + 1)))
    _, grad = head_log_likelihood(x, raw, degree)
    worst = 0.0
    eps = 1e-6
    for j in range(raw.shape[1]):
        plus, minus = raw.copy(), raw.copy()
        plus[:, j] += eps
        minus[:, j] -= eps
        ll_plus, _ = head_log_likelihood(x, plus, degree)
        ll_minus, _ = head_log_likelihood(x, minus, degree)
        numeric = (ll_plus - ll_minus) / (2.0 * eps)
        worst = max(worst, float(np.max(np.abs(numeric - grad[:, j]))))
    return worst


# ------------------------------------------------------------------ study --
def run_study(
    rho: float,
    *,
    degree: int = DEFAULT_CORE_DEGREE,
    draws: int = DEFAULT_DRAWS,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample = sample_pair(rho, draws, rng)
    model = train_pair(sample, degree=degree, steps=steps, seed=seed)

    grid = np.linspace(-6.0, 6.0, 401)
    law = model_pair_law(model, grid, degree)
    truth = pair_density(rho, grid)
    tv = total_variation(truth, law, grid)

    raw2, _ = model["conditioner"].forward(grid)
    slope = np.polyfit(grid[100:301], raw2[100:301, 0], 1)[0]
    width = float(np.exp(np.mean(raw2[100:301, 1])))
    return {
        "rho": rho,
        "degree": degree,
        "total_variation": tv,
        "theta_of_x": raw2[:, 0],
        "grid": grid,
        "slope": slope,
        "width": width,
        "final_nll": model["history"][-1]["nll"],
        "parameters": 2
        + 2 * (degree + 1)
        + sum(p.size for p in model["conditioner"].parameters()),
    }


def plot_study(
    results: list[dict[str, Any]],
    fixed_frame: dict[float, float],
    *,
    output_dir: Any = None,
) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    rhos = [r["rho"] for r in results]
    axes[0].semilogy(
        list(fixed_frame.keys()),
        list(fixed_frame.values()),
        "o-",
        color="#b0413e",
        label="fixed frame, $K = 8$",
    )
    axes[0].semilogy(
        rhos,
        [r["total_variation"] for r in results],
        "s-",
        color="#2a6f97",
        label=f"neural frame, core $K = {results[0]['degree']}$",
    )
    axes[0].set_xlabel(r"correlation $\rho$")
    axes[0].set_ylabel("total variation")
    axes[0].legend(fontsize=9, frameon=False)

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))
    for result, color in zip(results, colors, strict=True):
        grid = result["grid"]
        axes[1].plot(
            grid,
            result["theta_of_x"],
            color=color,
            label=rf"$\rho = {result['rho']:g}$",
        )
        axes[1].plot(grid, result["rho"] * grid, ls=":", lw=1.0, color="0.3")
    axes[1].set_xlim(-4.0, 4.0)
    axes[1].set_ylim(-4.5, 4.5)
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel(r"learned $\theta(x_1)$")
    axes[1].set_title(r"conditional frame (dotted: $\rho x_1$)")
    axes[1].legend(fontsize=8, frameon=False)
    figure.tight_layout()

    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "neural-frame-pair.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=DEFAULT_CORE_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rhos", type=float, nargs="+", default=list(DEFAULT_RHOS))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    check = gradient_check(args.degree)
    print(f"gradient check: max error {check:.2e}")

    from applications.pair_schmidt import run_study as fixed_frame_study

    results, fixed = [], {}
    for rho in args.rhos:
        result = run_study(
            rho, degree=args.degree, draws=args.draws, steps=args.steps, seed=args.seed
        )
        results.append(result)
        reference = fixed_frame_study(rho, degree=8, draws=args.draws, seed=args.seed)
        fixed[rho] = reference["total_variation"]
        print(
            f"rho {rho:5.2f}  fixed-frame TV {fixed[rho]:9.3e}  "
            f"neural TV {result['total_variation']:9.3e}  "
            f"theta slope {result['slope']:+.4f} (true {rho:+.2f})  "
            f"width {result['width']:.4f} (true {np.sqrt(1 - rho * rho):.4f})  "
            f"nll {result['final_nll']:.4f}  "
            f"params {result['parameters']}"
        )
    if args.plot:
        print(f"figure: {plot_study(results, fixed, output_dir=args.output)}")


if __name__ == "__main__":
    main()
