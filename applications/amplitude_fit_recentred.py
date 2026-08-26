"""Joint fit of a Fisher location and a complex amplitude, one channel.

The model transports the baseline by an arclength ``theta`` along the
family and fits a complex amplitude in the transported frame:
``p = p_{eta(theta)} |h_c(. ; eta(theta))|^2``.  The loss matches the
model's ratio coefficients to the empirical coefficients *of the moved
baseline*, so the data moments move with ``theta`` while the model stays
centred.  The near-flat direction that a coherent shift of ``c`` shares
with ``theta`` is removed from the optimiser's tangent space alongside
the norm and phase gauges, which puts the location parameter on the same
footing as the shape coefficients.  After convergence the amplitude is
transported along the orbit to the moment gauge ``R_1(c) = 0``, making
the reported ``theta`` the fitted Fisher location.

The study sweeps the displacement of a shifted-member target at fixed
small degree: the plain fit hits the displacement wall, the recentred
fit should not.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, null_space

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fitting_matrices,
    law_from_amplitude,
    ratio_coefficients_complex,
)
from applications.targets import (
    TARGETS,
    Target,
    empirical_coefficients,
    shifted_target,
    support_grid,
)

DEFAULT_DEGREE = 4
DEFAULT_DRAWS = 50_000
DEFAULT_SEED = 5
FIGURE_SUBDIRECTORY = "recentred_fit"

# natural-shift sweep ranges chosen to cross theta = 2 sqrt(K) at K = 4
SWEEPS = {
    "normal": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    "poisson": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
    "gamma": (0.05, 0.10, 0.15, 0.20, 0.25),
    "negative-binomial": (0.1, 0.2, 0.3, 0.4, 0.5),
}


# ------------------------------------------------------------------ family --
def with_mean(family: Any, baseline: Any, mean: float) -> Any:
    """Return the member with the given mean and the baseline's shape."""

    return dataclasses.replace(baseline, mean=float(mean))


def quadratic_coefficient(family: Any, baseline: Any) -> float:
    """Return ``a_2``, exact for a quadratic variance function."""

    mu = float(family.mean(baseline))
    h = max(0.1, 0.05 * abs(mu))
    values = [
        float(family.variance(with_mean(family, baseline, mu + s * h)))
        for s in (-1, 0, 1)
    ]
    return 0.5 * (values[0] - 2 * values[1] + values[2]) / h**2


def mean_at_arclength(family: Any, baseline: Any, theta: float) -> float:
    """Integrate ``dmu/dtheta = sqrt(V)`` from the baseline mean."""

    mu = float(family.mean(baseline))
    steps = 200
    dt = theta / steps
    for _ in range(steps):
        k1 = np.sqrt(float(family.variance(with_mean(family, baseline, mu))))
        half = mu + 0.5 * dt * k1
        k2 = np.sqrt(float(family.variance(with_mean(family, baseline, half))))
        mu = mu + dt * k2
    return mu


def flow_generator(degree: int, a2: float) -> np.ndarray:
    """Return the truncated antisymmetric generator ``G0``."""

    n = np.arange(degree)
    weights = 0.5 * np.sqrt((n + 1) * (1 + n * a2))
    matrix = np.zeros((degree + 1, degree + 1))
    matrix[n, n + 1] = weights
    matrix[n + 1, n] = -weights
    return matrix


# ------------------------------------------------------------------ the fit --
def moved_targets(
    family: Any, baseline: Any, sample: np.ndarray, theta: float, k_max: int
) -> np.ndarray:
    """Empirical coefficients of the sample at the transported baseline."""

    member = with_mean(family, baseline, mean_at_arclength(family, baseline, theta))
    coefficients, _ = empirical_coefficients(family, member, sample, k_max)
    return coefficients, member


def fit_recentred(
    family: Any,
    baseline: Any,
    sample: np.ndarray,
    degree: int,
    *,
    theta_box: tuple[float, float] = (-8.0, 8.0),
    max_iterations: int = 200,
    tolerance: float = 1e-11,
) -> dict[str, Any]:
    """Joint Levenberg--Marquardt over ``(theta, c)`` with the orbit gauged out."""

    a2 = quadratic_coefficient(family, baseline)
    generator = flow_generator(degree, a2)

    # moment initialisation: the arclength to the sample mean
    mu0 = float(family.mean(baseline))
    mu_hat = float(np.mean(sample))
    span = np.linspace(mu0, mu_hat, 2001)
    speeds = np.array(
        [
            1.0 / np.sqrt(float(family.variance(with_mean(family, baseline, m))))
            for m in span
        ]
    )
    theta = float(np.trapezoid(speeds, span))
    theta = float(np.clip(theta, theta_box[0], theta_box[1]))
    c = np.zeros(degree + 1, dtype=complex)
    c[0] = 1.0

    def pieces(theta_value):
        member = with_mean(
            family, baseline, mean_at_arclength(family, baseline, theta_value)
        )
        phi = fitting_matrices(family, member, degree)
        target, _ = empirical_coefficients(family, member, sample, phi.shape[0] - 1)
        return phi, np.asarray(target, dtype=float), member

    phi, target, member = pieces(theta)

    def residual(vec, tgt):
        return ratio_coefficients_complex(vec, phi)[1:] - tgt[1:]

    mu = 1e-3
    value = float(0.5 * residual(c, target) @ residual(c, target))
    history = [{"theta": theta, "objective": value, "coefficients": c.copy()}]
    h = 1e-4
    for iteration in range(max_iterations):
        a, b = np.real(c), np.imag(c)
        r = residual(c, target)
        rows, n, _ = phi[1:].shape
        flat = phi[1:].reshape(rows * n, n)
        jac_c = 2.0 * np.concatenate(
            ((flat @ a).reshape(rows, n), (flat @ b).reshape(rows, n)), axis=1
        )
        _, plus_target, _ = pieces(theta + h)
        _, minus_target, _ = pieces(theta - h)
        jac_theta = -(plus_target[1:] - minus_target[1:]) / (2 * h)

        gauge = np.vstack(
            (
                np.concatenate((a, b)),
                np.concatenate((-b, a)),
                np.concatenate((generator @ a, generator @ b)),
            )
        )
        tangent = null_space(gauge)
        jacobian = np.column_stack((jac_c @ tangent, jac_theta))
        gradient = jacobian.T @ r
        if np.linalg.norm(gradient) < tolerance:
            break
        hessian = jacobian.T @ jacobian

        accepted = False
        for _ in range(40):
            step = np.linalg.solve(hessian + mu * np.eye(hessian.shape[0]), -gradient)
            trial_real = np.concatenate((a, b)) + tangent @ step[:-1]
            trial_c = trial_real[: degree + 1] + 1j * trial_real[degree + 1 :]
            trial_c = trial_c / np.linalg.norm(trial_c)
            trial_theta = float(np.clip(theta + step[-1], theta_box[0], theta_box[1]))
            phi_t, target_t, member_t = pieces(trial_theta)

            def trial_residual():
                return ratio_coefficients_complex(trial_c, phi_t)[1:] - target_t[1:]

            trial_value = float(0.5 * trial_residual() @ trial_residual())
            if trial_value < value:
                c, theta = trial_c, trial_theta
                phi, target, member = phi_t, target_t, member_t
                value = trial_value
                history.append(
                    {"theta": theta, "objective": value, "coefficients": c.copy()}
                )
                mu = max(mu * 0.3, 1e-14)
                accepted = True
                break
            mu *= 3.0
        if not accepted:
            break

    # moment gauge: transport along the orbit until R_1(c) = 0
    for _ in range(20):
        r1 = float(ratio_coefficients_complex(c, phi)[1])
        slope_c = expm(1e-4 * generator) @ c
        phi1, target1, _ = pieces(theta + 1e-4)
        r1_plus = float(ratio_coefficients_complex(slope_c, phi1)[1])
        slope = (r1_plus - r1) / 1e-4
        if abs(slope) < 1e-12 or abs(r1) < 1e-10:
            break
        delta = float(np.clip(-r1 / slope, -0.2, 0.2))
        theta = float(np.clip(theta + delta, theta_box[0], theta_box[1]))
        c = expm(delta * generator) @ c
        c = c / np.linalg.norm(c)
        phi, target, member = pieces(theta)

    history.append({"theta": theta, "objective": value, "coefficients": c.copy()})
    return {
        "theta": theta,
        "coefficients": c,
        "member": member,
        "objective": value,
        "iterations": iteration + 1,
        "history": history,
    }


# ------------------------------------------------------------------ study --
def run_offset_sweep(
    name: str,
    *,
    degree: int = DEFAULT_DEGREE,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    offsets: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Sweep the displacement of a shifted target at fixed small degree."""

    if offsets is None:
        offsets = SWEEPS.get(name, (0.5, 1.0, 1.5, 2.0, 2.5))
    rng = np.random.default_rng(seed)
    family, baseline, _ = TARGETS[name]
    rows = []
    for offset in offsets:
        target = shifted_target(name, offset)
        grid = support_grid(family, baseline, target.members)
        sample = np.asarray(target.sample(draws, rng))

        # plain complex fit at the fixed baseline
        phi0 = fitting_matrices(family, baseline, degree)
        empirical, _ = empirical_coefficients(
            family, baseline, sample, phi0.shape[0] - 1
        )
        plain = continued_complex_fit(phi0, empirical)["complex"]
        plain_law = law_from_amplitude(target, plain["coefficients"], grid)
        plain_tv = _distance(target, plain_law, grid)

        # recentred joint fit
        fit = fit_recentred(family, baseline, sample, degree)
        moved = Target(
            label=target.label,
            family=family,
            baseline=fit["member"],
            sample=target.sample,
            density=target.density,
            members=target.members,
        )
        law = law_from_amplitude(moved, fit["coefficients"], grid)
        tv = _distance(target, law, grid)

        true_theta = _true_arclength(family, baseline, target.members[0])
        trace = []
        for entry in fit["history"]:
            step_member = with_mean(
                family, baseline, mean_at_arclength(family, baseline, entry["theta"])
            )
            step_target = Target(
                label=target.label,
                family=family,
                baseline=step_member,
                sample=target.sample,
                density=target.density,
                members=target.members,
            )
            step_law = law_from_amplitude(step_target, entry["coefficients"], grid)
            trace.append(
                {
                    "theta": entry["theta"],
                    "objective": entry["objective"],
                    "tv": _distance(target, step_law, grid),
                }
            )
        rows.append(
            {
                "offset": offset,
                "plain_tv": plain_tv,
                "recentred_tv": tv,
                "theta": fit["theta"],
                "true_theta": true_theta,
                "iterations": fit["iterations"],
                "trace": trace,
            }
        )
        print(
            f"  offset {offset:4.2f}  plain TV {plain_tv:9.3e}  "
            f"recentred TV {tv:9.3e}  theta {fit['theta']:+.3f} "
            f"(true {true_theta:+.3f})  iters {fit['iterations']}"
        )
    return {"name": name, "degree": degree, "rows": rows}


def _distance(target, law, grid):
    truth = np.asarray(target.density(grid), dtype=float)
    if target.family.is_lattice(target.baseline):
        return 0.5 * float(np.sum(np.abs(truth - law)))
    return 0.5 * float(np.trapezoid(np.abs(truth - law), grid))


def _true_arclength(family, baseline, member):
    mu0, mu1 = float(family.mean(baseline)), float(family.mean(member))
    grid = np.linspace(mu0, mu1, 4001)
    speeds = np.array(
        [
            1.0 / np.sqrt(float(family.variance(with_mean(family, baseline, m))))
            for m in grid
        ]
    )
    return float(np.trapezoid(speeds, grid))


def plot_sweep(result: dict[str, Any], *, output_dir: Any = None) -> str:
    rows = result["rows"]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    thetas = [row["true_theta"] for row in rows]
    axis.semilogy(
        thetas, [row["plain_tv"] for row in rows], "o-", label="plain complex fit"
    )
    axis.semilogy(
        thetas, [row["recentred_tv"] for row in rows], "s-", label="recentred fit"
    )
    wall = 2.0 * np.sqrt(result["degree"])
    axis.axvline(wall, color="0.6", ls=":", lw=1.0)
    axis.text(wall, axis.get_ylim()[1], r" $\theta=2\sqrt{K}$", va="top", fontsize=8)
    axis.set_xlabel("target displacement (Fisher distance)")
    axis.set_ylabel("total variation")
    axis.legend(fontsize=9)
    axis.set_title(f"{result['name']}, K = {result['degree']}")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}-K{result['degree']}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_convergence(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Three panels per accepted LM step: objective, theta error, and TV.

    The dashed horizontals in the right panel are the plain fit's final
    total variation at the same offset -- the level each trace must beat."""

    rows = result["rows"]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(rows)))
    for row, color in zip(rows, colors, strict=True):
        steps = np.arange(len(row["trace"]))
        label = rf"$\theta_\star = {row['true_theta']:.2f}$"
        axes[0].semilogy(
            steps, [e["objective"] for e in row["trace"]], color=color, label=label
        )
        theta_error = [
            max(abs(e["theta"] - row["true_theta"]), 1e-6) for e in row["trace"]
        ]
        tvs = [e["tv"] for e in row["trace"]]
        axes[1].semilogy(steps, theta_error, color=color)
        axes[2].semilogy(steps, tvs, color=color)
        # the last entry is the moment-gauge polish, not an LM step
        axes[1].plot(steps[-1], theta_error[-1], "o", mfc="none", color=color, ms=6)
        axes[2].plot(steps[-1], tvs[-1], "o", mfc="none", color=color, ms=6)
        axes[2].axhline(row["plain_tv"], color=color, ls="--", lw=0.8, alpha=0.6)
    axes[0].set_ylim(bottom=1e-12)
    axes[0].set_ylabel("objective")
    axes[1].set_ylabel(r"$|\hat\theta - \theta_\star|$")
    axes[2].set_ylabel("total variation")
    axes[1].text(
        0.03,
        0.03,
        "circles: after the moment gauge",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.4",
    )
    axes[2].text(
        0.97,
        0.5,
        "dashed: plain fit, final",
        transform=axes[2].transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color="0.4",
    )
    for axis in axes:
        axis.set_xlabel("accepted LM step")
    axes[0].legend(fontsize=8, title="displacement")
    figure.suptitle(f"{result['name']}, K = {result['degree']}: joint-fit convergence")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}-K{result['degree']}-convergence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(f"{args.family}, degree {args.degree}")
    result = run_offset_sweep(
        args.family, degree=args.degree, draws=args.draws, seed=args.seed
    )
    if args.plot:
        print(f"  figure: {plot_sweep(result, output_dir=args.output)}")
        print(f"  figure: {plot_convergence(result, output_dir=args.output)}")


if __name__ == "__main__":
    main()
