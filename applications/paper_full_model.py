"""The full fit model, one panel per principle, all six families.

The model p_{theta,c,eps} = p_{eta(theta)} [ (1-eps)|h_c|^2 + eps ] of the
note's fitting section has three ingredients beyond the real rank-one core,
and each earns a column:

  1  the complex amplitude: an equal mixture at the widest resolvable gap,
     fitted with a real and with a complex amplitude at fixed baseline;
  2  the movable frame: a single member displaced beyond the wall
     theta = 2 sqrt(K), fitted with the frame fixed and with the frame free;
  3  a law outside the class: the truncated baseline, whose jump no
     polynomial ratio represents, fitted with a real and with a complex
     amplitude.  The eps-mixture is deliberately not displayed as a
     fitted structure: the law does not determine the rank-two
     decomposition (the non-identifiability proposition), and held-out
     likelihood confirms it by selecting eps = 0 wherever the amplitude
     clears the data.

Every panel shows the sample, the reference, the target, and the fitted
law(s), with total variations quoted.
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

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fitting_matrices,
)
from applications.amplitude_fit_recentred import (
    fit_recentred,
    mean_at_arclength,
    with_mean,
)
from applications.paper_one_channel import (
    FIT_COLOUR,
    REFERENCE_COLOUR,
    SAMPLE_COLOUR,
    TARGET_COLOUR,
    continued_fit,
    separation_for,
)
from applications.targets import (
    FAMILY_MAX_DEGREE,
    TARGETS,
    Target,
    integrate,
    mixture_target,
    support_grid,
    truncated_target,
)

FAMILY_NAMES = ("normal", "poisson", "gamma", "binomial", "negative-binomial", "ghs")
FAMILY_LABELS = {
    "normal": "Normal",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "binomial": "Binomial",
    "negative-binomial": "Neg. binomial",
    "ghs": "GHS",
}
SAMPLE_SIZE = 10**5
DISPLAY_DEGREE = 12
RECENTRED_DEGREE = 4
EPSILON = 0.1
COMPARATOR_COLOUR = "#b0413e"
FIGURE_SUBDIRECTORY = "full-model"

# arclength displacement for the movable-frame column: beyond the wall
# 2 sqrt(K) = 4 at K = 4 for every family; the binomial lattice ends at
# N_bin = 12, which caps the reachable arclength just above 4.5
FRAME_THETA = {name: 6.0 for name in FAMILY_NAMES} | {"binomial": 4.6}


def display_degree(name: str) -> int:
    return min(DISPLAY_DEGREE, FAMILY_MAX_DEGREE.get(name, DISPLAY_DEGREE))


def member_target(name: str, member: Any) -> Target:
    family, baseline, _ = TARGETS[name]
    return Target(
        label="displaced member",
        family=family,
        baseline=baseline,
        sample=lambda size, rng: np.asarray(family.sample(member, size, rng=rng)),
        density=lambda x: np.asarray(family.prob(x, member), dtype=float),
        members=(member,),
    )


def standardised(target: Target) -> Target:
    """Move the baseline mean to the target mean, as the fitting section
    prescribes: the reference parameters absorb the low-order structure."""

    family, baseline = target.family, target.baseline
    wide = support_grid(family, baseline, target.members, 40.0)
    density = target.density(wide)
    mean = integrate(family, baseline, wide, density * wide)
    return dataclasses.replace(
        target, baseline=dataclasses.replace(baseline, mean=float(mean))
    )


def empirical(target: Target, sample: np.ndarray, k_max: int) -> np.ndarray:
    basis = np.asarray(target.family.basis(sample, k_max, target.baseline), dtype=float)
    return basis.mean(axis=0)


def fitted_law(
    family: Any, params: Any, coefficients: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    reference = np.asarray(family.prob(grid, params), dtype=float)
    amplitude = np.asarray(family.basis_dot(grid, coefficients, params))
    return reference * np.abs(amplitude) ** 2


def variation(target: Target, law: np.ndarray, grid: np.ndarray) -> float:
    difference = np.abs(law - target.density(grid))
    return 0.5 * integrate(target.family, target.baseline, grid, difference)


# ------------------------------------------------------------------ columns --
def column_complex(name: str, rng: Any) -> dict[str, Any]:
    target = standardised(mixture_target(name, separation_for(name, 2.0)))
    family, baseline = target.family, target.baseline
    degree = display_degree(name)
    sample = target.sample(SAMPLE_SIZE, rng)
    moments = empirical(target, sample, 2 * degree)
    grid = support_grid(family, baseline, target.members)

    real = continued_fit(family, baseline, moments, degree)
    complex_fit = continued_complex_fit(
        fitting_matrices(family, baseline, degree), moments
    )["complex"]["coefficients"]

    law_real = fitted_law(family, baseline, real, grid)
    law_complex = fitted_law(family, baseline, complex_fit, grid)
    return {
        "target": target,
        "sample": sample,
        "grid": grid,
        "laws": [
            ("real fit", law_real, variation(target, law_real, grid)),
            ("complex fit", law_complex, variation(target, law_complex, grid)),
        ],
        "title": "complex amplitude",
    }


def column_frame(name: str, rng: Any) -> dict[str, Any]:
    family, baseline, _ = TARGETS[name]
    theta = FRAME_THETA[name]
    member = with_mean(family, baseline, mean_at_arclength(family, baseline, theta))
    target = member_target(name, member)
    degree = min(RECENTRED_DEGREE, display_degree(name))
    sample = target.sample(SAMPLE_SIZE, rng)
    grid = support_grid(family, baseline, target.members)

    recentred = fit_recentred(family, baseline, sample, degree)
    law_recentred = fitted_law(
        family, recentred["member"], recentred["coefficients"], grid
    )
    return {
        "target": target,
        "sample": sample,
        "grid": grid,
        "laws": [
            ("recentred fit", law_recentred, variation(target, law_recentred, grid)),
        ],
        "title": rf"movable frame, $\theta_\star={theta:g}$",
    }


def column_truncated(name: str, rng: Any) -> dict[str, Any]:
    target = truncated_target(name)
    family, baseline = target.family, target.baseline
    degree = display_degree(name)
    sample = target.sample(SAMPLE_SIZE, rng)
    moments = empirical(target, sample, 2 * degree)
    grid = support_grid(family, baseline, target.members)

    real = continued_fit(family, baseline, moments, degree)
    complex_fit = continued_complex_fit(
        fitting_matrices(family, baseline, degree), moments
    )["complex"]["coefficients"]

    law_real = fitted_law(family, baseline, real, grid)
    law_complex = fitted_law(family, baseline, complex_fit, grid)
    return {
        "target": target,
        "sample": sample,
        "grid": grid,
        "laws": [
            ("real fit", law_real, variation(target, law_real, grid)),
            ("complex fit", law_complex, variation(target, law_complex, grid)),
        ],
        "title": "truncated baseline",
    }


# ------------------------------------------------------------------ drawing --
def draw_panel(
    axis: Any, result: dict[str, Any], *, leftmost: bool, legend: bool
) -> None:
    target, grid, sample = result["target"], result["grid"], result["sample"]
    truth = target.density(grid)
    family, baseline = target.family, target.baseline
    reference = np.asarray(family.prob(grid, baseline), dtype=float)

    envelope = np.maximum(truth / truth.max(), reference / reference.max())
    window = grid[envelope > 1e-5]
    low, high = float(window.min()), float(window.max())
    inside = (grid >= low) & (grid <= high)

    if family.is_lattice(baseline):
        counts = np.array(
            [np.count_nonzero(sample == v) for v in grid], dtype=float
        ) / len(sample)
        axis.bar(grid, counts, width=0.8, color=SAMPLE_COLOUR, alpha=0.75, zorder=0)
    else:
        axis.hist(
            sample,
            bins=70,
            range=(low, high),
            density=True,
            histtype="stepfilled",
            facecolor=SAMPLE_COLOUR,
            edgecolor="0.72",
            linewidth=0.5,
            alpha=0.55,
            zorder=0,
        )
    axis.plot(
        grid,
        reference,
        linestyle=(0, (5, 2)),
        color=REFERENCE_COLOUR,
        linewidth=1.1,
        zorder=1,
        label=r"reference $p_{\rm ref}$",
    )
    axis.plot(grid, truth, color=TARGET_COLOUR, linewidth=1.5, zorder=2, label="target")
    comparator_law = None
    for comparator_label, comparator_law, _ in result["laws"][:-1]:
        axis.plot(
            grid,
            comparator_law,
            linestyle=(0, (4, 1.5, 1, 1.5)),
            color=COMPARATOR_COLOUR,
            linewidth=1.2,
            zorder=3,
            label=comparator_label,
        )
    (fit_label, fit_law, fit_tv) = result["laws"][-1]
    axis.plot(
        grid,
        fit_law,
        linestyle=(0, (2.6, 1.6)),
        color=FIT_COLOUR,
        linewidth=2.4,
        zorder=5,
        label=fit_label,
    )

    axis.set_xlim(low, high)
    axis.set_yscale("log")
    top = float(max(truth[inside].max(), fit_law[inside].max()))
    floors = [truth[inside & (truth > 0)].min()]
    for _, law, _ in result["laws"]:
        visible = law[inside & (law > 1e-12)]
        if len(visible):
            floors.append(visible.min())
    bottom = max(0.3 * float(min(floors)), 1e-8, top * 1e-9)
    axis.set_ylim(bottom, top * 30.0)
    quoted = r"\,/\,".join(f"{tv:.1e}" for _, _, tv in result["laws"])
    axis.set_title(rf"{result['title']}:  $D={quoted}$", fontsize=8.5)
    axis.tick_params(labelsize=7.5)
    if leftmost:
        axis.set_ylabel("density", fontsize=9)
    if legend:
        axis.legend(
            frameon=False,
            fontsize=6.4,
            loc="upper left",
            handlelength=2.0,
            borderpad=0.15,
            labelspacing=0.25,
        )


def make_figure(names: tuple[str, ...], output: Path, seed: int) -> Path:
    figure, axes = plt.subplots(len(names), 3, figsize=(11.0, 2.35 * len(names) + 0.4))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        rng = np.random.default_rng(seed)
        columns = [
            column_complex(name, rng),
            column_frame(name, rng),
            column_truncated(name, rng),
        ]
        for col, result in enumerate(columns):
            draw_panel(
                axes[row, col],
                result,
                leftmost=(col == 0),
                legend=(row == 0),
            )
            print(
                f"  {name:18s} {result['title'][:16]:18s}"
                + "  ".join(f"{label} {tv:.2e}" for label, _, tv in result["laws"])
            )
        axes[row, 0].text(
            -0.24,
            0.5,
            FAMILY_LABELS[name],
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "full-model-fits.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="*", default=list(FAMILY_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if args.output is None
        else Path(args.output)
    )
    path = make_figure(tuple(args.families), output, args.seed)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
