"""The recentred-fit figure of the note: the displacement wall removed.

Two panels, Poisson and Normal at fixed degree K = 4: total variation of
the plain complex fit against the joint recentred fit, swept over the
displacement of a shifted-member target, with the wall at theta = 2 sqrt(K).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_recentred import run_offset_sweep

FAMILIES = ("poisson", "normal")
DEGREE = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/recentred_fit")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for axis, name in zip(axes, FAMILIES, strict=True):
        print(f"{name}, degree {DEGREE}")
        result = run_offset_sweep(name, degree=DEGREE, seed=args.seed)
        rows = result["rows"]
        thetas = [row["true_theta"] for row in rows]
        axis.semilogy(
            thetas,
            [row["plain_tv"] for row in rows],
            "o-",
            color="#b0413e",
            label="fixed baseline",
        )
        axis.semilogy(
            thetas,
            [row["recentred_tv"] for row in rows],
            "s-",
            color="#2a6f97",
            label="recentred",
        )
        wall = 2.0 * np.sqrt(DEGREE)
        axis.axvline(wall, color="0.6", ls=":", lw=1.0)
        axis.text(
            wall,
            axis.get_ylim()[1],
            r" $\theta=2\sqrt{K}$",
            va="top",
            fontsize=9,
            color="0.4",
        )
        axis.set_xlabel(r"displacement $d_{\mathrm{FR}}$")
        axis.set_title(name.capitalize())
    axes[0].set_ylabel("total variation")
    axes[0].legend(fontsize=9, frameon=False)
    figure.tight_layout()

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "recentred-sweep.pdf"
    figure.savefig(path)
    plt.close(figure)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
