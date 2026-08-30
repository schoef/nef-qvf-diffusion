"""The entanglement gauge: representative selection on the amplitude fibre.

A law fixes the amplitude only up to a phase function ("the fibre").
For a strictly positive two-site law the real fibre is trivial -- the
positive branch is the only real representative -- while the complex
fibre is large, and different representatives carry different
entanglement.

Two targets, noised sitewise (components N(+-d e^{-t}, 1) against the
N(0,1) baseline):

  mixture       q_t = (r1 r1' + r2 r2')/2      -- classical data;
  interference  q_t = |h1 h1' - h2 h2'|^2/Z    -- amplitude with a node.

Finding: for the mixture the positive branch sqrt(q_t) has LOWER
Schmidt entropy than the exact rank-two complex representative at every
slice -- classical mixtures never need the phase, and the chambers met
in fitting are truncation artifacts of the polynomial class, not
properties of the law's fibre.  For the interference law the predicted
crossover exists: the positive branch exceeds the signed representative
(S > log 2) exactly below the structure-birth time where the node
forms.  Both entropies grow smoothly from 0 along the schedule: the
semigroup distributes the structure in t.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = np.linspace(-9.0, 9.0, 481)
FIGURE_SUBDIRECTORY = "entanglement_gauge"


def component_ratio(grid: np.ndarray, mean: float) -> np.ndarray:
    """Density ratio N(mean, 1) / N(0, 1) on the grid."""

    return np.exp(mean * grid - 0.5 * mean * mean)


def representatives(d: float, t: float, w1: float = 0.5):
    """Positive-branch and complex rank-two amplitudes of the same law."""

    m = d * np.exp(-t)
    r1 = component_ratio(GRID, +m)
    r2 = component_ratio(GRID, -m)
    q = w1 * np.outer(r1, r1) + (1.0 - w1) * np.outer(r2, r2)
    h_pos = np.sqrt(q)
    h_cx = np.sqrt(w1) * np.outer(np.sqrt(r1), np.sqrt(r1)) + 1j * np.sqrt(
        1.0 - w1
    ) * np.outer(np.sqrt(r2), np.sqrt(r2))
    assert np.max(np.abs(np.abs(h_cx) ** 2 - q)) < 1e-9 * np.max(q)
    return q, h_pos, h_cx


def schmidt_entropy(h: np.ndarray) -> tuple[float, np.ndarray]:
    """Entanglement entropy of an amplitude kernel in L2(ref) x L2(ref)."""

    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    w = np.sqrt(ref * np.gradient(GRID))
    kernel = w[:, None] * h * w[None, :]
    s = np.linalg.svd(kernel, compute_uv=False)
    p = s**2 / np.sum(s**2)
    p = p[p > 1e-300]
    return float(-np.sum(p * np.log(p))), s / s[0]


def trough_ratio(q: np.ndarray) -> float:
    """Depth of the inter-mode trough of the joint density, trough / peak."""

    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    density = np.diagonal(q) * ref * ref
    centre = density[len(density) // 2]
    return float(centre / np.max(density))


def interference_representatives(d: float, t: float):
    """A law whose amplitude genuinely carries a node: q = |h1 h1' - h2 h2'|^2
    normalised.  The signed representative is rank two; the positive branch
    is |h| with a nodal kink."""

    m = d * np.exp(-t)
    r1 = component_ratio(GRID, +m)
    r2 = component_ratio(GRID, -m)
    signed = np.outer(np.sqrt(r1), np.sqrt(r1)) - np.outer(np.sqrt(r2), np.sqrt(r2))
    ref = np.exp(-0.5 * GRID**2) / np.sqrt(2.0 * np.pi)
    w = ref * np.gradient(GRID)
    norm = np.sqrt(np.sum(w[:, None] * signed**2 * w[None, :]))
    signed = signed / norm
    return np.abs(signed), signed


def run_sweep(d: float, times: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for t in times:
        q, h_pos, h_cx = representatives(d, float(t))
        h_ipos, h_isigned = interference_representatives(d, float(t))
        s_ipos, _ = schmidt_entropy(h_ipos)
        s_isigned, _ = schmidt_entropy(h_isigned)
        s_pos, spec_pos = schmidt_entropy(h_pos)
        s_cx, spec_cx = schmidt_entropy(h_cx)
        rows.append(
            {
                "t": float(t),
                "s_pos": s_pos,
                "s_cx": s_cx,
                "rank_cx": int(np.sum(spec_cx > 1e-10)),
                "trough": trough_ratio(q),
                "s_ipos": s_ipos,
                "s_isigned": s_isigned,
            }
        )
    return rows


def plot_sweep(rows: list[dict[str, Any]], d: float, output_dir: Any = None) -> str:
    t = [r["t"] for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(
        t, [r["s_pos"] for r in rows], "o-", color="#b0413e", label="positive branch"
    )
    axes[0].plot(
        t,
        [r["s_cx"] for r in rows],
        "s-",
        color="#2a6f97",
        label="complex branch (rank 2)",
    )
    axes[0].axhline(np.log(2.0), color="0.6", ls=":", lw=1.0)
    axes[0].text(t[0], np.log(2.0), r" $\log 2$", va="bottom", fontsize=8, color="0.4")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("Schmidt entropy $S$")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_title(rf"two representatives of one law, $d = {d:g}$")
    axes[1].plot(t, [r["trough"] for r in rows], "o-", color="0.3")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("inter-mode trough / peak")
    axes[1].invert_xaxis()
    axes[1].set_title("structure birth")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"entanglement-gauge-d{d:g}.png"
    figure.savefig(path, dpi=150)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=2.5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    times = np.linspace(2.0, 0.0, 17)
    rows = run_sweep(args.d, times)
    for r in rows:
        print(
            f"  t {r['t']:5.2f}  S_pos {r['s_pos']:8.5f}  S_cx {r['s_cx']:8.5f}"
            f"  trough {r['trough']:6.3f}  |  interference: "
            f"S_pos {r['s_ipos']:8.5f}  S_signed {r['s_isigned']:8.5f}"
        )
    if args.plot:
        print(f"figure: {plot_sweep(rows, args.d, args.output)}")


if __name__ == "__main__":
    main()
