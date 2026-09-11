"""1D law-level projections for (nSelJet, ht) in dileptonic ttbar.

There is no fit at this level: the law is represented by linear modes
whose coefficients are projections,

    R_k = E[phi_k(X)],  estimated as sample means with errors sd/sqrt(N),

and the truncated law pi(x) sum_{k<=K} R_k phi_k(x) overlays directly
on the truth.  Capacity usage is read off the spectrum decay and the
TV(K) curves; the plug-in truncation can dip negative in the tails
(the section-8 warning box), and the negative mass is reported rather
than hidden.

Four cases, projected on one half of the sample and scored in total
variation against the other half:

  m  = nSelJet - 2   on a Poisson baseline (control) and on the
                     moment-matched negative binomial;
  x  = ht - 60       on a moment-matched Gamma (frame a, fixed);
  u  = ht - 30 nSelJet on a moment-matched Gamma (frame b, kinematic).

Predictions registered before running:

  P1  the negative-binomial baseline reaches the TV floor by K ~ 3
      while the Poisson baseline needs roughly twice that -- baseline
      choice is mode compression;
  P2  the kinematic frame u is more Gamma-like than ht - 60: faster
      spectrum decay and a lower-K TV floor.  (The frames describe
      different 1D variables; the decisive frame race is the pair
      unfolding of the next step.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT

from applications.cms.www import publish
from nefqvf import (
    Gamma,
    GammaParams,
    NegativeBinomial,
    NegativeBinomialParams,
    Poisson,
    PoissonParams,
)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

NTUPLE = (
    "/eos/vbc/group/cms/robert.schoefbeck/CMGRDF_ntuples/"
    "v2-3-2_nJ2p_nB2p_2l/2018/TTLep_pow_nominal.root"
)
CACHE = Path("/users/robert.schoefbeck/ML/quicklook/nj_ht_w.npy")
OUT = Path("/users/robert.schoefbeck/ML/nef-qvf-diffusion/artifacts/cms-ttbar")
K_MAX = 12
COLORS = (ROOT.kRed, ROOT.kOrange + 1, ROOT.kGreen + 2, ROOT.kBlue, ROOT.kViolet)


def load() -> tuple[np.ndarray, np.ndarray]:
    if CACHE.exists():
        arr = np.load(CACHE)
        return arr[:, 0], arr[:, 1]
    df = ROOT.RDataFrame("Events", NTUPLE)
    arr = df.AsNumpy(columns=["nSelJet", "ht"])
    nj = np.asarray(arr["nSelJet"], dtype=float).ravel()
    ht = np.asarray(arr["ht"], dtype=float).ravel()
    return nj, ht


def projections(family, params, values: np.ndarray, k_max: int, chunk: int = 1_000_000):
    """OPS projections R_k with their sampling errors, streamed in chunks."""

    total = np.zeros(k_max + 1)
    total_sq = np.zeros(k_max + 1)
    for start in range(0, len(values), chunk):
        basis = np.asarray(
            family.basis(values[start : start + chunk], k_max, params), dtype=float
        )
        total += basis.sum(axis=0)
        total_sq += (basis**2).sum(axis=0)
    n = len(values)
    coefficients = total / n
    variance = np.maximum(total_sq / n - coefficients**2, 0.0)
    errors = np.sqrt(variance / n)
    return coefficients, errors


def truncated_law(family, params, coefficients, grid, k):
    basis = np.asarray(family.basis(grid, K_MAX, params), dtype=float)
    reference = np.asarray(family.prob(grid, params), dtype=float)
    return reference * (basis[:, : k + 1] @ coefficients[: k + 1])


def tv_discrete(law, counts):
    return 0.5 * float(np.sum(np.abs(law - counts)))


def tv_binned(law_grid, grid, edges, counts):
    """TV of a continuous law against a held-out histogram."""

    cell = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    law = np.interp(centres, grid, law_grid) * cell
    return 0.5 * float(np.sum(np.abs(law - counts))) + 0.5 * abs(
        1.0 - float(np.sum(law))
    )


def negative_mass(law_grid, grid):
    return -float(np.trapz(np.minimum(law_grid, 0.0), grid))


# ---------------------------------------------------------------- discrete --
def discrete_case(m_train, m_held):
    mu, var = m_train.mean(), m_train.var()
    cases = {
        "Poisson": (Poisson, PoissonParams(mean=mu)),
        "neg. binomial": (
            NegativeBinomial,
            NegativeBinomialParams(mean=mu, r=mu**2 / (var - mu)),
        ),
    }
    grid = np.arange(0.0, 14.0)
    held_counts = np.array([np.mean(m_held == k) for k in grid])

    results = {}
    for label, (family, params) in cases.items():
        coefficients, errors = projections(family, params, m_train, 2 * K_MAX)
        tvs, laws = [], {}
        for k in range(K_MAX + 1):
            law = truncated_law(family, params, coefficients, grid, k)
            tvs.append(tv_discrete(law, held_counts))
            laws[k] = law
        results[label] = {
            "coefficients": coefficients,
            "errors": errors,
            "tv": tvs,
            "laws": laws,
            "family": family,
            "params": params,
        }
        print(
            f"m on {label}: |R_k| "
            + " ".join(f"{abs(c):.4f}" for c in coefficients[1:8])
            + "  TV(K) "
            + " ".join(f"{t:.4f}" for t in tvs[:7])
        )
    return grid, held_counts, results


# -------------------------------------------------------------- continuous --
def continuous_case(values_train, values_held, label):
    mu, var = values_train.mean(), values_train.var()
    params = GammaParams(mean=mu, r=mu**2 / var)
    grid = np.linspace(1e-3, 1600.0, 3200)
    edges = np.linspace(0.0, 1600.0, 81)
    counts, _ = np.histogram(values_held, bins=edges)
    counts = counts / len(values_held)

    coefficients, errors = projections(Gamma, params, values_train, 2 * K_MAX)
    tvs, laws, negmass = [], {}, []
    for k in range(K_MAX + 1):
        law = truncated_law(Gamma, params, coefficients, grid, k)
        tvs.append(tv_binned(law, grid, edges, counts))
        negmass.append(negative_mass(law, grid))
        laws[k] = law
    print(
        f"{label}: shape nu {params.r:.3f}  |R_k| "
        + " ".join(f"{abs(c):.4f}" for c in coefficients[1:8])
        + "  TV(K) "
        + " ".join(f"{t:.4f}" for t in tvs[:7])
        + "  negmass(K_MAX) "
        + f"{negmass[-1]:.2e}"
    )
    return {
        "grid": grid,
        "edges": edges,
        "counts": counts,
        "coefficients": coefficients,
        "errors": errors,
        "tv": tvs,
        "laws": laws,
        "negmass": negmass,
        "params": params,
    }


# ------------------------------------------------------------------- plots --
def draw_discrete(grid, held_counts, results, canvas_pad):
    canvas_pad.SetLogy()
    h_data = ROOT.TH1D("hd", ";m = nSelJet - 2;probability", 14, -0.5, 13.5)
    for i, c in enumerate(held_counts):
        h_data.SetBinContent(i + 1, c)
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(3)
    h_data.SetMinimum(1e-8)
    h_data.SetMaximum(5.0)
    h_data.Draw("hist")
    keep = [h_data]
    legend = ROOT.TLegend(0.38, 0.55, 0.89, 0.89)
    legend.AddEntry(h_data, "data (held-out half)", "l")
    entry = 0
    for label, res in results.items():
        for k, style in ((0, 2), (2, 1), (4, 1)):
            h = h_data.Clone(f"h{label}{k}")
            for i in range(14):
                h.SetBinContent(i + 1, max(res["laws"][min(k, K_MAX)][i], 1e-12))
            color = ROOT.kRed if "Pois" in label else ROOT.kBlue
            h.SetLineColor(color + (k // 2))
            h.SetLineStyle(style)
            h.SetLineWidth(2)
            h.Draw("hist same")
            keep.append(h)
            legend.AddEntry(h, f"{label}, K={k}", "l")
            entry += 1
    legend.SetTextSize(0.03)
    legend.Draw()
    keep.append(legend)
    return keep


def draw_continuous(res, title, canvas_pad):
    canvas_pad.SetLogy()
    edges, counts = res["edges"], res["counts"]
    h_data = ROOT.TH1D(f"hc{title}", f";{title};density", len(edges) - 1, edges[0], edges[-1])
    cell = np.diff(edges)
    for i, c in enumerate(counts):
        h_data.SetBinContent(i + 1, c / cell[i])
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(3)
    h_data.SetMinimum(1e-9)
    h_data.Draw("hist")
    keep = [h_data]
    legend = ROOT.TLegend(0.45, 0.6, 0.89, 0.89)
    legend.AddEntry(h_data, "data (held-out half)", "l")
    for index, k in enumerate((0, 2, 4, 8, 12)):
        graph = ROOT.TGraph(len(res["grid"]))
        for i, (x, y) in enumerate(zip(res["grid"], res["laws"][k])):
            graph.SetPoint(i, x, max(y, 1e-12))
        graph.SetLineColor(COLORS[index])
        graph.SetLineWidth(2)
        graph.Draw("l same")
        keep.append(graph)
        legend.AddEntry(graph, f"K={k}  (TV {res['tv'][k]:.4f})", "l")
    legend.SetTextSize(0.03)
    legend.Draw()
    keep.append(legend)
    return keep


def draw_summary(results_m, res_a, res_b, canvas_pad_spec, canvas_pad_tv):
    canvas_pad_spec.SetLogy()
    frame = ROOT.TH1D("fs", ";mode k;|R_k| and its sampling error", 2 * K_MAX, 0.5, 2 * K_MAX + 0.5)
    frame.SetMinimum(1e-6)
    frame.SetMaximum(2.0)
    frame.Draw()
    keep = [frame]
    legend = ROOT.TLegend(0.5, 0.65, 0.89, 0.89)
    series = [
        ("m, Poisson", results_m["Poisson"], ROOT.kRed),
        ("m, neg. binomial", results_m["neg. binomial"], ROOT.kBlue),
        ("H_{T} - 60, Gamma", res_a, ROOT.kGreen + 2),
        ("u = H_{T} - 30 n, Gamma", res_b, ROOT.kViolet),
    ]
    for label, res, color in series:
        graph = ROOT.TGraphErrors(2 * K_MAX)
        for k in range(1, 2 * K_MAX + 1):
            graph.SetPoint(k - 1, k, max(abs(res["coefficients"][k]), 1e-12))
            graph.SetPointError(k - 1, 0.0, res["errors"][k])
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.7)
        graph.Draw("pl same")
        keep.append(graph)
        legend.AddEntry(graph, label, "pl")
    legend.SetTextSize(0.03)
    legend.Draw()
    keep.append(legend)

    canvas_pad_tv.SetLogy()
    frame2 = ROOT.TH1D("ft", ";truncation K;TV to held-out half", K_MAX + 1, -0.5, K_MAX + 0.5)
    frame2.SetMinimum(2e-4)
    frame2.SetMaximum(1.0)
    frame2.Draw()
    keep.append(frame2)
    legend2 = ROOT.TLegend(0.5, 0.65, 0.89, 0.89)
    for label, res, color in series:
        graph = ROOT.TGraph(K_MAX + 1)
        for k in range(K_MAX + 1):
            graph.SetPoint(k, k, res["tv"][k])
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(21)
        graph.SetMarkerSize(0.7)
        graph.SetLineWidth(2)
        graph.Draw("pl same")
        keep.append(graph)
        legend2.AddEntry(graph, label, "pl")
    legend2.SetTextSize(0.03)
    legend2.Draw()
    keep.append(legend2)
    return keep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nj, ht = load()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(nj))
    half = len(nj) // 2
    a_idx, b_idx = order[:half], order[half:]

    m_train, m_held = nj[a_idx] - 2, nj[b_idx] - 2
    grid_m, held_m, results_m = discrete_case(m_train, m_held)

    x = ht - 60.0
    res_a = continuous_case(x[a_idx], x[b_idx], "frame a: ht - 60")
    u = ht - 30.0 * nj
    res_b = continuous_case(u[a_idx], u[b_idx], "frame b: ht - 30 nSelJet")

    canvas = ROOT.TCanvas("c", "", 1600, 1000)
    canvas.Divide(2, 2)
    keep = []
    keep += draw_discrete(grid_m, held_m, results_m, canvas.cd(1))
    keep += draw_continuous(res_a, "H_{T} - 60 [GeV]", canvas.cd(2))
    keep += draw_continuous(res_b, "u = H_{T} - 30 nSelJet [GeV]", canvas.cd(3))
    canvas.cd(4)
    summary = ROOT.TCanvas("cs", "", 1600, 500)
    summary.Divide(2, 1)
    keep += draw_summary(results_m, res_a, res_b, summary.cd(1), summary.cd(2))

    canvas.SaveAs(str(OUT / "law-projections.pdf"))
    canvas.SaveAs(str(OUT / "law-projections.png"))
    summary.SaveAs(str(OUT / "law-projections-summary.pdf"))
    summary.SaveAs(str(OUT / "law-projections-summary.png"))
    publish(
        [
            OUT / "law-projections.png",
            OUT / "law-projections.pdf",
            OUT / "law-projections-summary.png",
            OUT / "law-projections-summary.pdf",
        ],
        "1d-projections",
    )


if __name__ == "__main__":
    main()
