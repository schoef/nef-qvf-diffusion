"""The site-2 frame race: H_T against log H_T across baseline families.

Still no fit: coefficients are projections.  What changes between the
cases is the frame -- the variable, the baseline family, and how many
low moments the frame absorbs before the first nonzero mode.  The
centering observation drives the design: a family with two parameters
plus a support shift has three handles, enough to zero R_1, R_2, R_3.

Cases (each published to its own folder under TT2l-study):

  ht/gamma            x = H_T - 60,        Gamma,  2 moments matched;
  log-ht/normal       v = log H_T,         Normal, 2 moments;
  log-ht/gamma        v = log H_T,         Gamma with free threshold,
                                           3 moments (feasible: theta
                                           = 3.31 < log 60);
  log-ht/ghs          v = log H_T shifted, GHS skew-matched, 3 moments;
  log-ht-m60/normal   v = log(H_T - 60),   Normal, 2 moments;
  log-ht-m60/ghs      v = log(H_T - 60) shifted, GHS, 3 moments.

All truncations are mapped back to H_T and scored in total variation
against one common held-out H_T histogram, so the numbers compare
across frames.

Predictions registered before running:

  P3  the three-moment frames (log-ht/gamma, log-ht/ghs) beat every
      two-moment frame and reach the sampling floor by K ~ 8;
  P4  among the two-moment log frames Normal and GHS start alike and
      GHS wins in the tail modes;
  P5  the linear-H_T Gamma stays worst.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT

from applications.cms.law_projections import K_MAX, load, negative_mass, projections
from applications.cms.www import publish
from nefqvf import GHS, GHSParams, Gamma, GammaParams, Normal, NormalParams

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

OUT = Path("/users/robert.schoefbeck/ML/nef-qvf-diffusion/artifacts/cms-ttbar/ht-frames")
HT_EDGES = np.linspace(0.0, 1600.0, 81)
COLORS = (ROOT.kRed, ROOT.kOrange + 1, ROOT.kGreen + 2, ROOT.kBlue,
          ROOT.kViolet, ROOT.kCyan + 2)


def three_moment_gamma(values: np.ndarray):
    """Shifted Gamma matching mean, variance, skewness; None if infeasible."""

    mu, sd = values.mean(), values.std()
    skew = float(((values - mu) ** 3).mean()) / sd**3
    if skew <= 0:
        return None
    nu = (2.0 / skew) ** 2
    scale = sd / np.sqrt(nu)
    theta = mu - nu * scale
    if theta >= values.min():
        return None
    return theta, GammaParams(mean=nu * scale, r=nu)


def three_moment_ghs(values: np.ndarray):
    """Shifted GHS matching mean, variance, skewness (always feasible)."""

    mu, sd = values.mean(), values.std()
    skew = float(((values - mu) ** 3).mean()) / sd**3
    r = 2.0 * sd**2 / (1.0 + skew**2 * sd**2)
    mean = skew * sd * r
    shift = mu - mean
    return shift, GHSParams(mean=mean, r=r)


def two_moment_normal(values: np.ndarray):
    return 0.0, NormalParams(mean=values.mean(), sigma=values.std())


def two_moment_gamma(values: np.ndarray):
    mu, var = values.mean(), values.var()
    return 0.0, GammaParams(mean=mu, r=mu**2 / var)


# The cases: name -> (folder, transform ht->v, dv/dht, frame builder, family)
def build_cases(ht_train):
    cases = {}
    cases["ht/gamma"] = (
        lambda ht: ht - 60.0, lambda ht: np.ones_like(ht),
        two_moment_gamma, Gamma, "x = H_{T} - 60, Gamma (2 moments)")
    cases["log-ht/normal"] = (
        np.log, lambda ht: 1.0 / ht,
        two_moment_normal, Normal, "v = log H_{T}, Normal (2 moments)")
    cases["log-ht/gamma"] = (
        np.log, lambda ht: 1.0 / ht,
        three_moment_gamma, Gamma, "v = log H_{T}, shifted Gamma (3 moments)")
    cases["log-ht/ghs"] = (
        np.log, lambda ht: 1.0 / ht,
        three_moment_ghs, GHS, "v = log H_{T}, shifted GHS (3 moments)")
    cases["log-ht-m60/normal"] = (
        lambda ht: np.log(ht - 60.0 + 1e-9), lambda ht: 1.0 / (ht - 60.0 + 1e-9),
        two_moment_normal, Normal, "v = log(H_{T}-60), Normal (2 moments)")
    cases["log-ht-m60/ghs"] = (
        lambda ht: np.log(ht - 60.0 + 1e-9), lambda ht: 1.0 / (ht - 60.0 + 1e-9),
        three_moment_ghs, GHS, "v = log(H_{T}-60), shifted GHS (3 moments)")
    return cases


def run_case(name, transform, jacobian, builder, family, ht_train, ht_held,
             held_counts):
    v_train = transform(ht_train)
    frame = builder(v_train)
    if frame is None:
        print(f"{name}: three-moment frame infeasible, skipped")
        return None
    shift, params = frame
    w_train = v_train - shift
    coefficients, errors = projections(family, params, w_train, 2 * K_MAX)

    centres = 0.5 * (HT_EDGES[:-1] + HT_EDGES[1:])
    cell = np.diff(HT_EDGES)
    w_centres = transform(np.maximum(centres, 60.5)) - shift
    basis = np.asarray(family.basis(w_centres, K_MAX, params), dtype=float)
    reference = np.asarray(family.prob(w_centres, params), dtype=float)
    jac = jacobian(np.maximum(centres, 60.5))
    inside = centres > 60.0

    tvs, laws = [], {}
    for k in range(K_MAX + 1):
        law_v = reference * (basis[:, : k + 1] @ coefficients[: k + 1])
        law_ht = np.where(inside, law_v * jac, 0.0)
        laws[k] = law_ht
        mass = law_ht * cell
        tvs.append(0.5 * float(np.sum(np.abs(mass - held_counts)))
                   + 0.5 * abs(1.0 - float(np.sum(mass))))
    negmass = -float(np.sum(np.minimum(laws[K_MAX], 0.0) * cell))
    print(
        f"{name}: shift {shift:+.3f}  |R_k| "
        + " ".join(f"{abs(c):.4f}" for c in coefficients[1:8])
        + "  TV(K) " + " ".join(f"{t:.4f}" for t in tvs[: K_MAX + 1: 2])
        + f"  negmass {negmass:.2e}"
    )
    return {
        "coefficients": coefficients, "errors": errors, "tv": tvs,
        "laws": laws, "centres": centres, "cell": cell,
    }


def draw_case(name, label, res, held_counts):
    canvas = ROOT.TCanvas(f"c{name.replace('/', '_')}", "", 900, 650)
    canvas.SetLogy()
    edges = HT_EDGES
    h_data = ROOT.TH1D(f"h{name}", f";H_{{T}} [GeV];density  --  {label}",
                       len(edges) - 1, edges[0], edges[-1])
    cell = np.diff(edges)
    for i, c in enumerate(held_counts):
        h_data.SetBinContent(i + 1, c / cell[i])
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(3)
    h_data.SetMinimum(1e-9)
    h_data.Draw("hist")
    keep = [h_data]
    legend = ROOT.TLegend(0.45, 0.6, 0.89, 0.89)
    legend.AddEntry(h_data, "data (held-out half)", "l")
    for index, k in enumerate((0, 2, 4, 8, 12)):
        graph = ROOT.TGraph(len(res["centres"]))
        for i, (x, y) in enumerate(zip(res["centres"], res["laws"][k])):
            graph.SetPoint(i, x, max(y, 1e-12))
        graph.SetLineColor(COLORS[index])
        graph.SetLineWidth(2)
        graph.Draw("l same")
        keep.append(graph)
        legend.AddEntry(graph, f"K={k}  (TV {res['tv'][k]:.4f})", "l")
    legend.SetTextSize(0.03)
    legend.Draw()
    stem = OUT / name.replace("/", "-")
    canvas.SaveAs(str(stem) + ".png")
    canvas.SaveAs(str(stem) + ".pdf")
    publish([str(stem) + ".png", str(stem) + ".pdf"], name)
    return keep


def draw_summary(results):
    canvas = ROOT.TCanvas("csum", "", 1600, 550)
    canvas.Divide(2, 1)
    keep = []
    pad = canvas.cd(1)
    pad.SetLogy()
    frame = ROOT.TH1D("fs2", ";mode k;|R_k| and its sampling error",
                      2 * K_MAX, 0.5, 2 * K_MAX + 0.5)
    frame.SetMinimum(1e-6)
    frame.SetMaximum(2.0)
    frame.Draw()
    keep.append(frame)
    legend = ROOT.TLegend(0.42, 0.62, 0.89, 0.89)
    for index, (name, res) in enumerate(results.items()):
        graph = ROOT.TGraphErrors(2 * K_MAX)
        for k in range(1, 2 * K_MAX + 1):
            graph.SetPoint(k - 1, k, max(abs(res["coefficients"][k]), 1e-12))
            graph.SetPointError(k - 1, 0.0, res["errors"][k])
        graph.SetLineColor(COLORS[index])
        graph.SetMarkerColor(COLORS[index])
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.6)
        graph.Draw("pl same")
        keep.append(graph)
        legend.AddEntry(graph, name, "pl")
    legend.SetTextSize(0.03)
    legend.Draw()
    keep.append(legend)

    pad = canvas.cd(2)
    pad.SetLogy()
    frame2 = ROOT.TH1D("ft2", ";truncation K;TV to held-out H_{T} (common bins)",
                       K_MAX + 1, -0.5, K_MAX + 0.5)
    frame2.SetMinimum(5e-4)
    frame2.SetMaximum(1.0)
    frame2.Draw()
    keep.append(frame2)
    legend2 = ROOT.TLegend(0.42, 0.62, 0.89, 0.89)
    for index, (name, res) in enumerate(results.items()):
        graph = ROOT.TGraph(K_MAX + 1)
        for k in range(K_MAX + 1):
            graph.SetPoint(k, k, res["tv"][k])
        graph.SetLineColor(COLORS[index])
        graph.SetMarkerColor(COLORS[index])
        graph.SetMarkerStyle(21)
        graph.SetMarkerSize(0.6)
        graph.SetLineWidth(2)
        graph.Draw("pl same")
        keep.append(graph)
        legend2.AddEntry(graph, name, "pl")
    legend2.SetTextSize(0.03)
    legend2.Draw()
    keep.append(legend2)
    canvas.SaveAs(str(OUT / "summary.png"))
    canvas.SaveAs(str(OUT / "summary.pdf"))
    publish([str(OUT / "summary.png"), str(OUT / "summary.pdf")],
            "ht-frames-summary")
    return keep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nj, ht = load()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ht))
    half = len(ht) // 2
    ht_train, ht_held = ht[order[:half]], ht[order[half:]]
    held_counts, _ = np.histogram(ht_held, bins=HT_EDGES)
    held_counts = held_counts / len(ht_held)

    results = {}
    keep = []
    for name, (transform, jacobian, builder, family, label) in build_cases(
        ht_train
    ).items():
        res = run_case(name, transform, jacobian, builder, family,
                       ht_train, ht_held, held_counts)
        if res is None:
            continue
        results[name] = res
        keep += draw_case(name, label, res, held_counts)
    keep += draw_summary(results)
    # browsable parent folders
    publish([], "ht")
    publish([], "log-ht")
    publish([], "log-ht-m60")
    publish([], "")


if __name__ == "__main__":
    main()
