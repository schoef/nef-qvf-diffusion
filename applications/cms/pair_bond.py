"""The joint law of (nSelJet, log H_T): staging in bond dimension.

The setup is frozen: site 1 is m = nSelJet - 2 on the moment-matched
negative binomial, site 2 is v = log H_T on the three-moment shifted
Gamma (theta, nu from the train half).  The pair coefficients are
projections,

    R_{n1 n2} = E[phi_{n1}(m) psi_{n2}(w)],   w = v - theta',

and the staged models preserve the marginals exactly: with the
cross-cut residual C = R - R_{.,0} R_{0,.} (whose first row and column
vanish identically),

    R^(r) = marginal product + sum_{c<=r} s_c u_c (x) w_c

from the singular decomposition of C.  r = 0 is exact independence;
the tensor-train bond dimension across the cut is at most 1 + r.

Validation against the held-out half, in bins of multiplicity:
distribution level (conditional laws p(v | m=k) overlaid on data) and
coefficient level (conditional coefficients E[psi_{n2}(w) | m=k],
data with sampling errors against the staged models).

Predictions registered before running:

  P6  two to three significant channels; s_1 dominant, r = 1 carries
      the populated bins (m <= 3) most of the way and r = 2 reaches
      their sampling floor;
  P7  channel 1 is the scale channel: u_1 dominated by phi_1(m),
      w_1 by psi_1(v) -- mean log H_T linear in multiplicity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT

from applications.cms.law_projections import load
from applications.cms.ht_frames import three_moment_gamma
from applications.cms.www import publish
from nefqvf import Gamma, NegativeBinomial, NegativeBinomialParams

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

OUT = Path("/users/robert.schoefbeck/ML/nef-qvf-diffusion/artifacts/cms-ttbar/pair")
K1, K2 = 6, 8
RANKS = (0, 1, 2, 3)
M_BINS = (0, 1, 2, 3, 4, 5)
COLORS = (ROOT.kRed, ROOT.kOrange + 1, ROOT.kGreen + 2, ROOT.kBlue)


def pair_projections(b1: np.ndarray, b2: np.ndarray, chunk: int = 1_000_000):
    """R_{n1 n2} = mean of the basis outer products, with sampling errors."""

    n = len(b1)
    total = np.zeros((b1.shape[1], b2.shape[1]))
    total_sq = np.zeros_like(total)
    for start in range(0, n, chunk):
        prod = np.einsum(
            "si,sj->ij", b1[start : start + chunk], b2[start : start + chunk]
        )
        total += prod
        total_sq += np.einsum(
            "si,sj->ij",
            b1[start : start + chunk] ** 2,
            b2[start : start + chunk] ** 2,
        )
    coefficients = total / n
    errors = np.sqrt(np.maximum(total_sq / n - coefficients**2, 0.0) / n)
    return coefficients, errors


def staged_models(R: np.ndarray):
    """Marginal-preserving models R^(r) from the SVD of the cross residual."""

    product = np.outer(R[:, 0], R[0, :])
    residual = R - product
    u, s, vt = np.linalg.svd(residual)
    models = {}
    for r in RANKS:
        models[r] = product + (u[:, :r] * s[:r]) @ vt[:r]
    return models, s, u, vt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nj, ht = load()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(nj))
    half = len(nj) // 2
    tr, he = order[:half], order[half:]
    m, v = nj - 2, np.log(ht)

    # frozen frames, train half only
    mu, var = m[tr].mean(), m[tr].var()
    nb_params = NegativeBinomialParams(mean=mu, r=mu**2 / (var - mu))
    theta, gamma_params = three_moment_gamma(v[tr])
    w = v - theta
    print(
        f"frozen frames: NB(mu={mu:.4f}, r={nb_params.r:.3f});"
        f" Gamma(theta={theta:.3f}, nu={gamma_params.r:.2f},"
        f" mean={gamma_params.mean:.3f})"
    )

    b1 = np.asarray(NegativeBinomial.basis(m[tr], K1, nb_params), dtype=float)
    b2 = np.asarray(Gamma.basis(w[tr], K2, gamma_params), dtype=float)
    R, dR = pair_projections(b1, b2)
    models, singular, u_vec, vt_vec = staged_models(R)

    noise = float(np.median(dR)) * np.sqrt(max(R.shape))
    print("cross-channel singular values:", " ".join(f"{x:.4f}" for x in singular[:6]))
    print(f"noise scale for a null channel ~ {noise:.5f}")
    print("channel 1 site-1 vector:", " ".join(f"{x:+.3f}" for x in u_vec[:, 0]))
    print("channel 1 site-2 vector:", " ".join(f"{x:+.3f}" for x in vt_vec[0]))
    print("channel 2 site-1 vector:", " ".join(f"{x:+.3f}" for x in u_vec[:, 1]))
    print("channel 2 site-2 vector:", " ".join(f"{x:+.3f}" for x in vt_vec[1]))

    # held-out conditional data per multiplicity bin
    grid = np.linspace(np.log(60.0), np.log(2500.0), 600)
    wg = grid - theta
    basis_g = np.asarray(Gamma.basis(wg, K2, gamma_params), dtype=float)
    ref_g = np.asarray(Gamma.prob(wg, gamma_params), dtype=float)
    phi_k = np.asarray(
        NegativeBinomial.basis(np.arange(0.0, 12.0), K1, nb_params), dtype=float
    )
    pi_k = np.asarray(
        NegativeBinomial.prob(np.arange(0.0, 12.0), nb_params), dtype=float
    )

    edges = np.linspace(np.log(60.0), np.log(2500.0), 61)
    keep = []

    # ------------------------------------------------- distribution level --
    canvas = ROOT.TCanvas("cc", "", 1800, 950)
    canvas.Divide(3, 2)
    tv_table = {r: [] for r in RANKS}
    for panel, k in enumerate(M_BINS):
        pad = canvas.cd(panel + 1)
        pad.SetLogy()
        sel = v[he][m[he] == k]
        counts, _ = np.histogram(sel, bins=edges)
        frac = len(sel) / len(he)
        density = counts / counts.sum() / np.diff(edges)
        h = ROOT.TH1D(f"h{k}", f";log H_{{T}}  (m = {k}, frac {frac:.4f});conditional density",
                      len(edges) - 1, edges[0], edges[-1])
        for i, c in enumerate(density):
            h.SetBinContent(i + 1, c)
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(3)
        h.SetMinimum(1e-6)
        h.Draw("hist")
        keep.append(h)
        legend = ROOT.TLegend(0.14, 0.16, 0.5, 0.45)
        legend.AddEntry(h, "data (held-out)", "l")
        centres = 0.5 * (edges[:-1] + edges[1:])
        for index, r in enumerate(RANKS):
            a = phi_k[k] @ models[r]          # coefficients of pi2 * sum a psi
            law = ref_g * (basis_g @ a)
            conditional = law / max(a[0], 1e-12)
            graph = ROOT.TGraph(len(grid))
            for i, (x, y) in enumerate(zip(grid, conditional)):
                graph.SetPoint(i, x, max(y, 1e-9))
            graph.SetLineColor(COLORS[index])
            graph.SetLineWidth(2)
            graph.Draw("l same")
            keep.append(graph)
            model_bins = np.interp(centres, grid, conditional) * np.diff(edges)
            tv = 0.5 * float(np.sum(np.abs(model_bins - counts / counts.sum())))
            tv_table[r].append(tv)
            legend.AddEntry(graph, f"rank {r}  (TV {tv:.4f})", "l")
        legend.SetTextSize(0.032)
        legend.Draw()
        keep.append(legend)
    canvas.SaveAs(str(OUT / "conditionals.png"))
    canvas.SaveAs(str(OUT / "conditionals.pdf"))

    for r in RANKS:
        print(f"rank {r}: conditional TV per bin "
              + " ".join(f"{t:.4f}" for t in tv_table[r]))

    # -------------------------------------------------- coefficient level --
    canvas2 = ROOT.TCanvas("cf", "", 1600, 950)
    canvas2.Divide(2, 2)
    for panel, n2 in enumerate((1, 2, 3, 4)):
        pad = canvas2.cd(panel + 1)
        frame = ROOT.TH1D(
            f"fr{n2}", f";m = nSelJet - 2;conditional coefficient  E[#psi_{{{n2}}} | m]",
            9, -0.5, 8.5)
        lo = min(-0.05, 1.3 * min((phi_k[k] @ models[3])[n2] / (phi_k[k] @ models[3])[0] for k in range(9)))
        hi = 1.3 * max((phi_k[k] @ models[3])[n2] / (phi_k[k] @ models[3])[0] for k in range(9))
        frame.SetMinimum(lo)
        frame.SetMaximum(max(hi, 0.1))
        frame.Draw()
        keep.append(frame)
        data_graph = ROOT.TGraphErrors(9)
        for k in range(9):
            sel = w[he][m[he] == k]
            if len(sel) < 50:
                continue
            values = np.asarray(Gamma.basis(sel, K2, gamma_params), dtype=float)[:, n2]
            data_graph.SetPoint(k, k, values.mean())
            data_graph.SetPointError(k, 0.0, values.std() / np.sqrt(len(values)))
        data_graph.SetMarkerStyle(20)
        data_graph.Draw("p same")
        keep.append(data_graph)
        legend = ROOT.TLegend(0.14, 0.6, 0.52, 0.88)
        legend.AddEntry(data_graph, "data (held-out)", "p")
        for index, r in enumerate(RANKS[:3]):
            graph = ROOT.TGraph(9)
            for k in range(9):
                a = phi_k[k] @ models[r]
                graph.SetPoint(k, k, a[n2] / max(a[0], 1e-12))
            graph.SetLineColor(COLORS[index])
            graph.SetLineWidth(2)
            graph.Draw("l same")
            keep.append(graph)
            legend.AddEntry(graph, f"rank {r}", "l")
        legend.SetTextSize(0.032)
        legend.Draw()
        keep.append(legend)
    canvas2.SaveAs(str(OUT / "conditional-coefficients.png"))
    canvas2.SaveAs(str(OUT / "conditional-coefficients.pdf"))

    # ------------------------------------------------------------ channels --
    canvas3 = ROOT.TCanvas("cs3", "", 1600, 500)
    canvas3.Divide(3, 1)
    pad = canvas3.cd(1)
    pad.SetLogy()
    spec = ROOT.TH1D("spec", ";channel;singular value of the cross residual",
                     len(singular), 0.5, len(singular) + 0.5)
    for i, s in enumerate(singular):
        spec.SetBinContent(i + 1, max(s, 1e-12))
    spec.SetLineWidth(3)
    spec.SetMinimum(1e-5)
    spec.Draw("hist")
    noise_line = ROOT.TLine(0.5, noise, len(singular) + 0.5, noise)
    noise_line.SetLineColor(ROOT.kRed)
    noise_line.SetLineStyle(2)
    noise_line.Draw()
    keep += [spec, noise_line]
    for pad_index, (vectors, title, count) in enumerate(
        ((u_vec.T, "site-1 channel u_c(n_1)", K1 + 1), (vt_vec, "site-2 channel w_c(n_2)", K2 + 1)),
        start=2,
    ):
        pad = canvas3.cd(pad_index)
        frame = ROOT.TH1D(f"fch{pad_index}", f";mode;{title}", count, -0.5, count - 0.5)
        frame.SetMinimum(-1.05)
        frame.SetMaximum(1.05)
        frame.Draw()
        keep.append(frame)
        legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
        for c in range(2):
            graph = ROOT.TGraph(count)
            for i in range(count):
                graph.SetPoint(i, i, vectors[c][i])
            graph.SetLineColor(COLORS[c])
            graph.SetMarkerColor(COLORS[c])
            graph.SetLineWidth(2)
            graph.SetMarkerStyle(20)
            graph.Draw("pl same")
            keep.append(graph)
            legend.AddEntry(graph, f"channel {c + 1}", "pl")
        legend.SetTextSize(0.035)
        legend.Draw()
        keep.append(legend)
    canvas3.SaveAs(str(OUT / "channels.png"))
    canvas3.SaveAs(str(OUT / "channels.pdf"))

    publish([OUT / "conditionals.png", OUT / "conditionals.pdf"], "pair/conditionals")
    publish(
        [OUT / "conditional-coefficients.png", OUT / "conditional-coefficients.pdf"],
        "pair/coefficients",
    )
    publish([OUT / "channels.png", OUT / "channels.pdf"], "pair/channels")
    publish([], "pair")


if __name__ == "__main__":
    main()
