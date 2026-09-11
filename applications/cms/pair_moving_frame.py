"""The moving NEF frame for (nSelJet, H_T): natural-parameter motion.

The fixed-frame staging spends its bond rank transporting the
conditional bulk (five channels, oscillatory conditionals).  Here the
site-2 reference itself moves along the Gamma family -- QVF-co-moving
scale, justified by the measured fact that the conditional pairs
(mu_m, sigma_m) lie on the Gamma QVF curve sigma = mu/sqrt(nu) to
within 7% -- and the residual tensor measures only shape beyond the
frame.  Coefficients remain projections, now in the fiberwise OPS.

Variants (frame parameters from the train half only, registered):

  F0      fixed frame: the three-moment shifted Gamma on log H_T
          (the previous study's frame; control);
  F2      fiberwise Gamma in linear H_T, shape nu_c =
          weighted mean of (mu_m/sigma_m)^2, mean field
          mu(m) = a + b m (weighted least squares: two numbers);
  F2theta F2 plus the kinematic support field theta(m) = 30 (m + 2),
          mean field on H_T - theta(m).

Staging as before: cross residual C (fiberwise-product part removed),
SVD channels, marginal-preserving models R^(r).

Predictions registered before running:

  MF-P1  the ladder collapses: s_1 drops by >= 5x under F2; under
         F2theta at most two significant channels remain and the
         discarded weight at rank 2 is below 1%;
  MF-P2  conditional TVs reach <= 0.02 in the populated bins
         (m <= 3) at rank <= 2, without oscillations;
  MF-P3  the surviving channels are the width drift (psi_2-dominated)
         and an edge remnant, the latter larger in F2 than F2theta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ROOT

from applications.cms.law_projections import load
from applications.cms.ht_frames import three_moment_gamma
from applications.cms.www import publish
from nefqvf import Gamma, GammaParams, NegativeBinomial, NegativeBinomialParams

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

OUT = Path(
    "/users/robert.schoefbeck/ML/nef-qvf-diffusion/artifacts/cms-ttbar/pair-moving"
)
K1, K2 = 6, 8
RANKS = (0, 1, 2, 3)
M_BINS = (0, 1, 2, 3, 4, 5)
M_MAX = 10
COLORS = (ROOT.kRed, ROOT.kOrange + 1, ROOT.kGreen + 2, ROOT.kBlue)


# ------------------------------------------------------------------- frames --
def bin_statistics(m, ht, minimum=2000):
    """Per-multiplicity means and variances on the train half."""

    stats = {}
    for k in range(M_MAX):
        sel = ht[m == k]
        if len(sel) < minimum:
            break
        stats[k] = (len(sel), sel.mean(), sel.var())
    return stats


def frame_f2(stats, kinematic: bool):
    """QVF-co-moving Gamma frame; optionally with the kinematic shift."""

    ks = np.array(sorted(stats))
    weights = np.array([stats[k][0] for k in ks], dtype=float)
    means = np.array([stats[k][1] for k in ks])
    variances = np.array([stats[k][2] for k in ks])
    theta = 30.0 * (ks + 2.0) if kinematic else np.zeros_like(ks, dtype=float)
    shifted = means - theta
    nu = float(np.average(shifted**2 / variances, weights=weights))
    design = np.column_stack([np.ones_like(ks, dtype=float), ks.astype(float)])
    w_matrix = np.diag(weights)
    coeffs = np.linalg.solve(
        design.T @ w_matrix @ design, design.T @ w_matrix @ shifted
    )
    a, b = float(coeffs[0]), float(coeffs[1])

    def mean_field(k):
        return a + b * k

    def theta_field(k):
        return 30.0 * (k + 2.0) if kinematic else 0.0

    label = "F2theta" if kinematic else "F2"
    print(f"{label}: nu_c {nu:.3f}  mean field {a:.1f} + {b:.1f} m")
    return nu, mean_field, theta_field


# -------------------------------------------------------------- projections --
def fiber_projections(m, ht, nu, mean_field, theta_field, nb_params):
    """R'_{n1 n2} = E[phi_{n1}(m) psi^{(m)}_{n2}(ht)], plus per-fiber data."""

    phi = np.asarray(
        NegativeBinomial.basis(np.arange(0.0, M_MAX), K1, nb_params), dtype=float
    )
    n = len(m)
    R = np.zeros((K1 + 1, K2 + 1))
    R2 = np.zeros_like(R)
    fiber = {}
    for k in range(M_MAX):
        sel = ht[m == k] - theta_field(k)
        if len(sel) == 0:
            continue
        params = GammaParams(mean=mean_field(k), r=nu)
        basis = np.asarray(Gamma.basis(sel, K2, params), dtype=float)
        mean_b = basis.mean(axis=0)
        mean_b2 = (basis**2).mean(axis=0)
        share = len(sel) / n
        R += share * np.outer(phi[k], mean_b)
        R2 += share * np.outer(phi[k] ** 2, mean_b2)
        fiber[k] = (params, share)
    errors = np.sqrt(np.maximum(R2 / n - R**2 / n, 0.0))
    return R, errors, phi, fiber


def staged_models(R):
    product = np.outer(R[:, 0], R[0, :]) / max(R[0, 0], 1e-12)
    residual = R - product
    u, s, vt = np.linalg.svd(residual)
    models = {r: product + (u[:, :r] * s[:r]) @ vt[:r] for r in RANKS}
    return models, s, u, vt


# ------------------------------------------------------------------- fixed --
def fixed_frame_spectrum(m, v, nb_params):
    """The F0 control: the previous fixed-frame ladder, recomputed."""

    theta, params = three_moment_gamma(v)
    w = v - theta
    phi_all = np.asarray(NegativeBinomial.basis(m, K1, nb_params), dtype=float)
    psi_all = np.asarray(Gamma.basis(w, K2, params), dtype=float)
    R = phi_all.T @ psi_all / len(m)
    _, s, _, _ = staged_models(R)
    return s


# -------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nj, ht = load()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(nj))
    half = len(nj) // 2
    tr, he = order[:half], order[half:]
    m, v = nj - 2, np.log(ht)

    mu, var = m[tr].mean(), m[tr].var()
    nb_params = NegativeBinomialParams(mean=mu, r=mu**2 / (var - mu))

    s_fixed = fixed_frame_spectrum(m[tr], v[tr], nb_params)
    print("F0 (fixed frame) channels:", " ".join(f"{x:.4f}" for x in s_fixed[:6]))

    stats = bin_statistics(m[tr], ht[tr])
    spectra = {"F0 fixed": s_fixed}
    variant_data = {}
    for kinematic in (False, True):
        name = "F2theta" if kinematic else "F2"
        nu, mean_field, theta_field = frame_f2(stats, kinematic)
        R, dR, phi, fiber = fiber_projections(
            m[tr], ht[tr], nu, mean_field, theta_field, nb_params
        )
        models, s, u_vec, vt_vec = staged_models(R)
        noise = float(np.median(dR)) * np.sqrt(max(R.shape))
        spectra[name] = s
        variant_data[name] = (nu, mean_field, theta_field, models, s)
        print(
            f"{name} channels: " + " ".join(f"{x:.4f}" for x in s[:6])
            + f"   noise ~ {noise:.5f}"
        )
        print(f"  {name} channel 1 site-2 vector: "
              + " ".join(f"{x:+.3f}" for x in vt_vec[0]))
        tail = [float(np.sum(s[r:] ** 2) / np.sum(s**2)) for r in RANKS]
        print("  discarded weight at rank r: "
              + " ".join(f"r={r}:{t:.4f}" for r, t in zip(RANKS, tail)))

    # ------------------------------------------------ conditional validation --
    keep = []
    for name in ("F2", "F2theta"):
        nu, mean_field, theta_field, models, s = variant_data[name]
        phi = np.asarray(
            NegativeBinomial.basis(np.arange(0.0, M_MAX), K1, nb_params), dtype=float
        )
        canvas = ROOT.TCanvas(f"c{name}", "", 1800, 950)
        canvas.Divide(3, 2)
        tv_table = {r: [] for r in RANKS}
        for panel, k in enumerate(M_BINS):
            pad = canvas.cd(panel + 1)
            pad.SetLogy()
            sel = ht[he][m[he] == k]
            edges = np.linspace(0.0, 1600.0, 81)
            counts, _ = np.histogram(sel, bins=edges)
            frac = len(sel) / len(he)
            density = counts / counts.sum() / np.diff(edges)
            h = ROOT.TH1D(
                f"h{name}{k}",
                f";H_{{T}} [GeV]  (m = {k}, frac {frac:.4f});conditional density"
                f"  --  {name}",
                len(edges) - 1, edges[0], edges[-1],
            )
            for i, c in enumerate(density):
                h.SetBinContent(i + 1, c)
            h.SetLineColor(ROOT.kBlack)
            h.SetLineWidth(3)
            h.SetMinimum(1e-8)
            h.Draw("hist")
            keep.append(h)
            legend = ROOT.TLegend(0.5, 0.6, 0.89, 0.89)
            legend.AddEntry(h, "data (held-out)", "l")
            grid = np.linspace(1.0, 1600.0, 1600)
            wg = grid - theta_field(k)
            inside = wg > 1e-6
            params = GammaParams(mean=mean_field(k), r=nu)
            basis_g = np.asarray(
                Gamma.basis(np.maximum(wg, 1e-6), K2, params), dtype=float
            )
            ref_g = np.asarray(
                Gamma.prob(np.maximum(wg, 1e-6), params), dtype=float
            )
            centres = 0.5 * (edges[:-1] + edges[1:])
            for index, r in enumerate(RANKS):
                a = phi[k] @ models[r]
                law = np.where(inside, ref_g * (basis_g @ a), 0.0)
                conditional = law / max(a[0], 1e-12)
                graph = ROOT.TGraph(len(grid))
                for i, (x, y) in enumerate(zip(grid, conditional)):
                    graph.SetPoint(i, x, max(y, 1e-12))
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
        canvas.SaveAs(str(OUT / f"conditionals-{name}.png"))
        canvas.SaveAs(str(OUT / f"conditionals-{name}.pdf"))
        for r in RANKS:
            print(f"{name} rank {r}: conditional TV "
                  + " ".join(f"{t:.4f}" for t in tv_table[r]))

    # ----------------------------------------------------------- spectra plot --
    canvas = ROOT.TCanvas("cs", "", 900, 650)
    canvas.SetLogy()
    frame = ROOT.TH1D("f", ";channel;singular value of the cross residual",
                      7, 0.5, 7.5)
    frame.SetMinimum(1e-4)
    frame.SetMaximum(2.0)
    frame.Draw()
    keep.append(frame)
    legend = ROOT.TLegend(0.55, 0.66, 0.89, 0.89)
    for index, (name, s) in enumerate(spectra.items()):
        graph = ROOT.TGraph(min(7, len(s)))
        for i in range(min(7, len(s))):
            graph.SetPoint(i, i + 1, max(s[i], 1e-12))
        graph.SetLineColor(COLORS[index % 4])
        graph.SetMarkerColor(COLORS[index % 4])
        graph.SetMarkerStyle(20)
        graph.SetLineWidth(2)
        graph.Draw("pl same")
        keep.append(graph)
        legend.AddEntry(graph, name, "pl")
    legend.SetTextSize(0.032)
    legend.Draw()
    keep.append(legend)
    canvas.SaveAs(str(OUT / "spectra.png"))
    canvas.SaveAs(str(OUT / "spectra.pdf"))

    publish([OUT / "spectra.png", OUT / "spectra.pdf"], "pair-moving/spectra")
    publish(
        [OUT / "conditionals-F2.png", OUT / "conditionals-F2.pdf"],
        "pair-moving/conditionals-F2",
    )
    publish(
        [OUT / "conditionals-F2theta.png", OUT / "conditionals-F2theta.pdf"],
        "pair-moving/conditionals-F2theta",
    )
    publish([], "pair-moving")


if __name__ == "__main__":
    main()
