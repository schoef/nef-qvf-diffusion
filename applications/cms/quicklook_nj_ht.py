"""Quick look at (nSelJet, ht) in dileptonic ttbar: supports, dispersion,
and the kinematic coupling of the two supports.  PyROOT plots."""

import math

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

DATA = "/users/robert.schoefbeck/ML/quicklook/nj_ht_w.npy"
OUT = "/users/robert.schoefbeck/ML/nef-qvf-diffusion/artifacts/cms-ttbar"


def negative_binomial(k, mu, fano):
    p = 1.0 / fano
    r = mu * p / (1.0 - p)
    return (
        math.exp(
            math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(p) + k * math.log(1.0 - p)
        )
    )


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    arr = np.load(DATA)
    nj, ht = arr[:, 0], arr[:, 1]
    m = nj - 2
    mu, fano = m.mean(), m.var() / m.mean()

    canvas = ROOT.TCanvas("c", "", 1500, 500)
    canvas.Divide(3, 1)

    # panel 1: multiplicity vs Poisson and negative binomial
    canvas.cd(1).SetLogy()
    h_nj = ROOT.TH1D("h_nj", ";m = nSelJet - 2;probability", 10, -0.5, 9.5)
    for k in range(10):
        h_nj.SetBinContent(k + 1, float(np.mean(m == k)))
    h_nj.SetLineColor(ROOT.kBlack)
    h_nj.SetLineWidth(2)
    h_nj.SetMinimum(1e-7)
    h_nj.Draw("hist")
    h_pois = h_nj.Clone("h_pois")
    h_nb = h_nj.Clone("h_nb")
    for k in range(10):
        h_pois.SetBinContent(k + 1, math.exp(-mu) * mu**k / math.factorial(k))
        h_nb.SetBinContent(k + 1, negative_binomial(k, mu, fano))
    h_pois.SetLineColor(ROOT.kRed)
    h_nb.SetLineColor(ROOT.kBlue)
    h_pois.Draw("hist same")
    h_nb.Draw("hist same")
    leg1 = ROOT.TLegend(0.45, 0.68, 0.88, 0.88)
    leg1.AddEntry(h_nj, "data (Fano %.2f)" % fano, "l")
    leg1.AddEntry(h_pois, "Poisson(%.2f)" % mu, "l")
    leg1.AddEntry(h_nb, "neg. binomial (moment)", "l")
    leg1.Draw()

    # panel 2: ht with moment-matched shifted Gamma
    canvas.cd(2).SetLogy()
    h_ht = ROOT.TH1D("h_ht", ";H_{T} [GeV];density", 100, 0.0, 1500.0)
    for v in ht:
        h_ht.Fill(v)
    h_ht.Scale(1.0 / (h_ht.Integral("width")))
    h_ht.SetLineColor(ROOT.kBlack)
    h_ht.SetLineWidth(2)
    h_ht.SetMinimum(1e-8)
    h_ht.Draw("hist")
    s = ht - 60.0
    shape = s.mean() ** 2 / s.var()
    scale = s.var() / s.mean()
    gamma_fn = ROOT.TF1(
        "g", "(x>[2])*pow((x-[2])/[1],[0]-1)*exp(-(x-[2])/[1])/([1]*TMath::Gamma([0]))",
        0.0, 1500.0)
    gamma_fn.SetParameters(shape, scale, 60.0)
    gamma_fn.SetNpx(600)
    gamma_fn.SetLineColor(ROOT.kBlue)
    gamma_fn.Draw("same")
    leg2 = ROOT.TLegend(0.4, 0.72, 0.88, 0.88)
    leg2.AddEntry(h_ht, "data", "l")
    leg2.AddEntry(gamma_fn, "Gamma(#nu=%.2f) on H_{T}-60" % shape, "l")
    leg2.Draw()

    # panel 3: the support coupling
    canvas.cd(3)
    h2 = ROOT.TH2D("h2", ";nSelJet;H_{T} [GeV]", 8, 1.5, 9.5, 60, 0.0, 1200.0)
    for a, b in zip(nj, ht):
        h2.Fill(a, b)
    h2.Draw("colz")
    ROOT.gPad.SetLogz()
    line = ROOT.TF1("kin", "30.0*x", 2.0, 9.5)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(2)
    line.Draw("same")
    prof = h2.ProfileX()
    prof.SetLineColor(ROOT.kBlack)
    prof.SetLineWidth(2)
    prof.Draw("same")

    canvas.SaveAs(OUT + "/quicklook-nj-ht.pdf")
    canvas.SaveAs(OUT + "/quicklook-nj-ht.png")
    print("saved", OUT + "/quicklook-nj-ht.png")


if __name__ == "__main__":
    main()
