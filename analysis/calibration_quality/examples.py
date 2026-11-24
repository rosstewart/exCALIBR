"""Example fits for the misfit / Yang-distance figure: one replicate each of a well-fit, an overfit (few controls) and
an underfit (misspecified truth) setting. For each sample: the observed scores, the true density, the fitted mixture
density (median over bootstrap fits) and the in-sample, held-out and fitted-vs-true distances.

    python -m analysis.calibration_quality.examples <out.json> [--B 50 --R 8]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.stats as sps

from analysis.calibration_quality import generate as G, run as Rn

EXAMPLES = {
    "well fit": dict(experiment="core", n_plp=40, n_blb=40, richness="mid", kappa=4.0, gamma=1.0, pi=0.25),
    "overfit (5 controls per class)": dict(experiment="core", n_plp=5, n_blb=5, richness="mid", kappa=2.5,
                                           gamma=1.0, pi=0.25),
    "underfit (two abnormal modes)": dict(experiment="misfit", n_plp=160, n_blb=160, richness="mid", kappa=4.0,
                                          gamma=1.0, pi=0.25, shape="bimodal", shape_par=2.0),
}


def true_density(x, c, sample):
    norm, abn = G._pdf_normal(x, c), G._pdf_abnormal(x, c)
    path = c["gamma"] * abn + (1 - c["gamma"]) * norm
    if sample == 0:
        lab = c["lab_gamma"] * abn + (1 - c["lab_gamma"]) * norm
        return (1 - c["eps_p"]) * lab + c["eps_p"] * norm
    if sample == 1:
        return c["eta_b"] * path + (1 - c["eta_b"]) * norm
    if sample == 2:
        return c["pi"] * path + (1 - c["pi"]) * norm
    return norm


def one(name, kw, B, R, rep=0):
    exp = kw.pop("experiment")
    c = G._cell(exp, **kw)
    d = G.make_data(G.rng_for(c, rep), c)
    ss, cols = Rn.build_scoreset(d["samples"])
    fits, seeds = Rn.fit_arm(ss, B, R)
    splits = {int(b): Rn.bootstrap_splits(ss, b) for b in seeds}
    x = np.linspace(min(-9, ss.scores.min()), max(4, ss.scores.max()), 400)
    out = dict(name=name, cell_id=c["cell_id"], x=x.tolist(), samples={})
    yt = Rn.yang_per_sample(fits, seeds, splits, cols, "train")
    yv = Rn.yang_per_sample(fits, seeds, splits, cols, "val")
    ytr = Rn.yang_true_per_sample(fits, seeds, splits, cols, c, rep)
    for j, i in enumerate(cols):
        dens = np.array([sum(w * sps.skewnorm.pdf(x, a, loc, sc) for (a, loc, sc), w in
                             zip(f["fit"]["component_params"], f["fit"]["weights"][j])) for f in fits])
        nm = G.NAMES[i]
        out["samples"][nm] = dict(scores=np.asarray(d["samples"][i]).tolist(), true=true_density(x, c, i).tolist(),
                                  fit=np.median(dens, 0).tolist(), fit_lo=np.percentile(dens, 5, 0).tolist(),
                                  fit_hi=np.percentile(dens, 95, 0).tolist(), yang_train=yt[nm], yang_val=yv[nm],
                                  yang_true=ytr[nm])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--R", type=int, default=8)
    a = ap.parse_args()
    res = [one(n, dict(kw), a.B, a.R) for n, kw in EXAMPLES.items()]
    with open(a.out, "w") as fh:
        json.dump(res, fh)
    for r in res:
        print(r["name"], {k: (round(v["yang_train"], 3), round(v["yang_val"], 3) if v["yang_val"] == v["yang_val"]
                              else None, round(v["yang_true"], 3)) for k, v in r["samples"].items()})


if __name__ == "__main__":
    main()
