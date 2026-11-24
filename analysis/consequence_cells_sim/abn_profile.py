"""Abnormal-profile overlap test on the cells-only refits (experimental, 2026-11-16).

Same statistic as the real-data check: per bootstrap, compose the P/LP PTV and P/LP missense cells (cell-size weighted),
take the normal component as the one with the largest composed B/LB weight, drop it, renormalise, and take the
total-variation distance of the two abnormal profiles. Overlap = 95th-percentile bound < 0.2.
The true TV comes from generate.truth (pathogenic class shares without the normal class).

    python -m analysis.consequence_cells_sim.abn_profile <cellfits_dir> [--results <grid results.csv> ...]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ABN = 0.05


def _sel(names, sample, csq):
    pos = "PBGS".index(sample)
    return np.array([n.split("|")[0][pos] != "-" and (csq is None or n.split("|")[1] == csq) for n in names])


def _compose(W, sel, n):
    return (n[sel][:, None] * W[sel]).sum(0) / n[sel].sum()


def true_tv(c):
    from analysis.consequence_cells_sim import generate as G
    p, m = G.truth(c, "PTV")["P"], G.truth(c, "missense")["P"]
    ab = [k for k in G.CLASSES if k != "nor"]
    pa = np.array([p[k] for k in ab]); ma = np.array([m[k] for k in ab])
    return float(0.5 * np.abs(pa / pa.sum() - ma / ma.sum()).sum()), float(m["nor"] - p["nor"])


def one(f):
    z = np.load(f)
    W, names, n = z["weights"], [str(x) for x in z["names"]], z["sizes"].astype(float)
    sp, sm, sb = _sel(names, "P", "PTV"), _sel(names, "P", "missense"), _sel(names, "B", None)
    tv, dn = [], []
    for w in W:
        b, p, m = _compose(w, sb, n), _compose(w, sp, n), _compose(w, sm, n)
        j = int(np.argmax(b))
        keep = np.arange(w.shape[1]) != j
        dn.append(m[j] - p[j])
        if min(1 - p[j], 1 - m[j]) < MIN_ABN:
            tv.append(np.nan)
            continue
        tv.append(0.5 * np.abs(p[keep] / p[keep].sum() - m[keep] / m[keep].sum()).sum())
    tv = np.asarray(tv)
    c = json.loads(str(z["cell"]))
    t_tv, t_dn = true_tv(c)
    q = lambda a, x: float(np.nanquantile(a, x)) if np.isfinite(a).any() else np.nan  # noqa: E731
    return dict(cell_id=c["cell_id"], rep=int(z["rep"]), assay=c["assay"], size=c["size"], kappa=c["kappa"],
                ptv_loc=c.get("ptv_loc", 1.0), hypomorph=bool(c.get("hypomorph", False)), n_boot=len(W),
                n_boot_tv=int(np.isfinite(tv).sum()), n_plp_ptv=int(n[sp].sum()), n_plp_mis=int(n[sm].sum()),
                true_tv=t_tv, tv_med=q(tv, 0.5), tv_lo=q(tv, 0.05), tv_up=q(tv, 0.95),
                true_dnorm=t_dn, dnorm_med=q(np.array(dn), 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cellfits_dir")
    ap.add_argument("--results", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.cellfits_dir)
    df = pd.DataFrame([one(f) for f in sorted(glob.glob(str(out / "fits" / "*.npz")))])
    df["overlap"] = df.tv_up < 0.2
    if a.results:
        res = pd.concat([pd.read_csv(r, low_memory=False) for r in a.results])
        cols = [c for c in res.columns if c.startswith(("pooled__missense_", "cell_all__missense_", "cell_mis__missense_",
                                                         "true_cell_mis__missense_"))
                and c.rsplit("missense_", 1)[1] in ("post_mae", "post_over", "path_benign", "ben_pathev")]
        df = df.merge(res[["cell_id", "rep"] + cols], on=["cell_id", "rep"], how="left")
    df.to_csv(out / "abn_profile.csv", index=False)
    g = df.groupby(["cell_id"]).agg(n=("rep", "size"), true_tv=("true_tv", "first"), tv_med=("tv_med", "median"),
                                   tv_up=("tv_up", "median"), overlap=("overlap", "mean"),
                                   n_ptv=("n_plp_ptv", "median"), n_mis=("n_plp_mis", "median"))
    g.to_csv(out / "abn_profile_by_cell.csv")
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
