"""Pathomechanism-based PTV-vs-missense comparison on cell-model fits (experimental, 2026-11-19).

For each bootstrap fit and each consequence c in {PTV, missense}, the P/LP density of c is modelled as in Supplementary
Section 1.9 (pathomechanism-aware prior and likelihood ratio):

    f_P,c(s) = gamma_c f_D,c(s) + (1 - gamma_c) f_B(s)

f_P,c is the cell-size-weighted composition of the P/LP cells of consequence c; f_B is the benign reference (B/LB and
synonymous compositions blended per benign_method, as in production). gamma_c (P(M=1|Y=1)) comes from production's
boundary estimator (point_ranges._pathomechanism_boundary_match_frac, evaluated at the observed P/LP scores of c) and
f_D,c from the same unmixing (point_ranges._unmix_density). Reported per pair:

    d_gamma  = gamma_missense - gamma_PTV  (share of P/LP whose mechanism the assay detects; drives benign evidence)
    tv_D     = 1/2 int |f_D,PTV - f_D,missense| ds  (detected pathogenic densities; drives pathogenic evidence)

with bootstrap medians and 5th/95th percentiles. Calls on tv_D (margin 0.2): overlap if q95 < 0.2, different if
q05 > 0.2, otherwise inconclusive.

    python -m analysis.consequence_cells_sim.mech_overlap real [--workers 16]
    python -m analysis.consequence_cells_sim.mech_overlap sim <cellfits_dir> [--workers 16]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration")
LARGE = ROOT / "missense_only_calibrations/cells/hpc_large_labeled/labeled"
OUT_REAL = ROOT / "consequence_cells_sim/real_overlap"
MARGIN = 0.2
GRID_N = 2001
REAL_PTV = ("stop_gained", "frameshift_variant")


# ── core ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _mix(x, params, w):
    from src.assay_calibration.fit_utils.cfusn.density_utils import joint_densities
    return joint_densities(np.asarray(x, float), params, np.asarray(w, float)).sum(axis=0)


def _compose(W, sel, n):
    return (n[sel][:, None] * W[sel]).sum(0) / n[sel].sum()


def benign_reference(W, n, sel_blb, sel_syn, benign_method):
    from src.assay_calibration.fit_utils.point_ranges import _get_benign_reference_weights
    rows, bi, si = [], None, None
    if sel_blb.any():
        bi = len(rows); rows.append(_compose(W, sel_blb, n))
    if sel_syn.any():
        si = len(rows); rows.append(_compose(W, sel_syn, n))
    return np.asarray(_get_benign_reference_weights(rows, benign_method, bi, si), float)


def detected(params, w_P, w_B, labeled_scores, grid):
    """(gamma, f_D on grid normalised to 1) for one consequence; f_D None when gamma is below production's floor."""
    from src.assay_calibration.fit_utils.point_ranges import (_PATHOMECHANISM_MATCH_FRAC_FLOOR,
                                                              _pathomechanism_boundary_match_frac, _unmix_density)
    g = _pathomechanism_boundary_match_frac(_mix(labeled_scores, params, w_B), _mix(labeled_scores, params, w_P))
    if not (np.isfinite(g) and 0 < g <= 1) or g < _PATHOMECHANISM_MATCH_FRAC_FLOOR:
        return g, None
    f = _unmix_density(_mix(grid, params, w_P), 1 - g, _mix(grid, params, w_B))
    f = np.clip(f, 0, None)
    z = np.trapz(f, grid)
    return g, (f / z if z > 0 else None)


def one_fit(params, W, n, sel, scores, grid, benign_method):
    """sel: dict ptv/mis/blb/syn -> cell masks; scores: dict ptv/mis -> observed P/LP scores."""
    w_B = benign_reference(W, n, sel["blb"], sel["syn"], benign_method)
    out = {}
    for c in ("ptv", "mis"):
        out[f"gamma_{c}"], out[f"fD_{c}"] = detected(params, _compose(W, sel[c], n), w_B, scores[c], grid)
    a, b = out["fD_ptv"], out["fD_mis"]
    out["tv_D"] = 0.5 * np.trapz(np.abs(a - b), grid) if (a is not None and b is not None) else np.nan
    out["d_gamma"] = out["gamma_mis"] - out["gamma_ptv"]
    return out


def summarise(fits):
    q = lambda a, x: float(np.nanquantile(a, x)) if np.isfinite(a).any() else np.nan  # noqa: E731
    r = {}
    for k in ("gamma_ptv", "gamma_mis", "d_gamma", "tv_D"):
        a = np.array([f[k] for f in fits], float)
        r.update({f"{k}_med": q(a, 0.5), f"{k}_q05": q(a, 0.05), f"{k}_q95": q(a, 0.95)})
    r["n_boot"] = len(fits)
    r["n_boot_tv"] = int(np.isfinite([f["tv_D"] for f in fits]).sum())
    r["call"] = ("overlap" if r["tv_D_q95"] < MARGIN else "different" if r["tv_D_q05"] > MARGIN
                 else "inconclusive" if np.isfinite(r["tv_D_med"]) else "untestable")
    return r


# ── real data ────────────────────────────────────────────────────────────────────────────────────────────────────

def _real_k(ds):
    cfg = json.load(open("src/igvf_configs/dataset_configs_aug_2026.json"))
    e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
    kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"     # same K rule as membership_cells.metrics
    return kc if kc in ("2c", "3c") else "3c"


def real_one(ds, min_n=10):
    from analysis.membership_cells.cells import Cells
    c = Cells.load(LARGE / "cells" / f"{ds}.npz")
    t = c.table
    n = t["n"].to_numpy(float)
    m = np.array([list(x) for x in t["mask"]])
    csq = t["consequence"].to_numpy()
    sel = dict(ptv=m[:, 0] & np.isin(csq, REAL_PTV), mis=m[:, 0] & (csq == "missense_variant"),
               blb=m[:, 1], syn=m[:, 3])
    rows = np.array([list(x) for x in t["mask"]])[c.cell_of_row]
    rc = csq[c.cell_of_row]
    scores = dict(ptv=c.scores[rows[:, 0] & np.isin(rc, REAL_PTV)], mis=c.scores[rows[:, 0] & (rc == "missense_variant")])
    K = _real_k(ds)
    row = dict(dataset=ds, K=K, n_plp_ptv=len(scores["ptv"]), n_plp_mis=len(scores["mis"]), n_blb=int(n[sel["blb"]].sum()))
    if min(row["n_plp_ptv"], row["n_plp_mis"]) < min_n or not (sel["blb"].any() or sel["syn"].any()):
        return row | dict(call="untestable")
    bm = json.load(open(LARGE / f"config_{K}.json")).get(ds, {}).get("benign_method", "avg")
    fits = []
    for f in sorted(glob.glob(str(LARGE / "raw" / f"{ds}__{K}__*.json.gz"))):
        for rec in json.load(gzip.open(f)).values():
            if not rec or "error" in rec:
                continue
            fit = rec[K]["fit"]
            lo, hi = fit["xlims"]
            fits.append(one_fit(fit["component_params"], np.asarray(fit["weights"], float), n, sel, scores,
                                np.linspace(lo, hi, GRID_N), bm))
    return row | summarise(fits) | dict(benign_method=bm)


def run_real(workers):
    from joblib import Parallel, delayed
    dss = sorted(Path(p).stem for p in glob.glob(str(LARGE / "cells" / "*.npz")))
    df = pd.DataFrame(Parallel(n_jobs=workers)(delayed(real_one)(d) for d in dss))
    o = pd.read_csv(OUT_REAL / "rel_yang_real_with_outcomes.csv")
    keep = ["dataset_internal", "assay_class"] + [f"{a}_{b}" for a in ("fn", "fp", "tp", "mcc")
                                                  for b in ("std_all", "std_mis", "cell_all", "cell_mis")]
    df = df.merge(o[[k for k in keep if k in o]], left_on="dataset", right_on="dataset_internal", how="left")
    df.drop(columns="dataset_internal").to_csv(OUT_REAL / "mech_overlap_real.csv", index=False)
    return df


def cell_arm_calibration(ds, plp="ptv", manual_prior=None, n_jobs=8):
    """Production calibration (process_component_fits, lp_anchored) of a cell-model arm composed from the stored large
    labeled cell fits: P/LP = the P/LP cells of `plp` ("ptv" or "all"), B/LB, gnomAD and Synonymous as in cell_all.
    manual_prior fixes the prior (e.g. the stored cell_all prior). With plp="all" and no manual prior this reproduces
    the stored cell_all calibration (checked on LDLR abundance: same prior, point boundaries within 0.001)."""
    import logging
    from analysis.consequence_cells_sim.run import compose_fits, membership, scoreset
    from analysis.membership_cells.cells import Cells
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    K = _real_k(ds)
    c = Cells.load(LARGE / "cells" / f"{ds}.npz")
    t = c.table
    m = np.array([list(x) for x in t["mask"]])[c.cell_of_row].astype(bool)
    if plp == "ptv":
        m[:, 0] &= np.isin(t["consequence"].to_numpy()[c.cell_of_row], REAL_PTV)
    ss, keep = scoreset(c.scores, m)
    M = membership(m, keep, c.cell_of_row, len(t))
    fits, seeds = [], []
    for f in sorted(glob.glob(str(LARGE / "raw" / f"{ds}__{K}__*.json.gz"))):
        for rec in json.load(gzip.open(f)).values():
            if rec and "error" not in rec:
                fits.append({"fit": rec[K]["fit"]}); seeds.append(int(rec[K]["bootstrap_seed"]))
    bm = json.load(open(LARGE / f"config_{K}.json")).get(ds, {}).get("benign_method", "avg")
    cfg = PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp", components=[int(K[0])], benign_method=bm,
                         use_median_prior=True, use_2c_equation=False, point_values=list(range(1, 9)), n_jobs=n_jobs,
                         acmg_mapping_method="lp_anchored", manual_prior=manual_prior)
    lg = logging.getLogger("mech_overlap"); lg.setLevel(logging.ERROR)
    return process_component_fits(fits=compose_fits(fits, M), scoreset=ss, config=cfg, n_c=int(K[0]), logger=lg,
                                  bootstrap_seeds=np.array(seeds))


# ── simulation ───────────────────────────────────────────────────────────────────────────────────────────────────

def sim_truth(c, grid):
    """True gamma_c and detected-density TV from generate.truth (score classes are Gaussian, sd 1)."""
    from scipy.stats import norm
    from analysis.consequence_cells_sim import generate as G
    k = c["kappa"]
    loc = dict(ptv=-k * c.get("ptv_loc", 1.0), abn=-k, int=-k / 2)
    out, f = {}, {}
    for csq, key in (("PTV", "ptv"), ("missense", "mis")):
        P = G.truth(c, csq)["P"]
        out[f"true_gamma_{key}"] = 1 - P["nor"]
        d = sum(P[j] * norm.pdf(grid, loc[j], 1) for j in loc)
        f[key] = d / np.trapz(d, grid)
    out["true_d_gamma"] = out["true_gamma_mis"] - out["true_gamma_ptv"]
    out["true_tv_D"] = 0.5 * np.trapz(np.abs(f["ptv"] - f["mis"]), grid)
    return out


def sim_one(f):
    from analysis.consequence_cells_sim import generate as G
    z = np.load(f)
    c = json.loads(str(z["cell"]))
    rep = int(z["rep"])
    names = [str(x) for x in z["names"]]
    n = z["sizes"].astype(float)
    pos = lambda s: "PBGS".index(s)  # noqa: E731
    has = lambda nm, s: nm.split("|")[0][pos(s)] != "-"  # noqa: E731
    csq = np.array([nm.split("|")[1] for nm in names])
    sel = dict(ptv=np.array([has(x, "P") for x in names]) & (csq == "PTV"),
               mis=np.array([has(x, "P") for x in names]) & (csq == "missense"),
               blb=np.array([has(x, "B") for x in names]), syn=np.array([has(x, "S") for x in names]))
    d = G.make_data(G.rng_for(c, rep), c)
    m, dc, sc = d["mask"], d["csq"], d["score"]
    scores = dict(ptv=sc[m[:, 0] & (dc == "PTV")], mis=sc[m[:, 0] & (dc == "missense")])
    grid = np.linspace(sc.min() - 0.5, sc.max() + 0.5, GRID_N)
    fits = [one_fit(z["params"][b], z["weights"][b], n, sel, scores, grid, "avg") for b in range(len(z["weights"]))]
    tgrid = np.linspace(-c["kappa"] * max(c.get("ptv_loc", 1.0), 1.0) - 6, 6, GRID_N)
    return dict(cell_id=c["cell_id"], rep=rep, experiment=c["experiment"], assay=c["assay"], size=c["size"],
                kappa=c["kappa"], ptv_loc=c.get("ptv_loc", 1.0), hypomorph=bool(c.get("hypomorph", False)),
                n_plp_ptv=len(scores["ptv"]), n_plp_mis=len(scores["mis"])) | sim_truth(c, tgrid) | summarise(fits)


def run_sim(cellfits_dir, workers):
    from joblib import Parallel, delayed
    d = Path(cellfits_dir)
    df = pd.DataFrame(Parallel(n_jobs=workers)(delayed(sim_one)(f) for f in sorted(glob.glob(str(d / "fits" / "*.npz")))))
    df.to_csv(d / "mech_overlap.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real"); r.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("sim"); s.add_argument("cellfits_dir"); s.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    if a.cmd == "real":
        df = run_real(a.workers)
        cols = ["dataset", "K", "n_plp_ptv", "n_plp_mis", "gamma_ptv_med", "gamma_mis_med", "d_gamma_med",
                "tv_D_med", "tv_D_q05", "tv_D_q95", "call", "fn_std_all", "fn_std_mis"]
        print(df[df.call != "untestable"][cols].round(3).to_string(index=False))
    else:
        df = run_sim(a.cellfits_dir, a.workers)
        g = df.groupby("cell_id").agg(true_tv=("true_tv_D", "first"), tv_med=("tv_D_med", "median"),
                                      tv_q95=("tv_D_q95", "median"), true_dg=("true_d_gamma", "first"),
                                      dg_med=("d_gamma_med", "median"),
                                      overlap=("call", lambda x: (x == "overlap").mean()),
                                      different=("call", lambda x: (x == "different").mean()))
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
