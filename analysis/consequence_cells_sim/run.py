"""Consequence-cell simulation: one replicate of one design cell, all arms.

Fitted arms (production fitter; skew locked to Gaussian except in the misfit cell; K = the true number):
    pooled      4-sample fit, all controls, whole population                  (real data: std_all)
    mis         4-sample refit: missense P/LP, all B/LB, missense population  (real data: std_mis / medium_plpmis)
    cells       one fit with a column per (membership mask x consequence [x SpliceAI>0.2 in fitness assays]) cell
Calibrations (production process_component_fits, benign_method avg, lp_anchored points):
    pooled, mis                          their own fits
    cell_all, cell_mis[, cell_mis_nosplsyn] composed from the cell fit         (real data: large cell runs)
    cell_<c>       consistent per-consequence pair: P/LP and population from consequence c's cells
    nu_<c>         fallback (a): population from c's cells, no P/LP (NU unmixing: prior and LR jointly)
    poolP_<c>      fallback (b): all P/LP with c's population (mismatched pair)
    true_<...>     true components, per-bootstrap per-cell weight EM, same compositions (cell_mis and cell_<c>)
Diagnostic: mm_<c>_ownprior_poolLR (pi-hat from cell_<c> with the pooled LR) and mm_<c>_poolprior_ownLR.
Truth: exact pi_c and LR_c of consequence c's kept population (generate.truth).

Metrics per (arm, evaluation consequence) on 10k held-out population draws of that consequence: prior error vs pi_c,
|log LR| error of the median curve (all draws and |true log LR| > log 2), posterior error and overstatement with the
conservative reported LR, reliability counts, points over / wrong vs truth points.

Pilot: python -m analysis.consequence_cells_sim.run --pilot [--reps 2 --B 50 --R 8 --n-jobs 12]
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd

from analysis.calibration_quality.run import (conservative_lr, lr_bounds, points_of, posterior, posterior_histograms,
                                              posterior_metrics, prior_diagnostics, truth_points)
from analysis.consequence_cells_sim import generate as G

warnings.filterwarnings("ignore")
POINTS = [1, 2, 3, 4, 5, 6, 7, 8]
MAPPING = "lp_anchored"     # the real-data cell and missense arms use lp_anchored
B_DEFAULT, R_DEFAULT = 50, 8
PLP, BLB, POP, SYN = 0, 1, 2, 3


# ── layouts ──────────────────────────────────────────────────────────────────────────────────────────────────────

def cell_keys(d, assay):
    tag = np.where(d["hi"] & (assay == "fitness"), "|SpliceAI>0.2", "")
    keys = ["".join("PBGS"[i] if m[i] else "-" for i in range(4)) + "|" + c + t
            for m, c, t in zip(d["mask"], d["csq"], tag)]
    names, cell_of_row = np.unique(keys, return_inverse=True)
    return list(names), cell_of_row


def layout(d, kind, c=None):
    """Membership mask of the calibration (or refit) scoreset for one arm; rows with an empty mask drop out."""
    m = d["mask"].copy()
    if kind in ("mis", "cell_mis", "cell_mis_nosplsyn"):
        m[d["csq"] != "missense", PLP] = False
        m[d["csq"] != "missense", POP] = False
        if kind == "cell_mis_nosplsyn":
            m[(d["csq"] == "synonymous") & d["hi"], SYN] = False
    elif kind in ("cell_c", "nu_c", "poolP_c"):
        m[d["csq"] != c, POP] = False
        if kind == "cell_c":
            m[d["csq"] != c, PLP] = False
        elif kind == "nu_c":
            m[:, PLP] = False
    return m


def scoreset(scores, mask):
    """All four standard columns kept (empty ones included), as production builds them."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    keep = mask.any(1)
    return BasicScoreset(scores[keep], mask[keep], sample_names=list(G.NAMES)), keep


def membership(mask, keep, cell_of_row, n_cells):
    """Rows: non-empty standard columns in order; entries: member counts per cell."""
    cols = [s for s in range(4) if mask[keep, s].any()]
    M = np.zeros((len(cols), n_cells))
    for r, s in enumerate(cols):
        np.add.at(M[r], cell_of_row[keep & mask[:, s]], 1.0)
    return M


def compose_fits(cell_fits, M):
    out = []
    for f in cell_fits:
        W = np.asarray(f["fit"]["weights"], float)
        out.append({"fit": {"component_params": f["fit"]["component_params"], "weights": (M @ W) / M.sum(1, keepdims=True),
                            "xlims": f["fit"]["xlims"]}})
    return out


# ── fits and calibration ─────────────────────────────────────────────────────────────────────────────────────────

def fit_arm(ss, K, B, R, gaussian):
    from src.assay_calibration.fit_utils.fit import Fit
    fitter = Fit(ss)
    fits, seeds = [], []
    for b in range(B):
        try:
            models, best, _ = fitter.run([K], bootstrap=True, bootstrap_seed=b, num_fits=R, master_seed=0,
                                         check_monotonic=True, force_gaussian=gaussian, core_limit=1, verbose=False)
            if best is None:
                continue
            fits.append({"fit": models[best]}); seeds.append(b)
        except Exception:
            continue
    return fits, np.array(seeds, int)


def true_cell_fits(ss_cells, c, seeds, xlims, gaussian):
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    from src.assay_calibration.fit_utils.fit import Fit
    params = G.true_component_params(c)
    out = []
    for b, xl in zip(seeds, xlims):
        job = Fit(ss_cells).generate_fit_jobs([G.n_components(c)], bootstrap=True, bootstrap_seed=int(b), num_fits=1,
                                              master_seed=0, check_monotonic=True, force_gaussian=gaussian)[0]
        obs, sa = job["train_observations"], job["train_sample_assignments"]
        W = np.vstack([_em_weights(obs[sa[:, j]], params, [1.0] * len(params)) for j in range(sa.shape[1])])
        out.append({"fit": {"component_params": params, "weights": W, "xlims": xl}})
    return out


def calibrate(ss, fits, seeds, K):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    cfg = PipelineConfig(dataset_csv="", dataset_name="sim", output_dir="/tmp", components=[K], benign_method="avg",
                         use_median_prior=True, use_2c_equation=False, point_values=POINTS, n_jobs=1,
                         acmg_mapping_method=MAPPING)
    lg = logging.getLogger("csqcells"); lg.setLevel(logging.ERROR)
    try:
        return process_component_fits(fits=fits, scoreset=ss, config=cfg, n_c=K, logger=lg, bootstrap_seeds=seeds)
    except Exception:
        return None


# ── evaluation ───────────────────────────────────────────────────────────────────────────────────────────────────

def evaluate(row, arm, csq, prior, l5, l50, l95, ev, c, cal=None):
    t = G.truth(c, csq)
    x, is_path = ev[csq]["x"], ev[csq]["is_path"]
    tl = G.true_log_lr(x, c, csq)
    p = f"{arm}__{csq}"
    row[f"{p}_prior"] = prior
    row[f"{p}_prior_err"] = prior - t["pi"]
    err = np.abs(l50 - tl)
    row[f"{p}_llr_mae"] = float(np.nanmean(err))
    tail = np.abs(tl) > np.log(2)
    row[f"{p}_llr_mae_tail"] = float(np.nanmean(err[tail])) if tail.any() else np.nan
    post_true = posterior(np.exp(tl), t["pi"])
    lr_c = conservative_lr(l5, l95)
    post = posterior(lr_c, prior)
    for k, v in posterior_metrics(post, post_true, lr_c, is_path).items():
        row[f"{p}_{k}"] = v
    for k, v in posterior_histograms(post, None, is_path).items():
        row[f"{p}_{k}"] = v
    if cal is not None:
        est = points_of(cal, x)
        tru = truth_points(tl, t["pi"], MAPPING)
        row[f"{p}_pts_over"] = float(((est > np.maximum(tru, 0)) | (est < np.minimum(tru, 0))).mean())
        row[f"{p}_pts_wrong"] = float((((est > 0) & (tru < 0)) | ((est < 0) & (tru > 0))).mean())
        row[f"{p}_pts_det"] = float((est != 0).mean())
        row[f"{p}_truth_pts_det"] = float((tru != 0).mean())


def run_cell(cell, rep, B=B_DEFAULT, R=R_DEFAULT):
    t0 = time.time()
    c = cell
    row = {k: (v if not isinstance(v, dict) else str(v)) for k, v in c.items()}
    row.update(rep=int(rep), B=B, R=R)
    for csq in G.CONSEQUENCES:
        row[f"true_pi__{csq}"] = G.truth(c, csq)["pi"]
    K = G.n_components(c)
    gaussian = c.get("shape", "gauss") == "gauss"
    try:
        d = G.make_data(G.rng_for(c, rep), c)
        ev, sc = d["eval"], d["score"]
        names, cell_of_row = cell_keys(d, c["assay"])
        row["n_cells"] = len(names)
        row["cell_sizes"] = ";".join(f"{n}:{k}" for n, k in zip(names, np.bincount(cell_of_row)))
        from src.assay_calibration.data_utils.dataset import BasicScoreset
        onehot = np.zeros((len(sc), len(names)), bool); onehot[np.arange(len(sc)), cell_of_row] = True
        ss_cells = BasicScoreset(sc, onehot, sample_names=names)
        cal_store = {}

        def calib_eval(arm, ss, fits, seeds, eval_csqs):
            cal = calibrate(ss, fits, seeds, K)
            row[f"{arm}_ok"] = cal is not None
            if cal is None:
                return None
            for k, v in prior_diagnostics(cal).items():
                row[f"{arm}_{k}"] = v
            for csq in eval_csqs:
                l5, l50, l95 = lr_bounds(cal, ev[csq]["x"])
                evaluate(row, arm, csq, float(cal["prior"]), l5, l50, l95, ev, c, cal)
            cal_store[arm] = cal
            return cal

        # pooled and missense-only refits
        for arm in ("pooled", "mis"):
            m = layout(d, "pooled" if arm == "pooled" else "mis")
            ss, _ = scoreset(sc, m)
            fits, seeds = fit_arm(ss, K, B, R, gaussian)
            row[f"{arm}_n_fits"] = len(fits)
            if fits:
                calib_eval(arm, ss, fits, seeds, G.CONSEQUENCES)
        # cells: one fit, many compositions
        cfits, cseeds = fit_arm(ss_cells, K, B, R, gaussian)
        row["cells_n_fits"] = len(cfits)
        xls = [f["fit"]["xlims"] for f in cfits]
        tfits = true_cell_fits(ss_cells, c, cseeds, xls, gaussian) if (cfits and gaussian) else []
        comps = [("cell_all", "pooled", None, G.CONSEQUENCES), ("cell_mis", "cell_mis", None, ("missense",))]
        if c["assay"] == "fitness":
            comps.append(("cell_mis_nosplsyn", "cell_mis_nosplsyn", None, ("missense",)))
        for csq in G.CONSEQUENCES:
            comps += [(f"cell_{csq}", "cell_c", csq, (csq,)), (f"nu_{csq}", "nu_c", csq, (csq,)),
                      (f"poolP_{csq}", "poolP_c", csq, (csq,))]
        for arm, kind, csq, evs in comps:
            m = layout(d, kind, csq)
            if not m[:, POP].any():
                row[f"{arm}_ok"] = False; continue
            ss, keep = scoreset(sc, m)
            M = membership(m, keep, cell_of_row, len(names))
            if cfits:
                calib_eval(arm, ss, compose_fits(cfits, M), cseeds, evs)
            if tfits and (arm == "cell_mis" or kind == "cell_c"):
                calib_eval(f"true_{arm}", ss, compose_fits(tfits, M), cseeds, evs)
        # mismatch diagnostics: own prior with pooled LR, pooled prior with own LR
        for csq in G.CONSEQUENCES:
            own, pool = cal_store.get(f"cell_{csq}"), cal_store.get("pooled")
            if own is None or pool is None:
                continue
            x = ev[csq]["x"]
            po, pp_ = lr_bounds(pool, x), lr_bounds(own, x)
            evaluate(row, "mm_ownprior_poolLR", csq, float(own["prior"]), *po, ev, c)
            evaluate(row, "mm_poolprior_ownLR", csq, float(pool["prior"]), *pp_, ev, c)
        row["status"] = "ok"
    except Exception as e:  # recorded, not raised
        import traceback
        row["status"] = f"error: {type(e).__name__}: {e} | {traceback.format_exc()[-300:]}"[:600]
    row["seconds"] = round(time.time() - t0, 1)
    return row


# ── pilot ────────────────────────────────────────────────────────────────────────────────────────────────────────

def pilot_cells():
    return [G.make_cell("core", "mid", "realistic", kappa=3.5, assay="fitness"),
            G.make_cell("core", "mid", "realistic", kappa=3.5, assay="abundance"),
            G.make_cell("core", "small", "extreme", kappa=2.0, assay="fitness"),
            G.make_cell("core", "large", "balanced", kappa=5.0, assay="abundance"),
            G.make_cell("slice", "mid", "realistic", assay="fitness", hypomorph=True, nu=0.3),
            G.make_cell("slice", "mid", "realistic", assay="fitness", n_blb=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--R", type=int, default=R_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--out", default="/data/ross/assay_calibration/consequence_cells_sim/pilot/pilot.csv")
    a = ap.parse_args()
    if a.pilot:
        import os
        from joblib import Parallel, delayed
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        cells = pilot_cells()
        rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(run_cell)(c, r, a.B, a.R) for c in cells for r in range(a.reps))
        df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
        print(df[["cell_id", "rep", "status", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
