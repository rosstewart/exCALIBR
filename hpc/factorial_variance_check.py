"""EXPERIMENT: is factorial's advantage an artifact of lower bootstrap-to-bootstrap
variance? Scores the v2-chosen factorial and canonical configs on the REPORT
bootstraps three ways:

  pct     production points (5th / 95th percentile of log LR+ across bootstraps)
  median  points from the median log LR+ across bootstraps (no variance term)
  oob     points from the median over ONLY the bootstraps in which the variant
          was held out (no variance term, no in-sample optimism); variants
          never held out in those bootstraps are excluded, identically for
          both models

plus, on the OOB median log LR+, a calibration slope (logistic regression of
P/LP vs B/LB on log LR+; 1 = honest magnitudes, < 1 = overconfident).

    nice -n 10 taskset -c 0-7 python hpc/factorial_variance_check.py PALB2 KCNH2 ...
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys, warnings, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factorial_panel as V1  # noqa: E402
import run_factorial_panel_v2 as V2  # noqa: E402


def _points(llr, tau_p, tau_b):
    pts = np.zeros(len(llr), int)
    for pv in range(1, len(tau_p) + 1):
        pts[llr >= tau_p[pv - 1]] = pv
    for pv in range(1, len(tau_b) + 1):
        pts[llr <= tau_b[pv - 1]] = -pv
    return pts


def _metrics(pts, P, B, rows=None):
    m = np.ones(len(pts), bool) if rows is None else rows
    p, b = pts[P & m], pts[B & m]
    tp, fn, tn, fp = int((p > 0).sum()), int((p < 0).sum()), int((b < 0).sum()), int((b > 0).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(TP=tp, FN=fn, TN=tn, FP=fp, coverage=(tp + fn + tn + fp) / max(len(p) + len(b), 1),
                mcc=(tp * tn - fp * fn) / den if den > 0 else 0.0,
                auroc=V1._auroc_points(p.astype(float), b.astype(float)))


def _calibration_slope(llr, y):
    from sklearn.linear_model import LogisticRegression
    ok = np.isfinite(llr)
    if ok.sum() < 10 or len(set(y[ok])) < 2:
        return np.nan
    x = np.clip(llr[ok], -20, 20)[:, None]
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(x, y[ok]).coef_[0, 0])


def analyse(key):
    import config as mv_config
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    s = json.load(open(V2.OUT / key / "scores.json"))
    ms_f, gene, gs, dataset, aux = V1.load(key, True)
    ms_c, *_ = V1.load(key, False)
    local = V1.OUT / key / "canonical_results.json.gz"
    cjson = local if local.exists() else (V1.TP53_RAWTEMP_CANONICAL if gs == "tp53" else Path(mv_config.PAPER_RESULTS_JSON))
    boots = list(V2.REPORT_BOOTS)
    # OOB membership per report bootstrap (rows of ms, identical for both models)
    oob = {b: set(np.asarray(V1.job_for(ms_c, b)["val_variant_indices"]).tolist()) for b in boots}
    out = {"key": key}
    for label, ms, js in (("factorial", ms_f, V2.gdir(key) / "factorial_results.json.gz"), ("canonical", ms_c, cjson)):
        cfg = s[label]["chosen"]
        an = build_gene_set_analysis(ms, gene, str(js), dataset_name=dataset, auxiliary_pathogenic_indices=aux,
                                     gene_set=gs)
        an.raw_boots = {k: v for k, v in an.raw_boots.items() if int(k) in set(boots)}
        order = [int(k) for k in an.raw_boots]
        an.configs = [cfg]
        kw = dict(V1.RUN_KW)
        if aux:
            kw.update(aux_path_percentile=50, aux_ben_percentile=50)
        an.run(**kw)
        r = an.results[cfg]
        L = np.asarray(r["lr_matrix"], float)
        assert L.shape[0] == len(order), f"{key}/{label}: {L.shape[0]} valid of {len(order)} bootstraps"
        tau_p, tau_b = np.asarray(r["tau_p_log"]), np.asarray(r["tau_b_log"])
        sa = np.asarray(ms.sample_assignments, bool)
        P, B = sa[:, an.p_idx], sa[:, an.b_idx]
        N = L.shape[1]
        oob_mask = np.zeros((len(order), N), bool)
        for i, b in enumerate(order):
            oob_mask[i, list(oob[b])] = True
        L_oob = np.where(oob_mask, L, np.nan)
        with np.errstate(all="ignore"):
            med_oob = np.nanmedian(L_oob, axis=0)
        has_oob = np.isfinite(med_oob)
        res = {
            "pct": _metrics(np.asarray(r["points"]), P, B),
            "median": _metrics(_points(np.nanmedian(L, axis=0), tau_p, tau_b), P, B),
            "oob": _metrics(_points(np.where(has_oob, med_oob, 0.0), tau_p, tau_b), P, B, rows=has_oob),
        }
        lab = (P | B) & has_oob
        res["calib_slope_oob"] = _calibration_slope(med_oob[lab], P[lab].astype(int))
        res["median_spread"] = float(np.nanmedian((np.nanpercentile(L, 95, axis=0) - np.nanpercentile(L, 5, axis=0))[P | B]))
        res["config"] = cfg
        out[label] = res
    json.dump(out, open(V2.gdir(key) / "variance_check.json", "w"), indent=1, default=float)
    for label in ("factorial", "canonical"):
        r = out[label]
        print(f"[{key}] {label:9s} {r['config']:7s} spread={r['median_spread']:.2f} calib_slope_oob={r['calib_slope_oob']:.2f}  " +
              "  ".join(f"{m}: mcc={r[m]['mcc']:.3f} cov={r[m]['coverage']:.3f} auc={r[m]['auroc']:.3f}" for m in ("pct", "median", "oob")),
              flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    for k in sys.argv[1:]:
        analyse(k)
