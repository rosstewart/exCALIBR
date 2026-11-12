"""Calibration-reliability diagnostics for the real calibrations, from the calibration-quality simulation.

Two checks, one per failure mode:
- prior reliability (weak-signal failure), in ACMG/AMP evidence points: posterior log-odds = prior log-odds + log LR,
  so a prior whose odds are off by a factor F shifts every posterior by log F, and one Tavtigian point is an odds
  factor of 350^(1/8) (POINT). The error |logit(fitted) - logit(true)| / log(POINT) therefore has the same meaning
  at every prior. Reported: a reliability score, P(prior error < 1 point), from a gradient-boosted classifier with an
  logistic (Platt) recalibration fitted on its grouped cross-validated predictions; and the predicted error in points
  (median, 90th percentile; below). The bootstrap 90% interval of the prior is shown for reference only: in the
  simulation it contains the true prior in ~75% of fits (bias that resampling does not see);
- prior error: the size of the prior's error on the log-odds scale,
  |logit(fitted prior) - logit(true prior)|, so that an error means the same at every prior. Gradient-boosted
  quantile regressions trained on the simulation predict its median and 90th percentile (log scale), each shifted by
  a constant from grouped cross-validation so that the bounds cover 50% and 90% of simulated fits (split-conformal).
  The prediction is reported as prior intervals around the fitted prior, logit(prior) +- predicted error: a typical
  (median) and a 90% interval. A linear model was not used: its coverage fails at fitted priors below 7% and above
  50% (it covers 27% and 66% at its 90% bound). The boosted model also under-covers at fitted priors above 50%
  and below 3%, where the simulation has few fits (the 0.01, 0.025 and 0.60 prior slices in the rerun add them);
- bad fit (systematically wrong evidence from a mixture that does not match the data): a fixed rule on the Yang
  distance, flagged when any sample's in-sample distance exceeds YANG_T, or its held-out distance does with at least
  YANG_MIN_HELDOUT held-out variants (config.YANG_GOOD_FIT_THRESHOLD = 0.2; distances are medians across bootstraps,
  production definition, analysis/yang_distance).

Prior-model features, all computed from a real calibration's production outputs:
- the log10 SD of the bootstrap priors on the probability and (delta method, SD / (p (1 - p))) the logit scale, and the share of bootstraps whose prior stays at the EM start (0.5), from the
  per-bootstrap priors recomputed with production code (analysis.calibration_quality.recompute_priors; only datasets
  whose recomputation reproduces the saved calibration exactly get them), with the simulation's definition
  (run.prior_diagnostics: priors in (0.001, 0.999));
- share of bootstraps discarded (calibration pct_bootstraps_kept);
- share of the labeled controls (P/LP and/or B/LB) with no evidence (in-sample points, variants csv);
- largest in-sample Yang distance over the samples.
The control separation (AUC) and control counts are still computed and reported, but not used by the model.
No-P/LP (NU) calibrations are not simulated: the model gives them a prediction, but its accuracy there is untested.
The mechanism-mix-bias slice is left out of training: control-composition bias cannot be seen in the fit.

    python -m analysis.calibration_quality.diagnostics [--out DIR] [--n-jobs 12] [--no-plots] [--datasets ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Every feature is defined whichever samples a calibration has (PN, PU or NU), so single-class calibrations get a
# prediction and the simulated PU / no-B/LB replicates are in the training set. The control separation and the
# smaller control count were dropped: undefined (or degenerate) without both classes, and they add nothing once
# these five are in (grouped-CV AUC 0.822 with them on two-class replicates, 0.825 without, on all).
# The fitted prior itself is not a feature: the simulated true priors take a few fixed values, so a model given the
# fitted prior learns the distance to the nearest simulated value as "error" (intervals ending at exactly 0.25 or
# 0.10 for precise calibrations). It still sets where each interval is centred.
FEATURES = ["log_prior_sd", "log_sd_logit", "frac_at_start", "pct_discarded", "no_evidence", "yang_train_max"]
FEATURE_LABELS = {
    "logit_prior": "fitted prior",
    "log_prior_sd": "prior SD (bootstrap)",
    "log_sd_logit": "prior SD, log-odds scale",
    "frac_at_start": "bootstraps with prior at EM start",
    "pct_discarded": "bootstraps discarded",
    "no_evidence": "controls with no evidence",
    "yang_train_max": "max in-sample Yang distance",
    "ctrl_auc": "P/LP vs B/LB separation (AUC)",
    "log_n_ctrl": "smaller control count",
}
ERR_CAP = 5.0  # log-odds error given to a replicate with no prior (calibration failed); errors are capped here
ERR_EPS = 0.02  # offset of the log transform of the error
QUANTILES = (0.5, 0.9)
POINT = float(np.log(350) / 8)  # log odds factor of one evidence point (Tavtigian et al.)
REL_COLORS = [(0.9, "#4a9b6e"), (0.75, "#93b85a"), (0.5, "#d9b440"), (0.25, "#e07b39"), (-1, "#c8423b")]


def rel_color(r):
    return next(c for lo, c in REL_COLORS if r >= lo)
YANG_T = 0.2
YANG_MIN_HELDOUT = 5
SAMPLES = {"plp": "P/LP", "blb": "B/LB", "gnomad": "gnomAD", "synonymous": "Synonymous"}
PLP, BLB = "Pathogenic/Likely Pathogenic", "Benign/Likely Benign"


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return np.nan
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def prior_stats(priors):
    """SD and share at the EM start of the bootstrap priors, as run.prior_diagnostics computes them."""
    pr = np.asarray(priors, float)
    kept = pr[np.isfinite(pr) & (pr > 0.001) & (pr < 0.999)]
    return (float(np.std(kept)) if len(kept) > 1 else np.nan,
            float(np.mean(np.abs(kept - 0.5) < 1e-3)) if len(kept) else np.nan)


# ── simulation: features, training, the Yang rule ────────────────────────────────────────────────────────────────

def simulation_table(out_dir, n_jobs=16):
    """All simulated replicates with the prior-model features (control separation recomputed from each replicate's
    regenerated controls, generate.rng_for, and cached)."""
    from joblib import Parallel, delayed
    from analysis.calibration_quality import generate as G
    from analysis.calibration_quality.summarize import load
    out_dir = Path(out_dir)
    df = load(out_dir)
    cache = out_dir / "summary" / "ctrl_auc.csv"
    if cache.exists():
        ca = pd.read_csv(cache)
    else:
        cells = {c["cell_id"]: c for c in G.design()}

        def one(cid, rep):
            c = G.for_rep(cells[cid], rep)
            d = G.make_data(G.rng_for(c, rep), c)
            p, b = d["samples"][0], d["samples"][1]
            return _auc(np.r_[np.ones(len(p)), np.zeros(len(b))], np.r_[p, b]) if len(p) and len(b) else np.nan
        ca = df[["cell_id", "rep"]].copy()
        ca["ctrl_auc"] = Parallel(n_jobs=n_jobs)(delayed(one)(c, r) for c, r in zip(ca.cell_id, ca.rep))
        cache.parent.mkdir(exist_ok=True)
        ca.to_csv(cache, index=False)
    df = df.merge(ca, on=["cell_id", "rep"], how="left")
    df["log_prior_sd"] = np.log10(df.fit_prior_sd.clip(lower=1e-4))
    fp = df.fit_prior.clip(1e-4, 1 - 1e-4)
    df["logit_prior"] = np.log(fp / (1 - fp))
    df["log_sd_logit"] = np.log10((df.fit_prior_sd / (fp * (1 - fp))).clip(lower=1e-4))
    df["frac_at_start"] = df.fit_frac_at_start
    df["pct_discarded"] = df.fit_pct_discarded
    df["no_evidence"] = 1 - df.fit_ctrl_determinate
    df["log_n_ctrl"] = np.log10(np.minimum(df.n_plp, df.n_blb).clip(lower=1))
    # held-out Yang in the simulation already counts only samples with >= run.YANG_MIN_N (5) held-out variants
    df["yang_flag"] = (df.yang_train_max > YANG_T) | (df.yang_val_max > YANG_T)
    df["over_share"] = (df.fit_vs_truth_pathogenic_over + df.fit_vs_truth_benign_over) / 2
    return df


def logodds_err(s):
    """|logit(fitted prior) - logit(true prior)|, capped at ERR_CAP; a replicate with no prior counts as the cap."""
    fp = s.fit_prior.clip(1e-4, 1 - 1e-4)
    e = np.abs(np.log(fp / (1 - fp)) - np.log(s.pi / (1 - s.pi)))
    return e.where(s.fit_prior.notna(), ERR_CAP).clip(upper=ERR_CAP).to_numpy()


def _hgb(q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=40, random_state=0)


def train(sim):
    """Gradient-boosted quantile regressions (median, 90th percentile) of log(log-odds prior error + ERR_EPS).
    Grouped (by simulated setting) 5-fold cross-validation gives a constant shift per quantile on the log scale that
    makes the cross-validated bounds cover 50% and 90% of fits (split-conformal); reported: coverage after the shift
    overall, by true prior and by fitted prior, the Spearman correlation of the predicted median with the error, and
    the simulated feature ranges (real values are clipped to them before prediction)."""
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold
    s = sim[sim.experiment != "mech_bias"].dropna(subset=FEATURES)
    X, err, grp = s[FEATURES].to_numpy(), logodds_err(s), s.cell_id.to_numpy()
    ly = np.log(err + ERR_EPS)
    cv = {q: np.zeros(len(s)) for q in QUANTILES}
    for tr, te in GroupKFold(5).split(X, ly, grp):
        for q in QUANTILES:
            cv[q][te] = _hgb(q).fit(X[tr], ly[tr]).predict(X[te])
    shift = {q: float(np.quantile(ly - cv[q], q)) for q in QUANTILES}
    pred = {q: np.exp(cv[q] + shift[q]) - ERR_EPS for q in QUANTILES}

    def cover(mask):
        return {f"q{int(100 * q)}": round(float((err[mask] <= pred[q][mask]).mean()), 3) for q in QUANTILES}
    fb = pd.cut(s.fit_prior, [0, 0.03, 0.07, 0.15, 0.3, 0.5, 1])
    # reliability: P(error < 1 evidence point)
    from sklearn.ensemble import HistGradientBoostingClassifier
    yc = (err < POINT).astype(int)
    clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 min_samples_leaf=40, random_state=0)
    pc = np.zeros(len(s))
    for tr, te in GroupKFold(5).split(X, yc, grp):
        pc[te] = clf().fit(X[tr], yc[tr]).predict_proba(X[te])[:, 1]
    # smooth (Platt) recalibration on the logit of the cross-validated probabilities; isotonic gives ties
    from sklearn.linear_model import LogisticRegression
    _lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    platt = LogisticRegression().fit(_lg(pc), yc)
    iso = lambda p: platt.predict_proba(_lg(p))[:, 1]
    rel = iso(pc)
    rb = pd.cut(rel, [-0.001, 0.25, 0.5, 0.75, 0.9, 1.0])

    def calib(mask):
        d = pd.DataFrame(dict(b=rb[mask], p=rel[mask], y=yc[mask]))
        return {str(k): (round(float(g.p.mean()), 3), round(float(g.y.mean()), 3), len(g))
                for k, g in d.groupby("b", observed=True)}
    from sklearn.metrics import roc_auc_score
    return dict(models={q: _hgb(q).fit(X, ly) for q in QUANTILES}, shift=shift, n=len(s),
                clf=clf().fit(X, yc), iso=iso, rel_auc=float(roc_auc_score(yc, rel)),
                rel_share_reliable=float(yc.mean()),
                rel_calibration=calib(np.ones(len(s), bool)),
                rel_calibration_by_true_prior={f"{p:g}": calib((s.pi == p).to_numpy()) for p in sorted(s.pi.unique())},
                rel_calibration_by_fitted_prior={str(b): (round(float(rel[(fb == b).to_numpy()].mean()), 3),
                                                          round(float(yc[(fb == b).to_numpy()].mean()), 3))
                                                 for b in fb.cat.categories},
                spearman=float(spearmanr(pred[0.5], err).correlation),
                coverage=cover(np.ones(len(s), bool)),
                coverage_by_true_prior={f"{p:g}": cover((s.pi == p).to_numpy()) for p in sorted(s.pi.unique())},
                coverage_by_fitted_prior={str(b): cover((fb == b).to_numpy()) for b in fb.cat.categories},
                coverage_by_mode={m: cover((s.experiment == m).to_numpy()) for m in ("pu", "no_blb", "core", "misfit")},
                range={f: (float(s[f].quantile(0.005)), float(s[f].quantile(0.995))) for f in FEATURES},
                sim_sorted={f: np.sort(s[f].to_numpy()) for f in FEATURES},
                sim_err_sorted=np.sort(pred[0.5]))


def yang_rule_summary(sim):
    """How often the Yang rule fires in the simulation: correctly specified, misspecified truths (by shape), and
    mechanism-mix bias; with the share of variants overstated with and without the flag."""
    kind = np.where(sim.experiment == "misfit", "misspecified",
                    np.where(sim.experiment == "mech_bias", "mechanism-mix bias", "correctly specified"))
    g = sim.assign(kind=kind).groupby("kind")
    out = {"flag_rate": g.yang_flag.mean().round(4).to_dict(),
           "overstated_share_flagged": sim[sim.yang_flag].groupby(kind[sim.yang_flag]).over_share.mean()
           .round(4).to_dict(),
           "overstated_share_unflagged": sim[~sim.yang_flag].groupby(kind[~sim.yang_flag]).over_share.mean()
           .round(4).to_dict()}
    mis = sim[sim.experiment == "misfit"]
    out["misspecified_flag_rate_by_shape"] = {f"{k[0]} {k[1]:g}": round(float(v), 3) for k, v in
                                              mis.groupby(["shape", "shape_par"]).yang_flag.mean().items()}
    out["misspecified_flag_rate_by_controls"] = {int(k): round(float(v), 3) for k, v in
                                                 mis.groupby("n_plp").yang_flag.mean().items()}
    return out


# ── real calibrations ───────────────────────────────────────────────────────────────────────────────────────────

def real_datasets(output_dir=None):
    """The calibrated datasets and their selected calibration, as in analyze_pipeline_output (section A1): every
    dataset in the pipeline output except discovery.DEFAULT_EXCLUDED_DATASETS, with its component resolved by
    supplementary_data._resolved_calibration."""
    from analysis import config as cfg
    from analysis.discovery import DEFAULT_EXCLUDED_DATASETS, discover_outputs
    from analysis.supplementary_data import _resolved_calibration
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    tree, model_selections, _ = discover_outputs(output_dir)
    with open(cfg.DATASET_CONFIGS) as f:
        dataset_configs = json.load(f)
    cals = {}
    for d in sorted(set(tree) - set(DEFAULT_EXCLUDED_DATASETS)):
        try:
            cals[d] = _resolved_calibration(d, tree, model_selections, dataset_configs, output_dir=output_dir)
        except FileNotFoundError:
            pass
    return cals


def real_features(cals, priors_json, output_dir=None):
    from analysis import config as cfg
    from analysis.yang_distance import heldout_yang_table
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    rp = json.loads(Path(priors_json).read_text())
    yt = pd.read_csv(cfg.EXCALIBR_DATASETS_TABLE_CSV)
    yh = heldout_yang_table()
    rows = []
    for d, c in cals.items():
        n_c = c.get("n_c")
        r = dict(dataset=d, n_c=n_c, prior=c.get("prior"))
        p = rp.get(d, {})
        r["priors_reproduced"] = bool(p.get("match"))
        r["prior_sd"], r["frac_at_start"] = prior_stats(p["priors"]) if p.get("match") else (np.nan, np.nan)
        r["boot_p5"], r["boot_p95"] = (p["saved_pct"][0], p["saved_pct"][2]) if p.get("match") else (np.nan, np.nan)
        r["log_prior_sd"] = np.log10(max(r["prior_sd"], 1e-4)) if np.isfinite(r["prior_sd"]) else np.nan
        pf = min(max(float(c["prior"]), 1e-4), 1 - 1e-4) if c.get("prior") is not None else np.nan
        r["logit_prior"] = np.log(pf / (1 - pf))
        r["log_sd_logit"] = (np.log10(max(r["prior_sd"] / (pf * (1 - pf)), 1e-4)) if np.isfinite(r["prior_sd"])
                             else np.nan)
        kept = c.get("pct_bootstraps_kept")
        r["pct_discarded"] = 1 - float(kept) if kept is not None else np.nan
        vp = output_dir / d / f"{d}_{n_c}_variants.csv"
        if vp.exists():
            v = pd.read_csv(vp, usecols=["score", "sample", "standard_points"])
            s = v["sample"].fillna("").astype(str)
            is_p, is_b = s.str.contains(PLP, regex=False), s.str.contains(BLB, regex=False)
            is_s = s.str.contains("Synonymous", regex=False)
            ctrl = v[is_p | is_b] if (is_p | is_b).any() else v[is_s]  # synonymous-only NU: the synonymous sample
            r["n_plp"], r["n_blb"] = int(is_p.sum()), int(is_b.sum())
            r["no_evidence"] = float((ctrl["standard_points"].fillna(0) == 0).mean()) if len(ctrl) else np.nan
            one = v[is_p ^ is_b]  # controls in exactly one class
            yp = one["sample"].astype(str).str.contains(PLP, regex=False)
            r["ctrl_auc"] = _auc(yp, one["score"]) if yp.any() and (~yp).any() else np.nan
            r["log_n_ctrl"] = np.log10(max(min(r["n_plp"], r["n_blb"]), 1))
        y = yt[(yt.dataset == d) & (yt.n_c.astype(str) == str(n_c))]
        y = y if len(y) else yt[yt.dataset == d]
        h = yh[yh.dataset == d]
        for k in SAMPLES:
            r[f"yang_in_{k}"] = float(y.iloc[-1][f"yang_dist_{k}"]) if len(y) else np.nan
            r[f"yang_val_{k}"] = float(h.iloc[0][f"yang_val_{k}"]) if len(h) else np.nan
            r[f"n_val_{k}"] = float(h.iloc[0][f"n_val_{k}"]) if len(h) else np.nan
        ins = [r[f"yang_in_{k}"] for k in SAMPLES]
        val = [r[f"yang_val_{k}"] if r[f"n_val_{k}"] >= YANG_MIN_HELDOUT else np.nan for k in SAMPLES]
        r["yang_train_max"] = float(np.nanmax(ins)) if np.isfinite(ins).any() else np.nan
        r["yang_val_max"] = float(np.nanmax(val)) if np.isfinite(val).any() else np.nan
        r["yang_flag"] = bool(np.nan_to_num(ins, nan=0).max() > YANG_T or np.nan_to_num(val, nan=0).max() > YANG_T)
        rows.append(r)
    return pd.DataFrame(rows)


def _expit(x):
    return 1 / (1 + np.exp(-x))


def predict(model, feats):
    """Predicted log-odds prior error (median, 90th percentile) and the prior intervals they give around the fitted
    prior; features clipped to the simulated range, `n_extrapolated` counting those outside it. `error_fifth` places
    the predicted median among the simulated fits' (1 = smallest error, 5 = largest)."""
    X = feats[FEATURES].astype(float).copy()
    out = pd.DataFrame(index=feats.index)
    out["n_extrapolated"] = sum(((X[f] < lo) | (X[f] > hi)).astype(int) for f, (lo, hi) in model["range"].items())
    for f, (lo, hi) in model["range"].items():
        X[f] = X[f].clip(lo, hi)
    ok = X.notna().all(axis=1).to_numpy()
    lp = feats["logit_prior"].to_numpy(float)
    for q in QUANTILES:
        e = np.full(len(X), np.nan)
        if ok.any():
            e[ok] = np.exp(model["models"][q].predict(X[ok].to_numpy()) + model["shift"][q]) - ERR_EPS
        e = np.maximum(e, 0)
        tag = f"q{int(100 * q)}"
        out[f"err_logodds_{tag}"] = e
        out[f"prior_lo_{tag}"], out[f"prior_hi_{tag}"] = _expit(lp - e), _expit(lp + e)
    rel = np.full(len(X), np.nan)
    if ok.any():
        rel[ok] = model["iso"](model["clf"].predict_proba(X[ok].to_numpy())[:, 1])
    out["prior_reliability"] = rel
    for q in QUANTILES:
        out[f"err_points_q{int(100 * q)}"] = out[f"err_logodds_q{int(100 * q)}"] / POINT
    ss = model["sim_err_sorted"]
    out["error_fifth"] = [np.nan if not np.isfinite(v) else min(int(5 * np.searchsorted(ss, v) / len(ss)) + 1, 5)
                          for v in out["err_logodds_q50"]]
    return out


# ── plots ───────────────────────────────────────────────────────────────────────────────────────────────────────────

FIFTH_COLORS = ["#4a9b6e", "#93b85a", "#d9b440", "#e07b39", "#c8423b"]


def _fmt(f, v):
    if not np.isfinite(v):
        return "n/a"
    if f == "log_prior_sd":
        return f"{10 ** v:.3f}"
    if f == "log_sd_logit":
        return f"{10 ** v:.2f}"
    if f == "logit_prior":
        return f"{1 / (1 + np.exp(-v)):.3f}"
    if f == "log_n_ctrl":
        return f"{10 ** v:.0f}"
    if f in ("pct_discarded", "no_evidence", "frac_at_start"):
        return f"{100 * v:.0f}%"
    return f"{v:.2f}"


def draw_panel(ax, row, model):
    """Diagnostics panel: prior-model features (value and simulated percentile), the fitted prior with its predicted
    50% and 90% ranges, and the per-sample Yang distances with the bad-fit flag."""
    from matplotlib.patches import Rectangle
    ax.axis("off")
    y = 0.99
    ax.text(0, y, "Reliability diagnostics", fontsize=13, weight="bold", va="top")
    y -= 0.06
    ax.text(0, y, "Prior (simulation-trained models)", fontsize=10.5, weight="bold", va="top")
    y -= 0.04
    for f in FEATURES:
        v = row.get(f, np.nan)
        lo, hi = model["range"][f]
        out = np.isfinite(v) and (v < lo or v > hi)
        ss = model["sim_sorted"][f]
        ptxt = f" (p{int(100 * np.searchsorted(ss, v) / len(ss))})" if np.isfinite(v) else ""
        ax.text(0, y, FEATURE_LABELS[f], fontsize=9, va="top")
        ax.text(1, y, _fmt(f, v) + ptxt + (" *" if out else ""), fontsize=9, va="top", ha="right",
                color="#b03030" if out else "black", family="monospace")
        y -= 0.034
    npl, nbl = row.get("n_plp", np.nan), row.get("n_blb", np.nan)
    mode = "PU (no B/LB)" if nbl == 0 else ("NU (no P/LP)" if npl == 0 else "P/LP and B/LB")
    ax.text(0, y, f"controls: {mode}", fontsize=8.5, color="#444", va="top")
    ax.text(1, y, f"{npl:.0f} / {nbl:.0f}" if np.isfinite(npl) else "n/a", fontsize=8.5, va="top", ha="right",
            family="monospace", color="#444")
    y -= 0.034
    rel = row.get("prior_reliability", np.nan)
    y -= 0.01
    ax.text(0, y, "Prior reliability", fontsize=10.5, va="top")
    if np.isfinite(rel):
        col = rel_color(rel)
        ax.text(1, y, f"{rel:.2f}", fontsize=13, weight="bold", va="top", ha="right", color=col)
        y -= 0.04
        ax.add_patch(Rectangle((0, y - 0.02), 1, 0.02, color="#eee", transform=ax.transAxes, clip_on=False))
        ax.add_patch(Rectangle((0, y - 0.02), rel, 0.02, color=col, transform=ax.transAxes, clip_on=False))
        y -= 0.03
        ax.text(0, y, "P(prior error shifts posteriors by < 1 evidence point)", fontsize=8, color="#444", va="top")
        y -= 0.035
        ax.text(0, y, "predicted prior error (points)", fontsize=9, va="top")
        ax.text(1, y, f"median {row['err_points_q50']:.2f}, 90% {row['err_points_q90']:.2f}", fontsize=9,
                va="top", ha="right", family="monospace")
        y -= 0.032
        ax.text(0, y, "fitted prior (bootstrap 90%)", fontsize=9, va="top")
        ax.text(1, y, f"{row['prior']:.3f} ({row['boot_p5']:.3f}-{row['boot_p95']:.3f})", fontsize=9, va="top",
                ha="right", family="monospace")
        y -= 0.028
        ax.text(0, y, "bootstrap intervals contain the true prior in ~75% of simulated fits", fontsize=7.5,
                color="#666", va="top")
        y -= 0.035
    else:
        reason = "per-bootstrap priors not reproduced" if not row.get("priors_reproduced") else "missing feature"
        ax.text(1, y, "n/a", fontsize=12, va="top", ha="right")
        y -= 0.04
        ax.text(0, y, reason, fontsize=8, color="#444", va="top")
        y -= 0.04
    ax.text(0, y, "* outside the simulated range (clipped)", fontsize=7.5, color="#666", va="top")
    y -= 0.05

    ax.text(0, y, f"Bad fit (Yang distance > {YANG_T:g})", fontsize=10.5, weight="bold", va="top")
    y -= 0.04
    ax.text(0.42, y, "in-sample", fontsize=8.5, va="top", ha="right", color="#444")
    ax.text(1, y, f"held-out (n)", fontsize=8.5, va="top", ha="right", color="#444")
    y -= 0.032
    for k, lab in SAMPLES.items():
        vi, vv, n = row.get(f"yang_in_{k}", np.nan), row.get(f"yang_val_{k}", np.nan), row.get(f"n_val_{k}", np.nan)
        if not (np.isfinite(vi) or np.isfinite(vv)):
            continue
        ax.text(0, y, lab, fontsize=9, va="top")
        ax.text(0.42, y, f"{vi:.3f}" if np.isfinite(vi) else "n/a", fontsize=9, va="top", ha="right",
                family="monospace", color="#b03030" if np.isfinite(vi) and vi > YANG_T else "black")
        counts = np.isfinite(n) and n >= YANG_MIN_HELDOUT
        vtxt = (f"{vv:.3f}" if np.isfinite(vv) else "n/a") + (f" ({int(n)})" if np.isfinite(n) else "")
        ax.text(1, y, vtxt, fontsize=9, va="top", ha="right", family="monospace",
                color=("#b03030" if counts and vv > YANG_T else ("black" if counts else "#999")))
        y -= 0.03
    y -= 0.008
    flag = bool(row.get("yang_flag"))
    ax.text(0, y, "FLAGGED: fit does not match the data" if flag else "not flagged", fontsize=11, weight="bold",
            va="top", color="#c8423b" if flag else "#4a9b6e")
    y -= 0.04
    ax.text(0, y, f"held-out values in grey have < {YANG_MIN_HELDOUT} held-out variants\n"
                  "(median across bootstraps) and are not used", fontsize=7.5, color="#666", va="top")


def plot_dataset(dataset, row, model, out_png):
    """The SD2 ("full" style) calibration figure for one dataset with the diagnostics panel on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import extended_data_appendix as A
    from analysis.plot_common import display_name
    scoreset, scoreset_2018, summ, fits, score_range, n_c, flipped, n_samples = A._load_full_style(
        dataset, None, None, None, None)
    fig = A.plot_calibration(dataset, scoreset_2018, scoreset, summ, fits, score_range, n_c, n_samples,
                             flipped=flipped)
    w, h = fig.get_size_inches()
    scale = w / (w + 4.4)
    fig.set_size_inches(w + 4.4, h)
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position([p.x0 * scale, p.y0, p.width * scale, p.height])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x * scale, y))
    fig.suptitle(display_name(dataset), x=0.02, ha="left", fontsize=15, weight="bold")
    draw_panel(fig.add_axes([scale + 0.012, 0.05, 1 - scale - 0.025, 0.88]), row, model)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def overview_figure(tab, model, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.plot_common import display_name
    t = tab.dropna(subset=["prior_reliability"]).copy().sort_values("prior_reliability")
    fig, axs = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    yy = np.arange(len(t))
    cols = [rel_color(r) for r in t.prior_reliability]
    ax.barh(yy, t.prior_reliability, color=cols, height=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{display_name(d)} ({p:.3f})" for d, p in zip(t.dataset, t.prior)], fontsize=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("prior reliability: P(prior error < 1 evidence point)")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1]
    t["yang_any"] = t[["yang_train_max", "yang_val_max"]].max(axis=1)
    ax.scatter(t.prior_reliability, t.yang_any, s=28, c=cols, alpha=0.9, edgecolor="white", lw=0.5)
    for _, r in t[t.prior_reliability < 0.5].iterrows():
        ax.annotate(display_name(r.dataset), (r.prior_reliability, r.yang_any), fontsize=6.5, xytext=(3, 2),
                    textcoords="offset points")
    ax.axhline(YANG_T, color="#c8423b", ls="--", lw=0.8)
    ax.set_xlabel("prior reliability")
    ax.set_xlim(0, 1)
    ax.set_ylabel(f"largest Yang distance (in-sample, or held-out with >= {YANG_MIN_HELDOUT})")
    ax.set_ylim(0, max(0.25, t.yang_any.max() * 1.1))
    ax.set_title(f"Prior error and fit quality (n = {len(t)})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    from joblib import Parallel, delayed
    from analysis import config as cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=cfg.CALIBRATION_QUALITY_DIR)
    ap.add_argument("--out", default=str(Path(cfg.CALIBRATION_QUALITY_DIR).parent / "real_diagnostics"))
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "per_dataset").mkdir(parents=True, exist_ok=True)

    sim = simulation_table(a.sim_dir, n_jobs=a.n_jobs)
    model = train(sim)
    yr = yang_rule_summary(sim)
    print(f"prior error model: {model['n']} replicates; grouped-CV Spearman {model['spearman']:.3f}; "
          f"coverage {model['coverage']}")
    print(f"  by true prior {model['coverage_by_true_prior']}")
    print(f"  by fitted prior {model['coverage_by_fitted_prior']}")
    print(f"  by mode {model['coverage_by_mode']}")
    print(f"prior reliability P(error < 1 point): share reliable {model['rel_share_reliable']:.3f}, "
          f"AUC {model['rel_auc']:.3f}")
    print(f"  calibration (mean predicted, observed, n) {model['rel_calibration']}")
    print(f"  by true prior {model['rel_calibration_by_true_prior']}")
    print(f"  by fitted prior (mean predicted, observed) {model['rel_calibration_by_fitted_prior']}")
    print(f"Yang rule in the simulation: {json.dumps(yr, indent=1)}")

    cals = real_datasets()
    feats = real_features(cals, out / "recomputed_priors.json")
    tab = pd.concat([feats, predict(model, feats)], axis=1)
    tab.to_csv(out / "real_diagnostics.csv", index=False)
    summary = {k: v for k, v in model.items() if k not in ("models", "clf", "iso", "sim_sorted", "sim_err_sorted")}
    summary.update(features=FEATURES, yang_rule=dict(threshold=YANG_T, min_heldout=YANG_MIN_HELDOUT,
                                                     simulation=yr, real_flagged=tab.loc[tab.yang_flag, "dataset"]
                                                     .tolist()))
    (out / "model_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    overview_figure(tab, model, out / "overview.png")
    print(f"real calibrations: {len(tab)}; priors reproduced {int(tab.priors_reproduced.sum())}; with a prediction "
          f"{int(tab.err_logodds_q50.notna().sum())}; Yang-flagged {int(tab.yang_flag.sum())}")
    print(tab[["dataset", "prior", "prior_sd", "frac_at_start", "pct_discarded", "no_evidence", "yang_train_max",
               "yang_val_max", "ctrl_auc", "n_plp", "n_blb", "n_extrapolated", "prior_reliability",
               "err_points_q50", "err_points_q90", "boot_p5", "boot_p95"]]
          .sort_values("prior_reliability").round(3).to_string(index=False))
    if a.no_plots:
        return
    for f in (out / "per_dataset").glob("*.png"):
        if not a.datasets:
            f.unlink()
    todo = [r for _, r in tab.iterrows() if not a.datasets or r.dataset in a.datasets]

    def one(r):
        try:
            return plot_dataset(r.dataset, r.to_dict(), model, out / "per_dataset" / f"{r.dataset}.png")
        except Exception as e:
            return f"FAILED {r.dataset}: {type(e).__name__}: {e}"
    res = Parallel(n_jobs=a.n_jobs, verbose=0)(delayed(one)(r) for r in todo)
    fails = [x for x in res if x.startswith("FAILED")]
    print(f"{len(res) - len(fails)} figures -> {out / 'per_dataset'}")
    for x in fails:
        print(" ", x)
    pngs = sorted(Path(x) for x in res if not x.startswith("FAILED"))
    if pngs and not a.datasets:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out / "real_diagnostics_all.pdf") as pdf:
            for p in pngs:
                img = plt.imread(p)
                fig = plt.figure(figsize=(img.shape[1] / 130, img.shape[0] / 130))
                fig.figimage(img)
                pdf.savefig(fig)
                plt.close(fig)
        print(f"combined -> {out / 'real_diagnostics_all.pdf'}")


if __name__ == "__main__":
    main()
