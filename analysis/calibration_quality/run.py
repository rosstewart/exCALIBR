"""Calibration-quality simulation: one replicate of one cell, three arms.

Arms (same data, same bootstrap resamples):
    truth   true lr+ and true prior -> points with the pipeline's thresholds (the ideal calibration)
    true    true components; per-bootstrap mixing weights estimated by weight-only EM on that bootstrap's
            resampled rows; then the production calibration (per-bootstrap prior, discard rule, median prior,
            5th/95th-percentile lr+, thresholds, post-processing). The best the pipeline can do with a perfect
            fit: bootstrap and control-sampling variability plus the intended percentile conservatism.
    fit     production fit (2 components, skew locked to Gaussian because the truth is Gaussian) and calibration.

Metrics compare fit vs truth, fit vs true, and true vs truth, on held-out draws (truly pathogenic, truly benign, and
population-like), plus the user-facing diagnostics (Yang distance per sample, prior spread / discards / EM-start
share across bootstraps).

Pilot (local):  python -m analysis.calibration_quality.run --pilot [--n-jobs 6]
Sanity check:   python -m analysis.calibration_quality.run --sanity
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd

from analysis.calibration_quality import generate as G

warnings.filterwarnings("ignore")
POINTS = [1, 2, 3, 4, 5, 6, 7, 8]
TIERS = (8, 4, 2, 1)
B_DEFAULT, R_DEFAULT = 100, 4
# Evidence-threshold mappings recorded for every replicate: column prefix per mapping. Tavtigian (the manuscript's
# mapping) is primary and unprefixed; lp_anchored (pathogenic thresholds follow the prior, benign thresholds fixed
# at the prior-0.1 values) is recorded alongside with an "lp_" prefix. The fits and priors are shared.
MAPPINGS = {"tavtigian": "", "lp_anchored": "lp_"}
YANG_N_GRID = 2000
YANG_MIN_N = 5


# ── scoreset, fits, calibration ──────────────────────────────────────────────────────────────────────────────────

def build_scoreset(samples):
    """BasicScoreset with all four standard sample columns, empty ones included, as production builds it: the
    scoreset compacts empty columns itself (`sample_assignments`), and process_component_fits shifts the sample
    indices accordingly, so passing pre-compacted columns would shift them twice. Production names let the pipeline
    assign roles and switch to synonymous-only or PU itself. Returns the scoreset and the original sample index of
    each non-empty (fit-weight) column."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    cols = [i for i, s in enumerate(samples) if len(s)]
    scores = np.concatenate([samples[i] for i in cols])
    sa = np.zeros((len(scores), len(samples)), bool)
    o = 0
    for i in cols:
        sa[o:o + len(samples[i]), i] = True
        o += len(samples[i])
    return BasicScoreset(scores, sa, sample_names=list(G.NAMES)), cols


def bootstrap_splits(ss, seed):
    """The production train/held-out split of one bootstrap (identical to the one Fit.run uses for this seed)."""
    from src.assay_calibration.fit_utils.fit import Fit
    jobs = Fit(ss).generate_fit_jobs([2], bootstrap=True, bootstrap_seed=int(seed), num_fits=1, master_seed=0,
                                     check_monotonic=True, force_gaussian=True)
    return jobs[0]


def fit_arm(ss, B, R):
    from src.assay_calibration.fit_utils.fit import Fit
    fitter = Fit(ss)
    fits, seeds = [], []
    for b in range(B):
        try:
            models, best, _ = fitter.run([2], bootstrap=True, bootstrap_seed=b, num_fits=R, master_seed=0,
                                         check_monotonic=True, force_gaussian=True, core_limit=1, verbose=False)
            if best is None:
                continue
            fits.append({"fit": models[best]})
            seeds.append(b)
        except Exception:
            continue
    return fits, np.array(seeds, int)


def true_arm(ss, c, seeds, splits, xlims):
    """True components, mixing weights re-estimated on each bootstrap's resampled training rows."""
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    params = G.true_component_params(c)
    fits = []
    for b in seeds:
        job = splits[b]
        obs, sa = job["train_observations"], job["train_sample_assignments"]
        W = np.vstack([_em_weights(obs[sa[:, j]], params, [0.5, 0.5]) if sa[:, j].any() else [0.0, 1.0]
                       for j in range(sa.shape[1])])
        fits.append({"fit": {"component_params": params, "weights": W, "xlims": xlims.get(b, xlims["data"])}})
    return fits


def calibrate(ss, fits, seeds, mapping="tavtigian", manual_prior=None):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    cfg = PipelineConfig(dataset_csv="", dataset_name="sim", output_dir="/tmp", components=[2],
                         benign_method="avg", use_median_prior=True, use_2c_equation=False,
                         point_values=POINTS, n_jobs=1, acmg_mapping_method=mapping, manual_prior=manual_prior)
    lg = logging.getLogger("calqual")
    lg.setLevel(logging.ERROR)
    return process_component_fits(fits=fits, scoreset=ss, config=cfg, n_c=2, logger=lg, bootstrap_seeds=seeds)


# ── evaluation ───────────────────────────────────────────────────────────────────────────────────────────────────

def truth_points(log_lr, prior, mapping="tavtigian"):
    from src.assay_calibration.fit_utils.fit import thresholds_from_prior
    lrP, lrB, _ = thresholds_from_prior(prior, POINTS, acmg_mapping_method=mapping)
    pts = np.zeros(len(log_lr), int)
    for pv, tp, tb in zip(POINTS, np.log(lrP), np.log(lrB)):
        pts[log_lr >= tp] = pv
        pts[log_lr <= tb] = -pv
    return pts


def to_tier(pts):
    """Floor points to the standard ACMG/AMP levels (+-1, 2, 4, 8), for comparison with acmgscaler."""
    a = np.abs(pts)
    out = np.zeros_like(a)
    for t in sorted(TIERS):
        out[a >= t] = t
    return np.sign(pts) * out


def prior_effect(ss, fits, seeds, cal, true_prior, x):
    """Effect of the prior's error alone on the evidence: the production calibration rerun with the prior fixed at
    the true value (manual_prior) on the bootstraps the fitted-prior calibration kept (a fixed prior keeps them all
    otherwise), compared with the same rerun at the fitted prior, on the evaluation variants `x` (the population /
    gnomAD reference set). Returns the share of variants whose points change and the mean |change|."""
    keep = np.asarray(cal.get("valid_mask", np.ones(len(fits), bool)), bool)
    fk, sk = [f for f, k in zip(fits, keep) if k], np.asarray(seeds)[keep]
    at_fit, at_true = calibrate(ss, fk, sk, manual_prior=float(cal["prior"])), calibrate(ss, fk, sk,
                                                                                       manual_prior=true_prior)
    if at_fit is None or at_true is None:
        return {}
    a, b = points_of(at_fit, x), points_of(at_true, x)
    return dict(prior_effect_pop_changed=float((a != b).mean()),
                prior_effect_pop_mean_abs=float(np.abs(a - b).mean()))


def points_of(cal, x):
    from src.assay_calibration.fit_utils.fit import assign_points
    return np.asarray(assign_points(x, {int(k): v for k, v in cal["point_ranges"].items()}), int)


def compare(est, ref):
    """Agreement of estimated points with reference points on the same variants."""
    d = est - ref
    over = (est > np.maximum(ref, 0)) | (est < np.minimum(ref, 0))  # more extreme than the reference, either way
    wrong = ((est > 0) & (ref < 0)) | ((est < 0) & (ref > 0))
    return dict(mae=float(np.abs(d).mean()), mean_abs=float(np.abs(est).mean()), over=float(over.mean()),
                wrong=float(wrong.mean()))


def prior_diagnostics(cal):
    pr = np.asarray(cal.get("priors", []), float)
    fin = pr[np.isfinite(pr)]
    kept = fin[(fin > 0.001) & (fin < 0.999)]
    return dict(prior=float(cal["prior"]) if cal.get("prior") is not None else np.nan,
                prior_sd=float(np.std(kept)) if len(kept) > 1 else np.nan,
                prior_iqr=float(np.subtract(*np.percentile(kept, [75, 25]))) if len(kept) > 1 else np.nan,
                pct_kept=float(cal.get("pct_bootstraps_kept", np.nan)),
                frac_at_start=float(np.mean(np.abs(kept - 0.5) < 1e-3)) if len(kept) else np.nan,
                n_valid=int(cal.get("n_valid_fits", len(kept))))


def yang_per_sample(fits, seeds, splits, cols, split):
    """Median across bootstraps of each sample's Yang distance (production definition, analysis/yang_distance)."""
    from analysis.yang_distance import _process_single_bootstrap
    keys = {G.NAMES[i]: j for j, i in enumerate(cols)}
    per = {k: [] for k in keys}
    for f, b in zip(fits, seeds):
        job = splits[int(b)]
        if split == "val" and job["val_observations"] is None:
            continue
        d = _process_single_bootstrap(int(b), "sim", "x", {int(b): {"x": f}}, {int(b): job}, keys,
                                      n_grid=YANG_N_GRID, split=split)
        if split == "val":  # only samples with enough held-out variants count, as in the manuscript
            sa = job["val_sample_assignments"]
            d = {k: (v if sa[:, keys[k]].sum() >= YANG_MIN_N else np.nan) for k, v in d.items()}
        for k, v in d.items():
            per[k].append(v)
    return {k: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan for k, v in per.items()}


# ── posteriors (exact; no evidence-level mapping) ───────────────────────────────────────────────────────────────

POST_BINS = (0.9, 0.99)  # calibration bins: posterior >= b (pathogenic end) and <= 1 - b (benign end)
POST_OVER_TOL = 0.01
# Posterior bin edges for the reliability curve and the true-vs-model histograms (finer near 0 and 1)
POST_EDGES = np.array([0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99,
                       0.995, 0.999, 1.0])


def _bin(post):
    return np.clip(np.searchsorted(POST_EDGES, post, side="right") - 1, 0, len(POST_EDGES) - 2)


def posterior_histograms(post, post_true=None, is_path=None):
    """Compact per-replicate counts, ';'-joined: with `is_path` (population draws), variants and true pathogenic
    variants per model-posterior bin (reliability curve); with `post_true`, the true x model posterior histogram,
    row-major over (true bin, model bin)."""
    out = {}
    b = _bin(post)
    K = len(POST_EDGES) - 1
    if is_path is not None:
        out["rel_n"] = ";".join(map(str, np.bincount(b, minlength=K)))
        out["rel_path"] = ";".join(map(str, np.bincount(b, weights=is_path.astype(float), minlength=K).astype(int)))
    if post_true is not None:
        h = np.bincount(_bin(post_true) * K + b, minlength=K * K)
        out["hist2d"] = ";".join(map(str, h))
    return out


def posterior(lr, prior):
    lr = np.asarray(lr, float)
    return prior * lr / (prior * lr + (1 - prior))


def conservative_lr(log_lr5, log_lr95):
    """The reported likelihood ratio: the 5th-percentile lr+ where it exceeds 1 (pathogenic evidence), else the
    95th percentile where it is below 1 (benign evidence), else 1 (the bootstrap interval crosses 1: no evidence)."""
    lr5, lr95 = np.exp(log_lr5), np.exp(log_lr95)
    return np.where(lr5 > 1, lr5, np.where(lr95 < 1, lr95, 1.0))


def lr_bounds(cal, x):
    """5th, 50th and 95th percentile of log lr+ across bootstrap fits at scores x (curves interpolated as in the
    variant tables; scores beyond the grid take the edge value)."""
    sr = np.asarray(cal["score_range"], float)
    llr = np.asarray(cal["log_lr_plus"], float)
    at = np.vstack([np.interp(x, sr, row) for row in llr])
    return [np.nanpercentile(at, q, axis=0) for q in (5, 50, 95)]


def posterior_metrics(post, post_true, lr_used, is_path=None):
    """Agreement of an estimated posterior with the exact one on the same variants. `lr_used` is the likelihood
    ratio behind `post` (its side defines the direction of the claim)."""
    claim = np.where(lr_used > 1, post - post_true, np.where(lr_used < 1, post_true - post, 0.0))
    out = dict(post_mae=float(np.abs(post - post_true).mean()),
               post_over=float((claim > POST_OVER_TOL).mean()),
               post_over_mean=float(np.clip(claim, 0, None).mean()))
    if is_path is not None:  # calibration: the observed pathogenic share among variants with extreme posteriors
        for b in POST_BINS:
            hi, lo = post >= b, post <= 1 - b
            tag = f"{int(round(100 * b))}"
            out[f"post_ge{tag}_share"] = float(hi.mean())
            out[f"post_ge{tag}_obs"] = float(is_path[hi].mean()) if hi.any() else np.nan
            low = f"{100 - int(round(100 * b)):02d}"  # e.g. le10 = posterior <= 0.10, le01 = posterior <= 0.01
            out[f"post_le{low}_share"] = float(lo.mean())
            out[f"post_le{low}_obs"] = float(is_path[lo].mean()) if lo.any() else np.nan
    return out


def posterior_rows(cal, ev, c, is_path_pop, arm):
    """Conservative (reported) and central posterior of one calibration vs the exact posterior."""
    row = {}
    prior = float(cal["prior"])
    for k, x in ev.items():
        true_post = posterior(np.exp(G.true_log_lr(x, c)), c["pi"])
        l5, l50, l95 = lr_bounds(cal, x)
        lr_c = conservative_lr(l5, l95)
        tl = G.true_log_lr(x, c)
        with np.errstate(divide="ignore"):
            rl = np.log(lr_c)
        # log-LR retention (mapping-free): mean |reported log LR| and mean |true log LR|; their ratio of totals over
        # replicates is the share of the truth's evidence retained. "_dir" keeps only reported evidence pointing the
        # same way as the truth (wrong-direction evidence counts as none).
        row[f"{arm}_{k}_llr_rep_abs"] = float(np.abs(rl).mean())
        row[f"{arm}_{k}_llr_rep_abs_dir"] = float(np.where(np.sign(rl) == np.sign(tl), np.abs(rl), 0.0).mean())
        row[f"{arm}_{k}_llr_true_abs"] = float(np.abs(tl).mean())
        cap = np.log(LLR_CAP)  # usable evidence: |log LR| capped at Very Strong (LR 350, 8 points at prior 0.10)
        row[f"{arm}_{k}_llr_rep_abs_dir_cap"] = float(np.where(np.sign(rl) == np.sign(tl), np.minimum(np.abs(rl), cap),
                                                               0.0).mean())
        row[f"{arm}_{k}_llr_true_abs_cap"] = float(np.minimum(np.abs(tl), cap).mean())
        post_c = posterior(lr_c, prior)
        for m, v in posterior_metrics(post_c, true_post, lr_c, is_path_pop if k == "population" else None).items():
            row[f"{arm}_{k}_{m}"] = v
        for m, v in posterior_histograms(post_c, true_post if k != "population" else None,
                                         is_path_pop if k == "population" else None).items():
            row[f"{arm}_{k}_{m}"] = v
        lr_m = np.exp(l50)
        row[f"{arm}_{k}_post_mae_central"] = float(np.abs(posterior(lr_m, prior) - true_post).mean())
    return row


TRUE_DRAWS = 3000
LLR_CAP = 350.0  # Tavtigian's very-strong odds (prior 0.10)


def yang_true_per_sample(fits, seeds, splits, cols, c, rep):
    """Median across bootstraps of the distance between each sample's fitted mixture and its true distribution (the
    held-out Yang distance with the observed held-out scores replaced by TRUE_DRAWS draws from the truth), normalized
    by the training score range as in the manuscript."""
    from analysis.yang_distance import _process_single_bootstrap
    import zlib
    rng = np.random.default_rng([zlib.crc32(c["cell_id"].encode()), int(rep), 99])
    draws = {j: G.sample_draws(rng, c, i, TRUE_DRAWS) for j, i in enumerate(cols)}
    obs = np.concatenate([draws[j] for j in range(len(cols))])
    sa = np.zeros((len(obs), len(cols)), bool)
    o = 0
    for j in range(len(cols)):
        sa[o:o + TRUE_DRAWS, j] = True
        o += TRUE_DRAWS
    keys = {G.NAMES[i]: j for j, i in enumerate(cols)}
    per = {k: [] for k in keys}
    for f, b in zip(fits, seeds):
        job = dict(splits[int(b)])
        job["true_observations"], job["true_sample_assignments"] = obs, sa
        d = _process_single_bootstrap(int(b), "sim", "x", {int(b): {"x": f}}, {int(b): job}, keys,
                                      n_grid=YANG_N_GRID, split="true")
        for k, v in d.items():
            per[k].append(v)
    return {k: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan for k, v in per.items()}


# ── one replicate ────────────────────────────────────────────────────────────────────────────────────────────────

SHORT = {"Pathogenic/Likely Pathogenic": "plp", "Benign/Likely Benign": "blb", "gnomAD": "pop", "Synonymous": "syn"}


def run_cell(cell, rep, B=B_DEFAULT, R=R_DEFAULT):
    t0 = time.time()
    c = G.for_rep(cell, rep)
    row = {**c, "rep": int(rep), "B": B, "R": R}
    rng = G.rng_for(c, rep)
    d = G.make_data(rng, c)
    ss, cols = build_scoreset(d["samples"])
    ev = d["eval"]
    try:
        fits, seeds = fit_arm(ss, B, R)
        row["n_fits"] = len(fits)
        if not fits:
            raise RuntimeError("no fits")
        splits = {int(b): bootstrap_splits(ss, b) for b in seeds}
        xl = {int(b): tuple(f["fit"]["xlims"]) for f, b in zip(fits, seeds)}
        xl["data"] = (float(ss.scores.min()), float(ss.scores.max()))
        arms = {"fit": (fits, seeds)}
        if c.get("shape", "gauss") == "gauss":  # the true-component arm needs a two-Gaussian truth
            arms["true"] = (true_arm(ss, c, seeds, splits, xl), seeds)
        for mapping, pre in MAPPINGS.items():
            truth = {k: truth_points(G.true_log_lr(x, c), c["pi"], mapping) for k, x in ev.items()}
            pts = {}
            for arm, (fs, sd) in arms.items():
                cal = calibrate(ss, fs, sd, mapping)
                if cal is None:  # no bootstrap kept a valid prior: the pipeline gives no calibration
                    row[f"{pre}{arm}_no_calibration"] = True
                    pts[arm] = {k: np.zeros(len(x), int) for k, x in ev.items()}
                    if not pre:
                        row[f"{arm}_prior"], row[f"{arm}_prior_err"] = np.nan, np.nan
                    continue
                row[f"{pre}{arm}_no_calibration"] = False
                if not pre:  # the prior and its diagnostics do not depend on the mapping
                    for k, v in prior_diagnostics(cal).items():
                        row[f"{arm}_{k}"] = v
                    row[f"{arm}_prior_err"] = row[f"{arm}_prior"] - c["pi"]
                pts[arm] = {k: points_of(cal, x) for k, x in ev.items()}
                if not pre:  # posteriors do not depend on the evidence-level mapping
                    row.update(posterior_rows(cal, ev, c, d["eval_pop_is_path"], arm))
                if not pre and arm == "fit":  # prior-error effect on the population variants' evidence
                    row.update({f"fit_{k}": v for k, v in
                                prior_effect(ss, fs, sd, cal, c["pi"], ev["population"]).items()})
                # labeled controls (P/LP, B/LB); with neither (synonymous-only NU), the synonymous sample, as the
                # real-data diagnostic does (diagnostics.real_features)
                ctrl = [s for s in (d["samples"][0], d["samples"][1]) if len(s)] or \
                    [s for s in (d["samples"][3],) if len(s)]
                if ctrl:
                    cp = points_of(cal, np.concatenate(ctrl))
                    row[f"{pre}{arm}_ctrl_determinate"] = float((cp != 0).mean())
            for est, ref in [p for p in (("fit", "truth"), ("fit", "true"), ("true", "truth"))
                             if all(a in pts or a == "truth" for a in p)]:
                for k in ev:
                    e = pts[est][k]
                    r = truth[k] if ref == "truth" else pts[ref][k]
                    for m, v in compare(e, r).items():
                        row[f"{pre}{est}_vs_{ref}_{k}_{m}"] = v
                    if ref == "truth":
                        for m, v in compare(to_tier(e), to_tier(r)).items():
                            row[f"{pre}{est}_vs_{ref}_{k}_tier_{m}"] = v
            for k in ev:
                row[f"{pre}truth_{k}_mean_abs"] = float(np.abs(truth[k]).mean())
                row[f"{pre}truth_{k}_tier_mean_abs"] = float(np.abs(to_tier(truth[k])).mean())
        for split in ("train", "val"):
            for name, v in yang_per_sample(fits, seeds, splits, cols, split).items():
                row[f"yang_{split}_{SHORT[name]}"] = v
        for name, v in yang_true_per_sample(fits, seeds, splits, cols, c, rep).items():
            row[f"yang_true_{SHORT[name]}"] = v
        row["status"] = "ok"
    except Exception as e:  # recorded, not raised: a failed calibration is itself a result
        row["status"] = f"error: {type(e).__name__}: {e}"[:300]
    row["seconds"] = round(time.time() - t0, 1)
    return row


# ── sanity check and pilot ───────────────────────────────────────────────────────────────────────────────────────

def sanity(B=20):
    """True components AND true weights in every bootstrap: the pipeline should return the true prior and the
    truth points (confirms the truth mapping matches the pipeline's)."""
    out = []
    for c in [G._cell("core", 80, 80, "high", 4.0, 1.0, 0.25), G._cell("core", 80, 80, "high", 2.5, 0.5, 0.10),
              G._cell("core", 20, 20, "mid", 1.5, 0.25, 0.10)]:
        d = G.make_data(G.rng_for(c, 0), c)
        ss, cols = build_scoreset(d["samples"])
        W = G.true_weights(c)[cols]
        xl = (float(ss.scores.min()), float(ss.scores.max()))
        fits = [{"fit": {"component_params": G.true_component_params(c), "weights": W, "xlims": xl}}
                for _ in range(B)]
        cal = calibrate(ss, fits, np.arange(B))
        if cal is None:  # every prior discarded (identical fits): report and move on
            out.append(dict(cell=c["cell_id"], prior=np.nan, true_prior=c["pi"]))
            continue
        r = dict(cell=c["cell_id"], prior=float(cal["prior"]), true_prior=c["pi"])
        for k, x in d["eval"].items():
            m = compare(points_of(cal, x), truth_points(G.true_log_lr(x, c), c["pi"]))
            r[f"{k}_mae"], r[f"{k}_wrong"] = m["mae"], m["wrong"]
        out.append(r)
    print(pd.DataFrame(out).round(4).to_string(index=False))


PILOT = [("core", 80, 80, "high", 4.0, 1.0, 0.25), ("core", 20, 20, "mid", 2.5, 0.5, 0.25),
         ("core", 5, 5, "low", 1.5, 0.25, 0.10), ("core", 10, 10, "low", 2.5, 0.25, 0.25),
         ("no_blb", 20, 0, "mid", 2.5, 1.0, 0.25), ("core", 40, 40, "mid", 1.5, 1.0, 0.10)]


def pilot(reps=3, B=B_DEFAULT, R=R_DEFAULT, n_jobs=6, out=None):
    from joblib import Parallel, delayed
    cells = [G._cell(*p) for p in PILOT]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_cell)(c, r, B, R) for c in cells for r in range(reps))
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    cols = ["cell_id", "rep", "status", "seconds", "fit_prior_err", "true_prior_err",
            "fit_vs_truth_pathogenic_mae", "fit_vs_truth_benign_mae", "true_vs_truth_pathogenic_mae",
            "true_vs_truth_benign_mae", "fit_vs_true_pathogenic_mae", "fit_vs_truth_pathogenic_over",
            "yang_val_plp", "yang_val_pop", "fit_pct_kept"]
    print(df[[k for k in cols if k in df]].round(3).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--R", type=int, default=R_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sanity:
        sanity()
    if a.pilot:
        pilot(a.reps, a.B, a.R, a.n_jobs, a.out)


if __name__ == "__main__":
    main()
