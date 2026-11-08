"""(1) Empirical distribution-shift test, missense vs non-missense P/LP, per dataset.
(2) Does adding a PTV sample to the missense-only refit regularise it?

(1) Two tests on the same P/LP:
    raw     two-sample KS of missense vs non-missense P/LP scores; p from 2000 label permutations
            (exact for small n); effect = difference in medians / pooled P/LP IQR.
    fitted  baseline components fixed (first 100 bootstraps); per fit, P/LP mixing weights estimated
            separately from the fit's training missense and non-missense P/LP rows (weights-only EM);
            statistic = median over fits of the total-variation distance between the two weight vectors.
            Null: 50 random relabelings of the same P/LP rows (same sizes), same statistic; p = share of
            null statistics >= observed.
(2) For each medium refit run (missense-only, +1 component, + PTV sample, PTV pull capped at PTV P/LP, and
    their +1 versions), on every missense variant:
    shift  mean |points - all-P/LP points| (all P/LP = medium_baseline, same prior population)
    width  mean bootstrap band width of log LR+ (95th - 5th percentile) at missense scores
    Both relative to the plain missense-only refit (plpmis).

Usage:  python -m analysis.missense_sensitivity.shift_and_ptv [--n-jobs 32]
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
S = ROOT / "sens_popmis_lp"
PLP = "Pathogenic/Likely Pathogenic"
PTV_RUNS = ["plpmis", "plpmis_p1", "plpmis_ptv", "plpmis_ptv_p1", "plpmis_ptv_pullptv", "plpmis_ptv_pullptv_p1"]


def _tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def shift_task(ds, df_ds, fits_ds, n_perm_raw=2000, n_perm_fit=50, seed=0):
    from scipy.stats import ks_2samp
    from analysis.missense_sensitivity.compare_runs import _missense_map
    from analysis.missense_sensitivity.matched_subsample import training_rows
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release))
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    if PLP not in names:
        return dict(dataset=ds, status="no P/LP")
    pc = names.index(PLP)
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    mis = _missense_map(ds, df_ds)["is_missense"].values
    plp = sa[:, pc].astype(bool)
    xm, xo = sc[plp & mis], sc[plp & ~mis]
    out = dict(dataset=ds, status="ok", n_plp=int(plp.sum()), n_mis=len(xm), n_other=len(xo))
    if len(xm) < 2 or len(xo) < 2:
        out["status"] = "too few in one group"
        return out
    rng = np.random.default_rng(seed)
    allx = np.concatenate([xm, xo])
    d0 = ks_2samp(xm, xo).statistic
    null = [ks_2samp(*np.split(rng.permutation(allx), [len(xm)])).statistic for _ in range(n_perm_raw)]
    iqr = np.subtract(*np.percentile(allx, [75, 25])) or np.nan
    out.update(ks=float(d0), ks_p=float((np.sum(np.asarray(null) >= d0) + 1) / (n_perm_raw + 1)),
               median_shift_iqr=float((np.median(xm) - np.median(xo)) / iqr))
    # fitted-density test
    keys = sorted(fits_ds, key=int)
    obs, nul = [], []
    plp_idx = np.flatnonzero(plp)
    relabels = []
    for _ in range(n_perm_fit):
        m = np.zeros(len(sc), bool)
        m[rng.choice(plp_idx, size=len(xm), replace=False)] = True
        relabels.append(m)
    for k in keys:
        rec = next(iter(fits_ds[k].values()))["fit"]
        P, w = rec["component_params"], rec["weights"][pc]
        rows, onehot = training_rows(sa, sc, int(k))
        rp = rows[onehot[rows, pc]]
        rm, ro = rp[mis[rp]], rp[~mis[rp]]
        if len(rm) and len(ro):
            obs.append(_tv(_em_weights(sc[rm], P, w), _em_weights(sc[ro], P, w)))
        for m in relabels[:5]:  # 5 relabelings per fit x 100 fits = 500 null draws
            a, b = rp[m[rp]], rp[~m[rp]]
            if len(a) and len(b):
                nul.append(_tv(_em_weights(sc[a], P, w), _em_weights(sc[b], P, w)))
    if obs:
        o = float(np.median(obs))
        # compare the observed median with medians of null draws grouped like the observed (per fit)
        nul = np.asarray(nul)
        nmed = [np.median(rng.choice(nul, size=len(obs), replace=True)) for _ in range(500)]
        out.update(tv_weights=o, tv_null_median=float(np.median(nul)),
                   tv_p=float((np.sum(np.asarray(nmed) >= o) + 1) / (len(nmed) + 1)))
    return out


def _variant_points(run, ds):
    f = sorted((S / run / ds).glob(f"{ds}_*_variants.csv"))
    if not f:
        return None
    d = pd.read_csv(f[0], usecols=lambda c: c in {"variant_id", "score", "standard_points", "aa_ref", "aa_alt"})
    a, r = d.aa_alt.astype(str), d.aa_ref.astype(str)
    d = d[d.aa_alt.notna() & d.aa_ref.notna() & (a != r) & ~a.isin(["*", "Ter", "nan", ""])]
    return d.drop_duplicates("variant_id").set_index("variant_id")


def _width(run, ds, x):
    f = sorted((S / run / ds).glob(f"{ds}_*_lr_values.json.gz"))
    if not f:
        return np.nan
    lr = json.load(gzip.open(f[0], "rt"))
    sr = np.asarray(lr["score_range"])
    wdt = np.asarray(lr["log_lr_plus_p95"], float) - np.asarray(lr["log_lr_plus_p5"], float)
    ok = np.isfinite(wdt)
    return float(np.nanmean(np.interp(x, sr[ok], wdt[ok]))) if ok.any() else np.nan


def ptv_table(dss):
    rows = []
    for ds in dss:
        base = _variant_points("medium_baseline", ds)
        if base is None:
            continue
        r = dict(dataset=ds)
        for run in PTV_RUNS:
            v = _variant_points(f"medium_{run}", ds)
            if v is None:
                continue
            j = base.index.intersection(v.index)
            r[f"shift_{run}"] = float(np.mean(np.abs(v.loc[j].standard_points - base.loc[j].standard_points)))
            r[f"width_{run}"] = _width(f"medium_{run}", ds, base.loc[j].score.values)
        r["width_all"] = _width("medium_baseline", ds, base.score.values)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    fits = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results_first100.json.gz"))
    dss = sorted(fits)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in dss}
    del df_full
    sh = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(delayed(shift_task)(ds, slices[ds], fits[ds]) for ds in dss))
    pt = ptv_table(dss)
    d = sh.merge(pt, on="dataset", how="outer")
    out = ROOT / "analysis_v3" / "shift_and_ptv.csv"
    d.to_csv(out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    ok = d[d.status == "ok"].copy()
    ok["shift_raw"] = ok.ks_p < 0.05
    ok["shift_fit"] = ok.tv_p < 0.05
    print(f"(1) shift tests, {len(ok)} datasets with both missense and non-missense P/LP:")
    print(f"    raw KS p<0.05: {int(ok.shift_raw.sum())}   fitted-weights p<0.05: {int(ok.shift_fit.sum())}   both: "
          f"{int((ok.shift_raw & ok.shift_fit).sum())}")
    print(ok.sort_values("tv_p")[["dataset", "n_mis", "n_other", "ks", "ks_p", "median_shift_iqr", "tv_weights",
                                  "tv_null_median", "tv_p"]].round(3).to_string(index=False))
    print(f"\n(2) PTV runs vs missense-only refit, saved to {out}")
    for run in PTV_RUNS[1:]:
        if f"shift_{run}" not in d:
            continue
        x = d.dropna(subset=[f"shift_{run}", "shift_plpmis"])
        print(f"  {run:24s} shift vs all P/LP: median {x[f'shift_{run}'].median():.3f} (plpmis {x.shift_plpmis.median():.3f})"
              f"; width ratio vs plpmis: median {(x[f'width_{run}'] / x.width_plpmis).median():.3f}")


if __name__ == "__main__":
    main()
