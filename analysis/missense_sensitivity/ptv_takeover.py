"""E4: does adding a PTV sample "take over" the P/LP fit?

Compares the missense run (missense P/LP and B/LB, gnomAD-missense population,
synonymous) with the missense + PTV run (same samples plus a fit-only PTV
sample). Both have the same P/LP sample, so the quality of the fitted P/LP
density can be compared directly:
  * held-out mean log density of P/LP variants under the P/LP mixture, averaged
    over bootstraps (each bootstrap scores only its own held-out P/LP rows);
  * in-sample KS distance between the P/LP empirical CDF and the median fitted
    P/LP CDF across bootstraps;
and related to the PTV/P/LP count ratio and the PTV-vs-P/LP score difference
(two-sample KS). A takeover = the PTV run fits P/LP worse.

Usage:
  python -m analysis.missense_sensitivity.ptv_takeover [--n-jobs 32]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
PLP = "Pathogenic/Likely Pathogenic"
RUNS = {"missense": dict(missense_controls_only=True, population_type="gnomAD_missense_nsSNV"),
        "missense_ptv": dict(missense_controls_only=True, population_type="gnomAD_missense_nsSNV",
                             extra_sample="ptv")}


def _scoreset(ds, df_ds, kw):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    return load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                      clinvar_release=release, **kw))


def _fit_quality(ss, fits_ds, splits_ds):
    from src.assay_calibration.fit_utils.cfusn import density_utils
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    pc = names.index(PLP)
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    plp_rows = np.flatnonzero(sa[:, pc])
    x = np.sort(sc[plp_rows])
    grid = np.linspace(x.min() - 1e-9, x.max() + 1e-9, 400)
    ho, cdfs, n_c = [], [], None
    for seed, by_nc in fits_ds.items():
        rec = next(iter(by_nc.values()))
        n_c = len(rec["fit"]["component_params"])
        p, w = rec["fit"]["component_params"], rec["fit"]["weights"][pc]
        split = splits_ds.get(int(seed)) or {}
        val = split.get("val_variant_indices")
        if val is not None:
            vr = np.intersect1d(np.asarray(val, int), plp_rows)
            if len(vr):
                d = density_utils.joint_densities(sc[vr], p, w).sum(axis=0)
                ho.append(np.mean(np.log(np.clip(d, 1e-300, None))))
        dens = density_utils.joint_densities(grid, p, w).sum(axis=0)
        c = np.cumsum(dens)
        cdfs.append(c / c[-1] if c[-1] > 0 else c)
    med = np.median(np.array(cdfs), axis=0)
    ecdf = np.searchsorted(x, grid, side="right") / len(x)
    return dict(n_plp=len(plp_rows), n_c=n_c,
                heldout_ll=float(np.mean(ho)) if ho else np.nan,
                ks_plp_fit=float(np.max(np.abs(ecdf - med))))


def _task(ds, df_ds, fits, splits):
    from scipy.stats import ks_2samp
    out = dict(dataset=ds)
    for run, kw in RUNS.items():
        if ds not in fits[run]:
            return None
        ss = _scoreset(ds, df_ds, kw)
        names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
        if PLP not in names:
            return None
        q = _fit_quality(ss, fits[run][ds], splits[run].get(ds, {}))
        out.update({f"{k}_{run}": v for k, v in q.items()})
        if run == "missense_ptv":
            sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
            if "ptv" in names:
                ptv = sc[sa[:, names.index("ptv")]]
                plp = sc[sa[:, names.index(PLP)]]
                out["n_ptv"] = len(ptv)
                out["ks_ptv_vs_plp"] = float(ks_2samp(ptv, plp).statistic)
                out["ptv_median_minus_plp_median"] = float(np.median(ptv) - np.median(plp))
            else:
                out["n_ptv"] = 0
    out["d_heldout_ll"] = out["heldout_ll_missense_ptv"] - out["heldout_ll_missense"]
    out["d_ks_plp_fit"] = out["ks_plp_fit_missense_ptv"] - out["ks_plp_fit_missense"]
    out["ptv_plp_ratio"] = out.get("n_ptv", 0) / out["n_plp_missense"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "analysis_v3" / "e4_ptv_takeover.csv"))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    fits = {r: json.load(gzip.open(ROOT / f"explorer_jobs_pp_large_{r}" / "bootstrap_results.json.gz"))
            for r in RUNS}
    splits = {r: pickle.load(open(ROOT / f"explorer_jobs_pp_large_{r}" / "dataset_splits.pkl", "rb"))
              for r in RUNS}
    datasets = sorted(set(fits["missense"]) & set(fits["missense_ptv"]))
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in datasets}
    del df_full
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_task)(ds, slices[ds], {r: {ds: fits[r][ds]} for r in RUNS},
                       {r: {ds: splits[r].get(ds, {})} for r in RUNS}) for ds in datasets)
    df = pd.DataFrame([r for r in res if r])
    df.to_csv(args.out, index=False)
    from scipy.stats import spearmanr
    pd.set_option("display.width", 250)
    has = df[df.n_ptv > 0]
    print(f"{len(df)} datasets, {len(has)} with PTVs")
    print(f"held-out P/LP log density, PTV - missense: median {has.d_heldout_ll.median():+.3f}; "
          f"worse in {(has.d_heldout_ll < -0.05).sum()}, better in {(has.d_heldout_ll > 0.05).sum()}")
    print(f"P/LP CDF misfit (KS), PTV - missense: median {has.d_ks_plp_fit.median():+.3f}; "
          f"worse (>+0.05) in {(has.d_ks_plp_fit > 0.05).sum()}")
    for c in ["ptv_plp_ratio", "ks_ptv_vs_plp"]:
        r1 = spearmanr(has[c], has.d_heldout_ll, nan_policy="omit")
        r2 = spearmanr(has[c], has.d_ks_plp_fit, nan_policy="omit")
        print(f"  {c}: spearman vs d_heldout_ll {r1.statistic:+.2f} (p={r1.pvalue:.3g}); "
              f"vs d_ks {r2.statistic:+.2f} (p={r2.pvalue:.3g})")
    cols = ["dataset", "n_plp_missense", "n_ptv", "ptv_plp_ratio", "ks_ptv_vs_plp", "heldout_ll_missense",
            "heldout_ll_missense_ptv", "d_heldout_ll", "ks_plp_fit_missense", "ks_plp_fit_missense_ptv"]
    print(has.sort_values("d_heldout_ll")[cols].head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
