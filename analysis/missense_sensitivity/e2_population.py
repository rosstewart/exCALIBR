"""E2: how the prior population changes the prior and missense evidence.

Compares the baseline calibration (prior over all gnomAD variants in the
dataset) with recalibrations of the SAME baseline fits whose prior population
is (a) gnomAD missense SNVs or (b) all assayed missense SNVs
(run_igvf_batch.py --population-type ...). lr+ is unchanged; only the prior,
and through it the evidence thresholds, move.

The population runs have no OOB evidence (the row set changes, so the OOB
splits no longer index the scoreset), so every run is scored here on IN-BAG
standard_points, on the same missense controls and VUS as compare_runs.py.

Usage:
  python -m analysis.missense_sensitivity.e2_population [--root ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.missense_sensitivity.compare_runs import (
    PLP, BLB, _find_outputs, _metrics, _thresholds)

RUNS = {"all_gnomAD": "calib_baseline",
        "gnomAD_missense": "calib_baseline_pop_gnomAD_missense_nsSNV",
        "all_missense": "calib_baseline_pop_all_missense_nsSNV"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ross/assay_calibration/missense_only_calibrations")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-jobs", type=int, default=32)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or root / "analysis_v2")
    out.mkdir(parents=True, exist_ok=True)
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.compare_runs import _missense_map

    datasets = sorted(p.name for p in (root / RUNS["all_gnomAD"]).iterdir()
                      if p.is_dir() and _find_outputs(root / RUNS["all_gnomAD"], p.name)[1])
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in datasets}
    del df_full
    maps = dict(zip(datasets, Parallel(n_jobs=args.n_jobs)(
        delayed(_missense_map)(ds, slices[ds]) for ds in datasets)))

    rows = []
    for ds in datasets:
        cal_b, var_b = _find_outputs(root / RUNS["all_gnomAD"], ds)
        dfb = pd.read_csv(var_b)
        s = dfb["sample"].fillna("")
        mis = maps[ds].set_index("variant_id")["is_missense"].reindex(dfb["variant_id"]).fillna(False).values
        plp_ids = dfb.loc[mis & s.str.contains(PLP, regex=False).values, "variant_id"]
        blb_ids = dfb.loc[mis & s.str.contains(BLB, regex=False).values, "variant_id"]
        vus_ids = dfb.loc[mis & dfb["is_vus"].fillna(False).astype(bool).values
                          & ~s.str.contains(PLP, regex=False).values
                          & ~s.str.contains(BLB, regex=False).values, "variant_id"]
        for run, d in RUNS.items():
            cal_p, var_p = _find_outputs(root / d, ds)
            if var_p is None:
                continue
            cal = json.load(open(cal_p))
            df = pd.read_csv(var_p)
            pts = df.set_index("variant_id")["standard_points"].astype(float)
            p = pts.reindex(plp_ids).dropna()
            b = pts.reindex(blb_ids).dropna()
            v = pts.reindex(vus_ids).dropna()
            m = _metrics(int((p > 0).sum()), int((b > 0).sum()), int((b < 0).sum()),
                         int((p < 0).sum()), int((p == 0).sum()), int((b == 0).sum()))
            pmax, bmin = _thresholds(cal)
            m.update(dataset=ds, run=run, prior=cal.get("prior"), benign_method=cal.get("benign_method"),
                     max_path_pts=pmax, max_benign_pts=bmin, n_vus=len(v),
                     vus_path=float((v > 0).mean()) if len(v) else np.nan,
                     vus_benign=float((v < 0).mean()) if len(v) else np.nan,
                     vus_det=float((v != 0).mean()) if len(v) else np.nan,
                     plp_mean_pts=float(p.mean()) if len(p) else np.nan,
                     vus_mean_path_pts=float(v.clip(lower=0).mean()) if len(v) else np.nan)
            rows.append(m)
    res = pd.DataFrame(rows)
    res.to_csv(out / "e2_population_per_dataset.csv", index=False)

    piv = res.pivot(index="dataset", columns="run", values="prior")
    print(f"{piv.dropna().shape[0]} datasets with all three runs")
    print("\nPrior (median [IQR]) by population:")
    for r in RUNS:
        q = piv[r].quantile([.25, .5, .75])
        print(f"  {r:16s} {q[.5]:.3f} [{q[.25]:.3f}, {q[.75]:.3f}]")
    for r in ["gnomAD_missense", "all_missense"]:
        d = (piv[r] - piv["all_gnomAD"]).dropna()
        ratio = (piv[r] / piv["all_gnomAD"]).dropna()
        print(f"  {r} - all_gnomAD: median {d.median():+.3f}, ratio median {ratio.median():.2f}, "
              f"up {int((d > 0.005).sum())} / down {int((d < -0.005).sum())} of {len(d)}")
    cols = ["mcc", "accuracy", "determinate_rate", "plp_correct_all", "blb_correct_all",
            "vus_det", "vus_path", "vus_benign"]
    print("\nPer-dataset medians (in-bag points):")
    print(res.groupby("run")[cols].median().loc[list(RUNS)].round(3).to_string())
    print(f"\nWrote {out / 'e2_population_per_dataset.csv'}")


if __name__ == "__main__":
    main()
