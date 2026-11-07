"""E5: make TT3 section 3.1's "overlap" conditions testable, per dataset.

(a) Synonymous as benign controls: do synonymous variants look like the benign
    reference? Weight level (baseline fits): per bootstrap, the difference in
    abnormal-component weight between the synonymous and B/LB samples,
    d_syn = w_S[k*] - w_B[k*], where k* is the component most enriched in P/LP
    relative to B/LB (median over bootstraps). Score level: two-sample KS between
    missense B/LB and synonymous scores.
(b) PTVs as pathogenic controls: do PTVs look like missense P/LP? Weight level
    (missense + PTV fits): d_ptv = w_PTV[k*] - w_P[k*]; score level: KS between
    PTV and missense P/LP scores.
(c) Splicing-pathogenic synonymous variants (assays that detect splicing): the
    share of synonymous variants with SpliceAI delta score > 0.2 (eta_S proxy) and
    their score shift towards abnormal, in units of the low-SpliceAI synonymous
    spread.

A weight-level difference whose 95% bootstrap interval excludes 0 and exceeds
an equivalence margin (default 0.05) flags the sample as not exchangeable.

Usage:
  python -m analysis.missense_sensitivity.eligibility [--margin 0.05] [--n-jobs 32]
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
PLP, BLB, SYN = "Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "Synonymous"
SAI = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]


def _scoreset(ds, df_ds, **kw):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    return load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                      clinvar_release=release, **kw))


def _weight_diff(fits_ds, names, a, b, ref_p=PLP, ref_b=BLB):
    """Per-bootstrap w_a[k*] - w_b[k*]; k* = argmax median(w_refP - w_refB)."""
    if any(n not in names for n in (a, b, ref_p, ref_b)):
        return None
    W = np.array([np.asarray(next(iter(r.values()))["fit"]["weights"]) for r in fits_ds.values()])
    ip, ib, ia, ibb = (names.index(n) for n in (ref_p, ref_b, a, b))
    k = int(np.argmax(np.median(W[:, ip, :] - W[:, ib, :], axis=0)))
    d = W[:, ia, k] - W[:, ibb, k]
    return dict(median=float(np.median(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


def _task(ds, df_ds, fits_base, fits_ptv, margin):
    from scipy.stats import ks_2samp
    from analysis.missense_sensitivity.compare_runs import _missense_map
    out = dict(dataset=ds, splice_measure=str(df_ds["splice_measure"].iloc[0]))
    ss = _scoreset(ds, df_ds)
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    mis = _missense_map(ds, df_ds)["is_missense"].values
    # (a) synonymous vs B/LB
    wd = _weight_diff(fits_base, names, SYN, BLB)
    if wd:
        out.update({f"d_syn_{k}": v for k, v in wd.items()})
        out["syn_flag"] = bool((wd["lo"] > margin) or (wd["hi"] < -margin))
    if SYN in names and BLB in names:
        bm = sc[sa[:, names.index(BLB)] & mis]
        sy = sc[sa[:, names.index(SYN)]]
        out["n_blb_missense"], out["n_syn"] = len(bm), len(sy)
        if len(bm) >= 3 and len(sy) >= 3:
            t = ks_2samp(bm, sy)
            out["ks_blbmis_vs_syn"], out["ks_blbmis_vs_syn_p"] = float(t.statistic), float(t.pvalue)
    # (b) PTV vs missense P/LP (missense + PTV run)
    if fits_ptv is not None:
        sp = _scoreset(ds, df_ds, missense_controls_only=True, population_type="gnomAD_missense_nsSNV",
                       extra_sample="ptv")
        pn = [n for n, c in zip(sp.sample_names, sp.sample_counts) if c > 0]
        wd = _weight_diff(fits_ptv, pn, "ptv", PLP, ref_b=BLB if BLB in pn else SYN)
        if wd:
            out.update({f"d_ptv_{k}": v for k, v in wd.items()})
            out["ptv_flag"] = bool((wd["lo"] > margin) or (wd["hi"] < -margin))
        if "ptv" in pn and PLP in pn:
            psa, psc = np.asarray(sp.sample_assignments), np.asarray(sp.scores)
            pt, pl = psc[psa[:, pn.index("ptv")]], psc[psa[:, pn.index(PLP)]]
            out["n_ptv"], out["n_plp_missense"] = len(pt), len(pl)
            if len(pt) >= 3 and len(pl) >= 3:
                t = ks_2samp(pt, pl)
                out["ks_ptv_vs_plpmis"], out["ks_ptv_vs_plpmis_p"] = float(t.statistic), float(t.pvalue)
    # (c) splicing-pathogenic synonymous
    d = df_ds.copy()
    syn = d["most_severe_mutational_consequence"].astype(str).str.contains("synonymous")
    if syn.sum() >= 5 and all(c in d for c in SAI):
        sai = d.loc[syn, SAI].astype(float).max(axis=1)
        s = d.loc[syn, "auth_reported_score"].astype(float)
        hi, lo = s[sai > 0.2].dropna(), s[sai <= 0.2].dropna()
        out["n_syn_raw"], out["syn_high_sai_frac"] = int(syn.sum()), float((sai > 0.2).mean())
        # abnormal direction from P/LP vs B/LB medians in the scoreset
        if PLP in names and BLB in names and len(hi) and len(lo) > 3:
            sign = np.sign(np.median(sc[sa[:, names.index(PLP)]]) - np.median(sc[sa[:, names.index(BLB)]]))
            iqr = np.subtract(*np.percentile(lo, [75, 25])) or np.nan
            out["syn_high_sai_shift_iqr"] = float(sign * (np.median(hi) - np.median(lo)) / iqr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "analysis_v3" / "e5_eligibility.csv"))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    fb = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results.json.gz"))
    fp = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_missense_ptv" / "bootstrap_results.json.gz"))
    datasets = sorted(fb)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in datasets}
    del df_full
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_task)(ds, slices[ds], fb[ds], fp.get(ds), args.margin) for ds in datasets)
    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(f"{len(df)} datasets")
    if "syn_flag" in df:
        s = df.dropna(subset=["d_syn_median"])
        print(f"(a) synonymous vs B/LB abnormal-component weight: {len(s)} datasets; flagged "
              f"{int(s.syn_flag.sum())} (margin {args.margin}); median d {s.d_syn_median.median():+.3f}")
    if "ptv_flag" in df:
        s = df.dropna(subset=["d_ptv_median"])
        print(f"(b) PTV vs missense P/LP: {len(s)} datasets; flagged {int(s.ptv_flag.sum())}; "
              f"median d {s.d_ptv_median.median():+.3f}")
    s = df.dropna(subset=["syn_high_sai_frac"]) if "syn_high_sai_frac" in df else df.iloc[:0]
    for sm, g in s.groupby("splice_measure"):
        print(f"(c) splice_measure={sm}: {len(g)} datasets; median share of synonymous with SpliceAI>0.2 "
              f"{g.syn_high_sai_frac.median():.3f}; median abnormal shift {g.get('syn_high_sai_shift_iqr', pd.Series()).median():+.2f} IQR")


if __name__ == "__main__":
    main()
