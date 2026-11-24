"""E9: allele-frequency-aware gnomAD priors on the baseline fits.

The production prior is the label-shift EM over the population sample with every
distinct gnomAD variant weighted equally (per-variant prior). Weighting each
population variant by its allele frequency gives the per-encounter prior: the
probability that a variant met when sequencing a random person is pathogenic.
The fitted P/LP and benign-reference densities are unchanged, so for each
bootstrap fit the weighted EM is

    q <- sum_v a_v r_v(q) / sum_v a_v,   r_v(q) = q f_P(x_v) / (q f_P(x_v) + (1-q) f_B(x_v)).

Variants:
  unweighted    a_v = 1 (reproduces the production estimate, before the discard rule)
  af            a_v = allele frequency
  af_cap1e-3    a_v = min(AF, 1e-3): limits the pull of common variants
  af_lt1e-3     a_v = AF, restricted to AF < 1e-3 (drops near-BS1/BA1 variants)
and AF-stratified priors (unweighted EM within AF bins): singletons-like
(AF < 1e-5), rare (1e-5 to 1e-4), uncommon (1e-4 to 1e-3), common (>= 1e-3).

Reported per dataset: the median over bootstraps (NaN where the EM hits 0 or 1)
and the effective number of population variants, (sum a)^2 / sum a^2.

Usage:  python -m analysis.missense_sensitivity.af_prior [--n-jobs 32]
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
BINS = [(0, 1e-5, "af_lt1e-5"), (1e-5, 1e-4, "af_1e-5_1e-4"), (1e-4, 1e-3, "af_1e-4_1e-3"), (1e-3, 1.01, "af_ge1e-3")]


def weighted_prior_em(f_p, f_b, a, q0=0.5, tol=1e-10, max_iter=10000):
    a = np.asarray(a, float)
    if a.sum() <= 0 or len(a) == 0:
        return np.nan
    q = q0
    for _ in range(max_iter):
        r = q * f_p / np.maximum(q * f_p + (1 - q) * f_b, 1e-300)
        q_new = float((a * r).sum() / a.sum())
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = q_new
    return np.nan if (q <= 0.001 or q >= 0.999) else q


def _row_af(ss, df_ds):
    """Max gnomAD MAF per scoreset row (NaN if none)."""
    maf = []
    for all_idx, (_, variants) in enumerate(ss.get_variants_by_id().items()):
        if ss._keep_mask[all_idx]:
            vals = [getattr(v, "gnomad_MAF", None) for v in variants]
            vals = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
            maf.append(max(vals) if vals else np.nan)
    return np.asarray(maf)


def _task(ds, df_ds, fits_ds, cal):
    from analysis.prior_simulation.simulate_priors import fit_gnomad_densities
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release))
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    if PLP not in names or not ({BLB, SYN} & set(names)):
        return dict(dataset=ds, status="PU/NU skipped")
    gname = "population" if "population" in names else "gnomAD"
    idx = dict(pathogenic_idx=names.index(PLP), benign_idx=names.index(BLB) if BLB in names else None,
               gnomad_idx=names.index(gname), synonymous_idx=names.index(SYN) if SYN in names else None)
    bm = cal.get("benign_method", "avg")
    if idx["synonymous_idx"] is None and bm in ("avg", "synonymous"):
        bm = "benign"
    if idx["benign_idx"] is None and bm in ("avg", "benign"):
        bm = "synonymous"
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    af = _row_af(ss, df_ds)
    g = sa[:, idx["gnomad_idx"]]
    af_g = af[g]
    have = np.isfinite(af_g) & (af_g > 0)
    variants = {
        "unweighted": np.ones(g.sum()),
        "af": np.where(have, af_g, 0.0),
        "af_cap1e-3": np.where(have, np.minimum(af_g, 1e-3), 0.0),
        "af_lt1e-3": np.where(have & (af_g < 1e-3), af_g, 0.0),
    }
    for lo, hi, lab in BINS:
        variants[lab] = (have & (af_g >= lo) & (af_g < hi)).astype(float)
    est = {k: [] for k in variants}
    for rec in fits_ds.values():
        m = next(iter(rec.values()))["fit"]
        try:
            f_p, f_b = fit_gnomad_densities(m, sc, sa, bm, **idx)
        except Exception:
            continue
        for k, a in variants.items():
            est[k].append(weighted_prior_em(f_p, f_b, a))
    out = dict(dataset=ds, status="ok", benign_method=bm, prior_production=cal.get("prior"),
               n_gnomad=int(g.sum()), n_with_af=int(have.sum()))
    for k, a in variants.items():
        v = np.asarray(est[k], float)
        out[f"prior_{k}"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        out[f"neff_{k}"] = float(a.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "analysis_v3" / "e9_af_prior.csv"))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.compare_runs import _find_outputs
    fits = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results.json.gz"))
    datasets = sorted(fits)
    cals = {}
    for ds in datasets:
        c, _ = _find_outputs(ROOT / "calib_baseline", ds)
        if c is not None:
            cals[ds] = json.load(open(c))
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in cals}
    del df_full
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_task)(ds, slices[ds], fits[ds], cals[ds]) for ds in cals)
    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    ok = df[df.status == "ok"]
    pd.set_option("display.width", 250)
    print(f"{len(ok)} datasets (PU/NU skipped: {(df.status != 'ok').sum()})")
    cols = [c for c in ok.columns if c.startswith("prior_")]
    print(ok[cols].describe(percentiles=[.25, .5, .75]).loc[["count", "25%", "50%", "75%"]].round(4).to_string())
    print(ok[[c for c in ok.columns if c.startswith("neff_")]].median().round(1).to_string())


if __name__ == "__main__":
    main()
