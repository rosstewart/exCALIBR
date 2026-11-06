"""Post-fit missense-only P/LP sample for the baseline fits.

For every baseline bootstrap fit, keep the fitted components (and every other
sample's weights) fixed and re-estimate only the P/LP sample's mixture
weights (weights-only EM with frozen component_params, run to convergence,
initialised at the fitted all-variant-type P/LP weights) from that
bootstrap's own P/LP training rows restricted to missense.

The training rows are reproduced exactly as Fit.run built them (master seed
0, so the composition seed is the bootstrap seed): makeOneHot assigns each
multi-label variant to one sample, then sample_specific_bootstrap resamples
each sample with replacement. Rows keep their resampling multiplicity, and
only rows the one-hot assigned to P/LP are used. Checks per bootstrap: the
reproduced held-out rows equal the splits pickle's val_variant_indices, and
re-estimating from ALL reproduced P/LP training rows returns the fitted P/LP
weights (reported as reproduce_tv).

The output is a bootstrap-fits file with the same structure as the input, so
the ordinary calibration (run_igvf_batch.py --precomputed-fits) produces the
"pathogenic-missense" lr+, prior and point ranges. Combine with the standard
calibration as: pathogenic points from this run, benign points from standard
(compare_runs.py, run baseline_plpmis).

Usage:
  python -m analysis.missense_sensitivity.reweight_plp_missense \
      --fits   .../explorer_jobs_pp_large_baseline/bootstrap_results.json.gz \
      --splits .../explorer_jobs_pp_large_baseline/dataset_splits.pkl \
      --out    .../explorer_jobs_pp_large_baseline/bootstrap_results_plp_missense.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

PLP = "Pathogenic/Likely Pathogenic"


def _em_weights(x, params, w0, max_iter=5000, tol=1e-10):
    from src.assay_calibration.fit_utils.cfusn import density_utils
    w = np.asarray(w0, dtype=float)
    w = np.clip(w, 1e-12, None)
    w /= w.sum()
    for _ in range(max_iter):
        post = density_utils.component_posteriors(x, params, w)
        w_new = np.asarray(post).mean(1)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def _process(dataset, df_ds, fits_ds, splits_ds):
    from analysis.missense_sensitivity.compare_runs import _missense_map
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df

    release = "2018" if dataset.endswith("_clinvar_2018") else "2025"
    pcfg = PipelineConfig(dataset_csv="", dataset_name=dataset, output_dir="/tmp",
                          clinvar_release=release)
    ss = load_dataset_from_df(df_ds, pcfg)
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    if PLP not in names:
        return dataset, fits_ds, dict(dataset=dataset, status="no P/LP sample")
    p_col = names.index(PLP)                       # index into fit weights
    sa = np.asarray(ss.sample_assignments)
    scores = np.asarray(ss.scores)
    mis = _missense_map(dataset, df_ds)["is_missense"].values
    assert len(mis) == len(scores)
    plp_mis = sa[:, p_col].astype(bool) & mis
    n_plp, n_plp_mis = int(sa[:, p_col].sum()), int(plp_mis.sum())
    from src.assay_calibration.fit_utils.fit import makeOneHot, sample_specific_bootstrap
    out, n_fallback, shifts, repro, n_val_mismatch = {}, 0, [], [], 0
    for seed, by_nc in fits_ds.items():
        rec0 = next(iter(by_nc.values()))
        b = int(rec0.get("bootstrap_seed", seed))
        onehot = makeOneHot(sa, rng=np.random.RandomState(b))
        include = onehot.any(axis=1) & ~np.isnan(scores)
        orig = np.flatnonzero(include)
        tr, va = sample_specific_bootstrap(onehot[include], b)
        split = splits_ds.get(int(seed))
        if split is not None and split.get("val_variant_indices") is not None:
            if not np.array_equal(np.sort(orig[va]), np.sort(np.asarray(split["val_variant_indices"], int))):
                n_val_mismatch += 1
        rows = orig[tr]                                  # with multiplicity
        rows_p = rows[onehot[rows, p_col]]               # assigned to P/LP
        rows_pm = rows_p[mis[rows_p]]
        out[seed] = {}
        for nc, rec in by_nc.items():
            rec = json.loads(json.dumps(rec))            # deep copy
            w = rec["fit"]["weights"]
            params = rec["fit"]["component_params"]
            if len(rows_p):
                w_all = _em_weights(scores[rows_p], params, w[p_col])
                repro.append(float(np.abs(w_all - np.asarray(w[p_col])).sum() / 2))
            if len(rows_pm) == 0:
                n_fallback += 1                          # keep fitted all-type weights
            else:
                new = _em_weights(scores[rows_pm], params, w[p_col])
                shifts.append(float(np.abs(new - np.asarray(w[p_col])).sum() / 2))
                w[p_col] = new.tolist()
            out[seed][nc] = rec
    return dataset, out, dict(dataset=dataset, status="ok", n_plp=n_plp, n_plp_missense=n_plp_mis,
                              n_bootstraps=len(fits_ds), n_no_train_missense=n_fallback,
                              n_val_mismatch=n_val_mismatch,
                              reproduce_tv_max=float(np.max(repro)) if repro else np.nan,
                              median_tv_shift=float(np.median(shifts)) if shifts else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-tsv", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    import pandas as pd
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df

    fits = json.load(gzip.open(args.fits))
    splits = pickle.load(open(args.splits, "rb"))
    tsv = args.dataset_tsv or _cfg.DATASET_TSV
    df_full = load_master_df(tsv)
    slices = {ds: _filter_dataset_df(df_full, ds, tsv) for ds in fits}
    del df_full
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process)(ds, slices[ds], fits[ds], splits.get(ds, {})) for ds in sorted(fits))
    new = {ds: f for ds, f, _ in res}
    with gzip.open(args.out, "wt") as fh:
        json.dump(new, fh)
    summ = pd.DataFrame([s for _, _, s in res])
    summ.to_csv(args.out.replace(".json.gz", "_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
