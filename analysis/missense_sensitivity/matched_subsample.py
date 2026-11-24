"""E1: is the evidence change from missense-only P/LP a count effect or a composition effect?

For each dataset with non-missense P/LP, re-estimate the P/LP mixing weights of
every baseline bootstrap fit (components fixed, exact one-hot + resampled
training rows; see reweight_plp_missense.py) from
  * "missense": the training P/LP rows that are missense;
  * "subset_r" (r = 0..R-1): the training P/LP rows that fall in a fixed random
    subset of the dataset's P/LP rows of size n_missense (any variant type);
  * "all": every training P/LP row (sanity check: reproduces the baseline).
Each arm is calibrated in-process with the production code path
(run_igvf_batch._run_one_combo, the dataset's baseline config), and scored on
IN-BAG points for the dataset's missense P/LP and missense VUS.

If the missense arm falls inside the random-subset distribution, the change is
explained by control count; if outside, by composition.

Usage:
  python -m analysis.missense_sensitivity.matched_subsample [--n-subsets 10] [--n-jobs 64]
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
JOBS = Path("/data/ross/assay_calibration/missense_only_calibrations/explorer_jobs_pp_large_baseline")
PLP = "Pathogenic/Likely Pathogenic"
# calibration settings for every arm (set from the CLI in main; module-level so joblib workers see them
# through the task arguments below)
CALIB_OPTS = {"population_type": "gnomAD", "acmg_mapping_method": "tavtigian"}


def training_rows(sa, scores, seed):
    """(rows with multiplicity, one-hot matrix) exactly as Fit.run built bootstrap `seed`
    (master seed 0)."""
    from src.assay_calibration.fit_utils.fit import makeOneHot, sample_specific_bootstrap
    onehot = makeOneHot(sa, rng=np.random.RandomState(seed))
    include = onehot.any(axis=1) & ~np.isnan(scores)
    orig = np.flatnonzero(include)
    tr, _ = sample_specific_bootstrap(onehot[include], seed)
    return orig[tr], onehot


def _calibrate(ds, df_ds, fits, seeds, n_c, benign_method, overrides, release, tmp, keep_dir=None):
    import run_igvf_batch as rib
    args = Namespace(dataset="", min_clinvar_star=1, population_type=CALIB_OPTS["population_type"], manual_prior=None,
                     acmg_mapping_method=CALIB_OPTS["acmg_mapping_method"])
    out = Path(tempfile.mkdtemp(dir=tmp))
    try:
        rib._run_one_combo(
            n_c, benign_method, fits, np.asarray(seeds), df_ds, ds, release, str(out), args, 1,
            liberal_monotonicity=overrides.get("liberal_monotonicity", True),
            postprocess_point_ranges=overrides.get("postprocess_point_ranges", True))
        cal = json.load(open(next(out.glob("*_calibration.json"))))
        if keep_dir is not None:  # keep the calibration, LR curves and variant table for per-dataset figures
            Path(keep_dir).mkdir(parents=True, exist_ok=True)
            for f in list(out.glob("*_calibration.json")) + list(out.glob("*_lr_values.json.gz")) \
                    + list(out.glob("*_variants.csv")):
                shutil.copy2(f, Path(keep_dir) / f.name)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return cal


def _score(cal, scores_plp, scores_vus):
    from src.assay_calibration.fit_utils.fit import assign_points
    pr = {int(k): v for k, v in (cal.get("point_ranges") or {}).items()}
    p = assign_points(np.asarray(scores_plp, float), pr) if len(scores_plp) else np.array([])
    v = assign_points(np.asarray(scores_vus, float), pr) if len(scores_vus) else np.array([])
    lev = [k for k, r in pr.items() if r and k > 0]
    return dict(prior=cal.get("prior"), max_path=max(lev, default=0),
                plp_path_frac=float((p > 0).mean()) if len(p) else np.nan,
                plp_mean_path=float(np.clip(p, 0, None).mean()) if len(p) else np.nan,
                plp_ben_frac=float((p < 0).mean()) if len(p) else np.nan,
                vus_path_frac=float((v > 0).mean()) if len(v) else np.nan,
                vus_mean_path=float(np.clip(v, 0, None).mean()) if len(v) else np.nan)


def _dataset_task(ds, df_ds, fits_ds, cfg_entry, n_subsets, tmp, keep_root=None, calib_opts=None):
    if calib_opts:
        CALIB_OPTS.update(calib_opts)
    import run_igvf_batch as rib
    from analysis.missense_sensitivity.compare_runs import _missense_map
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df

    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    n_c, benign_method, overrides = rib.parse_dataset_config(cfg_entry)
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release))
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    if PLP not in names:
        return []
    pc = names.index(PLP)
    sa, scores = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    mmap = _missense_map(ds, df_ds)
    mis = mmap["is_missense"].values
    is_plp = sa[:, pc].astype(bool)
    n_mis = int((is_plp & mis).sum())
    n_all = int(is_plp.sum())
    if n_mis == n_all or n_mis == 0:
        return [dict(dataset=ds, arm="skip", n_plp=n_all, n_plp_missense=n_mis)]
    # scoring set: missense P/LP and missense VUS (VUS rows are not in the scoreset
    # samples, so take them from the baseline variants CSV)
    var = next((ROOT / "calib_baseline" / ds).glob(f"{ds}_*_variants.csv"))
    dv = pd.read_csv(var)
    mis_ids = set(mmap.loc[mmap.is_missense, "variant_id"])
    s = dv["sample"].fillna("")
    vus = dv[dv.variant_id.isin(mis_ids) & dv.is_vus.fillna(False).astype(bool)
             & ~s.str.contains(PLP, regex=False) & ~s.str.contains("Benign/Likely Benign", regex=False)]
    sc_plp, sc_vus = scores[is_plp & mis], vus["score"].values

    seeds = sorted(fits_ds, key=int)
    recs = [fits_ds[k][n_c] for k in seeds if fits_ds[k].get(n_c)]
    seeds = [int(k) for k in seeds if fits_ds[k].get(n_c)]
    rows_per_seed = [training_rows(sa, scores, b) for b in seeds]

    rng = np.random.default_rng(12345)
    plp_rows = np.flatnonzero(is_plp)
    arms = {"all": is_plp, "missense": is_plp & mis}
    for r in range(n_subsets):
        m = np.zeros(len(scores), bool)
        m[rng.choice(plp_rows, size=n_mis, replace=False)] = True
        arms[f"subset_{r}"] = m
    out = []
    for arm, member in arms.items():
        new_fits = []
        for rec, (rows, onehot) in zip(recs, rows_per_seed):
            rec = json.loads(json.dumps(rec))
            rp = rows[onehot[rows, pc] & member[rows]]
            if len(rp):
                rec["fit"]["weights"][pc] = _em_weights(
                    scores[rp], rec["fit"]["component_params"], rec["fit"]["weights"][pc]).tolist()
            new_fits.append(rec)
        try:
            cal = _calibrate(ds, df_ds, new_fits, seeds, n_c, benign_method, overrides, release, tmp,
                             keep_dir=(Path(keep_root) / ds / arm) if keep_root else None)
            row = _score(cal, sc_plp, sc_vus)
        except Exception as e:  # keep the batch going
            row = dict(error=repr(e)[:200])
        row.update(dataset=ds, arm=arm, n_plp=n_all, n_plp_missense=n_mis)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subsets", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=64)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--config", default="src/igvf_configs/dataset_configs_aug_2026.json")
    ap.add_argument("--out", default=str(ROOT / "analysis_v3" / "e1_matched_subsample.csv"))
    ap.add_argument("--population-type", default="gnomAD",
                    help="prior population for every arm (e.g. gnomAD_missense_nsSNV for missense evidence)")
    ap.add_argument("--acmg-mapping-method", default="tavtigian", choices=["tavtigian", "lp_anchored"])
    ap.add_argument("--fits", default=str(JOBS / "bootstrap_results.json.gz"),
                    help="bootstrap fits to re-weight (e.g. bootstrap_results_first100.json.gz to match the medium refits)")
    ap.add_argument("--keep-root", default=None,
                    help="keep each arm's calibration outputs under <keep-root>/<dataset>/<arm>/")
    ap.add_argument("--tmp", default="/tmp/claude-9996/-home-rcstewart-exCALIBR/"
                    "32036f78-48fd-4171-a682-ba397e58233c/scratchpad/e1_tmp")
    args = ap.parse_args()
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df

    fits = json.load(gzip.open(args.fits))
    cfg = json.load(open(args.config))
    datasets = sorted(args.datasets or fits)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in datasets}
    del df_full

    def _cfg_for(ds):
        return cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", ""))

    parts = Path(args.out.replace(".csv", "_parts"))
    parts.mkdir(parents=True, exist_ok=True)

    def _run(ds):
        f = parts / f"{ds}.csv"
        if f.exists():
            return
        rows = _dataset_task(ds, slices[ds], fits[ds], _cfg_for(ds), args.n_subsets, args.tmp, args.keep_root,
                             dict(population_type=args.population_type, acmg_mapping_method=args.acmg_mapping_method))
        pd.DataFrame(rows).to_csv(f, index=False)

    todo = [ds for ds in datasets if _cfg_for(ds) is not None and not (parts / f"{ds}.csv").exists()]
    print(f"{len(todo)} datasets to run ({len(datasets) - len(todo)} already done)")
    Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(_run)(ds) for ds in todo)
    # datasets without P/LP write an empty part file
    df = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("*.csv")) if f.stat().st_size > 2], ignore_index=True)
    df = df[df.arm != "skip"]
    df.to_csv(args.out, index=False)

    # missense arm vs random-subset distribution, per dataset
    sub = df[df.arm.str.startswith("subset_")]
    summ = []
    for ds, g in df[df.arm.isin(["all", "missense"])].groupby("dataset"):
        sg = sub[sub.dataset == ds]
        row = dict(dataset=ds, n_plp=g.n_plp.iloc[0], n_plp_missense=g.n_plp_missense.iloc[0])
        for m in ["plp_mean_path", "vus_mean_path", "prior", "max_path"]:
            mv = g.loc[g.arm == "missense", m].values
            av = g.loc[g.arm == "all", m].values
            sv = sg[m].dropna().values
            if not len(mv) or not len(sv):
                continue
            row[f"{m}_all"] = av[0] if len(av) else np.nan
            row[f"{m}_missense"] = mv[0]
            row[f"{m}_subset_median"] = np.median(sv)
            # two-sided empirical percentile of the missense arm among the subsets
            row[f"{m}_pct"] = float((sv < mv[0]).mean() + 0.5 * (sv == mv[0]).mean())
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
