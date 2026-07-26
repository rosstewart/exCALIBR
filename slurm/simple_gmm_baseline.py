#!/usr/bin/env python3
"""
Simple two-component GMM baseline — run locally (no SLURM, no bootstrap,
no restarts), across all datasets.

Fits a single deterministic sklearn GaussianMixture per dataset per variant,
then computes log LR+ and maps it to ACMG evidence points using the same
downstream machinery as run_pipeline.py / run_igvf_batch.py (calculate_score_ranges,
compute_variant_table, save_results) — so outputs are directly comparable to the
main pipeline's. Unlike the main pipeline, point_ranges here is calculate_score_ranges's
raw output: no enforce_monotonicity_point_ranges, no extend_points_to_xlims:

    <output_dir>/<dataset_name>/<dataset_name>_<variant>_calibration.json
    <output_dir>/<dataset_name>/<dataset_name>_<variant>_lr_values.json.gz
    <output_dir>/<dataset_name>/<dataset_name>_<variant>_variants.csv

sample_num convention: 0 = P/LP, 1 = B/LB, 2 = gnomAD/population, 3 = Synonymous.

Two pooling variants per dataset (mirrors plot_four_datasets_gmm_scores(mode=
'plp_blb') in src/assay_calibration/plot_utils/utils.py):
  plp_blb        P/LP (0) + B/LB (1)
  plp_blb_synon  P/LP (0) + [B/LB (1) UNION Synonymous (3)]

Neither variant uses a gnomAD/population sample, so there is no population-based
prior estimation here (see get_fit_prior in fit_utils/point_ranges.py, which
requires a population sample and is not called by this script). The prior is
instead a fixed hyperparameter (default 0.1) passed straight to
calculate_score_ranges — no bootstrap variance, no restarts.

Usage
-----
  python slurm/simple_gmm_baseline.py --output-dir DIR [--dataframe F] [--n-jobs N] [--prior P]
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SAMPLE_NUM_PLP = 0
SAMPLE_NUM_BLB = 1
SAMPLE_NUM_SYNON = 3

_DEFAULT_DATAFRAME = (
    "/data/ross/assay_calibration/dataframe/"
    "integrated_variant_effect_dataset_merged_89datasets.tsv.gz"
)
_DEFAULT_PRIOR = 0.1


def _fit_variant(scoreset, dataset_name, variant, prior, score_range_points,
                  point_values, acmg_mapping_method):
    from sklearn.mixture import GaussianMixture
    from src.assay_calibration.plot_utils.utils import (
        _col_idx_for_sample_num, _update_weights,
    )
    from src.assay_calibration.fit_utils.cfusn.density_utils import mixture_pdf
    from src.assay_calibration.fit_utils.fit import calculate_score_ranges

    plp_col = _col_idx_for_sample_num(scoreset, SAMPLE_NUM_PLP)
    if plp_col is None:
        return {"dataset_name": dataset_name, "variant": variant,
                "status": "skipped", "reason": "no P/LP sample"}

    blb_col = _col_idx_for_sample_num(scoreset, SAMPLE_NUM_BLB)
    synon_col = _col_idx_for_sample_num(scoreset, SAMPLE_NUM_SYNON)
    plp_scores = scoreset.scores[scoreset.sample_assignments[:, plp_col]]

    if variant == "plp_blb":
        if blb_col is None:
            return {"dataset_name": dataset_name, "variant": variant,
                     "status": "skipped", "reason": "no B/LB sample"}
        benign_scores = scoreset.scores[scoreset.sample_assignments[:, blb_col]]
    elif variant == "plp_blb_synon":
        parts = []
        if blb_col is not None:
            parts.append(scoreset.scores[scoreset.sample_assignments[:, blb_col]])
        if synon_col is not None:
            parts.append(scoreset.scores[scoreset.sample_assignments[:, synon_col]])
        if not parts:
            return {"dataset_name": dataset_name, "variant": variant,
                     "status": "skipped", "reason": "no B/LB or Synonymous sample"}
        benign_scores = np.concatenate(parts)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # --- Single deterministic 2-component GMM fit (no bootstrap, no restarts) ---
    combined = np.concatenate([plp_scores, benign_scores]).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                           random_state=42, n_init=10)
    gmm.fit(combined)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_[:, 0, 0])

    # GaussianMixture's component order is arbitrary; relabel deterministically
    # so component 0 is always "pathogenic-like" — the one closer to the
    # P/LP-only mean.
    plp_mean = plp_scores.mean()
    path_idx = int(np.argmin(np.abs(means - plp_mean)))
    order = [path_idx, 1 - path_idx]
    means, stds = means[order], stds[order]
    # skewnorm canonical (a, loc, scale) with a=0 == plain Gaussian; reuses the
    # existing density machinery (mixture_pdf) unmodified.
    params = [(0.0, float(means[0]), float(stds[0])),
              (0.0, float(means[1]), float(stds[1]))]

    # Per-sample mixing weights, re-estimated via fixed-parameter EM (same
    # helper plot_four_datasets_gmm_scores uses for its overlay curves).
    w_plp = _update_weights(plp_scores, means, stds)
    w_benign = _update_weights(benign_scores, means, stds)

    # Per-*present-sample* weights in scoreset.samples order (P/LP, B/LB,
    # gnomAD, Synonymous — whichever exist), needed by plot_scoreset_best_config
    # / sample_density, which index fit['fit']['weights'][sample_column_idx]
    # for every present sample, not just the two pooled into this variant's fit.
    per_sample_weights = [
        _update_weights(sample_scores, means, stds)
        for sample_scores, _sample_name in scoreset.samples
    ]
    # xlims gates sample_density's NaN-masking (plot_utils/utils.py:sample_density) —
    # in the real pipeline it's per-bootstrap-fit train-range, used to hide density
    # outside what that particular fit was actually trained on. There's no such
    # gap to guard here (one fit, no gnomAD involved in fitting), so use the full
    # observed score range — same bounds as score_range — rather than the pooled
    # P/LP+benign fitting range, or samples like gnomAD that extend beyond it
    # (which never entered the fit) would get masked out of their own panel.
    observed_scores = scoreset.scores[scoreset.sample_assignments.any(1)]
    fit_xlims = (float(observed_scores.min()), float(observed_scores.max()))
    single_fit_obj = {"fit": {
        "component_params": params,
        "weights": per_sample_weights,
        "xlims": fit_xlims,
    }}

    score_range = np.linspace(observed_scores.min(), observed_scores.max(),
                               score_range_points)

    log_fp = mixture_pdf(score_range, params, w_plp)
    log_fb = mixture_pdf(score_range, params, w_benign)
    log_lr_plus = log_fp - log_fb

    point_ranges_p, point_ranges_b, C = calculate_score_ranges(
        log_lr_plus, log_lr_plus, prior, score_range, point_values,
        acmg_mapping_method=acmg_mapping_method,
    )
    point_ranges = {**point_ranges_p, **point_ranges_b}

    scoreset_flipped = bool(plp_scores.mean() > benign_scores.mean())

    # No post-processing (no enforce_monotonicity_point_ranges, no
    # extend_points_to_xlims): point_ranges is the raw output of
    # calculate_score_ranges's per-score-point threshold crossing, unlike
    # generate_visualizations's non-acmg_bayes (point-based) branch which does both.

    calibration = {
        "prior": float(prior),
        "priors": [float(prior)],
        "point_ranges": point_ranges,
        "score_range": score_range,
        "log_lr_plus": log_lr_plus.reshape(1, -1),
        "log_fp": log_fp.reshape(1, -1),
        "log_fb": log_fb.reshape(1, -1),
        "C": C,
        "scoreset_flipped": scoreset_flipped,
        "acmg_mapping_method": acmg_mapping_method,
    }

    return {
        "dataset_name": dataset_name,
        "variant": variant,
        "status": "fit",
        "calibration": calibration,
        "single_fit": single_fit_obj,
        "pathogenic_mean": float(means[0]), "pathogenic_std": float(stds[0]),
        "benign_mean": float(means[1]), "benign_std": float(stds[1]),
        "n_plp": int(len(plp_scores)), "n_benign_pool": int(len(benign_scores)),
    }


def _generate_visualization(result, scoreset, config, logger, output_dir):
    """Render <dataset_name>_<variant>_visualization.png via the exact same
    plot_scoreset_best_config call generate_visualizations() uses
    (pipeline/visualize.py:142-186) — same fork+timeout pattern, since a hung
    Agg-backend render can't be interrupted with signal.SIGALRM. `fits=[single_fit]`
    (length 1) makes plot_scoreset_best_config's own "if len(log_lr_plus) == 1"
    branch plot just the one deterministic LR+ curve instead of a 5th/50th/95th
    percentile band — there's no bootstrap variance to show.
    """
    import signal
    from src.assay_calibration.plot_utils.utils import plot_scoreset_best_config

    dataset_name = result["dataset_name"]
    variant = result["variant"]
    calibration = result["calibration"]
    single_fit_obj = result["single_fit"]

    figure_path = Path(output_dir) / f"{dataset_name}_{variant}_visualization.png"

    child_pid = os.fork()
    if child_pid == 0:
        try:
            fig = plot_scoreset_best_config(
                dataset=dataset_name,
                scoreset=scoreset,
                indv_summary=calibration,
                fits=[single_fit_obj],
                score_range=np.asarray(calibration["score_range"]),
                config=f"({config.benign_method})",
                n_c=variant,
                n_samples=len([s for s in scoreset.samples]),
                relax=False,
                flipped=calibration.get("scoreset_flipped", False),
            )
            fig.savefig(figure_path, dpi=150)
            os._exit(0)
        except Exception:
            os._exit(1)
    else:
        import time
        start = time.monotonic()
        exit_status = None
        while time.monotonic() - start < 600:
            pid, st = os.waitpid(child_pid, os.WNOHANG)
            if pid != 0:
                exit_status = st
                break
            time.sleep(1)
        if exit_status is None:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            logger.warning("  Visualization timed out after 600s, skipping")
        elif os.WIFEXITED(exit_status) and os.WEXITSTATUS(exit_status) == 0:
            logger.info(f"  Saved visualization: {figure_path}")
        else:
            logger.error("  Visualization process failed")


def _save_variant_result(result, scoreset, output_dir, generate_viz):
    """Write calibration.json / lr_values.json.gz / variants.csv (and,
    unless generate_viz=False, visualization.png), matching the on-disk
    layout save_results()/compute_variant_table()/generate_visualizations()
    produce for the main pipeline (with the variant name standing in for the
    usual "2c"/"3c" component key).
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import setup_logging, save_results
    from src.assay_calibration.pipeline.variant_evidence import compute_variant_table

    dataset_name = result["dataset_name"]
    variant = result["variant"]
    calibration = result["calibration"]

    ds_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_output_dir, exist_ok=True)
    logger = setup_logging(ds_output_dir, f"{dataset_name}_{variant}")

    config = PipelineConfig(
        dataset_csv="",
        dataset_name=dataset_name,
        output_dir=ds_output_dir,
        components=[2],
        benign_method=("benign" if variant == "plp_blb" else "avg"),
        compute_oob=False,
        acmg_mapping_method=calibration["acmg_mapping_method"],
    )

    variant_df = compute_variant_table(
        scoreset=scoreset, calibration=calibration, config=config, logger=logger)
    table_path = os.path.join(ds_output_dir, f"{dataset_name}_{variant}_variants.csv")
    variant_df.to_csv(table_path, index=False)
    logger.info(f"  Saved: {table_path} ({len(variant_df)} variants)")

    save_results(results={variant: calibration}, bootstrap_results=None,
                 config=config, logger=logger, selected_k=None)

    if generate_viz:
        _generate_visualization(result, scoreset, config, logger, ds_output_dir)


def _requires_2018(df, dataset):
    genes_2018 = {"BRCA1", "PTEN", "MSH2", "TP53"}
    return df[df["Dataset"] == dataset]["Gene"].iloc[0] in genes_2018


def _process_dataset(df_ds, dataset, clinvar_release, population_type,
                      prior, output_dir, score_range_points, point_values,
                      acmg_mapping_method, generate_viz):
    from src.assay_calibration.data_utils.dataset import Scoreset

    dataset_name = dataset if clinvar_release == "2026" else f"{dataset}_clinvar_{clinvar_release}"
    variants = ("plp_blb", "plp_blb_synon")
    try:
        kw = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
        if population_type:
            kw["population_type"] = population_type
        scoreset = Scoreset(df_ds, **kw)
    except (ValueError, KeyError) as e:
        print(f"  {dataset_name} skipping: {e}")
        return [{"dataset_name": dataset_name, "variant": v,
                 "status": "skipped", "reason": str(e)} for v in variants]

    results = []
    for variant in variants:
        result = _fit_variant(
            scoreset, dataset_name, variant, prior, score_range_points,
            point_values, acmg_mapping_method)
        if result["status"] == "fit":
            _save_variant_result(result, scoreset, output_dir, generate_viz)
            # Don't carry the (large, non-JSON-serializable) calibration/fit
            # objects into the summary JSON — full versions are already on
            # disk per-dataset via _save_variant_result.
            result = {k: v for k, v in result.items()
                      if k not in ("calibration", "single_fit")}
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple 2-component GMM baseline (local, non-SLURM, "
                     "no bootstrap, fixed prior)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--population-type", default=None)
    parser.add_argument("--prior", type=float, default=_DEFAULT_PRIOR,
                       help=f"Fixed prior used for every dataset (default: {_DEFAULT_PRIOR}); "
                            "neither variant uses a gnomAD/population sample, so there is no "
                            "empirical prior estimate to fall back on.")
    parser.add_argument("--acmg-mapping-method", default="tavtigian",
                       choices=["tavtigian", "piecewise", "strict_additive"])
    parser.add_argument("--score-range-points", type=int, default=2000)
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip generating <name>_<variant>_visualization.png "
                            "(calibration/lr_values/variants are always written)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sep = "\t" if args.dataframe.endswith((".tsv.gz", ".tsv")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)

    datasets = df["Dataset"].unique()
    print(f"Datasets: {len(datasets)}")
    partitions = {ds: df[df["Dataset"] == ds] for ds in datasets}

    point_values = [1, 2, 3, 4, 5, 6, 7, 8]
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_dataset)(
            partitions[ds], ds,
            clinvar_release="2018" if _requires_2018(df, ds) else "2026",
            population_type=args.population_type,
            prior=args.prior,
            output_dir=args.output_dir,
            score_range_points=args.score_range_points,
            point_values=point_values,
            acmg_mapping_method=args.acmg_mapping_method,
            generate_viz=not args.no_viz,
        )
        for ds in datasets
    )

    flat = [r for rs in results if rs for r in rs]
    out_path = os.path.join(args.output_dir, "simple_gmm_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)

    n_fit = sum(1 for r in flat if r["status"] == "fit")
    print(f"\nDone: {n_fit} fit, {len(flat) - n_fit} skipped, "
          f"out of {len(flat)} (datasets x variants)")
    print(f"Per-dataset calibration/lr_values/variants written under {args.output_dir}/<dataset_name>/")
    print(f"Summary index saved to {out_path}")


if __name__ == "__main__":
    main()
