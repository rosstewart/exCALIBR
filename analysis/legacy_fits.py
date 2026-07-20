"""
Bridge: assemble (scoreset, indv_summary, fits, score_range, scoreset_flipped)
directly from run_pipeline.py / run_igvf_batch.py output + the master
dataframe + a precomputed bootstrap-fits file — replacing
src.assay_calibration.plot_utils.utils.load_dataset_for_plot's dependency on
hand-curated point_assignment_*/{dataset}/*.pkl files.

Once assembled, these objects are handed unmodified to the existing plotting
functions in src/assay_calibration/plot_utils/utils.py (plot_scoreset_best_config,
plot_scoreset_final_pillar_project_v2, plot_scoreset_cartoon_overview,
plot_four_datasets_publication, etc.) — output is pixel-identical to before,
given identical underlying fit data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis import config as cfg
from analysis.discovery import discover_outputs, resolve_component, _parse_dataset_config_entry, load_master_df


def _load_calibration_and_lr(
    calib_path: Path, lr_path: Path
) -> Dict:
    """Load a calibration dict suitable for plotting from saved pipeline files.

    Moved from run_igvf_batch.py::_load_calibration_from_disk (identical logic) —
    supports both the compact (percentile-only) lr_values format and the legacy
    format that stored all bootstrap rows.
    """
    import gzip as _gzip

    with open(calib_path) as f:
        c = json.load(f)
    with _gzip.open(lr_path, "rt") as f:
        lr = json.load(f)

    # `log_lr_plus` in the returned dict must stay the FULL per-bootstrap
    # matrix whenever one is available -- plot_scoreset_best_config computes
    # its own percentiles from it (np.nanpercentile(log_lr_plus, [5,50,95])).
    # Collapsing to the 3-row [p5,p50,p95] summary here and calling it
    # `log_lr_plus` was a real bug: that plotting code would then take
    # percentiles *of the percentiles* (effectively ~90% p5 + ~10% p50 for
    # the "5th percentile" curve), which measurably inflates the 5th
    # percentile curve wherever the median is far above the true p5 (exactly
    # the bootstrap-instability signature of a dataset with very few benign
    # controls) -- confirmed on ASPA_Grønbæk-Thygesen_2024_abundance: this
    # bug alone moved the reconstructed 5th-percentile peak from 1.51 (the
    # correct value, computed once from the real 944-bootstrap matrix) to
    # 4.23, enough to spuriously cross the +1 evidence threshold in the plot
    # even though point_ranges (computed once, correctly, by the pipeline
    # itself) are genuinely empty everywhere for that dataset.
    if "log_lr_plus_p5" in lr:
        # Only the compact percentile summary was ever saved (no raw
        # per-bootstrap matrix on disk to fall back to) -- this is the one
        # case where re-percentiling downstream is unavoidable and will
        # carry the same small bias described above.
        log_lr_pct = np.array([lr["log_lr_plus_p5"], lr["log_lr_plus_p50"], lr["log_lr_plus_p95"]])
        priors_pct = lr.get("priors_pct", [lr["prior"]] * 3)
        log_lr_plus_full = log_lr_pct
    else:
        llr = np.asarray(lr["log_lr_plus"])
        log_lr_pct = np.nanpercentile(llr, [5, 50, 95], axis=0)
        priors_pct = [lr.get("prior")] * 3
        log_lr_plus_full = llr

    return {
        "prior": c["prior"],
        "priors_pct": priors_pct,
        "priors": priors_pct,
        "log_lr_plus": log_lr_plus_full,
        "point_ranges": {int(k): v for k, v in c["point_ranges"].items()},
        "score_range": lr["score_range"],
        "log_lr_pct": log_lr_pct,
        "C": None,
        "scoreset_flipped": c.get("scoreset_flipped", False),
    }


def resolve_component_for(
    dataset: str,
    output_dir: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve (n_c, benign_method) for a dataset using --dataset-configs JSON,
    falling back to whatever component is actually on disk in output_dir.

    Replaces src.assay_calibration.plot_utils.utils.import_dataset_configurations's
    hardcoded ~90-dataset table with the same JSON config file run_igvf_batch.py
    itself was driven by.
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS

    dataset_configs = None
    if dataset_configs_path and Path(dataset_configs_path).exists():
        with open(dataset_configs_path) as f:
            dataset_configs = json.load(f)

    tree, model_selections, _ = discover_outputs(Path(output_dir) / dataset)
    if not tree:
        # dataset dir may be flat (no per-dataset subdir) — try output_dir directly
        tree, model_selections, _ = discover_outputs(Path(output_dir))
        tree = {dataset: tree[dataset]} if dataset in tree else tree

    available_comps = list(tree.get(dataset, {}).keys())
    if not available_comps:
        # Nothing discovered on disk — fall back purely to config
        if dataset_configs and dataset in dataset_configs:
            comp = _parse_dataset_config_entry(dataset_configs[dataset])
            n_c, _, benign = comp.partition("_")
            return comp.split("_")[0], (benign or "avg")
        return "3c", "avg"

    comp = resolve_component(dataset, available_comps, model_selections, dataset_configs)
    n_c = comp.split("_")[0]
    benign_method = comp.split("_", 1)[1] if "_" in comp else "avg"
    return n_c, benign_method


def load_scoreset_and_fits(
    dataset: str,
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    precomputed_fits: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
    n_c: Optional[str] = None,
    benign_method: Optional[str] = None,
    method: Optional[str] = None,
    pipeline_dataset: Optional[str] = None,
    clinvar_release: Optional[str] = None,
) -> Tuple[object, Dict, list, np.ndarray, str, int, bool]:
    """Assemble everything the legacy plotting functions need for one dataset.

    `dataset` controls which rows are pulled from the master dataframe and
    (unless `clinvar_release` is given explicitly) which ClinVar release is
    used to build the Scoreset. `pipeline_dataset` (defaults to `dataset`)
    controls which on-disk pipeline output (calibration/LR/fits) is loaded —
    pass this separately when the two diverge, e.g. genes that were only ever
    run through the pipeline under their "_clinvar_2018" name (BRCA1/MSH2/
    PTEN/TP53) but where you want a Scoreset built with clinvar_release="2026"
    for a "current ClinVar" panel reusing that same fit.

    Returns
    -------
    (scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped)
    matching the tuple order legacy scripts unpacked from their pickle files
    (see test/plot_MSH2_ex.py, test/plot_author_calibration_confusion.py).

    Raises FileNotFoundError / KeyError if the calibration, LR-values, or
    precomputed-fits entry for this dataset can't be found — callers should
    catch and skip rather than silently substitute different data.
    """
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    dataset_tsv = dataset_tsv or cfg.DATASET_TSV
    precomputed_fits = precomputed_fits or cfg.PRECOMPUTED_FITS
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS
    pipeline_dataset = pipeline_dataset or dataset

    if n_c is None or benign_method is None:
        n_c, benign_method = resolve_component_for(pipeline_dataset, str(output_dir), dataset_configs_path)
    comp = f"{n_c}_{benign_method}"

    ds_output_dir = output_dir / pipeline_dataset
    if not ds_output_dir.exists():
        ds_output_dir = output_dir  # flat layout fallback

    method_token = f"_{method}" if method else ""
    calib_path = ds_output_dir / f"{pipeline_dataset}{method_token}_{comp}_calibration.json"
    lr_path = ds_output_dir / f"{pipeline_dataset}{method_token}_{comp}_lr_values.json.gz"
    if not calib_path.exists():
        # try bare n_c (no benign_method suffix), the older naming convention
        calib_path = ds_output_dir / f"{pipeline_dataset}{method_token}_{n_c}_calibration.json"
        lr_path = ds_output_dir / f"{pipeline_dataset}{method_token}_{n_c}_lr_values.json.gz"
    if not calib_path.exists() or not lr_path.exists():
        raise FileNotFoundError(
            f"Calibration/LR files not found for {pipeline_dataset} ({comp}) under {ds_output_dir}"
        )

    indv_summary = _load_calibration_and_lr(calib_path, lr_path)
    score_range = np.asarray(indv_summary["score_range"])
    scoreset_flipped = bool(indv_summary.get("scoreset_flipped", False))

    # Scoreset, built the same way the pipeline itself builds it (see
    # src/assay_calibration/pipeline/utils.py::load_dataset_from_df).
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df

    df_full = load_master_df(dataset_tsv)
    csv_name = dataset.replace("_clinvar_2018", "")
    df_ds = df_full[df_full["Dataset"] == csv_name].copy()
    if df_ds.empty:
        from analysis.author_labels import load_name_mapping
        old_to_new, _ = load_name_mapping(str(dataset_tsv))
        alt = old_to_new.get(csv_name)
        if alt:
            df_ds = df_full[df_full["Dataset"] == alt].copy()
    if df_ds.empty:
        raise ValueError(f"Dataset '{csv_name}' not found in {dataset_tsv}")
    df_ds["Dataset"] = dataset

    if clinvar_release is None:
        clinvar_release = "2018" if "_clinvar_2018" in dataset else "2026"
    pcfg = PipelineConfig(
        dataset_csv=str(dataset_tsv),
        dataset_name=dataset,
        output_dir="/tmp",
        clinvar_release=clinvar_release,
    )
    scoreset = load_dataset_from_df(df_ds, pcfg)
    n_samples = len([s for s in scoreset.samples])

    # Full per-bootstrap fits (component_params/weights), for density-overlay plots.
    from src.assay_calibration.pipeline.visualize import load_precomputed_fits

    bootstrap_results = load_precomputed_fits(precomputed_fits, pipeline_dataset)
    fits = [
        seed_results[n_c] for seed_results in bootstrap_results.values()
        if isinstance(seed_results, dict) and seed_results.get(n_c) is not None
    ]
    if not fits:
        raise KeyError(f"No '{n_c}' fits found for '{pipeline_dataset}' in {precomputed_fits}")

    return scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped
