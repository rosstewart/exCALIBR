"""
Robustness analysis: how sensitive is ExCALIBR's calibration to shrinking
(downsampling) or discordant (mislabeled) P/LP and B/LB control counts?

`test/downsample_discordance_test.ipynb` already generated perturbed-input
calibration runs via `Scoreset.from_scoreset(base_scoreset,
downsample_n_variants=N, perturbation_seed=seed)` (N in [1,2,4,8,16,32,64])
and `Scoreset.from_scoreset(base_scoreset, discordance_pct=pct,
perturbation_seed=seed)` (pct in [0.01, 0.10]), each with 10 seeds, run
through the normal run_igvf_batch.py pipeline. Output lives in a FLAT
directory tree under analysis.config.ROBUSTNESS_OUTPUT_DIR, one subdirectory
per condition:

    {base_dataset}_ds{N}_s{seed}/{base_dataset}_ds{N}_s{seed}_{n_c}c_variants.csv
    {base_dataset}_ds{N}_s{seed}/..._calibration.json
    {base_dataset}_ds{N}_s{seed}/..._lr_values.json.gz
    {base_dataset}_disc{pct:.2f}_s{seed}/...   (same pattern)

IMPORTANT: a perturbed condition's own confusion matrix is not directly
useful. Downsampled conditions have as few as 1 P/LP or 1 B/LB variant, and
`Scoreset.from_scoreset`'s downsample path filters `_scores`/
`_sample_assignments`/etc. but never re-slices `_ids`/`_variant_codes`/
`variants`/`dataframe` -- so a downsampled condition's `variant_id` values
(themselves just positional `variant_0`, `variant_1`, ... placeholders,
since the generator flattened every perturbed Scoreset to bare
(score, sample_assignments) rows before running the pipeline) do not
correspond by position to the reference dataset either. There is no reliable
per-variant join between a robustness condition and the reference dataset.

Instead, every function here applies a condition's `point_ranges`
(calibration thresholds) to the FULL, UNPERTURBED reference dataset's own
variant scores (see `recompute_points_from_calibration` in
analysis.discovery) -- this isolates "how much did the calibration threshold
shift under this perturbation" from "how few variants are in this
condition's own tiny confusion matrix". Confusion matrices, summary metrics,
and LR+ curves in this module are therefore always computed against the
reference population; only the point_ranges/LR+ curve shape comes from the
perturbed condition.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import config as cfg
from analysis.discovery import (
    discover_outputs, resolve_component, recompute_points_from_calibration,
)
from analysis.confusion import build_confusion_matrix, make_single_confusion_figure
from analysis.plot_common import save_and_show, sample_matches, is_notebook

# Genes whose main-pipeline output is only ever saved under a
# "_clinvar_2018"-suffixed dataset name (see run_igvf_batch.py /
# run_acmgscaler_all.py's identical GENES_2018 convention).
GENES_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}


# ---------------------------------------------------------------------------
# Condition directory parsing / discovery
# ---------------------------------------------------------------------------

# Inverse of the generator's own suffix-stripping regex
# ('_ds[0-9]+_s[0-9]+$|_disc[0-9.]+_s[0-9]+$', see hpc/prepare.py
# --name-strip in test/downsample_discordance_test.ipynb) -- provably
# consistent with how the condition names were built, not a fresh guess.
#
# GENES_2018 datasets (BRCA1/MSH2/PTEN/TP53) additionally carry a trailing
# "_clinvar_2018" *after* the seed (e.g. "BRCA1_Findlay_2018_ds16_s0_
# clinvar_2018", reproducing the main pipeline's own naming convention for
# those genes). That suffix MUST be optional here, or those 3 of the 4 base
# datasets on disk fail to parse entirely and silently vanish from
# discover_robustness_base_datasets.
_COND_RE = re.compile(
    r"^(?P<base>.+)_(?:ds(?P<n>\d+)|disc(?P<pct>\d+\.\d+))_s(?P<seed>\d+)(?:_clinvar_2018)?$"
)

# Extracts the "{n_c}[_{benign_method}]" component token from a
# *_calibration.json filename, e.g. "..._3c_avg_calibration.json" -> "3c_avg".
_COMP_RE = re.compile(r"_(?P<comp>(?:2c|3c)(?:_[A-Za-z]+)?)_calibration\.json$")


def _infer_component(calibration_path: Path) -> str:
    """The "{n_c}[_{benign_method}]" component token a calibration.json's own
    filename was saved under -- used to find the SAME component in every
    robustness condition dir (which may hold more than one, e.g. both
    "_3c_calibration.json" and "_3c_avg_calibration.json"), rather than
    picking whichever sorts first alphabetically."""
    m = _COMP_RE.search(calibration_path.name)
    if not m:
        raise ValueError(f"Could not infer component token from {calibration_path.name}")
    return m.group("comp")


def parse_condition_dirname(dirname: str) -> Optional[dict]:
    """Parse one robustness condition directory name.

    Returns {"base_dataset", "perturbation_type": "downsample"|"discordance",
    "level": int|float, "seed": int}, or None if dirname doesn't match either
    pattern (e.g. the sibling 'logs/' directory) -- callers skip cleanly.
    """
    m = _COND_RE.match(dirname)
    if m is None:
        return None
    if m.group("n") is not None:
        return {
            "base_dataset": m.group("base"),
            "perturbation_type": "downsample",
            "level": int(m.group("n")),
            "seed": int(m.group("seed")),
        }
    return {
        "base_dataset": m.group("base"),
        "perturbation_type": "discordance",
        "level": float(m.group("pct")),
        "seed": int(m.group("seed")),
    }


def discover_robustness_conditions(
    base_dataset: str, robustness_output_dir: Optional[str] = None,
) -> List[dict]:
    """All condition dirs for `base_dataset` under robustness_output_dir,
    sorted by (perturbation_type, level, seed). [] if the directory is
    missing or nothing matches (caller skips that base_dataset cleanly)."""
    root = Path(robustness_output_dir or cfg.ROBUSTNESS_OUTPUT_DIR)
    if not root.is_dir():
        return []
    conditions = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_condition_dirname(child.name)
        if parsed is None or parsed["base_dataset"] != base_dataset:
            continue
        parsed["dir"] = child
        conditions.append(parsed)
    conditions.sort(key=lambda c: (c["perturbation_type"], c["level"], c["seed"]))
    return conditions


def discover_robustness_base_datasets(robustness_output_dir: Optional[str] = None) -> List[str]:
    """Distinct base_dataset names with >=1 condition dir on disk -- what the
    notebook loops over instead of a hardcoded dataset list."""
    root = Path(robustness_output_dir or cfg.ROBUSTNESS_OUTPUT_DIR)
    if not root.is_dir():
        return []
    bases = set()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_condition_dirname(child.name)
        if parsed is not None:
            bases.add(parsed["base_dataset"])
    return sorted(bases)


def find_condition_calibration(condition_dir: Path, component: Optional[str] = None) -> Optional[Path]:
    """The *_calibration.json inside one condition dir matching `component`
    (e.g. "3c_avg", inferred from the reference dataset's own resolved
    component via _infer_component) -- some condition dirs hold more than one
    (e.g. both "_3c_" and "_3c_avg_"), so this must match the SAME one the
    reference is using rather than picking whichever sorts first
    alphabetically. Falls back to the first calibration.json found if
    `component` is None or doesn't match anything. None if the dir has none."""
    if component:
        candidate = condition_dir / f"{condition_dir.name}_{component}_calibration.json"
        if candidate.exists():
            return candidate
    matches = sorted(condition_dir.glob("*_calibration.json"))
    return matches[0] if matches else None


def find_condition_lr_values(condition_dir: Path, component: Optional[str] = None) -> Optional[Path]:
    """The *_lr_values.json.gz inside one condition dir, matched to the same
    stem as find_condition_calibration's result."""
    cal_path = find_condition_calibration(condition_dir, component)
    if cal_path is None:
        return None
    lr_path = cal_path.parent / cal_path.name.replace("_calibration.json", "_lr_values.json.gz")
    return lr_path if lr_path.exists() else None


# ---------------------------------------------------------------------------
# Reference (unperturbed) dataset loading
# ---------------------------------------------------------------------------

def load_reference_variants(
    base_dataset: str,
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
    tree: Optional[Dict] = None,
    model_selections: Optional[Dict] = None,
    calibrations: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Load the reference (full, unperturbed) dataset's variants (score +
    sample [+ standard_points] columns) plus its own resolved
    calibration.json path -- every robustness condition's point_ranges get
    applied to this same fixed population.

    Preferred: base_dataset's (or f"{base_dataset}_clinvar_2018" for genes
    in GENES_2018) *_variants.csv from the MAIN pipeline output_dir (default
    cfg.OUTPUT_DIR) -- the same variants/config every other notebook section
    already uses.
    Fallback: build fresh from the master dataframe via
    analysis.legacy_fits.load_scoreset_and_fits +
    analysis.comparison_methods.build_variants_df_from_scoreset, if nothing
    is found in output_dir.

    tree/model_selections/calibrations : pass in the triple already returned
    by analysis.discovery.discover_outputs(output_dir) (e.g. section 1's own
    globals in analyze_pipeline_output.py) to skip re-walking the entire
    (~89-dataset) output_dir tree on every call -- callers looping over
    several robustness base datasets would otherwise re-discover the whole
    main pipeline output from scratch once per base dataset. Falls back to
    discovering it here (as before) if not given.

    Raises FileNotFoundError if neither path succeeds.
    """
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS

    dataset_configs = None
    if dataset_configs_path and Path(dataset_configs_path).exists():
        with open(dataset_configs_path) as f:
            dataset_configs = json.load(f)

    if tree is None or model_selections is None or calibrations is None:
        tree, model_selections, calibrations = discover_outputs(output_dir)

    candidates = [base_dataset]
    if base_dataset.split("_")[0] in GENES_2018:
        candidates.append(f"{base_dataset}_clinvar_2018")

    for pipeline_dataset in candidates:
        if pipeline_dataset not in tree:
            continue
        comp = resolve_component(
            pipeline_dataset, list(tree[pipeline_dataset].keys()), model_selections, dataset_configs,
        )
        method_dict = tree[pipeline_dataset][comp]
        csv_path = method_dict.get("default") or next(
            (p for p in method_dict.values() if p is not None), None,
        )
        cal_path = calibrations.get(pipeline_dataset, {}).get("default", {}).get(comp)
        if csv_path is None or cal_path is None:
            continue
        df = pd.read_csv(csv_path)
        return df, cal_path

    # Fallback: build fresh from the master dataframe.
    try:
        from analysis.legacy_fits import load_scoreset_and_fits
        from analysis.comparison_methods import build_variants_df_from_scoreset

        pipeline_dataset = candidates[-1] if len(candidates) > 1 else base_dataset
        scoreset, indv_summary, *_ = load_scoreset_and_fits(
            base_dataset, output_dir=str(output_dir), dataset_tsv=dataset_tsv,
            dataset_configs_path=dataset_configs_path, pipeline_dataset=pipeline_dataset,
        )
        df = build_variants_df_from_scoreset(scoreset)
        # load_scoreset_and_fits doesn't return the calibration.json path
        # itself, only its parsed contents -- re-resolve it the same way
        # resolve_component_for did internally.
        from analysis.legacy_fits import resolve_component_for
        n_c, benign_method = resolve_component_for(pipeline_dataset, str(output_dir), dataset_configs_path)
        comp = f"{n_c}_{benign_method}"
        ds_dir = output_dir / pipeline_dataset
        if not ds_dir.exists():
            ds_dir = output_dir
        cal_path = ds_dir / f"{pipeline_dataset}_{comp}_calibration.json"
        if not cal_path.exists():
            cal_path = ds_dir / f"{pipeline_dataset}_{n_c}_calibration.json"
        if not cal_path.exists():
            raise FileNotFoundError(f"No calibration.json found for {pipeline_dataset} to pair with rebuilt Scoreset")
        return df, cal_path
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load reference variants for '{base_dataset}' from {output_dir} "
            f"(tried {candidates}), and fallback Scoreset rebuild failed: {e}"
        )


# ---------------------------------------------------------------------------
# Confusion matrices + scalar metrics
# ---------------------------------------------------------------------------

def compute_robustness_confusion_matrices(
    base_dataset: str,
    reference_df: pd.DataFrame,
    reference_calibration_path: Path,
    robustness_output_dir: Optional[str] = None,
) -> Dict[Tuple[str, float, int], pd.DataFrame]:
    """One confusion matrix per discovered condition (that condition's
    point_ranges applied to reference_df), plus one ("reference", 0, 0)
    baseline entry: the reference dataset's own point_ranges applied to
    itself. Both use build_confusion_matrix(..., use_oob=False) -- a
    self-consistent scoring convention across every point in the comparison
    (including the baseline), so an observed robustness curve isn't an
    artifact of comparing OOB-scored vs freshly-assigned standard_points.
    """
    matrices: Dict[Tuple[str, float, int], pd.DataFrame] = {}
    component = _infer_component(reference_calibration_path)

    ref_scored = recompute_points_from_calibration(reference_df, reference_calibration_path)
    ref_mat = build_confusion_matrix(ref_scored, use_oob=False, label=f"{base_dataset}/reference")
    if ref_mat is not None:
        matrices[("reference", 0, 0)] = ref_mat

    for cond in discover_robustness_conditions(base_dataset, robustness_output_dir):
        cal_path = find_condition_calibration(cond["dir"], component)
        if cal_path is None:
            print(f"  SKIP {cond['dir'].name}: no calibration.json")
            continue
        try:
            df_scored = recompute_points_from_calibration(reference_df, cal_path)
        except ValueError as e:
            print(f"  SKIP {cond['dir'].name}: {e}")
            continue
        label = f"{base_dataset}/{cond['perturbation_type']}={cond['level']}/seed={cond['seed']}"
        mat = build_confusion_matrix(df_scored, use_oob=False, label=label)
        if mat is not None:
            matrices[(cond["perturbation_type"], cond["level"], cond["seed"])] = mat

    return matrices


def robustness_confusion_matrices_to_metrics(
    matrices: Dict[Tuple[str, float, int], pd.DataFrame], base_dataset: str,
) -> pd.DataFrame:
    """One row per (perturbation_type, level, seed) matrix (plus the
    'reference' baseline row), columns: base_dataset, perturbation_type,
    level, seed, then every key from compute_classification_metrics (accuracy,
    coverage, dor_standard, sensitivity, specificity, mcc, lr_plus_*, ...)."""
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    rows = []
    for (ptype, level, seed), mat in matrices.items():
        metrics = compute_classification_metrics(mat)
        rows.append({"base_dataset": base_dataset, "perturbation_type": ptype,
                      "level": level, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def run_robustness_analysis(
    base_dataset: str,
    output_dir: Optional[str] = None,
    robustness_output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Convenience wrapper: load_reference_variants ->
    compute_robustness_confusion_matrices -> robustness_confusion_matrices_to_metrics.
    Returns None (prints a SKIP reason) if reference loading fails or no
    conditions were found on disk."""
    try:
        reference_df, ref_cal_path = load_reference_variants(
            base_dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
            dataset_configs_path=dataset_configs_path,
        )
    except FileNotFoundError as e:
        print(f"  SKIP {base_dataset}: {e}")
        return None

    matrices = compute_robustness_confusion_matrices(
        base_dataset, reference_df, ref_cal_path, robustness_output_dir,
    )
    if not matrices:
        print(f"  SKIP {base_dataset}: no robustness conditions found under "
              f"{robustness_output_dir or cfg.ROBUSTNESS_OUTPUT_DIR}")
        return None
    return robustness_confusion_matrices_to_metrics(matrices, base_dataset)


# ---------------------------------------------------------------------------
# Per-condition / per-level data loaders (fits, LR percentiles, point ranges)
#
# NOTE: every *_lr_values.json.gz under ROBUSTNESS_OUTPUT_DIR only ever holds
# the compact [p5,p50,p95] percentile summary (confirmed on disk -- these
# runs never wrote the full per-bootstrap log_lr_plus matrix), so there is no
# "pool raw bootstraps across seeds" path for the LR+ curve. What IS on disk,
# separately, is a full per-bootstrap MIXTURE-FIT record for every condition
# (analysis.config.ROBUSTNESS_BOOTSTRAP_RESULTS, same {condition_dirname:
# {seed: {"2c"/"3c": fit}}} format as the main pipeline's PRECOMPUTED_FITS) --
# that's what backs the density row below. For the LR+ curve itself, each
# seed already carries its own correctly-computed [p5,p50,p95] curve (derived
# from that seed's own bootstraps by the pipeline); "pooling across seeds" for
# LR+ means looking at the SPREAD of those 10 already-computed curves, not
# re-deriving a curve from raw bootstraps.
# ---------------------------------------------------------------------------

# ROBUSTNESS_BOOTSTRAP_RESULTS is a single ~114MB gzip JSON holding every
# condition's bootstrap fits keyed by condition dirname (350 conditions x
# ~1000 bootstraps). load_precomputed_fits decompresses + json.loads the
# WHOLE file on every call; pooled_fits_for_level calls load_condition_fits
# once per seed (10x per level), so without caching a single
# plot_robustness_config_summary call would reparse the full multi-hundred-MB
# file 10 times. Cache the parsed top-level dict per (path) for the lifetime
# of the process instead.
_bootstrap_results_cache: Dict[str, dict] = {}


def _load_all_bootstrap_results(path: str) -> dict:
    if path not in _bootstrap_results_cache:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _bootstrap_results_cache[path] = json.load(f)
    return _bootstrap_results_cache[path]


def load_condition_fits(
    condition_dir: Path, component: str, bootstrap_results_path: Optional[str] = None,
    bootstrap_results_data: Optional[dict] = None,
) -> Optional[List[dict]]:
    """One condition's full per-bootstrap component fits (component_params +
    weights, one entry per bootstrap seed) from ROBUSTNESS_BOOTSTRAP_RESULTS,
    keyed by this condition's own directory name -- same record shape
    sample_density() expects (see legacy_fits.load_scoreset_and_fits's
    identical use of load_precomputed_fits for the main pipeline). None if
    this condition has no entry / no fit for `component`'s n_c token.

    If `bootstrap_results_data` is given, it's used directly instead of
    reading/decompressing `bootstrap_results_path` -- lets callers that
    already loaded (and, ideally, pre-filtered down to just the conditions
    they need) the big gzipped JSON pass it straight through, instead of
    every caller (e.g. one per joblib worker process) independently paying
    the full ~114MB decompression cost via `_load_all_bootstrap_results`'s
    process-local cache."""
    n_c = component.split("_")[0]
    if bootstrap_results_data is not None:
        all_results = bootstrap_results_data
    else:
        path = bootstrap_results_path or cfg.ROBUSTNESS_BOOTSTRAP_RESULTS
        all_results = _load_all_bootstrap_results(path)
    bootstrap_results = all_results.get(condition_dir.name)
    if bootstrap_results is None:
        return None
    fits = [
        seed_results[n_c] for seed_results in bootstrap_results.values()
        if isinstance(seed_results, dict) and seed_results.get(n_c) is not None
    ]
    return fits or None


def pooled_fits_for_level(
    base_dataset: str, perturbation_type: str, level, component: str,
    robustness_output_dir: Optional[str] = None, bootstrap_results_path: Optional[str] = None,
    max_fits: Optional[int] = 2000, bootstrap_results_data: Optional[dict] = None,
) -> List[dict]:
    """Every seed's per-bootstrap fits at this (perturbation_type, level),
    concatenated into one flat list ("flatten into bootstrap x seed") --
    directly what sample_density() needs to draw a 5th/50th/95th percentile
    density band representing genuine bootstrap x seed uncertainty.

    sample_density's cost scales with len(fits) x len(score_range) x
    n_components x n_samples -- pooling all 10 seeds x ~1000 bootstraps each
    (10,000 fits) takes ~100s per figure, and this function gets called for
    9 levels x 4 datasets. A single seed's own ~1000 bootstraps is already
    what the pipeline treats as enough for a stable percentile estimate
    everywhere else in this codebase, so cap the pooled set at `max_fits`
    (default 2000, i.e. roughly 2 seeds' worth) via a fixed-seed random
    subsample -- keeps runtime bounded without materially changing the
    5th/50th/95th percentile band. Pass max_fits=None to disable.
    """
    fits: List[dict] = []
    for cond in discover_robustness_conditions(base_dataset, robustness_output_dir):
        if cond["perturbation_type"] != perturbation_type or cond["level"] != level:
            continue
        seed_fits = load_condition_fits(cond["dir"], component, bootstrap_results_path, bootstrap_results_data)
        if seed_fits:
            fits.extend(seed_fits)
    if max_fits is not None and len(fits) > max_fits:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(fits), size=max_fits, replace=False)
        fits = [fits[i] for i in idx]
    return fits


def load_condition_lr_percentiles(condition_dir: Path, component: str) -> Optional[dict]:
    """One condition's own [p5,p50,p95] Log LR+ percentile curve (computed
    once, correctly, across THAT seed's own bootstraps by the pipeline) plus
    score_range/prior, via the same loader used everywhere else in
    analysis/. None if calibration.json/lr_values.json.gz are missing."""
    from analysis.legacy_fits import _load_calibration_and_lr

    cal_path = find_condition_calibration(condition_dir, component)
    lr_path = find_condition_lr_values(condition_dir, component)
    if cal_path is None or lr_path is None:
        return None
    return _load_calibration_and_lr(cal_path, lr_path)


def load_condition_point_ranges(condition_dir: Path, component: str) -> Optional[dict]:
    """One condition's own point_ranges dict ({point_value: [[lo,hi], ...]})
    straight from its calibration.json -- None if missing."""
    cal_path = find_condition_calibration(condition_dir, component)
    if cal_path is None:
        return None
    with open(cal_path) as f:
        cal = json.load(f)
    return cal.get("point_ranges")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_LEVEL_XLABELS = {"downsample": "Downsample N (control count)", "discordance": "Discordance fraction"}
_METRIC_YLABELS = {
    "accuracy": "Accuracy", "coverage": "Coverage", "dor_standard": "DOR",
    "sensitivity": "Sensitivity", "specificity": "Specificity",
    "lr_plus_standard": "LR+", "lr_plus_pathogenic": "LR+ (pathogenic)",
    "lr_plus_benign": "LR+ (benign)", "mcc": "MCC",
}


def _reference_value(summary_df: pd.DataFrame, metric: str) -> Optional[float]:
    ref_rows = summary_df[summary_df["perturbation_type"] == "reference"]
    if ref_rows.empty:
        return None
    return float(ref_rows.iloc[0][metric])


def plot_downsample_robustness_curve(
    summary_df: pd.DataFrame, base_dataset: str,
    metrics: List[str] = ("accuracy", "coverage", "dor_standard"),
    figure_dir: Optional[Path] = None,
):
    """Median line + IQR ribbon (across the 10 seeds) vs downsample_n (log2
    scale, explicit ticks at 1,2,4,8,16,32,64), individual seed values as
    scatter points, dashed horizontal reference line at the unperturbed
    value. One subplot per metric (never shared y-axis -- accuracy/coverage
    are 0-1, DOR is unbounded)."""
    sub = summary_df[summary_df["perturbation_type"] == "downsample"].copy()
    if sub.empty:
        print(f"  SKIP downsample robustness curve for {base_dataset}: no downsample conditions")
        return None

    levels = sorted(sub["level"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        medians, p25s, p75s, xs_scatter, ys_scatter = [], [], [], [], []
        for level in levels:
            vals = sub.loc[sub["level"] == level, metric].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                medians.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
                continue
            p25, p50, p75 = np.percentile(vals, [25, 50, 75])
            medians.append(p50); p25s.append(p25); p75s.append(p75)
            xs_scatter.extend([level] * len(vals))
            ys_scatter.extend(vals.tolist())

        ax.fill_between(levels, p25s, p75s, alpha=0.25, color="C0")
        ax.plot(levels, medians, marker="o", color="C0", label="median")
        ax.scatter(xs_scatter, ys_scatter, alpha=0.4, s=15, color="C0")

        ref_val = _reference_value(summary_df, metric)
        if ref_val is not None and np.isfinite(ref_val):
            ax.axhline(ref_val, linestyle="--", color="black", alpha=0.6, label="reference")

        ax.set_xscale("log", base=2)
        ax.set_xticks(levels)
        ax.set_xticklabels([str(l) for l in levels])
        ax.set_xlabel(_LEVEL_XLABELS["downsample"])
        ax.set_ylabel(_METRIC_YLABELS.get(metric, metric))
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(linewidth=0.5, alpha=0.3)

    fig.suptitle(f"{base_dataset}: downsampling robustness", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if figure_dir is not None:
        save_and_show(fig, Path(figure_dir) / f"robustness_downsample_{base_dataset}.png")
    return fig


def plot_discordance_robustness_distribution(
    summary_df: pd.DataFrame, base_dataset: str,
    metrics: List[str] = ("accuracy", "coverage", "dor_standard"),
    figure_dir: Optional[Path] = None,
):
    """Box plot + jittered strip points per discordance level (only 2
    categorical x-positions -- a line+ribbon would misleadingly imply a
    trend between adjacent points), dashed reference line, one subplot per
    metric."""
    sub = summary_df[summary_df["perturbation_type"] == "discordance"].copy()
    if sub.empty:
        print(f"  SKIP discordance robustness distribution for {base_dataset}: no discordance conditions")
        return None

    levels = sorted(sub["level"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]
    rng = np.random.RandomState(0)

    for ax, metric in zip(axes, metrics):
        box_data = []
        for level in levels:
            vals = sub.loc[sub["level"] == level, metric].values
            vals = vals[np.isfinite(vals)]
            box_data.append(vals)

        widths = [max(levels) * 0.15] * len(levels) if max(levels) > 0 else [0.01] * len(levels)
        ax.boxplot(box_data, positions=levels, widths=widths, showfliers=False)
        for level, vals in zip(levels, box_data):
            jitter = rng.uniform(-widths[0] * 0.3, widths[0] * 0.3, size=len(vals))
            ax.scatter(np.full(len(vals), level) + jitter, vals, alpha=0.5, s=15, color="C0")

        ref_val = _reference_value(summary_df, metric)
        if ref_val is not None and np.isfinite(ref_val):
            ax.axhline(ref_val, linestyle="--", color="black", alpha=0.6, label="reference")

        ax.set_xlim(min(levels) - widths[0], max(levels) + widths[0])
        ax.set_xlabel(_LEVEL_XLABELS["discordance"])
        ax.set_ylabel(_METRIC_YLABELS.get(metric, metric))
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(linewidth=0.5, alpha=0.3)

    fig.suptitle(f"{base_dataset}: label discordance robustness", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if figure_dir is not None:
        save_and_show(fig, Path(figure_dir) / f"robustness_discordance_{base_dataset}.png")
    return fig


def plot_robustness_config_summary(
    base_dataset: str,
    perturbation_type: str,
    level,
    reference_df: pd.DataFrame,
    reference_calibration_path: Path,
    figure_dir: Optional[Path] = None,
    robustness_output_dir: Optional[str] = None,
    bootstrap_results_path: Optional[str] = None,
    bootstrap_results_data: Optional[dict] = None,
    show: bool = True,
):
    """One figure per (base_dataset, perturbation_type, level), in the same
    3-row layout as src.assay_calibration.plot_utils.utils.plot_scoreset_best_config
    (fits / point assignments / Log LR+), aggregated across all 10 seeds at
    that level:

    Row 0 (density fits): the REFERENCE dataset's own score histograms
    (fixed population -- see module docstring) with mixture-density curves
    computed from every seed's per-bootstrap fit at this level, flattened
    into one bootstrap x seed pool (10 seeds x ~1000 bootstraps each) and
    shaded by that pool's 5th/50th/95th percentile density, via the same
    sample_density() plot_scoreset_best_config uses.

    Row 1 (point assignments): all 10 seeds' point-range bars overlaid at
    low, additive alpha -- score regions where more seeds agree on a point
    value read as more opaque.

    Row 2 (Log LR+): each of the 10 seeds already carries its own correctly-
    computed [5th,50th,95th] percentile curve (across THAT seed's own
    bootstraps). The shaded bands here are the seed-to-seed (IQR) spread of
    the 5th-percentile curve and, separately, of the 95th-percentile curve --
    uncertainty *across seeds* in where those curves sit, distinct from the
    within-seed bootstrap uncertainty already baked into each curve. Each
    seed's own median curve is drawn as a thin gray line underneath. The
    reference dataset's own Log LR+ curve overlays in bold black, with
    evidence-tier threshold lines for context.
    """
    from src.assay_calibration.plot_utils.utils import (
        sample_density, add_thresholds, log_thresholds_with_ylim_pad,
    )
    from analysis.calibration_plots import _SAMPLE_ORDER, _SAMPLE_COLORS, _SAMPLE_LABELS
    from analysis.legacy_fits import _load_calibration_and_lr

    conditions = [
        c for c in discover_robustness_conditions(base_dataset, robustness_output_dir)
        if c["perturbation_type"] == perturbation_type and c["level"] == level
    ]
    if not conditions:
        print(f"  SKIP config summary for {base_dataset}/{perturbation_type}={level}: no seeds found")
        return None

    component = _infer_component(reference_calibration_path)

    with open(reference_calibration_path) as f:
        ref_cal = json.load(f)
    ref_lr_path = reference_calibration_path.parent / reference_calibration_path.name.replace(
        "_calibration.json", "_lr_values.json.gz",
    )
    ref_data = _load_calibration_and_lr(reference_calibration_path, ref_lr_path) if ref_lr_path.exists() else None
    if ref_data is None:
        print(f"  SKIP config summary for {base_dataset}/{perturbation_type}={level}: "
              f"no reference lr_values.json.gz at {ref_lr_path}")
        return None
    score_range = np.asarray(ref_data["score_range"])

    samples_present = [s for s in _SAMPLE_ORDER if sample_matches(reference_df, s).any()]
    n_samples = len(samples_present)
    if n_samples == 0:
        print(f"  SKIP config summary for {base_dataset}/{perturbation_type}={level}: "
              f"reference has no recognized sample categories")
        return None

    pooled_fits = pooled_fits_for_level(
        base_dataset, perturbation_type, level, component, robustness_output_dir, bootstrap_results_path,
        bootstrap_results_data=bootstrap_results_data,
    )
    if not pooled_fits:
        print(f"  WARNING config summary for {base_dataset}/{perturbation_type}={level}: "
              f"no per-bootstrap fits found (row 0 will be histogram-only)")

    per_seed_lr, per_seed_point_ranges = [], []
    for cond in conditions:
        lr = load_condition_lr_percentiles(cond["dir"], component)
        if lr is not None:
            per_seed_lr.append((cond["seed"], lr))
        pr = load_condition_point_ranges(cond["dir"], component)
        if pr is not None:
            per_seed_point_ranges.append((cond["seed"], pr))

    fig, ax = plt.subplots(3, n_samples, figsize=(6 * n_samples, 16), squeeze=False,
                            gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    # ---- Row 0: reference histograms + pooled bootstrap x seed density ----
    for col, sample_label in enumerate(samples_present):
        ax_fit = ax[0, col]
        mask = sample_matches(reference_df, sample_label)
        sample_scores = reference_df.loc[mask, "score"].values
        n = len(sample_scores)
        if n > 1:
            q1, q3 = np.percentile(sample_scores, [25, 75])
            iqr = q3 - q1
            fd_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
            score_width = sample_scores.max() - sample_scores.min()
            bins = min(100, int(score_width / fd_width)) if fd_width > 0 else 50
            bins = max(bins, 10)
            ax_fit.hist(sample_scores, bins=bins, density=True, alpha=0.5,
                        color=_SAMPLE_COLORS.get(sample_label, "#A0A0A0"))
        max_hist_density = max([p.get_height() for p in ax_fit.patches]) if ax_fit.patches else 1.0

        if pooled_fits:
            density = sample_density(score_range, pooled_fits, col)
            for comp_num in range(density.shape[1]):
                comp_density = density[:, comp_num, :]
                d = np.nanpercentile(comp_density, [5, 50, 95], axis=0)
                ax_fit.plot(score_range, d[1], color=f"C{comp_num}", linestyle="--",
                            label=f"Comp {comp_num + 1}")
                ax_fit.fill_between(score_range, d[0], d[2], color=f"C{comp_num}", alpha=0.15)
            total = np.nansum(density, axis=1)
            t = np.nanpercentile(total, [5, 50, 95], axis=0)
            ax_fit.plot(score_range, t[1], color="black", alpha=0.6)
            ax_fit.fill_between(score_range, t[0], t[2], color="gray", alpha=0.25)
            ax_fit.legend(fontsize=7)

        ax_fit.set_title(f"{_SAMPLE_LABELS.get(sample_label, sample_label)} (reference n={int(mask.sum()):,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        if max_hist_density:
            ax_fit.set_ylim([0, max_hist_density * 1.1])
        ax_fit.grid(linewidth=0.5, alpha=0.3)

    xlim = ax[0, 0].get_xlim()

    # ---- Row 1: point-assignment overlay across seeds ----
    all_point_values = sorted({int(k) for _, pr in per_seed_point_ranges for k in pr.keys()})
    for col in range(n_samples):
        ax_pts = ax[1, col]
        for seed, pr in per_seed_point_ranges:
            for k, ranges in pr.items():
                point_val = int(k)
                y = all_point_values.index(point_val)
                for sr in ranges:
                    x0 = xlim[0] if np.isneginf(sr[0]) else max(sr[0], xlim[0])
                    x1 = xlim[1] if np.isposinf(sr[1]) else min(sr[1], xlim[1])
                    ax_pts.plot([x0, x1], [y, y], color="red" if point_val > 0 else "blue",
                                linewidth=4, alpha=0.12, solid_capstyle="butt")
        ax_pts.set_ylim(-1, max(len(all_point_values), 1))
        ax_pts.set_yticks(range(len(all_point_values)),
                           labels=[f"{v:+d}" if v != 0 else "0" for v in all_point_values])
        ax_pts.set_xlim(xlim)
        ax_pts.set_xlabel("Score")
        ax_pts.set_ylabel("Points")
        ax_pts.set_title(f"Point assignments ({len(per_seed_point_ranges)} seeds overlaid)", fontsize=10)
        ax_pts.grid(linewidth=0.5, alpha=0.3)

    # ---- Row 2: seed-to-seed spread of each seed's own LR+ percentile curves ----
    point_values_all = sorted({abs(int(k)) for k in ref_cal["point_ranges"].keys()})
    tauP, tauB, ylim_top, ylim_bottom = log_thresholds_with_ylim_pad(ref_cal["prior"], point_values_all)

    p5_lo = p5_mid = p5_hi = p95_lo = p95_mid = p95_hi = None
    p50_curves = []
    if per_seed_lr:
        p5_curves, p95_curves = [], []
        for seed, lr in per_seed_lr:
            curve_score = np.asarray(lr["score_range"])
            pct = np.asarray(lr["log_lr_pct"])  # shape (3, N): p5, p50, p95
            p5_curves.append(np.interp(score_range, curve_score, pct[0]))
            p50_curves.append(np.interp(score_range, curve_score, pct[1]))
            p95_curves.append(np.interp(score_range, curve_score, pct[2]))
        p5_curves = np.array(p5_curves)
        p95_curves = np.array(p95_curves)
        p5_lo, p5_mid, p5_hi = np.nanpercentile(p5_curves, [25, 50, 75], axis=0)
        p95_lo, p95_mid, p95_hi = np.nanpercentile(p95_curves, [25, 50, 75], axis=0)

    for col in range(n_samples):
        ax_lr = ax[2, col]
        if per_seed_lr:
            for c in p50_curves:
                ax_lr.plot(score_range, c, color="gray", alpha=0.25, linewidth=0.7)
            ax_lr.fill_between(score_range, p5_lo, p5_hi, color="red", alpha=0.15,
                                label="5th pct. (seed IQR)")
            ax_lr.plot(score_range, p5_mid, color="red", alpha=0.7, linewidth=1)
            ax_lr.fill_between(score_range, p95_lo, p95_hi, color="blue", alpha=0.15,
                                label="95th pct. (seed IQR)")
            ax_lr.plot(score_range, p95_mid, color="blue", alpha=0.7, linewidth=1)

        ax_lr.plot(ref_data["score_range"], ref_data["log_lr_pct"][1], color="black",
                   linewidth=2, label="reference (median)")
        add_thresholds(tauP, tauB, ax_lr)
        ax_lr.set_ylim([ylim_bottom, ylim_top])
        ax_lr.set_xlim(xlim)
        ax_lr.set_xlabel("Score")
        ax_lr.set_ylabel("Log LR+")
        ax_lr.set_title("Log LR+ (seed-to-seed spread)", fontsize=10)
        if col == n_samples - 1:
            ax_lr.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        ax_lr.grid(linewidth=0.5, alpha=0.3)

    n_seeds_used = len({s for s, _ in per_seed_lr} | {s for s, _ in per_seed_point_ranges})
    fig.suptitle(
        f"{base_dataset}: {perturbation_type}={level} "
        f"({n_seeds_used} seeds, {len(pooled_fits)} pooled bootstrap fits)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if figure_dir is not None:
        out_dir = Path(figure_dir) / base_dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"robustness_config_{base_dataset}_{perturbation_type}_{level}.png"
        if show:
            save_and_show(fig, out_path)
        else:
            # Rendered in a worker process (see _run_config_summary_job) --
            # never attempt inline display here, regardless of is_notebook();
            # the batch driver displays the saved PNG back in the main
            # process once all workers finish.
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"  Saved: {out_path}")
            plt.close(fig)
    return fig


def run_config_summary_plots(
    base_dataset: str,
    reference_df: pd.DataFrame,
    reference_calibration_path: Path,
    downsample_levels: List[int] = (1, 2, 4, 8, 16, 32, 64),
    discordance_levels: List[float] = (0.01, 0.10),
    figure_dir: Optional[Path] = None,
    robustness_output_dir: Optional[str] = None,
    bootstrap_results_path: Optional[str] = None,
):
    """Convenience: plot_robustness_config_summary for every downsample and
    discordance level found for `base_dataset`, in one call."""
    for level in downsample_levels:
        plot_robustness_config_summary(
            base_dataset, "downsample", level, reference_df, reference_calibration_path,
            figure_dir, robustness_output_dir, bootstrap_results_path,
        )
    for level in discordance_levels:
        plot_robustness_config_summary(
            base_dataset, "discordance", level, reference_df, reference_calibration_path,
            figure_dir, robustness_output_dir, bootstrap_results_path,
        )


def _run_config_summary_job(
    base_dataset, perturbation_type, level, reference_df, reference_calibration_path,
    figure_dir, robustness_output_dir, bootstrap_results_path, bootstrap_results_data,
):
    """Module-level (not a closure) so joblib/loky can pickle this call --
    renders and saves one (base_dataset, perturbation_type, level) figure to
    disk and returns only a small (key, error, saved_path) tuple, never the
    Figure itself. `show=False` so this never attempts inline display from
    inside the worker -- even a true subprocess can't show anything back in
    the notebook's own kernel, and if joblib ever ran this in-process (e.g.
    n_jobs resolves to 1), attempting `plt.show()`/switching backends here
    would corrupt the *notebook's own* matplotlib state for every plot after
    it. The batch driver displays each saved PNG back in the main process
    once every worker finishes (see run_config_summary_plots_batch).

    `bootstrap_results_data`, when given, is this base_dataset's own
    pre-filtered slice of ROBUSTNESS_BOOTSTRAP_RESULTS (built once in the
    main process by run_config_summary_plots_batch) -- avoids every worker
    independently decompressing the full ~114MB gzipped JSON just to look up
    a handful of its own condition dirs' entries.

    This is the actual cost driver run_config_summary_plots_batch
    parallelizes: pooled_fits_for_level's sample_density call scales with
    len(fits) x len(score_range) x n_components x n_samples (~100s/figure
    per its own docstring)."""
    key = (base_dataset, perturbation_type, level)
    out_path = (
        Path(figure_dir) / base_dataset
        / f"robustness_config_{base_dataset}_{perturbation_type}_{level}.png"
    ) if figure_dir is not None else None
    try:
        plot_robustness_config_summary(
            base_dataset, perturbation_type, level, reference_df, reference_calibration_path,
            figure_dir=figure_dir, robustness_output_dir=robustness_output_dir,
            bootstrap_results_path=bootstrap_results_path,
            bootstrap_results_data=bootstrap_results_data, show=False,
        )
        return key, None, out_path
    except Exception as e:
        return key, e, out_path


def run_config_summary_plots_batch(
    jobs: List[Tuple[str, pd.DataFrame, Path]],
    downsample_levels: List[int] = (1, 2, 4, 8, 16, 32, 64),
    discordance_levels: List[float] = (0.01, 0.10),
    figure_dir: Optional[Path] = None,
    robustness_output_dir: Optional[str] = None,
    bootstrap_results_path: Optional[str] = None,
    n_jobs: int = -1,
):
    """Parallel version of calling run_config_summary_plots once per
    (base_dataset, reference_df, reference_calibration_path) triple in
    `jobs`. Each (base_dataset, perturbation_type, level) figure is an
    independent, self-contained unit of work, so this flattens every base
    dataset x level combination (e.g. 5 base datasets x 9 levels = 45 jobs)
    into one joblib.Parallel batch, instead of looping base datasets
    serially with only (or not even) their own 9 levels parallelized --
    saturates far more cores when several base datasets are present.

    n_jobs=-1 (default) caps at min(len(tasks), affinity-visible cores) --
    more workers than tasks just wastes process-startup for no benefit.

    ROBUSTNESS_BOOTSTRAP_RESULTS (the per-condition per-bootstrap fits used
    for row 0's density overlay) is decompressed exactly once here, in the
    main process, then sliced per base_dataset (only that dataset's own
    condition-dir keys) before dispatch -- each worker gets just its own
    small slice instead of independently decompressing the full ~114MB
    gzipped JSON, which is what made this "parallel" batch nearly as slow as
    serial before (every one of the up to len(tasks) worker processes paid
    that cost on its own).
    """
    path = bootstrap_results_path or cfg.ROBUSTNESS_BOOTSTRAP_RESULTS
    all_bootstrap_results = None
    if path and Path(path).exists():
        print(f"  Loading {path} once (shared across workers)...")
        all_bootstrap_results = _load_all_bootstrap_results(path)

    def _slice_for(base_dataset):
        if all_bootstrap_results is None:
            return None
        cond_names = {c["dir"].name for c in discover_robustness_conditions(base_dataset, robustness_output_dir)}
        return {k: v for k, v in all_bootstrap_results.items() if k in cond_names}

    tasks = [
        (base_dataset, ptype, level, reference_df, reference_calibration_path,
         figure_dir, robustness_output_dir, bootstrap_results_path, _slice_for(base_dataset))
        for base_dataset, reference_df, reference_calibration_path in jobs
        for ptype, level in (
            [("downsample", lv) for lv in downsample_levels]
            + [("discordance", lv) for lv in discordance_levels]
        )
    ]
    if not tasks:
        return

    import os
    from joblib import Parallel, delayed
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 1
    n_requested = len(tasks) if n_jobs == -1 else n_jobs
    n_jobs = max(1, min(n_requested, n_cores, len(tasks)))

    print(f"  Rendering {len(tasks)} robustness config-summary figure(s) "
          f"across {n_jobs} worker(s)...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_config_summary_job)(*task) for task in tasks
    )

    if is_notebook():
        from IPython.display import Image, display
        for _, err, out_path in results:
            if err is None and out_path is not None and out_path.exists():
                display(Image(filename=str(out_path)))

    for key, err, _ in results:
        if err is not None:
            base_dataset, ptype, level = key
            print(f"  WARNING: config summary failed for {base_dataset}/{ptype}={level}: {err}")


def plot_robustness_confusion_grid(
    matrices: Dict[Tuple[str, float, int], pd.DataFrame], base_dataset: str, figure_dir: Path,
):
    """One aggregate confusion heatmap per (perturbation_type, level) --
    reuses analysis.confusion.make_single_confusion_figure as-is, passing
    that level's 10 seed matrices as the list to aggregate, plus one more for
    the reference baseline. No new heatmap drawing code."""
    by_level: Dict[Tuple[str, float], List[pd.DataFrame]] = {}
    for (ptype, level, seed), mat in matrices.items():
        by_level.setdefault((ptype, level), []).append(mat)

    out_dir = Path(figure_dir) / base_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for (ptype, level), mats in sorted(by_level.items()):
        make_single_confusion_figure(
            mats, [f"seed{i}" for i in range(len(mats))],
            label=base_dataset, figure_dir=out_dir, tag=f"{ptype}_{level}",
        )


# ---------------------------------------------------------------------------
# Bootstrap-count-reduction plotting (tests/benchmark_bootstrap_reduction.py)
#
# That script computes exactly ONE calibration per (dataset, bootstrap-count
# level) -- e.g. N in {1000, 500, 250, 100, 50, 20} -- not repeated seeds per
# level the way the downsample/discordance robustness conditions above do.
# So there is no seed-to-seed spread to measure at a fixed level, and
# deliberately not adding one (re-running each level several times with
# independent bootstrap subsamples just to get that spread) was a conscious
# cost call, not an oversight -- each level already carries its OWN free
# bootstrap-resampling uncertainty: process_component_fits computes
# [p5,p50,p95] across whatever bootstrap fits that level was actually given
# (1000 down to 20), so comparing every level's own band directly already
# shows (a) how the median curve drifts as N shrinks and (b) how each
# level's own reported uncertainty widens as N shrinks -- with zero extra
# fitting beyond what tests/benchmark_bootstrap_reduction.py already ran.
# ---------------------------------------------------------------------------

def plot_bootstrap_reduction_config_summary(
    dataset_name: str,
    reference_df: pd.DataFrame,
    levels_data: Dict[int, dict],
    figure_dir: Optional[Path] = None,
    show: bool = True,
):
    """One figure per dataset, same 3-row layout as
    src.assay_calibration.plot_utils.utils.plot_scoreset_best_config /
    plot_robustness_config_summary above (fits / point assignments / Log
    LR+), but overlaying bootstrap-COUNT LEVELS instead of seeds -- see the
    module-level comment just above for why each level's own [p5,p50,p95]
    band is already a valid, free uncertainty measure here.

    reference_df : DataFrame with "score" and "sample" (pipe-separated
        multi-label, matching analysis.plot_common.sample_matches) columns
        for the FULL, unperturbed dataset -- the fixed population every
        level's histogram/LR+ curve is compared against, mirroring
        plot_robustness_config_summary's reference_df.
    levels_data : {N: {"calib_path": Path, "lr_path": Path,
                        "fits": Optional[List[dict]]}}, one entry per
        bootstrap-count level. The largest N is the reference/baseline level
        (bold curve). "fits" -- that level's own pool of per-bootstrap
        component fits (e.g. sliced from the same precomputed-fits file
        tests/benchmark_bootstrap_reduction.py reads) -- is optional; Row 0
        falls back to histogram-only for any level it's omitted for.
    """
    from src.assay_calibration.plot_utils.utils import (
        sample_density, add_thresholds, log_thresholds_with_ylim_pad,
    )
    from analysis.calibration_plots import _SAMPLE_ORDER, _SAMPLE_LABELS
    from analysis.legacy_fits import _load_calibration_and_lr

    if not levels_data:
        print(f"  SKIP bootstrap-reduction config summary for {dataset_name}: no levels")
        return None

    levels_sorted = sorted(levels_data.keys(), reverse=True)
    baseline_n = levels_sorted[0]
    cmap = plt.get_cmap("viridis")
    level_colors = {N: cmap(i / max(1, len(levels_sorted) - 1)) for i, N in enumerate(levels_sorted)}

    baseline_entry = levels_data[baseline_n]
    with open(baseline_entry["calib_path"]) as f:
        ref_cal = json.load(f)
    ref_data = _load_calibration_and_lr(baseline_entry["calib_path"], baseline_entry["lr_path"])
    score_range = np.asarray(ref_data["score_range"])

    samples_present = [s for s in _SAMPLE_ORDER if sample_matches(reference_df, s).any()]
    n_samples = len(samples_present)
    if n_samples == 0:
        print(f"  SKIP bootstrap-reduction config summary for {dataset_name}: "
              f"reference has no recognized sample categories")
        return None

    fig, ax = plt.subplots(3, n_samples, figsize=(6 * n_samples, 16), squeeze=False,
                            gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    # ---- Row 0: reference histogram + per-level density (if fits given) ----
    for col, sample_label in enumerate(samples_present):
        ax_fit = ax[0, col]
        mask = sample_matches(reference_df, sample_label)
        sample_scores = reference_df.loc[mask, "score"].values
        n = len(sample_scores)
        if n > 1:
            q1, q3 = np.percentile(sample_scores, [25, 75])
            iqr = q3 - q1
            fd_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
            score_width = sample_scores.max() - sample_scores.min()
            bins = min(100, int(score_width / fd_width)) if fd_width > 0 else 50
            bins = max(bins, 10)
            ax_fit.hist(sample_scores, bins=bins, density=True, alpha=0.4, color="#A0A0A0")
        max_hist_density = max([p.get_height() for p in ax_fit.patches]) if ax_fit.patches else 1.0

        for N in levels_sorted:
            fits = levels_data[N].get("fits")
            if not fits:
                continue
            try:
                density = sample_density(score_range, fits, col)
            except IndexError:
                # Same root cause as PTEN_Mighell_2018_clinvar_2018's
                # get_fit_prior IndexError (see
                # tests/benchmark_bootstrap_reduction.py's run_one_level
                # docstring): a precomputed fit's "weights" is shaped for
                # however many samples the dataset had when the
                # precomputed-fits file was generated, which can mismatch
                # the CURRENT reference_df's sample count/order if that
                # dataset's sample composition changed since. Skip just this
                # (dataset, level)'s density curve rather than losing the
                # whole dataset's figure -- the other levels' curves and
                # rows 1/2 (points, LR+, which don't index into "fits") are
                # unaffected.
                print(f"    Row 0 density skipped for N={N}, sample={sample_label}: "
                      f"IndexError (likely a sample-composition mismatch between "
                      f"this level's precomputed fits and the current dataframe)")
                continue
            total = np.nansum(density, axis=1)
            med = np.nanpercentile(total, 50, axis=0)
            ax_fit.plot(score_range, med, color=level_colors[N],
                        linewidth=2.2 if N == baseline_n else 1.2,
                        alpha=1.0 if N == baseline_n else 0.85,
                        label=f"N={N}")

        ax_fit.set_title(f"{_SAMPLE_LABELS.get(sample_label, sample_label)} (n={int(mask.sum()):,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        if max_hist_density:
            ax_fit.set_ylim([0, max_hist_density * 1.1])
        if ax_fit.get_legend_handles_labels()[0]:
            ax_fit.legend(fontsize=7)
        ax_fit.grid(linewidth=0.5, alpha=0.3)

    xlim = ax[0, 0].get_xlim()

    # ---- Row 1: point-assignment overlay across levels ----
    per_level_point_ranges = []
    for N in levels_sorted:
        with open(levels_data[N]["calib_path"]) as f:
            cal = json.load(f)
        pr = cal.get("point_ranges")
        if pr:
            per_level_point_ranges.append((N, pr))

    all_point_values = sorted({int(k) for _, pr in per_level_point_ranges for k in pr.keys()})
    for col in range(n_samples):
        ax_pts = ax[1, col]
        for N, pr in per_level_point_ranges:
            for k, ranges in pr.items():
                point_val = int(k)
                y = all_point_values.index(point_val)
                for sr in ranges:
                    x0 = xlim[0] if np.isneginf(sr[0]) else max(sr[0], xlim[0])
                    x1 = xlim[1] if np.isposinf(sr[1]) else min(sr[1], xlim[1])
                    ax_pts.plot([x0, x1], [y, y], color=level_colors[N],
                                linewidth=5, alpha=0.5, solid_capstyle="butt")
        ax_pts.set_ylim(-1, max(len(all_point_values), 1))
        ax_pts.set_yticks(range(len(all_point_values)),
                           labels=[f"{v:+d}" if v != 0 else "0" for v in all_point_values])
        ax_pts.set_xlim(xlim)
        ax_pts.set_xlabel("Score")
        ax_pts.set_ylabel("Points")
        ax_pts.set_title(f"Point assignments ({len(per_level_point_ranges)} levels overlaid)", fontsize=10)
        ax_pts.grid(linewidth=0.5, alpha=0.3)

    # ---- Row 2: each level's own [p5,p50,p95] LR+ band ----
    point_values_all = sorted({abs(int(k)) for k in ref_cal["point_ranges"].keys()})
    tauP, tauB, ylim_top, ylim_bottom = log_thresholds_with_ylim_pad(ref_cal["prior"], point_values_all)

    for col in range(n_samples):
        ax_lr = ax[2, col]
        for N in levels_sorted:
            entry = levels_data[N]
            data = ref_data if N == baseline_n else _load_calibration_and_lr(entry["calib_path"], entry["lr_path"])
            curve_score = np.asarray(data["score_range"])
            pct = np.asarray(data["log_lr_pct"])
            p5 = np.interp(score_range, curve_score, pct[0])
            p50 = np.interp(score_range, curve_score, pct[1])
            p95 = np.interp(score_range, curve_score, pct[2])
            lw = 2.2 if N == baseline_n else 1.2
            ax_lr.plot(score_range, p50, color=level_colors[N], linewidth=lw, label=f"N={N}")
            ax_lr.fill_between(score_range, p5, p95, color=level_colors[N], alpha=0.10)
        add_thresholds(tauP, tauB, ax_lr)
        ax_lr.set_ylim([ylim_bottom, ylim_top])
        ax_lr.set_xlim(xlim)
        ax_lr.set_xlabel("Score")
        ax_lr.set_ylabel("Log LR+")
        ax_lr.set_title("Log LR+ (median + own [p5,p95] band per level)", fontsize=10)
        if col == n_samples - 1:
            ax_lr.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        ax_lr.grid(linewidth=0.5, alpha=0.3)

    fig.suptitle(f"{dataset_name}: bootstrap-count reduction ({len(levels_sorted)} levels: {levels_sorted})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if figure_dir is not None:
        out_dir = Path(figure_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bootstrap_reduction_config_{dataset_name}.png"
        if show:
            save_and_show(fig, out_path)
        else:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Fit-number (restart-count) comparison plotting
# (tests/benchmark_num_fits_dataframe.py's summary.csv)
# ---------------------------------------------------------------------------

def compute_delta_std_column(summary_df: pd.DataFrame, train_lls_path) -> pd.DataFrame:
    """Add a "delta_std" column: `delta` (raw train_ll units, not
    interpretable across datasets with very different likelihood scales)
    divided by that (dataset, n_c)'s OWN restart-to-restart standard
    deviation (computed across all its valid best_of_100 restarts, from
    tests/benchmark_num_fits_dataframe.py's train_lls.json). This turns
    "how much worse is best-of-N" into a dimensionless "how many SDs of this
    dataset's own restart-to-restart noise" measure, comparable across
    datasets.

    train_lls.json is keyed "{dataset}|{n_c}c" (see
    tests/benchmark_num_fits_dataframe.py's train_lls_all dict) -- must
    match summary_df's own "dataset"/"n_c" columns exactly.
    """
    with open(train_lls_path) as f:
        train_lls = json.load(f)

    std_by_key = {}
    for key, lls in train_lls.items():
        arr = np.asarray(lls, dtype=float)
        valid = arr[np.isfinite(arr)]
        std_by_key[key] = float(valid.std()) if len(valid) > 1 else np.nan

    df = summary_df.copy()
    keys = df["dataset"].astype(str) + "|" + df["n_c"].astype(int).astype(str) + "c"
    df["restart_std"] = keys.map(std_by_key)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["delta_std"] = df["delta"] / df["restart_std"]
    df.loc[~np.isfinite(df["restart_std"]) | (df["restart_std"] == 0), "delta_std"] = np.nan

    # geometric_mean_lr_pct = 100 * exp(delta): mathematically bounded in
    # (0%, 100%] regardless of the raw LL values' sign/scale (delta is a
    # DIFFERENCE of two per-observation average log-densities, so exp(delta)
    # is the ratio of their GEOMETRIC MEANS -- the difference structure is
    # what makes this scale-invariant, not anything about the LL values
    # themselves). Do NOT read this as a "quality percentage", though:
    # densities aren't probabilities (they can exceed 1), so "90%" here does
    # NOT mean "90% as many correct classifications" or any other
    # intuitively-linear/bounded notion of quality -- it is specifically a
    # geometric-mean likelihood *ratio*, nothing more. delta_std above is
    # the more defensible interpretable metric; keep this one only as a
    # secondary, explicitly-labeled number.
    df["geometric_mean_lr_pct"] = 100.0 * np.exp(df["delta"])
    return df


def summarize_delta_std_table(summary_df: pd.DataFrame, levels=(1, 8, 20, 50, 100),
                              metric: str = "delta_std") -> pd.DataFrame:
    """Median (IQR: 25th-75th percentile) of `metric` across every (dataset,
    n_c) row, at each of `levels` -- the exact numbers behind the
    "fits-per-bootstrap" table in docs/configuration.md. Call
    compute_delta_std_column first if `metric="delta_std"` isn't already a
    column."""
    rows = []
    for N in levels:
        vals = summary_df.loc[summary_df["num_fits"] == N, metric].values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            rows.append({"num_fits": N, "median": np.nan, "p25": np.nan, "p75": np.nan, "n": 0})
            continue
        p25, p50, p75 = np.percentile(vals, [25, 50, 75])
        rows.append({"num_fits": N, "median": p50, "p25": p25, "p75": p75, "n": len(vals)})
    return pd.DataFrame(rows)


def plot_fit_number_comparison_curve(
    summary_df: pd.DataFrame,
    metric: str = "delta",
    figure_dir: Optional[Path] = None,
    label: str = "all_datasets",
):
    """Median + IQR ribbon of `metric` (default "delta" = mean_best_of_N
    train_ll minus the best-of-all-fits baseline, from
    tests/benchmark_num_fits_dataframe.py's summary.csv) vs. num_fits
    (restart count), pooling every (dataset, n_c) row at each level --
    same median-line + IQR-ribbon + scatter + dashed-reference-line design
    as plot_downsample_robustness_curve above, adapted from "seed spread at
    one dataset, one level" to "cross-dataset spread at one restart count".
    """
    levels = sorted(summary_df["num_fits"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))

    medians, p25s, p75s, xs_scatter, ys_scatter = [], [], [], [], []
    for level in levels:
        vals = summary_df.loc[summary_df["num_fits"] == level, metric].values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            medians.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
            continue
        p25, p50, p75 = np.percentile(vals, [25, 50, 75])
        medians.append(p50); p25s.append(p25); p75s.append(p75)
        xs_scatter.extend([level] * len(vals))
        ys_scatter.extend(vals.tolist())

    ax.fill_between(levels, p25s, p75s, alpha=0.25, color="C0")
    ax.plot(levels, medians, marker="o", color="C0", label="median")
    ax.scatter(xs_scatter, ys_scatter, alpha=0.3, s=12, color="C0")

    # "No degradation" reference: 0 for the two difference-based metrics
    # (delta, delta_std), 100 for geometric_mean_lr_pct (a ratio expressed
    # as a percentage, not a difference) -- see
    # compute_delta_std_column's docstring for why this is bounded (0%,
    # 100%] and, just as importantly, why it is NOT a "quality percentage".
    ref_value = 100.0 if metric == "geometric_mean_lr_pct" else 0.0
    ax.axhline(ref_value, linestyle="--", color="black", alpha=0.6, label="no degradation")

    _YLABELS = {
        "delta": "delta (best_of_N − best_of_all train_ll)",
        "delta_std": "delta / restart-to-restart SD (dimensionless)",
        "geometric_mean_lr_pct": "geometric-mean likelihood ratio vs. best_of_all (%)",
    }
    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Restart count (num_fits)")
    ax.set_ylabel(_YLABELS.get(metric, metric))
    ax.set_title(f"{label}: fit quality vs. restart count")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.5, alpha=0.3)
    fig.tight_layout()

    if figure_dir is not None:
        out_dir = Path(figure_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fit_number_comparison_{label}.png"
        save_and_show(fig, out_path)
        print(f"  Saved: {out_path}")
    return fig
