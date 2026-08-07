"""
Shared analyze-core for the FGFR/TP53/LABEL-seq/CARD11/predictor-mv/combined
gene-sets: build an MVCalibrationAnalysis and run it.

Every ingestion module in multivariate_data/ uses the same fixed sample-role
ordering (pathogenic=0, benign=1, gnomad=2, synonymous=3, with any extra
samples -- TP53's RPV, CARD11's BENTA/CADINS -- appended after), so one
function covers every gene-set's analyze driver below instead of
reconstructing the same MVCalibrationAnalysis call per gene-set.
"""

import hashlib
import os
import pickle
from collections import Counter
from typing import Optional

import pandas as pd

from .mv_calibration import MVCalibrationAnalysis, _PARTIAL_PATTERN_MODES

# Display names for the four partial_pattern_mode values, matching how
# they've been discussed: "none" (no gate at all -- always trust the global
# weights), "old_gate" (historical asymmetric behavior), "pu_unmix"
# (population-unmixing when one side lacks pattern coverage), "conservative"
# (project down to the largest jointly-supported sub-pattern).
MODE_DISPLAY_NAMES = {
    "trust_global": "none",
    "gate": "old_gate",
    "local_unmixing": "pu_unmix",
    "project": "conservative",
}


def build_gene_set_analysis(
    ms,
    gene: str,
    fits_json_path: str,
    dataset_name: Optional[str] = None,
    dataset_suffix: str = "_mv",
    auxiliary_pathogenic_indices: Optional[list] = None,
    benign_method: str = "avg",
    mvcal_kwargs: Optional[dict] = None,
) -> MVCalibrationAnalysis:
    """Build (but do not run) an MVCalibrationAnalysis for one gene's fits.

    ``dataset_name`` should be the exact results-JSON key hpc/prepare.py
    saved this gene's fits under (see
    multivariate_data/common.gene_set_dataset_label) -- pass it explicitly
    rather than relying on ``dataset_suffix`` reconstructing
    ``f"{gene}{dataset_suffix}"``, since this package's gene-set labels
    (``"{gene}_{gene_set}_mv"``) don't match that pattern for
    gene_set != "" (e.g. FGFR1's fits are keyed "FGFR1_fgfr_mv", not
    "FGFR1_mv"). ``dataset_suffix`` still applies where dataset_name is
    left as None (e.g. predictor-mv's "_predictors_mv" convention already
    matches its own dataset_suffix by construction).
    """
    return MVCalibrationAnalysis(
        ms, gene, fits_json_path,
        pathogenic_idx=0, benign_idx=1, gnomad_idx=2, synonymous_idx=3,
        auxiliary_pathogenic_indices=auxiliary_pathogenic_indices,
        benign_method=benign_method,
        dataset_name=dataset_name,
        dataset_suffix=dataset_suffix,
        **(mvcal_kwargs or {}),
    )


def run_gene_set_analysis(
    ms,
    gene: str,
    fits_json_path: str,
    dataset_name: Optional[str] = None,
    dataset_suffix: str = "_mv",
    auxiliary_pathogenic_indices: Optional[list] = None,
    benign_method: str = "avg",
    mvcal_kwargs: Optional[dict] = None,
    **run_kwargs,
) -> MVCalibrationAnalysis:
    """Build and run an MVCalibrationAnalysis for one gene's fits.

    ``run_kwargs`` are forwarded to ``MVCalibrationAnalysis.run(...)``
    (e.g. ``path_percentile``, ``min_valid_boots``,
    ``reestimate_marginal_weights``).
    """
    analysis = build_gene_set_analysis(
        ms, gene, fits_json_path,
        dataset_name=dataset_name, dataset_suffix=dataset_suffix,
        auxiliary_pathogenic_indices=auxiliary_pathogenic_indices,
        benign_method=benign_method, mvcal_kwargs=mvcal_kwargs,
    )
    analysis.run(**run_kwargs)
    return analysis


def run_gene_set_analysis_cached(
    ms,
    gene: str,
    fits_json_path: str,
    cache_dir: str,
    dataset_name: Optional[str] = None,
    dataset_suffix: str = "_mv",
    auxiliary_pathogenic_indices: Optional[list] = None,
    benign_method: str = "avg",
    mvcal_kwargs: Optional[dict] = None,
    force_recompute: bool = False,
    **run_kwargs,
) -> MVCalibrationAnalysis:
    """Like run_gene_set_analysis, but caches ``analysis.results`` (the
    output of ``.run()``, covering every config at once) to disk.

    ``.run()`` re-parses every bootstrap fit in ``fits_json_path`` and
    re-estimates priors/thresholds/points for all configs on every call --
    cheap next to precompute_mv_plot_data_cached's per-config bootstrap
    sweeps, but not free, and pointless to repeat when only plot aesthetics
    are being iterated on. Cache key covers everything that affects the
    result: the gene, the fits file's identity (path + mtime + size, so a
    re-fit/re-aggregate invalidates old cache entries automatically), and
    every run_kwargs value (path_percentile, partial_pattern_mode, etc.).
    """
    analysis = build_gene_set_analysis(
        ms, gene, fits_json_path,
        dataset_name=dataset_name, dataset_suffix=dataset_suffix,
        auxiliary_pathogenic_indices=auxiliary_pathogenic_indices,
        benign_method=benign_method, mvcal_kwargs=mvcal_kwargs,
    )

    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(fits_json_path)
    key_material = repr({
        "gene": gene,
        "fits_json_path": fits_json_path,
        "fits_mtime": st.st_mtime,
        "fits_size": st.st_size,
        "dataset_name": dataset_name,
        "dataset_suffix": dataset_suffix,
        "auxiliary_pathogenic_indices": auxiliary_pathogenic_indices,
        "benign_method": benign_method,
        "run_kwargs": sorted(run_kwargs.items()),
    })
    key = hashlib.sha1(key_material.encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{gene}_run_{key}.pkl")

    if not force_recompute and os.path.exists(cache_path):
        print(f"  Loading cached analysis.run() results: {cache_path}")
        with open(cache_path, "rb") as f:
            analysis.results = pickle.load(f)
        return analysis

    analysis.run(**run_kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump(analysis.results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached analysis.run() results: {cache_path}")
    return analysis


def report_configs_and_modes(
    analysis: MVCalibrationAnalysis,
    modes=_PARTIAL_PATTERN_MODES,
    mode_labels: Optional[dict] = None,
    **run_kwargs,
) -> pd.DataFrame:
    """One flat table, one row per (config, mode): n_boots, prior, C_path/
    C_ben, path/negative correct-and-wrong %, point distribution.

    Runs ``analysis`` once per mode (via compare_partial_pattern_modes) so
    every mode is evaluated against the exact same underlying bootstrap
    fits -- only the partial-pattern handling differs between rows.
    ``run_kwargs`` (path_percentile, min_valid_boots, etc.) are held fixed
    across all modes.
    """
    mode_labels = mode_labels if mode_labels is not None else MODE_DISPLAY_NAMES
    all_results = analysis.compare_partial_pattern_modes(modes=modes, **run_kwargs)

    rows = []
    for mode, results in all_results.items():
        label = mode_labels.get(mode, mode)
        for config, r in results.items():
            if r is None:
                rows.append({"config": config, "mode": label, "status": "failed"})
                continue
            rows.append({
                "config": config,
                "mode": label,
                "n_boots": r["n_valid"],
                "prior": round(r["median_prior"], 4),
                "C_path": round(r["C_path"], 1),
                "C_ben": round(r["C_ben"], 1),
                "path_correct%": round(r["path_correct"] * 100, 1),
                "path_wrong%": round(r["path_wrong"] * 100, 1),
                "neg_correct%": round(r["neg_correct"] * 100, 1),
                "neg_wrong%": round(r["neg_wrong"] * 100, 1),
                "point_dist": dict(sorted(Counter(r["points"]).items())),
            })
    return pd.DataFrame(rows)
