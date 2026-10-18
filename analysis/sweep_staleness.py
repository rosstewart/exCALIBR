"""
Which datasets of the bootstrap-count (Supplementary Table S5) and restart-
count (Supplementary Table S6) sweeps need rerunning, so only those are rerun.

A calibrated dataset is stale for a sweep when any of:
  - it is missing from the sweep's output;
  - any fit-relevant column of its rows differs between the master TSV the
    sweep read (automatic backups of that date, SWEEP_INPUT_TSV) and the
    current TSV -- annotation-only columns (REVEL, MutPred2, AlphaMissense,
    ClinGen-repo, author annotations, ClinVar review dates, ...) are ignored,
    since the fits never read them;
  - its selected (n_c, benign_method) differs between the config the sweep
    used and the current dataset config (skipped if that file is gone).
A dataset whose sweep output was rewritten after the current master TSV was
last modified (i.e. rerun on the current data) is never stale on the input
test.
(Calibration-file timestamps alone are not used: the whole canonical
pipeline was rerun after both sweeps, so every dataset would look stale.)

    python -m analysis.sweep_staleness            # print stale datasets + rerun commands
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from analysis import config as cfg

SWEEP_CONFIG_USED = str(Path(cfg.DATASET_CONFIGS).parent / "dataset_configs_jul_2026.json")

_TSV = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
# Master TSV content each sweep read: the automatic backups taken when the
# TSV was next replaced (bootstrap sweep ran 2026-07-31 09:39-20:41, restart
# sweep finished 2026-08-01 04:47).
SWEEP_INPUT_TSV = {
    "bootstrap_reduction": _TSV + ".bak-20260731182344",
    "restart": _TSV + ".bak-20260813213300",
}
# Columns the fitting code (Scoreset / load_dataset_from_df) never reads.
# rna_score and nucleotide_or_aa are only used downstream (evidence export).
_ANNOTATION_ONLY = ("REVEL", "MutPred2", "AM_", "ClinGen_repo", "auth_", "_train", "HGNC",
                    "clinvar_date_last_reviewed", "Mondo", "rna_score", "nucleotide_or_aa")


def fit_input_changes(old_tsv: str, datasets: Iterable[str]) -> Dict[str, str]:
    """dataset -> description of fit-relevant row changes vs `old_tsv`."""
    import pandas as pd
    from analysis.discovery import load_master_df
    cur = load_master_df(str(cfg.DATASET_TSV))
    old = pd.read_csv(old_tsv, sep="\t", low_memory=False, compression="gzip")
    cols = sorted(c for c in set(old.columns) & set(cur.columns) if not any(k in c for k in _ANNOTATION_ONLY))
    out = {}
    for d in datasets:
        name = d.replace("_clinvar_2018", "")
        a = old[old["Dataset"] == name]
        b = cur[cur["Dataset"] == name]
        if a.empty:
            out[d] = "absent from the TSV the sweep read"
            continue
        if len(a) != len(b):
            out[d] = f"row count {len(a)} -> {len(b)}"
            continue
        a = a[cols].astype(str).sort_values(cols).reset_index(drop=True)
        b = b[cols].astype(str).sort_values(cols).reset_index(drop=True)
        diff = [c for c in cols if not a[c].equals(b[c])]
        if diff:
            out[d] = "changed: " + ", ".join(diff[:6]) + (" ..." if len(diff) > 6 else "")
    return out


def _component(configs: Dict, dataset: str):
    c = configs.get(dataset) or configs.get(dataset.replace("_clinvar_2018", "")) or {}
    return (c.get("n_c"), c.get("benign_method"))


def _config_changes(datasets: Iterable[str]) -> Dict[str, str]:
    # The sweeps' scripts defaulted to dataset_configs_jul_2026.json, which no
    # longer exists under that name; without it this test is skipped (reported).
    if not Path(SWEEP_CONFIG_USED).exists():
        print(f"  NOTE: {SWEEP_CONFIG_USED} not found; component-change test skipped")
        return {}
    old = json.load(open(SWEEP_CONFIG_USED))
    new = json.load(open(cfg.DATASET_CONFIGS))
    out = {}
    for d in datasets:
        if _component(old, d) != _component(new, d):
            out[d] = f"selected component changed {_component(old, d)} -> {_component(new, d)}"
    return out


def bootstrap_reduction_stale(datasets: Iterable[str], sweep_dir: str = None) -> Dict[str, str]:
    from tests.benchmark_bootstrap_reduction import _KNOWN_NAME_ALIASES
    sweep_dir = Path(sweep_dir or cfg.BOOTSTRAP_REDUCTION_OUTPUT_DIR)
    found = {}
    for d in sweep_dir.iterdir() if sweep_dir.exists() else []:
        # Covered only if every level produced a calibration (a level folder
        # can exist with nothing in it when that level's fit failed).
        levels = [lv for lv in d.glob("level_*") if lv.is_dir()]
        if d.is_dir() and levels and all(any(lv.glob("*_calibration.json")) for lv in levels):
            mt = max(p.stat().st_mtime for p in d.rglob("*_calibration.json"))
            found[_KNOWN_NAME_ALIASES.get(d.name, d.name)] = mt
    datasets = list(datasets)
    stale = _config_changes(datasets)
    tsv_mtime = Path(cfg.DATASET_TSV).stat().st_mtime
    stale.update(fit_input_changes(SWEEP_INPUT_TSV["bootstrap_reduction"],
                                   [d for d in datasets if d in found and found[d] <= tsv_mtime]))
    for d in datasets:
        if d not in found:
            stale[d] = "missing from sweep"
    return dict(sorted(stale.items()))


def restart_sweep_name_map(sweep_names) -> Dict[str, str]:
    """Restart-sweep dataset name -> pipeline dataset name."""
    from hpc.prepare import _requires_2018
    from analysis.discovery import load_master_df
    from tests.benchmark_bootstrap_reduction import _KNOWN_NAME_ALIASES
    df = load_master_df(str(cfg.DATASET_TSV))
    tsv_names = set(df["Dataset"].unique())
    out = {}
    for raw in sweep_names:
        d = _KNOWN_NAME_ALIASES.get(raw, raw)
        out[raw] = d + "_clinvar_2018" if d in tsv_names and _requires_2018(df, d) else d
    return out


def restart_sweep_found(summary_csv: str = None) -> set:
    """Sweep dataset names mapped to pipeline names (adds the _clinvar_2018
    suffix where the sweep used 2018 ClinVar controls)."""
    import pandas as pd
    from hpc.prepare import _requires_2018
    from analysis.discovery import load_master_df
    from tests.benchmark_bootstrap_reduction import _KNOWN_NAME_ALIASES
    summary = pd.read_csv(summary_csv or cfg.FIT_NUMBER_COMPARISON_SUMMARY_CSV)
    df = load_master_df(str(cfg.DATASET_TSV))
    tsv_names = set(df["Dataset"].unique())
    out = set()
    for d in summary["dataset"].unique():
        d = _KNOWN_NAME_ALIASES.get(d, d)
        if d not in tsv_names:  # old name no longer in the dataframe: counts as not covered
            out.add(d)
            continue
        out.add(d + "_clinvar_2018" if _requires_2018(df, d) else d)
    return out


def restart_rerun_since_tsv(datasets: Iterable[str], checkpoint_dir: Path) -> set:
    """Datasets whose restart-sweep checkpoints were all written after the
    current master TSV was last modified (named as checkpoint_path_for does)."""
    import re
    tsv_mtime = Path(cfg.DATASET_TSV).stat().st_mtime
    out = set()
    for d in datasets:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", d.replace("_clinvar_2018", ""))
        ckpts = list(Path(checkpoint_dir).glob(f"{safe}__*c_*.pkl"))
        if ckpts and min(c.stat().st_mtime for c in ckpts) > tsv_mtime:
            out.add(d)
    return out


def restart_sweep_stale(datasets: Iterable[str], summary_csv: str = None,
                        selected_nc: Dict[str, int] = None) -> Dict[str, str]:
    """Datasets whose restart-sweep results are missing or out of date.
    `selected_nc` ({dataset: component count}, e.g. from
    analysis.discovery.resolve_component): a dataset also counts as stale
    when the sweep has no rows at its selected component count (Supp Table
    S6 uses only those rows)."""
    import pandas as pd

    summary_csv = summary_csv or cfg.FIT_NUMBER_COMPARISON_SUMMARY_CSV
    found = restart_sweep_found(summary_csv) if Path(summary_csv).exists() else set()
    datasets = list(datasets)
    stale = _config_changes(datasets)
    rerun = restart_rerun_since_tsv(datasets, Path(summary_csv).parent / "checkpoints")
    stale.update(fit_input_changes(SWEEP_INPUT_TSV["restart"], [d for d in datasets if d in found and d not in rerun]))
    if selected_nc and Path(summary_csv).exists():
        summary = pd.read_csv(summary_csv, usecols=["dataset", "n_c"])
        names = restart_sweep_name_map(summary["dataset"].unique())
        swept = {(names[d], int(n)) for d, n in zip(summary["dataset"], summary["n_c"])}
        for d in datasets:
            if d in found and d in selected_nc and (d, int(selected_nc[d])) not in swept:
                stale.setdefault(d, f"no sweep rows at the selected {int(selected_nc[d])}-component model")
    for d in datasets:
        if d not in found:
            stale[d] = "missing from sweep"
    return dict(sorted(stale.items()))


def rerun_commands(bsr_stale: Dict[str, str], restart_stale: Dict[str, str]) -> Dict[str, str]:
    bsr = " ".join(bsr_stale)
    rs = " ".join(d.replace("_clinvar_2018", "") for d in restart_stale)
    return {
        "bootstrap_reduction": (
            f"python tests/benchmark_bootstrap_reduction.py --output-dir {cfg.BOOTSTRAP_REDUCTION_OUTPUT_DIR} "
            f"--precomputed-fits {cfg.PRECOMPUTED_FITS} --config-file {cfg.DATASET_CONFIGS} --update-existing --datasets {bsr}"
        ) if bsr else "",
        "restart_sweep": (
            f"python tests/benchmark_num_fits_dataframe.py --results-dir {Path(cfg.FIT_NUMBER_COMPARISON_SUMMARY_CSV).parent} "
            f"--config-file {cfg.DATASET_CONFIGS} --update-existing --datasets {rs}"
        ) if rs else "",
    }


if __name__ == "__main__":
    from analysis.discovery import discover_outputs, DEFAULT_EXCLUDED_DATASETS
    tree, _, _ = discover_outputs(Path(cfg.OUTPUT_DIR))
    ds = sorted(d for d in tree if d not in DEFAULT_EXCLUDED_DATASETS)
    b, r = bootstrap_reduction_stale(ds), restart_sweep_stale(ds)
    for name, st in (("bootstrap-count sweep (Supp Table S5)", b), ("restart-count sweep (Supp Table S6)", r)):
        print(f"\n{name}: {len(st)} of {len(ds)} datasets stale")
        for d, why in st.items():
            print(f"  {d}: {why}")
    for k, v in rerun_commands(b, r).items():
        print(f"\n{k} rerun:\n  {v}")


# ---------------------------------------------------------------------------
# Side runs of the pipeline (pathomechanism prior, GMM baseline, SpliceAI
# ablation), all run on 2026-08-19 from the master TSV of that date.
# ---------------------------------------------------------------------------
# Datasets whose fit inputs changed between that TSV and the current one,
# found on 2026-10-06: the pathomechanism run's *_variants.csv (same exporter
# as the canonical run) matched the canonical output exactly (variant_id,
# score, sample, is_vus, is_snv) for every calibrated dataset but these three
# (rows dropped in the current TSV). The GMM baseline and SpliceAI runs read
# the same TSV but export variants differently (ClinVar release / splice
# filter), so they cannot be compared to the canonical output directly.
SIDE_RUN_CHANGED_DATASETS = {
    "LDLR_Tabet_2025_abundance": "rows dropped since 2026-08-19 (2676 -> 1811 exported variants)",
    "LDLR_Tabet_2025_uptake": "rows dropped since 2026-08-19 (2745 -> 1878 exported variants)",
    "TSC2_IGVF": "rows dropped since 2026-08-19 (1927 -> 1786 exported variants)",
}
_FIT_INPUT_COLS = ["variant_id", "score", "sample", "is_vus", "is_snv"]


def _rerun_after_tsv(dataset_dirs) -> bool:
    """True when every output file in `dataset_dirs` postdates the current master TSV."""
    tsv_mtime = Path(cfg.DATASET_TSV).stat().st_mtime
    files = [f for d in dataset_dirs for f in Path(d).glob("*_calibration.json")]
    return bool(files) and min(f.stat().st_mtime for f in files) > tsv_mtime


def side_run_stale(run_root, datasets: Iterable[str], conditions: Iterable[str] = ("",)) -> Dict[str, str]:
    """dataset -> reason, for a side run under `run_root` (or each condition
    subfolder of it): datasets in SIDE_RUN_CHANGED_DATASETS not rerun since
    the current master TSV was written. Coverage (missing datasets) is checked
    separately by the caller."""
    root = Path(run_root)
    stale = {}
    for d in datasets:
        if d not in SIDE_RUN_CHANGED_DATASETS:
            continue
        dirs = [root / c / d for c in conditions if (root / c / d).exists()]
        if not _rerun_after_tsv(dirs):
            stale[d] = SIDE_RUN_CHANGED_DATASETS[d]
    return stale


def canonical_input_changes(run_root, datasets: Iterable[str], tree, model_selections, dataset_configs) -> Dict[str, str]:
    """dataset -> reason, comparing a side run's *_variants.csv fit-input
    columns with the canonical run's, for runs that export variants exactly
    as the canonical pipeline does (e.g. the pathomechanism rerun)."""
    import pandas as pd
    from analysis.discovery import resolve_component

    def load(p):
        x = pd.read_csv(p, usecols=lambda c: c in _FIT_INPUT_COLS, low_memory=False)
        return x[_FIT_INPUT_COLS].astype(str).sort_values(_FIT_INPUT_COLS).reset_index(drop=True)

    out = {}
    for d in datasets:
        comp = resolve_component(d, list(tree[d].keys()), model_selections, dataset_configs)
        canon = Path(cfg.OUTPUT_DIR) / d / f"{d}_{comp}_variants.csv"
        side = Path(run_root) / d / canon.name
        if not side.exists():
            cands = sorted((Path(run_root) / d).glob("*_variants.csv"))
            if not cands:
                out[d] = "missing from run"
                continue
            side = cands[0]
        a, b = load(canon), load(side)
        if len(a) != len(b):
            out[d] = f"exported variants {len(b)} -> {len(a)}"
        elif not a.equals(b):
            out[d] = "changed: " + ", ".join(c for c in _FIT_INPUT_COLS if not a[c].equals(b[c]))
    return out
