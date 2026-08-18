"""
Maintainer-only tool: package a minimal, MSH2-only bundle that lets someone
else reproduce Figure 4 via `analysis.figure4.driver` without needing the
full pipeline output, master TSV, or bootstrap-fits file.

This script itself needs the full-scale pipeline artifacts (and so, unlike
`analysis.figure4.driver`, freely depends on `analysis.config` for their
locations) -- only its *output* (the small bundle directory) is meant to be
handed to someone else. Run once whenever the MSH2 calibration or the
pipeline-wide confusion matrices change.

    cd exCALIBR && python -m analysis.figure4.export_msh2_bundle --bundle-dir /path/to/bundle

Writes, under --bundle-dir:
    calibration/MSH2_Jia_2021_clinvar_2018_<comp>_calibration.json
    calibration/MSH2_Jia_2021_clinvar_2018_<comp>_lr_values.json.gz (trimmed)
    scoreset_2018.csv.gz (+ scoreset_2018_snv.csv.gz)  -- panel a's Scoreset,
                                see analysis.figure4.scoreset_io
    scoreset_2025.csv.gz (+ scoreset_2025_snv.csv.gz)  -- panel b/e's Scoreset
    dataset_configs.json      ({"MSH2_Jia_2021_clinvar_2018": {...}} only)
    bootstrap_fits.json.gz    (precomputed fits, MSH2_Jia_2021_clinvar_2018 key only)
    revel/                    (the 4 REVEL files _build_panel_ef_data reads)
    panel_c.json              (precomputed cross-dataset confusion aggregate --
                                see analysis.figure4.panel_c_io -- the one thing
                                Figure 4 needs that genuinely isn't MSH2-specific)
    dataset.tsv.gz            (MSH2_Jia_2021 rows of the master TSV -- only
                                written with --include-dataset-tsv; the
                                scoreset_*.csv.gz files above already cover
                                what driver.py needs from it, at a fraction
                                of the size, so most bundles can skip it)

Reproduce with:
    python -m analysis.figure4.driver --bundle bundle/ --figure-dir out/
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

from analysis import config as cfg
from analysis.author_labels import attach_author_labels
from analysis.confusion import (
    build_author_confusion_matrix,
    build_author_vus_coverage,
    build_confusion_matrix,
    build_vus_coverage,
    _aggregate_coverage_pct,
)
from analysis.discovery import discover_outputs, load_all_variants
from analysis.figure4.driver import MSH2_DATASET
from analysis.figure4.panel_c_io import save_panel_c_bundle
from analysis.legacy_fits import resolve_component_for

MSH2_PIPELINE_KEY = f"{MSH2_DATASET}_clinvar_2018"


def _export_calibration_files(output_dir: str, dataset_configs_path: str, bundle_dir: Path) -> None:
    n_c, benign_method = resolve_component_for(MSH2_PIPELINE_KEY, output_dir, dataset_configs_path)
    comp = f"{n_c}_{benign_method}"
    src_dir = Path(output_dir) / MSH2_PIPELINE_KEY
    calib_src = src_dir / f"{MSH2_PIPELINE_KEY}_{comp}_calibration.json"
    lr_src = src_dir / f"{MSH2_PIPELINE_KEY}_{comp}_lr_values.json.gz"
    if not calib_src.exists() or not lr_src.exists():
        # older naming convention -- bare n_c, no benign_method suffix
        calib_src = src_dir / f"{MSH2_PIPELINE_KEY}_{n_c}_calibration.json"
        lr_src = src_dir / f"{MSH2_PIPELINE_KEY}_{n_c}_lr_values.json.gz"
    if not calib_src.exists() or not lr_src.exists():
        raise FileNotFoundError(f"MSH2 calibration/LR files not found under {src_dir}")

    dest_dir = bundle_dir / "calibration" / MSH2_PIPELINE_KEY
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(calib_src, dest_dir / calib_src.name)

    # lr_values.json.gz carries several bootstrap-percentile arrays
    # (log_lr_plus_p5/p50/p95, log_lr_benign_p5/p50/p95, priors_pct,
    # prior_benign_pct, ...) that legacy_fits._load_calibration_and_lr loads
    # into indv_summary, but panels.py only ever reads `score_range` out of
    # it for Figure 4 (point_ranges/prior come from calibration.json
    # instead -- see build_figure4's docstring). _load_calibration_and_lr
    # does still require log_lr_plus_p5/p50/p95 + priors_pct (+, subtly,
    # "prior" itself: `lr.get("priors_pct", [lr["prior"]] * 3)` evaluates
    # `lr["prior"]` to build its *default* value even when "priors_pct" is
    # already present, since Python evaluates .get()'s arguments eagerly --
    # so "prior" must stay even though nothing downstream reads it) to
    # exist or it KeyErrors, so those are kept with their real values --
    # only the *_benign_* / prior_benign_pct / dataset / n_c fields
    # (genuinely unused by this loader, not just by Figure 4) are dropped.
    with gzip.open(lr_src, "rt", encoding="utf-8") as f:
        lr_full = json.load(f)
    trimmed = {
        k: lr_full[k]
        for k in ("score_range", "log_lr_plus_p5", "log_lr_plus_p50", "log_lr_plus_p95", "priors_pct", "prior")
        if k in lr_full
    }
    lr_dest = dest_dir / lr_src.name
    with gzip.open(lr_dest, "wt", encoding="utf-8") as f:
        json.dump(trimmed, f)
    print(f"  Copied {calib_src.name}, wrote trimmed {lr_src.name} "
          f"({lr_src.stat().st_size / 1e3:.0f} KB -> {lr_dest.stat().st_size / 1e3:.0f} KB)")


def _export_dataset_tsv(dataset_tsv: str, bundle_dir: Path) -> None:
    """Only needed as a fallback for driver.py's dataset_tsv-based Scoreset
    reconstruction -- skipped by default (see --include-dataset-tsv) now that
    _export_scoresets below covers what panels a/b/e actually need from it,
    at a small fraction of the size."""
    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    df_full = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)
    df_msh2 = df_full[df_full["Dataset"] == MSH2_DATASET]
    if df_msh2.empty:
        raise ValueError(f"'{MSH2_DATASET}' not found in {dataset_tsv}")
    out_path = bundle_dir / "dataset.tsv.gz"
    df_msh2.to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {out_path} ({len(df_msh2):,} rows, vs. {len(df_full):,} in the full TSV)")


def _export_scoresets(
    output_dir: str, dataset_tsv: str, precomputed_fits: str, dataset_configs_path: str, bundle_dir: Path,
) -> None:
    """Build both MSH2 Scoresets (2018-ClinVar-release for panel a's mixture
    fit, current-release for panel b/e's "All SNVs" distribution) the same
    way driver.py itself would, then cache just the 4 attributes
    analysis.figure4.panels actually reads off each -- see
    analysis.figure4.scoreset_io's docstring -- instead of the recipient
    needing the full master TSV + rerunning this same pipeline-dataframe
    construction (splice filtering, ClinVar-release parsing, ...) themselves.
    """
    from analysis.legacy_fits import load_scoreset_and_fits
    from analysis.figure4.scoreset_io import save_scoreset_bundle

    scoreset_2018, _, _, _, n_c, _, _ = load_scoreset_and_fits(
        MSH2_DATASET, output_dir=output_dir, dataset_tsv=dataset_tsv,
        precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
        pipeline_dataset=MSH2_PIPELINE_KEY, clinvar_release="2018",
    )
    scoreset, *_ = load_scoreset_and_fits(
        MSH2_DATASET, output_dir=output_dir, dataset_tsv=dataset_tsv,
        precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
        pipeline_dataset=MSH2_PIPELINE_KEY, clinvar_release="2025", n_c=n_c,
    )

    path_2018 = bundle_dir / "scoreset_2018.csv.gz"
    path_2025 = bundle_dir / "scoreset_2025.csv.gz"
    save_scoreset_bundle(path_2018, scoreset_2018)
    save_scoreset_bundle(path_2025, scoreset)
    size_kb = sum(p.stat().st_size for p in bundle_dir.glob("scoreset_*.csv.gz")) / 1e3
    print(f"  Wrote {path_2018.name}, {path_2025.name} (+ their _snv siblings), {size_kb:.0f} KB total")


def _export_dataset_configs(dataset_configs_path: str, bundle_dir: Path) -> None:
    with open(dataset_configs_path) as f:
        full = json.load(f)
    if MSH2_PIPELINE_KEY not in full:
        raise KeyError(f"'{MSH2_PIPELINE_KEY}' not found in {dataset_configs_path}")
    out_path = bundle_dir / "dataset_configs.json"
    with open(out_path, "w") as f:
        json.dump({MSH2_PIPELINE_KEY: full[MSH2_PIPELINE_KEY]}, f, indent=2)
    print(f"  Wrote {out_path}")


def _export_bootstrap_fits(precomputed_fits: str, bundle_dir: Path) -> None:
    with gzip.open(precomputed_fits, "rt", encoding="utf-8") as f:
        full = json.load(f)
    if MSH2_PIPELINE_KEY not in full:
        raise KeyError(f"'{MSH2_PIPELINE_KEY}' not found in {precomputed_fits}")
    out_path = bundle_dir / "bootstrap_fits.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump({MSH2_PIPELINE_KEY: full[MSH2_PIPELINE_KEY]}, f)
    print(f"  Wrote {out_path} (1 dataset, vs. {len(full)} in the full fits file)")


def _export_revel_files(revel_dir: str, bundle_dir: Path) -> None:
    dest = bundle_dir / "revel"
    dest.mkdir(parents=True, exist_ok=True)
    names = [
        "REVEL_gene_specific_calibration_thresholds_20260118.csv",
        "MSH2_REVEL_scores.tsv",
        "MSH2_REVEL_labeled.txt",
        "REVEL_heatmap_data_pillar.csv",
    ]
    total_in, total_out = 0, 0
    for name in names:
        src = Path(revel_dir) / name
        if not src.exists():
            raise FileNotFoundError(f"{src} not found")
        # gzip-compressed on write -- driver._revel_path transparently prefers
        # a "<name>.gz" over the plain file, and pandas infers the compression
        # from that extension, so no change is needed on the reading side
        # beyond that path resolution.
        out_path = dest / f"{name}.gz"
        with open(src, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        total_in += src.stat().st_size
        total_out += out_path.stat().st_size
    print(f"  Compressed {len(names)} REVEL file(s) to {dest} "
          f"({total_in / 1e6:.1f} MB -> {total_out / 1e6:.1f} MB)")


def _export_panel_c(output_dir: str, dataset_tsv: str, dataset_configs_path: str, bundle_dir: Path) -> None:
    """Rebuild panel c's cross-dataset confusion matrices + VUS coverage
    fresh from the full pipeline output (the one part of this export that
    genuinely needs it -- see analysis.figure4.panel_c_io's docstring), then
    cache the result to a small JSON so the recipient never has to redo this.
    """
    tree, model_selections, calibrations = discover_outputs(Path(output_dir))
    if not tree:
        raise RuntimeError(f"No pipeline output discovered under {output_dir}")

    with open(dataset_configs_path) as f:
        dataset_configs = json.load(f)

    df = load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=None, calibrations=calibrations, min_controls=0,
    )
    if df.empty:
        raise RuntimeError("load_all_variants returned no rows")
    df = attach_author_labels(df, dataset_tsv)

    primary_method = sorted(df["method"].unique())[0]
    df_m = df[df["method"] == primary_method]
    # F9/TP53 excluded -- those two genes use separate classifier models to
    # integrate multiple datasets (same exclusion analyze_pipeline_output.py's
    # section 3a1/7 apply to the danzs_oob/auths_oob it passes build_figure4
    # directly), so this bundled panel_c.json stays consistent with a
    # notebook-driven reproduction rather than silently including them here.
    F9_TP53_GENES = {"F9", "TP53"}
    dataset_names = sorted(
        d for d in df_m["dataset"].unique() if d.split("_")[0] not in F9_TP53_GENES
    )

    danzs, auths, vus, auth_vus = [], [], [], []
    for dataset in dataset_names:
        df_ds = df_m[df_m["dataset"] == dataset]
        danzs.append(build_confusion_matrix(df_ds, use_oob=True, label=f"{dataset}/{primary_method}"))
        auths.append(build_author_confusion_matrix(df_ds, use_oob=True))
        vus.append(build_vus_coverage(df_ds, use_oob=True, label=f"{dataset}/{primary_method}"))
        auth_vus.append(build_author_vus_coverage(df_ds))

    # Same pairing analysis.figure4.panels.plot_panel_c itself applies: only
    # pool the ExCALIBR side over datasets that also have a real author call,
    # so the two aggregate matrices describe the same population.
    paired = [(d, a) for d, a in zip(danzs, auths) if d is not None and a is not None]
    danz_agg = sum(d for d, _ in paired)
    auth_agg = sum(a for _, a in paired)

    vus_pct_danz = _aggregate_coverage_pct(vus)
    vus_pct_auth = _aggregate_coverage_pct(auth_vus)

    out_path = bundle_dir / "panel_c.json"
    save_panel_c_bundle(str(out_path), danz_agg, auth_agg, len(paired), vus_pct_danz, vus_pct_auth)
    print(f"  Wrote {out_path} (aggregate over {len(paired)}/{len(dataset_names)} datasets)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", required=True, help="Output directory for the bundle.")
    parser.add_argument("--output-dir", default=None, help="Full pipeline output tree. Defaults to analysis.config.OUTPUT_DIR.")
    parser.add_argument("--dataset-tsv", default=None, help="Full master TSV. Defaults to analysis.config.DATASET_TSV.")
    parser.add_argument("--dataset-configs", default=None, help="Full dataset-configs JSON. Defaults to analysis.config.DATASET_CONFIGS.")
    parser.add_argument("--precomputed-fits", default=None, help="Full bootstrap-fits JSON. Defaults to analysis.config.PRECOMPUTED_FITS.")
    parser.add_argument("--revel-dir", default=None, help="Full REVEL data directory. Defaults to analysis.config.YILE_DIR.")
    parser.add_argument("--include-dataset-tsv", action="store_true",
                         help="Also write dataset.tsv.gz (MSH2 rows of the master TSV). Off by "
                              "default: analysis.figure4.driver only falls back to it if the "
                              "scoreset_2018.csv.gz/scoreset_2025.csv.gz files this script always "
                              "writes are missing, so most bundles don't need it.")
    args = parser.parse_args()

    output_dir = args.output_dir or cfg.OUTPUT_DIR
    dataset_tsv = args.dataset_tsv or cfg.DATASET_TSV
    dataset_configs_path = args.dataset_configs or cfg.DATASET_CONFIGS
    precomputed_fits = args.precomputed_fits or cfg.PRECOMPUTED_FITS
    revel_dir = args.revel_dir or cfg.YILE_DIR

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting MSH2-only Figure 4 bundle to {bundle_dir} ...")
    _export_calibration_files(output_dir, dataset_configs_path, bundle_dir)
    _export_scoresets(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, bundle_dir)
    if args.include_dataset_tsv:
        _export_dataset_tsv(dataset_tsv, bundle_dir)
    _export_dataset_configs(dataset_configs_path, bundle_dir)
    _export_bootstrap_fits(precomputed_fits, bundle_dir)
    _export_revel_files(revel_dir, bundle_dir)
    _export_panel_c(output_dir, dataset_tsv, dataset_configs_path, bundle_dir)
    print("Done.")


if __name__ == "__main__":
    main()
