#!/usr/bin/env python
"""
Re-calibrate BRCA2_Huang_2025_SGE with ExCALIBR on Huang et al.'s own VarCall
model score instead of the master TSV's auth_reported_score.

Builds a one-dataset copy of the master TSV's BRCA2_Huang_2025_SGE rows with
auth_reported_score replaced by Table S3's "Model based functional score"
(joined on the c. change -- 6,959/6,959 unique matches), renamed to
BRCA2_Huang_2025_SGE_VarCallScore, then runs run_pipeline.py on it with the
canonical run's settings (3 components, benign_method=benign, ClinVar 2025,
OOB) at --preset light. Everything else about the variants -- ClinVar/gnomAD/
synonymous sample membership, Flag / splice filtering -- is untouched, so the
fitting samples are the same four as the canonical run; only the score differs.

Output goes to its own directory (not config.OUTPUT_DIR) so it never shows up
in analyze_pipeline_output.py's discover_outputs. compare_varcall_brca2_huang.py
picks it up from there.

Usage
-----
python analysis/run_varcall_score_calibration_brca2_huang.py
python analysis/run_varcall_score_calibration_brca2_huang.py --preset medium --n-jobs 16
python analysis/run_varcall_score_calibration_brca2_huang.py --build-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from analysis import config

SOURCE_DATASET = "BRCA2_Huang_2025_SGE"
VARCALL_SCORE_DATASET = "BRCA2_Huang_2025_SGE_VarCallScore"
DEFAULT_VARCALL = "/data/ross/assay_calibration/BRCA2_Huang/BRCA2_Huang_VarCall.csv"
DEFAULT_WORK_DIR = "/data/ross/assay_calibration/BRCA2_Huang/excalibr_varcall_score"
VARCALL_SCORE_COL = "Model based functional score"
VARCALL_HGVS_C_COL = "Coding sequence change (c.)"

# Canonical run's settings for BRCA2_Huang_2025_SGE: dataset_configs_aug_2026
# gives {"n_c": "3c", "benign_method": "benign"}, and ClinVar 2025 is the
# release that reproduces its 390 P/LP / 674 B/LB / 2,721-variant scoreset
# (2026 gives 425 / 1,395 / 2,892).
PIPELINE_ARGS = [
    "--components", "3",
    "--benign-method", "benign",
    "--clinvar-release", "2025",
    "--oob",
]


def build_varcall_score_table(master_tsv: str, varcall_path: str) -> pd.DataFrame:
    sep = "\t" if master_tsv.endswith((".tsv", ".tsv.gz")) else ","
    df = pd.read_csv(master_tsv, sep=sep, low_memory=False)
    df = df[df["Dataset"] == SOURCE_DATASET].copy()
    if df.empty:
        raise ValueError(f"{SOURCE_DATASET} not found in {master_tsv}")

    vc = pd.read_csv(varcall_path, skiprows=1)
    vc.columns = [c.strip() for c in vc.columns]
    vc = vc[vc[VARCALL_HGVS_C_COL].notna() & vc[VARCALL_SCORE_COL].notna()]
    score_by_c = dict(zip(vc[VARCALL_HGVS_C_COL].astype(str).str.strip(),
                          pd.to_numeric(vc[VARCALL_SCORE_COL], errors="raise")))

    c_short = df["hgvs_c"].astype(str).str.rsplit(":", n=1).str[-1]
    new_score = c_short.map(score_by_c)
    n_missing = int(new_score.isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing}/{len(df)} {SOURCE_DATASET} rows have no VarCall score, "
                         f"e.g. {c_short[new_score.isna()].head(5).tolist()}")

    df["auth_reported_score_original"] = df["auth_reported_score"]
    df["auth_reported_score"] = new_score.values
    df["Dataset"] = VARCALL_SCORE_DATASET
    print(f"Built {VARCALL_SCORE_DATASET}: {len(df):,} rows, every auth_reported_score replaced by "
          f"VarCall '{VARCALL_SCORE_COL}' (corr with original: "
          f"{df['auth_reported_score'].corr(pd.to_numeric(df['auth_reported_score_original'], errors='coerce')):.3f})")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-tsv", default=config.DATASET_TSV)
    parser.add_argument("--varcall", default=DEFAULT_VARCALL, help="Huang et al. Table S3 CSV")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                        help="Holds the temporary table and the pipeline output (<work-dir>/<dataset>/)")
    parser.add_argument("--preset", default="light", help="run_pipeline.py --preset (default: light)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--build-only", action="store_true", help="Write the table, don't run the pipeline")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    table_path = work_dir / f"{VARCALL_SCORE_DATASET}.tsv.gz"

    df = build_varcall_score_table(str(args.master_tsv), args.varcall)
    df.to_csv(table_path, sep="\t", index=False, compression="gzip")
    print(f"  Saved: {table_path}")
    if args.build_only:
        return

    cmd = [
        sys.executable, str(_ROOT / "run_pipeline.py"),
        "--dataset", str(table_path),
        "--name", VARCALL_SCORE_DATASET,
        "--output-dir", str(work_dir / VARCALL_SCORE_DATASET),
        "--preset", args.preset,
        "--n-jobs", str(args.n_jobs),
        *PIPELINE_ARGS,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)


if __name__ == "__main__":
    main()
