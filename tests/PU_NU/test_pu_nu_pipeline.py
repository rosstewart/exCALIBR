"""End-to-end smoke tests for the PU (positive-unlabeled) and NU
(negative-unlabeled) pipeline modes -- see docs/input-formats.md#pnpunu-modes-
missing-class-inference.

These run the actual `run_pipeline.py` CLI (not just unit-level fitting code)
against `example/brca_findlay_PU_example.csv` and `example/brca_findlay_NU_example.csv`
-- gutted versions of `example/brca_findlay_example.csv` with the Benign/
Synonymous labels (PU) or Pathogenic label (NU) stripped out, leaving only
gnomAD/population plus one labeled class, exactly the minimum-viable sample
count PU/NU mode requires.

Regression test for the `n_samples < 3` hard-block in
BootstrapRunner._load_dataset (src/assay_calibration/pipeline/fit_bootstrap.py)
that made every PU/NU dataset (2 populated sample categories) impossible to
run through run_pipeline.py -- fixed to `< 2`, matching the threshold already
used elsewhere in the codebase (hpc/prepare.py, multivariate_data/combined.py,
multivariate_data/common.py).

Run with:
    source activate excalibr
    pytest tests/PU_NU/test_pu_nu_pipeline.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "example"


def _run_pipeline(dataset_csv: str, name: str, output_dir: Path):
    cmd = [
        sys.executable, str(REPO_ROOT / "run_pipeline.py"),
        "--dataset", str(EXAMPLE_DIR / dataset_csv),
        "--name", name,
        "--sample-names", "Pathogenic/Likely Pathogenic", "Benign/Likely Benign",
        "gnomAD", "Synonymous",
        "--preset", "light",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"run_pipeline.py failed for {dataset_csv}:\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    return result


@pytest.mark.parametrize("dataset_csv,name", [
    ("brca_findlay_PU_example.csv", "brca_findlay_PU_smoke"),
    ("brca_findlay_NU_example.csv", "brca_findlay_NU_smoke"),
])
def test_pu_nu_pipeline_runs_end_to_end(tmp_path, dataset_csv, name):
    output_dir = tmp_path / name
    _run_pipeline(dataset_csv, name, output_dir)

    calibration_path = output_dir / f"{name}_3c_calibration.json"
    variants_path = output_dir / f"{name}_3c_variants.csv"
    assert calibration_path.exists(), f"missing {calibration_path}"
    assert variants_path.exists(), f"missing {variants_path}"

    with open(calibration_path) as f:
        calibration = json.load(f)
    assert "prior" in calibration
    assert calibration["prior"] is not None


def test_pu_dataset_has_no_benign_or_synonymous_labels():
    import pandas as pd
    df = pd.read_csv(EXAMPLE_DIR / "brca_findlay_PU_example.csv")
    tokens = {
        t for v in df["sample_assignments"].dropna()
        for t in str(v).split(",") if t
    }
    assert tokens <= {"0", "2"}, f"PU example should only carry labels 0/2, found {tokens}"


def test_nu_dataset_has_no_pathogenic_labels():
    import pandas as pd
    df = pd.read_csv(EXAMPLE_DIR / "brca_findlay_NU_example.csv")
    tokens = {
        t for v in df["sample_assignments"].dropna()
        for t in str(v).split(",") if t
    }
    assert tokens <= {"1", "2", "3"}, f"NU example should only carry labels 1/2/3, found {tokens}"
