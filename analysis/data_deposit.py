"""
Files for the Zenodo data record that accompanies the paper (separate from
the software record of release v1.1.0): the bootstrap fits for the calibrated
datasets.

  bootstrap_fits.json.gz              regular ExCALIBR fits (config.PRECOMPUTED_FITS)
  bootstrap_fits_skew_locked.json.gz  skew-locked refits (config.SKEW_LOCKED_BOOTSTRAP_RESULTS)
  sample_order.json                   per dataset, the sample of each row of a fit's `weights`
  README.md                           contents, structure, checksums

The bootstrap training/held-out splits are not deposited: the released code
regenerates them exactly with the default master seed (0), under which each
bootstrap's composition is keyed by its seed. `verify_regenerated_splits`
checks this against the production splits (config.SPLITS_FILE) for every
dataset, so the README's statement is tested on each build.

    python -m analysis.data_deposit [--out DIR]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from analysis import config as cfg

FILES = ["bootstrap_fits.json.gz", "bootstrap_fits_skew_locked.json.gz", "sample_order.json"]
_SPLIT_KEYS = ("train_observations", "train_sample_assignments", "val_observations",
               "val_sample_assignments", "val_variant_indices")


def _scoreset_and_nc(dataset: str):
    from analysis.legacy_fits import load_scoreset_and_fits, resolve_component_for

    n_c, benign_method = resolve_component_for(dataset)
    scoreset, _, _, _, n_c, _, _ = load_scoreset_and_fits(dataset, n_c=n_c, benign_method=benign_method)
    return scoreset, n_c


def verify_regenerated_splits(datasets: Iterable[str], seeds: Sequence[int] = (0, 1, 499, 999)) -> (Dict, List[str]):
    """Regenerate bootstrap splits with the released code (master seed 0) and
    compare them exactly with the production splits the fits were trained on.
    Returns ({dataset: sample names in fit-weight row order}, problems)."""
    from src.assay_calibration.fit_utils.fit import Fit, DEFAULT_MASTER_SEED

    with open(cfg.SPLITS_FILE, "rb") as f:
        production = pickle.load(f)
    order, problems = {}, []
    for d in datasets:
        scoreset, n_c = _scoreset_and_nc(d)
        order[d] = [str(s[1]) for s in scoreset.samples]
        if d not in production:
            problems.append(f"{d}: not in {cfg.SPLITS_FILE}")
            continue
        fitter = Fit(scoreset)
        for s in seeds:
            if s not in production[d]:
                continue
            job = fitter.generate_fit_jobs(component_range=[int(str(n_c).rstrip("c")[0])], bootstrap_seed=s,
                                           master_seed=DEFAULT_MASTER_SEED, check_monotonic=True, num_fits=1)[0]
            # Composition (which variants, which sample) must match exactly;
            # scores can differ in the last bit (floating-point order of the
            # score preprocessing; at most 4e-16 seen).
            same = all(np.array_equal(job[k], production[d][s][k]) for k in _SPLIT_KEYS
                       if not k.endswith("_observations"))
            same = same and all(np.allclose(job[k], production[d][s][k], rtol=0, atol=1e-12, equal_nan=True)
                                for k in _SPLIT_KEYS if k.endswith("_observations"))
            if not same:
                problems.append(f"{d} seed {s}: regenerated split differs from {cfg.SPLITS_FILE}")
                break
    return order, problems


def _subset_fits(src: str, datasets: Iterable[str], out_path: Path) -> Dict[str, int]:
    """Copy of a bootstrap-fits JSON restricted to `datasets` (the paper's
    calibrated set); returns {dataset: number of bootstrap iterations}."""
    with gzip.open(src, "rt") as f:
        fits = json.load(f)
    keep = {d: fits[d] for d in datasets if d in fits}
    with gzip.open(out_path, "wt") as f:
        json.dump(keep, f)
    return {d: len(v) for d, v in keep.items()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


README = """# ExCALIBR bootstrap fits

Data record accompanying Stewart, Zeiberg, Jain *et al.*, "Gene-based calibration of
high-throughput functional assays for clinical variant classification". The
software is archived separately (ExCALIBR v1.1.0, https://doi.org/10.5281/zenodo.23197027).
The input variant-effect dataset is available from the IGVF data portal
(accession IGVFDS4011IBUI).

{n_datasets} calibrated datasets, {n_boot} bootstrap iterations each (median).

## Files

| File | Size | SHA-256 |
|---|---|---|
{file_rows}

## bootstrap_fits.json.gz, bootstrap_fits_skew_locked.json.gz

Gzipped JSON: `{{dataset: {{bootstrap_seed: {{n_components: record}}}}}}`, with
`n_components` "2c" or "3c" (the skew-locked refits contain only each
dataset's selected component count). Each record holds the fit of that
bootstrap iteration:

- `fit.component_params`: one `[skewness, location, scale]` per mixture component
  (skew normal, scipy.stats.skewnorm parameterization)
- `fit.weights`: mixing weights, one row per sample (row order in
  `sample_order.json`), one column per component
- `val_ll`: mean log-likelihood of the iteration's held-out variants
- other keys: fitting diagnostics

In the skew-locked refits every component's skewness is fixed at 0.

## sample_order.json

`{{dataset: [sample, ...]}}`: the sample of each row of `fit.weights`.

## Bootstrap samples

The training and held-out variants of each bootstrap iteration are not
included: ExCALIBR v1.1.0 regenerates them from the input dataset with the
default master seed (0), under which each iteration's composition is keyed by
its `bootstrap_seed`. Checked against the splits used for the paper for every
dataset: the same variants in the same samples, with scores equal up to
floating-point rounding (differences below 1e-12).

Generated by `analysis/data_deposit.py` (git {git_sha}).
"""


def build_deposit(datasets: List[str], out_dir: Path = None) -> Dict:
    """Write the deposit folder; returns {"files": {...}, "problems": [...]}."""
    out_dir = Path(out_dir or cfg.ZENODO_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "bootstrap_splits.npz"  # no longer deposited (splits regenerate exactly)
    if stale.exists():
        stale.unlink()
    order, problems = verify_regenerated_splits(datasets)
    (out_dir / "sample_order.json").write_text(json.dumps(order, indent=1))
    n_reg = _subset_fits(cfg.PRECOMPUTED_FITS, datasets, out_dir / "bootstrap_fits.json.gz")
    n_skew = _subset_fits(cfg.SKEW_LOCKED_BOOTSTRAP_RESULTS, datasets, out_dir / "bootstrap_fits_skew_locked.json.gz")
    if len(n_reg) != len(datasets):
        problems.append(f"regular fits: {len(n_reg)} of {len(datasets)} datasets")
    if len(n_skew) != len(datasets):
        problems.append(f"skew-locked fits: {len(n_skew)} of {len(datasets)} datasets")
    rows = "\n".join(f"| `{f}` | {(out_dir / f).stat().st_size / 1e6:,.1f} MB | `{_sha256(out_dir / f)}` |"
                     for f in FILES)
    (out_dir / "README.md").write_text(README.format(
        n_datasets=len(datasets), n_boot=int(np.median(list(n_reg.values()))) if n_reg else 0,
        file_rows=rows, git_sha=_git_sha()))
    return {"files": {f: (out_dir / f).stat().st_size for f in FILES + ["README.md"]}, "problems": problems}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: config.ZENODO_DATA_DIR")
    args = ap.parse_args()
    from analysis.discovery import discover_outputs, DEFAULT_EXCLUDED_DATASETS
    tree, _, _ = discover_outputs(Path(cfg.OUTPUT_DIR))
    res = build_deposit(sorted(d for d in tree if d not in DEFAULT_EXCLUDED_DATASETS), args.out)
    print(json.dumps(res, indent=1))
