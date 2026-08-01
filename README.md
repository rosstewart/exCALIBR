<p align="center">
  <img src="assets/excalibr_logo.png" alt="ExCALIBR logo" width="200">
</p>

# ExCALIBR

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.04.29.651326-b31b1b)](https://doi.org/10.1101/2025.04.29.651326)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pipeline for calibrating functional assays to clinical variant interpretation scales (ACMG/AMP evidence levels) using bootstrap skew-normal mixture model fitting and Bayesian calibration.

## Installation

```bash
git clone https://github.com/rosstewart/exCALIBR
cd exCALIBR

# CPU-only environment (standard runs)
conda env create -f excalibr.yml
conda activate excalibr
pip install -e .

# GPU environment (JAX + CUDA 12, for GPU-accelerated fitting)
conda env create -f excalibr-gpu.yml
conda activate excalibr
pip install -e .
```

## Quick Start

### Interactive (single dataset)

The simplest way to run ExCALIBR on your own data is the **BasicScoreset**
format: a CSV with just a `score` column (your assay's per-variant score)
and a `sample_assignments` column you fill in yourself, marking which rows
are known-Pathogenic, known-Benign, gnomAD/population, or Synonymous
controls (see [Input Data Formats](#input-data-formats) below for exactly
how to fill that column in). No ClinVar lookups or extra metadata needed.

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
```

This fits a bootstrap mixture model to the score distribution, calibrates
it against the labeled control groups, and writes a calibration JSON,
score-distribution plot, and per-variant evidence table to
`./calibration_output/` (see [Output Files](docs/output-files.md)).

By default this runs 20 bootstrap iterations × 8 fits each — fast, good
for a first look, and takes a few minutes on a laptop. A few flags you'll
likely want to reach for right away:

| Flag | What it does |
|------|--------------|
| `--n-bootstraps` / `--fits-per-bootstrap` | Trade speed for stability/quality — see the [quality vs. speed presets](docs/configuration.md#quality-vs-speed-presets) for concrete numbers (defaults are the fastest "Light" preset). |
| `--n-jobs` | How many CPU cores to use (`-1` = all available, the default). |
| `--output-dir` | Where results are written (default: `./calibration_output`). |
| `--components` | Which mixture-model sizes to fit (default: `3`; pass `2 3` to fit both and auto-select). |
| `--sample-names` | Required for BasicScoreset CSVs — labels your `sample_assignments` columns in order (see above). |
| `--manual-prior` | Override the estimated prior if you have your own domain-knowledge estimate — see [Prior estimation](docs/configuration.md#prior-estimation). |

If your data already comes with ClinVar/gnomAD annotations (e.g. an IGVF-
or PillarProject-formatted table), you can skip manually assigning sample
groups — see [Input Data Formats](#input-data-formats) below.

### Production batch (many datasets)

See the [Batch HPC Workflow](docs/batch-hpc-workflow.md).

## Input Data Formats

Three formats are supported — see [docs/input-formats.md](docs/input-formats.md) for full details and example commands:

1. **BasicScoreset** (start here) — just a `score` column and a `sample_assignments` column you fill in yourself, marking which variants are Pathogenic/Benign/gnomAD/Synonymous controls. No ClinVar metadata needed. Example: `example/brca_findlay_example.csv`.
2. **IGVF / PillarProject format** — a rich per-variant metadata table; sample groups are instead derived automatically from ClinVar/gnomAD/consequence columns already present in the file, so you don't assign them by hand. Example: `example/MSH2_Jia_2021.csv`.
3. **MaveDB format** — MaveDB-style CSV with functional classification columns.

Multi-assay (multivariate) calibration and calibration of computational variant-effect predictor scores (REVEL, AlphaMissense, etc.) are also supported — see the note at the bottom of [docs/input-formats.md](docs/input-formats.md#3-mavedb-format).

## Documentation

- [Input Data Formats](docs/input-formats.md) — the three supported input formats, in detail
- [Batch HPC Workflow](docs/batch-hpc-workflow.md) — calibrating many datasets across a SLURM cluster
- [Script Reference](docs/script-reference.md) — every script and its key flags
- [Output Files](docs/output-files.md) — what gets written, and what's in `calibration.json`
- [Configuration Options](docs/configuration.md) — bootstrap presets, priors, component selection, postprocessing, bidirectional-assay detection
- [Troubleshooting](docs/troubleshooting.md)

## Citation

If you use ExCALIBR, please cite:

```
Gene-based calibration of high-throughput functional assays for clinical variant classification.
Daniel Zeiberg, Ross Stewart, et al.
bioRxiv 2025.04.29.651326; doi: https://doi.org/10.1101/2025.04.29.651326
```

For the IGVF dataset and format:

```
A scalable approach to resolving variants of uncertain significance.
Malvika Tejura, Yile Chen, Abbye E. McEwen, Ross Stewart, Yuriy Sverchkov, Florent Laval, et al.
bioRxiv 2026.02.14.705848; doi: https://doi.org/10.64898/2026.02.14.705848
```

## License

[MIT License](LICENSE)

## Contact

[stewart.ro@northeastern.edu](mailto:stewart.ro@northeastern.edu)
