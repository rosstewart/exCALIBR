<p align="center">
  <img src="assets/excalibr_logo.png" alt="ExCALIBR logo" width="56">
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

Start with **BasicScoreset**: a `score` column + a `sample_assignments` column marking Pathogenic/Benign/gnomAD/Synonymous rows yourself (see [Input Data Formats](#input-data-formats)).

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
```

Writes a calibration JSON, score-distribution plot, and per-variant evidence table to `./calibration_output/` (see [Output Files](docs/output-files.md)). Default: `--preset light` — fast; use `--preset medium`/`large`/`xl`/`finest` for higher quality (see [presets](docs/configuration.md#quality-vs-speed-presets)).

### Common options

| Flag | Default | What it does |
|---|---|---|
| `--preset` | `light` | Quality/speed level: `light`/`medium`/`large`/`xl`/`finest` — see [presets](docs/configuration.md#quality-vs-speed-presets) |
| `--n-jobs` | `-1` (all cores) | Parallel workers (`1` = single-threaded, e.g. for debugging) |
| `--device` | `cpu` | `cpu`, `gpu` (JAX-batched, default GPU), or `cuda:N` to pin to GPU N. `run_pipeline.py` runs one dataset at a time so only one GPU is ever used; for multi-dataset GPU runs use the HPC workflow (`hpc/prepare.py`), one process per GPU. |
| `--output-dir` | `./calibration_output` | Output location |
| `--components` | `3` | Mixture-model size(s) to fit; `2 3` fits both and auto-selects |
| `--sample-names` | — | Required for BasicScoreset; labels `sample_assignments` columns in order |
| `--manual-prior` | — | Supply your own prior instead of estimating — see [Prior estimation](docs/configuration.md#prior-estimation) |
| `--benign-method` | `avg` | `avg`/`benign`/`synonymous` — see [Benign sample method](docs/configuration.md#benign-sample-method) |
| `--conservative-monotonicity` | off | Stricter evidence-threshold enforcement |
| `--no-auto-bidirectional` | off (auto-detect on) | For LoF/GoF-style assays — see [Bidirectional detection](docs/configuration.md#bidirectional-assay-auto-detection) |
| `--no-postprocess` | off | Raw, unprocessed LR-threshold intervals (debugging only) — see [Point-range postprocessing](docs/configuration.md#point-range-postprocessing) |
| `--pathomechanism-prior` | off | Mechanism-aware pathogenic-direction prior, for assays that only detect one disease mechanism — see [Pathomechanism prior](docs/configuration.md#pathomechanism-prior-advanced) |
| `--precomputed-fits` | — | Skip fitting, load existing bootstrap fits |
| `--oob` | off | Compute out-of-bag per-variant evidence |

Full flag reference: [docs/script-reference.md](docs/script-reference.md).

### Production batch (many datasets)

See the [Batch HPC Workflow](docs/batch-hpc-workflow.md).

## Input Data Formats

Three formats are supported — see [docs/input-formats.md](docs/input-formats.md) for full details and example commands:

1. **BasicScoreset** (start here) — just a `score` column and a `sample_assignments` column you fill in yourself, marking which variants are Pathogenic/Benign/gnomAD/Synonymous controls. No ClinVar metadata needed. Example: `example/brca_findlay_example.csv`. PU/NU-mode examples: `example/brca_findlay_PU_example.csv`, `example/brca_findlay_NU_example.csv`.
2. **IGVF / PillarProject format** — a rich per-variant metadata table; sample groups are instead derived automatically from ClinVar/gnomAD/consequence columns already present in the file, so you don't assign them by hand. Example: `example/MSH2_Jia_2021.csv`.
3. **MaveDB format** — MaveDB-style CSV with functional classification columns.

Multi-assay (multivariate) calibration and calibration of computational variant-effect predictor scores (REVEL, AlphaMissense, etc.) are also supported — see the note at the bottom of [docs/input-formats.md](docs/input-formats.md#3-mavedb-format).

## Documentation

- [Input Data Formats](docs/input-formats.md) — the three supported input formats, in detail
- [Batch HPC Workflow](docs/batch-hpc-workflow.md) — calibrating many datasets across a SLURM cluster
- [Script Reference](docs/script-reference.md) — every script and its key flags
- [Output Files](docs/output-files.md) — what gets written, and what's in `calibration.json`
- [Configuration Options](docs/configuration.md) — bootstrap presets, priors, component selection, postprocessing, bidirectional-assay detection
- [GPU Acceleration](docs/gpu-acceleration.md) — JAX/CUDA setup, JIT compilation details, and benchmark timings
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

## Software

Software developed by Ross Stewart and Daniel Zeiberg.

## License

[MIT License](LICENSE)

## Contact

[stewart.ro@northeastern.edu](mailto:stewart.ro@northeastern.edu)
