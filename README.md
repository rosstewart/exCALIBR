# ExCALIBR

A pipeline for calibrating functional assays to clinical variant interpretation scales (ACMG/AMP evidence levels) using bootstrap skew-normal mixture model fitting and Bayesian calibration.

## Overview

ExCALIBR takes variant effect scores from functional assays and calibrates them using:

- **Bootstrap mixture modeling** to estimate probability distributions for pathogenic, benign, population, and synonymous variants
- **Bayesian calibration** to compute likelihood ratios and ACMG evidence thresholds
- **Statistical model selection** to choose optimal component counts (2c vs 3c)
- **Flexible execution** via SLURM clusters, parallel processing, or single-CPU

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

```bash
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021
```

Defaults: 20 bootstrap iterations × 8 fits each — fast, good for a first look. For higher-quality (slower) results, see the [quality vs. speed presets](docs/configuration.md#quality-vs-speed-presets), or jump straight to `--n-bootstraps 1000 --fits-per-bootstrap 100`.

### Production batch (many datasets)

See the [Batch HPC Workflow](docs/batch-hpc-workflow.md).

## Input Data Formats

Three formats are supported — see [docs/input-formats.md](docs/input-formats.md) for full details and example commands:

1. **IGVF / PillarProject format** (standard) — a rich per-variant metadata table; sample groups (Pathogenic/Benign/gnomAD/Synonymous) are derived automatically from ClinVar/gnomAD/consequence columns. Example: `example/MSH2_Jia_2021.csv`.
2. **BasicScoreset** (bare-bones CSV) — just a `score` column and a `sample_assignments` column you fill in yourself. No ClinVar metadata needed. Example: `example/brca_findlay_example.csv`.
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
