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

Defaults: 20 bootstrap iterations × 8 fits each. For production-quality results, use `--n-bootstraps 1000 --fits-per-bootstrap 100` (or the SLURM batch workflow below).

### Production batch (many datasets)

See the [Batch HPC Workflow](#batch-hpc-workflow) section below.

## Input Data Formats

Three formats are supported:

### 1. IGVF / PillarProject format (standard)

Multi-sample variant score TSV with `score`, `sample`, and `Dataset` columns. ClinVar-labeled samples (P/LP, B/LB, gnomAD, synonymous) are auto-detected. This is the format produced by the IGVF Coding Variants Focus Group pipeline (Tejura et al. 2026, bioRxiv 2026.02.14.705848).

```bash
python run_pipeline.py --dataset variants.tsv.gz --name MyGene_MyLab_2025
```

Required columns: `score`, `sample` (integer index or comma-separated for multilabel), `Dataset`.

Sample indices:
- `0`: Pathogenic/Likely Pathogenic (ClinVar P/LP)
- `1`: Benign/Likely Benign (ClinVar B/LB, optional)
- `2`: gnomAD / population
- `3`: Synonymous (optional)

At least one of Benign or Synonymous samples is required.

### 2. BasicScoreset (bare-bones CSV)

Minimal format: `score` column + `sample_assignments` integer column. No ClinVar required. Use `--sample-names` to label samples.

```bash
python run_pipeline.py \
    --dataset basic.csv --name MyGene \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD"
```

### 3. MaveDB format

MaveDB-style CSV with functional classification columns. Used for batch runs via `slurm/prepare.py mavedb`.

Implementation: `src/assay_calibration/data_utils/dataset.py` — `BasicScoreset`, `Scoreset`, `MultiScoreset`, `BasicMultiScoreset`.

## Batch HPC Workflow

For calibrating many datasets, use the decoupled batch workflow: fitting runs on a compute cluster, then calibration runs from precomputed fits.

### Step 1 — Generate job manifest

```bash
python slurm/prepare.py default \
    --output-dir /path/to/run \
    --dataframe variants.tsv.gz

# Other subcommands:
#   pillar_project  — same as default, ClinVar 2025 default instead of 2026
#   mavedb          — MaveDB-formatted CSV input
#   basicscoreset   — bare-bones CSV with sample_assignments column
#   multivariate    — multi-assay per-gene fitting (MultiScoreset)
#   predictor-mv    — multi-assay from per-gene predictor CSV files
```

Outputs `<output_dir>/jobs/job_index.json` and `jobs/array_NNNN.pkl` files.

### Step 2 — Submit fits

**SLURM CPU array:**
```bash
bash slurm/submit_array.sh /path/to/run
```

Key env-var overrides: `SLURM_ACCOUNT`, `SLURM_PARTITION` (default: `short`), `SLURM_TIME` (default: `23:59:00`), `SLURM_MEM` (default: `1G`), `SLURM_CPUS` (default: `8`), `PYTHON` (default: system python).

**SLURM GPU array:**
```bash
bash slurm/submit_array_gpu.sh /path/to/run
```

Additional overrides: `N_GPUS` (default: `32`), `CUDA_MODULE`, `CONDA_MODULE`.

**Local CPU:**
```bash
bash slurm/run_local_array.sh /path/to/run START END [concurrency]
# Example: bash slurm/run_local_array.sh /path/to/run 0 999 8
```

**Local GPU:**
```bash
PYTHON=/path/to/envs/excalibr/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash slurm/run_local_array_gpu.sh /path/to/run START END
```

### Step 3 — Monitor progress

```bash
python slurm/count_bootstraps.py /path/to/run [--verbose]
```

### Step 4 — Aggregate fit results

```bash
python slurm/aggregate_results.py /path/to/run
# Writes /path/to/run/bootstrap_results.json.gz
```

### Step 5 — Calibrate all datasets

```bash
python run_igvf_batch.py \
    --dataset variants.tsv.gz \
    --precomputed-fits /path/to/run/bootstrap_results.json.gz \
    --dataset-configs src/igvf_configs/my_configs.json \
    --output-dir ./calibration_output
```

### Step 6 — Collect outputs

```bash
python src/collect_calibration_outputs.py ./calibration_output ./collected \
    --dataset-configs src/igvf_configs/my_configs.json
# Writes collected/json/ and collected/png/, then tars for download
```

## Script Reference

### `run_pipeline.py` — Single-dataset interactive pipeline

Fits, selects models, generates visualizations, and saves calibration JSON for one dataset.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Path to input CSV/TSV |
| `--name` | (required) | Dataset name for output files |
| `--n-bootstraps` | 20 | Bootstrap iterations (use 1000 for production) |
| `--fits-per-bootstrap` | 8 | Fits per bootstrap (use 100 for production) |
| `--components K [K ...]` | `2 3` | Component counts to fit (integers 2–10) |
| `--mode` | `parallel` | `parallel`, `single`, or `slurm` |
| `--device` | `cpu` | `cpu` or `gpu` (GPU: routes through JAX batch) |
| `--precomputed-fits` | — | Skip fitting; load existing bootstrap fits JSON |
| `--output-dir` | `./calibration_output` | Output directory |
| `--no-postprocess` | off | Skip monotonicity enforcement and extend-to-limits on point ranges. Intended for bidirectional assays (e.g. LoF/GoF in one assay) where standard monotonicity assumptions do not hold. |
| `--conservative-monotonicity` | off | Use stricter monotonicity enforcement (default is liberal) |
| `--manual-prior` | — | Override prior probability (0–1); skip estimation |
| `--benign-method` | `avg` | `avg`, `benign`, or `synonymous` |
| `--oob` | off | Compute out-of-bag per-variant evidence |
| `--seed` | — | Master seed for full reproducibility |
| `--sample-names` | — | Override sample labels (order must match data columns) |

---

### `run_igvf_batch.py` — Batch calibration from precomputed fits

Calibrates many datasets at once from a pre-aggregated bootstrap fits file. Does not run any fitting.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Integrated TSV with all datasets |
| `--precomputed-fits` | (required) | Path to `bootstrap_results.json.gz` from `aggregate_results.py` |
| `--dataset-configs` | — | JSON mapping dataset names to `[n_c, benign_method, {overrides}]` |
| `--datasets` | all | Only process these dataset names |
| `--output-dir` | `./igvf_output` | Output directory |
| `--n-jobs` | 1 | Datasets in parallel (outer) |
| `--n-jobs-inner` | -1 | Parallel jobs within each dataset |
| `--skip-existing` | off | Resume a failed run (skip completed datasets) |
| `--all-configs` | off | Run all 4 (2c/3c)×(avg/benign) combos per dataset with comparison plot |
| `--no-postprocess` | off | Skip point-range postprocessing (same as `run_pipeline.py`) |
| `--manual-prior` | — | Override prior for all datasets |
| `--oob` | off | Compute OOB per-variant evidence |
| `--generate-config-template` | — | Write a blank dataset config JSON and exit |

---

### `slurm/prepare.py` — Job manifest generation

Generates array job files from a dataset dataframe. Must be run before submitting fits.

```bash
python slurm/prepare.py <subcommand> --output-dir /path/to/run [options]
```

Subcommands:
- `default` — IGVF/PillarProject TSV, ClinVar 2026
- `pillar_project` — same as `default`, ClinVar 2025 default
- `mavedb` — MaveDB CSV with `--score-cols` and `--dataframe`
- `basicscoreset` — bare-bones CSV or directory of CSVs
- `multivariate` — multi-assay fitting; groups genes with >1 dataset
- `predictor-mv` — multi-assay from per-gene predictor CSVs

Common options: `--n-bootstraps` (default: 1000), `--components`, `--datasets`, `--target-array-size`, `--n-jobs`.

---

### `slurm/submit_array.sh` — SLURM CPU array submission

```bash
bash slurm/submit_array.sh /path/to/run
```

Env-var overrides (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_ACCOUNT` | `predrag` | SLURM account |
| `SLURM_PARTITION` | `short` | Partition |
| `SLURM_TIME` | `23:59:00` | Wall time |
| `SLURM_MEM` | `1G` | Memory per task |
| `SLURM_CPUS` | `8` | CPUs per task |
| `MAX_CONCURRENT` | `50` | Max simultaneous array tasks |
| `PYTHON` | system python | Python executable |

---

### `slurm/submit_array_gpu.sh` — SLURM GPU array submission

```bash
N_GPUS=32 bash slurm/submit_array_gpu.sh /path/to/run
```

Same overrides as `submit_array.sh` plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `N_GPUS` | `32` | Number of GPU jobs (each covers a consecutive range) |
| `SLURM_GRES` | `gpu:1` | GPU resource request |
| `CUDA_MODULE` | `cuda/12.1.1` | CUDA module to load |
| `CONDA_MODULE` | `anaconda3/2024.06` | Conda module to load |

---

### `slurm/run_local_array.sh` — Local CPU execution

```bash
bash slurm/run_local_array.sh /path/to/run START END [concurrency]
```

Runs array tasks `START` to `END` locally using `xargs -P` for parallelism.

---

### `slurm/run_local_array_gpu.sh` — Local GPU execution

```bash
PYTHON=/path/to/envs/excalibr/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash slurm/run_local_array_gpu.sh /path/to/run START END [gpu_id]
```

Runs all tasks in one process to keep the JAX JIT cache alive across tasks.

---

### `slurm/run_array_task.py` — Array task worker

Called by the submission scripts; not intended to be invoked directly.

```
python slurm/run_array_task.py <output_dir> <array_idx> [--end <end_idx>] [--device {cpu,gpu}]
```

---

### `slurm/count_bootstraps.py` — Progress monitoring

```bash
python slurm/count_bootstraps.py /path/to/run [--verbose]
```

Prints a per-dataset table of completed vs. expected bootstrap iterations.

---

### `slurm/aggregate_results.py` — Aggregate fit results

```bash
python slurm/aggregate_results.py /path/to/run [output_file]
```

Merges all per-bootstrap `*.pkl` files into `bootstrap_results.json.gz`, which is the input for `run_igvf_batch.py --precomputed-fits`.

---

### `src/collect_calibration_outputs.py` — Collect batch outputs

```bash
python src/collect_calibration_outputs.py <input_dir> <output_dir> \
    [--dataset-configs src/igvf_configs/my_configs.json]
```

Copies the selected-model calibration JSON and PNG for each dataset into flat `json/` and `png/` subdirectories, then tars the result for download.

## Output Files

For each run, `run_pipeline.py` and `run_igvf_batch.py` produce:

| File | Description |
|------|-------------|
| `<name>_<Kc>_calibration.json` | Calibration thresholds, prior, point ranges, fit metadata |
| `<name>_<Kc>_visualization.png` | Score distribution plot with calibrated thresholds |
| `<name>_<Kc>_variants.csv` | Per-variant point assignment table |
| `<name>_<Kc>_lr_values.json.gz` | Full LR+ curves over score range |
| `<name>_model_selection.json` | 2c vs. 3c bootstrap test results |
| `<name>_bootstrap_fits.json.gz` | Saved bootstrap fit results (when fitting fresh) |

The `calibration.json` includes:
```json
{
  "prior": 0.0034,
  "point_ranges": {
    "1": [[0.12, 0.45]],
    "2": [[0.45, 0.78]],
    "-1": [[-0.45, -0.12]],
    "-2": [[-0.78, -0.45]]
  },
  "scoreset_flipped": true,
  "n_valid_fits": 998,
  "C_range": [3.17, 3.45]
}
```

## Configuration Options

### Bootstrap parameters

```bash
# Interactive / exploratory (defaults)
python run_pipeline.py --dataset my.csv --name MyGene
# → 20 bootstraps × 8 fits

# Production quality
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 100
```

### Component selection

```bash
# Default: fit both 2c and 3c, auto-select
python run_pipeline.py --dataset my.csv --name MyGene --components 2 3

# Force 3-component only
python run_pipeline.py --dataset my.csv --name MyGene --components 3

# Fit 5-component model
python run_pipeline.py --dataset my.csv --name MyGene --components 5
```

### Prior estimation

```bash
# Empirical EM estimation (default)
python run_pipeline.py --dataset my.csv --name MyGene

# Manual prior
python run_pipeline.py --dataset my.csv --name MyGene --manual-prior 0.001

# Use 5th/95th percentile thresholds instead of median prior
python run_pipeline.py --dataset my.csv --name MyGene --no-median-prior
```

### Benign sample method

```bash
--benign-method avg         # Average benign and synonymous (default when both exist)
--benign-method benign      # Use benign (ClinVar B/LB) only
--benign-method synonymous  # Use synonymous only
```

### Point-range postprocessing

```bash
# Default: enforce monotonicity + extend to score-axis limits
python run_pipeline.py --dataset my.csv --name MyGene

# Disable postprocessing — intended for bidirectional assays (LoF/GoF)
python run_pipeline.py --dataset my.csv --name MyGene --no-postprocess

# Conservative (stricter) monotonicity enforcement
python run_pipeline.py --dataset my.csv --name MyGene --conservative-monotonicity
```

## Troubleshooting

**"Insufficient samples"**
- Need at least 3 sample categories
- Check for empty samples (all NaN scores)

**SLURM jobs fail**
- Verify `SLURM_ACCOUNT` and `SLURM_PARTITION` match your cluster
- Check `module avail anaconda` and set `CONDA_MODULE` accordingly
- Check logs in `<output_dir>/logs/`

**Low number of valid fits**
- Increase `--n-bootstraps` or `--fits-per-bootstrap`
- Check for score range issues (all variants at same score)

**GPU runs (JAX)**
- Requires `excalibr-gpu.yml` environment (Python 3.11 + JAX)
- Test: `python -c "import jax; print(jax.devices())"`
- Use `CUDA_VISIBLE_DEVICES=0` to pin to a specific GPU

## Citation

If you use ExCALIBR, please cite:

```
Gene-based calibration of high-throughput functional assays for clinical variant classification.
Daniel Zeiberg, Malvika Tejura, Abbye E. McEwen, Shawn Fayer, Vikas Pejaver, Alan F. Rubin,
Lea M. Starita, Douglas M. Fowler, Anne O'Donnell-Luria, Predrag Radivojac
bioRxiv 2025.04.29.651326; doi: https://doi.org/10.1101/2025.04.29.651326
```

For the IGVF dataset and format:

```
A scalable approach to resolving variants of uncertain significance.
Malvika Tejura, Yile Chen, Abbye E. McEwen, Ross Stewart, et al.
bioRxiv 2026.02.14.705848; doi: https://doi.org/10.64898/2026.02.14.705848
```

## License

[MIT License](LICENSE)

## Contact

[stewart.ro@northeastern.edu](mailto:stewart.ro@northeastern.edu)
