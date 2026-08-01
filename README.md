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

Minimal format: `score` column + `sample_assignments` column. No ClinVar
metadata required — you just tell it which rows belong to which sample
group yourself. `sample_assignments` is an integer per row (`0`, `1`,
`2`, `3`, ...), or a comma-separated string for variants that belong to
more than one group (e.g. `"1,2"`). By convention, column `0` = Pathogenic,
`1` = Benign, `2` = gnomAD/population, `3` = Synonymous — use
`--sample-names` to relabel them if your groups don't match that
convention.

`example/brca_findlay_example.csv` is a ready-to-run BasicScoreset example
(BRCA1 SGE functional scores from Findlay et al. 2018, with rows labeled
by ClinVar/population group membership):

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
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
| `--no-auto-bidirectional` | off (auto-detection on by default) | Disable automatic bidirectional-assay detection (see [Bidirectional assay auto-detection](#bidirectional-assay-auto-detection) below). |
| `--pathogenic-percentile` | `5.0` | Conservative (pathogenic-direction) percentile for all bootstrap LR+/threshold calculations |
| `--benign-percentile` | `100 - pathogenic-percentile` | Upper (benign-direction) percentile; set independently to decouple from `--pathogenic-percentile` |
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
| `--no-auto-bidirectional` | off (auto-detection on by default) | Disable automatic bidirectional-assay detection (see [Bidirectional assay auto-detection](#bidirectional-assay-auto-detection) below). Global, batch-wide — not settable per-dataset via `--dataset-configs`. |
| `--pathogenic-percentile` | `5.0` | Conservative (pathogenic-direction) percentile, batch-wide |
| `--benign-percentile` | `100 - pathogenic-percentile` | Upper (benign-direction) percentile, batch-wide; set independently to decouple |
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

The calibration works by repeatedly refitting the model to resampled
("bootstrapped") versions of your data, then combining the results — this
is what makes the output stable and gives you a confidence range instead
of a single brittle fit. Two settings control how much of this resampling
is done:

- `--n-bootstraps`: how many resampled versions of the dataset get fit.
  This is the main quality/speed dial.
- `--fits-per-bootstrap`: for each resampled version, how many times the
  fitting is retried from a different random starting point (only the
  best-scoring attempt is kept). A secondary dial.

Turning either one up gives a more stable, reproducible result, at the
cost of more compute time.

#### Quality vs. speed presets

We tested how much the final result changes as `--n-bootstraps` is
lowered, using 87 real assay datasets and comparing each reduced run
against a much larger (~1000-bootstrap) run of the same dataset as a
"ground truth" reference. The easiest number to interpret is: *out of all
the variants in a dataset, what percent end up in a different ACMG
category (e.g. "Likely Benign" instead of "Benign") than they would with
the much larger reference run?*

| `--n-bootstraps` tested | Typical dataset | A harder-than-typical dataset (worst ~10%) |
|---|---|---|
| 20  | ~14 in 100 variants change category | up to ~61 in 100 |
| 50  | ~9 in 100 variants change category  | up to ~51 in 100 |
| 100 | ~6 in 100 variants change category  | up to ~34 in 100 |
| 250 | ~3 in 100 variants change category  | up to ~17 in 100 |
| 500 | ~2 in 100 variants change category  | up to ~10 in 100 |

Overall classification performance (how well the calibration separates
pathogenic from benign variants) barely changes even at 20 bootstraps for
a typical dataset — it's specific individual variants near a category
boundary that are more likely to move. So the numbers above should be read
as "how many variants might sit close enough to a boundary to flip," not
as "the calibration is unreliable."

**Caveat:** this comparison only varies `--n-bootstraps`; it always used
`--fits-per-bootstrap 100`, not the pipeline's default of 8. The presets
below assume the two dials affect quality in a similar way, which we
believe is reasonable but have not separately confirmed.

Based on this, here are five presets to choose from:

| Preset | `--n-bootstraps` | `--fits-per-bootstrap` | What to expect |
|---|---|---|---|
| Light (default) | 20   | 8   | Fastest option, good for a first look. ~14 in 100 variants could land in a different category than a much larger run. |
| Medium           | 100  | 8   | Noticeably more stable, still practical to run on a laptop/desktop. ~6 in 100 variants could shift. |
| Large            | 500  | 8   | Good for a result you plan to rely on. ~2 in 100 variants could shift. Best run on a shared server or cluster. |
| XL               | 1000 | 8   | Matches the bootstrap count used as the reference standard above, so drift should be minimal — but we only directly confirmed this at `--fits-per-bootstrap 100`, not 8. Needs a server/cluster. |
| Finest           | 1000 | 100 | The reference-quality configuration itself. Very slow — intended for a compute cluster, not a personal computer. |

```bash
# Light (default) — fast, exploratory
python run_pipeline.py --dataset my.csv --name MyGene

# Medium — better stability, still practical on a laptop/desktop
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 100 --fits-per-bootstrap 8

# Large
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 500 --fits-per-bootstrap 8

# XL
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 8

# Finest — the reference-quality configuration; run on a cluster
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 100
```

For Large/XL/Finest on many datasets, prefer the
[batch HPC workflow](#batch-hpc-workflow) (SLURM array jobs, one dataset
per node) instead of running them one at a time on your own computer.

#### Speed estimates

How long a run takes mostly depends on two things: how big your preset is
(`--n-bootstraps × --fits-per-bootstrap`) and how many CPU cores your
computer can devote to the job (`--n-jobs`; use `--n-jobs -1` to use all
available cores). Almost all your CPU cores can work on this at the same
time, so more cores means a roughly proportional speedup.

We timed the example dataset (`example/MSH2_Jia_2021.csv`, 1579 variants)
at the Light preset (20 bootstraps × 8 fits) and scaled that measurement
up for the other presets, assuming the same proportional speedup on more
cores. Real times will vary by dataset size and computer, but this should
give a reasonable ballpark:

| Preset | 4 cores (typical laptop) | 16 cores (workstation) | 64 cores (server/cluster node) |
|---|---|---|---|
| Light  | ~2.5 hours | ~35 min | ~10 min |
| Medium | ~12 hours  | ~3 hours | ~45 min |
| Large  | ~2.5 days  | ~16 hours | ~4 hours |
| XL     | ~5 days    | ~1.3 days | ~8 hours |
| Finest | ~2 months  | ~16 days | ~4 days |

A few notes:
- These are for a single dataset, single gene/assay. If you're calibrating
  many datasets at once, use the [batch HPC workflow](#batch-hpc-workflow)
  so datasets run in parallel across a cluster instead of one after another.
- Most of this time (well over 90%, in our test) goes to the bootstrap
  fitting step itself; the plotting/export steps that follow take well
  under a minute regardless of preset.
- Fitting only a 3-component model (the default) is slightly slower than
  fitting only a 2-component model (roughly 15-20% slower in our test) —
  fitting both (`--components 2 3`) takes about as long as the two added
  together.

You can reproduce or extend these measurements with
`tests/benchmark_run_pipeline_speed.py`.

### Component selection

```bash
# Default: 3-component only (assumed at least as good as 2c for most assays,
# not always true, but a reasonable default for typical usage)
python run_pipeline.py --dataset my.csv --name MyGene

# Fit both 2c and 3c and auto-select the better one
python run_pipeline.py --dataset my.csv --name MyGene --components 2 3

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

# Disable postprocessing entirely — intended for bidirectional assays (LoF/GoF)
# where standard monotonicity assumptions do not hold
python run_pipeline.py --dataset my.csv --name MyGene --no-postprocess

# Conservative (stricter) monotonicity enforcement
python run_pipeline.py --dataset my.csv --name MyGene --conservative-monotonicity
```

### Bidirectional assay auto-detection

Some assays (e.g. LoF/GoF in one assay) show pathogenic-leaning evidence on
**both** sides of a benign region — the standard single-direction
monotonicity/extend-to-limits postprocessing assumes one direction and is
inappropriate for these. Instead of requiring `--no-postprocess` by hand,
`run_pipeline.py` and `run_igvf_batch.py` **auto-detect this pattern by
default** (`n_c >= 3` fits only):

- For each bootstrap fit, mixture components are sorted along the score
  axis and labeled pathogenic-like/benign-like by comparing the pathogenic
  and benign samples' mixture weights (PU/NU sample-availability cases fall
  back to gnomAD in place of whichever of pathogenic/benign is unavailable).
  A fit is flagged if a benign-like component has a pathogenic-like
  component on each side.
- If a majority of bootstrap fits are flagged, standard monotonicity
  enforcement is skipped for that dataset's **pathogenic** tiers (each side
  of the benign region is independently re-nested and extended toward its
  own axis limit, honoring `--conservative-monotonicity` per side). Benign
  tiers are assumed to never be bidirectional: they still get cleaned up
  (noisy same-tier fragmentation merged in liberal mode; the standard
  strict "evidence goes back to indeterminate" removal in conservative
  mode), but are never extended to the axis limit.
- Otherwise, standard postprocessing runs unchanged.

```bash
# Default: auto-detection on
python run_pipeline.py --dataset my.csv --name MyGene --components 3

# Disable auto-detection; use standard postprocessing unconditionally
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional

# Disable auto-detection AND all postprocessing (old bidirectional-assay workflow)
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional --no-postprocess
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
