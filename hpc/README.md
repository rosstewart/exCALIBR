# ExCALIBR SLURM Workflow

All scripts live in `hpc/` and accept an `<output_dir>` argument so they can be
run from any directory.

---

## Step-by-step

### 1. Generate the job manifest

All manifest modes are unified in `hpc/prepare.py`.  Run with a sub-command:

```bash
# Standard univariate (most common):
python hpc/prepare.py default \
    --output-dir /projects/pedjas_lab/stewart.ro/my_run \
    --dataframe /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz \
    --config-file src/igvf_configs/dataset_configs_jan_2026.json \
    --target-array-size 1000 \
    --n-jobs 30

# MaveDB / label-seq input:
python hpc/prepare.py mavedb \
    --output-dir /projects/pedjas_lab/stewart.ro/my_run \
    --dataframe /data/ross/labelseq/my_data.csv.gz \
    --score-cols "Median activation" "Median PemR" \
    --splice-measure no --n-jobs 30

# BasicScoreset (pre-built sample_assignments column):
python hpc/prepare.py basicscoreset \
    --output-dir /projects/pedjas_lab/stewart.ro/my_run \
    --data-dir /data/ross/predictor_scores/single_gene_calibration_data

# Multi-assay multivariate:
python hpc/prepare.py multivariate \
    --output-dir /projects/pedjas_lab/stewart.ro/my_run \
    --genes BRCA1 BRCA2 --components 2 3 --constraints both

# Predictor-score multivariate (REVEL / MutPred2 / AlphaMissense):
python hpc/prepare.py predictor-mv \
    --output-dir /projects/pedjas_lab/stewart.ro/my_run \
    --data-dir /data/ross/predictor_scores/single_gene_calibration_data \
    --genes BRCA1 --components 2 3

# List available multi-assay genes without generating jobs:
python hpc/prepare.py multivariate --output-dir /tmp/ignore --list-only
```

This writes:
- `<output_dir>/jobs/job_index.json` — lightweight index of all tasks
- `<output_dir>/jobs/array_NNNN.pkl` — one file per SLURM array task
- `<output_dir>/datasets.txt` — list of dataset names being fitted

Datasets are **not** pre-created on disk; the worker creates them on demand.

To skip specific datasets at run time, create:
```
<output_dir>/skip_datasets.txt
```
One dataset name per line; `#` comments are supported.

---

### 2. Submit the array job

```bash
bash hpc/submit_array.sh <output_dir>
```

**Tunable parameters** (set as environment variables before calling):

| Variable | Default | Description |
|---|---|---|
| `SLURM_ACCOUNT` | `predrag` | SLURM billing account |
| `SLURM_JOB_NAME` | `excalibr_bootstrap` | Job name shown in `squeue` |
| `SLURM_TIME` | `23:59:00` | Wall-clock limit per task |
| `SLURM_MEM` | `4G` | Memory per task |
| `SLURM_CPUS` | `12` | CPUs per task (used for parallel fitting) |
| `SLURM_PARTITION` | `short` | Partition / queue |
| `MAX_CONCURRENT` | `50` | Max simultaneously running array tasks |
| `PYTHON` | `/home/rcstewart/.conda/envs/pillar_project/bin/python` | Python interpreter to use |

Example with overrides:

```bash
SLURM_MEM=8G SLURM_CPUS=24 SLURM_TIME=11:00:00 \
    bash hpc/submit_array.sh /projects/pedjas_lab/stewart.ro/my_run
```

Each worker reads `SLURM_CPUS_PER_TASK` from the environment to set its own
`ProcessPoolExecutor` worker count — no extra flags needed.

---

### 3. Monitor progress

```bash
python hpc/count_bootstraps.py <output_dir>
# or with per-dataset missing-seed details:
python hpc/count_bootstraps.py <output_dir> --verbose
```

Shows completed / expected bootstraps per dataset, broken down by component (2c / 3c).

---

### 4. Re-submit failed / partial tasks (optional)

The worker script is **resume-safe**: if `bootstrap_<N>_best_fits.pkl` already exists
and contains a result for `2c` but not `3c`, only the missing component is fitted and
the file is updated in place.  Re-submitting the same array job is therefore safe.

To re-run only specific tasks you can also call the worker directly:

```bash
python hpc/run_array_task.py <output_dir> <array_idx>
```

---

### 5. Aggregate results

Once all tasks are complete:

```bash
python hpc/aggregate_results.py <output_dir>
# writes <output_dir>/bootstrap_results.json.gz

# or to a custom path:
python hpc/aggregate_results.py <output_dir> /path/to/results.json.gz
```

---

## Output layout

```
<output_dir>/
  jobs/
    job_index.json          ← task index; read by all hpc/ scripts
    array_0000.pkl
    array_0001.pkl
    ...
  logs/
    array_<JOB>_<TASK>.out
    array_<JOB>_<TASK>.err
  datasets.txt              ← dataset list written by prepare_batch_jobs.py
  skip_datasets.txt         ← (optional) datasets to skip at runtime
  <dataset_name>/
    bootstrap_0_best_fits.pkl
    bootstrap_1_best_fits.pkl
    ...
  bootstrap_results.json.gz ← written by aggregate_results.py
```
