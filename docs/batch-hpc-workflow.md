← [Back to README](../README.md)

# Batch HPC Workflow

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
