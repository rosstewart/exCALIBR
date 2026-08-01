← [Back to README](../README.md)

# Script Reference

### `run_pipeline.py` — Single-dataset interactive pipeline

Fits, selects models, generates visualizations, and saves calibration JSON for one dataset.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | Path to input CSV/TSV |
| `--name` | (required) | Dataset name for output files |
| `--preset` | `light` | Quality/speed level: `light`/`medium`/`large`/`xl`/`finest` (see [presets](configuration.md#quality-vs-speed-presets)) |
| `--n-bootstraps` / `--fits-per-bootstrap` | from `--preset` | Advanced: override the preset's underlying bootstrap count / fits-per-bootstrap directly |
| `--components K [K ...]` | `3` | Component counts to fit (integers 2–10; pass `2 3` to fit both and compare) |
| `--mode` | `parallel` | `parallel` or `single` (single-process debugging; not for cluster execution — see [batch HPC workflow](batch-hpc-workflow.md)) |
| `--n-jobs` | `-1` | Parallel workers for bootstrap fitting (`-1` = all CPUs) |
| `--device` | `cpu` | `cpu` or `gpu` (GPU: routes through JAX batch, untested as of authoring) |
| `--precomputed-fits` | — | Skip fitting; load existing bootstrap fits JSON |
| `--output-dir` | `./calibration_output` | Output directory |
| `--manual-prior` | — | Override the estimated prior probability (0–1) with your own value; skips estimation (see [Prior estimation](configuration.md#prior-estimation)) |
| `--benign-method` | `avg` | `avg`, `benign`, or `synonymous` |
| `--conservative-monotonicity` | off | Use stricter monotonicity enforcement (default is liberal) |
| `--no-postprocess` | off | Return raw, unprocessed LR-threshold intervals (debugging tool; not needed for bidirectional assays, see [Bidirectional assay auto-detection](configuration.md#bidirectional-assay-auto-detection)) |
| `--pathogenic-percentile` | `5.0` | Conservative (pathogenic-direction) percentile for all bootstrap LR+/threshold calculations |
| `--benign-percentile` | `100 - pathogenic-percentile` | Upper (benign-direction) percentile; set independently to decouple from `--pathogenic-percentile` |
| `--pathomechanism-prior` | off | Enable the mechanism-aware pathogenic-direction prior (see [Pathomechanism prior](configuration.md#pathomechanism-prior-advanced)) |
| `--no-auto-bidirectional` | off (auto-detection on by default) | Disable automatic bidirectional-assay detection (see [Bidirectional assay auto-detection](configuration.md#bidirectional-assay-auto-detection) below). |
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
| `--dataset-configs` | — | (Advanced, most users can skip) JSON mapping dataset names to `[n_c, benign_method, {overrides}]` for per-dataset control. Omit it and every dataset in `--precomputed-fits` is processed with sensible defaults (2c+3c, avg, model selection). |
| `--datasets` | all | Only process these dataset names |
| `--output-dir` | `./igvf_output` | Output directory |
| `--n-jobs` | 1 | Datasets in parallel (outer) |
| `--n-jobs-inner` | -1 | Parallel jobs within each dataset |
| `--skip-existing` | off | Resume a failed run (skip completed datasets) |
| `--all-configs` | off | Run all 4 (2c/3c)×(avg/benign) combos per dataset with comparison plot |
| `--no-postprocess` | off | Return raw, unprocessed LR-threshold intervals (same as `run_pipeline.py`; a debugging tool, not needed for bidirectional assays) |
| `--no-auto-bidirectional` | off (auto-detection on by default) | Disable automatic bidirectional-assay detection (see [Bidirectional assay auto-detection](configuration.md#bidirectional-assay-auto-detection) below). Global, batch-wide — not settable per-dataset via `--dataset-configs`. |
| `--pathogenic-percentile` | `5.0` | Conservative (pathogenic-direction) percentile, batch-wide |
| `--benign-percentile` | `100 - pathogenic-percentile` | Upper (benign-direction) percentile, batch-wide; set independently to decouple |
| `--pathomechanism-prior` | off | Enable the mechanism-aware pathogenic-direction prior, batch-wide (see [Pathomechanism prior](configuration.md#pathomechanism-prior-advanced)) |
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

Copies the selected-model calibration JSON and PNG for each dataset into flat `json/` and `png/` subdirectories, then tars the result for download. If run over SSH, also prints an `scp` command (using your actual username/hostname) to pull the archive down.
