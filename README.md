# Assay Calibration Pipeline

A pipeline for calibrating functional assays using bootstrap skew normal mixture model fitting and Bayesian calibration.

## Overview

This pipeline takes variant effect scores from functional assays and calibrates them to clinical interpretation scales (ACMG/AMP evidence levels) using:

- **Bootstrap mixture modeling** to estimate probability distributions for pathogenic, benign, population, and synonymous variants
- **Bayesian calibration** to compute likelihood ratios and evidence thresholds
- **Statistical model selection** to choose optimal component counts (2c vs 3c)
- **Flexible execution** via SLURM clusters, parallel processing, or single-CPU

## Installation

```bash
# Clone repository
git clone https://github.com/rosstewart/assay_calibration
cd assay_calibration

# Install dependencies
pip install -r requirements.txt

# Requires the assay_calibration package
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Run calibration with default settings (2c and 3c models, parallel execution)
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021

# This will:
# 1. Fit 1000 bootstrap iterations with 100 fits each
# 2. Automatically select between 2c and 3c models
# 3. Generate calibration thresholds and visualizations
# 4. Save results to ./calibration_output/
```

### Input Data Format

Your CSV must contain these columns (or alternatively input a pre-formatted IGVF scoreset):

| Column | Description | Example |
|--------|-------------|---------|
| `score` | Variant effect score | 0.523 |
| `sample` | Sample assignment index (can be multilabel) | "1,2" |
| `Dataset` | Dataset name (optional) | "MyGene_MyLab_2025" |

**Required sample indices (these indices cannot be changed):**
- `0: Pathogenic/Likely Pathogenic` - ClinVar P/LP variants
- `1: Benign/Likely Benign` - ClinVar B/LB variants (optional)
- `2: gnomAD` or `population` - Population variants
- `3: Synonymous` - Synonymous variants (optional)

Note: Must have either Benign or Synonymous samples. Variants can belong to multiple samples via separating each index with ",".

## Execution Modes

### 1. Parallel Execution (Recommended)

Fast and efficient for moderate datasets:

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --mode parallel \
  --n-jobs -1  # Use all available CPUs
```

### 2. SLURM Cluster

For large-scale runs on HPC clusters:

```bash
# Generate SLURM job array (customize for your cluster environment)
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --mode slurm \
  --slurm-account my_account \
  --slurm-partition short \
  --slurm-time 12 \
  --slurm-conda-env my_conda_env \
  --slurm-modules "module load anaconda3/2024.06"

# This creates job files in ./calibration_output/slurm_jobs/
# Review the generated submit.sh script and customize if needed:
nano ./calibration_output/slurm_jobs/submit.sh

# Then submit:
cd ./calibration_output/slurm_jobs
sbatch submit.sh

# After jobs complete, collect results:
python -c "
from utils import collect_slurm_results
from config import PipelineConfig
from pathlib import Path

config = PipelineConfig(
    dataset_csv='example/MSH2_Jia_2021.csv',
    dataset_name='MSH2_Jia_2021',
    output_dir='./calibration_output'
)

results = collect_slurm_results(
    Path('./calibration_output/slurm_jobs'),
    config
)

# Continue with visualization...
"
```

### 3. Single-CPU (Debugging)

Slowest but easiest to debug:

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --mode single
```

## Model Selection

### Automatic Selection (Default)

The pipeline automatically tests 2-component vs 3-component models:

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --components 2 3  # Fit both models
```

**Conservative selection (default):** Uses 5th percentile test - selects 3c only if 95% of bootstrap samples show improvement.

**P-value selection:** Uses Wilcoxon signed-rank test at α=0.05:

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --components 2 3 \
  --no-conservative
```

### Manual Selection

Fit only specific component count:

```bash
# 2-component model only
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --components 2

# 3-component model only
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --components 3
```

## Configuration Options

### Bootstrap Parameters

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --n-bootstraps 500 \        # Default: 1000
  --fits-per-bootstrap 50     # Default: 100
```

### Prior Estimation

```bash
# Use EM estimation (default)
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021

# Use equation for 2c
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --use-equation
  --components 2

# Use 5th/95th percentile thresholds instead of median prior
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --no-median-prior
```

### Benign Sample Method

```bash
# Average benign and synonymous (default if both samples exist)
--benign-method avg

# Use benign samples only. (default if no synonymous sample) 
--benign-method benign

# Use synonymous variants only (default if no benign sample)
--benign-method synonymous
```

### ClinVar Options

```bash
python run_pipeline.py \
  --dataset example/MSH2_Jia_2021.csv \
  --name MSH2_Jia_2021 \
  --clinvar-release 2025 \     # Default: 2025
  --min-clinvar-star 1          # Minimum review stars (default: 1)
```

Note: this functionality is only supported within IGVF-formatted scoresets.

## Output Files

The pipeline generates:

### Essential Outputs

1. **Calibration JSON** (`MSH2_Jia_2021_2c_calibration.json`)
   ```json
   {
     "dataset": "MSH2_Jia_2021",
     "component": "2c",
     "prior": 0.0034,
     "point_ranges": {
       "1": [[0.12, 0.45]],
       "2": [[0.45, 0.78]],
       ...
       "-1": [[-0.45, -0.12]],
       "-2": [[-0.78, -0.45]]
     },
     "scoreset_flipped": true,
     "n_valid_fits": 998,
     "config": {...}
   }
   ```

2. **Visualization** (`MSH2_Jia_2021_2c_visualization.png`)
   - Density plots for each sample
   - Calibration thresholds overlaid

3. **Model Selection Results** (`MSH2_Jia_2021_model_selection.json`)
   ```json
   {
     "selected_k": 2,
     "conservative_k": 2,
     "p_value": 0.1234,
     "mean_diff": 0.0012,
     "fifth_percentile": -0.0003,
     ...
   }
   ```

### Optional Outputs

4. **Full Results** (with `--save-fits`)
   - `MSH2_Jia_2021_2c_full.json.gz` - Complete calibration data
   - `MSH2_Jia_2021_bootstrap_fits.pkl` - All bootstrap fit results

5. **Log Files** (`logs/MSH2_Jia_2021_pipeline.log`)

Optionally, change the output directory with `--output-dir /path/to/output`.


## Troubleshooting

### Common Issues


**"Insufficient samples"**
- Need at least 3 sample categories
- Check for empty samples (all NaN scores)

**SLURM jobs fail**
- Check account/partition settings
- Verify conda environment is activated in submission script
- Check logs in `./calibration_output/slurm_jobs/logs/`

**Low number of valid fits**
- Increase `--n-bootstraps` or `--fits-per-bootstrap`
- Check for score range issues (all variants at same score)
- Review log files for warnings

## Citation

If you use this pipeline, please cite:

```
Gene-based calibration of high-throughput functional assays for clinical variant classification.
Daniel Zeiberg, Malvika Tejura, Abbye E. McEwen, Shawn Fayer, Vikas Pejaver, Alan F. Rubin, Lea M. Starita, Douglas M. Fowler, Anne O’Donnell-Luria, Predrag Radivojac
bioRxiv 2025.04.29.651326; doi: https://doi.org/10.1101/2025.04.29.651326
```

## Contributing

Contributions are welcome! Please submit issues or pull requests to help improve the project.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please contact [stewart.ro@northeastern.edu](mailto:stewart.ro@northeastern.edu).
