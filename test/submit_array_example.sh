#!/bin/bash
#SBATCH --account=predrag
#SBATCH --job-name=calibration_bootstrap
#SBATCH --output=/projects/pedjas_lab/stewart.ro/assay_calibration/explorer_jobs_predictors_multivariate/all_c4_unc/logs/array_%A_%a.out
#SBATCH --error=/projects/pedjas_lab/stewart.ro/assay_calibration/explorer_jobs_predictors_multivariate/all_c4_unc/logs/array_%A_%a.err
#SBATCH --array=0-999
#SBATCH --time=23:59:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=12
#SBATCH --partition=short

# Create logs directory
mkdir -p /projects/pedjas_lab/stewart.ro/assay_calibration/explorer_jobs_predictors_multivariate/all_c4_unc/logs

#module load anaconda3/2024.06
#source $HOME/.bashrc
#conda activate pillar_project

/home/stewart.ro/.conda/envs/pillar_project/bin/python run_array_task.py /projects/pedjas_lab/stewart.ro/assay_calibration/explorer_jobs_predictors_multivariate/all_c4_unc/jobs $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID completed"
