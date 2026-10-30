#!/bin/bash
# Submit the cross-fitted classifier calibration as a SLURM array.
#
# Usage (on the cluster, from the repo):
#   bash hpc/submit_classifier_cal.sh <bundle_dir>
#
# <bundle_dir> comes from hpc/prepare_classifier_cal.py. Each array task runs
# TASKS_PER_INDEX entries of <bundle_dir>/tasks.json with that many worker
# processes (python -m mv_analysis.production_oob excalcv --task-file ...),
# writing <bundle_dir>/production_oob/<key>/split<i>_excalcv.npz. Finished
# splits are skipped, so a resubmission only reruns what is missing.
#
# SLURM parameters can be overridden via environment variables (as submit_array.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
BUNDLE="$(realpath "${1:?Usage: $0 <bundle_dir>}")"
TASKS="$BUNDLE/tasks.json"
[ -f "$TASKS" ] || { echo "Error: $TASKS not found (run hpc/prepare_classifier_cal.py)"; exit 1; }

TASKS_PER_INDEX="${TASKS_PER_INDEX:-4}"
N_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASKS'))))")
MAX_IDX=$(( (N_TASKS + TASKS_PER_INDEX - 1) / TASKS_PER_INDEX - 1 ))
mkdir -p "$BUNDLE/logs"

ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_classifier_cal}"
TIME="${SLURM_TIME:-04:00:00}"
MEM="${SLURM_MEM:-8G}"
CPUS="${SLURM_CPUS:-$TASKS_PER_INDEX}"
PARTITION="${SLURM_PARTITION:-short}"
MAX_CONCURRENT="${MAX_CONCURRENT:-200}"
PYTHON="${PYTHON:-/home/stewart.ro/.conda/envs/pillar_project/bin/python}"

JOB_ID=$(sbatch --parsable << SBATCH_SCRIPT
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${BUNDLE}/logs/cal_%A_%a.out
#SBATCH --error=${BUNDLE}/logs/cal_%A_%a.err
#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --partition=${PARTITION}

# one BLAS thread per worker process (see submit_array.sh)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TQDM_DISABLE=1
export MV_PKNN_DIR=${BUNDLE} PYTHONPATH=${REPO}
cd ${REPO}
${PYTHON} -m mv_analysis.production_oob excalcv --task-file ${TASKS} --task-index \${SLURM_ARRAY_TASK_ID} \\
    --tasks-per-index ${TASKS_PER_INDEX} --workers ${TASKS_PER_INDEX}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${N_TASKS} splits, ${TASKS_PER_INDEX} per array task (indices 0-${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  mem / cpus : ${MEM} / ${CPUS}   partition ${PARTITION}, time limit ${TIME}"
echo "  Outputs    : ${BUNDLE}/production_oob/<key>/split<i>_excalcv.npz"
echo "Progress:   ls ${BUNDLE}/production_oob/*/split*_excalcv.npz | wc -l   (of ${N_TASKS})"
