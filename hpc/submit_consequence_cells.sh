#!/bin/bash
# Submit the consequence-cell simulation grid as a SLURM array.
#
# Usage (on the cluster, from the repo):
#   bash hpc/submit_consequence_cells.sh <out_dir>
#
# <out_dir>/tasks.json comes from
#   python -m analysis.consequence_cells_sim.tasks make <out_dir> [--reps 20] [--B 50] [--R 8]
# The tasks are split over at most N_ARRAYS array indices; each index runs its slice with WORKERS processes and
# writes <out_dir>/runs/<task id>.csv. Finished tasks are skipped, so a resubmission only reruns what is missing.
# Collect with
#   python -m analysis.consequence_cells_sim.tasks collect <out_dir>
#
# SLURM parameters can be overridden via environment variables (as submit_array.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
OUT="$(realpath "${1:?Usage: $0 <out_dir>}")"
TASKS="$OUT/tasks.json"
[ -f "$TASKS" ] || { echo "Error: $TASKS not found (run analysis.consequence_cells_sim.tasks make)"; exit 1; }

WORKERS="${WORKERS:-16}"
N_ARRAYS="${N_ARRAYS:-50}"
N_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASKS'))))")
PER_INDEX="${PER_INDEX:-$(( (N_TASKS + N_ARRAYS - 1) / N_ARRAYS ))}"
MAX_IDX=$(( (N_TASKS + PER_INDEX - 1) / PER_INDEX - 1 ))
mkdir -p "$OUT/logs" "$OUT/runs"

ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_csqcells_sim}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-32G}"
CPUS="${SLURM_CPUS:-$WORKERS}"
PARTITION="${SLURM_PARTITION:-short}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
PYTHON="${PYTHON:-/home/stewart.ro/.conda/envs/pillar_project/bin/python}"

JOB_ID=$(sbatch --parsable << SBATCH_SCRIPT
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT}/logs/sim_%A_%a.out
#SBATCH --error=${OUT}/logs/sim_%A_%a.err
#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --partition=${PARTITION}

# one BLAS thread per worker process (see submit_array.sh)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TQDM_DISABLE=1
export PYTHONPATH=${REPO}
cd ${REPO}
# keep on one line: a backslash continuation here becomes a stray argument
${PYTHON} -m analysis.consequence_cells_sim.tasks run ${OUT} --index \${SLURM_ARRAY_TASK_ID} --per-index ${PER_INDEX} --workers ${WORKERS}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${N_TASKS}, ${PER_INDEX} per array task (indices 0-${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  workers    : ${WORKERS}   mem / cpus : ${MEM} / ${CPUS}   partition ${PARTITION}, time limit ${TIME}"
echo "Progress:   ls ${OUT}/runs/*.csv | wc -l   (of ${N_TASKS})"
