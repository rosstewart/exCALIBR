#!/bin/bash
# Submit the control-composition (E3) and overlap (E6) simulation grids as a SLURM array.
#
# Usage (on the cluster, from the repo):
#   bash hpc/submit_composition_sim.sh <out_dir>
#
# <out_dir>/tasks.json comes from
#   python -m analysis.composition_simulation.tasks make <out_dir>
# Each array index runs TASKS_PER_INDEX tasks with that many worker processes and
# writes <out_dir>/runs/<task id>.csv. Finished tasks are skipped, so a
# resubmission only reruns what is missing. Collect with
#   python -m analysis.composition_simulation.tasks collect <out_dir>
#
# SLURM parameters can be overridden via environment variables (as submit_array.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
OUT="$(realpath "${1:?Usage: $0 <out_dir>}")"
TASKS="$OUT/tasks.json"
[ -f "$TASKS" ] || { echo "Error: $TASKS not found (run analysis.composition_simulation.tasks make)"; exit 1; }

TASKS_PER_INDEX="${TASKS_PER_INDEX:-16}"
N_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASKS'))))")
MAX_IDX=$(( (N_TASKS + TASKS_PER_INDEX - 1) / TASKS_PER_INDEX - 1 ))
mkdir -p "$OUT/logs"

ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_composition_sim}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-16G}"
CPUS="${SLURM_CPUS:-$TASKS_PER_INDEX}"
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
${PYTHON} -m analysis.composition_simulation.tasks run ${OUT} --index \${SLURM_ARRAY_TASK_ID} --per-index ${TASKS_PER_INDEX} --workers ${TASKS_PER_INDEX}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${N_TASKS}, ${TASKS_PER_INDEX} per array task (indices 0-${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  mem / cpus : ${MEM} / ${CPUS}   partition ${PARTITION}, time limit ${TIME}"
echo "Progress:   ls ${OUT}/runs/*.csv | wc -l   (of ${N_TASKS})"
