#!/bin/bash
# Submit a SLURM array job for ExCALIBR bootstrap fits.
#
# Usage:
#   bash hpc/submit_array.sh <output_dir>
#
# The job array size is read automatically from <output_dir>/jobs/job_index.json.
# All SLURM parameters can be overridden via environment variables (see README.md).
#
# Example:
#   bash hpc/submit_array.sh /projects/pedjas_lab/stewart.ro/my_run
#   SLURM_MEM=8G SLURM_CPUS=24 bash hpc/submit_array.sh /projects/pedjas_lab/stewart.ro/my_run
#
# Task layout is a submission-time choice (no need to re-run prepare.py). The
# default is one task per concurrent slot (TASKS=MAX_CONCURRENT): with a fixed
# number of slots, extra tasks only queue behind the first wave and start
# their long fits late (docs/fit_speed_report.md, "Static vs dynamic").
#   TASKS=150 bash hpc/submit_array.sh <dir>   # e.g. when one-wave tasks would exceed SLURM_TIME
#   TASKS=auto bash hpc/submit_array.sh <dir>  # count chosen by make_layout for MAX_CONCURRENT x SLURM_CPUS
#   TASKS=prepare bash hpc/submit_array.sh <dir>  # prepare.py's own layout
# hpc/make_layout.py balances the compact jobs (dataset x bootstrap) across the
# tasks from prepare.py's fit-time estimates, longest tasks first, and prints
# the predicted CPU use (and a smaller SLURM_CPUS if extra cores would sit
# idle). Scheduling only: fit results are bitwise unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:?Usage: $0 <output_dir>}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

JOB_INDEX="$OUTPUT_DIR/jobs/job_index.json"
if [ ! -f "$JOB_INDEX" ]; then
    echo "Error: $JOB_INDEX not found."
    echo "Run prepare_batch_jobs.py first to generate the job manifest."
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"

# ── SLURM parameters ───────────────────────────────────────────────────────────
# Override any of these with environment variables before calling this script.
ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_bootstrap}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-1G}"
CPUS="${SLURM_CPUS:-16}"
PARTITION="${SLURM_PARTITION:-short}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"        # max simultaneously running tasks
PYTHON="${PYTHON:-/home/stewart.ro/.conda/envs/pillar_project/bin/python}"
TASKS="${TASKS:-$MAX_CONCURRENT}"             # default: one task per concurrent slot (one wave);
                                              # N or auto: re-lay out; "prepare": prepare.py's layout
# ──────────────────────────────────────────────────────────────────────────────

LAYOUT_ARG=""
if [ "$TASKS" != "prepare" ]; then
    LAYOUT_FILE=$(${PYTHON} ${SCRIPT_DIR}/make_layout.py "$OUTPUT_DIR" --tasks "$TASKS" \
        --cpus "$CPUS" --max-concurrent "$MAX_CONCURRENT" | tail -n 1)
    NUM_ARRAYS=$(python3 -c "import json; print(json.load(open('$LAYOUT_FILE'))['n_tasks'])")
    LAYOUT_ARG="--layout ${LAYOUT_FILE}"
else
    NUM_ARRAYS=$(python3 -c "import json; print(json.load(open('$JOB_INDEX'))['num_arrays'])")
fi
MAX_IDX=$((NUM_ARRAYS - 1))

JOB_ID=$(sbatch --parsable << SBATCH_SCRIPT
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_DIR}/logs/array_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/array_%A_%a.err
#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --partition=${PARTITION}

# See run_local_array.sh's identical exports / run_array_task.py's own
# os.environ.setdefault safety net: without this, each ProcessPoolExecutor
# worker below can spawn its own multi-threaded BLAS trying to use every
# visible core, oversubscribing --cpus-per-task by up to its square.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${PYTHON} ${SCRIPT_DIR}/run_array_task.py ${OUTPUT_DIR} \${SLURM_ARRAY_TASK_ID} ${LAYOUT_ARG}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${NUM_ARRAYS}  (indices 0–${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
if [ -n "$LAYOUT_ARG" ]; then echo "  Layout     : ${LAYOUT_FILE}"; fi
echo "  mem / cpus : ${MEM} / ${CPUS}"
echo "  Partition  : ${PARTITION}   time limit: ${TIME}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Logs       : ${OUTPUT_DIR}/logs/"
echo ""
echo "Monitor:   python ${SCRIPT_DIR}/count_bootstraps.py ${OUTPUT_DIR}"
echo "Aggregate: python ${SCRIPT_DIR}/aggregate_results.py ${OUTPUT_DIR}"
