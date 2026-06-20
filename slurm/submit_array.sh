#!/bin/bash
# Submit a SLURM array job for ExCALIBR bootstrap fits.
#
# Usage:
#   bash slurm/submit_array.sh <output_dir>
#
# The job array size is read automatically from <output_dir>/jobs/job_index.json.
# All SLURM parameters can be overridden via environment variables (see README.md).
#
# Example:
#   bash slurm/submit_array.sh /projects/pedjas_lab/stewart.ro/my_run
#   SLURM_MEM=8G SLURM_CPUS=24 bash slurm/submit_array.sh /projects/pedjas_lab/stewart.ro/my_run

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

# Read number of array tasks from the job index
NUM_ARRAYS=$(python3 -c "import json; print(json.load(open('$JOB_INDEX'))['num_arrays'])")
MAX_IDX=$((NUM_ARRAYS - 1))

mkdir -p "$OUTPUT_DIR/logs"

# ── SLURM parameters ───────────────────────────────────────────────────────────
# Override any of these with environment variables before calling this script.
ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_bootstrap}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-4G}"
CPUS="${SLURM_CPUS:-12}"
PARTITION="${SLURM_PARTITION:-short}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"        # max simultaneously running tasks
PYTHON="${PYTHON:-/home/rcstewart/.conda/envs/pillar_project/bin/python}"
# ──────────────────────────────────────────────────────────────────────────────

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

${PYTHON} ${SCRIPT_DIR}/run_array_task.py ${OUTPUT_DIR} \${SLURM_ARRAY_TASK_ID}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${NUM_ARRAYS}  (indices 0–${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  mem / cpus : ${MEM} / ${CPUS}"
echo "  Partition  : ${PARTITION}   time limit: ${TIME}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Logs       : ${OUTPUT_DIR}/logs/"
echo ""
echo "Monitor:   python ${SCRIPT_DIR}/count_bootstraps.py ${OUTPUT_DIR}"
echo "Aggregate: python ${SCRIPT_DIR}/aggregate_results.py ${OUTPUT_DIR}"
