#!/bin/bash
# GPU counterpart of submit_array.sh — same job manifest, but runs each array
# task's fits batched through src/assay_calibration/fit_utils/jax_batch on a
# GPU instead of one fit per CPU core.
#
# Untested on GPU as of authoring — run tests/test_batch_em_parity.py first
# (see its docstring) before trusting this for a real run.
#
# Usage:
#   bash slurm/submit_array_gpu.sh <output_dir>
#
# Example:
#   bash slurm/submit_array_gpu.sh /projects/pedjas_lab/stewart.ro/my_run
#   SLURM_GRES=gpu:1 SLURM_PARTITION=gpu bash slurm/submit_array_gpu.sh /projects/pedjas_lab/stewart.ro/my_run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:?Usage: $0 <output_dir>}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

JOB_INDEX="$OUTPUT_DIR/jobs/job_index.json"
if [ ! -f "$JOB_INDEX" ]; then
    echo "Error: $JOB_INDEX not found."
    echo "Run prepare.py first to generate the job manifest."
    exit 1
fi

NUM_ARRAYS=$(python3 -c "import json; print(json.load(open('$JOB_INDEX'))['num_arrays'])")
MAX_IDX=$((NUM_ARRAYS - 1))

mkdir -p "$OUTPUT_DIR/logs"

# ── SLURM parameters ───────────────────────────────────────────────────────────
# Override any of these with environment variables before calling this script.
# GPU batching processes many jobs per array task at once (see jax_batch's
# module docstring), so far fewer concurrent array tasks are typically needed
# than the CPU path's default of 50.
ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_bootstrap_gpu}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-16G}"
CPUS="${SLURM_CPUS:-4}"
PARTITION="${SLURM_PARTITION:-gpu}"
GRES="${SLURM_GRES:-gpu:1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
PYTHON="${PYTHON:-/home/stewart.ro/.conda/envs/pillar_project/bin/python}"
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
#SBATCH --gres=${GRES}

${PYTHON} ${SCRIPT_DIR}/run_array_task.py ${OUTPUT_DIR} \${SLURM_ARRAY_TASK_ID} --device gpu
SBATCH_SCRIPT
)

echo "Submitted GPU job array ${JOB_ID}"
echo "  Tasks      : ${NUM_ARRAYS}  (indices 0–${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  mem / cpus : ${MEM} / ${CPUS}   gres: ${GRES}"
echo "  Partition  : ${PARTITION}   time limit: ${TIME}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Logs       : ${OUTPUT_DIR}/logs/"
echo ""
echo "Monitor:   python ${SCRIPT_DIR}/count_bootstraps.py ${OUTPUT_DIR}"
echo "Aggregate: python ${SCRIPT_DIR}/aggregate_results.py ${OUTPUT_DIR}"
