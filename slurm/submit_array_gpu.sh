#!/bin/bash
# GPU counterpart of submit_array.sh — same job manifest, but runs all fits for
# a range of array tasks in one Python process via jax_batch.
#
# Key design: splits the full task range across N_GPUS SLURM jobs, each
# covering consecutive tasks in a single Python process. This amortizes JAX JIT
# compilation (~15 min) over many tasks rather than recompiling once per task.
#
# Usage:
#   bash slurm/submit_array_gpu.sh <output_dir>
#
# Override defaults via environment variables:
#   N_GPUS=8 bash slurm/submit_array_gpu.sh /projects/pedjas_lab/stewart.ro/my_run
#   N_GPUS=4 SLURM_PARTITION=gpu bash slurm/submit_array_gpu.sh /projects/pedjas_lab/stewart.ro/my_run

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
ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_bootstrap_gpu}"
TIME="${SLURM_TIME:-23:59:00}"
MEM="${SLURM_MEM:-16G}"
CPUS="${SLURM_CPUS:-4}"
PARTITION="${SLURM_PARTITION:-gpu}"
GRES="${SLURM_GRES:-gpu:1}"
N_GPUS="${N_GPUS:-8}"
PYTHON="${PYTHON:-/home/stewart.ro/.conda/envs/pillar_project/bin/python}"
# ──────────────────────────────────────────────────────────────────────────────

# Each SLURM array element covers this many consecutive array tasks
TASKS_PER_GPU=$(( (NUM_ARRAYS + N_GPUS - 1) / N_GPUS ))

JOB_ID=$(sbatch --parsable << SBATCH_SCRIPT
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_DIR}/logs/gpu_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/logs/gpu_%A_%a.err
#SBATCH --array=0-$((N_GPUS - 1))
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}

START=\$(( \${SLURM_ARRAY_TASK_ID} * ${TASKS_PER_GPU} ))
END=\$(( START + ${TASKS_PER_GPU} - 1 ))
[ \${END} -gt ${MAX_IDX} ] && END=${MAX_IDX}

echo "GPU job \${SLURM_ARRAY_TASK_ID}: array tasks \${START}–\${END}"
${PYTHON} ${SCRIPT_DIR}/run_array_task.py ${OUTPUT_DIR} \${START} --end \${END} --device gpu
SBATCH_SCRIPT
)

echo "Submitted GPU job array ${JOB_ID}"
echo "  Tasks      : ${NUM_ARRAYS}  (indices 0–${MAX_IDX})"
echo "  GPU jobs   : ${N_GPUS}  (${TASKS_PER_GPU} tasks/job, all run concurrently)"
echo "  mem / cpus : ${MEM} / ${CPUS}   gres: ${GRES}"
echo "  Partition  : ${PARTITION}   time limit: ${TIME}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Logs       : ${OUTPUT_DIR}/logs/gpu_<jobid>_<0..${N_GPUS}-1>.{out,err}"
echo ""
echo "Monitor:   python ${SCRIPT_DIR}/count_bootstraps.py ${OUTPUT_DIR}"
echo "Aggregate: python ${SCRIPT_DIR}/aggregate_results.py ${OUTPUT_DIR}"
