#!/bin/bash
# Submit consequence-paper strata fits (csq_analysis.fits) as a SLURM array.
#
# Usage (on the cluster, from the repo):
#   bash hpc/submit_csq.sh <bundle_dir>
#
# <bundle_dir> holds tasks.json, bundle.json and rows.pkl from
#   python -m csq_analysis.fits make <bundle_dir> --configs ... --B 100 --R 20
# Tasks are split over at most N_ARRAYS array indices; each index runs its slice with WORKERS processes and writes
# <bundle_dir>/fits/<config>/<dataset>__<s0>-<s1>.json.gz (+ strata/<config>/<dataset>.json). Finished tasks are
# skipped, so a resubmission only reruns what is missing. Tar fits/ and strata/ back when done.
#
# SLURM parameters can be overridden via environment variables (as submit_array.sh).
# MODULE=csq_analysis.sim submits a paper simulation bundle (python -m csq_analysis.sim make <bundle_dir>) instead;
# it writes <bundle_dir>/runs/<task>.csv and needs no rows.pkl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
OUT="$(realpath "${1:?Usage: $0 <bundle_dir>}")"
TASKS="$OUT/tasks.json"
MODULE="${MODULE:-csq_analysis.fits}"
[ -f "$TASKS" ] || { echo "Error: $TASKS not found (run ${MODULE} make)"; exit 1; }
if [ "$MODULE" = "csq_analysis.fits" ]; then
    [ -f "$OUT/rows.pkl" ] || { echo "Error: $OUT/rows.pkl not found"; exit 1; }
fi

WORKERS="${WORKERS:-16}"
N_ARRAYS="${N_ARRAYS:-50}"
N_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASKS'))))")
PER_INDEX="${PER_INDEX:-$(( (N_TASKS + N_ARRAYS - 1) / N_ARRAYS ))}"
MAX_IDX=$(( (N_TASKS + PER_INDEX - 1) / PER_INDEX - 1 ))
mkdir -p "$OUT/logs" "$OUT/fits" "$OUT/strata"

ACCOUNT="${SLURM_ACCOUNT:-predrag}"
JOB_NAME="${SLURM_JOB_NAME:-excalibr_csq_fits}"
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
#SBATCH --output=${OUT}/logs/fit_%A_%a.out
#SBATCH --error=${OUT}/logs/fit_%A_%a.err
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
${PYTHON} -m ${MODULE} run ${OUT} --index \${SLURM_ARRAY_TASK_ID} --per-index ${PER_INDEX} --workers ${WORKERS}
SBATCH_SCRIPT
)

echo "Submitted job array ${JOB_ID}"
echo "  Tasks      : ${N_TASKS}, ${PER_INDEX} per array task (indices 0-${MAX_IDX}, max ${MAX_CONCURRENT} concurrent)"
echo "  workers    : ${WORKERS}   mem / cpus : ${MEM} / ${CPUS}   partition ${PARTITION}, time limit ${TIME}"
if [ "$MODULE" = "csq_analysis.fits" ]; then
    echo "Progress:   find ${OUT}/fits -name '*.json.gz' | wc -l   (of ${N_TASKS})"
else
    echo "Progress:   ls ${OUT}/runs/*.csv | wc -l   (of ${N_TASKS})"
fi
