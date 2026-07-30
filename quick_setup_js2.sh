#!/bin/bash
# Jetstream2 quick setup for exCALIBR GPU fits.
#
# Sets up the environment, clones the repo, creates the conda env (with JAX
# CUDA), runs prepare.py to build the job manifest, then prints the GPU launch
# command for this instance's index range.
#
# Usage:
#   bash quick_setup_js2.sh <output_dir> [--dataframe <path>] [prepare.py options]
#
# Arguments:
#   output_dir   Directory containing (or to write) the job manifest and results.
#                If jobs/job_index.json already exists (e.g. you uploaded it),
#                prepare.py is skipped entirely — no dataframe needed.
#
# Options:
#   --dataframe <path>   Input dataframe (.csv/.csv.gz). Required only if
#                        jobs/job_index.json does not already exist.
#   Any remaining args are forwarded to prepare.py (e.g. --components 3).
#
# Examples:
#   # jobs dir already uploaded — no dataframe needed:
#   bash quick_setup_js2.sh my_run
#
#   # run prepare.py from scratch:
#   bash quick_setup_js2.sh my_run --dataframe ../data.csv.gz --components 3
#
# After setup, launch GPU fits (fill in START END for this instance's range):
#   CUDA_VISIBLE_DEVICES=0 nohup bash exCALIBR/slurm/run_local_array_gpu.sh \
#       <abs_output_dir> START END > <abs_output_dir>/logs/runner_START_END.log 2>&1 &

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> [--dataframe <path>] [prepare.py options...]}"
shift 1

# Parse --dataframe from remaining args; everything else goes to prepare.py
DATAFRAME=""
PREPARE_EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataframe) DATAFRAME="$2"; shift 2 ;;
        *)           PREPARE_EXTRA_ARGS="$PREPARE_EXTRA_ARGS $1"; shift ;;
    esac
done

REPO_URL="https://github.com/rosstewart/exCALIBR.git"
REPO_DIR="exCALIBR"
BRANCH="gpu"

# ── module ────────────────────────────────────────────────────────────────────
echo "== module load miniforge =="
module load miniforge

if ! grep -qxF "module load miniforge" "$HOME/.bashrc" 2>/dev/null; then
    echo "module load miniforge" >> "$HOME/.bashrc"
    echo "  added to ~/.bashrc"
else
    echo "  already in ~/.bashrc"
fi

# ── repo ──────────────────────────────────────────────────────────────────────
echo "== clone repo =="
if [ -d "$REPO_DIR" ]; then
    echo "  ${REPO_DIR} already exists, skipping clone"
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "== checkout ${BRANCH} branch =="
git fetch origin
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "$BRANCH"
    git merge --ff-only "origin/${BRANCH}"
else
    git checkout -b "$BRANCH" "origin/${BRANCH}"
fi

# ── conda env ─────────────────────────────────────────────────────────────────
echo "== conda env =="
CONDA_BASE="$(conda info --base)"
if conda env list | grep -qE "^excalibr\s"; then
    echo "  excalibr env already exists, skipping create"
else
    conda env create -f excalibr-gpu.yml
fi
# Use explicit python path — conda activate in non-interactive scripts
# does not reliably update PATH for subsequent commands.
PYTHON="${CONDA_BASE}/envs/excalibr/bin/python"

# ── smoke test JAX GPU ────────────────────────────────────────────────────────
echo "== JAX GPU smoke test =="
"$PYTHON" -c "
import jax
devs = jax.devices()
print(f'  JAX devices: {devs}')
if not any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devs):
    print('  WARNING: no GPU device found — JAX will run on CPU')
else:
    print('  GPU detected OK')
"

# ── job manifest ──────────────────────────────────────────────────────────────
echo "== job manifest =="
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS="$(realpath "${OUTPUT_DIR}")"

if [ -f "${OUTPUT_ABS}/jobs/job_index.json" ]; then
    echo "  job_index.json found — skipping prepare.py"
elif [ -n "$DATAFRAME" ]; then
    DATAFRAME_ABS="$(realpath "$DATAFRAME")"
    echo "  running prepare.py with dataframe: ${DATAFRAME_ABS}"
    # shellcheck disable=SC2086
    "$PYTHON" slurm/prepare.py \
        --dataframe "$DATAFRAME_ABS" \
        --output-dir "$OUTPUT_ABS" \
        --n-bootstraps 1000 \
        --num-fits 100 \
        $PREPARE_EXTRA_ARGS
else
    echo "ERROR: ${OUTPUT_ABS}/jobs/job_index.json not found and --dataframe not provided."
    echo "Either upload the jobs directory or pass --dataframe <path>."
    exit 1
fi

NUM_ARRAYS=$("$PYTHON" -c "import json; print(json.load(open('${OUTPUT_ABS}/jobs/job_index.json'))['num_arrays'])")
REPO_ABS="$(pwd)"

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "Setup complete."
echo "  Repo   : ${REPO_ABS}"
echo "  Output : ${OUTPUT_ABS}"
echo "  Tasks  : ${NUM_ARRAYS} (indices 0–$((NUM_ARRAYS - 1)))"
echo ""
echo "Launch GPU fits — split the task range across instances (fill in START END):"
echo ""
echo "  mkdir -p ${OUTPUT_ABS}/logs"
echo "  PYTHON=${PYTHON} CUDA_VISIBLE_DEVICES=0 nohup \\"
echo "      bash ${REPO_ABS}/slurm/run_local_array_gpu.sh \\"
echo "      ${OUTPUT_ABS} START END \\"
echo "      > ${OUTPUT_ABS}/logs/runner_START_END.log 2>&1 &"
echo "  disown"
echo ""
echo "Monitor : ${PYTHON} ${REPO_ABS}/slurm/count_bootstraps.py ${OUTPUT_ABS}"
echo "Aggregate: ${PYTHON} ${REPO_ABS}/slurm/aggregate_results.py ${OUTPUT_ABS}"
