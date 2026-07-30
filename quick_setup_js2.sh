#!/bin/bash
# Jetstream2 quick setup for exCALIBR multivariate 4c/unc fits.
#
# Assumes the input dataframe is already uploaded, one directory above wherever
# this script is run from, as: ../labelseq_dataframe_processed.csv.gz
#
# Does: module load -> clone -> checkout multivariate -> conda env create ->
# activate -> prepare.py multivariate (job manifest). Prints the
# run_local_array.sh command at the end — fill in START/END yourself for
# whatever index range this instance should handle.
#
# Usage:
#   bash quick_setup_js2.sh

set -euo pipefail

REPO_URL="https://github.com/rosstewart/exCALIBR.git"
REPO_DIR="exCALIBR"
BRANCH="multivariate"
DATAFRAME="../labelseq_dataframe_processed.csv.gz"
OUTPUT_DIR="labelseq_4c_fits"

echo "== module load miniforge =="
module load miniforge

echo "== persist module load for future shells =="
if ! grep -qxF "module load miniforge" "$HOME/.bashrc" 2>/dev/null; then
    echo "module load miniforge" >> "$HOME/.bashrc"
    echo "  added to ~/.bashrc"
else
    echo "  already in ~/.bashrc"
fi

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

echo "== conda env =="
if conda env list | grep -qE "^excalibr\s"; then
    echo "  excalibr env already exists, skipping create"
else
    conda env create -f excalibr.yml
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate excalibr

echo "== prepare.py multivariate (job manifest) =="
if [ -f "${OUTPUT_DIR}/jobs/job_index.json" ]; then
    echo "  ${OUTPUT_DIR}/jobs/job_index.json already exists, skipping prepare"
else
    python slurm/prepare.py multivariate \
        --dataframe "$DATAFRAME" \
        --output-dir "$OUTPUT_DIR" \
        --components 4 \
        --constraints unc \
        --population-type gnomAD \
        --n-bootstraps 1000 \
        --num-fits 100
fi

REPO_ABS="$(pwd)"
OUTPUT_ABS="${REPO_ABS}/${OUTPUT_DIR}"

echo ""
echo "Setup complete. Repo at: ${REPO_ABS}"
echo ""
echo "Run this from ANY directory (uses absolute paths) to launch fits for this"
echo "instance's index range (fill in START/END, e.g. 0 499):"
echo "  module load miniforge && conda activate excalibr"
echo "  nohup bash ${REPO_ABS}/slurm/run_local_array.sh ${OUTPUT_ABS} START END > ${OUTPUT_ABS}/runner_START_END.log 2>&1 &"
echo "  disown"
echo ""
echo "Monitor: python ${REPO_ABS}/slurm/count_bootstraps.py ${OUTPUT_ABS}"
