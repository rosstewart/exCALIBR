#!/bin/bash
# Local (non-SLURM) GPU counterpart of run_local_array.sh — runs a range of
# array indices on this machine's GPU, batched through
# src/assay_calibration/fit_utils/jax_batch instead of a CPU ProcessPoolExecutor.
#
# Untested on GPU as of authoring — run tests/test_batch_em_parity.py first
# (see its docstring) before trusting this for a real run.
#
# Usage:
#   bash slurm/run_local_array_gpu.sh <output_dir> <start_idx> <end_idx>
#
# Unlike run_local_array.sh there's no `concurrency` argument: GPU batching
# already parallelizes across all of an array task's jobs internally, and
# running multiple array indices at once would just contend for the same
# device. Set CUDA_VISIBLE_DEVICES to pick which GPU to use on a
# multi-GPU/shared machine (see nvidia-smi).
#
# Examples:
#   bash slurm/run_local_array_gpu.sh /shared/mv_run 0   9
#   CUDA_VISIBLE_DEVICES=1 bash slurm/run_local_array_gpu.sh /shared/mv_run 0 9

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:?Usage: $0 <output_dir> <start_idx> <end_idx>}"
START="${2:?Usage: $0 <output_dir> <start_idx> <end_idx>}"
END="${3:?Usage: $0 <output_dir> <start_idx> <end_idx>}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Running GPU array indices ${START}-${END} on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, JAX default>}"
echo "  logs: ${LOG_DIR}/array_NNNN.log"

PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

# Run all indices in ONE Python process so JAX JIT cache stays alive across
# array tasks. A fresh process per task would recompile (~10 min) every time,
# turning a ~11h run into a week of compilation overhead.
log="${LOG_DIR}/array_$(printf '%04d' "$START")_to_$(printf '%04d' "$END").log"
$PYTHON "${SCRIPT_DIR}/run_array_task.py" "$OUTPUT_DIR" "$START" \
    --end "$END" --device gpu \
    2>&1 | tee "$log"

echo "Done: indices ${START}-${END} on $(hostname)"
