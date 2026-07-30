#!/bin/bash
# Local (non-SLURM) replacement for submit_array.sh — runs a range of array
# indices on this machine using run_array_task.py's own internal
# ProcessPoolExecutor for parallelism (sized via SLURM_CPUS_PER_TASK).
#
# Usage:
#   bash slurm/run_local_array.sh <output_dir> <start_idx> <end_idx> [concurrency]
#
# concurrency (default 1): how many array indices to run at once. Each one
# gets nproc/concurrency cores. Leave at 1 unless a single array task's job
# count is too small to saturate all cores on its own (e.g. tiny
# --target-array-size chunks) — concurrency > 1 trades per-task core count
# for less idle time between tasks.
#
# Examples:
#   bash slurm/run_local_array.sh /shared/mv_run 0   499
#   bash slurm/run_local_array.sh /shared/mv_run 500 999

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:?Usage: $0 <output_dir> <start_idx> <end_idx> [concurrency]}"
START="${2:?Usage: $0 <output_dir> <start_idx> <end_idx> [concurrency]}"
END="${3:?Usage: $0 <output_dir> <start_idx> <end_idx> [concurrency]}"
CONCURRENCY="${4:-1}"

TOTAL_CPUS="$(nproc)"
CPUS_PER_TASK=$(( TOTAL_CPUS / CONCURRENCY ))
if [ "$CPUS_PER_TASK" -lt 1 ]; then CPUS_PER_TASK=1; fi
export SLURM_CPUS_PER_TASK="$CPUS_PER_TASK"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Running array indices ${START}-${END} on $(hostname)"
echo "  concurrency: ${CONCURRENCY}  cpus/task: ${CPUS_PER_TASK}  (of ${TOTAL_CPUS} total)"
echo "  logs: ${LOG_DIR}/array_NNNN.log"

seq "$START" "$END" | xargs -P "$CONCURRENCY" -I IDX \
    bash -c "python '${SCRIPT_DIR}/run_array_task.py' '${OUTPUT_DIR}' IDX \
             2>&1 | tee '${LOG_DIR}/array_\$(printf \"%04d\" IDX).log'"

echo "Done: indices ${START}-${END} on $(hostname)"
