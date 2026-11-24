#!/bin/bash
# Chained GPU worker: drain several prepared runs in priority order on one GPU,
# so a GPU this job holds is never released while work remains. Each stage is a
# run_task_worker.py (lock-based claiming, so any number of these -- and the
# per-run workers already running -- can share a run without duplicating tasks).
cd /home/stewart_ro_neu/exCALIBR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
P=/home/stewart_ro_neu/.conda/envs/excalibr/bin/python
W="$P hpc/run_task_worker.py"
nvidia-smi -L
stage() { echo "=== stage $1 ($(date +%H:%M:%S))"; shift; "$@"; echo "=== stage done, exit $? ($(date +%H:%M:%S))"; }
stage tsc2_uv $W /home/stewart_ro_neu/exCALIBR/jobs/jobs_tsc2_uv --device gpu --gpu-init cpu
stage repair  $W /home/stewart_ro_neu/exCALIBR/jobs/jobs_floorfull_prioritized_100b_102626 --device gpu --gpu-init cpu \
              --tasks /home/stewart_ro_neu/exCALIBR/jobs/jobs_floorfull_prioritized_100b_102626/repair_remaining.txt --require-valid
stage pilot   $W /home/stewart_ro_neu/exCALIBR/jobs/jobs_predictor_transform_pilot_20b_102726 --device gpu --gpu-init cpu
stage rerun   $W /home/stewart_ro_neu/exCALIBR/jobs/jobs_floorfull_prioritized_100b_102626_cpuinit --device gpu --gpu-init cpu \
              --first KCNH2_mv,TP53_tp53_mv,TP53_combined_mv,F9_combined_mv,BRCA2_combined_mv,PALB2_mv
echo "chain done"
