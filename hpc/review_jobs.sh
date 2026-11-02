#!/bin/bash
# Cluster jobs for the review analyses (bootstrap / restart budget), in order.
# Run each step on the cluster from the repo root; nothing here runs locally.
#
#   bash hpc/review_jobs.sh production      <out_root> [n_bootstraps=200] [restarts=greedy]
#   (data not on the cluster: build locally, copy, submit there)
#   bash hpc/review_jobs.sh build  <job_dir> [n_bootstraps=200] [restarts=greedy]   # local
#   bash hpc/review_jobs.sh submit <job_dir>                                       # cluster
#   (optional, only if the production run's checks say more is needed)
#   bash hpc/review_jobs.sh restart-pilot   <out_root>
#   bash hpc/review_jobs.sh bootstrap-pilot <out_root> <restarts: greedy|N>
#
# restart-pilot   50 bootstraps of representative models (TP53 functional,
#                 BRCA1/GCK/F9 functional, BRCA1/TP53 predictors, BRCA1/TP53
#                 combined), 3 and 4 components, with the full greedy
#                 sign-pattern restart set (1 Gaussian + 1+K MSN + 1+2K CFUSN);
#                 every restart's held-out log-likelihood is kept
#                 (EXCALIBR_SAVE_RESTART_LL=1 ->
#                 bootstrap_<seed>_restart_val_ll.json). Analyse with
#                 python -m mv_analysis.review_budget restarts <run_dir>
#                 to choose the minimum number of restarts.
# bootstrap-pilot the same models with 1000 bootstraps at that restart budget;
#                 python -m mv_analysis.review_budget bootstraps <final_dir>
#                 subsamples them to 100..1000 to choose the bootstrap count.
# production      every paper model (functional, predictor, combined; all
#                 assay groups) at the chosen budget, 3 and 4 components (so
#                 either component rule can be applied), pathogenic share 0.1
#                 as the paper's fits. Merge into its own final dir with
#                 mv_analysis/merge_condition_results.py and point
#                 MV_PAPER_FITS_DIR at it for an alternative result set; never
#                 overwrite the paper's final dir.
#
# Submission uses hpc/submit_array.sh (SLURM settings via its environment
# variables, e.g. SLURM_CPUS, SLURM_TIME, MAX_CONCURRENT); sbatch inherits the
# environment, which is how EXCALIBR_SAVE_RESTART_LL reaches the workers.
set -euo pipefail
STEP=${1:?step}
OUT=${2:?out_root}
DF=${DATAFRAME:-/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz}
PRED=${PREDICTOR_DATA_DIR:-/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data}
PILOT_GENES="TP53 BRCA1 GCK F9"
COMMON=(--components 3 4 --pathogenic-share 0.1 --dataframe "$DF" --predictor-data-dir "$PRED")

restarts_args() {  # greedy -> --greedy-restarts; N -> --num-fits N
    if [ "$1" = "greedy" ]; then echo "--greedy-restarts"; else echo "--num-fits $1"; fi
}

case "$STEP" in
  restart-pilot)
    python hpc/prepare.py mv_all --output-dir "$OUT/review_restart_pilot_50b" \
        --pipelines tp53 integrated predictor-mv combined --genes $PILOT_GENES \
        --n-bootstraps 50 --greedy-restarts "${COMMON[@]}"
    EXCALIBR_SAVE_RESTART_LL=1 SLURM_TIME=${SLURM_TIME:-23:59:00} \
        bash hpc/submit_array.sh "$OUT/review_restart_pilot_50b"
    ;;
  bootstrap-pilot)
    R=${3:?restarts}
    python hpc/prepare.py mv_all --output-dir "$OUT/review_bootstrap_pilot_1000b_r$R" \
        --pipelines tp53 integrated predictor-mv combined --genes $PILOT_GENES \
        --n-bootstraps 1000 $(restarts_args "$R") "${COMMON[@]}"
    bash hpc/submit_array.sh "$OUT/review_bootstrap_pilot_1000b_r$R"
    ;;
  production)
    # default: greedy restarts + 200 bootstraps. Every restart's held-out
    # log-likelihood is kept, so restart saturation is checked on this run
    # itself (review_budget restarts), and the bootstraps are subsampled to
    # check they suffice (review_budget bootstraps; more can be appended later
    # without refitting these).
    B=${3:-200}; R=${4:-greedy}
    python hpc/prepare.py mv_all --output-dir "$OUT/review_production_${B}b_r$R" \
        --pipelines tp53 integrated tsc2 fgfr predictor-mv combined \
        --n-bootstraps "$B" $(restarts_args "$R") "${COMMON[@]}"
    EXCALIBR_SAVE_RESTART_LL=1 bash hpc/submit_array.sh "$OUT/review_production_${B}b_r$R"
    ;;
  build)
    # where the input data live (no cluster access to them): only prepare the
    # job dir. The job pickles carry their data and save paths are re-derived
    # from the dir at run time, so copy <job_dir> to the cluster and run
    #   bash hpc/review_jobs.sh submit <job_dir_on_cluster>
    B=${3:-200}; R=${4:-greedy}
    python hpc/prepare.py mv_all --output-dir "$OUT" \
        --pipelines tp53 integrated tsc2 fgfr predictor-mv combined \
        --n-bootstraps "$B" $(restarts_args "$R") "${COMMON[@]}"
    ;;
  submit)
    EXCALIBR_SAVE_RESTART_LL=1 bash hpc/submit_array.sh "$OUT"
    ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
