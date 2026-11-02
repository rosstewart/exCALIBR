#!/usr/bin/env python3
"""Bundle the semi-synthetic classifier calibration simulation for the cluster.

The simulation (mv_analysis/classifier_calibration_sim.py, stage "run": one
replicate world per paper dataset, every classifier under raw, ExCALIBR and
local calibration) needs only each dataset's P-KNN prep (prep/<key>.npz,
.json) and its out-of-bag fits (production_oob/<key>/mv.npz, whose bootstrap
log LR+ define the true likelihood ratio). This copies them into <bundle_dir>
with the same layout, so MV_PKNN_DIR=<bundle_dir> on the cluster resolves
every path, and writes <bundle_dir>/sim_tasks.json, the [key, replicate] list
the array tasks index into.

Usage (here):
    python hpc/prepare_classifier_sim.py <bundle_dir> [--reps 20] [--keys KEY ...]
then copy <bundle_dir> to the cluster, run hpc/submit_classifier_sim.sh there,
copy <bundle_dir>/classifier_calibration_sim/runs/ back into
$PAPER_PRODUCTION_OOB_DIR/classifier_calibration_sim/runs/ here, and summarize:
    python -m mv_analysis.classifier_calibration_sim summarize
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mv_analysis import config  # noqa: E402
from mv_analysis.classifier_calibration_sim import tasks  # noqa: E402
from mv_analysis.production_oob import OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()
    b = Path(args.bundle_dir)
    (b / "prep").mkdir(parents=True, exist_ok=True)
    t = tasks(args.keys, args.reps)  # replicates not yet run here
    for key in sorted({k for k, _ in t}):
        for ext in ("npz", "json"):
            shutil.copy2(Path(config.PKNN_DIR) / "prep" / f"{key}.{ext}", b / "prep" / f"{key}.{ext}")
        (b / "production_oob" / key).mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / key / "mv.npz", b / "production_oob" / key / "mv.npz")
    (b / "sim_tasks.json").write_text(json.dumps(t))
    print(f"{len(t)} tasks over {len({k for k, _ in t})} datasets -> {b}")


if __name__ == "__main__":
    main()
