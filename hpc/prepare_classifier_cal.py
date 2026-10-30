#!/usr/bin/env python3
"""Bundle the inputs of the cross-fitted classifier calibration for the cluster.

The calibration (mv_analysis/production_oob.py, stage "excalcv": one
supervised.fit_excalibr_crossfit per model, paper dataset and out-of-bag
split) needs only each dataset's P-KNN prep (prep/<key>.npz, .json) and its
out-of-bag fits (production_oob/<key>/mv.npz). This copies them into
<bundle_dir> with the same layout, so MV_PKNN_DIR=<bundle_dir> on the cluster
resolves every path, and writes <bundle_dir>/tasks.json, the [key, split]
list the array tasks index into.

Usage (here):
    python hpc/prepare_classifier_cal.py <bundle_dir> [--all] [--keys KEY ...]
then copy <bundle_dir> to the cluster, run hpc/submit_classifier_cal.sh there,
copy <bundle_dir>/production_oob/*/split*_excalcv.npz back into
$MV_PKNN_DIR/production_oob/ here, and collect:
    python mv_analysis/analyze_mv_pipeline_output.py --sections production-oob

--all lists every split (default: only splits without a split<i>_excalcv.npz
here), e.g. after a change to the calibration.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mv_analysis import config  # noqa: E402
from mv_analysis.production_oob import OUT, keys_with_both_classes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir")
    ap.add_argument("--all", action="store_true", help="list every split, not only those missing here")
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()
    b = Path(args.bundle_dir)
    (b / "prep").mkdir(parents=True, exist_ok=True)
    tasks = []
    for key in keys_with_both_classes(args.keys):
        mv = OUT / key / "mv.npz"
        if not mv.exists():
            print(f"  skip {key}: no out-of-bag fits (production_oob.py mv)")
            continue
        for ext in ("npz", "json"):
            shutil.copy2(Path(config.PKNN_DIR) / "prep" / f"{key}.{ext}", b / "prep" / f"{key}.{ext}")
        (b / "production_oob" / key).mkdir(parents=True, exist_ok=True)
        shutil.copy2(mv, b / "production_oob" / key / "mv.npz")
        n = np.load(mv)["oob"].shape[0]
        tasks += [[key, i] for i in range(n) if args.all or not (OUT / key / f"split{i}_excalcv.npz").exists()]
    (b / "tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks over {len({k for k, _ in tasks})} datasets -> {b}")


if __name__ == "__main__":
    main()
