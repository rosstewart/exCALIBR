"""HPC task runner for membership-cell fits (C1 labeled scale, C2 assay scale).

Make the bundle here (builds the cell matrices, so cluster tasks need no dataframe):
    python -m analysis.membership_cells.tasks make <bundle> [--B 100] [--R 8] [--block 5] [--scales labeled assay]
Run one array index on the cluster (each index runs --per-index tasks with as many workers):
    python -m analysis.membership_cells.tasks run <bundle> --index 0 --per-index 16
After copying <bundle>/<scale>/raw/ back here, compose (needs the dataframe) and calibrate locally:
    python -m analysis.membership_cells.run_light --compose-only --scale <scale> --out <bundle>/<scale> --datasets ...

Each task = (scale, dataset, K, block of bootstrap seeds) and writes <bundle>/<scale>/raw/<ds>__<K>c__<a>-<b>.json.gz;
finished tasks are skipped, so a resubmission only reruns what is missing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def make(bundle, B, R, block, scales, datasets=None, config="src/igvf_configs/dataset_configs_aug_2026.json", n_jobs=30):
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.membership_cells.run_light import build, _cfg_nc
    bundle = Path(bundle)
    cfg = json.load(open(config))
    import gzip
    fits = json.load(gzip.open("/data/ross/assay_calibration/missense_only_calibrations/explorer_jobs_pp_large_baseline/"
                               "bootstrap_results_first100.json.gz"))
    dss = sorted(datasets or fits)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in dss}
    del df_full
    tasks = []
    for scale in scales:
        out = bundle / scale
        Parallel(n_jobs=n_jobs)(delayed(build)(ds, slices[ds], scale, out) for ds in dss)
        for ds in dss:
            nc = _cfg_nc(cfg, ds)
            if nc is None or not (out / "cells" / f"{ds}.npz").exists():
                continue
            for K in sorted({min(nc, 3), 3}):
                for s0 in range(0, B, block):
                    tasks.append(dict(scale=scale, ds=ds, K=K, seeds=list(range(s0, min(s0 + block, B))), R=R))
    (bundle / "tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks over {len(dss)} datasets x {scales} -> {bundle / 'tasks.json'}")


def _run_one(bundle, t):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from analysis.membership_cells.run_light import task
    return task(t["ds"], t["K"], t["seeds"], t["R"], Path(bundle) / t["scale"])


def run(bundle, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(bundle) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    res = Parallel(n_jobs=workers or len(mine) or 1)(delayed(_run_one)(bundle, t) for t in mine)
    for r in res:
        if str(r).startswith("FAILED"):
            print(r)


def status(bundle):
    bundle = Path(bundle)
    tasks = json.loads((bundle / "tasks.json").read_text())
    done = sum((bundle / t["scale"] / "raw" / f"{t['ds']}__{t['K']}c__{t['seeds'][0]}-{t['seeds'][-1]}.json.gz").exists()
               for t in tasks)
    print(f"{done} of {len(tasks)} tasks finished")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("bundle")
    m.add_argument("--B", type=int, default=100)
    m.add_argument("--R", type=int, default=8)
    m.add_argument("--block", type=int, default=5)
    m.add_argument("--scales", nargs="+", default=["labeled", "assay"])
    m.add_argument("--datasets", nargs="*", default=None)
    r = sub.add_parser("run")
    r.add_argument("bundle")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, default=16)
    r.add_argument("--workers", type=int, default=0)
    s = sub.add_parser("status")
    s.add_argument("bundle")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.bundle, a.B, a.R, a.block, a.scales, a.datasets)
    elif a.cmd == "run":
        run(a.bundle, a.index, a.per_index, a.workers)
    else:
        status(a.bundle)


if __name__ == "__main__":
    main()
