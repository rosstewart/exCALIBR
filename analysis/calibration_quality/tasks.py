"""Task-file runner for the calibration-quality grid, for local or SLURM use (same pattern as
analysis.composition_simulation.tasks).

Make the task file (every design cell x replicate):
    python -m analysis.calibration_quality.tasks make <out_dir> [--reps 30] [--B 100] [--R 4]
Run a slice (array index i runs tasks [i * per_index, (i + 1) * per_index) with --workers processes):
    python -m analysis.calibration_quality.tasks run <out_dir> --index 0 --per-index 256 --workers 16
Collect finished tasks into <out_dir>/results.csv:
    python -m analysis.calibration_quality.tasks collect <out_dir>

Each task writes <out_dir>/runs/<cell id>__rep<r>.csv (one row); finished tasks are skipped, so a resubmission only
reruns what is missing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _task_id(t):
    return f"{t['cell']['cell_id']}__rep{t['rep']}"


def make(out_dir, reps, B, R):
    from analysis.calibration_quality.generate import design
    tasks = [dict(cell=c, rep=r, B=B, R=R) for r in range(reps) for c in design()]  # rep-major: early reps first
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks ({len(tasks) // reps} cells x {reps} reps) -> {out / 'tasks.json'}")


def _run_one(out_dir, t):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    f = Path(out_dir) / "runs" / f"{_task_id(t)}.csv"
    if f.exists():
        return
    from analysis.calibration_quality.run import run_cell
    row = run_cell(t["cell"], t["rep"], t["B"], t["R"])
    tmp = f.with_suffix(".tmp")
    pd.DataFrame([row]).to_csv(tmp, index=False)
    tmp.rename(f)


def run(out_dir, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(out_dir) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    Parallel(n_jobs=workers or 1, verbose=5)(delayed(_run_one)(out_dir, t) for t in mine)


def collect(out_dir):
    out = Path(out_dir)
    fs = sorted((out / "runs").glob("*.csv"))
    n = len(json.loads((out / "tasks.json").read_text()))
    print(f"{len(fs)} of {n} tasks finished")
    if fs:
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        df.to_csv(out / "results.csv", index=False)
        ok = (df["status"] == "ok").sum()
        print(f"  {len(df)} rows ({ok} ok) -> {out / 'results.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--reps", type=int, default=30)
    m.add_argument("--B", type=int, default=100)
    m.add_argument("--R", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, required=True)
    r.add_argument("--workers", type=int, default=16)
    c = sub.add_parser("collect")
    c.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.reps, a.B, a.R)
    elif a.cmd == "run":
        run(a.out_dir, a.index, a.per_index, a.workers)
    else:
        collect(a.out_dir)


if __name__ == "__main__":
    main()
