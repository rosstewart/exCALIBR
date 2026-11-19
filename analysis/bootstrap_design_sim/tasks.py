"""Task-file runner for the bootstrap-design simulation (same pattern as
analysis.calibration_quality.tasks).

    python -m analysis.bootstrap_design_sim.tasks make <out_dir> [--reps 10] [--B 30] [--R 1]
    python -m analysis.bootstrap_design_sim.tasks run <out_dir> --index 0 --per-index 4 --workers 16
    python -m analysis.bootstrap_design_sim.tasks collect <out_dir>

Each task (one design cell x replicate, every scheme on the same data) writes
<out_dir>/runs/<cell id>__rep<r>.csv; finished tasks are skipped. Submit on the
cluster with hpc/submit_calibration_quality.sh's pattern (SIM_MODULE below).
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
    from analysis.bootstrap_design_sim.generate import design
    tasks = [dict(cell=c, rep=r, B=B, R=R) for r in range(reps) for c in design()]
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
    from analysis.bootstrap_design_sim.run import run_cell
    rows = run_cell(t["cell"], t["rep"], t["B"], t["R"])
    tmp = f.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
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
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    df.to_csv(out / "results.csv", index=False)
    print(f"{len(fs)} of {n} tasks finished -> {out / 'results.csv'}")
    if len(df):
        cols = ["oracle_recall", "oracle_recall_rare", "wrong_direction", "overstatement", "determinate", "mcc",
                "lr_mae", "spread"]
        print(df.groupby("scheme")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("out_dir"); m.add_argument("--reps", type=int, default=10)
    m.add_argument("--B", type=int, default=30); m.add_argument("--R", type=int, default=1)
    r = sp.add_parser("run"); r.add_argument("out_dir"); r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, required=True); r.add_argument("--workers", type=int, default=1)
    c = sp.add_parser("collect"); c.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.reps, a.B, a.R)
    elif a.cmd == "run":
        run(a.out_dir, a.index, a.per_index, a.workers)
    else:
        collect(a.out_dir)
