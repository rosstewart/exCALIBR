"""Task-file runner for the E3 (simulate) and E6 (overlap) grids, for local or SLURM use.

Make a task file (one JSON list of keyword dicts for the module's run_cell):
    python -m analysis.composition_simulation.tasks make <out_dir> [--reps 10] [--reps-e6 30] [--B 50] [--R 4]
Run a slice of it (each array index runs --per-index tasks with as many workers):
    python -m analysis.composition_simulation.tasks run <out_dir> --index 0 --per-index 4
Collect finished tasks:
    python -m analysis.composition_simulation.tasks collect <out_dir>

Each task writes <out_dir>/runs/<task id>.csv; finished tasks are skipped, so a
resubmission only reruns what is missing.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import pandas as pd

# E3: scenario x number of missense P/LP x PTV share of P/LP
E3_SCENARIOS = ["null", "ptv_extreme", "mis_hypomorph", "ptv_escape"]
E3_N_MIS = [3, 8, 20, 60]
E3_PTV_SHARE = [0.25, 0.5, 0.8]
# E6: overlap settings
E6_SETTINGS = ["overlap_same", "overlap_milder"]


def _task_id(t):
    if t["module"] == "simulate":
        return f"e3__{t['scen']}__n{t['n_mis']}__ptv{t['ptv_share']}__rep{t['rep']}"
    return f"e6__{t['setting']}__rep{t['rep']}"


def make(out_dir, reps, reps_e6, B, R, which):
    tasks = []
    if "e3" in which:
        tasks += [dict(module="simulate", scen=s, n_mis=n, ptv_share=p, rep=r, B=B, R=R)
                  for s in E3_SCENARIOS for n in E3_N_MIS for p in E3_PTV_SHARE for r in range(reps)]
    if "e6" in which:
        tasks += [dict(module="overlap", setting=s, rep=r, B=B, R=R) for s in E6_SETTINGS for r in range(reps_e6)]
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks -> {out / 'tasks.json'}")


def _run_one(out_dir, t):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    f = Path(out_dir) / "runs" / f"{_task_id(t)}.csv"
    if f.exists():
        return
    mod = importlib.import_module(f"analysis.composition_simulation.{t['module']}")
    kw = {k: v for k, v in t.items() if k != "module"}
    rows = mod.run_cell(**kw)
    tmp = f.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(f)


def run(out_dir, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(out_dir) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    Parallel(n_jobs=workers or len(mine) or 1)(delayed(_run_one)(out_dir, t) for t in mine)


def collect(out_dir):
    out = Path(out_dir)
    fs = sorted((out / "runs").glob("*.csv"))
    n = len(json.loads((out / "tasks.json").read_text()))
    print(f"{len(fs)} of {n} tasks finished")
    # note: scenario "null" must not be parsed as NaN, hence keep_default_na=False
    for pre, name in [("e3__", "e3_results.csv"), ("e6__", "e6_results.csv")]:
        sel = [f for f in fs if f.name.startswith(pre)]
        if sel:
            pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in sel], ignore_index=True).to_csv(out / name, index=False)
            print(f"  {len(sel)} -> {out / name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--reps", type=int, default=10)
    m.add_argument("--reps-e6", type=int, default=30)
    m.add_argument("--B", type=int, default=50)
    m.add_argument("--R", type=int, default=4)
    m.add_argument("--which", nargs="+", default=["e3", "e6"])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, default=4)
    r.add_argument("--workers", type=int, default=0)
    c = sub.add_parser("collect")
    c.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.reps, a.reps_e6, a.B, a.R, a.which)
    elif a.cmd == "run":
        run(a.out_dir, a.index, a.per_index, a.workers)
    else:
        collect(a.out_dir)


if __name__ == "__main__":
    main()
