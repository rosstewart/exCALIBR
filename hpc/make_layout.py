#!/usr/bin/env python
"""Lay a prepared run's compact jobs (dataset x bootstrap) out into SLURM
array tasks for a given machine, at submission time.

    python hpc/make_layout.py <output_dir> --tasks N|auto --cpus C
        [--max-concurrent M] [--out PATH] [--quiet]

--tasks N     exactly N tasks (capped at the number of compact jobs), work
              balanced across them with hpc/packing.py's fit-time estimates.
--tasks auto  the task count minimising the predicted whole-array wall time
              for M concurrent tasks of C CPUs (needs --max-concurrent).
--cpus C      CPUs per task (the SLURM --cpus-per-task you will submit with).

Writes <output_dir>/jobs/layout_<N>t_<C>c.json (or --out): for each task, the
(array_idx, local_idx) storage locations of its compact jobs, longest task
first; pass it to run_array_task.py --layout (submit_array.sh does this when
TASKS is set). Prints the path on the last line of stdout. Reads
jobs/costs.json (written by prepare.py); for older runs without it the job
pickles are read once and costs.json is written.

Scheduling only: which task runs a compact job cannot change its fits
(CPU path; bitwise).
"""
import argparse
import glob
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from hpc import packing as P  # noqa: E402


def load_costs(jobs_dir):
    path = os.path.join(jobs_dir, "costs.json")
    if os.path.exists(path):
        with open(path) as f:
            jobs = json.load(f)["jobs"]
    else:
        with open(os.path.join(jobs_dir, "job_index.json")) as f:
            index = json.load(f)
        by_arr = {}
        for e in index["job_index"]:
            by_arr.setdefault(e["array_idx"], []).append(e)
        jobs = []
        for a in sorted(by_arr):
            with open(os.path.join(jobs_dir, f"array_{a:04d}.pkl"), "rb") as f:
                arr = pickle.load(f)
            for e in sorted(by_arr[a], key=lambda e: e["local_idx"]):
                jobs.append({"global_idx": e["global_idx"], "array_idx": a, "local_idx": e["local_idx"],
                             "fit_costs": [round(c, 3) for c in P.cjob_fit_costs(arr[e["local_idx"]])]})
        jobs.sort(key=lambda j: j["global_idx"])
        with open(path, "w") as f:
            json.dump({"model": "hpc/packing.py v1", "jobs": jobs}, f)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--tasks", required=True, help="number of array tasks, or 'auto'")
    ap.add_argument("--cpus", type=int, required=True, help="CPUs per task")
    ap.add_argument("--max-concurrent", type=int, default=None,
                    help="concurrent tasks (needed for --tasks auto; used for the prediction)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    jobs_dir = os.path.join(os.path.abspath(args.output_dir), "jobs")
    jobs = load_costs(jobs_dir)
    costs = [j["fit_costs"] for j in jobs]
    if args.tasks == "auto":
        if not args.max_concurrent:
            sys.exit("--tasks auto needs --max-concurrent")
        bins, _ = P.choose_tasks(costs, args.cpus, args.max_concurrent)
    else:
        bins = P.assign(costs, int(args.tasks), args.cpus)
    st = P.evaluate(bins, costs, args.cpus, args.max_concurrent)
    best_c, table = P.suggest_cpus(costs, bins, options=tuple(sorted({1, 2, 4, 8, 16, 32, args.cpus})))
    out = args.out or os.path.join(jobs_dir, f"layout_{len(bins)}t_{args.cpus}c.json")
    layout = {"n_tasks": len(bins), "cpus_per_task": args.cpus, "max_concurrent": args.max_concurrent,
              "requested_tasks": args.tasks, "cost_model": "hpc/packing.py v1",
              "predicted": {"cpu_use_within_tasks": round(st["utilisation"], 4),
                            "longest_task_rel_hours": round(st["max_task"] / 3600, 2),
                            "array_rel_hours": round(st["run_wall"] / 3600, 2) if st["run_wall"] else None},
              "tasks": [[[jobs[i]["array_idx"], jobs[i]["local_idx"]] for i in b] for b in bins]}
    with open(out, "w") as f:
        json.dump(layout, f)
    if not args.quiet:
        sizes = [len(b) for b in bins]
        print(f"Layout: {len(bins)} tasks x {args.cpus} CPUs ({min(sizes)}-{max(sizes)} compact jobs each)",
              file=sys.stderr)
        msg = (f"  predicted CPU use within tasks {st['utilisation']:.0%}, longest task "
               f"{st['max_task'] / 3600:.1f} h")
        if st["run_wall"]:
            msg += f", whole array {st['run_wall'] / 3600:.1f} h at {args.max_concurrent} concurrent"
        print(msg + " (hours on a local core: compare, don't schedule by them)", file=sys.stderr)
        # The same layout at other CPU counts: CPU use within tasks falls when
        # a task has fewer fits than CPUs, so many small tasks want few CPUs.
        row = []
        for c, t in sorted(table.items()):
            if c in (1, 2, 4, 8, 16, 32, args.cpus):
                row.append(f"{c}: {t['utilisation']:.0%} / {t['max_task'] / 3600:.1f} h")
        print("  same tasks at SLURM_CPUS = (CPU use / longest task)  " + ";  ".join(row), file=sys.stderr)
        if st["utilisation"] < 0.8:
            good = max([c for c, t in table.items() if t["utilisation"] >= 0.9] or [1])
            print(f"  warning: {1 - st['utilisation']:.0%} of the reserved cores would sit idle; "
                  f"use fewer CPUs per task (SLURM_CPUS={good} keeps >= 90% busy) or fewer tasks",
                  file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
