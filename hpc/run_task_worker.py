#!/usr/bin/env python3
"""Work-stealing worker: run whichever array tasks of a prepared run are still
unfinished, claiming each with a lock so concurrent workers never share one.

Meant for topping up a run whose fixed-range jobs (submit_array_gpu.sh gives
each GPU job a contiguous block of tasks) are falling behind: start any number
of these alongside them, on GPU or CPU partitions, and they drain the backlog.

    python hpc/run_task_worker.py <output_dir> --device {cpu,gpu}
        [--range-job ARRAY_JOB_ID --tasks-per-job N]

A task counts as finished once every bootstrap it covers has its
bootstrap_<seed>_best_fits.pkl (run_array_task itself also skips completed
components). Claims are directories under <output_dir>/locks/ (mkdir is
atomic); a claim whose owning SLURM job is no longer in the queue is stale and
may be taken over.

--range-job/--tasks-per-job describe the fixed-range array already running
(element r covers tasks r*N .. r*N+N-1, walking forward): workers then take
each range's tasks from its END, and never one at or just ahead of where that
range's job currently is, so the two meet in the middle with at most one task
of overlap. A range whose job has left the queue is treated as unowned.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_array_task import run_array_task  # noqa: E402


def _me():
    a, t = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{a}_{t}" if a and t else os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")


def _in_queue(job_id):
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return True                      # can't tell: assume alive, don't steal
    return bool(out.stdout.strip())


def _conditions(out_dir):
    """dataset -> condition tag (None if untagged), read from the array
    pickles -- job_index.json does not record it, but run_array_task routes
    a tagged dataset's results to <output_dir>/<condition>/<dataset>."""
    import pickle
    cache = out_dir / "locks" / "conditions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    cond = {}
    for f in sorted((out_dir / "jobs").glob("array_*.pkl")):
        with open(f, "rb") as fh:
            for cjob in pickle.load(fh):
                cond.setdefault(cjob["dataset_name"], cjob.get("condition"))
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cond))
    tmp.replace(cache)
    return cond


def _task_done(out_dir, jobs, cond):
    for j in jobs:
        c = cond.get(j["dataset_name"])
        base = out_dir / c if c else out_dir
        if not (base / j["dataset_name"] / f"bootstrap_{j['bootstrap_seed']}_best_fits.pkl").exists():
            return False
    return True


def _range_position(out_dir, range_job, r):
    """(alive, current task) of fixed-range array element r, from its log."""
    if not _in_queue(f"{range_job}_{r}"):
        return False, None
    log = out_dir / "logs" / f"gpu_{range_job}_{r}.out"
    cur = None
    try:
        for m in re.finditer(r"^Array task (\d+):", log.read_text(), re.M):
            cur = int(m.group(1))
    except OSError:
        pass
    return True, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--range-job", default=None)
    ap.add_argument("--tasks-per-job", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the claimable tasks in claim order and exit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    index = json.loads((out_dir / "jobs" / "job_index.json").read_text())
    by_task = defaultdict(list)
    for j in index["job_index"]:
        by_task[j["array_idx"]].append(j)
    n_tasks = index["num_arrays"]
    locks = out_dir / "locks"
    locks.mkdir(exist_ok=True)
    cond = _conditions(out_dir)
    me = _me()

    per = args.tasks_per_job
    if args.range_job and per:
        # distance from the end of its fixed range: each range's last task first
        order = sorted(range(n_tasks), key=lambda t: ((t // per) * per + per - 1 - t, t))
    else:
        order = list(range(n_tasks))[::-1]

    print(f"worker {me}: device={args.device}, {n_tasks} tasks", flush=True)
    if args.dry_run:
        positions = {}
        claimable = []
        for t in order:
            if _task_done(out_dir, by_task[t], cond) or (locks / f"task_{t:04d}").exists():
                continue
            if args.range_job and per:
                r = t // per
                if r not in positions:
                    positions[r] = _range_position(out_dir, args.range_job, r)
                alive, cur = positions[r]
                if alive and (cur is None or t <= cur + 1):
                    continue
            claimable.append(t)
        print(f"{len(claimable)} claimable, in order: {claimable}")
        print("range positions:", {r: p for r, p in sorted(positions.items())})
        return
    n_run = 0
    n_fail = 0
    while True:
        claimed = None
        positions = {}
        for t in order:
            if _task_done(out_dir, by_task[t], cond):
                continue
            if args.range_job and per:
                r = t // per
                if r not in positions:
                    positions[r] = _range_position(out_dir, args.range_job, r)
                alive, cur = positions[r]
                # the range job walks forward from r*per: leave it the task it
                # is on and the one after (cur is None: it hasn't logged yet)
                if alive and (cur is None or t <= cur + 1):
                    continue
            lock = locks / f"task_{t:04d}"
            try:
                lock.mkdir()
            except FileExistsError:
                try:
                    owner = (lock / "owner").read_text().strip()
                except OSError:
                    continue                 # claim being written right now
                if owner == me or _in_queue(owner) or _task_done(out_dir, by_task[t], cond):
                    continue
                shutil.rmtree(lock, ignore_errors=True)   # stale: owner gone
                try:
                    lock.mkdir()
                except FileExistsError:
                    continue
            (lock / "owner").write_text(me + "\n")
            claimed = t
            break
        if claimed is None:
            print(f"worker {me}: nothing left to claim after {n_run} task(s)", flush=True)
            return
        print(f"worker {me}: task {claimed} ({time.strftime('%H:%M:%S')})", flush=True)
        try:
            run_array_task(str(out_dir), claimed, device=args.device)
            n_fail = 0
        except SystemExit:
            raise
        except Exception:
            # Release the claim so a healthy worker can take the task, and
            # give up after two failures in a row: a broken node or device
            # would otherwise churn through (and fail) the whole backlog.
            traceback.print_exc()
            shutil.rmtree(locks / f"task_{claimed:04d}", ignore_errors=True)
            n_fail += 1
            if n_fail >= 2:
                print(f"worker {me}: two consecutive failures, exiting", flush=True)
                sys.exit(1)
        n_run += 1


if __name__ == "__main__":
    main()
