"""hpc/packing.py and hpc/prepare.py array layouts (scheduling only)."""
import json
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hpc"))

from hpc import packing as P  # noqa: E402


def _fit(K, mode, greedy_slots=None):
    kw = {"force_gaussian": mode == "gaussian", "latent_q": 1 if mode == "q1" else 2}
    j = {"num_components": K, "kwargs": kw}
    if greedy_slots is not None:
        j["greedy_meta"] = {"mode": mode, "n_slots": greedy_slots, "n_patterns": 4}
    return j


def _cjob(name, seed, N, p, Ks=(2, 3, 4)):
    cj = {"dataset_name": name, "bootstrap_seed": seed, "num_fits_total": 3 * len(Ks),
          "multivariate": True, "shared_data": {"train_observations": np.zeros((N, p))}}
    for K in Ks:
        cj[f"jobs_{K}c_unc"] = [_fit(K, m) for m in ("gaussian", "q1", "q2")]
    return cj


def _bundle():
    jobs = []
    for name, N, p in (("big", 5000, 6), ("mid", 1500, 3), ("small", 700, 10)):
        for s in range(12):
            jobs.append(_cjob(name, s, N, p))
    return jobs


def test_fit_cost_ordering():
    c = lambda j: P.fit_cost(j, 1000, 6)  # noqa: E731
    assert c(_fit(4, "q2")) > c(_fit(4, "gaussian")) > 0
    assert c(_fit(4, "q2")) > c(_fit(4, "q1"))
    assert c(_fit(6, "q2")) > c(_fit(3, "q2"))
    assert P.fit_cost(_fit(4, "q2"), 5000, 6) > P.fit_cost(_fit(4, "q2"), 500, 6)
    assert c(_fit(4, "q2", greedy_slots=3)) == pytest.approx(4 * c(_fit(4, "q2")))


def test_assign_covers_every_job_once_and_balances():
    jobs = _bundle()
    costs = [P.cjob_fit_costs(c) for c in jobs]
    for n in (1, 5, 7, len(jobs)):
        bins = P.assign(costs, n)
        flat = sorted(i for b in bins for i in b)
        assert flat == list(range(len(jobs)))
        walls = [P.makespan([x for i in b for x in costs[i]], 16) for b in bins]
        assert walls == sorted(walls, reverse=True)  # longest task first
    leg = P.evaluate(P.contiguous(len(jobs), 6), costs, 16)
    bal = P.evaluate(P.assign(costs, 6), costs, 16)
    assert bal["max_task"] <= leg["max_task"]


def test_choose_tasks_not_worse_than_one_per_task():
    jobs = _bundle()
    costs = [P.cjob_fit_costs(c) for c in jobs]
    bins, st = P.choose_tasks(costs, cpus=16, slots=4)
    assert sorted(i for b in bins for i in b) == list(range(len(jobs)))
    one = P.evaluate([[i] for i in range(len(jobs))], costs, 16, 4)
    assert st["run_wall"] <= one["run_wall"] * 1.02


@pytest.mark.parametrize("packing", ["contiguous", "balanced"])
def test_save_manifest_layouts(tmp_path, packing):
    import prepare
    jobs = _bundle()
    old = dict(prepare._PACK_OPTS)
    prepare._PACK_OPTS.update(packing=packing)
    try:
        total, n_arr = prepare._save_manifest(str(tmp_path), jobs, 5, 12)
    finally:
        prepare._PACK_OPTS.clear()
        prepare._PACK_OPTS.update(old)
    index = json.load(open(tmp_path / "jobs" / "job_index.json"))
    assert total == len(jobs) and index["num_arrays"] == n_arr
    # integer target keeps the legacy task count
    jpa = -(-len(jobs) // 5)
    assert n_arr == -(-len(jobs) // jpa)
    seen = []
    for a in range(n_arr):
        with open(tmp_path / "jobs" / f"array_{a:04d}.pkl", "rb") as f:
            arr = pickle.load(f)
        ent = [e for e in index["job_index"] if e["array_idx"] == a]
        assert len(ent) == len(arr)
        for e, cj in zip(ent, arr):
            # global_idx is the position in all_jobs under every layout
            assert jobs[e["global_idx"]]["dataset_name"] == cj["dataset_name"]
            assert jobs[e["global_idx"]]["bootstrap_seed"] == cj["bootstrap_seed"]
            seen.append(e["global_idx"])
    assert sorted(seen) == list(range(len(jobs)))
    if packing == "contiguous":
        # legacy layout: consecutive slices, global_idx = start + local_idx
        assert all(e["global_idx"] == e["array_idx"] * jpa + e["local_idx"] for e in index["job_index"])
    costs = json.load(open(tmp_path / "jobs" / "costs.json"))["jobs"]
    assert [c["global_idx"] for c in costs] == list(range(len(jobs)))


@pytest.mark.parametrize("tasks", ["7", "auto", "1000"])
def test_submission_layout_roundtrip(tmp_path, tasks):
    """make_layout.py lays the stored compact jobs out for the machine;
    run_array_task's --layout loader returns exactly those jobs, each once."""
    import subprocess
    import prepare
    import run_array_task as rat
    jobs = _bundle()
    prepare._save_manifest(str(tmp_path), jobs, 5, 12)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "hpc", "make_layout.py"), str(tmp_path),
                          "--tasks", tasks, "--cpus", "4", "--max-concurrent", "3"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    layout = json.load(open(out))
    expect = len(jobs) if tasks == "1000" else (7 if tasks == "7" else None)
    if expect is not None:
        assert layout["n_tasks"] == expect
    got = []
    for t in range(layout["n_tasks"]):
        for cj in rat._load_layout_task(tmp_path / "jobs", out, t):
            got.append((cj["dataset_name"], cj["bootstrap_seed"]))
    assert sorted(got) == sorted((c["dataset_name"], c["bootstrap_seed"]) for c in jobs)
