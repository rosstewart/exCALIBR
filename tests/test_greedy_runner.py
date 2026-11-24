"""hpc/run_array_task.py greedy-restart bookkeeping, CPU and GPU drivers.

The fit itself is replaced by a deterministic fake (val_ll a hash of the
candidate's lambdaIndex/fit_seed/K/mode), so these test only the runner:
every bootstrap is saved (including K=4, q=2, whose slot 7 -- and slot 6 once
the current pattern is >= 36 -- has no in-range pattern), the restart count
per (bootstrap, K) is what the search actually ran, and the GPU driver's
round-by-round search makes exactly the CPU frontier's candidates and choices.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc"))
import run_array_task as rat  # noqa: E402
from src.assay_calibration.fit_utils.fit import next_greedy_lambda  # noqa: E402


def _fake_val_ll(job):
    kw = job["kwargs"]
    h = hash((kw["lambdaIndex"], kw["fit_seed"], job["num_components"],
              kw["latent_q"], kw["force_gaussian"])) % 10_000
    if h % 97 == 0:
        return None                      # an occasional failed fit
    return {"val_ll": -float(h), "lambdaIndex": kw["lambdaIndex"]}


def _fake_execute(job):
    return _fake_val_ll(job)


def _fake_run_gpu(specs, use_gpu_init=True, max_em_iters=None):
    assert use_gpu_init is False
    out = [(bs, lbl, sd, job["fit_idx"], _fake_val_ll(job)) for job, bs, lbl, sd in specs]
    return out[::-1]                      # the GPU path reorders results


def _seed_jobs(K, bs):
    jobs = []
    for mode, q, fg, fidx in (("gaussian", 2, True, 0), ("q1", 1, False, 1), ("q2", 2, False, 2)):
        n_patterns = min((2 ** q) ** K, 100)
        n_slots = 0 if mode == "gaussian" else min(q * K, n_patterns - 1)
        jobs.append(dict(
            fit_idx=fidx, num_components=K, constrained=False, init_method="kmeans",
            init_constraint_adjustment=0,
            kwargs=dict(lambdaIndex=0, latent_q=q, force_gaussian=fg,
                        fit_seed=1000 * bs + 10 * K + fidx, greedy_restarts=True),
            greedy_meta=dict(mode=mode, n_patterns=n_patterns, n_slots=n_slots)))
    return jobs


def _make_run(root, n_boot=4):
    (root / "jobs").mkdir(parents=True)
    arr = [{"dataset_name": "D_mv", "bootstrap_seed": bs, "shared_data": {},
            "multivariate": True, "jobs_3c": _seed_jobs(3, bs), "jobs_4c": _seed_jobs(4, bs)}
           for bs in range(n_boot)]
    with open(root / "jobs" / "array_0000.pkl", "wb") as f:
        pickle.dump(arr, f)
    return root


def _expected_ll(bs, K):
    """Reference greedy search (run_greedy's loop) on the fake val_ll."""
    out = []
    for mode, q, fg, fidx in (("gaussian", 2, True, 0), ("q1", 1, False, 1), ("q2", 2, False, 2)):
        base = _seed_jobs(K, bs)[fidx]
        job = dict(base, kwargs=dict(base["kwargs"]))
        r = _fake_val_ll(job)
        out.append((fidx, r["val_ll"] if r else None))
    nxt = 3
    for mode, q, fidx in (("q1", 1, 1), ("q2", 2, 2)):
        base = _seed_jobs(K, bs)[fidx]
        gm = base["greedy_meta"]
        r = _fake_val_ll(base)
        best, cur = (r["val_ll"] if r else -np.inf), 0
        start = nxt
        for slot in range(gm["n_slots"]):
            cand = next_greedy_lambda(cur, slot, gm["n_patterns"])
            if cand is None:
                continue
            fi = nxt
            nxt += 1
            kw = dict(base["kwargs"], lambdaIndex=cand,
                      fit_seed=(base["kwargs"]["fit_seed"] + fi * 2654435761) % (2**31 - 1))
            r = _fake_val_ll(dict(base, kwargs=kw))
            v = r["val_ll"] if r else -np.inf
            out.append((fi, r["val_ll"] if r else None))
            if v > best:
                best, cur = v, cand
        nxt = start + gm["n_slots"]
    return sorted(out)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_greedy_saves_every_bootstrap(tmp_path, monkeypatch, device):
    monkeypatch.setenv("EXCALIBR_SAVE_RESTART_LL", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setattr(rat.Fit, "execute_fit_job", staticmethod(_fake_execute))
    import src.assay_calibration.fit_utils.jax_batch.interop as interop
    monkeypatch.setattr(interop, "run_gpu", _fake_run_gpu, raising=False)
    root = _make_run(tmp_path / device)
    rat.run_array_task(str(root), 0, device=device, gpu_init="cpu")
    for bs in range(4):
        d = root / "D_mv"
        best = pickle.load(open(d / f"bootstrap_{bs}_best_fits.pkl", "rb"))
        assert set(best) == {"3c", "4c"}
        lls = json.loads((d / f"bootstrap_{bs}_restart_val_ll.json").read_text())
        for K in (3, 4):
            got = sorted((e["fit_idx"], e["val_ll"]) for e in lls if e["label"] == f"{K}c")
            assert got == _expected_ll(bs, K)


def test_greedy_gpu_rejects_gpu_init(tmp_path, monkeypatch):
    root = _make_run(tmp_path / "g")
    with pytest.raises(ValueError):
        rat.run_array_task(str(root), 0, device="gpu", gpu_init="gpu")
