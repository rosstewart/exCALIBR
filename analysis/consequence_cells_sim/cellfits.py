"""Cells-only refits that keep the per-cell weights, for the abnormal-profile overlap test (experimental, 2026-11-15).

The grid runs (run.py) do not store cell weights. This refits only the cell model for the K >= 3 cells (PTV-location
cells and the hypomorph slice) on the same regenerated data, with the same fitter settings and bootstrap seeds, and
saves one npz per task: weights (B, n_cells, K), component params (B, K, 3), cell names and sizes.

    python -m analysis.consequence_cells_sim.cellfits make <out_dir> [--reps 20] [--B 50] [--R 8]
    python -m analysis.consequence_cells_sim.cellfits run <out_dir> --index 0 --per-index 16 --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def fit_k(c):
    from analysis.consequence_cells_sim.generate import n_components
    return int(c.get("fit_K") or n_components(c))


def _cells(experiment=None):
    from analysis.consequence_cells_sim.generate import design
    if experiment:
        return [c for c in design() if c["experiment"] == experiment]
    return [c for c in design() if fit_k(c) >= 3 and c["experiment"] != "ptvtv"]   # the 2026-11-15 set


def make(out_dir, reps, B, R, experiment=None):
    tasks = [dict(cell=c, rep=r, B=B, R=R) for r in range(reps) for c in _cells(experiment)]
    out = Path(out_dir)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks ({len(tasks) // reps} cells x {reps} reps) -> {out / 'tasks.json'}")


def _run_one(out_dir, t):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    c, rep = t["cell"], t["rep"]
    f = Path(out_dir) / "fits" / f"{c['cell_id']}__rep{rep}.npz"
    if f.exists():
        return
    from analysis.consequence_cells_sim import generate as G
    from analysis.consequence_cells_sim.run import cell_keys, fit_arm
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    d = G.make_data(G.rng_for(c, rep), c)
    sc = d["score"]
    names, cell_of_row = cell_keys(d, c["assay"])
    onehot = np.zeros((len(sc), len(names)), bool)
    onehot[np.arange(len(sc)), cell_of_row] = True
    K = fit_k(c)
    fits, seeds = fit_arm(BasicScoreset(sc, onehot, sample_names=names), K, t["B"], t["R"],
                          c.get("shape", "gauss") == "gauss")
    W = np.array([np.asarray(x["fit"]["weights"], float) for x in fits]).reshape(len(fits), len(names), K)
    P = np.array([np.asarray(x["fit"]["component_params"], float) for x in fits]).reshape(len(fits), K, -1)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez_compressed(tmp, weights=W, params=P, seeds=seeds, names=np.array(names),
                        sizes=np.bincount(cell_of_row, minlength=len(names)),
                        true_params=np.asarray(G.true_component_params(c), float), cell=json.dumps(c), rep=rep)
    tmp.rename(f)


def run(out_dir, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(out_dir) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    Parallel(n_jobs=workers or 1, verbose=5)(delayed(_run_one)(out_dir, t) for t in mine)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--reps", type=int, default=20)
    m.add_argument("--B", type=int, default=50)
    m.add_argument("--R", type=int, default=8)
    m.add_argument("--experiment", default=None, help="only cells of this experiment (e.g. ptvtv)")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, required=True)
    r.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.reps, a.B, a.R, a.experiment)
    else:
        run(a.out_dir, a.index, a.per_index, a.workers)


if __name__ == "__main__":
    main()
