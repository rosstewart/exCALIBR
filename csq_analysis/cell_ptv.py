"""PTV-specific cell-model calibration (P/LP = PTV cells, prior fixed at the stored cell_all prior) for every
canonical dataset with PTV P/LP, from the stored large-labeled cell fits. PRELIMINARY (in-bag, lp_anchored,
500 bootstraps x 8 restarts). Point ranges cached as JSON per dataset.

    python -m csq_analysis.cell_ptv [--workers 8] [--inner 8]
"""
from __future__ import annotations

import argparse
import json

from csq_analysis import config as C

OUT = C.CACHE / "cell_ptv"


def stored(ds, kc, arm):
    f = sorted((C.CELLS_LARGE / f"calib_{arm}_{kc}" / ds).glob("*_calibration.json"))
    return json.loads(f[0].read_text()) if f else None


def cell_k(ds):
    from analysis.consequence_cells_sim.mech_overlap import _real_k
    return _real_k(ds)


def one(ds, inner=8):
    f = OUT / f"{ds}.json"
    if f.exists():
        return json.loads(f.read_text())
    import logging
    logging.disable(logging.WARNING)
    from analysis.consequence_cells_sim.mech_overlap import cell_arm_calibration
    kc = cell_k(ds)
    base = stored(ds, kc, "cell_all")
    rec = dict(dataset=ds, K=kc)
    if base is None:
        rec["status"] = "no cell_all"
    else:
        try:
            cal = cell_arm_calibration(ds, "ptv", manual_prior=float(base["prior"]), n_jobs=inner)
            rec.update(status="ok", prior=float(base["prior"]),
                       point_ranges={str(k): v for k, v in cal["point_ranges"].items()})
        except Exception as e:
            rec.update(status=f"error: {repr(e)[:200]}")
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    return rec


def main():
    from joblib import Parallel, delayed
    from csq_analysis.annotate import load_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--min-ptv", type=int, default=1)
    a = ap.parse_args()
    r = load_rows()
    n = r[r.plp & (r.csq == "PTV")].groupby("dataset").size()
    dss = sorted(n[n >= a.min_ptv].index)
    print(f"{len(dss)} datasets with >= {a.min_ptv} PTV P/LP")
    res = Parallel(n_jobs=a.workers, verbose=5)(delayed(one)(ds, a.inner) for ds in dss)
    bad = [x for x in res if x.get("status") != "ok"]
    print(f"done: {len(res) - len(bad)} ok; problems: {[(x['dataset'], x['status']) for x in bad]}")


if __name__ == "__main__":
    main()
