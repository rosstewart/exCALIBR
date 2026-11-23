"""Control and VUS metrics for consequence-routed scoring of the paper arms (pilot: in-bag points).

A strategy maps each consequence class to an arm, with a default arm for every other class:
    pool          every variant from the pooled arm (strata ExCALIBR-equivalent)
    routed        missense -> mis, PTV -> ptv_poolprior, synonymous/splice/intronic -> their arms, else pool
    mis:<arm>     missense -> <arm>, else pool        (one per missense benign-reference / prior variant)
    ptv:<arm>     PTV -> <arm>, else pool              (one per PTV prior variant)
A consequence arm is used only where its calibration succeeded and the dataset has >= min_n P/LP of that
consequence; otherwise the strategy's default arm scores the variant. Counts are per measurement.

    python -m csq_analysis.metrics <fit_dir> --config S1_m5_R20 --mapping lp_anchored [--min 3] [--oob]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from csq_analysis import strata_uni as U

ROUTED = {"missense": "mis", "PTV": "ptv_poolprior", "synonymous": "syn", "splice": "splice",
          "intronic": "intronic"}


def strategies():
    s = {"pool": {}, "routed": dict(ROUTED)}
    for a in U.ARMS:
        if a.startswith("mis"):
            s[f"mis:{a}"] = {"missense": a}
        if a.startswith("ptv"):
            s[f"ptv:{a}"] = {"PTV": a}
    return s


def _ranges(cal):
    if not cal or cal.get("status") != "ok":
        return None
    return {int(k): [(float(a), float(b)) for a, b in v] for k, v in cal["point_ranges"].items() if v}


def points(fit_dir, cfgname, mapping, rows, min_n=3, strats=None, oob=False):
    """Long table, one row per scored variant (controls and VUS) of every dataset in rows, with one points column
    per strategy, plus csq / plp / blb / vus / dataset. rows must be the fit dir's rows.pkl (row order within a
    dataset = strata row positions). oob: controls take their out-of-bag points where the arm's calibration has
    them (calibrate --oob), else in-bag; VUS are never fitted, so they always take the full calibration's points."""
    from src.assay_calibration.fit_utils.fit import assign_points
    from csq_analysis.calibrate import load
    strats = strats or strategies()
    arms = sorted({a for m in strats.values() for a in m.values()} | {"pool"})
    out = []
    n_missing_oob = 0
    for ds, r in rows.groupby("dataset", sort=False):
        cals = {a: load(fit_dir, cfgname, a, mapping, ds) for a in arms}
        rng = {a: _ranges(c) for a, c in cals.items()}
        if rng["pool"] is None:
            continue
        x = r.score.to_numpy(float)
        pts = {}
        for a, v in rng.items():
            if v is None:
                pts[a] = None
                continue
            p = np.asarray(assign_points(x, v), int)
            if oob:
                o = cals[a].get("oob")
                if o is None:
                    n_missing_oob += 1
                else:
                    p[np.asarray(o["rows"], int)] = np.asarray(o["points"], int)
            pts[a] = p
        keep = ((r.plp ^ r.blb) | r.vus).to_numpy()
        r = r[keep]
        pts = {a: (p[keep] if p is not None else None) for a, p in pts.items()}
        n_plp = r[r.plp].csq.value_counts()
        cols = {}
        for name, route in strats.items():
            p = pts["pool"].copy()
            for csq, a in route.items():
                if pts.get(a) is not None and n_plp.get(csq, 0) >= min_n:
                    sel = (r.csq == csq).to_numpy()
                    p[sel] = pts[a][sel]
            cols[name] = p
        out.append(pd.DataFrame(dict(dataset=ds, csq=r.csq.to_numpy(), plp=r.plp.to_numpy(), blb=r.blb.to_numpy(),
                                     vus=r.vus.to_numpy(), **cols)))
    if oob and n_missing_oob:
        print(f"warning: {n_missing_oob} (dataset, arm) calibrations without OOB points scored in-bag")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarise(t, strats):
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics as M
    rows = []
    sets = {"all": t.plp | t.blb, "missense": (t.plp | t.blb) & t.csq.eq("missense"),
            "PTV": t.plp & t.csq.eq("PTV"), "non-missense": (t.plp | t.blb) & t.csq.ne("missense")}
    for sname, sel in sets.items():
        sub = t[sel]
        for s in strats:
            v_p, v_b = sub[sub.plp][s], sub[sub.blb][s]
            mat = pd.DataFrame([[int((v_b < 0).sum()), int((v_b == 0).sum()), int((v_b > 0).sum())],
                                [int((v_p < 0).sum()), int((v_p == 0).sum()), int((v_p > 0).sum())]],
                               index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"])
            m = M(mat) if mat.values.sum() else {}
            rows.append(dict(controls=sname, strategy=s, n_plp=len(v_p), n_blb=len(v_b), TP=int((v_p > 0).sum()),
                             FN=int((v_p < 0).sum()), FP=int((v_b > 0).sum()), TN=int((v_b < 0).sum()),
                             mcc=m.get("mcc"), dor=m.get("dor_standard"), determinate=m.get("coverage")))
        vus = t[t.vus & sel.reindex(t.index, fill_value=False)] if sname != "all" else t[t.vus]
        for s in strats:
            rows.append(dict(controls=f"VUS ({sname})", strategy=s, n_vus=len(vus),
                             vus_benign=int((vus[s] < 0).sum()), vus_pathogenic=int((vus[s] > 0).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit_dir")
    ap.add_argument("--config", default="S1_m5")
    ap.add_argument("--mapping", default="lp_anchored")
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--oob", action="store_true")
    a = ap.parse_args()
    from pathlib import Path
    import json
    from csq_analysis.fits import ROWS_FILE
    dss = json.loads((Path(a.fit_dir) / "bundle.json").read_text())["datasets"]
    rows = pd.read_pickle(Path(a.fit_dir) / ROWS_FILE)
    rows = rows[rows.dataset.isin(dss)]
    st = strategies()
    t = points(a.fit_dir, a.config, a.mapping, rows, a.min, st, oob=a.oob)
    s = summarise(t, st)
    out = Path(a.fit_dir) / "metrics"
    out.mkdir(exist_ok=True)
    tag = "oob" if a.oob else "inbag"
    s.to_csv(out / f"summary_{a.config}_{a.mapping}_min{a.min}_{tag}.csv", index=False)
    t.to_pickle(out / f"points_{a.config}_{a.mapping}_min{a.min}_{tag}.pkl")
    pd.set_option("display.width", 220)
    print(s[s.controls.isin(["all", "missense"])].to_string(index=False))


if __name__ == "__main__":
    main()
