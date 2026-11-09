"""Control-based performance (in-bag standard points) for the membership-cell arms vs standard ExCALIBR.

Arms (all lp_anchored): standard all P/LP (first 100 bootstraps, prior all gnomAD), standard missense refit
(prior gnomAD missense), cell all P/LP, cell missense, and the two cell arms without SpliceAI-flagged synonymous
in the benign reference (splice-detecting assays; elsewhere they equal their parent arm).
Control sets (ClinVar, from the cell table): missense P/LP vs missense B/LB, and all P/LP vs all B/LB.
Metrics as compare_runs._metrics (the manuscript's): MCC at the method's own operating point (determinate
controls), determinate rate, accuracy, sensitivity/specificity among determinate, share of each class with
correct-direction evidence. Per dataset and pooled over datasets.

Usage:  python -m analysis.membership_cells.metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
H = Path(__import__("os").environ.get("CELLS_DIR", str(ROOT / "cells" / "hpc_medium" / "labeled")))
S = ROOT / "sens_popmis_lp"


def _pts(d, ds):
    f = sorted((d / ds).glob(f"{ds}_*_variants.csv"))
    if not f:
        return None
    x = pd.read_csv(f[0], usecols=["variant_id", "standard_points"]).drop_duplicates("variant_id")
    return x.set_index("variant_id").standard_points.astype(float)


def main(force_k=None):
    import sys
    force_k = force_k or (sys.argv[1] if len(sys.argv) > 1 else None)   # e.g. "3c": cell arms at 3c for every dataset
    from analysis.membership_cells.cells import Cells
    from analysis.missense_sensitivity.compare_runs import _metrics
    cfg = json.load(open("src/igvf_configs/dataset_configs_aug_2026.json"))
    rows = []
    for npz in sorted((H / "cells").glob("*.npz")):
        ds = npz.stem
        e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
        kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"
        kc = kc if kc in ("2c", "3c") else "3c"
        if force_k:
            kc = force_k
        c = Cells.load(npz)
        t = c.table
        cell = t.index.values[c.cell_of_row] if False else c.cell_of_row
        mask = np.array([list(m) for m in t["mask"]])[cell]
        mis = np.array([str(x).startswith("missense_variant") for x in t["consequence"]])[cell]
        ids = pd.Index(c.variant_ids)
        sets = {"missense": (ids[mask[:, 0] & mis], ids[mask[:, 1] & mis]),
                "all": (ids[mask[:, 0]], ids[mask[:, 1]])}
        arms = {"std_all": S / "medium_baseline_popall", "std_mis": S / "medium_plpmis",
                "cell_all": H / f"calib_cell_all_{kc}", "cell_mis": H / f"calib_cell_mis_{kc}",
                "cell_all_nosplsyn": H / f"calib_cell_all_nosplsyn_{kc}",
                "cell_mis_nosplsyn": H / f"calib_cell_mis_nosplsyn_{kc}"}
        splice = bool(c.meta.get("detects_splice"))
        for arm, d in arms.items():
            p = _pts(d, ds)
            if p is None and arm.endswith("nosplsyn") and not splice:   # equals its parent arm here
                p = _pts(arms[arm.replace("_nosplsyn", "")], ds)
            if p is None:
                continue
            for sname, (pid, bid) in sets.items():
                pp, bb = p.reindex(pid).dropna(), p.reindex(bid).dropna()
                m = _metrics(int((pp > 0).sum()), int((bb > 0).sum()), int((bb < 0).sum()), int((pp < 0).sum()),
                             int((pp == 0).sum()), int((bb == 0).sum()))
                rows.append(dict(dataset=ds, arm=arm, controls=sname, splice=splice, **m))
    d = pd.DataFrame(rows)
    d.to_csv(H / ("metrics_inbag.csv" if not force_k else f"metrics_inbag_cells_{force_k}.csv"), index=False)
    pd.set_option("display.width", 250)
    order = ["std_all", "std_mis", "cell_all", "cell_mis", "cell_all_nosplsyn", "cell_mis_nosplsyn"]
    for sname in ["missense", "all"]:
        x = d[(d.controls == sname) & (d.n_plp > 0) & (d.n_blb > 0)]
        common = x.groupby("dataset").arm.nunique()
        x = x[x.dataset.isin(common[common == len(order)].index)]
        per = x.groupby("arm")[["mcc", "determinate_rate", "plp_correct_all", "blb_correct_all"]].median()
        pool = x.groupby("arm")[["tp", "fp", "tn", "fn", "ir_plp", "ir_blb"]].sum()
        pooled = pd.DataFrame([dict(arm=a, **_metrics(*[int(pool.loc[a, k]) for k in ["tp", "fp", "tn", "fn", "ir_plp", "ir_blb"]]))
                               for a in pool.index]).set_index("arm")
        out = per.add_prefix("median_").join(pooled[["mcc", "determinate_rate", "sensitivity_det", "specificity_det",
                                                     "plp_correct_all", "blb_correct_all"]].add_prefix("pooled_"))
        print(f"\n== {sname} controls ({x.dataset.nunique()} datasets with both classes, all arms present)")
        print(out.reindex(order).round(3).to_string())
        sp = x[x.splice]
        if len(sp):
            print(f"   splice-detecting subset ({sp.dataset.nunique()} datasets): median MCC")
            print(sp.groupby("arm").mcc.median().reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
