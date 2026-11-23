"""PRELIMINARY confusion matrices: consequence-agnostic vs consequence-specific calibration (2026-11-20).

All arms are the existing experimental runs: in-bag points, lp_anchored mapping. Standard arms are the first 100
production bootstraps (sens_popmis_lp); cell arms are the large-labeled cell fits (500 bootstraps x 8 restarts);
cell_ptv is csq_analysis.cell_ptv (pooled prior). Points are assigned from each arm's global point ranges to the
production scoreset rows (csq_analysis.annotate), and ClinVar labels are the production P/LP / B/LB memberships,
so every strategy is scored on the identical control set. Counts are per measurement, as in the manuscript.

Strategies (how each control gets its evidence):
    standard pooled     std_all for every variant (ExCALIBR as published, at this budget and mapping)
    standard routed     missense -> std_mis (missense-only P/LP refit), others -> std_all
    cell pooled         cell_all (strata model, all P/LP) for every variant
    cell routed         missense -> cell_mis, PTV -> cell_ptv, others -> cell_all
    hybrid              cell routed in flagged datasets (Δγ q95 < -0.1 or TV_D q05 > 0.2), standard pooled elsewhere
A consequence arm is used only where the dataset has >= MIN P/LP of that consequence, else the pooled arm of the
same model.

    python -m csq_analysis.prelim_confusion [--min 3]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from csq_analysis import config as C  # noqa: E402

S = C.DATA / "missense_only_calibrations/sens_popmis_lp"
STRATS = ["standard pooled", "standard routed", "cell pooled", "cell routed", "hybrid"]
SETS = {"all controls": lambda r: np.ones(len(r), bool), "missense controls": lambda r: r.csq.eq("missense").to_numpy(),
        "non-missense controls": lambda r: r.csq.ne("missense").to_numpy()}
INK, MUTED = "#2b2b2a", "#6b6a64"


def _ranges(cal):
    if not cal:
        return None
    return {int(k): [(float(a), float(b)) for a, b in v] for k, v in cal["point_ranges"].items() if v}


def _cal(d):
    f = sorted(d.glob("*_calibration.json")) if d.exists() else []
    return json.loads(f[0].read_text()) if f else None


def arm_ranges(ds):
    from csq_analysis.cell_ptv import cell_k, OUT as PTV
    kc = cell_k(ds)
    out = {"std_all": _ranges(_cal(S / "medium_baseline_popall" / ds)),
           "std_mis": _ranges(_cal(S / "medium_plpmis" / ds)),
           "cell_all": _ranges(_cal(C.CELLS_LARGE / f"calib_cell_all_{kc}" / ds)),
           "cell_mis": _ranges(_cal(C.CELLS_LARGE / f"calib_cell_mis_{kc}" / ds))}
    f = PTV / f"{ds}.json"
    rec = json.loads(f.read_text()) if f.exists() else {}
    out["cell_ptv"] = _ranges(rec) if rec.get("status") == "ok" else None
    return out


def flagged():
    from analysis.consequence_cells_sim.plot_mech_overlap import load
    d = load()
    return set(d.dataset[d.flag])


def score(min_n=3):
    """Long table: one row per control (P/LP or B/LB row of a canonical dataset) with points under every strategy."""
    from src.assay_calibration.fit_utils.fit import assign_points
    from csq_analysis.annotate import load_rows
    rows = load_rows()
    rows = rows[(rows.plp ^ rows.blb) & rows.dataset.isin(C.canonical_datasets())]   # conflicting P+B rows dropped
    flags = flagged()
    out, skipped = [], []
    for ds, r in rows.groupby("dataset"):
        a = arm_ranges(ds)
        if a["std_all"] is None or a["cell_all"] is None:
            skipped.append(ds)
            continue
        x = r.score.to_numpy(float)
        pts = {k: (np.asarray(assign_points(x, v), int) if v is not None else None) for k, v in a.items()}
        n_plp = r[r.plp].csq.value_counts()
        use_mis_std = pts["std_mis"] is not None and n_plp.get("missense", 0) >= min_n
        use_mis_cell = pts["cell_mis"] is not None and n_plp.get("missense", 0) >= min_n
        use_ptv_cell = pts["cell_ptv"] is not None and n_plp.get("PTV", 0) >= min_n
        mis, ptv = r.csq.eq("missense").to_numpy(), r.csq.eq("PTV").to_numpy()
        s_std = pts["std_all"]
        s_std_r = np.where(mis & use_mis_std, pts["std_mis"] if use_mis_std else 0, s_std)
        s_cell = pts["cell_all"]
        s_cell_r = np.where(mis & use_mis_cell, pts["cell_mis"] if use_mis_cell else 0, s_cell)
        s_cell_r = np.where(ptv & use_ptv_cell, pts["cell_ptv"] if use_ptv_cell else 0, s_cell_r)
        s_hyb = s_cell_r if ds in flags else s_std
        out.append(pd.DataFrame({"dataset": ds, "csq": r.csq.to_numpy(), "plp": r.plp.to_numpy(),
                                 "flagged": ds in flags, "standard pooled": s_std, "standard routed": s_std_r,
                                 "cell pooled": s_cell, "cell routed": s_cell_r, "hybrid": s_hyb}))
    t = pd.concat(out, ignore_index=True)
    print(f"{t.dataset.nunique()} datasets scored; skipped (missing std_all or cell_all): {skipped}")
    return t


def matrix(sub, col):
    p, b = sub[sub.plp][col], sub[~sub.plp][col]
    cnt = lambda v: [int((v < 0).sum()), int((v == 0).sum()), int((v > 0).sum())]
    return pd.DataFrame([cnt(b), cnt(p)], index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"])


def summary(t):
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics as M
    rows = []
    for sname, sel in SETS.items():
        sub = t[sel(t)]
        for s in STRATS:
            m = M(matrix(sub, s))
            rows.append(dict(controls=sname, strategy=s, **{k: m[k] for k in ("TP", "FN", "FP", "TN", "uncertain",
                                                                               "coverage", "mcc", "dor_standard")}))
    return pd.DataFrame(rows)


def figure(t, path, min_n):
    from analysis.confusion import draw_diverging_confusion_heatmap
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics as M
    lab = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indet.", "Normal": "Benign", "Abnormal": "Pathogenic"}
    fig, axes = plt.subplots(len(SETS), len(STRATS), figsize=(4.1 * len(STRATS), 3.5 * len(SETS)))
    for i, (sname, sel) in enumerate(SETS.items()):
        sub = t[sel(t)]
        for j, s in enumerate(STRATS):
            ax = axes[i, j]
            mat = matrix(sub, s)
            draw_diverging_confusion_heatmap(ax, mat, fontsize=11)
            ax.set_xticks(np.arange(3) + 0.5, [lab[c] for c in mat.columns], fontsize=8.5)
            ax.set_yticks(np.arange(2) + 0.5, [lab[r] for r in mat.index], fontsize=9)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            m = M(mat)
            ax.set_title(f"{s}" if i == 0 else "", fontsize=11, fontweight="bold", color=INK, pad=8)
            ax.text(0.5, -0.17, f"MCC {m['mcc']:.3f} · DOR {m['dor_standard']:.0f} · determinate {100 * m['coverage']:.1f}%",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
            if j == 0:
                ax.set_ylabel(f"{sname}\n(n = {len(sub):,})", fontsize=10.5, color=INK)
    n_ds, n_fl = t.dataset.nunique(), t[t.flagged].dataset.nunique()
    fig.suptitle(f"PRELIMINARY confusion matrices, {n_ds} canonical datasets ({n_fl} flagged); in-bag, lp_anchored; "
                 f"consequence arms used where ≥ {min_n} P/LP of that consequence", fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def by_consequence(t):
    rows = []
    for (c, plp), sub in t.groupby(["csq", "plp"]):
        for s in STRATS:
            v = sub[s]
            rows.append(dict(csq=c, label="P/LP" if plp else "B/LB", strategy=s, n=len(v), benign=int((v < 0).sum()),
                             indeterminate=int((v == 0).sum()), pathogenic=int((v > 0).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=3)
    a = ap.parse_args()
    t = score(a.min)
    tag = f"min{a.min}"
    s = summary(t)
    s.to_csv(C.PRELIM / f"P5_confusion_summary_{tag}.csv", index=False)
    by_consequence(t).to_csv(C.PRELIM / f"P5_by_consequence_{tag}.csv", index=False)
    figure(t, C.PRELIM / f"P5_confusion_{tag}.png", a.min)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
