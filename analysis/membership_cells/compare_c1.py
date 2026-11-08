"""C1 light membership-cell results vs the other routes to missense evidence, per dataset.

Points of every missense variant (standard points, in-bag), compared with the all-P/LP standard calibration
(medium_baseline_popall, prior over all gnomAD):
  cell_mis           membership-cell fit, missense P/LP + gnomAD-missense prior        (light: 20 bootstraps)
  cell_mis_3c        same at 3 components (2c datasets)
  cell_mis_nosplsyn  cell_mis without SpliceAI-flagged synonymous in the benign reference (splice assays)
  reweight           post-hoc re-weighting, gnomAD-missense prior   (large baseline, 500 bootstraps)
  refit              missense-only refit, gnomAD-missense prior     (medium, 100 bootstraps)
  subset_refit_max   largest change among the 3 count-matched random all-P/LP subset refits (count null)
Sanity: cell_all vs the standard calibration (should be ~0).

Outputs <cells>/light_c1/compare_c1.csv and compare_c1.png (dot plot: mean |delta points| vs all P/LP per dataset,
one dot per route, sorted by number of missense P/LP; benign- and pathogenic-call flips in the CSV).

Usage:  python -m analysis.membership_cells.compare_c1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
S = ROOT / "sens_popmis_lp"
C = ROOT / "cells" / "light_c1"


def _pts(d, ds):
    f = sorted((d / ds).glob(f"{ds}_*_variants.csv"))
    if not f:
        return None
    x = pd.read_csv(f[0], usecols=lambda c: c in {"variant_id", "standard_points", "aa_ref", "aa_alt"})
    a, r = x.aa_alt.astype(str), x.aa_ref.astype(str)
    mis = x.aa_alt.notna() & x.aa_ref.notna() & (a != r) & ~a.isin(["*", "Ter", "nan", ""])
    return x[mis].drop_duplicates("variant_id").set_index("variant_id").standard_points.astype(float)


def _cmp(p0, p1):
    if p0 is None or p1 is None:
        return dict(mean_abs=np.nan, ben_flip=np.nan, path_flip=np.nan, dmax=np.nan)
    j = p0.index.intersection(p1.index)
    a, b = p0.loc[j].values, p1.loc[j].values
    return dict(mean_abs=float(np.mean(np.abs(b - a))), ben_flip=float(np.mean((a < 0) != (b < 0))),
                path_flip=float(np.mean((a > 0) != (b > 0))), dmax=float(b.max() - a.max()))


def main():
    cfg = json.load(open("src/igvf_configs/dataset_configs_aug_2026.json"))
    rep = pd.read_csv(C / "compose_report.csv")
    dss = sorted(rep.dataset.unique())
    nmis = {}
    for f in (S / "e1_full" / "e1_matched_subsample_parts").glob("*.csv"):
        if f.stat().st_size > 2:
            d = pd.read_csv(f)
            nmis[f.stem] = int(d.n_plp_missense.iloc[0]) if "n_plp_missense" in d else np.nan
    rows = []
    for ds in dss:
        e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
        kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"
        kc = kc if kc in ("2c", "3c") else "3c"
        base = _pts(S / "medium_baseline_popall", ds)
        routes = {
            "cell_mis": _pts(C / f"calib_cell_mis_{kc}", ds),
            "cell_mis_3c": _pts(C / "calib_cell_mis_3c", ds) if kc == "2c" else None,
            "cell_mis_nosplsyn": _pts(C / f"calib_cell_mis_nosplsyn_{kc}", ds),
            "reweight": _pts(S / "large_baseline_plpmis_reweight", ds),
            "refit": _pts(S / "medium_plpmis", ds),
        }
        r = dict(dataset=ds, K=kc, n_mis=nmis.get(ds, np.nan))
        for k, p in routes.items():
            r.update({f"{k}_{m}": v for m, v in _cmp(base, p).items()})
        subs = [_cmp(base, _pts(S / f"medium_plpsub_s{i}", ds)) for i in range(3)]
        r["subset_refit_max_mean_abs"] = np.nanmax([x["mean_abs"] for x in subs]) if subs else np.nan
        r.update({f"sanity_cell_all_{m}": v for m, v in _cmp(base, _pts(C / f"calib_cell_all_{kc}", ds)).items()})
        r["cell_vs_refit_mean_abs"] = _cmp(routes["refit"], routes["cell_mis"])["mean_abs"]
        r["cell_vs_reweight_mean_abs"] = _cmp(routes["reweight"], routes["cell_mis"])["mean_abs"]
        rows.append(r)
    d = pd.DataFrame(rows).sort_values(["n_mis", "dataset"])
    d.to_csv(C / "compare_c1.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    cols = ["dataset", "K", "n_mis", "sanity_cell_all_mean_abs", "cell_mis_mean_abs", "reweight_mean_abs",
            "refit_mean_abs", "subset_refit_max_mean_abs", "cell_mis_ben_flip", "refit_ben_flip",
            "cell_vs_refit_mean_abs", "cell_vs_reweight_mean_abs", "cell_mis_nosplsyn_mean_abs"]
    print(d[cols].round(2).to_string(index=False))
    _plot(d, C / "compare_c1.png")


def _plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.missense_sensitivity.plot_stability_rules import display_name
    d = d.reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.32 * n + 2.2), gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.05},
                           sharey=True)
    y = np.arange(n)
    for yy in y[::2]:
        for a in ax:
            a.axhspan(yy - 0.5, yy + 0.5, color="#f4f3f0", lw=0, zorder=0)
    a = ax[0]
    a.hlines(y, 0, d.subset_refit_max_mean_abs, color="#1baf7a", lw=6, alpha=0.35, label="random subset refits (largest)")
    for col, c, m, lab in [("reweight_mean_abs", "#eb6834", "o", "re-weighting"), ("refit_mean_abs", "#6250d6", "s",
                           "missense-only refit"), ("cell_mis_mean_abs", "#e87ba4", "D", "membership-cell fit"),
                           ("cell_mis_3c_mean_abs", "#a50f5f", "d", "membership-cell fit, 3c"),
                           ("sanity_cell_all_mean_abs", "#52514e", "|", "cell fit, all P/LP (sanity)")]:
        if col in d:
            a.scatter(d[col], y, color=c, marker=m, s=34, zorder=3, label=lab, edgecolor="white", lw=0.5)
    a.set_yticks(y); a.set_yticklabels([display_name(x) for x in d.dataset], fontsize=8)
    a.set_ylim(n - 0.5, -0.5); a.grid(axis="x", color="#e4e3df"); a.set_axisbelow(True)
    a.set_xlabel("mean |change in points| per missense variant vs all-P/LP calibration")
    a.legend(loc="lower right", fontsize=8, frameon=False)
    a.set_title("C1 light membership-cell fit vs other routes to missense evidence (20 / 100 / 500 bootstraps)",
                fontsize=10, loc="left")
    b = ax[1]
    b.barh(y, d.n_mis.fillna(0), color="#52514e", height=0.5)
    b.set_xscale("symlog", linthresh=1); b.set_xlabel("missense P/LP (n)")
    for v in (5, 10):
        b.axvline(v, color="#52514e", ls=":", lw=1)
    for s in ("top", "right"):
        a.spines[s].set_visible(False); b.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
