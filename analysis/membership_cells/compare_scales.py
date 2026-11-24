"""Membership-cell evidence across runs: medium (100 bootstraps) vs large (500), labeled vs assay scale.

Pairs (each at the dataset's config K, arms cell_all and cell_mis; in-bag standard points of missense variants):
  boot    medium labeled (100 bootstraps)  ->  large labeled (500)     bootstrap-count stability
  scale   large labeled                    ->  large assay             adding unlabeled assayed cells (C2)
Per dataset: mean |delta points|, share of variants whose points change, evidence-direction flips (P<->B and
to/from indeterminate), delta prior, both runs' pct_bootstraps_kept.

Outputs <large labeled>/compare_scales.csv and compare_scales.png.
Usage:  python -m analysis.membership_cells.compare_scales
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.membership_cells.compare_c1 import _pts

C = Path("/data/ross/assay_calibration/missense_only_calibrations/cells")
RUNS = {"medium_labeled": C / "hpc_medium" / "labeled", "large_labeled": C / "hpc_large_labeled" / "labeled",
        "large_assay": C / "hpc_large_assay" / "assay"}
PAIRS = {"boot": ("medium_labeled", "large_labeled"), "scale": ("large_labeled", "large_assay")}


def _cal(d, ds):
    f = sorted(glob.glob(str(d / ds / "*_calibration.json")))
    return json.load(open(f[0])) if f else None


def main():
    cfg = json.load(open("src/igvf_configs/dataset_configs_aug_2026.json"))
    rows = []
    dss = sorted(p.stem for p in (RUNS["large_labeled"] / "cells").glob("*.npz"))
    for ds in dss:
        e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
        kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"
        kc = kc if kc in ("2c", "3c") else "3c"
        for arm in ("cell_all", "cell_mis"):
            for pair, (a, b) in PAIRS.items():
                da, db = RUNS[a] / f"calib_{arm}_{kc}", RUNS[b] / f"calib_{arm}_{kc}"
                pa, pb = _pts(da, ds), _pts(db, ds)
                ca, cb = _cal(da, ds), _cal(db, ds)
                if pa is None or pb is None:
                    continue
                x = pd.concat([pa.rename("a"), pb.rename("b")], axis=1).dropna()
                sa, sb = np.sign(x.a), np.sign(x.b)
                rows.append(dict(dataset=ds, arm=arm, K=kc, pair=pair, n=len(x),
                                 mean_abs_dpts=float((x.b - x.a).abs().mean()),
                                 frac_changed=float((x.b != x.a).mean()),
                                 flips_PB=int(((sa * sb) < 0).sum()),
                                 to_from_indet=int(((sa == 0) != (sb == 0)).sum()),
                                 gain_P=int(((sb > 0) & (sa <= 0)).sum()), lose_P=int(((sa > 0) & (sb <= 0)).sum()),
                                 gain_B=int(((sb < 0) & (sa >= 0)).sum()), lose_B=int(((sa < 0) & (sb >= 0)).sum()),
                                 prior_a=ca["prior"] if ca else np.nan, prior_b=cb["prior"] if cb else np.nan,
                                 kept_a=ca.get("pct_bootstraps_kept") if ca else np.nan,
                                 kept_b=cb.get("pct_bootstraps_kept") if cb else np.nan))
    d = pd.DataFrame(rows)
    d["dprior"] = d.prior_b - d.prior_a
    out = RUNS["large_labeled"]
    d.to_csv(out / "compare_scales.csv", index=False)
    pd.set_option("display.width", 250)
    for pair in PAIRS:
        x = d[d.pair == pair]
        print(f"\n== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]}")
        s = x.groupby("arm").agg(datasets=("dataset", "nunique"), median_mean_abs_dpts=("mean_abs_dpts", "median"),
                                 p90_mean_abs_dpts=("mean_abs_dpts", lambda v: v.quantile(.9)),
                                 median_frac_changed=("frac_changed", "median"), flips_PB=("flips_PB", "sum"),
                                 to_from_indet=("to_from_indet", "sum"), n=("n", "sum"),
                                 median_abs_dprior=("dprior", lambda v: v.abs().median()),
                                 max_abs_dprior=("dprior", lambda v: v.abs().max()))
        print(s.round(3).to_string())
        print("  largest changes:")
        print(x.sort_values("mean_abs_dpts", ascending=False).head(8)[
            ["dataset", "arm", "K", "n", "mean_abs_dpts", "flips_PB", "gain_P", "lose_P", "gain_B", "lose_B",
             "prior_a", "prior_b", "kept_a", "kept_b"]].round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, pair in zip(axs, PAIRS):
        for arm, col in (("cell_all", "#4C72B0"), ("cell_mis", "#DD8452")):
            x = d[(d.pair == pair) & (d.arm == arm)].sort_values("mean_abs_dpts")
            ax.plot(np.arange(len(x)), x.mean_abs_dpts.values, "o", ms=3.5, color=col, label=arm)
        ax.set_title(f"{PAIRS[pair][0]} → {PAIRS[pair][1]}")
        ax.set_xlabel("datasets (sorted)")
        ax.axhline(0.5, color="grey", lw=.6, ls=":")
    axs[0].set_ylabel("mean |Δ points|, missense variants")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "compare_scales.png", dpi=150)


if __name__ == "__main__":
    main()
