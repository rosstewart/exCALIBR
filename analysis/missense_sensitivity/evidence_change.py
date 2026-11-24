"""Which datasets change missense evidence substantially under missense re-weighting, missense-only refit,
or random count-matched subset refits (and how much the prior population alone moves it)?

Per dataset, standard points of every missense variant (VUS and controls) from the corrected sensitivity
calibrations (sens_popmis_lp; lp_anchored mapping), compared pairwise on shared variant ids:
  reweight      large_baseline            -> large_baseline_plpmis_reweight   (gnomAD missense prior; 500 boots)
  refit         medium_baseline           -> medium_plpmis                    (gnomAD missense prior; 100 boots)
  subset_refit  medium_baseline           -> medium_plpsub_s0..2 (each; count-matched null for 'refit')
  prior_pop     medium_baseline_popall    -> medium_baseline                  (all gnomAD -> gnomAD missense prior)
Metrics per comparison: mean |delta points|, share of missense variants with |delta| >= 2, share whose
pathogenic call flips (>0 vs <=0) or benign call flips (<0 vs >=0), and the change in the highest pathogenic
level among missense variants.

Usage:  python -m analysis.missense_sensitivity.evidence_change
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
S = ROOT / "sens_popmis_lp"
PAIRS = {"reweight": ("large_baseline", ["large_baseline_plpmis_reweight"]),
         "refit": ("medium_baseline", ["medium_plpmis"]),
         "subset_refit": ("medium_baseline", ["medium_plpsub_s0", "medium_plpsub_s1", "medium_plpsub_s2"]),
         "prior_pop": ("medium_baseline_popall", ["medium_baseline"])}


def _pts(run, ds):
    f = sorted((S / run / ds).glob(f"{ds}_*_variants.csv"))
    if not f:
        return None
    d = pd.read_csv(f[0], usecols=lambda c: c in {"variant_id", "standard_points", "aa_ref", "aa_alt"})
    a, r = d.aa_alt.astype(str), d.aa_ref.astype(str)
    mis = d.aa_alt.notna() & d.aa_ref.notna() & (a != r) & ~a.isin(["*", "Ter", "nan", ""])
    return d[mis].drop_duplicates("variant_id").set_index("variant_id").standard_points.astype(float)


def _cmp(p0, p1):
    j = p0.index.intersection(p1.index)
    a, b = p0.loc[j].values, p1.loc[j].values
    return dict(n=len(j), mean_abs=float(np.mean(np.abs(b - a))), frac_ge2=float(np.mean(np.abs(b - a) >= 2)),
                path_flip=float(np.mean((a > 0) != (b > 0))), ben_flip=float(np.mean((a < 0) != (b < 0))),
                dmax=float(b.max() - a.max()), dmin=float(b.min() - a.min()))


def main():
    dss = sorted(p.name for p in (S / "medium_baseline").iterdir() if p.is_dir())
    rows = []
    for ds in dss:
        for name, (base, alts) in PAIRS.items():
            p0 = _pts(base, ds)
            if p0 is None:
                continue
            res = [_cmp(p0, p1) for p1 in (_pts(a, ds) for a in alts) if p1 is not None]
            if not res:
                continue
            r = {k: (max(x[k] for x in res) if k != "n" else res[0][k]) for k in res[0]}  # subsets: largest change
            rows.append(dict(dataset=ds, comparison=name, **r))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "analysis_v3" / "evidence_change_sens.csv", index=False)
    w = d.pivot(index="dataset", columns="comparison", values=["mean_abs", "frac_ge2", "path_flip", "ben_flip", "dmax"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w.to_csv(ROOT / "analysis_v3" / "evidence_change_sens_wide.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print("median over datasets:")
    print(d.groupby("comparison")[["mean_abs", "frac_ge2", "path_flip", "ben_flip"]].median().round(3))
    for c in ["reweight", "refit", "subset_refit", "prior_pop"]:
        s = d[d.comparison == c]
        big = s[(s.mean_abs >= 0.5) | (s.frac_ge2 >= 0.10) | (s.dmax.abs() >= 2)]
        print(f"\n{c}: {len(big)}/{len(s)} datasets with mean|d| >= 0.5, >=10% of missense moving >= 2 points, or top level moving >= 2")
        print(big.sort_values("mean_abs", ascending=False)[["dataset", "n", "mean_abs", "frac_ge2", "path_flip", "ben_flip", "dmax"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
