"""The paper's performance results at every evidence percentile, and the
choice of percentile per evidence type.

For each p in review_common.PERCENTILES, review/percentile/p<p>/ holds a copy
of the inputs the performance figures read, with every joint-model quantity at
percentile p (pathogenic evidence from the p-th percentile of the log LR+ over
bootstrap fits, benign from the (100-p)-th):
  functional|predictor|combined results tables   from the existing sweep
      (percentile_sweep/p<p>/; p = 5 the paper's), joint rows pinned to the
      paper's (p = 5) canonical configuration so only the percentile varies
  production_oob/production_oob_comparison.csv   joint out-of-bag (MV) and
      in-sample (mv_insample) rows recomputed at p (review_common); P-KNN and
      classifier rows unchanged (their own 5th/95th percentiles)
  strength/strength_points.csv                   joint points of controls and
      gnomAD variants at p; VUS rows (not model rows for most genes) dropped
  everything else the figures read (independent comparators, P-KNN) copied
      unchanged: those do not depend on the joint model's percentile
Then the performance figures are drawn into p<p>/accuracy/ with
MV_PAPER_FIGURES_DIR pointing at p<p>.

review/percentile/selection/ compares the percentiles per evidence type by
  (1) calibration: the likelihood ratio controls show at each cumulative
      evidence level (>= k, <= -k) against the LR the level claims, out of bag
      (review_vep_calibration.calibration); a percentile overstates when the
      upper CI of a pathogenic level, or the lower CI of a benign level, misses
      the claim;
  (2) out-of-bag MCC and determinate rate (median over genes, gene level);
  (3) all of them, as figures.

Usage: python -m mv_analysis.review_percentile_sets [--figures]
"""
import argparse
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

MAIN = Path(config.PAPER_FIGURES_DIR)
TABLES = {"functional": "functional/functional_all_results_table.csv",
          "predictor": "predictor/predictor_results_table.csv",
          "combined": "combined/combined_results_table.csv"}
COPY_DIRS = ("functional", "predictor", "combined", "pooled", "pknn", "production_oob", "strength", "correlation")
FIGURES = ("make_performance_figures.py", "make_panel_performance.py")
MIN_LEVEL_N = 20  # controls a cumulative level needs before its calibration is judged


def _canonical(table):
    """(gene_set, gene, cluster) -> the paper's canonical MV config."""
    from mv_analysis.baselines import EVIDENCE_DIRECTION
    t = pd.read_csv(table)
    t = t[(t["threshold"] == EVIDENCE_DIRECTION) & t["method"].str.startswith("MV")]
    if "gene_set" not in t.columns:
        t = t.assign(gene_set=None)
    c = config.select_canonical_mv_rows(t)
    return {(gs, g, cl): cf for gs, g, cl, cf in zip(c["gene_set"], c["gene"], c["cluster"], c["config"])}


def _pin(t, canon):
    """Keep only the canonical config's rows of a results table (rows without a config stay)."""
    if "config" not in t.columns:
        return t
    gs = t["gene_set"] if "gene_set" in t.columns else pd.Series([None] * len(t), index=t.index)
    keep = [pd.isna(cf) or canon.get((s, g, cl)) in (None, cf)
            for s, g, cl, cf in zip(gs, t["gene"], t["cluster"], t["config"])]
    return t[keep]


def _oob_comparison(p, main_cmp):
    from mv_analysis import report
    from mv_analysis.baselines import EVIDENCE_DIRECTION
    rows = []
    for key, m in R.paper_keys():
        o = R.oob_points(key, p)
        never = int((o["n_oob"] == 0).sum())
        o = o[o["n_oob"] > 0]
        ins = R.insample_points(key, p) if p != 5 else R.insample_paper_points(key)
        for method, pts in (("MV", o["points"].to_numpy(float)), ("mv_insample", ins[o["row"].to_numpy()])):
            r = [x for x in report._rows_for_points(method, np.nan_to_num(pts, nan=0.0), o["label"].to_numpy(int))
                 if x["threshold"] == EVIDENCE_DIRECTION][0]
            rows.append({"key": key, "panel": m["panel"], "gene_set": m["gene_set"], "gene": m["gene"],
                         "n_never_oob": never, "n_unscored": int(np.isnan(pts).sum()),
                         "min_oob_splits": int(o["n_oob"].min()), "det_rate": r["determinate"] / r["total"], **r})
    new = pd.DataFrame(rows)
    other = main_cmp[~main_cmp["method"].isin(["MV", "mv_insample"])]
    return pd.concat([new, other], ignore_index=True)


def _strength(p, main_sp):
    """strength_points with joint points at p for model rows (controls, gnomAD)."""
    out = []
    for key, x in main_sp.groupby("key"):
        x = x[x["sample"] != "VUS"].copy()
        if R.SUBSET_FROM and key not in KEYS():
            continue
        if p != 5 or R.ALT_FITS:
            d, _ = R.prep(key)
            pos = {str(v): i for i, v in enumerate(d["variants"])}
            ins = R.insample_points(key, p)
            j = x["variant"].astype(str).map(pos)
            x = x[j.notna()]
            x["joint_points"] = ins[j.dropna().astype(int).to_numpy()]
        out.append(x)
    out = pd.concat(out, ignore_index=True)
    # back to the figures' own key form
    out["key"] = np.where(out["panel"] == "predictor", out["key"].str.replace("^predictor__", "predictor-mv__", regex=True),
                          out["key"])
    return out


def KEYS():
    return {k for k, _ in R.paper_keys()}


def build(p, figures=False):
    dst = R.out_dir(f"percentile/p{p}")
    if R.SUBSET_FROM:  # alternative fits / a subset of models: the out-of-bag
        # comparison and the strength points only (the results tables and
        # figures are the paper fits' own)
        for sub in ("production_oob", "strength"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
        main_cmp = pd.read_csv(MAIN / "production_oob/production_oob_comparison.csv")
        _oob_comparison(p, main_cmp[main_cmp["key"].isin(KEYS())]).to_csv(
            dst / "production_oob/production_oob_comparison.csv", index=False)
        _strength(p, R.strength_points()).to_csv(dst / "strength/strength_points.csv", index=False)
        return dst
    for sub in COPY_DIRS:
        if (MAIN / sub).exists():
            shutil.copytree(MAIN / sub, dst / sub, dirs_exist_ok=True)
    src = MAIN if p == 5 else MAIN / "percentile_sweep" / f"p{p}"
    for panel, rel in TABLES.items():
        canon = _canonical(MAIN / rel)
        t = pd.read_csv(src / rel)
        _pin(t, canon).to_csv(dst / rel, index=False)
    main_cmp = pd.read_csv(MAIN / "production_oob/production_oob_comparison.csv")
    _oob_comparison(p, main_cmp).to_csv(dst / "production_oob/production_oob_comparison.csv", index=False)
    _strength(p, R.strength_points()).to_csv(dst / "strength/strength_points.csv", index=False)
    (dst / "accuracy").mkdir(exist_ok=True)
    if figures:
        env = dict(os.environ, MV_PAPER_FIGURES_DIR=str(dst))
        for script in FIGURES:
            subprocess.run([sys.executable, str(Path(config._ROOT) / "mv_analysis/figures/scripts" / script)],
                           env=env, check=True)
    return dst


def selection():
    from mv_analysis import review_vep_calibration as V
    from mv_analysis import baselines as B
    out = R.out_dir("percentile/selection")
    cal = []
    for panel in ("functional", "predictor", "combined"):
        if not R.paper_keys(panel):
            continue
        t = V.control_table(panel)
        c = V.calibration(t, panel)
        cal.append(c[c["method"].str.startswith("joint") | (c["method"] == "nonconflicting comparator")])
    cal = pd.concat(cal, ignore_index=True)
    cal.to_csv(out / "calibration_by_percentile.csv", index=False)
    perf = []
    for p in R.PERCENTILES:
        cmp = pd.read_csv(R.REVIEW_DIR / f"percentile/p{p}/production_oob/production_oob_comparison.csv")
        mv = cmp[cmp["method"] == "MV"]
        for panel, x in mv.groupby("panel"):
            x = x[(x["n_pathogenic"] > 0) & (x["n_benign"] > 0)]
            g = B.gene_level(x, by=("panel", "gene"))  # BRCA2's groups count once
            defined = g[B.mcc_defined(g)]
            tot = x[["TP", "TN", "FP", "FN", "total"]].sum()
            den = np.sqrt(float((tot.TP + tot.FP) * (tot.TP + tot.FN) * (tot.TN + tot.FP) * (tot.TN + tot.FN)))
            perf.append(dict(panel=panel, percentile=p, n_genes=len(g), n_genes_mcc_defined=len(defined),
                             median_mcc=defined["mcc"].median(),
                             median_det_rate=(g["determinate"] / g["total"]).median(),
                             pooled_mcc=(tot.TP * tot.TN - tot.FP * tot.FN) / den if den else np.nan,
                             pooled_det_rate=(tot.TP + tot.TN + tot.FP + tot.FN) / tot.total,
                             errors=int(tot.FP + tot.FN)))
    perf = pd.DataFrame(perf)
    perf.to_csv(out / "performance_by_percentile.csv", index=False)
    j = cal[(cal["scope"] == "cumulative") & cal["method"].str.startswith("joint")].copy()
    j["percentile"] = j["method"].str.extract(r"p(\d+)").astype(int)
    # a level is judged only with enough controls to test its claim
    j["overstated"] = j["overstated"].astype(bool) & ((j["n_plp"] + j["n_blb"]) >= MIN_LEVEL_N)
    over = j.groupby(["panel", "percentile"])["overstated"].any().rename("any_level_overstated").reset_index()
    j[j["overstated"]].to_csv(out / "overstated_levels.csv", index=False)
    summ = perf.merge(over, on=["panel", "percentile"], how="left")
    rec = []
    for panel, x in summ.groupby("panel"):
        ok = x[~x["any_level_overstated"].fillna(False)]
        rec.append(dict(panel=panel,
                        by_calibration=int(ok["percentile"].max()) if len(ok) else None,
                        by_oob_mcc=int(x.loc[x["median_mcc"].idxmax(), "percentile"])))
    summ.to_csv(out / "summary_by_percentile.csv", index=False)
    pd.DataFrame(rec).to_csv(out / "recommended_percentile.csv", index=False)
    _selection_figure(cal, perf, out)
    print(pd.DataFrame(rec).to_string(index=False))
    return summ


def _selection_figure(cal, perf, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ("functional", "predictor", "combined")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for c, panel in enumerate(panels):
        ax = axes[0, c]
        x = perf[perf["panel"] == panel].sort_values("percentile")
        ax.plot(x["percentile"], x["median_mcc"], "o-", color="#1d1d1b", label="median MCC (out of bag)")
        ax.set_ylabel("median MCC")
        ax2 = ax.twinx()
        ax2.plot(x["percentile"], x["median_det_rate"], "s--", color="#d55e00", label="median determinate rate")
        ax2.set_ylabel("median determinate rate", color="#d55e00")
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel("percentile")
        ax = axes[1, c]
        d = cal[(cal["panel"] == panel) & (cal["scope"] == "cumulative") & cal["method"].str.startswith("joint")]
        for lv, col in zip((1, 2, 4, -1, -2, -4), ("#f4a582", "#d6604d", "#b2182b", "#92c5de", "#4393c3", "#2166ac")):
            y = d[d["level"] == lv].copy()
            if y.empty:
                continue
            y["percentile"] = y["method"].str.extract(r"p(\d+)").astype(int)
            y = y.sort_values("percentile")
            ratio = np.log(y["observed_lr"].clip(1e-4, 1e5) / y["claimed_lr"]) * np.sign(lv)
            ax.plot(y["percentile"], ratio, "o-", color=col, label=f"{'>=' if lv > 0 else '<='} {lv:+d}")
        ax.axhline(0, color="#a3a39d", lw=0.6, ls=":")
        ax.set_xlabel("percentile")
        ax.set_ylabel("log(observed / claimed LR), in the evidence's direction")
        if c == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_percentile_selection.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--percentiles", nargs="*", type=int, default=list(R.PERCENTILES))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    for p in args.percentiles:
        print(f"p{p}: {build(p, args.figures)}", flush=True)
    selection()


if __name__ == "__main__":
    main()
