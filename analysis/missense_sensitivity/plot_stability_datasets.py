"""E7 per-dataset figures: the calibrations behind each stability-rule call, for a priority list of datasets.

For each dataset, one figure with
  A  pathogenic evidence: log10 lr+ (5th bootstrap percentile, the conservative pathogenic curve) vs score
  B  evidence points vs score (the calibrated point thresholds; benign points negative)
  C  where the P/LP sit: missense vs non-missense P/LP scores (strip), so composition differences are visible
  D  the numbers each rule uses, and each rule's call
Arms: all P/LP (black), missense P/LP (orange), the random all-P/LP subsets of the missense size (thin blue).
Rows: E1 (baseline fits, P/LP weights re-estimated, components fixed); with --refit and the medium calibrations
present, a second row from full refits (baseline first 100 bootstraps / plpmis / plpsub seeds 0-2).

Usage:  python -m analysis.missense_sensitivity.plot_stability_datasets [--refit] [--datasets ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
KEEP = ROOT / "analysis_v3" / "e1_priority_calibrations"
OUTDIR = ROOT / "figs_stability_rules" / "per_dataset"
PLP = "Pathogenic/Likely Pathogenic"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_SUB, C_MIS = "#2a78d6", "#eb6834"

# priority list: why each dataset is worth looking at
PRIORITY = {
    "TP53_Giacomelli_2018_p53WT_Nutlin3_clinvar_2018": "composition, n large: missense P/LP give MORE evidence than all P/LP",
    "GCK_Gersing_2024_abundance": "composition by mean points only (-0.33); highest level unchanged",
    "PTEN_Matreyek_2018_clinvar_2018": "n-based rule says composition; missense sits at the edge of the subset range (count?)",
    "CHEK2_McCarthy-Leo_2024": "n = 7 (count-limited by n), yet random subsets are stable and missense moves a level",
    "BRCA2_IGVF": "n = 2: subsets unstable AND missense outside their range",
    "KCNE1_Muhammad_2024_trafficking_WT_background_DN": "n = 3: subsets span 0-8 points; missense 0 vs all P/LP 8",
    "KCNH2_Kozek_Glazer_2020": "n = 2 but every subset gives the same calls: n rule over-flags",
    "BRCA2_Sahu_2023_exon13_SGE": "n = 3, stable under subsets: n rule over-flags",
    "TSC2_IGVF": "n = 61 yet 1 of 10 subsets drops the top level: threshold at a level boundary",
    "BAP1_Waters_2024": "n = 11, just above the floor, unstable under subsets",
    "RAD51C_Olvera-León_2024": "n = 8: missense at the top of a wide subset range",
    "PALB2_IGVF": "n = 3: wide subset spread (0-4.7 mean points)",
    "LDLR_Tabet_2025_uptake": "control: n = 358, all arms agree",
    # no missense P/LP: the missense-only calibration runs in NU mode (refit row only)
    "XRCC2_IGVF": "all 6 P/LP non-missense: missense-only calibration becomes NU",
    "PALB2_Boonen_2026_SGE": "all 4 P/LP non-missense: missense-only calibration becomes NU",
}


def _load_arm(d):
    d = Path(d)
    try:
        cal = json.load(open(next(d.glob("*_calibration.json"))))
        lr = json.load(gzip.open(next(d.glob("*_lr_values.json.gz"))))
    except StopIteration:
        return None
    return cal, lr


def _points_curve(cal, x):
    from src.assay_calibration.fit_utils.fit import assign_points
    pr = {int(k): v for k, v in (cal.get("point_ranges") or {}).items()}
    return assign_points(np.asarray(x, float), pr)


def _plp_scores(ds):
    """Missense and non-missense P/LP scores, from the baseline variants table."""
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.compare_runs import _missense_map
    dv = pd.read_csv(next((ROOT / "calib_baseline" / ds).glob(f"{ds}_*_variants.csv")))
    df = _filter_dataset_df(load_master_df(_cfg.DATASET_TSV), ds, _cfg.DATASET_TSV)
    mis = set(_missense_map(ds, df).query("is_missense").variant_id)
    p = dv[dv["sample"].fillna("").str.contains(PLP, regex=False)]
    return p[p.variant_id.isin(mis)].score.values, p[~p.variant_id.isin(mis)].score.values


def _row(axs, arms, x_plp_m, x_plp_o, label):
    """arms: dict name -> (cal, lr) with names 'all', 'missense', 'subset_*'."""
    lo = min(np.min(a[1]["score_range"]) for a in arms.values())
    hi = max(np.max(a[1]["score_range"]) for a in arms.values())
    x = np.linspace(lo, hi, 600)
    for name, (cal, lr) in sorted(arms.items(), key=lambda kv: (kv[0] != "subset", kv[0])):
        sr = np.asarray(lr["score_range"], float)
        y = np.asarray(lr["log_lr_plus_p5"], float) / np.log(10)
        sub = name.startswith("subset")
        kw = dict(color=C_SUB, lw=0.8, alpha=0.55, zorder=2) if sub else \
            dict(color=C_MIS if name == "missense" else INK, lw=2, zorder=4 if name == "missense" else 3)
        axs[0].plot(sr, y, **kw)
        axs[1].step(x, _points_curve(cal, x), where="mid", **kw)
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[1].axhline(0, color=INK2, lw=0.8)
    axs[0].set_ylabel(f"{label}\nlog10 lr+ (5th pct)", fontsize=8)
    axs[1].set_ylabel("points", fontsize=8)
    rng = np.random.default_rng(0)
    axs[2].scatter(x_plp_m, rng.uniform(0.6, 1.4, len(x_plp_m)), s=10, color=C_MIS, alpha=0.8, label="missense P/LP")
    axs[2].scatter(x_plp_o, rng.uniform(-0.4, 0.4, len(x_plp_o)), s=10, color=INK2, alpha=0.6, label="other P/LP")
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels([f"other\n(n={len(x_plp_o)})", f"missense\n(n={len(x_plp_m)})"], fontsize=7.5)
    axs[2].set_ylim(-0.8, 1.8)
    for a in axs[:3]:
        a.set_xlim(lo, hi)
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)


def _rule_text(r):
    yn = lambda b: "FLAG" if b else "-"
    if r.n_mis == 0:
        return (f"P/LP: {int(r.n_plp)} all, 0 missense\n\nmissense-only calibration has no P/LP:\n"
                "NU mode, so its 'pathogenic' evidence means\nfunctionally abnormal, not pathogenic.\n\n"
                "No re-weighting or random subsets possible.\nCount rules: n < 5 FLAG, n < 10 FLAG\n\n"
                f"proposed: {r.proposed}")
    lines = [
        f"P/LP: {int(r.n_plp)} all, {int(r.n_mis)} missense",
        "",
        "mean points per missense P/LP",
        f"  all {r.mean_all:.2f}  missense {r.mean_mis:.2f}",
        f"  subsets {r.mean_lo:.2f}-{r.mean_hi:.2f} (median {r.mean_sub:.2f})",
        "highest pathogenic level",
        f"  all {r.lev_all:.0f}  missense {r.lev_mis:.0f}  subsets {r.lev_lo:.0f}-{r.lev_hi:.0f}",
        f"  subsets that change it: {int(r.lev_n_changed)}/{int(r.n_sub)}",
        "",
        "candidate rules",
        f"  n < 5                     {yn(r.n_mis < 5)}",
        f"  n < 10                    {yn(r.n_mis < 10)}",
        f"  missense moves level >= 1 {yn(abs(r.lev_mis - r.lev_all) >= 1)}",
        f"  missense moves mean >= 0.5 {yn(abs(r.mean_mis - r.mean_all) >= 0.5)}",
        "",
        "reference outcomes (random subsets)",
        f"  count-unstable            {yn(r.count_unstable)}",
        f"  composition               {yn(r.composition)}",
        "",
        f"proposed (n-based): {r.proposed}",
        f"empirical (subsets): {r.empirical}",
    ]
    return "\n".join(lines)


def draw_dataset(ds, r, why, refit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.missense_sensitivity.plot_stability_rules import display_name
    e1 = {a.name: _load_arm(a) for a in sorted((KEEP / ds).glob("*")) if a.is_dir()}
    e1 = {k: v for k, v in e1.items() if v}
    rows = [("E1 re-weighting", e1)] if e1 else []
    if refit:
        rf = {"all": _load_arm(ROOT / "calib_medium_baseline" / ds), "missense": _load_arm(ROOT / "calib_medium_plpmis" / ds)}
        for i in range(3):
            rf[f"subset_{i}"] = _load_arm(ROOT / f"calib_medium_plpsub_s{i}" / ds)
        rf = {k: v for k, v in rf.items() if v}
        if "all" in rf and "missense" in rf:
            rows.append(("full refit" + (" (missense = NU)" if r.n_mis == 0 else ""), rf))
    if not rows:
        return None
    xm, xo = _plp_scores(ds)
    fig = plt.figure(figsize=(16, 3.3 * len(rows) + 1.0))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[2.4, 2.4, 1.8, 2.0], wspace=0.32, hspace=0.35,
                          left=0.06, right=0.99, top=1 - 0.8 / (3.3 * len(rows) + 1.0), bottom=0.12 / len(rows))
    for i, (lab, arms) in enumerate(rows):
        axs = [fig.add_subplot(gs[i, j]) for j in range(3)]
        _row(axs, arms, xm, xo, lab)
        if i == 0:
            axs[0].set_title("A  pathogenic evidence curve", fontsize=9, loc="left")
            axs[1].set_title("B  calibrated points", fontsize=9, loc="left")
            axs[2].set_title("C  P/LP scores by variant type", fontsize=9, loc="left")
        if i == len(rows) - 1:
            for a in axs:
                a.set_xlabel("assay score", fontsize=8)
    ta = fig.add_subplot(gs[:, 3])
    ta.axis("off")
    ta.text(0, 1, _rule_text(r), va="top", ha="left", family="monospace", fontsize=7.6, transform=ta.transAxes)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, lw=2), Line2D([], [], color=C_MIS, lw=2), Line2D([], [], color=C_SUB, lw=0.8)]
    fig.legend(h, ["all P/LP", "missense P/LP", "random all-P/LP subsets, missense size"], loc="upper right",
               ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"{display_name(ds)}  -  {why}", x=0.06, ha="left", fontsize=10.5, y=1 - 0.15 / fig.get_figheight())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{display_name(ds)}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    from analysis.missense_sensitivity.plot_stability_rules import e1_table, classify
    t = e1_table().set_index("dataset", drop=False)
    for c in ["mean_all", "mean_mis", "mean_lo", "mean_hi", "mean_sub", "lev_all", "lev_mis", "lev_lo", "lev_hi"]:
        t[c] = t[c].astype(float)
    rules, prop = classify(t)
    t["count_unstable"], t["composition"], t["proposed"] = rules["count-unstable*"], rules["composition*"], prop
    t["empirical"] = np.select([t.count_unstable & t.composition, t.count_unstable, t.composition],
                               ["count-unstable + composition", "count-unstable", "composition"], "stable")
    for ds in args.datasets or PRIORITY:
        if not (KEEP / ds).exists() and not args.refit:
            print("no kept calibrations:", ds)
            continue
        print(draw_dataset(ds, t.loc[ds], PRIORITY.get(ds, ""), args.refit))


if __name__ == "__main__":
    main()
