"""E7 figure: per dataset, what each candidate stability rule says for missense-restricted P/LP evidence,
next to the evidence it is judged on and the proposed two-part rule.

Rows = datasets with non-missense P/LP, sorted by the number of missense P/LP.
Columns:
  A  number of P/LP, all (light) and missense (dark); log scale; n = 5 and n = 10 marked
  B  mean pathogenic points per missense P/LP: all P/LP (black), missense P/LP (orange),
     range over random all-P/LP subsets of the same size (blue bar). Source: E1 re-weighting
     of the baseline fits (components fixed); with --refit, a second row of marks from the
     medium full refits (plpmis vs plpsub seeds 0-2, vs the first 100 baseline bootstraps).
  C  highest pathogenic level reached, same encoding
  D  rule matrix: each candidate rule's flag (filled = flagged), and the two reference
     outcomes it is judged against:
       count-unstable  = a random subset moves the highest level by >= 1 or the mean points by >= 0.5
       composition     = the missense value lies outside the random-subset range and differs from it
                         by >= 0.25 points (or >= 1 level)
  E  proposed rule: count-limited (n < N_FLOOR), else composition-discordant (missense moves the
     highest level by >= 1, or the mean points by >= 0.25 outside the subset range), else stable.

Usage:  python -m analysis.missense_sensitivity.plot_stability_rules [--refit]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
OUTDIR = ROOT / "figs_stability_rules"
N_FLOOR = 10
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_SUB, C_MIS = "#2a78d6", "#eb6834"
CAT = {"no missense P/LP (NU)": "#52514e", "count-limited": "#eda100", "composition-discordant": "#6250d6", "stable": "#d9d8d3"}


def display_name(ds):
    try:
        from analysis.plot_common import display_name as dn
        return dn(ds)
    except Exception:
        return ds.replace("_clinvar_2018", "")


def e1_table():
    parts = sorted((ROOT / "analysis_v3" / "e1_matched_subsample_parts").glob("*.csv"))
    d = pd.concat([pd.read_csv(f) for f in parts if f.stat().st_size > 2], ignore_index=True)  # keeps the skip rows
    # datasets whose P/LP are all non-missense: the missense-only calibration has no P/LP (NU mode), so
    # there is nothing to re-weight or subsample; listed as their own category
    nu = d[(d.arm == "skip") & (d.n_plp_missense == 0)]
    d = d[d.arm != "skip"]
    rows = [dict(dataset=r.dataset, n_mis=0, n_plp=int(r.n_plp), n_sub=0, lev_n_changed=0) for r in nu.itertuples()]
    for ds, g in d.groupby("dataset"):
        a, m = g[g.arm == "all"].iloc[0], g[g.arm == "missense"].iloc[0]
        s = g[g.arm.str.startswith("subset_")]
        rows.append(dict(dataset=ds, n_mis=int(m.n_plp_missense), n_plp=int(m.n_plp),
                         mean_all=a.plp_mean_path, mean_mis=m.plp_mean_path,
                         mean_lo=s.plp_mean_path.min(), mean_hi=s.plp_mean_path.max(),
                         mean_sub=s.plp_mean_path.median(),
                         lev_all=a.max_path, lev_mis=m.max_path, lev_lo=s.max_path.min(), lev_hi=s.max_path.max(),
                         n_sub=len(s), lev_n_changed=int((s.max_path != a.max_path).sum())))
    return pd.DataFrame(rows)


def refit_table():
    """Medium full refits: missense-only P/LP vs three matched-count random all-P/LP subsets, scored like E1
    (in-bag points of the dataset's missense P/LP). Missing calibrations -> NaN."""
    from analysis.missense_sensitivity.matched_subsample import _score
    from analysis.missense_sensitivity.compare_runs import _find_outputs, _missense_map
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    df_full = load_master_df(_cfg.DATASET_TSV)
    out = []
    for cdir in sorted(glob.glob(str(ROOT / "calib_medium_baseline" / "*"))):
        ds = Path(cdir).name
        cals = {}
        for run in ["baseline", "plpmis", "plpsub_s0", "plpsub_s1", "plpsub_s2"]:
            c, _ = _find_outputs(ROOT / f"calib_medium_{run}", ds)
            cals[run] = json.load(open(c)) if c else None
        if any(v is None for v in cals.values()):
            continue
        dfd = _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV)
        mm = _missense_map(ds, dfd)
        var = next(Path(cdir).glob(f"{ds}_*_variants.csv"))
        dv = pd.read_csv(var)
        s = dv["sample"].fillna("")
        mis_ids = set(mm.loc[mm.is_missense, "variant_id"])
        plp = dv[dv.variant_id.isin(mis_ids) & s.str.contains("Pathogenic/Likely Pathogenic", regex=False)]
        sc = plp["score"].values
        r = {k: _score(v, sc, []) for k, v in cals.items()}
        subs_m = [r[f"plpsub_s{i}"]["plp_mean_path"] for i in range(3)]
        subs_l = [r[f"plpsub_s{i}"]["max_path"] for i in range(3)]
        out.append(dict(dataset=ds, rf_mean_all=r["baseline"]["plp_mean_path"], rf_mean_mis=r["plpmis"]["plp_mean_path"],
                        rf_mean_lo=np.nanmin(subs_m), rf_mean_hi=np.nanmax(subs_m),
                        rf_lev_all=r["baseline"]["max_path"], rf_lev_mis=r["plpmis"]["max_path"],
                        rf_lev_lo=min(subs_l), rf_lev_hi=max(subs_l)))
    return pd.DataFrame(out)


def classify(t, p=""):
    """Rules and outcomes from columns with prefix p ('' = E1 re-weighting, 'rf_' = refits)."""
    g = lambda k: t[p + k]
    d_lev = g("lev_mis") - g("lev_all")
    d_mean = g("mean_mis") - g("mean_all")
    out_sub = (g("mean_mis") < g("mean_lo") - 1e-9) | (g("mean_mis") > g("mean_hi") + 1e-9)
    dist_sub = np.maximum(g("mean_lo") - g("mean_mis"), g("mean_mis") - g("mean_hi")).clip(lower=0)
    lev_out = (g("lev_mis") < g("lev_lo")) | (g("lev_mis") > g("lev_hi"))
    r = pd.DataFrame(index=t.index)
    r["n < 5"] = t.n_mis < 5
    r["n < 10"] = t.n_mis < 10
    r["moves level ≥ 1"] = d_lev.abs() >= 1
    r["moves mean ≥ 0.5"] = d_mean.abs() >= 0.5
    r["count-unstable*"] = ((g("lev_hi") - g("lev_lo")) >= 1) | ((g("mean_hi") - g("mean_lo")) >= 0.5) \
        | ((g("mean_hi") - g("mean_all")).abs() >= 0.5) | ((g("mean_lo") - g("mean_all")).abs() >= 0.5)
    r["composition*"] = (out_sub & (dist_sub >= 0.25)) | (lev_out & (d_lev.abs() >= 1))
    disc = (d_lev.abs() >= 1) | (out_sub & (d_mean.abs() >= 0.25))
    prop = np.where(t.n_mis == 0, "no missense P/LP (NU)", np.where(t.n_mis < N_FLOOR, "count-limited",
                                                 np.where(disc, "composition-discordant", "stable")))
    return r, pd.Series(prop, index=t.index)


def draw(t, refit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    t = t.sort_values(["n_mis", "dataset"]).reset_index(drop=True)
    rules, prop = classify(t)
    have_rf = refit and "rf_mean_mis" in t and t.rf_mean_mis.notna().any()
    if have_rf:
        rules_rf, prop_rf = classify(t, "rf_")
    n = len(t)
    fig = plt.figure(figsize=(17, 0.26 * n + 3.0))
    widths = [1.3, 2.6, 2.0, 3.0, 1.6] + ([1.6] if have_rf else [])
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.08, left=0.2, right=0.99, top=1 - 1.9 / (0.26 * n + 3.0),
                          bottom=0.75 / (0.26 * n + 3.0))
    ax = [fig.add_subplot(gs[0, i]) for i in range(len(widths))]
    y = np.arange(n)
    for a in ax:
        a.set_ylim(n - 0.5, -0.5)
        a.tick_params(axis="y", length=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for yy in y[::2]:
            a.axhspan(yy - 0.5, yy + 0.5, color="#f4f3f0", zorder=0, lw=0)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([display_name(d) for d in t.dataset], fontsize=7.5, color=INK)
    for a in ax[1:]:
        a.set_yticks([])
    # A: counts
    ax[0].barh(y, t.n_plp, color="#c9c8c2", height=0.7, label="all P/LP")
    ax[0].barh(y, t.n_mis, color=INK2, height=0.4, label="missense P/LP")
    ax[0].set_xscale("log")
    ax[0].legend(loc="lower right", fontsize=7, frameon=False, handlelength=1)
    for v in (5, 10):
        ax[0].axvline(v, color=INK2, ls=":", lw=1)
    ax[0].set_title("A  P/LP (n)", fontsize=9, loc="left", color=INK)
    # B, C: value panels
    for a, k, title in [(ax[1], "mean", "B  mean pathogenic points per missense P/LP"), (ax[2], "lev", "C  highest pathogenic level")]:
        off = 0.17 if have_rf else 0.0
        a.hlines(y - off, t[f"{k}_lo"], t[f"{k}_hi"], color=C_SUB, lw=5, alpha=0.45, zorder=2, label="random subsets, same n (range)")
        a.scatter(t[f"{k}_all"], y - off, s=18, color=INK, zorder=3, label="all P/LP")
        a.scatter(t[f"{k}_mis"], y - off, s=22, color=C_MIS, edgecolor="white", lw=0.6, zorder=4, label="missense P/LP")
        if have_rf:
            a.hlines(y + off, t[f"rf_{k}_lo"], t[f"rf_{k}_hi"], color=C_SUB, lw=5, alpha=0.45, zorder=2)
            a.scatter(t[f"rf_{k}_all"], y + off, s=18, color=INK, marker="s", zorder=3)
            a.scatter(t[f"rf_{k}_mis"], y + off, s=22, color=C_MIS, marker="s", edgecolor="white", lw=0.6, zorder=4)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.set_title(title, fontsize=9, loc="left", color=INK)
    for i in np.flatnonzero(t.n_mis.values == 0):
        ax[1].text(0.02, i, "no missense P/LP: missense-only calibration is NU (no pathogenic controls)",
                   transform=ax[1].get_yaxis_transform(), va="center", fontsize=7, color=INK2)
    ax[2].set_xticks(range(0, 9))
    h, l = ax[1].get_legend_handles_labels()
    if have_rf:
        from matplotlib.lines import Line2D
        h += [Line2D([], [], marker="o", ls="", color=INK2), Line2D([], [], marker="s", ls="", color=INK2)]
        l += ["E1 re-weighting (upper)", "full refit (lower)"]
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.2, 1 - 0.15 / fig.get_figheight()), fontsize=8, frameon=False,
               ncol=len(h), handletextpad=0.3, columnspacing=1.2)
    # D: rule matrix
    cols = list(rules.columns)
    a = ax[3]
    for j, c in enumerate(cols):
        ref = c.endswith("*")
        for i in range(n):
            f = bool(rules[c].iloc[i])
            a.add_patch(Rectangle((j - 0.32, i - 0.32), 0.64, 0.64, facecolor=(C_SUB if ref else INK2) if f else "white",
                                  edgecolor=C_SUB if ref else INK2, lw=0.8, zorder=3))
    a.set_xlim(-0.6, len(cols) - 0.4)
    a.set_xticks(range(len(cols)))
    a.set_xticklabels(cols, rotation=35, ha="left", fontsize=7.5)
    a.xaxis.tick_top()
    a.tick_params(axis="x", length=0)
    a.axvline(3.5, color=INK2, lw=0.8)
    a.text(0, -0.5 / n, "D  candidate rules (gray) | reference outcomes (blue)", transform=a.transAxes, fontsize=9,
           color=INK, ha="left", va="top")
    a.xaxis.set_label_position("top")
    # E: proposed
    for col, pr, ttl in [(4, prop, f"E  proposed (E1)")] + ([(5, prop_rf, "F  proposed (refit)")] if have_rf else []):
        a = ax[col]
        for i, v in enumerate(pr):
            a.add_patch(Rectangle((0.02, i - 0.4), 0.96, 0.8, facecolor=CAT[v], edgecolor="none", zorder=3))
            a.text(0.5, i, v, ha="center", va="center", fontsize=6.8, zorder=4,
                   color="white" if v in ("composition-discordant", "no missense P/LP (NU)") else INK)
        a.set_xlim(0, 1)
        a.set_xticks([])
        a.set_title(ttl, fontsize=9, loc="left", color=INK)
    for a in ax[1:3]:
        a.xaxis.tick_top()
        a.tick_params(axis="x", labelsize=7.5)
    ax[0].xaxis.tick_top()
    ax[0].tick_params(axis="x", labelsize=7.5)
    for a in ax[:3]:
        a.title.set_position((0, 1 + 40 / (fig.get_figheight() * 72)))
    # precision / recall footer
    lines = []
    for nm, rr, pp in [("E1 re-weighting", rules, prop)] + ([("refit", rules_rf, prop_rf)] if have_rf else []):
        tgt = rr["count-unstable*"] | rr["composition*"]
        flag_prop = pp != "stable"
        parts = []
        for c in cols[:4]:
            f = rr[c]
            parts.append(f"{c}: P {(f & tgt).sum() / max(f.sum(), 1):.2f} R {(f & tgt).sum() / max(tgt.sum(), 1):.2f}")
        parts.append(f"proposed: P {(flag_prop & tgt).sum() / max(flag_prop.sum(), 1):.2f} "
                     f"R {(flag_prop & tgt).sum() / max(tgt.sum(), 1):.2f}")
        lines.append(f"{nm} (target = count-unstable or composition, {int(tgt.sum())}/{n}):  " + " | ".join(parts))
    fig.text(0.01, 0.004, "\n".join(lines) + "\n* reference outcomes from the random matched-size subsets (E1: 10 subsets, "
             "components fixed; refit: 3 subsets, full refits, 100 bootstraps). Points are in-bag, for the dataset's missense P/LP.",
             fontsize=7, color=INK2, va="bottom")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tab = pd.concat([t, rules.add_prefix("rule_"), prop.rename("proposed")], axis=1)
    if have_rf:
        tab = pd.concat([tab, rules_rf.add_prefix("rf_rule_"), prop_rf.rename("rf_proposed")], axis=1)
    tab.to_csv(path.with_suffix(".csv"), index=False)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    t = e1_table()
    if args.refit:
        t = t.merge(refit_table(), on="dataset", how="left")
    tab = draw(t, args.refit, OUTDIR / ("stability_rules_refit.png" if args.refit else "stability_rules_e1.png"))
    print(tab.proposed.value_counts().to_string())


if __name__ == "__main__":
    main()
