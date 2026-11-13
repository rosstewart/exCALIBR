"""Is the joint model's weaker predictor evidence a correction of VEP
overcalling, or overconservatism?

For the predictor and combined panels, on the paper's evaluation controls:

  calibration     for each method and claimed evidence level (points floored to
                  +-1, 2, 4, 8), the likelihood ratio the controls actually show
                  at that level (P/LP vs B/LB odds at the level over their odds
                  overall, pooled over genes Mantel-Haenszel style; 95% CI from
                  resampling controls within gene) against the LR the level
                  claims (c^(level/8) at the method's prior). Cumulative levels
                  (>= k, <= -k) are reported too, as OddsPath does. Methods:
                  joint out-of-bag at each percentile in review_common.PERCENTILES;
                  Chen et al.'s gene-specific calibration of each predictor alone;
                  their nonconflicting combination (the paper's comparator);
                  the strongest single predictor ("UV max"). Chen's points are not
                  out of bag: their labelled sets overlap these controls.
  discordance     controls grouped by (comparator sign, joint sign): the LR each
                  group shows. Where the comparator gives pathogenic evidence and
                  the joint model none, an LR near 1 means the joint model was
                  right to withhold it; an LR well above the comparator's claim
                  means it was overconservative.
  population      share of gnomAD variants given pathogenic evidence per method
                  (joint in-sample: gnomAD variants are fitted, not held out).

Outputs: review/vep_calibration/{calibration,discordance,population}.csv and
fig_vep_calibration.pdf/png.

Usage: python -m mv_analysis.review_vep_calibration
"""
import warnings

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

N_BOOT = 1000
PANELS = ("predictor", "combined")
CHEN_SINGLE = {"REVEL": "indep_REVEL", "AlphaMissense": "indep_AM", "MutPred2": "indep_MP2"}


def control_table(panel, percentiles=R.PERCENTILES):
    """One row per evaluation control: gene, label, joint out-of-bag points at
    each percentile (and its out-of-bag prior), comparator points."""
    sp = R.strength_points()
    sp = sp[(sp["panel"] == panel) & sp["sample"].isin(["P/LP", "B/LB"])]
    rows = []
    for key, m in R.paper_keys(panel):
        o = R.oob_points(key, percentiles[0]).rename(columns={"points": f"joint_p{percentiles[0]}"})
        for p in percentiles[1:]:
            o[f"joint_p{p}"] = R.oob_points(key, p)["points"].to_numpy()
        o = o[o["n_oob"] > 0]
        s = sp[sp["key"] == key].drop_duplicates("variant").set_index("variant")
        o = o.join(s[["indep_points"] + list(CHEN_SINGLE.values())], on="variant")
        if o["indep_points"].isna().all():
            print(f"  {key}: not in strength_points (no comparator points); left out", flush=True)
            continue
        assert o["indep_points"].notna().all(), f"{key}: some controls missing from strength_points"
        assert ((o["label"] == 1) == (s.loc[o["variant"], "sample"].to_numpy() == "P/LP")).all(), f"{key}: labels differ"
        o.insert(0, "gene", m["gene"] + (f"_c{m['cluster_idx']}" if m["cluster_idx"] else ""))
        o.insert(1, "key", key)
        rows.append(o)
    t = pd.concat(rows, ignore_index=True)
    single = t[list(CHEN_SINGLE.values())].to_numpy(float)
    idx = np.nanargmax(np.where(np.isnan(single), -1, np.abs(single)), axis=1)
    t["uv_max"] = np.where(np.isnan(single).all(1), 0, single[np.arange(len(t)), idx])
    return t


def _methods(t, panel):
    m = {f"joint (p{p})": (f"joint_p{p}", "oob") for p in R.PERCENTILES}
    m["nonconflicting comparator"] = ("indep_points", "chen")
    if panel == "predictor":
        m.update({f"{k} alone (Chen)": (c, "chen") for k, c in CHEN_SINGLE.items()})
        m["strongest predictor (Chen)"] = ("uv_max", "chen")
    return m


def _claimed(level, prior):
    """LR a level claims at each prior (array): c(prior)^(level/8)."""
    lc = np.array([R.log_c(round(float(p), 4)) for p in np.atleast_1d(prior)])
    return np.exp(level / 8 * lc)


def _counts(gene_codes, y, mask, n_genes):
    n_p = np.bincount(gene_codes[mask & y], minlength=n_genes)
    n_b = np.bincount(gene_codes[mask & ~y], minlength=n_genes)
    return n_p, n_b


def calibration(t, panel, seed=0):
    rng = np.random.default_rng(seed)
    gcode, genes = pd.factorize(t["gene"])
    G = len(genes)
    y = t["label"].to_numpy() == 1
    P, B = np.bincount(gcode[y], minlength=G), np.bincount(gcode[~y], minlength=G)
    by_gene = [np.flatnonzero(gcode == g) for g in range(G)]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in by_gene]) for _ in range(N_BOOT)]
    recs = []
    for name, (col, kind) in _methods(t, panel).items():
        pts = R.floor_level(t[col].to_numpy(float))
        prior = t["oob_prior"].to_numpy(float) if kind == "oob" else np.full(len(t), R.GENOME_WIDE_PRIOR)
        for lv in (-8, -4, -2, -1, 1, 2, 4, 8):
            for scope in ("level", "cumulative"):
                if scope == "level":
                    w = pts == lv
                else:
                    w = pts >= lv if lv > 0 else pts <= lv
                if not w.any():
                    continue
                n_p, n_b = _counts(gcode, y, w, G)
                lr = R.mh_lr(n_p, n_b, P, B)
                bs = []
                for b in boots:
                    yb, wb, gb = y[b], w[b], gcode[b]
                    bp, bb = _counts(gb, yb, wb, G)
                    bs.append(R.mh_lr(bp, bb, np.bincount(gb[yb], minlength=G), np.bincount(gb[~yb], minlength=G)))
                bs = np.asarray(bs, float)
                bs = bs[np.isfinite(bs) & (bs > 0)]
                claimed = float(np.exp(np.mean(np.log(_claimed(lv, prior[w])))))
                recs.append(dict(panel=panel, method=name, level=lv, scope=scope, n_plp=int(n_p.sum()),
                                 n_blb=int(n_b.sum()), observed_lr=lr,
                                 lr_lo=np.percentile(bs, 2.5) if len(bs) else np.nan,
                                 lr_hi=np.percentile(bs, 97.5) if len(bs) else np.nan,
                                 claimed_lr=claimed,
                                 overstated=(lv > 0 and len(bs) and np.percentile(bs, 97.5) < claimed)
                                 or (lv < 0 and len(bs) and np.percentile(bs, 2.5) > claimed)))
    return pd.DataFrame(recs)


def discordance(t, panel, p=5):
    j = np.sign(R.floor_level(t[f"joint_p{p}"]))
    c = np.sign(R.floor_level(t["indep_points"]))
    gcode, genes = pd.factorize(t["gene"])
    G = len(genes)
    y = t["label"].to_numpy() == 1
    P, B = np.bincount(gcode[y], minlength=G), np.bincount(gcode[~y], minlength=G)
    name = {1: "pathogenic", 0: "none", -1: "benign"}
    recs = []
    for cs in (1, 0, -1):
        for js in (1, 0, -1):
            w = (c == cs) & (j == js)
            if not w.any():
                continue
            n_p, n_b = _counts(gcode, y, w, G)
            recs.append(dict(panel=panel, percentile=p, comparator=name[cs], joint=name[js],
                             n_plp=int(n_p.sum()), n_blb=int(n_b.sum()),
                             observed_lr=R.mh_lr(n_p, n_b, P, B)))
    return pd.DataFrame(recs)


def population(panel):
    sp = R.strength_points()
    g = sp[(sp["panel"] == panel) & (sp["sample"] == "gnomAD")]
    keys = {k for k, _ in R.paper_keys(panel)}
    g = g[g["key"].isin(keys)]
    if R.ALT_FITS:  # the alternative fits' in-sample points (p = 5) for the joint model
        parts = []
        for key, x in g.groupby("key"):
            d, _ = R.prep(key)
            pos = {str(v): i for i, v in enumerate(d["variants"])}
            j = x["variant"].astype(str).map(pos)
            x = x[j.notna()].copy()
            x["joint_points"] = R.insample_paper_points(key)[j.dropna().astype(int).to_numpy()]
            parts.append(x)
        g = pd.concat(parts, ignore_index=True) if parts else g.iloc[:0]
    cols ={"joint (in-sample, p5)": "joint_points", "nonconflicting comparator": "indep_points"}
    if panel == "predictor":
        cols.update({f"{k} alone (Chen)": c for k, c in CHEN_SINGLE.items()})
    recs = []
    for gene, x in [("all", g)] + list(g.groupby("gene")):
        for name, col in cols.items():
            v = x[col].to_numpy(float)
            recs.append(dict(panel=panel, gene=gene, method=name, n=int(np.isfinite(v).sum()),
                             frac_pathogenic=float(np.nanmean(v >= 1)), frac_strong=float(np.nanmean(v >= 4))))
    return pd.DataFrame(recs)


def figure(cal, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["joint (p5)", "joint (p10)", "joint (p25)", "joint (p50)", "nonconflicting comparator",
            "REVEL alone (Chen)", "strongest predictor (Chen)"]
    colors = dict(zip(show, ["#1d1d1b", "#4a4a46", "#7a7a74", "#a3a39d", "#d55e00", "#cc79a7", "#e69f00"]))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for r, panel in enumerate(PANELS):
        for c, scope in enumerate(("level", "cumulative")):
            ax = axes[r, c]
            d = cal[(cal["panel"] == panel) & (cal["scope"] == scope)]
            for k, mth in enumerate([m for m in show if m in set(d["method"])]):
                x = d[d["method"] == mth]
                off = (k - len(show) / 2) * 0.05
                ax.errorbar(np.log(x["claimed_lr"]) + off, np.log(x["observed_lr"].clip(1e-3, 1e4)),
                            yerr=[np.log(x["observed_lr"].clip(1e-3, 1e4)) - np.log(x["lr_lo"].clip(1e-3, 1e4)),
                                  np.log(x["lr_hi"].clip(1e-3, 1e4)) - np.log(x["observed_lr"].clip(1e-3, 1e4))],
                            fmt="o", ms=3, lw=0.6, color=colors[mth], label=mth)
            lim = [-8, 8]
            ax.plot(lim, lim, color="#a3a39d", lw=0.6, ls=":")
            ax.set_xlabel("claimed log LR (level threshold)")
            ax.set_ylabel("observed log LR among controls")
            ax.set_title(f"{panel}: {'each level' if scope == 'level' else 'cumulative (>= k / <= -k)'}", fontsize=8)
            ax.axhline(0, color="#e4e4df", lw=0.5)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_vep_calibration.{ext}", dpi=200)
    plt.close(fig)


def run():
    warnings.filterwarnings("ignore")
    out = R.out_dir("vep_calibration")
    cal, dis, pop = [], [], []
    for panel in PANELS:
        t = control_table(panel)
        t.to_csv(out / f"controls_{panel}.csv", index=False)
        cal.append(calibration(t, panel))
        dis.append(pd.concat([discordance(t, panel, p) for p in R.PERCENTILES]))
        pop.append(population(panel))
    cal, dis, pop = (pd.concat(x, ignore_index=True) for x in (cal, dis, pop))
    cal.to_csv(out / "calibration.csv", index=False)
    dis.to_csv(out / "discordance.csv", index=False)
    pop.to_csv(out / "population.csv", index=False)
    figure(cal, out)
    print(f"wrote {out}")
    return cal, dis, pop


if __name__ == "__main__":
    run()
