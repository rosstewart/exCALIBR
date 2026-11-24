"""Summary tables and figures for the consequence-cell simulation.

    python -m analysis.consequence_cells_sim.summarize <results.csv> [--out <dir>]

Tables (CSV in <out>):
  arm_metrics.csv     per arm x evaluation consequence x assay x size x mix x kappa: replicate medians of |prior err|
                      (pp), posterior MAE, overstatement share, |log LR| error (all / tails), points over / wrong
  headline.csv        the same pooled over the core grid, per arm x consequence x assay
  min_cells.csv       per arm x consequence x assay: the smallest size class at which |prior err| <= 5 pp in >= 90%
                      of replicates, with the corresponding class-c P/LP and population cell sizes
Figures: reliability_<csq>.png (observed pathogenic share vs reported posterior, by arm), prior_err_vs_size.png.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["prior_err", "post_mae", "post_over", "llr_mae", "llr_mae_tail", "pts_over", "pts_wrong"]
ARM_ORDER = ["pooled", "mis", "cell_all", "cell_mis", "cell_mis_nosplsyn", "cell_synonymous", "cell_missense",
             "cell_PTV", "nu_synonymous", "nu_missense", "nu_PTV", "poolP_synonymous", "poolP_missense", "poolP_PTV",
             "true_cell_mis", "true_cell_synonymous", "true_cell_missense", "true_cell_PTV", "mm_ownprior_poolLR",
             "mm_poolprior_ownLR"]
SIZE_ORDER = ["small", "mid", "large"]


def long_form(df):
    rows = []
    pat = re.compile(r"^(?P<arm>.+?)__(?P<csq>synonymous|missense|PTV)_(?P<m>.+)$")
    keep = [c for c in df.columns if pat.match(c) and pat.match(c).group("m") in METRICS]
    base = ["cell_id", "experiment", "assay", "size", "mix", "kappa", "rep"]
    m = df[base + keep].melt(id_vars=base, var_name="col", value_name="v")
    g = m["col"].str.extract(pat)
    m = pd.concat([m.drop(columns="col"), g], axis=1)
    m["v"] = pd.to_numeric(m["v"], errors="coerce")
    w = m.pivot_table(index=base + ["arm", "csq"], columns="m", values="v").reset_index()
    if "prior_err" in w:
        w["abs_prior_err_pp"] = 100 * w["prior_err"].abs()
    return w


def reliability(df, csq, arms, out):
    edges = np.array([0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995,
                      0.999, 1.0])
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for arm in arms:
        cn, cp = f"{arm}__{csq}_rel_n", f"{arm}__{csq}_rel_path"
        if cn not in df:
            continue
        n = np.zeros(len(mid)); p = np.zeros(len(mid))
        for a, b in zip(df[cn].dropna(), df[cp].dropna()):
            n += np.array(str(a).split(";"), float); p += np.array(str(b).split(";"), float)
        ok = n >= 20
        if ok.any():
            ax.plot(mid[ok], p[ok] / n[ok], "o-", ms=3, lw=1.2, label=arm)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    ax.set_xlabel("reported posterior (bin midpoint)"); ax.set_ylabel("observed pathogenic share")
    ax.set_title(f"Reliability, {csq} population draws (bins with >= 20 variants)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/reliability_{csq}.png", dpi=130); plt.close(fig)


def main(res, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(res, low_memory=False)
    print(f"{len(df)} rows; status ok: {(df.status == 'ok').sum()}")
    df = df[df.status == "ok"]
    w = long_form(df)
    agg = {m: "median" for m in METRICS + ["abs_prior_err_pp"] if m in w}
    w.groupby(["arm", "csq", "assay", "size", "mix", "kappa"]).agg(n=("rep", "size"), **{k: (k, v) for k, v in agg.items()}) \
        .reset_index().to_csv(f"{out}/arm_metrics.csv", index=False)
    core = w[w.experiment.eq("core")] if w.experiment.eq("core").any() else w
    head = core.groupby(["csq", "assay", "arm"]).agg(n=("rep", "size"), **{k: (k, v) for k, v in agg.items()}).reset_index()
    head["arm"] = pd.Categorical(head["arm"], [a for a in ARM_ORDER if a in set(head.arm)] +
                                 sorted(set(head.arm) - set(ARM_ORDER)), ordered=True)
    head = head.sort_values(["csq", "assay", "arm"]); head.to_csv(f"{out}/headline.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    print(head.round(4).to_string(index=False))
    mins = []
    for (arm, csq, assay), x in core.groupby(["arm", "csq", "assay"]):
        best = None
        for sz in SIZE_ORDER:
            y = x[x["size"] == sz]["abs_prior_err_pp"].dropna()
            if len(y) and (y <= 5).mean() >= 0.9:
                best = sz; break
        mins.append(dict(arm=arm, csq=csq, assay=assay, min_size_prior_within_5pp=best or "none"))
    pd.DataFrame(mins).to_csv(f"{out}/min_cells.csv", index=False)
    for csq in ("missense", "PTV", "synonymous"):
        arms = ["pooled", "mis", "cell_all", "cell_mis", f"cell_{csq}", f"nu_{csq}", f"poolP_{csq}", f"true_cell_{csq}"]
        reliability(df, csq, arms, out)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, csq in zip(axs, ("missense", "PTV", "synonymous")):
        for arm in ["pooled", "mis", "cell_all", f"cell_{csq}", f"nu_{csq}", f"poolP_{csq}"]:
            y = core[(core.arm == arm) & (core.csq == csq)].groupby("size")["abs_prior_err_pp"].median().reindex(SIZE_ORDER)
            if y.notna().any():
                ax.plot(SIZE_ORDER, y.values, "o-", ms=3, label=arm)
        ax.set_title(csq, fontsize=9); ax.set_xlabel("size class"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False)
    axs[0].set_ylabel("median |prior error| (pp)")
    fig.tight_layout(); fig.savefig(f"{out}/prior_err_vs_size.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.results, a.out or os.path.join(os.path.dirname(os.path.abspath(a.results)), "summary"))
