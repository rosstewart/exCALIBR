"""Summaries of the calibration-quality simulation (results.csv from analysis.calibration_quality.tasks collect, and
acmgscaler_rows.csv from analysis.calibration_quality.acmgscaler_arm).

Outcomes per replicate (threshold mapping: Tavtigian, the manuscript's; lp_anchored columns carry an "lp_" prefix):
    prior unreliable     |fit prior - true prior| > 0.10
    fit unreliable       per-variant MAE of the fit's points vs the true-component pipeline (same bootstraps,
                         5th/95th-percentile rule) > 1 point, for truly pathogenic or truly benign variants: model-fit
                         error, with the intended percentile conservatism removed
    overstated           > 5% of truly pathogenic or truly benign variants get stronger evidence than the truth
    retention            mean |points| of the fit as a share of the truth's, per class (power)
The P/LP mechanism-mix bias cells are summarized separately (sample size cannot protect against that bias).

    python -m analysis.calibration_quality.summarize <out_dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRIOR_TOL = 0.10
FIT_MAE_TOL = 1.0
OVER_TOL = 0.05
P_OK = 0.10  # a scenario is "reliable" when at most 10% of its replicates are unreliable
GRID = ["kappa", "gamma", "richness", "n_plp"]
DIAGNOSTICS = {  # user-facing diagnostic -> column (higher = worse)
    "Yang distance, max over samples (held out)": "yang_val_max",
    "Yang distance, max over samples (in sample)": "yang_train_max",
    "SD of the prior across bootstraps": "fit_prior_sd",
    "IQR of the prior across bootstraps": "fit_prior_iqr",
    "share of bootstraps discarded": "fit_pct_discarded",
    "share of bootstraps at the EM starting value": "fit_frac_at_start",
}


def load(out_dir):
    df = pd.read_csv(Path(out_dir) / "results.csv", low_memory=False)
    df = df[df["status"] == "ok"].copy()
    return add_outcomes(df)


def add_outcomes(df, pre=""):
    # misspecified-truth replicates have no true-component arm: those columns are absent or all NaN
    for k in ("true_prior_err", f"{pre}fit_vs_true_pathogenic_mae", f"{pre}fit_vs_true_benign_mae"):
        if k not in df:
            df[k] = np.nan
    df[f"{pre}prior_bad"] = df["fit_prior_err"].abs().gt(PRIOR_TOL) | df["fit_prior_err"].isna()
    df[f"{pre}true_prior_bad"] = df["true_prior_err"].abs().gt(PRIOR_TOL) | df["true_prior_err"].isna()
    for k in ("pathogenic", "benign"):
        df[f"{pre}fit_bad_{k}"] = df[f"{pre}fit_vs_true_{k}_mae"] > FIT_MAE_TOL
        df[f"{pre}over_{k}"] = df[f"{pre}fit_vs_truth_{k}_over"] > OVER_TOL
        tm = df[f"{pre}truth_{k}_mean_abs"]
        df[f"{pre}retention_{k}"] = np.where(tm > 0, df[f"{pre}fit_vs_truth_{k}_mean_abs"] / tm, np.nan)
    df[f"{pre}fit_bad"] = df[f"{pre}fit_bad_pathogenic"] | df[f"{pre}fit_bad_benign"]
    df[f"{pre}overstated"] = df[f"{pre}over_pathogenic"] | df[f"{pre}over_benign"]
    df[f"{pre}any_bad"] = df["prior_bad"] | df[f"{pre}fit_bad"] | df[f"{pre}overstated"]
    if not pre:
        yv = [c for c in df if c.startswith("yang_val_")]
        yt = [c for c in df if c.startswith("yang_train_")]
        df["yang_val_max"] = df[yv].max(axis=1)
        df["yang_train_max"] = df[yt].max(axis=1)
        df["fit_pct_discarded"] = 1 - df["fit_pct_kept"]
    return df


def degradation(df, outcome):
    """Share of replicates with the outcome per core-grid cell (pooled over the two priors)."""
    c = df[df["experiment"] == "core"]
    return c.groupby(GRID)[outcome].mean().rename(outcome).reset_index()


def minimum_controls(df, outcome):
    """Smallest controls per class at which at most P_OK of replicates have the outcome, for each separation,
    measured fraction and population/synonymous richness (NaN: none of the simulated counts suffices)."""
    d = degradation(df, outcome)
    rows = []
    for key, g in d.groupby(["kappa", "gamma", "richness"]):
        ok = g.sort_values("n_plp")
        ok = ok[ok[outcome] <= P_OK]["n_plp"]
        rows.append(dict(zip(["kappa", "gamma", "richness"], key), min_controls=ok.min() if len(ok) else np.nan))
    return pd.DataFrame(rows)


def minimum_controls_retention(df, target=0.5):
    """Smallest controls per class at which the fit keeps at least `target` of the truth's evidence (ratio of cell
    totals, figures.retention_by_cell); NaN where no simulated count reaches it, or where the truth itself gives
    too little evidence to rate."""
    from analysis.calibration_quality.figures import retention_by_cell
    c = df[df["experiment"] == "core"]
    r = c.groupby(GRID).apply(retention_by_cell).rename("retention").reset_index()
    rows = []
    for key, g in r.groupby(["kappa", "gamma", "richness"]):
        ok = g[g["retention"] >= target]["n_plp"]
        rated = g["retention"].notna().any()
        rows.append(dict(zip(["kappa", "gamma", "richness"], key),
                         min_controls=ok.min() if len(ok) else (np.nan if rated else -1)))
    return pd.DataFrame(rows)


def latex_min_controls_table(prior_mc, ret_mc):
    """Minimum controls per class, by population/synonymous sample size, measured fraction gamma and separation
    kappa: for the prior (at most 10% of replicates more than 10 pp off) and for evidence (at least half of the
    truth's evidence retained). '>80': not reached with 80; '--': the truth itself gives too little evidence."""
    def cell(v):
        if v == -1:
            return "--"
        return "$>$80" if not np.isfinite(v) else f"{int(v)}"
    rich = [("low", "50 / 0"), ("mid", "200 / 100"), ("high", "1000 / 500")]
    ks = [1.5, 2.5, 4.0]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{Prior} & \multicolumn{3}{c}{Evidence} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Population / synonymous & $\gamma$ & " + " & ".join(f"$\\kappa={k:g}$" for k in ks * 2) + r" \\",
             r"\midrule"]
    for r, lab in rich:
        for i, g in enumerate([0.25, 0.5, 1.0]):
            pv = [prior_mc[(prior_mc.richness == r) & (prior_mc.gamma == g) & (prior_mc.kappa == k)]["min_controls"]
                  .iloc[0] for k in ks]
            ev = [ret_mc[(ret_mc.richness == r) & (ret_mc.gamma == g) & (ret_mc.kappa == k)]["min_controls"]
                  .iloc[0] for k in ks]
            lines.append(f"{lab if i == 0 else ''} & {g:g} & " + " & ".join(cell(v) for v in pv + ev) + r" \\")
        if r != "high":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def auc(score, label):
    """Area under the ROC curve of `score` (higher = predicts the label), NaN-safe."""
    m = np.isfinite(score)
    s, y = np.asarray(score)[m], np.asarray(label)[m].astype(bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def diagnostic_table(df, outcome, exclude=("mech_bias",)):
    d = df[~df["experiment"].isin(exclude)]
    rows = []
    for name, col in DIAGNOSTICS.items():
        rows.append(dict(diagnostic=name, column=col, auc=auc(d[col].to_numpy(float), d[outcome].to_numpy())))
    return pd.DataFrame(rows)


def threshold_table(df, outcome, col, cutoffs, exclude=("mech_bias",)):
    """Sensitivity, specificity and positive predictive value of `col > cutoff` for the outcome."""
    d = df[~df["experiment"].isin(exclude)]
    y = d[outcome].to_numpy(bool)
    x = d[col].to_numpy(float)
    rows = []
    for t in cutoffs:
        f = np.nan_to_num(x, nan=np.inf) > t  # a missing diagnostic (e.g. no prior) counts as flagged
        tp, fp = (f & y).sum(), (f & ~y).sum()
        rows.append(dict(cutoff=t, flagged=f.mean(), sensitivity=tp / max(y.sum(), 1),
                         specificity=((~f) & ~y).sum() / max((~y).sum(), 1), ppv=tp / max(tp + fp, 1)))
    return pd.DataFrame(rows)


def slices(df):
    out = ["prior_bad", "true_prior_bad", "fit_bad", "overstated", "retention_pathogenic", "retention_benign"]
    return df[df["experiment"] != "core"].groupby(["experiment", "n_plp", "n_blb", "kappa", "gamma", "lab_gamma",
                                                    "eps_p", "eta_b", "pi"])[out].mean().reset_index()


def mech_bias(df):
    d = df[df["experiment"] == "mech_bias"].copy()
    d["mix_shift"] = d["lab_gamma"] - d["gamma"]
    return d.groupby(["gamma", "lab_gamma", "kappa", "n_plp"])[
        ["fit_vs_truth_pathogenic_over", "fit_prior_err", "yang_val_max", "fit_prior_sd"]].mean().reset_index()


def acmgscaler_compare(out_dir, df):
    """acmgscaler (fixed prior 0.10) vs the ExCALIBR fit, both on the ACMG/AMP levels vs the truth on those levels."""
    f = Path(out_dir) / "acmgscaler_rows.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f, low_memory=False)
    key = ["cell_id", "rep"]
    keep = key + [c for c in a if c.startswith("acmg") or c.startswith("lp_acmg")]
    m = df.merge(a[keep], on=key, how="left")
    m["acmg_ok"] = m["acmg_status"] == "ok"
    for k in ("pathogenic", "benign"):
        m[f"acmg_over_{k}"] = m[f"acmg_vs_truth_{k}_tier_over"] > OVER_TOL
        m[f"exc_over_{k}"] = m[f"fit_vs_truth_{k}_tier_over"] > OVER_TOL
    g = m.groupby(["experiment", "pi", "n_plp"]).agg(
        acmg_calibratable=("acmg_ok", "mean"),
        acmg_mae_path=("acmg_vs_truth_pathogenic_tier_mae", "mean"),
        exc_mae_path=("fit_vs_truth_pathogenic_tier_mae", "mean"),
        acmg_mae_ben=("acmg_vs_truth_benign_tier_mae", "mean"),
        exc_mae_ben=("fit_vs_truth_benign_tier_mae", "mean"),
        acmg_over_path=("acmg_vs_truth_pathogenic_tier_over", "mean"),
        exc_over_path=("fit_vs_truth_pathogenic_tier_over", "mean"),
        acmg_over_ben=("acmg_vs_truth_benign_tier_over", "mean"),
        exc_over_ben=("fit_vs_truth_benign_tier_over", "mean")).reset_index()
    return m, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    df = load(out)
    pd.set_option("display.width", 250)
    print(f"{len(df)} replicates, {df['cell_id'].nunique()} cells")
    tabs = out / "summary"
    tabs.mkdir(exist_ok=True)
    for o in ("prior_bad", "true_prior_bad", "fit_bad", "overstated"):
        degradation(df, o).to_csv(tabs / f"degradation_{o}.csv", index=False)
        mc = minimum_controls(df, o)
        mc.to_csv(tabs / f"min_controls_{o}.csv", index=False)
        print(f"\nminimum controls per class with <= {P_OK:.0%} replicates '{o}':")
        print(mc.pivot_table(index=["richness", "gamma"], columns="kappa", values="min_controls").to_string())
    for o in ("prior_bad", "fit_bad"):
        dt = diagnostic_table(df, o)
        dt.to_csv(tabs / f"diagnostic_auc_{o}.csv", index=False)
        print(f"\ndiagnostic AUC for '{o}' (excluding mechanism-mix bias):")
        print(dt.round(3).to_string(index=False))
    yt = threshold_table(df, "fit_bad", "yang_val_max", [0.05, 0.1, 0.15, 0.2, 0.3])
    yt.to_csv(tabs / "threshold_yang_fit_bad.csv", index=False)
    print("\nheld-out Yang distance cut-offs vs fit unreliable:")
    print(yt.round(3).to_string(index=False))
    pt = threshold_table(df, "prior_bad", "fit_prior_sd", [0.02, 0.05, 0.1, 0.15, 0.2])
    pt.to_csv(tabs / "threshold_priorsd_prior_bad.csv", index=False)
    print("\nprior SD cut-offs vs prior unreliable:")
    print(pt.round(3).to_string(index=False))
    slices(df).to_csv(tabs / "slices.csv", index=False)
    mech_bias(df).to_csv(tabs / "mech_bias.csv", index=False)
    r = acmgscaler_compare(out, df)
    if r is not None:
        r[1].to_csv(tabs / "acmgscaler_vs_excalibr.csv", index=False)
        print("\nacmgscaler vs ExCALIBR (ACMG/AMP levels vs truth), prior slice:")
        print(r[1][r[1]["experiment"].isin(["prior", "core"])].groupby(["pi"]).mean(numeric_only=True).round(3)
              .to_string())
    print(f"\ntables -> {tabs}")


if __name__ == "__main__":
    main()
