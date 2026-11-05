"""Shared summaries of the paired prior-simulation results (test_priors.py
outputs), used by the report page (report.py), the supplementary figure
(figures.py) and the manuscript build (analyze_pipeline_output.py A11h), so
every number quoted anywhere is computed once, the same way.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from analysis.prior_simulation.simulate_priors import ASSAYS, DEFAULT_OUT_DIR

KEYS = ["experiment", "regime", "true_prior", "delta", "rho", "eps_p", "delta_b", "lab_m1_frac", "assay"]
A1, A2, A3 = ASSAYS


def mean_ci(x):
    """Mean, 95% t-interval and n of the finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    m = x.mean()
    h = sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h, len(x)


def summarize(reps):
    """One row per setting and assay: mean [95% CI] for each estimator."""
    rows = []
    for key, g in reps.groupby(KEYS, sort=False):
        o, f = mean_ci(g["oracle"]), mean_ci(g["fit"])
        rows.append({**dict(zip(KEYS, key)), "expected": g["expected_prior"].iloc[0],
                     "o_m": o[0], "o_lo": o[1], "o_hi": o[2], "o_n": o[3],
                     "f_m": f[0], "f_lo": f[1], "f_hi": f[2], "f_n": f[3],
                     "at05": g["fit_frac_at_em_init"].mean()})
    return pd.DataFrame(rows)


def sel(df, **kw):
    for k, v in kw.items():
        df = df[np.isclose(df[k], v)] if isinstance(v, float) else df[df[k] == v]
    return df


def load(data_dir=DEFAULT_OUT_DIR):
    data_dir = Path(data_dir)
    reps = pd.read_csv(data_dir / "paired_replicates.csv")
    tests = pd.read_csv(data_dir / "paired_tests.csv")
    return reps, tests, summarize(reps)


def key_numbers(reps, tests, summ):
    """Every number quoted in the report and the supplement text."""
    one = lambda df, a, **kw: sel(df, assay=a, **kw).iloc[0]
    k = {}
    k["n_o"] = int(reps.groupby(KEYS)["oracle"].size().max())
    k["n_f"] = int(reps.groupby(KEYS)["fit"].count().max())
    fitted = reps[reps["fit"].notna()]
    k["n_boot"] = int(fitted["fit_n_boot"].max()) if "fit_n_boot" in fitted else 10
    k["n_restarts"] = int(fitted["fit_n_restarts"].max()) if "fit_n_restarts" in fitted else 1
    base = sel(summ, experiment="1_clean", regime="rich", true_prior=0.25)
    k["b1"], k["b2"], k["b3"] = (one(base, a) for a in ASSAYS)

    bt = tests[(tests.test == "bias_vs_truth") & (tests.experiment == "1_clean") & (tests.regime == "rich")]
    k["bt"] = bt
    k["n_rich_sig"] = int((bt.p_holm < 0.05).sum())
    ab = bt.assign(ab=bt.mean_diff.abs())
    k["o_bias_max"] = ab[ab.arm == "oracle"].ab.max()
    k["o_bias_nsig"] = int((ab[ab.arm == "oracle"].p_holm < 0.05).sum())
    k["f23_bias_max"] = ab[(ab.arm == "fit") & (ab.assay != A1)].ab.max()
    fa1 = bt[(bt.arm == "fit") & (bt.assay == A1)]
    k["fa1_max"] = fa1.mean_diff.abs().max()
    k["fa1_nsig"] = int((fa1.p_holm < 0.05).sum())

    dv = tests[tests.test == "dev_vs_expectation"]
    k["n_dev"], k["n_dev_sig"] = int(dv.p_holm.notna().sum()), int((dv.p_holm < 0.05).sum())
    dvi = dv[np.abs(dv.reference - 0.5) > 1e-6]  # exclude the unidentifiable cells (expectation 0.5)
    k["o_dev_max"] = dvi[dvi.arm == "oracle"].mean_diff.abs().max()

    fo = tests[(tests.test == "fit_vs_oracle") & tests.experiment.isin(["1_clean", "4_separation", "5_correlation"])
               & (tests.regime == "rich")]
    k["fo23"] = fo[fo.assay != A1]["mean_diff"].abs().max()

    mech = sel(summ, experiment="3_mech_bias")
    k["mech"] = mech
    k["m75"] = {a: one(mech, a, lab_m1_frac=0.75) for a in ASSAYS}
    k["a3_mech_err"] = (sel(mech, assay=A3).f_m - 0.25).abs().max()
    k["a3_mech_err_o"] = (sel(mech, assay=A3).o_m - 0.25).abs().max()
    k["np20"] = {a: one(sel(summ, experiment="2_noise_plp"), a, eps_p=0.2) for a in ASSAYS}
    k["nb20"] = {a: one(sel(summ, experiment="2_noise_blb"), a, delta_b=0.2) for a in ASSAYS}

    kc = sel(summ, experiment="1_clean", regime="KCNH2_Kozek-like")
    k["kc"] = kc
    kt = tests[(tests.test == "bias_vs_truth") & (tests.regime == "KCNH2_Kozek-like") & (tests.arm == "fit")
               & (tests.p_holm < 0.05)]
    k["kc_lo"], k["kc_hi"] = kt["mean_diff"].min(), kt["mean_diff"].max()
    k["kc_at05"] = sel(kc, assay=A1)["at05"].mean()

    sep = sel(summ, experiment="4_separation", true_prior=0.25)
    k["sep15"] = one(sep, A1, delta=1.5)
    k["sep25"] = one(sel(summ, experiment="4_separation", true_prior=0.1), A1, delta=2.5)
    k["sep25_25"] = one(sel(summ, experiment="4_separation", true_prior=0.25), A1, delta=2.5)
    # Fit minus true components for the minority-mechanism assay at weaker separation
    # (Delta = 2.5, independent mechanisms) and with correlated mechanisms (rho = 0.6).
    fo_all = tests[(tests.test == "fit_vs_oracle") & (tests.assay == A1) & (tests.regime == "rich")]
    ws = fo_all[(fo_all.experiment == "4_separation") & np.isclose(fo_all.delta, 2.5)]
    k["ws_gap_lo"], k["ws_gap_hi"] = ws.mean_diff.min(), ws.mean_diff.max()
    k["ws_all_sig"] = bool((ws.p_holm < 0.05).all())
    # Fit minus true components (paired, same datasets) for the M1-only assay across the separation series.
    sep_fo = fo_all[fo_all.experiment == "4_separation"]
    k["gap_sep"] = {(round(r.true_prior, 2), float(r.delta)): (float(r.mean_diff), float(r.p_holm))
                    for r in sep_fo.itertuples()}
    fo23_sep = tests[(tests.test == "fit_vs_oracle") & (tests.experiment == "4_separation") & (tests.assay != A1)
                     & np.isclose(tests.true_prior, 0.25)]
    k["fo23_sep25"] = fo23_sep.mean_diff.abs().max()
    g15 = fo_all[(fo_all.experiment == "4_separation") & np.isclose(fo_all.delta, 1.5) & np.isclose(fo_all.true_prior, 0.1)]
    k["sep15_gap10"], k["sep15_gap10_sig"] = float(g15.mean_diff.iloc[0]), bool(g15.p_holm.iloc[0] < 0.05)
    k["sep15_10"] = one(sel(summ, experiment="4_separation", true_prior=0.1), A1, delta=1.5)
    rc = fo_all[(fo_all.experiment == "5_correlation") & np.isclose(fo_all.delta, 2.5) & np.isclose(fo_all.rho, 0.6)]
    k["rho6_gap_max"] = rc.mean_diff.abs().max()
    # Identifiability: spread of the true-component estimate vs. P/B separation, and the cells where the
    # P/LP and B/LB score distributions coincide for an assay (no P/LP act through its mechanism).
    sd = reps.groupby(KEYS)["oracle"].std().rename("o_sd").reset_index()
    k["sd_base"] = {a: one(sd, a, experiment="1_clean", regime="rich", true_prior=0.25).o_sd for a in ASSAYS}
    k["sd_sep15"] = {a: one(sd, a, experiment="4_separation", true_prior=0.25, delta=1.5).o_sd for a in ASSAYS}
    k["unid_a1"] = one(mech, A1, lab_m1_frac=0.0)
    k["unid_a2"] = one(mech, A2, lab_m1_frac=1.0)
    sep_o = sel(summ, experiment="4_separation", delta=1.5, true_prior=0.25)
    k["sep15_o_err"] = (sep_o.o_m - sep_o.true_prior).abs().max()
    return k


# Settings shown in the supplementary table, in order: (label, selector).
TABLE_SETTINGS = (
    [(f"Clean, $P(Y=1) = {p:.2f}$", dict(experiment="1_clean", regime="rich", true_prior=p))
     for p in (0.05, 0.10, 0.25, 0.40)]
    + [("20\\% of P/LP benign", dict(experiment="2_noise_plp", eps_p=0.2)),
       ("20\\% of B/LB pathogenic", dict(experiment="2_noise_blb", delta_b=0.2)),
       ("75\\% of P/LP via M1", dict(experiment="3_mech_bias", lab_m1_frac=0.75)),
       ("$\\kappa = 1.5$ SD", dict(experiment="4_separation", true_prior=0.25, delta=1.5)),
       ("$\\rho = 0.6$, $\\kappa = 2.5$", dict(experiment="5_correlation", true_prior=0.25, delta=2.5, rho=0.6))]
    + [(f"3/0/10/6 samples, $P(Y=1) = {p:.2f}$", dict(experiment="1_clean", regime="KCNH2_Kozek-like", true_prior=p))
       for p in (0.10, 0.25)]
)


def table_rows(summ):
    """[(setting_label, row)] for each setting in TABLE_SETTINGS and each assay."""
    out = []
    for label, kw in TABLE_SETTINGS:
        for a in ASSAYS:
            out.append((label, sel(summ, assay=a, **kw).iloc[0]))
    return out


# ── sample-size experiment (test_priors.py --design sample_size) ─────────────────────────────────────────────

SAMPLES = ("P/LP", "B/LB", "gnomAD", "synonymous")
BASE_COUNTS = dict(zip(SAMPLES, (100, 100, 500, 200)))


def load_sample_size(data_dir=DEFAULT_OUT_DIR):
    path = Path(data_dir) / "sample_size_replicates.csv"
    return pd.read_csv(path) if path.exists() else None


def _boot_ci(x, stat, n_boot=2000, seed=0):
    """Percentile-bootstrap 95% CI of stat(x) over replicates."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def _err_stats(est, truth):
    """Bias [95% CI], SD [bootstrap CI], RMSE and its bias^2 / variance parts, n of the finite estimates."""
    x = np.asarray(est, dtype=float)
    x = x[np.isfinite(x)]
    m, lo, hi, n = mean_ci(x)
    sd = x.std(ddof=1) if n > 1 else np.nan
    sd_lo, sd_hi = _boot_ci(x, lambda v: v.std(ddof=1))
    bias = m - truth
    return {"bias": bias, "bias_lo": lo - truth, "bias_hi": hi - truth, "sd": sd, "sd_lo": sd_lo, "sd_hi": sd_hi,
            "bias2": bias ** 2, "var": sd ** 2, "rmse": np.sqrt(bias ** 2 + sd ** 2), "n": n}


def sample_size_summary(reps):
    """One row per (sample reduced, size, assay, arm). The base setting is repeated as size = base for each sample.
    arm: 'oracle' (true components, after the discard rule), 'oracle_raw' (before it), 'fit'."""
    rows = []
    for regime, g in reps.groupby("regime", sort=False):
        if regime == "base":
            targets = [(s, BASE_COUNTS[s]) for s in SAMPLES]
        else:
            s, n = regime.split(" = ")
            targets = [(s, int(n))]
        for assay, ga in g.groupby("assay", sort=False):
            truth = ga["true_prior"].iloc[0]
            fitted = ga[ga["fit_n_boot"] > 0]
            disc = {"oracle": ga["oracle"].isna().mean(), "oracle_raw": 0.0,
                    "fit": 1 - (fitted["fit_n_boot_ok"].sum() / max(fitted["fit_n_boot"].sum(), 1))}
            for arm, col in (("oracle", "oracle"), ("oracle_raw", "oracle_raw"), ("fit", "fit")):
                st = _err_stats(ga[col], truth)
                for s, n in targets:
                    rows.append({"sample": s, "size": n, "is_base": regime == "base", "assay": assay, "arm": arm,
                                 "discard": disc[arm], **st})
    out = pd.DataFrame(rows)
    out["sample"] = pd.Categorical(out["sample"], SAMPLES, ordered=True)
    return out.sort_values(["sample", "size", "assay", "arm"], ascending=[True, False, True, True]).reset_index(drop=True)


def sample_size_key_numbers(ss):
    """For each assay and arm, the error at the smallest size of each sample, relative to the base."""
    k = {}
    for (assay, arm), g in ss.groupby(["assay", "arm"], observed=True):
        base = g[g.is_base].iloc[0]
        k[(assay, arm, "base")] = base
        for s in SAMPLES:
            gs = g[(g["sample"] == s) & ~g.is_base]
            k[(assay, arm, s)] = gs.sort_values("size").iloc[0]
    return k


# ── boundary-prior policy test (test_priors.py --design policy) ──────────────────────────────────────────────

POLICIES = {
    "P0": "discard ≤0.001 / ≥0.999 (current)",
    "P1": "keep all",
    "P2a": "clip to [0.001, 0.999]",
    "P2b": "clip to [0.0001, 0.9999]",
    "P3": "clip [0.001, 0.999]; EM-start bootstraps → 0.10",
    "P4": "0.10 if >50% of bootstraps invalid or at EM start, else P0",
}
FALLBACK_PRIOR = 0.10
EM_START_TOL = 1e-3


def _policy_value(v, policy):
    """Dataset-level prior from one dataset's raw bootstrap priors `v` under `policy`; (value, fell_back)."""
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, False
    bad = (v <= 0.001) | (v >= 0.999)
    at_start = np.abs(v - 0.5) < EM_START_TOL
    if policy == "P0":
        k = v[~bad]
        return (float(np.median(k)) if len(k) else np.nan), False
    if policy == "P1":
        return float(np.median(v)), False
    if policy == "P2a":
        return float(np.median(np.clip(v, 0.001, 0.999))), False
    if policy == "P2b":
        return float(np.median(np.clip(v, 0.0001, 0.9999))), False
    if policy == "P3":
        c = np.clip(v, 0.001, 0.999)
        c[at_start] = FALLBACK_PRIOR
        return float(np.median(c)), bool(at_start.mean() > 0.5)
    if policy == "P4":
        if (bad | at_start).mean() > 0.5:
            return FALLBACK_PRIOR, True
        k = v[~bad]
        return (float(np.median(k)) if len(k) else FALLBACK_PRIOR), not len(k)
    raise ValueError(policy)


def _oracle_policy_value(x, policy):
    """Single true-component estimate under a policy (P4 needs a bootstrap distribution, so it is not defined)."""
    if not np.isfinite(x):
        return np.nan
    if policy == "P0":
        return x if 0.001 < x < 0.999 else np.nan
    if policy == "P1":
        return x
    if policy in ("P2a", "P3"):
        y = min(max(x, 0.001), 0.999)
        return FALLBACK_PRIOR if policy == "P3" and abs(x - 0.5) < EM_START_TOL else y
    if policy == "P2b":
        return min(max(x, 0.0001), 0.9999)
    return np.nan


def load_policy(data_dir=DEFAULT_OUT_DIR):
    path = Path(data_dir) / "policy_replicates.csv"
    return pd.read_csv(path) if path.exists() else None


def policy_summary(reps):
    """One row per (experiment, regime, true prior, assay, arm, policy): bias, SD, RMSE of the dataset-level prior,
    the share of datasets with no usable prior (NaN), and the fallback rate."""
    rows = []
    keys = ["experiment", "regime", "true_prior", "assay"]
    for key, g in reps.groupby(keys, sort=False):
        truth = key[2]
        fitted = g[g["fit_n_boot"] > 0]
        boots = [np.array([float(t) for t in str(s).split(";") if t != ""], dtype=float)
                 for s in fitted["fit_boot_raw"].fillna("")]
        for policy in POLICIES:
            vals = [_policy_value(b, policy) for b in boots]
            est = np.array([v for v, _ in vals], dtype=float)
            fb = np.mean([f for _, f in vals]) if vals else np.nan
            st = _err_stats(est, truth)
            rows.append({**dict(zip(keys, key)), "arm": "fit", "policy": policy, "nan_frac": np.mean(~np.isfinite(est)),
                         "fallback": fb, "floor": np.mean(est[np.isfinite(est)] <= 0.001) if np.isfinite(est).any()
                         else np.nan, "mean": st["bias"] + truth, **st})
            if policy != "P4":
                est_o = np.array([_oracle_policy_value(x, policy) for x in g["oracle_raw"]], dtype=float)
                st = _err_stats(est_o, truth)
                rows.append({**dict(zip(keys, key)), "arm": "oracle", "policy": policy,
                             "nan_frac": np.mean(~np.isfinite(est_o)), "fallback": np.nan,
                             "floor": np.mean(est_o[np.isfinite(est_o)] <= 0.001) if np.isfinite(est_o).any() else np.nan,
                             "mean": st["bias"] + truth, **st})
    return pd.DataFrame(rows)


# ── EM starting value (test_priors.py --design policy, columns *_start01) ──────────────────────────────────────

def _discard_median(s):
    """Dataset prior under the current rule from a ';'-joined string of raw bootstrap priors (NaN if none kept)."""
    v = np.array([float(t) for t in str(s).split(";") if t != ""], dtype=float)
    v = v[np.isfinite(v) & (v > 0.001) & (v < 0.999)]
    return float(np.median(v)) if len(v) else np.nan


def start_value_summary(reps):
    """One row per (experiment, regime, true prior, assay, arm, start): bias, SD and RMSE of the prior under the
    current discard rule with the EM started at 0.5 (production) or 0.1, the share of estimates still at the start
    (|p - start| < EM_START_TOL), and the share of datasets with no prior. None if the run predates the 0.1 start."""
    if "oracle_raw_start01" not in reps.columns:
        return None
    rows = []
    keys = ["experiment", "regime", "true_prior", "assay"]
    for key, g in reps.groupby(keys, sort=False):
        truth = key[2]
        fitted = g[g["fit_n_boot"] > 0]
        for start, ocol, fcol in ((0.5, "oracle_raw", "fit_boot_raw"), (0.1, "oracle_raw_start01", "fit_boot_raw_start01")):
            o = g[ocol].to_numpy(dtype=float)
            o_kept = np.where((o > 0.001) & (o < 0.999), o, np.nan)
            st = _err_stats(o_kept, truth)
            rows.append({**dict(zip(keys, key)), "arm": "oracle", "start": start, "at_start": float(np.mean(
                np.abs(o - start) < EM_START_TOL)), "nan_frac": float(np.mean(~np.isfinite(o_kept))),
                "mean": st["bias"] + truth, **st})
            boots = [np.array([float(t) for t in str(s).split(";") if t != ""], dtype=float)
                     for s in fitted[fcol].fillna("")]
            f = np.array([_discard_median(s) for s in fitted[fcol].fillna("")], dtype=float)
            allb = np.concatenate(boots) if boots else np.array([])
            st = _err_stats(f, truth)
            rows.append({**dict(zip(keys, key)), "arm": "fit", "start": start,
                         "at_start": float(np.mean(np.abs(allb[np.isfinite(allb)] - start) < EM_START_TOL))
                         if len(allb) else np.nan, "nan_frac": float(np.mean(~np.isfinite(f))),
                         "mean": st["bias"] + truth, **st})
    return pd.DataFrame(rows)
