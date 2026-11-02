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
    rc = fo_all[(fo_all.experiment == "5_correlation") & np.isclose(fo_all.delta, 2.5) & np.isclose(fo_all.rho, 0.6)]
    k["rho6_gap_max"] = rc.mean_diff.abs().max()
    sep_o = sel(summ, experiment="4_separation", delta=1.5)
    k["sep15_o_err"] = (sep_o.o_m - sep_o.true_prior).abs().max()
    return k


# Settings shown in the supplementary table, in order: (label, selector).
TABLE_SETTINGS = (
    [(f"Clean, $P(Y=1) = {p:.2f}$", dict(experiment="1_clean", regime="rich", true_prior=p))
     for p in (0.05, 0.10, 0.25, 0.40)]
    + [("20\\% of P/LP benign", dict(experiment="2_noise_plp", eps_p=0.2)),
       ("20\\% of B/LB pathogenic", dict(experiment="2_noise_blb", delta_b=0.2)),
       ("75\\% of P/LP via M1", dict(experiment="3_mech_bias", lab_m1_frac=0.75)),
       ("$\\Delta = 1.5$ SD", dict(experiment="4_separation", true_prior=0.25, delta=1.5)),
       ("$\\rho = 0.6$, $\\Delta = 2.5$", dict(experiment="5_correlation", true_prior=0.25, delta=2.5, rho=0.6))]
    + [(f"3/0/10/6 controls, $P(Y=1) = {p:.2f}$", dict(experiment="1_clean", regime="KCNH2_Kozek-like", true_prior=p))
       for p in (0.10, 0.25)]
)


def table_rows(summ):
    """[(setting_label, row)] for each setting in TABLE_SETTINGS and each assay."""
    out = []
    for label, kw in TABLE_SETTINGS:
        for a in ASSAYS:
            out.append((label, sel(summ, assay=a, **kw).iloc[0]))
    return out
