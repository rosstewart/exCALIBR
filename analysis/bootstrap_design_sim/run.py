"""One simulation replicate: fit every bootstrap scheme on the same data and
score held-out missense draws against the oracle (the true LR).

Schemes (fit_utils/fit.py bootstrap_scheme; the fit's samples are always the
strata, as in production):
  S0     composite samples x readout pattern (today's scheme, resampling the
         standard samples while fitting strata)
  S1     stratum only
  S2n3 / S2n5 / S2n10   stratum x pattern, thin cells collapsed into the nearest
         cell (n_min 3 / 5 / 10)
  S3     S2n5 with McCarthy-Snowden n-1 draws

Composite weights for the LR are the "missense" arm (P/LP missense vs B/LB
missense), as in the combined model. Points use the production rule (5th /
95th percentile of the bootstrap log LR) at the TRUE prior, so differences
reflect the bootstrap spread alone, not prior estimation. Oracle points are
the true log LR at the same thresholds.

Metrics (per scheme): oracle recall (share of oracle-determinate held-out
variants given same-direction points; primary), wrong-direction rate (points
opposite the oracle's direction, or opposite the label for oracle-neutral
variants), overstatement (|points| > |oracle points|, same direction),
determinate rate, MCC and DOR on the held-out labels, LR recovery (mean bias
and median absolute error of the median log LR vs the true log LR) and
median bootstrap spread (p95 - p5). Oracle recall is also split by whether
the held-out variant's readout pattern is rare (< 5% of the training rows).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SCHEMES = {
    "S0": dict(bootstrap_scheme="pattern", composite=True),
    "S1": dict(bootstrap_scheme="stratum"),
    "S2n3": dict(bootstrap_scheme="stratum_pattern", bootstrap_min_cell=3),
    "S2n5": dict(bootstrap_scheme="stratum_pattern", bootstrap_min_cell=5),
    "S2n10": dict(bootstrap_scheme="stratum_pattern", bootstrap_min_cell=10),
    "S3": dict(bootstrap_scheme="stratum_pattern", bootstrap_min_cell=5, bootstrap_n_minus_one=True),
}


class _SimMS:
    """Minimal multiscoreset: what Fit, build_strata and StrataView read."""

    def __init__(self, X, members, D):
        self._scores = np.asarray(X, float)
        self._sample_assignments = np.asarray(members, bool)
        self.sample_counts = self._sample_assignments.sum(axis=0)
        self.dataset_names = [f"r{j}" for j in range(D)]
        self.kept_variants = [f"v{i}" for i in range(len(X))]
        self.categorical_dims = {}
        self.is_multivariate = True
        self.n_assays = D

    @property
    def scores(self):
        return self._scores

    @property
    def sample_assignments(self):
        return self._sample_assignments[:, self.sample_counts > 0]

    @property
    def sample_names(self):
        n = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]
        return [x for x, k in zip(n, self.sample_counts) if k > 0]


def _log_lr(params, wP, wB, X):
    from scipy.special import logsumexp
    from src.assay_calibration.multivariate_analysis.mv_calibration import _comp_logpdf
    K = len(params)
    lf = np.stack([_comp_logpdf(X, params, c) for c in range(K)])
    return logsumexp(np.log(wP + 1e-300)[:, None] + lf, axis=0) - logsumexp(np.log(wB + 1e-300)[:, None] + lf, axis=0)


def _points(lo, hi, lc, pvals=range(1, 9)):
    pts = np.zeros(len(lo), int)
    for pv in pvals:
        pts[lo >= pv / 8 * lc] = pv
    for pv in pvals:
        pts[hi <= -pv / 8 * lc] = -pv
    return pts


def _mcc(tp, tn, fp, fn):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else np.nan


def run_cell(cell, rep, B=30, R=1, K=3, latent_q=1, schemes=None):
    from analysis.bootstrap_design_sim.generate import make_data, rng_for, true_log_lr
    from src.assay_calibration.fit_utils.fit import Fit
    from src.assay_calibration.fit_utils.strata import build_strata, StrataView, compose_weights
    from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
    from src.assay_calibration.multivariate_analysis.mv_calibration import MVCalibrationAnalysis
    warnings.filterwarnings("ignore")
    data = make_data(rng_for(cell, rep), cell)
    ms = _SimMS(data["X"], data["members"], cell["D"])
    ann = pd.DataFrame({"consequence": data["consequence"], "splice": False})
    strata = build_strata(ms, ann, min_stratum=5)
    view = StrataView(ms, strata)
    M = strata.composition("missense")
    lc = float(np.log(get_tavtigian_constant(round(cell["pi"], 4))))
    Xe, ye = data["eval_X"], data["eval_y"]
    true = true_log_lr(Xe, cell)
    oracle = _points(true, true, lc)
    tr_pat = pd.Series([tuple(r) for r in ~np.isnan(data["X"])]).value_counts(normalize=True)
    rare = np.array([tr_pat.get(tuple(r), 0.0) < 0.05 for r in ~np.isnan(Xe)])
    rows = []
    for name, spec in (schemes or SCHEMES).items():
        kw = {k: v for k, v in spec.items() if k != "composite"}
        if spec.get("composite"):
            kw["bootstrap_assignments"] = ms._sample_assignments
        L = []
        for b in range(B):
            jobs = Fit(view).generate_fit_jobs([K], bootstrap_seed=b, master_seed=0, num_fits=R, latent_q=latent_q,
                                               init_strategy="kmeans", check_monotonic=False, min_overlap_rows=1,
                                               verbose_overlap=False, **kw)
            res = [Fit.execute_fit_job(j) for j in jobs]
            res = [r for r in res if r is not None and np.isfinite(r.get("val_ll", -np.inf))]
            if not res:
                continue
            best = max(res, key=lambda r: r["val_ll"])
            fit = MVCalibrationAnalysis._reconstitute_params(None, best["fit"])
            if fit is None:
                continue
            Wc = compose_weights(np.asarray(fit["weights"], float), M)
            L.append(_log_lr(fit["component_params"], Wc[0], Wc[1], Xe))
        if not L:
            rows.append(dict(cell_id=cell["cell_id"], rep=rep, scheme=name, n_boot=0))
            continue
        L = np.asarray(L)
        lo, hi, med = np.nanpercentile(L, 5, 0), np.nanpercentile(L, 95, 0), np.nanmedian(L, 0)
        pts = _points(lo, hi, lc)
        det_or = oracle != 0
        same = np.sign(pts) == np.sign(oracle)
        wrong = np.where(det_or, (pts != 0) & ~same, (pts != 0) & (np.sign(pts) != np.where(ye == 1, 1, -1)))
        tp = int(((pts > 0) & (ye == 1)).sum()); tn = int(((pts < 0) & (ye == 0)).sum())
        fp = int(((pts > 0) & (ye == 0)).sum()); fn = int(((pts < 0) & (ye == 1)).sum())
        rows.append(dict(
            cell_id=cell["cell_id"], rep=rep, scheme=name, n_boot=len(L), n_strata=len(strata.names),
            oracle_recall=float((same & (pts != 0))[det_or].mean()) if det_or.any() else np.nan,
            oracle_recall_rare=float((same & (pts != 0))[det_or & rare].mean()) if (det_or & rare).any() else np.nan,
            oracle_recall_common=float((same & (pts != 0))[det_or & ~rare].mean()) if (det_or & ~rare).any() else np.nan,
            wrong_direction=float(wrong.mean()),
            overstatement=float(((np.abs(pts) > np.abs(oracle)) & same & (pts != 0)).mean()),
            determinate=float((pts != 0).mean()), oracle_determinate=float(det_or.mean()),
            mcc=_mcc(tp, tn, fp, fn), dor=(tp * tn) / (fp * fn) if fp * fn else np.inf,
            lr_bias=float(np.nanmean(med - true)), lr_mae=float(np.nanmedian(np.abs(med - true))),
            spread=float(np.nanmedian(hi - lo)), rare_share=float(rare.mean()),
        ))
    return rows
