#!/usr/bin/env python3
"""Bootstrap-to-bootstrap LR variance: single-assay (scalar) vs. multivariate
CFUSN(q=2), across dimensionality, missingness, and bootstrap strategy.

Motivation: real multivariate calibrations look far noisier bootstrap-to-
bootstrap than single-assay ones, which calls into question whether the
production percentile defaults (path=5/ben=95 -- mv_calibration.Analysis.run,
pipeline.config.pathogenic_percentile) are still the right conservatism knob as
dimensionality and missingness grow. The existing sim_missingness_recovery.py
measures point-estimate recovery against ground truth; nothing measured spread
across resampled refits. This does.

Every multivariate fit here is q=2 (latent_q=2, the production default); p=1
uses the true scalar path (multivariate=False), which is what single-assay
calibrations actually run. No q=1 CFUSN proxy is fitted anywhere.

Bootstrap strategies compared (production code reused directly):
    unstratified      -- i.i.d. resample within each sample class (seeded local
                         copy of fit.get_bootstrap_indices' logic; the
                         production function takes no seed and is unused)
    sample_specific   -- fit.sample_specific_bootstrap (univariate default)
    pattern_stratified-- fit.pattern_stratified_bootstrap (MV default).
                         Identical to sample_specific when nothing is missing,
                         which is a built-in sanity check at p=1/none.

Research output only -- this changes no production behaviour.

Usage:
    # smoke
    python -u tests/cfusn_simulations/sim_bootstrap_variance.py \
        --p-grid 1 8 --missingness none mcar_heavy --n-bootstraps 3
    # full grid, parallel
    python -u tests/cfusn_simulations/sim_bootstrap_variance.py \
        --p-grid 1 2 4 16 --n-bootstraps 10 --n-jobs -1
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import (
    tryToFit, sample_specific_bootstrap, pattern_stratified_bootstrap,
    derive_bootstrap_seed, derive_fit_seed,
)
from src.assay_calibration.fit_utils.cfusn.density_utils import log_joint_densities
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, inject_missingness, inject_block_missingness,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Sample classes: index 0 = pathogenic (minority), 1 = benign.
PATH_CLASS, BEN_CLASS = 0, 1
# Mixing weights per sample class over the K=2 shared components
# (component 0 = "benign-like", component 1 = "pathogenic-like").
COMP_PROBS = np.array([[0.15, 0.85],    # pathogenic sample class
                       [0.95, 0.05]])   # benign sample class
N_COMPONENTS = 2
MASTER_SEED = 12345

# Per-dimension ground-truth template, tiled up to any p. Only the first
# N_INFORMATIVE_DIMS carry real separation; the rest are near-null. That is
# what real multi-assay panels look like, and it is the point of the study:
# growing p adds parameters to estimate without adding proportional signal.
N_INFORMATIVE_DIMS = 2
MU_INFORMATIVE = 0.60
MU_NULL = 0.08
GAMMA_SCALE = 0.30
DELTA_TEMPLATE = np.array([[0.70, 0.20], [0.50, -0.15], [0.35, 0.10], [0.25, -0.10]])
DELTA_NULL_SCALE = 0.15      # near-null dims get a heavily shrunk Delta row
DELTA_BENIGN_SCALE = 0.25    # benign-like component is near symmetric

MISSINGNESS_CONDITIONS = ("none", "mcar_light", "mcar_heavy", "block_tp53like")
MCAR_LIGHT_FRAC = 0.15
MCAR_HEAVY_FRAC = 0.50
PCT_PAIRS = ((5, 95), (10, 90), (15, 85), (20, 80))
# Drop query points below this percentile of the true mixture density (see
# make_condition_data): the far tails are where log-LR is not estimable.
QUERY_DENSITY_PCT = 20.0


# ── Ground truth ────────────────────────────────────────────────────────────

def build_true_params(p):
    """Two CFUSN components in p dims, q=2 for p>=2 and q=1 for p=1.

    Skew enters only through Delta (the alternate form sample_cfusn uses);
    there is no alpha parameterization in this codebase's simulators.
    """
    q = 2 if p >= 2 else 1
    mu_mag = np.where(np.arange(p) < N_INFORMATIVE_DIMS, MU_INFORMATIVE, MU_NULL)
    delta_rows = DELTA_TEMPLATE[np.arange(p) % len(DELTA_TEMPLATE)][:, :q]
    shrink = np.where(np.arange(p) < N_INFORMATIVE_DIMS, 1.0, DELTA_NULL_SCALE)
    delta_path = delta_rows * shrink[:, None]

    Gamma = GAMMA_SCALE * np.eye(p)
    benign = (-mu_mag, DELTA_BENIGN_SCALE * delta_path, Gamma)
    pathogenic = (mu_mag, delta_path, Gamma)
    return [benign, pathogenic]


def _missingness_spec(p, condition):
    """(kind, args) describing how to NaN out a (N, p) array."""
    if condition == "none" or p < 2:
        return ("none", None)
    if condition == "mcar_light":
        return ("mcar", np.full(p, MCAR_LIGHT_FRAC))
    if condition == "mcar_heavy":
        return ("mcar", np.full(p, MCAR_HEAVY_FRAC))
    if condition == "block_tp53like":
        # One co-observed block (like TP53's always-together 8-readout assay)
        # plus 1-2 individually very sparse dims (KawOligo-like).
        if p >= 4:
            blocks = [list(range(0, max(2, p // 2))), [p - 1], [p - 2]]
            fracs = [0.25, 0.90, 0.90]
        else:
            blocks = [[0], [p - 1]]
            fracs = [0.25, 0.90]
        return ("block", (blocks, fracs))
    raise ValueError(f"unknown missingness condition: {condition}")


def _restore_empty_rows(X_missing, X_full, rng):
    """Guarantee every row keeps at least one observed dimension.

    Injecting missingness into an already-complete matrix can mask every
    dimension of a row (at p=4 the block spec below empties ~20% of rows, and
    mcar_heavy ~6%), but a variant with no measurement at all cannot occur in
    real data: the MV matrix is assembled from variants that some assay scored,
    and the univariate loader drops NaN-score rows outright
    (data_utils/dataset.py:274-284).

    Such rows are not merely unrealistic, they break EM: they contribute
    nothing to the observed-data likelihood but still enter the M-step through
    the no-observed-dims prior branch, and removing them turned a 6.68e+01
    likelihood decrease at iteration 197 into a clean 52-iteration monotone fit.
    """
    empty = np.isnan(X_missing).all(axis=1)
    if not empty.any():
        return X_missing
    for i in np.where(empty)[0]:
        d = rng.randint(X_missing.shape[1])
        X_missing[i, d] = X_full[i, d]
    return X_missing


def _apply_missingness(X, spec, rng):
    kind, args = spec
    if kind == "none":
        return X
    if kind == "mcar":
        out = inject_missingness(X, args, rng)
    else:
        blocks, fracs = args
        out = inject_block_missingness(X, blocks, fracs, rng)
    return _restore_empty_rows(out, X, rng)


def make_condition_data(p, condition, n_path, n_ben, n_query, data_seed):
    """Fixed dataset + fixed query set for one (p, missingness) condition.

    The query set is drawn once from the TRUE mixture (50/50 over components)
    and reused across every strategy and every bootstrap, so the log-LR vectors
    are directly comparable. Under missingness, a second query set carrying
    representative observed/missing patterns is produced -- that is where
    pattern stratification is supposed to matter.
    """
    params = build_true_params(p)
    rng = np.random.RandomState(data_seed)
    X, sa, _ = sample_cfusn_mixture(
        params, COMP_PROBS, np.array([n_path, n_ben]), rng
    )
    spec = _missingness_spec(p, condition)
    X = _apply_missingness(X, spec, rng)

    q_rng = np.random.RandomState(data_seed + 7919)
    # Oversample, then keep only the BULK of the true mixture. Query points in
    # the far tails have vanishing density under some bootstrap refits, so their
    # log-LR swings by hundreds of log units and dominates any spread statistic
    # -- that is a property of evaluating a ratio where neither density is
    # estimable, not a property of the bootstrap strategy under test. Calibration
    # only has to be trustworthy where variants actually live, so the query set
    # is restricted to points above the QUERY_DENSITY_PCT-th percentile of the
    # true mixture's own density.
    Xq_all, _, _ = sample_cfusn_mixture(
        params, np.array([[0.5, 0.5]]), np.array([int(n_query * 1.6)]), q_rng
    )
    # multivariate=True regardless of p: build_true_params always returns
    # alternate-form (mu, Delta, Gamma) triples, and at p=1 Delta is (1,1) so
    # this takes the q=1 restricted-MSN branch. (The FITTED params at p=1 are
    # canonical (a, loc, scale) and are evaluated with multivariate=False in
    # _log_lr -- the two forms must not be confused: passing the alternate-form
    # triple to the scalar path silently broadcasts into a (K,K,N) array.)
    true_logdens = logsumexp(
        log_joint_densities(Xq_all, params, np.array([0.5, 0.5]), multivariate=True),
        axis=0,
    )
    keep_q = true_logdens >= np.percentile(true_logdens, QUERY_DENSITY_PCT)
    Xq = Xq_all[keep_q][:n_query]
    Xq_missing = _apply_missingness(Xq.copy(), spec, q_rng) if spec[0] != "none" else None
    return X, sa, Xq, Xq_missing


def pattern_diagnostics(X):
    """How much stratification structure the data actually has -- at high p
    under MCAR nearly every row is its own pattern, which makes
    pattern_stratified degenerate toward "no resampling at all"."""
    obs = ~np.isnan(X)
    patterns = [tuple(row.tolist()) for row in obs]
    counts = {}
    for pat in patterns:
        counts[pat] = counts.get(pat, 0) + 1
    n_singleton = sum(1 for c in counts.values() if c == 1)
    return dict(
        n_patterns=len(counts),
        frac_singleton_rows=n_singleton / len(patterns),
        frac_complete_rows=float(obs.all(axis=1).mean()),
    )


# ── Bootstrap strategies ────────────────────────────────────────────────────

def unstratified_bootstrap(sample_assignments, bootstrap_seed=None, max_tries=100):
    """Seeded equivalent of fit.get_bootstrap_indices: a plain i.i.d. resample
    over ALL rows, ignoring sample class entirely.

    This is the real "why does stratification exist" baseline -- the
    pathogenic/benign mix is free to vary bootstrap-to-bootstrap on top of the
    missingness-composition drift that pattern_stratified_bootstrap's docstring
    describes. (An earlier version resampled within each class, which made this
    bit-identical to sample_specific_bootstrap and measured nothing -- it
    produced exactly equal median_band at p=1 and p=8.)

    The production function takes no seed (global np.random.choice) and has no
    call sites, so it cannot be used directly in a reproducible study.
    """
    rng = np.random.RandomState(bootstrap_seed)
    n = sample_assignments.shape[0]
    idx = np.arange(n)
    for _ in range(max_tries):
        train = rng.choice(idx, size=n, replace=True)
        evl = np.setdiff1d(idx, train)
        # Both classes must survive or there is nothing to calibrate; this is
        # itself a symptom of the unstratified scheme, so count the retries.
        if len(evl) and sample_assignments[train].sum(axis=0).min() >= 2:
            return train, evl
    raise ValueError("unstratified bootstrap could not retain both classes")


def draw_indices(strategy, X, sa, bootstrap_seed):
    if strategy == "unstratified":
        return unstratified_bootstrap(sa, bootstrap_seed)
    if strategy == "sample_specific":
        return sample_specific_bootstrap(sa, bootstrap_seed)
    if strategy == "pattern_stratified":
        return pattern_stratified_bootstrap(X, sa, bootstrap_seed)
    raise ValueError(f"unknown strategy: {strategy}")


# ── Fitting + LR evaluation ─────────────────────────────────────────────────

def _log_lr(params, weights, Xq, mv):
    """Production log-LR: pathogenic-class mixture vs. benign-class mixture
    over the same fitted components. Permutation-invariant in the component
    labels, so bootstrap label switching does not contaminate it.

    Reuses density_utils.log_joint_densities -> _single_component_logpdf, the
    same density the EM and mv_calibration._sn_logpdf use (so NaNs in query
    points are handled by the production missing-data path).
    """
    w = np.asarray(weights, dtype=float)
    w_p = np.clip(w[PATH_CLASS], 1e-300, None)
    w_b = np.clip(w[BEN_CLASS], 1e-300, None)
    x = Xq if mv else np.asarray(Xq).ravel()
    log_fp = logsumexp(log_joint_densities(x, params, w_p, multivariate=mv), axis=0)
    log_fb = logsumexp(log_joint_densities(x, params, w_b, multivariate=mv), axis=0)
    return np.asarray(log_fp - log_fb, dtype=float).ravel()


def _pathogenic_weight(weights):
    """Weight the pathogenic sample class puts on its most pathogenic-enriched
    component, identified by the w_p/w_b ratio rather than by index -- again so
    bootstrap component label switching does not show up as variance."""
    w = np.asarray(weights, dtype=float)
    ratio = w[PATH_CLASS] / np.clip(w[BEN_CLASS], 1e-300, None)
    return float(w[PATH_CLASS][int(np.argmax(ratio))])


def run_one_bootstrap(task):
    """One resample + refit + LR evaluation. Module-level and picklable so it
    can run under ProcessPoolExecutor."""
    (p, condition, strategy, boot_idx,
     n_path, n_ben, n_query, data_seed, max_em_iters) = task

    X, sa, Xq, Xq_missing = make_condition_data(
        p, condition, n_path, n_ben, n_query, data_seed
    )
    mv = p >= 2

    boot_seed = derive_bootstrap_seed(MASTER_SEED, boot_idx)
    try:
        train_idx, _ = draw_indices(strategy, X, sa, boot_seed)
    except ValueError:
        return dict(ok=False)

    obs = X[train_idx] if mv else X[train_idx].ravel()
    result = tryToFit(
        obs, sa[train_idx], num_components=N_COMPONENTS, constrained=False,
        init_method="kmeans", init_constraint_adjustment="scale",
        multivariate=mv, latent_q=2, check_monotonic=False, num_fits=1,
        fit_seed=derive_fit_seed(MASTER_SEED, boot_idx, N_COMPONENTS, 0),
        verbose=False, verbose_init=False, max_em_iters=max_em_iters,
    )
    params = result.get("component_params", [])
    weights = result.get("weights")
    if not params or any(len(pp) == 0 for pp in params) or weights is None:
        return dict(ok=False)

    out = dict(
        ok=True,
        log_lr=_log_lr(params, weights, Xq, mv),
        path_weight=_pathogenic_weight(weights),
    )
    if Xq_missing is not None:
        out["log_lr_missing"] = _log_lr(params, weights, Xq_missing, mv)
    return out


# ── Aggregation ─────────────────────────────────────────────────────────────

def _spread_metrics(mat, prefix=""):
    """mat: (n_bootstraps, n_query) log-LR. Reduced exactly the way production
    does it -- percentiles down axis 0 (mv_calibration.py:1591-1593)."""
    if mat.size == 0 or mat.shape[0] < 2:
        return {f"{prefix}median_band": np.nan, f"{prefix}median_iqr": np.nan,
                f"{prefix}median_std": np.nan, f"{prefix}median_mad": np.nan,
                f"{prefix}sign_unstable_frac": np.nan}
    lo = np.nanpercentile(mat, 5, axis=0)
    hi = np.nanpercentile(mat, 95, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)
    med = np.nanmedian(mat, axis=0)
    band = hi - lo
    return {
        f"{prefix}median_band": float(np.nanmedian(band)),
        # Robust companion to median_band. The 5/95 band is an extreme-order
        # statistic: with few bootstraps it is essentially the full range of the
        # draws, so a single wandering refit sets it and the estimate itself has
        # enormous variance (observed directly at n_bootstraps=10, where band
        # widths swung between 21 and 689 log units across adjacent cells with
        # no stable ordering). The IQR uses interior order statistics and stays
        # meaningful at any bootstrap count, so it is the metric to read when
        # the two disagree.
        f"{prefix}median_iqr": float(np.nanmedian(q75 - q25)),
        f"{prefix}median_std": float(np.nanmedian(np.nanstd(mat, axis=0))),
        f"{prefix}median_mad": float(np.nanmedian(
            np.nanmedian(np.abs(mat - med[None, :]), axis=0))),
        # fraction of query points whose 5-95 band straddles log-LR = 0, i.e.
        # where bootstrapping cannot even agree on the direction of evidence
        f"{prefix}sign_unstable_frac": float(np.mean((lo < 0) & (hi > 0) & ~np.isnan(med))),
    }


def percentile_sensitivity(mat):
    """Cheap, no refitting: how much evidence each percentile pair keeps.

    For each query point the conservative (toward-null) bootstrap estimate is
    the low percentile when the median favours pathogenic and the high
    percentile when it favours benign -- exactly what 5/95 does in production.
    """
    rows = []
    med = np.nanmedian(mat, axis=0)
    for path_pct, ben_pct in PCT_PAIRS:
        lo = np.nanpercentile(mat, path_pct, axis=0)
        hi = np.nanpercentile(mat, ben_pct, axis=0)
        conservative = np.where(med >= 0, lo, hi)
        rows.append(dict(
            path_percentile=path_pct,
            ben_percentile=ben_pct,
            median_band=float(np.nanmedian(hi - lo)),
            mean_abs_conservative_log_lr=float(np.nanmean(np.abs(conservative))),
            # fraction of points where the conservative estimate still points
            # the same way as the median -- evidence survives the haircut
            evidence_retained_frac=float(np.nanmean(np.sign(conservative) == np.sign(med))),
            median_shrinkage=float(np.nanmedian(np.abs(med) - np.abs(conservative))),
        ))
    return rows


def run_condition(p, condition, strategy, args, executor=None):
    tasks = [(p, condition, strategy, b, args.n_path, args.n_benign,
              args.n_query, args.data_seed, args.max_em_iters)
             for b in range(args.n_bootstraps)]
    if executor is None:
        results = [run_one_bootstrap(t) for t in tasks]
    else:
        results = list(executor.map(run_one_bootstrap, tasks))

    good = [r for r in results if r.get("ok")]
    n_valid, n_failed = len(good), len(results) - len(good)

    X, _, _, Xq_missing = make_condition_data(
        p, condition, args.n_path, args.n_benign, args.n_query, args.data_seed
    )
    row = dict(p=p, missingness=condition, strategy=strategy,
               n_valid=n_valid, n_failed=n_failed, **pattern_diagnostics(X))

    if not good:
        row.update(_spread_metrics(np.empty((0, 0))))
        row.update(path_weight_std=np.nan, path_weight_cv=np.nan)
        return row, np.empty((0, 0)), None

    mat = np.vstack([r["log_lr"] for r in good])
    row.update(_spread_metrics(mat))
    if Xq_missing is not None and "log_lr_missing" in good[0]:
        mat_missing = np.vstack([r["log_lr_missing"] for r in good])
        row.update(_spread_metrics(mat_missing, prefix="miss_"))

    weights = np.array([r["path_weight"] for r in good])
    row["path_weight_std"] = float(np.nanstd(weights))
    row["path_weight_cv"] = float(np.nanstd(weights) / max(abs(np.nanmean(weights)), 1e-12))
    return row, mat, None


def run_sweep(args, executor=None):
    rows, matrices = [], {}
    for p in args.p_grid:
        conditions = ["none"] if p < 2 else list(args.missingness)
        for condition in conditions:
            for strategy in args.strategies:
                row, mat, _ = run_condition(p, condition, strategy, args, executor)
                print(f"  p={p:>2} {condition:>14} {strategy:>18}  "
                      f"valid={row['n_valid']:>3}  "
                      f"median_band={row['median_band']:.3f} "
                      f"median_iqr={row['median_iqr']:.3f}")
                rows.append(row)
                matrices[(p, condition, strategy)] = mat
    return rows, matrices


def flag_high_variance(rows, flag_multiple):
    """Flag conditions whose LR spread exceeds a fixed multiple of the
    p=1 / none / sample_specific baseline."""
    baseline = next((r["median_band"] for r in rows
                     if r["p"] == 1 and r["missingness"] == "none"
                     and r["strategy"] == "sample_specific"), np.nan)
    for r in rows:
        r["baseline_median_band"] = baseline
        r["variance_ratio"] = (r["median_band"] / baseline
                               if baseline and np.isfinite(baseline) and baseline > 0
                               else np.nan)
        r["flagged"] = bool(np.isfinite(r["variance_ratio"])
                            and r["variance_ratio"] > flag_multiple)
    return baseline


# ── Reporting ───────────────────────────────────────────────────────────────

def _report(rows, baseline, flag_multiple):
    print(f"\n{'═' * 110}")
    print("  BOOTSTRAP LR VARIANCE by (dimensionality × missingness × strategy)")
    print(f"{'═' * 110}")
    print(f"  {'p':>3}  {'missingness':>14}  {'strategy':>18}  {'band':>8}  "
          f"{'std':>8}  {'signbad':>8}  {'w_cv':>8}  {'ratio':>7}  "
          f"{'npat':>5}  {'valid':>5}  flag")
    for r in rows:
        print(f"  {r['p']:>3}  {r['missingness']:>14}  {r['strategy']:>18}  "
              f"{r['median_band']:>8.3f}  {r['median_iqr']:>8.3f}  "
              f"{r['median_std']:>8.3f}  "
              f"{r['sign_unstable_frac']:>8.3f}  {r['path_weight_cv']:>8.3f}  "
              f"{r['variance_ratio']:>7.2f}  {r['n_patterns']:>5}  "
              f"{r['n_valid']:>5}  {'***' if r['flagged'] else ''}")
    print(f"\n  baseline (p=1/none/sample_specific) median_band = {baseline:.3f}; "
          f"*** = >{flag_multiple}x baseline")


def _report_pct(pct_rows):
    if not pct_rows:
        print("\n  No conditions flagged -- percentile sensitivity skipped.")
        return
    print(f"\n{'═' * 110}")
    print("  PERCENTILE SENSITIVITY for flagged conditions (no refitting)")
    print(f"{'═' * 110}")
    print(f"  {'p':>3}  {'missingness':>14}  {'strategy':>18}  {'pair':>8}  "
          f"{'band':>8}  {'|cons|LR':>9}  {'retained':>9}  {'shrink':>8}")
    for r in pct_rows:
        pair = f"{r['path_percentile']}/{r['ben_percentile']}"
        print(f"  {r['p']:>3}  {r['missingness']:>14}  {r['strategy']:>18}  "
              f"{pair:>8}  {r['median_band']:>8.3f}  "
              f"{r['mean_abs_conservative_log_lr']:>9.3f}  "
              f"{r['evidence_retained_frac']:>9.3f}  {r['median_shrinkage']:>8.3f}")


def _write_csv(rows, stem):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{stem}_{ts}.csv"
    if rows:
        # Union of keys, not rows[0]'s: only conditions WITH missingness carry
        # the miss_* metrics (the second, pattern-bearing query set), so the
        # first row's keys are a strict subset of what later rows hold.
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-grid", type=int, nargs="+", default=[1, 2, 4, 16])
    parser.add_argument("--missingness", nargs="+", default=list(MISSINGNESS_CONDITIONS),
                        choices=list(MISSINGNESS_CONDITIONS))
    parser.add_argument("--strategies", nargs="+",
                        default=["unstratified", "sample_specific", "pattern_stratified"],
                        choices=["unstratified", "sample_specific", "pattern_stratified"])
    parser.add_argument("--n-bootstraps", type=int, default=10)
    parser.add_argument("--n-path", type=int, default=60)
    parser.add_argument("--n-benign", type=int, default=400)
    parser.add_argument("--n-query", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=20260918)
    parser.add_argument("--flag-multiple", type=float, default=3.0)
    parser.add_argument("--max-em-iters", type=int, default=10000,
                        help="EM iteration ceiling per fit. Fits that hit it are "
                             "truncated rather than converged; under heavy "
                             "missingness EM converges linearly at a rate set by "
                             "the fraction of missing information, so this bounds "
                             "runtime at the cost of stopping some fits early.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="-1 (default) uses all cores; 1 runs sequentially")
    args = parser.parse_args()

    print(f"p grid={args.p_grid} (p=1 -> scalar path, p>=2 -> CFUSN q=2)")
    print(f"missingness={args.missingness}  strategies={args.strategies}")
    print(f"n_bootstraps={args.n_bootstraps}  n_path={args.n_path}  "
          f"n_benign={args.n_benign}  n_query={args.n_query}  n_jobs={args.n_jobs}\n")

    if args.n_jobs and args.n_jobs != 1:
        with ProcessPoolExecutor(max_workers=None if args.n_jobs < 0 else args.n_jobs) as ex:
            rows, matrices = run_sweep(args, ex)
    else:
        rows, matrices = run_sweep(args, None)

    baseline = flag_high_variance(rows, args.flag_multiple)
    _report(rows, baseline, args.flag_multiple)

    pct_rows = []
    for r in rows:
        if not r["flagged"]:
            continue
        mat = matrices[(r["p"], r["missingness"], r["strategy"])]
        if mat.size == 0 or mat.shape[0] < 2:
            continue
        for pr in percentile_sensitivity(mat):
            pct_rows.append(dict(p=r["p"], missingness=r["missingness"],
                                 strategy=r["strategy"], **pr))
    _report_pct(pct_rows)

    _write_csv(rows, "sim_bootstrap_variance")
    _write_csv(pct_rows, "sim_bootstrap_variance_pctsens")
    print()


if __name__ == "__main__":
    main()
