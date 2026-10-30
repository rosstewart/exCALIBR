"""Supervised classifier baselines for ExCALIBR-MV: a classifier trained on
labelled (P/LP vs B/LB) variants, whose score is turned into a log likelihood
ratio so it can be thresholded with the same evidence thresholds as the joint
model.

Calibration used for the paper (fit_excalibr_predict): the labelled
training variants are split, within each class, into a fitting set (2/3)
and a calibration set (1/3). The classifier is trained on the fitting set;
ExCALIBR (one 3-component fit, best of 8 restarts, monotonic, benign
reference averaging B/LB and synonymous, as in the ExCALIBR pipeline) is then
fitted to its scores on the calibration controls together with the
population and synonymous variants, from which ExCALIBR's own prior and log
LR+ curve give each evaluation variant's log LR+. Rows used to fit the
classifier, and evaluation rows, are excluded from every calibration sample.

Secondary (fit_calibrate_predict, "-local" in the outputs): the labelled
training variants are split, within each class, into a fitting set (2/3) and
a calibration set (1/3). The classifier is trained on the fitting set only;
its score is then calibrated on the calibration set, which it never saw, by a
local likelihood ratio in the manner of Pejaver et al. (2022): for a query
score, the fraction of each class's calibration variants among the nearest
calibration scores (max(10, 10%) of them),

    LR+ = [(k_P + 0.5) / (N_P + 1)] / [(k_B + 0.5) / (N_B + 1)],

with Jeffreys pseudo-counts so a window of one class gives a finite ratio.
This does not assume the classifier's probabilities are calibrated. The
evaluation variants are disjoint from both sets (out of bag).

fit_predict (kept for reference, "-raw" in the outputs) instead assumes the
probability is calibrated and removes the training class balance,
log LR+ = logit(p) - logit(training P/LP fraction).

Ported from the 2026-10-15 baselines analysis (baselines_101526); models:
  B-LR         L2 logistic regression, strength by CV log-loss; missing scores
               set to the training median, plus per-assay missingness indicators
  B-LR-nomiss  the same without missingness indicators
  B-spline     logistic regression on cubic-spline features (with indicators)
  B-GB         gradient-boosted trees (native missing-value handling)
  B-GB-cal     B-GB with sigmoid (Platt) recalibration
"""
import numpy as np

MODELS = ("B-LR", "B-LR-nomiss", "B-spline", "B-GB", "B-GB-cal")
CLIP = 1e-9


def _design_lr(Xtr, Xte, miss_ind=True, splines=False):
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing_any = np.isnan(Xtr).any(0)
    spl = None
    if splines:
        from sklearn.preprocessing import SplineTransformer
        spl = SplineTransformer(n_knots=5, degree=3, extrapolation="linear").fit(np.where(np.isnan(Xtr), med, Xtr))

    def f(X):
        miss = np.isnan(X)
        Z = np.where(miss, med, X)
        if spl is not None:
            Z = spl.transform(Z)
        keep = missing_any if miss_ind else np.zeros(X.shape[1], bool)
        return np.hstack([Z, miss[:, keep].astype(float)])
    A, B = f(Xtr), f(Xte)
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (A - mu) / sd, (B - mu) / sd


def fit_predict(model, Xtr, ytr, Xte, seed=0, min_class=5):
    """log LR+ for Xte from `model` trained on (Xtr, ytr), assuming its
    probability is calibrated; None when the smaller class has fewer than
    `min_class` training variants."""
    ytr = np.asarray(ytr, int)
    p = _fit_proba(model, Xtr, ytr, Xte, seed, min_class)
    if p is None:
        return None
    pi = ytr.mean()
    return np.log(p / (1 - p)) - np.log(pi / (1 - pi))


def local_log_lr(cal_s, cal_y, q_s, frac=0.1, min_k=10):
    """Local log likelihood ratio at each query score from the nearest
    calibration scores (Pejaver et al. 2022 style; Jeffreys pseudo-counts)."""
    cal_s, cal_y, q_s = np.asarray(cal_s, float), np.asarray(cal_y, int), np.asarray(q_s, float)
    n = len(cal_s)
    k = min(n, max(min_k, int(np.ceil(frac * n))))
    n_p, n_b = cal_y.sum(), (1 - cal_y).sum()
    idx = np.argsort(np.abs(q_s[:, None] - cal_s[None, :]), axis=1, kind="stable")[:, :k]
    k_p = cal_y[idx].sum(1)
    k_b = k - k_p
    return np.log((k_p + 0.5) / (n_p + 1)) - np.log((k_b + 0.5) / (n_b + 1))


def fit_calibrate_predict(model, Xtr, ytr, Xte, seed=0, min_class=5, min_cal=3, cal_frac=1 / 3):
    """log LR+ for Xte: `model` trained on a fitting subset of (Xtr, ytr) and
    its score calibrated on the held-out rest (local_log_lr). None when the
    fitting set has fewer than `min_class` or the calibration set fewer than
    `min_cal` variants of either class."""
    ytr = np.asarray(ytr, int)
    rng = np.random.default_rng(seed)
    cal = np.zeros(len(ytr), bool)
    for c in (0, 1):
        i = np.flatnonzero(ytr == c)
        cal[rng.choice(i, int(round(cal_frac * len(i))), replace=False)] = True
    fit = ~cal
    if min(ytr[cal].sum(), (1 - ytr[cal]).sum()) < min_cal:
        return None
    p = _fit_proba(model, Xtr[fit], ytr[fit], np.vstack([Xtr[cal], Xte]), seed, min_class)
    if p is None:
        return None
    s = np.log(p / (1 - p))
    return local_log_lr(s[:cal.sum()], ytr[cal], s[cal.sum():])


def _fit_proba(model, Xtr, ytr, Xte, seed=0, min_class=5):
    """Classifier probability for Xte from `model` trained on (Xtr, ytr);
    None when the smaller class has fewer than `min_class` training variants."""
    import warnings
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    ytr = np.asarray(ytr, int)
    n_min = int(min(ytr.sum(), (1 - ytr).sum()))
    if n_min < min_class:
        return None
    cv = int(max(2, min(5, n_min)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model in ("B-LR", "B-LR-nomiss", "B-spline"):
            A, B = _design_lr(Xtr, Xte, miss_ind=(model != "B-LR-nomiss"), splines=(model == "B-spline"))
            clf = LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=cv, scoring="neg_log_loss", max_iter=5000)
            p = clf.fit(A, ytr).predict_proba(B)[:, 1]
        else:
            early = len(ytr) >= 60  # too few rows to hold out an early-stopping split otherwise
            gb = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=500 if early else 100, l2_regularization=1.0,
                min_samples_leaf=max(2, min(5, n_min // 2), n_min // 10), early_stopping=early,
                validation_fraction=0.15, n_iter_no_change=20, scoring="loss", random_state=seed)
            if model == "B-GB":
                p = gb.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            else:
                p = CalibratedClassifierCV(gb, method="sigmoid", cv=min(3, cv)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return np.clip(p, CLIP, 1 - CLIP)


def fit_excalibr_predict(model, X, y, sa, fit_rows, cal_rows, te_rows, exclude, seed=0, min_class=5, min_cal=3):
    """(log LR+ for te_rows, prior): `model` trained on fit_rows, its logit
    score calibrated by one ExCALIBR fit on the calibration samples -- sa's
    P/LP and B/LB columns restricted to cal_rows, its population and
    synonymous columns -- with `exclude` rows (evaluation and fitting rows)
    and unscored rows removed. None when a class is too small."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    from src.assay_calibration.fit_utils.point_ranges import get_fit_prior, compute_single_fit_log_densities
    y = np.asarray(y, int)
    S = np.asarray(sa, bool).copy()
    S[:, 0] &= cal_rows
    S[:, 1] &= cal_rows
    S &= (~exclude & ~np.isnan(X).all(1))[:, None]
    if min(S[:, 0].sum(), S[:, 1].sum()) < min_cal:
        return None
    rows = S.any(1)
    p = _fit_proba(model, X[fit_rows], y[fit_rows], np.vstack([X[rows], X[te_rows]]), seed, min_class)
    if p is None:
        return None
    s = np.log(p / (1 - p))
    s_cal, s_te = s[:rows.sum()], s[rows.sum():]
    Sr = S[rows]
    syn = 3 if Sr[:, 3].any() else None
    models, best, _ = Fit(BasicScoreset(s_cal, Sr)).run([3], num_fits=8, bootstrap=False, check_monotonic=True,
                                                       core_limit=1)
    if best is None:
        return None
    fit = {"fit": models[best]}
    prior, _ = get_fit_prior(fit, s_cal, "avg", 0, 1, 2, syn, sample_assignments=Sr)
    if not (np.isfinite(prior) and 0 < prior < 1):
        return None
    grid = np.linspace(s.min(), s.max(), 512)
    lfp, lfb = compute_single_fit_log_densities(fit, prior, grid, "avg", pathogenic_idx=0, benign_idx=1,
                                                gnomad_idx=2, synonymous_idx=syn)[:2]
    return np.interp(s_te, grid, lfp - lfb), float(prior)


SAMPLE_ROLES = {"pathogenic": ("Pathogenic/Likely Pathogenic", "P/LP"),
                "benign": ("Benign/Likely Benign", "B/LB"),
                "population": ("population", "gnomAD"),
                "synonymous": ("Synonymous", "synonymous")}


def sample_columns(sample_names):
    """{role: column index} of the calibration samples, found by name: models
    differ in which samples they have and in what order (TP53's fourth is its
    reduced-penetrance sample, which is not a calibration sample)."""
    names = [str(n) for n in sample_names]
    return {role: next((names.index(a) for a in alias if a in names), None) for role, alias in SAMPLE_ROLES.items()}


def fit_excalibr_crossfit(model, X, y, sa, sample_names, tr_rows, te_rows, exclude, seed=0, k=5,
                          min_class=5, min_cal=3, return_fit=False):
    """log LR+ for te_rows from `model`, its logit score calibrated by one
    ExCALIBR fit, by cross-fitting with no refit: the tr_rows controls are
    split into k stratified folds and k classifiers are trained, each on all
    folds but one. Every calibration score comes from one of these
    classifiers and never from one trained on that variant: each control is
    scored by the classifier that left its fold out, and each in-bag
    population or synonymous variant (unlabelled, so in no training set) by
    the classifier of a randomly assigned fold. A test variant is scored by
    all k classifiers; each score is converted to log LR+ by the one ExCALIBR
    curve and the k values are averaged. Calibration and test scores thus come
    from the same classifiers. Returns (log LR+, prior), or with return_fit
    also the calibration scores, their samples and the fitted curve; a string
    giving the reason when it cannot be done."""
    from sklearn.model_selection import StratifiedKFold
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    from src.assay_calibration.fit_utils.point_ranges import get_fit_prior, compute_single_fit_log_densities
    y = np.asarray(y, int)
    tr_rows, te_rows, exclude = (np.asarray(m_, bool) for m_ in (tr_rows, te_rows, exclude))
    col = sample_columns(sample_names)
    if col["pathogenic"] is None or col["benign"] is None or col["population"] is None:
        return "model lacks a P/LP, B/LB or population sample"
    tr = np.flatnonzero(tr_rows)
    n_p, n_b = int(y[tr].sum()), int((1 - y[tr]).sum())
    if min(n_p, n_b) < min_class:
        return f"too few controls in bag (P/LP {n_p}, B/LB {n_b}; need {min_class} each)"
    k = int(min(k, n_p, n_b))
    fold_min = max(2, (min_class * (k - 1)) // k)  # each fold classifier trains on (k-1)/k of them
    # calibration samples: in-bag controls and in-bag population / synonymous variants
    S = np.zeros((len(y), 4), bool)
    S[tr, 0] = y[tr] == 1
    S[tr, 1] = y[tr] == 0
    S[:, 2] = np.asarray(sa, bool)[:, col["population"]]
    if col["synonymous"] is not None:
        S[:, 3] = np.asarray(sa, bool)[:, col["synonymous"]]
    S[:, 2:] &= ~exclude[:, None]
    S &= ~np.isnan(X).all(1)[:, None]
    rows = S.any(1)
    other = np.flatnonzero(rows & ~tr_rows)
    te = np.flatnonzero(te_rows)
    rng = np.random.default_rng(seed)
    other_fold = rng.integers(0, k, len(other))
    s = np.full(len(y), np.nan)
    s_te = np.full((k, len(te)), np.nan)
    for f, (f_tr, f_te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(tr, y[tr])):
        q = np.r_[tr[f_te], other[other_fold == f], te]
        p = _fit_proba(model, X[tr[f_tr]], y[tr[f_tr]], X[q], seed, fold_min)
        if p is None:
            return f"too few controls in a cross-fitting fold ({k} folds)"
        lg = np.log(p / (1 - p))
        n1 = len(f_te) + int((other_fold == f).sum())
        s[q[:n1]] = lg[:n1]
        s_te[f] = lg[n1:]
    s_cal, Sr = s[rows], S[rows]
    if not Sr[:, 3].any():
        Sr = Sr[:, :3]
    syn = 3 if Sr.shape[1] == 4 else None
    if min(Sr[:, 0].sum(), Sr[:, 1].sum()) < min_cal:
        return "too few calibration controls"
    models, best, _ = Fit(BasicScoreset(s_cal, Sr)).run([3], num_fits=8, bootstrap=False, check_monotonic=True,
                                                       core_limit=1)
    if best is None:
        return "ExCALIBR fit failed (no monotonic fit in 8 restarts)"
    fit = {"fit": models[best]}
    w = models[best].get("weights") if isinstance(models[best], dict) else getattr(models[best], "weights", None)
    if w is None:
        return "ExCALIBR fit degenerate (no sample weights)"
    prior, _ = get_fit_prior(fit, s_cal, "avg", 0, 1, 2, syn, sample_assignments=Sr)
    if not (np.isfinite(prior) and 0 < prior < 1):
        # get_fit_prior returns NaN when its EM prior saturates (<= 0.001 or
        # >= 0.999), the fits ExCALIBR discards from its own bootstraps
        return "ExCALIBR prior saturated (<= 0.001 or >= 0.999)" if np.isnan(prior) else f"invalid ExCALIBR prior ({prior})"
    grid = np.linspace(min(np.nanmin(s_cal), np.nanmin(s_te)), max(np.nanmax(s_cal), np.nanmax(s_te)), 512)
    lfp, lfb = compute_single_fit_log_densities(fit, prior, grid, "avg", pathogenic_idx=0, benign_idx=1,
                                                gnomad_idx=2, synonymous_idx=syn)[:2]
    llr = np.interp(s_te, grid, lfp - lfb).mean(0)
    if return_fit:
        return llr, float(prior), dict(scores=s_cal, samples=Sr, grid=grid, log_fp=lfp, log_fb=lfb,
                                       log_lr=lfp - lfb, fit=fit, test_scores=s_te, n_folds=k)
    return llr, float(prior)
