"""Supervised classifier baselines for ExCALIBR-MV: a classifier trained on
labelled (P/LP vs B/LB) variants, whose probability is converted to a log
likelihood ratio by removing the training class balance,

    log LR+ = logit(p) - logit(training P/LP fraction),

so it can be thresholded with the same evidence thresholds as the joint model.
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
    """log LR+ for Xte from `model` trained on (Xtr, ytr); None when the
    smaller class has fewer than `min_class` training variants."""
    import warnings
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    ytr = np.asarray(ytr, int)
    n_min = int(min(ytr.sum(), (1 - ytr).sum()))
    if n_min < min_class:
        return None
    pi = ytr.mean()
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
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p)) - np.log(pi / (1 - pi))
