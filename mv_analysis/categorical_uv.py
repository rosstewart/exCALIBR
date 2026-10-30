"""Univariate calibration of a categorical readout (e.g. TP53 oligomerization:
tetramer, dimer or monomer), the categorical analogue of an ExCALIBR fit.

With one categorical readout a mixture of categorical components is not
identifiable beyond each sample's level distribution, so the model reduces to:

  class distributions  P(level | P/LP) and P(level | B/LB), smoothed by a
                       Dirichlet pseudo-count (the design choice below)
  prior                gnomAD's level distribution fitted as a mixture of the
                       two, by the same EM as ExCALIBR
                       (point_ranges.estimate_prior_from_class_densities)
  evidence             each level's likelihood ratio over bootstrap
                       iterations (each sample resampled with replacement):
                       pathogenic points at the 5th percentile, benign at the
                       95th, converted with the Tavtigian constant of the
                       median prior, as ExCALIBR's evidence rule

Designs (smoothing of the class distributions):
  frequency  Jeffreys pseudo-count 0.5 per level
  shrunk     pseudo-counts tau * ((1 - lam) * pooled frequency + lam / L),
             the prior the multivariate categorical model uses
             (fit_utils/cfusn/categorical.py: tau 3, lam 0.1)

The pooled frequency comes from gnomAD (pool="gnomad", the default, as the
multivariate model's since 2026-10-28) or from every sample (pool="all", the
multivariate default before then). The paper's TP53 KawOligo calibration
uses pool="all" to match the stored TP53 categorical fits
(jobs_tp53_categorical_4c_tau0.1_100b_3f_102726); switch both to "gnomad" at
the next TP53 refit. On the paper's data both give points -1 / 0 / +4.
  ordinal    shrunk, with the likelihood ratio made non-decreasing over the
             level order (pool-adjacent-violators, weighted by counts), for
             readouts ordered by pathogenicity

Usage (TP53 KawOligo; writes <config.TP53_EXTRA_UV_CALIB_DIR>/TP53_KawOligo_categorical/):
  python mv_analysis/categorical_uv.py [--pool all|gnomad]
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant  # noqa: E402
from src.assay_calibration.fit_utils.point_ranges import estimate_prior_from_class_densities  # noqa: E402

DESIGNS = ("frequency", "shrunk", "ordinal")
TAU, LAM = 3.0, 0.1


def _class_probs(counts, pooled, design):
    L = len(counts)
    if design == "frequency":
        alpha = np.full(L, 0.5)
    else:
        alpha = TAU * ((1 - LAM) * pooled + LAM / L)
    p = counts + alpha
    return p / p.sum()


def _monotone_lr(lr, weights):
    """Pool-adjacent-violators: the weighted least-squares non-decreasing fit
    to log LR over the level order."""
    blocks = [[float(np.log(v)), float(w), 1] for v, w in zip(lr, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            w = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out += [np.exp(v)] * n
    return np.asarray(out)


def _one(codes_p, codes_b, codes_g, L, design, pool="gnomad"):
    cp, cb, cg = (np.bincount(c, minlength=L).astype(float) for c in (codes_p, codes_b, codes_g))
    pooled = cg if pool == "gnomad" and cg.sum() > 0 else cp + cb + cg
    pooled = pooled / max(pooled.sum(), 1)
    fp, fb = _class_probs(cp, pooled, design), _class_probs(cb, pooled, design)
    lr = fp / fb
    if design == "ordinal":
        lr = _monotone_lr(lr, cp + cb + 1.0)
    prior = estimate_prior_from_class_densities(fp[codes_g], fb[codes_g], mode="standard")
    return lr, prior


def calibrate(codes_p, codes_b, codes_g, level_names, design, n_boot=1000, seed=0,
              path_percentile=5.0, pool="gnomad"):
    """{design, level_names, prior, tavtigian_c, points{level: int},
    lr_median{level}, lr_lo{level}, lr_hi{level}, n{P/LP, B/LB, gnomAD}}."""
    L = len(level_names)
    rng = np.random.default_rng(seed)
    lrs, priors = [], []
    for _ in range(n_boot):
        bp = rng.choice(codes_p, len(codes_p)) if len(codes_p) else codes_p
        bb = rng.choice(codes_b, len(codes_b)) if len(codes_b) else codes_b
        bg = rng.choice(codes_g, len(codes_g)) if len(codes_g) else codes_g
        lr, prior = _one(bp, bb, bg, L, design, pool)
        lrs.append(lr)
        priors.append(prior)
    lrs, priors = np.asarray(lrs), np.asarray(priors, float)
    ok = np.isfinite(priors) & (priors > 0) & (priors < 1)
    prior = float(np.median(priors[ok]))
    C = float(get_tavtigian_constant(prior))
    lo = np.percentile(lrs[ok], path_percentile, axis=0)
    hi = np.percentile(lrs[ok], 100 - path_percentile, axis=0)
    points = {}
    for lv, a, b in zip(level_names, lo, hi):
        p = int(np.clip(np.floor(8 * np.log(a) / np.log(C)), 0, 8)) if a > 1 else 0
        n = int(np.clip(np.ceil(8 * np.log(b) / np.log(C)), -8, 0)) if b < 1 else 0
        points[lv] = p if p > 0 else n
    return dict(design=design, pool=pool, level_names=list(level_names), prior=prior, tavtigian_c=C,
                n_valid_bootstraps=int(ok.sum()), points=points,
                lr_median=dict(zip(level_names, np.median(lrs[ok], axis=0).round(4).tolist())),
                lr_lo=dict(zip(level_names, lo.round(4).tolist())), lr_hi=dict(zip(level_names, hi.round(4).tolist())),
                n={"P/LP": int(len(codes_p)), "B/LB": int(len(codes_b)), "gnomAD": int(len(codes_g))})


def _mcc(points, labels):
    tp = int(((points > 0) & (labels == 1)).sum()); tn = int(((points < 0) & (labels == 0)).sum())
    fp = int(((points > 0) & (labels == 0)).sum()); fn = int(((points < 0) & (labels == 1)).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0, (tp + tn + fp + fn) / max(len(labels), 1)


def main(pool="all"):
    from mv_analysis import config
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_multiscoresets_for_gene_set("tp53")["TP53"]
    dim = "KawOligo"
    levels = ms.categorical_dims[dim]
    x = np.asarray(ms.scores, float)[:, list(ms.dataset_names).index(dim)]
    sa = np.asarray(ms.sample_assignments, bool)
    m = np.isfinite(x)
    codes = {s: x[m & sa[:, i]].astype(int) for i, s in enumerate(("P/LP", "B/LB", "gnomAD"))}
    ctrl = m & (sa[:, 0] | sa[:, 1])
    labels = sa[ctrl, 0].astype(int)
    out_dir = Path(config.TP53_EXTRA_UV_CALIB_DIR) / "TP53_KawOligo_categorical"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"KawOligo: {len(codes['P/LP'])} P/LP, {len(codes['B/LB'])} B/LB, {len(codes['gnomAD'])} gnomAD; levels {levels}")
    for lv_i, lv in enumerate(levels):
        print(f"  {lv}: P/LP {int((codes['P/LP'] == lv_i).sum())}, B/LB {int((codes['B/LB'] == lv_i).sum())}, "
              f"gnomAD {int((codes['gnomAD'] == lv_i).sum())}")
    for design in DESIGNS:
        res = calibrate(codes["P/LP"], codes["B/LB"], codes["gnomAD"], levels, design, pool=pool)
        pts = np.array([res["points"][levels[int(c)]] for c in x[ctrl]])
        res["controls_mcc"], res["controls_determinate"] = _mcc(pts, labels)
        (out_dir / f"TP53_KawOligo_{design}_calibration.json").write_text(json.dumps(res, indent=1))
        print(f"{design:9s} prior {res['prior']:.3f}  points {res['points']}  "
              f"LR median {res['lr_median']}  controls MCC {res['controls_mcc']:.2f} "
              f"(determinate {100 * res['controls_determinate']:.0f}%)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("all", "gnomad"), default="all",
                    help="pooled frequency of the shrunk prior (all: matches the stored TP53 fits)")
    main(ap.parse_args().pool)
