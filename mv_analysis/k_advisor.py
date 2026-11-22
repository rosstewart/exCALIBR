"""Pre-fit component diagnostic (advisory only -- nothing reads its output).

Input: a small K-grid run prepared like production (same rows, strata or
standard samples, and bootstrap scheme) with ``--components 2 3 4 5 6`` and ~10 bootstraps
(hpc/prepare.py mv_all ...). Every bootstrap fit records ``val_ll``, the
held-out log-likelihood of its out-of-bag rows, and a given bootstrap seed
has the same split at every K, so K is compared on identical held-out rows.

Per gene x gene set, one page:
  (a) held-out log-likelihood vs K (mean +- SE over bootstraps, paired
      differences to the best K), with the suggested K* marked;
  (b) BIC and ICL vs K, from each bootstrap's parameters on all rows;
  (c) k-means silhouette vs K on standardized complete rows (rows with every
      readout; skipped when there are fewer than 30);
  (d) rows per stratum (calibration strata highlighted) and rows per readout;
  (e) the fitted components for K = 2..6 on the gene's main readout x
      AlphaMissense (or its two best-covered readouts), from the first valid
      bootstrap, as 1-sigma ellipses of each component's 2D marginal, with
      the labelled controls.
A summary CSV lists, per gene x gene set, the suggestion K* = the smallest K
whose mean held-out log-likelihood is within one paired SE of the best,
capped so every calibration stratum has >= 10 labelled rows per component.

Usage (with MV_PAPER_FITS_DIR pointing at the K-grid run):
    python -m mv_analysis.k_advisor --out <dir> [--gene-sets combined integrated ...]
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _fits_by_k(raw_boots):
    """{K: {seed: (val_ll, fit dict)}} over unconstrained configs."""
    out = {}
    for seed, bd in raw_boots.items():
        for cfg, fr in bd.items():
            if fr is None or not cfg.endswith("_unc"):
                continue
            k = int(cfg.split("c_")[0])
            out.setdefault(k, {})[int(seed)] = (fr.get("val_ll", np.nan), fr.get("fit", fr))
    return out


def _heldout_table(fits):
    ks = sorted(fits)
    seeds = sorted(set.intersection(*[set(fits[k]) for k in ks])) if ks else []
    V = np.array([[fits[k][s][0] for k in ks] for s in seeds], dtype=float)
    rows = []
    if len(V) == 0:
        return pd.DataFrame(), None
    best = int(np.nanargmax(np.nanmean(V, axis=0)))
    for j, k in enumerate(ks):
        d = V[:, j] - V[:, best]
        rows.append(dict(K=k, n_boot=int(np.isfinite(V[:, j]).sum()), heldout_ll=np.nanmean(V[:, j]),
                         heldout_se=np.nanstd(V[:, j], ddof=1) / np.sqrt(np.isfinite(V[:, j]).sum()),
                         diff_to_best=np.nanmean(d),
                         diff_se=np.nanstd(d, ddof=1) / np.sqrt(np.isfinite(d).sum()) if j != best else 0.0))
    return pd.DataFrame(rows), ks[best]


def _loglik_terms(an, fit, X):
    """Per-row log-density under the fitted mixture of all rows (population
    weights averaged over samples) and the per-row component posteriors, for
    BIC/ICL."""
    from scipy.special import logsumexp
    from src.assay_calibration.multivariate_analysis.mv_calibration import _comp_logpdf
    P = fit["component_params"]
    W = np.asarray(fit["weights"], dtype=float)
    w = W.mean(axis=0)
    lf = np.stack([np.log(w[c] + 1e-300) + _comp_logpdf(X, P, c) for c in range(len(P))])
    ll = logsumexp(lf, axis=0)
    post = np.exp(lf - ll)
    return ll, post


def _n_params(fit, D):
    K = len(fit["component_params"])
    q = int(fit.get("latent_q", 1) or 1)
    S = np.asarray(fit["weights"]).shape[0]
    return K * (D + D * q + D * (D + 1) / 2) + S * (K - 1)


def _silhouette(X, ks, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    C = X[np.isfinite(X).all(axis=1)]
    if len(C) < 30:
        return {k: np.nan for k in ks}
    C = (C - C.mean(0)) / np.where(C.std(0) > 0, C.std(0), 1)
    out = {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(C)
        out[k] = silhouette_score(C, lab, sample_size=min(len(C), 3000), random_state=seed)
    return out


def _ellipse(ax, mu, cov, color):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    ax.add_patch(Ellipse(mu, 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]), angle=ang, fill=False, color=color, lw=1.2))


def _component_moments(params, c, dims):
    """Mean and covariance of component c restricted to ``dims`` (skew-normal
    / CFUSN moments: mean mu + Delta*sqrt(2/pi), cov Gamma + Delta Delta^T (1 - 2/pi))."""
    mu, Delta, Gamma = params[c][:3] if isinstance(params[c], (tuple, list)) else (params[c].mu, params[c].Delta, params[c].Gamma)
    mu = np.asarray(mu, float); Gamma = np.asarray(Gamma, float); Delta = np.asarray(Delta, float)
    if Delta.ndim == 1:
        Delta = Delta[:, None]
    m = mu + Delta.sum(axis=1) * np.sqrt(2 / np.pi)
    S = Gamma + Delta @ Delta.T * (1 - 2 / np.pi)
    d = list(dims)
    return m[d], S[np.ix_(d, d)]


def gene_page(gene, gene_set, dataset, out_dir, pdf):
    import matplotlib.pyplot as plt
    from mv_analysis import config
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.fit_utils.strata import ROLES
    rj = config.PAPER_RESULTS_JSON
    cl = int(dataset.split("_cluster")[1].split("_")[0]) - 1 if "_cluster" in dataset else 0
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(gene_set, gene, cluster_idx=cl, results_json=rj)
        an = build_gene_set_analysis(ms, gene.lower(), rj, dataset_name=dataset, gene_set=gene_set)
    fits = _fits_by_k(an.raw_boots)
    if not fits:
        return None
    ho, k_best = _heldout_table(fits)
    X = np.asarray(ms.scores, float)
    obs = np.isfinite(X).any(axis=1)
    Xo = X[obs]
    names = list(ms.dataset_names)
    # BIC / ICL per K over bootstraps (all observed rows, approximate: population-averaged weights)
    bic, icl = {}, {}
    for k in sorted(fits):
        b_, i_ = [], []
        for seed, (_, fr) in list(fits[k].items())[:5]:
            fit = an._reconstitute_params(fr)
            if fit is None:
                continue
            ll, post = _loglik_terms(an, fit, Xo)
            p = _n_params(fit, X.shape[1])
            b = -2 * ll.sum() + p * np.log(len(Xo))
            ent = -np.sum(post * np.log(post + 1e-300))
            b_.append(b); i_.append(b + 2 * ent)
        bic[k], icl[k] = (np.mean(b_) if b_ else np.nan), (np.mean(i_) if i_ else np.nan)
    sil = _silhouette(X, sorted(fits))
    # calibration strata sizes -> cap
    strata = getattr(an, "strata", None)
    cap_note, k_cap = "", max(fits)
    if strata is not None:
        M = strata.composition(an.strata_arm)
        cal = [c for c in range(len(strata.names)) if (M[:2, c] > 0).any()]
        nmin = min(strata.counts[c] for c in cal) if cal else 0
        tot_p = M[0].sum(); tot_b = M[1].sum()
        k_cap = int(max(2, min(tot_p, tot_b) // 10))
        cap_note = f"P/LP composite {int(tot_p)}, B/LB composite {int(tot_b)}; smallest calibration stratum {int(nmin)}"
    else:  # standard model: the same cap from the P/LP and B/LB samples
        SA = np.asarray(ms._sample_assignments, bool)
        tot_p, tot_b = int(SA[:, 0].sum()), int(SA[:, 1].sum())
        k_cap = int(max(2, min(tot_p, tot_b) // 10))
        cap_note = f"P/LP sample {tot_p}, B/LB sample {tot_b} (standard model)"
    within = ho[(ho.diff_to_best >= -ho.diff_se.where(ho.diff_se > 0, ho.heldout_se))]
    k_star = int(min(within.K.min(), k_cap)) if len(within) else k_best

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 6)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.errorbar(ho.K, ho.heldout_ll, yerr=ho.heldout_se, marker="o", color="#333")
    ax.axvline(k_star, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("K"); ax.set_ylabel("held-out log-likelihood (mean over bootstraps)")
    ax.set_title(f"(a) held-out LL; K*={k_star} (suggestion only)", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[0, 2])
    kk = sorted(fits)
    ax.plot(kk, [bic[k] for k in kk], marker="o", label="BIC")
    ax.plot(kk, [icl[k] for k in kk], marker="s", label="ICL")
    ax.legend(fontsize=8); ax.set_xlabel("K"); ax.set_title("(b) BIC / ICL (lower better)", fontsize=10, loc="left")
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(kk, [sil[k] for k in kk], marker="o", color="#2e86c1")
    ax.set_xlabel("K"); ax.set_title(f"(c) k-means silhouette\n(complete rows: {int(np.isfinite(X).all(1).sum())})",
                                    fontsize=10, loc="left")
    ax = fig.add_subplot(gs[0, 4:6])
    if strata is not None:
        M = strata.composition(an.strata_arm)
        cal = (M[:2] > 0).any(axis=0)
        order = np.argsort(-strata.counts)
        ax.barh(range(len(order)), strata.counts[order], color=np.where(cal[order], "#c0392b", "#aaa"))
        ax.set_yticks(range(len(order))); ax.set_yticklabels([strata.names[i] for i in order], fontsize=7)
        ax.set_xscale("log"); ax.invert_yaxis()
        ax.set_title(f"(d) rows per stratum (red = calibration, arm {an.strata_arm})\n{cap_note}", fontsize=9, loc="left")
    else:
        SA = np.asarray(ms._sample_assignments, bool)
        sn = list(getattr(ms, "sample_names", None) or [f"sample {i}" for i in range(SA.shape[1])])
        ax.barh(range(SA.shape[1]), SA.sum(0), color=["#c0392b" if i < 2 else "#aaa" for i in range(SA.shape[1])])
        ax.set_yticks(range(SA.shape[1])); ax.set_yticklabels(sn[:SA.shape[1]], fontsize=7)
        ax.set_xscale("log"); ax.invert_yaxis()
        ax.set_title(f"(d) rows per sample (standard model; overlapping)\n{cap_note}", fontsize=9, loc="left")
    # (e) components on two readouts
    pred = [i for i, n in enumerate(names) if n == "AlphaMissense"]
    func = [i for i, n in enumerate(names) if n not in ("REVEL", "MutPred2", "AlphaMissense")]
    cov = np.isfinite(X).sum(0)
    if pred and func:
        dims = [max(func, key=lambda i: cov[i]), pred[0]]
    else:
        dims = list(np.argsort(-cov)[:2])
    sa = np.asarray(ms._sample_assignments, bool)
    P_, B_ = sa[:, 0], sa[:, 1]
    for j, k in enumerate(kk[:6]):
        ax = fig.add_subplot(gs[1, j])
        both = np.isfinite(X[:, dims]).all(1)
        ax.scatter(X[both & B_, dims[0]], X[both & B_, dims[1]], s=4, c="#1D7AAB", alpha=0.5)
        ax.scatter(X[both & P_, dims[0]], X[both & P_, dims[1]], s=4, c="#CA7682", alpha=0.6)
        fr = next(iter(fits[k].values()))[1]
        fit = an._reconstitute_params(fr)
        if fit is not None:
            cols = plt.cm.tab10(np.arange(k))
            for c in range(k):
                try:
                    m, S = _component_moments(fit["component_params"], c, dims)
                    _ellipse(ax, m, S, cols[c])
                except Exception:
                    pass
        ax.set_title(f"(e) K={k}", fontsize=10, loc="left")
        ax.set_xlabel(names[dims[0]], fontsize=8)
        if j == 0:
            ax.set_ylabel(names[dims[1]], fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{gene} -- {gene_set} ({dataset}): rows {int(obs.sum())}, readouts {len(names)}", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    fig.savefig(Path(out_dir) / f"k_{dataset}.png", dpi=90)
    plt.close(fig)
    rec = dict(gene=gene, gene_set=gene_set, dataset=dataset, K_suggested=k_star, K_best_heldout=k_best, K_cap=k_cap,
               **{f"heldout_K{int(r.K)}": r.heldout_ll for r in ho.itertuples()},
               **{f"diff_K{int(r.K)}": r.diff_to_best for r in ho.itertuples()},
               **{f"BIC_K{k}": bic[k] for k in kk}, **{f"ICL_K{k}": icl[k] for k in kk},
               **{f"silhouette_K{k}": sil[k] for k in kk})
    return rec


def main():
    from matplotlib.backends.backend_pdf import PdfPages
    from mv_analysis import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gene-sets", nargs="+", default=None)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with gzip.open(config.PAPER_RESULTS_JSON, "rt") as f:
        keys = sorted(json.load(f).keys())
    recs = []
    with PdfPages(out / "k_diagnostic_pages.pdf") as pdf:
        for ds in keys:
            gene = ds.split("_")[0]
            rest = ds[len(gene) + 1:]
            gs = ("fgfr" if "_fgfr_" in ds else "predictor" if rest.startswith("predictors") else
                  "integrated" if rest.startswith(("mv", "cluster")) else rest.split("_")[0])
            if args.gene_sets and gs not in args.gene_sets:
                continue
            try:
                r = gene_page(gene, gs, ds, out, pdf)
            except Exception as e:  # report, keep going
                print(f"{ds}: {e}")
                continue
            if r:
                recs.append(r)
                print(f"{ds}: K* = {r['K_suggested']} (best held-out {r['K_best_heldout']})")
    pd.DataFrame(recs).to_csv(out / "k_diagnostic_summary.csv", index=False)


if __name__ == "__main__":
    main()
