"""E9b: allele-frequency-weighted (per-encounter) prior pooled across genes.

Per gene, one dataset (the one with the most gnomAD variants) and one baseline bootstrap fit (the one
whose unweighted prior is the median) give f_P, f_B at the gene's gnomAD variants. One shared prior q is
then fitted by the weighted label-shift EM over all genes' variants together,
    q <- sum_g sum_v a_v r_v(q) / sum_g sum_v a_v,
for the same weights as af_prior.py. Interval: 95% percentile over 1000 resamples of genes.
Also reports which variants carry the weight (share of total AF in the top 10 variants).

Usage:  python -m analysis.missense_sensitivity.af_prior_pooled
"""
from __future__ import annotations

import gzip
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _gene_densities(ds, df_ds, fits_ds, cal):
    from analysis.missense_sensitivity.af_prior import _row_af, PLP, BLB, SYN, weighted_prior_em
    from analysis.prior_simulation.simulate_priors import fit_gnomad_densities
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release))
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    if PLP not in names or not ({BLB, SYN} & set(names)):
        return None
    gname = "population" if "population" in names else "gnomAD"
    idx = dict(pathogenic_idx=names.index(PLP), benign_idx=names.index(BLB) if BLB in names else None,
               gnomad_idx=names.index(gname), synonymous_idx=names.index(SYN) if SYN in names else None)
    bm = cal.get("benign_method", "avg")
    if idx["synonymous_idx"] is None and bm in ("avg", "synonymous"):
        bm = "benign"
    if idx["benign_idx"] is None and bm in ("avg", "benign"):
        bm = "synonymous"
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    af = _row_af(ss, df_ds)[sa[:, idx["gnomad_idx"]]]
    dens = []
    for rec in fits_ds.values():
        try:
            f_p, f_b = fit_gnomad_densities(next(iter(rec.values()))["fit"], sc, sa, bm, **idx)
        except Exception:
            continue
        dens.append((weighted_prior_em(f_p, f_b, np.ones(len(f_p))), f_p, f_b))
    dens = [d for d in dens if np.isfinite(d[0])]
    if not dens:
        return None
    dens.sort(key=lambda d: d[0])
    q, f_p, f_b = dens[len(dens) // 2]
    return dict(dataset=ds, f_p=f_p, f_b=f_b, af=af, q_unweighted=q)


def pooled_em(genes, wfun, q0=0.2, tol=1e-10, it=20000):
    fp = np.concatenate([g["f_p"] for g in genes]); fb = np.concatenate([g["f_b"] for g in genes])
    a = np.concatenate([wfun(g["af"]) for g in genes])
    q = q0
    for _ in range(it):
        r = q * fp / np.maximum(q * fp + (1 - q) * fb, 1e-300)
        qn = float((a * r).sum() / a.sum())
        if abs(qn - q) < tol:
            return qn, a
        q = qn
    return q, a


def main():
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.af_prior import ROOT
    from analysis.missense_sensitivity.compare_runs import _find_outputs
    fits = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results.json.gz"))
    df_full = load_master_df(_cfg.DATASET_TSV)
    gene_of, n_g = {}, {}
    for ds in fits:
        d = _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV)
        gene_of[ds] = str(d["Gene"].iloc[0])
        n_g[ds] = int(d["gnomad_MAF"].notna().sum())
    pick = {}
    for ds, g in gene_of.items():
        if g not in pick or n_g[ds] > n_g[pick[g]]:
            pick[g] = ds
    cals = {}
    for ds in pick.values():
        c, _ = _find_outputs(ROOT / "calib_baseline", ds)
        if c:
            cals[ds] = json.load(open(c))
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in cals}
    del df_full
    res = Parallel(n_jobs=24)(delayed(_gene_densities)(ds, slices[ds], fits[ds], cals[ds]) for ds in cals)
    genes = [g for g in res if g is not None]
    for g in genes:
        g["af"] = np.where(np.isfinite(g["af"]) & (g["af"] > 0), g["af"], 0.0)
    W = {"unweighted": lambda af: np.ones(len(af)), "af": lambda af: af,
         "af_cap1e-3": lambda af: np.minimum(af, 1e-3), "af_lt1e-3": lambda af: np.where(af < 1e-3, af, 0.0)}
    rng = np.random.default_rng(0)
    rows = []
    for k, f in W.items():
        q, a = pooled_em(genes, f)
        boots = [pooled_em([genes[i] for i in rng.integers(0, len(genes), len(genes))], f)[0] for _ in range(1000)]
        top = np.sort(a)[::-1]
        rows.append(dict(weights=k, n_genes=len(genes), pooled_prior=q, lo=np.percentile(boots, 2.5),
                         hi=np.percentile(boots, 97.5), n_eff=a.sum() ** 2 / (a ** 2).sum(),
                         top10_share=top[:10].sum() / a.sum()))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "analysis_v3" / "e9b_af_prior_pooled.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("median per-gene unweighted prior:", round(float(np.median([g["q_unweighted"] for g in genes])), 3))


if __name__ == "__main__":
    main()
