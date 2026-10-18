"""EXPERIMENT: weight-divergence penalty and a dedicated RPV component vs the
standard fit, on TP53 -- does a small sample (RPV) get a better fit when
components are pushed to be sample-specific?

Arms (all opt-in features; production defaults untouched):
  baseline      standard K-component fit
  wdiv_<lam>    + weight-divergence penalty (cfusn/weight_divergence.py),
                groups [[P/LP], [B/LB, Synonymous], [RPV]] (gnomAD unpenalised)
  dedicated     (K-1)-component standard fit + one RPV-only component grown
                with add_component_to_fit (base frozen; K total)

Datasets:
  od    tetramerization domain only (residues 319-360, KawOligo categorical;
        tests/cfusn_simulations/pilot_tp53_od_fit.py) -- PRIMARY. Also one
        no-bootstrap Gaussian fit per arm for marginal plots.
  full  full TP53, production preprocessing (KawOligo jitter, raw TempSens,
        Kato PCA2) -- SECONDARY.

K=4, 10 paired bootstraps x 3 restarts (standard Gaussian/q1/q2 cycling),
restart selection by held-out LL (standardized space, as production).
Checkpoint per (dataset, arm, bootstrap, restart).

    nice -n 10 taskset -c 0-7  python tests/cfusn_simulations/sim_weight_divergence_tp53.py run --dataset od   --workers 8
    nice -n 10 taskset -c 8-39 python tests/cfusn_simulations/sim_weight_divergence_tp53.py run --dataset full --workers 32
    python tests/cfusn_simulations/sim_weight_divergence_tp53.py analyze
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

from src.assay_calibration.fit_utils.fit import Fit, _weighted_val_ll, _unstandardize_component_params  # noqa: E402
from src.assay_calibration.fit_utils.cfusn import categorical as cat  # noqa: E402
from src.assay_calibration.fit_utils.cfusn import weight_divergence as wd  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.fit import add_component_to_fit  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.density_utils import (  # noqa: E402
    get_likelihood, _single_component_logpdf,
)

DEFAULT_OUT = _HERE / "results" / "weight_divergence_tp53"
K = 4
NUM_FITS = 3
N_BOOT = 10
MAX_EM_ITERS = 3000
LAMBDAS = (0.03, 0.1, 0.3, 1.0)
ARMS = ("baseline",) + tuple(f"wdiv_{lam:g}" for lam in LAMBDAS) + ("dedicated",)
GROUPS = [["P/LP"], ["B/LB", "Synonymous"], ["RPV"]]
RPV = "RPV"


# ── data ─────────────────────────────────────────────────────────────────────
def build_dataset(name):
    if name == "od":
        import pilot_tp53_od_fit as P
        return P.build_od_scoreset()
    import sim_kawoligo_categorical_tp53 as KC
    ms = KC.build_ms("jitter")
    return SimpleNamespace(scores=ms.scores, sample_assignments=ms.sample_assignments,
                           dataset_names=list(ms.dataset_names), sample_names=list(ms.sample_names),
                           categorical_dims=dict(getattr(ms, "categorical_dims", {}) or {}),
                           ids=np.asarray(ms._all_ids), scoreset_name="TP53")


def _job_kwargs(b, extra=None):
    kw = dict(num_fits=NUM_FITS, check_monotonic=False, master_seed=0, max_em_iters=MAX_EM_ITERS,
              verbose_overlap=False)
    if b == "single":
        kw.update(num_fits=1, bootstrap=False, bootstrap_seed=0)
    else:
        kw.update(bootstrap=True, bootstrap_seed=int(b))
    kw.update(extra or {})
    return kw


def arm_jobs(ss, arm, b):
    """Jobs for one (arm, bootstrap); 'dedicated' returns its (K-1) base jobs."""
    extra = {}
    if arm.startswith("wdiv_"):
        extra["weight_divergence"] = {"lambda": float(arm.split("_", 1)[1]), "groups": GROUPS}
    k = K - 1 if arm == "dedicated" else K
    return Fit(ss).generate_fit_jobs([k], **_job_kwargs(b, extra))


def _raw(job, key):
    X = job[key]
    if X is None:
        return None
    return X * job["kwargs"]["_scale_std"] + job["kwargs"]["_scale_mean"]


def _restandardize(params, mean, std):
    out = []
    for mu, D, G in params:
        mu, D, G = (np.asarray(a, dtype=float) for a in (mu, D, G))
        out.append(((mu - mean) / std, D / std if D.ndim == 1 else D / std[:, None], G / np.outer(std, std)))
    return out


def _mode(job):
    kw = job["kwargs"]
    return "gaussian" if kw.get("force_gaussian") else f"q{kw.get('latent_q', 2)}"


def _grow_dedicated(job, base_res, rpv_idx):
    """Grow one RPV-only component on a (K-1) base fit, in standardized space."""
    fit = base_res["fit"]
    kw = job["kwargs"]
    mean, std = np.asarray(kw["_scale_mean"]), np.asarray(kw["_scale_std"])
    cont = fit["cat_spec"]["cont_positions"] if fit.get("cat_spec") else list(range(len(mean)))
    params_std = _restandardize(fit["component_params"], mean[cont], std[cont])
    mode = _mode(job)
    latent_q = 1 if mode == "q1" else 2
    if latent_q == 1:                       # execute_fit_job padded the q=1 winner to (p, 2)
        params_std = [(m, np.asarray(D)[:, 0] if np.ndim(D) == 2 else D, G) for m, D, G in params_std]
    base_std = dict(fit, component_params=params_std)
    grow_kw = dict(fit_seed=kw.get("fit_seed"), max_em_iters=MAX_EM_ITERS, verbose=False,
                   n_mc_truncated=kw.get("n_mc_truncated", 500), force_gaussian=(mode == "gaussian"),
                   min_rows=2)
    if fit.get("cat_spec"):
        grow_kw.update(cat_spec=fit["cat_spec"], cat_alpha=fit.get("cat_alpha"))
    grown = add_component_to_fit(job["train_observations"], job["train_sample_assignments"], base_std,
                                 target_sample_idx=rpv_idx, latent_q=latent_q, **grow_kw)
    grown.pop("history", None)
    grown.pop("likelihoods", None)
    val_ll = None
    if job.get("val_observations") is not None:
        val_ll = _weighted_val_ll(job["val_observations"], job["val_sample_assignments"],
                                  grown["component_params"], grown["weights"], True, kw, cat_fit=grown)
    grown["component_params"] = _unstandardize_component_params(grown["component_params"], mean[cont], std[cont])
    return {"fit": grown, "val_ll": val_ll}


def _run_task(path, dataset, arm, b, fit_idx):
    if path.exists():
        return str(path), 0.0, None
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ss = build_dataset(dataset)
    jobs = arm_jobs(ss, arm, b)
    job = [j for j in jobs if j["fit_idx"] == fit_idx][0]
    if arm == "dedicated":
        base = Fit.execute_fit_job(job)
        res = _grow_dedicated(job, base, list(ss.sample_names).index(RPV)) if base else None
    else:
        res = Fit.execute_fit_job(job)
    out = {"res": res, "seconds": time.time() - t0, "mode": _mode(job), "arm": arm, "dataset": dataset,
           "bootstrap": b, "fit_idx": fit_idx,
           "train_raw": _raw(job, "train_observations"), "train_sa": job["train_sample_assignments"],
           "val_raw": _raw(job, "val_observations"), "val_sa": job.get("val_sample_assignments"),
           "val_variant_indices": job.get("val_variant_indices")}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    tmp.rename(path)
    return str(path), out["seconds"], (res or {}).get("val_ll")


def run(out_dir, dataset, n_boot, workers, max_em_iters):
    global MAX_EM_ITERS
    MAX_EM_ITERS = max_em_iters
    d = out_dir / dataset
    d.mkdir(parents=True, exist_ok=True)
    boots = (["single"] if dataset == "od" else []) + list(range(n_boot))
    todo = []
    for b in boots:
        n_fits = 1 if b == "single" else NUM_FITS
        for arm in ARMS:
            for f in range(n_fits):
                p = d / f"{arm}_b{b}_f{f}.pkl"
                if not p.exists():
                    todo.append((p, dataset, arm, b, f))
    print(f"[{dataset}] {len(todo)} fits to run ({workers} workers)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_task, *t): t[0] for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                path, secs, vll = fut.result()
                print(f"[{i}/{len(todo)}] {Path(path).name}: {secs:.0f}s val_ll={vll}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {futs[fut].name}: FAILED {e!r}", flush=True)
    print("ALL DONE", flush=True)


# ── analysis ─────────────────────────────────────────────────────────────────
def load_best(d, arm, b):
    n_fits = 1 if b == "single" else NUM_FITS
    entries = []
    for f in range(n_fits):
        p = d / f"{arm}_b{b}_f{f}.pkl"
        if p.exists():
            e = pickle.load(open(p, "rb"))
            r = e.get("res")
            if r and r.get("fit") and (b == "single" or (r.get("val_ll") is not None and np.isfinite(r["val_ll"]))):
                entries.append(e)
    if not entries:
        return None, []
    if b == "single":
        return entries[0], entries
    return max(entries, key=lambda e: e["res"]["val_ll"]), entries


def _split(fit, X):
    if fit.get("cat_spec"):
        spec = cat.CatSpec.from_dict(fit["cat_spec"])
        Xc, labels, _ = cat.split_observations(X, spec)
        return Xc, cat.cat_log_pdfs(labels, [np.asarray(P) for P in fit["cat_probs"]]), spec
    return X, None, None


def comp_logpdfs(fit, X):
    """(K, N) full-row component log-densities (categorical term included)."""
    Xc, extra, _ = _split(fit, X)
    params = fit["component_params"]
    ok = ~np.all(np.isnan(Xc), axis=1)
    lp = np.zeros((len(params), len(X)))
    if ok.any():
        lp[:, ok] = np.array([np.atleast_1d(_single_component_logpdf(Xc[ok], p, True)).ravel() for p in params])
    if extra is not None:
        lp = lp + extra
    return lp


def log_mix(fit, X, w):
    from scipy.special import logsumexp
    with np.errstate(divide="ignore"):
        return logsumexp(comp_logpdfs(fit, X) + np.log(np.asarray(w))[:, None], axis=0)


def ll_per_row(fit, X, sa, s=None):
    """Mean held-out log-likelihood per row, optionally for sample s only."""
    Xc, extra, _ = _split(fit, X)
    W = np.asarray(fit["weights"])
    if s is not None:
        rows = sa[:, s]
        if not rows.any():
            return np.nan
        Xc, sa, X = Xc[rows], sa[rows][:, [s]], X[rows]
        W = W[[s]]
        extra = None if extra is None else extra[:, rows]
    return get_likelihood(Xc, sa, fit["component_params"], W, multivariate=True, extra_log_pdfs=extra) / len(Xc)


def marginal_l1(fit, X, sa, s, ref_X):
    """Mean over dims of the L1 distance between sample s's histogram and its
    fitted marginal (categorical dims: total variation x 2)."""
    W = np.asarray(fit["weights"])[s]
    params = fit["component_params"]
    spec = cat.CatSpec.from_dict(fit["cat_spec"]) if fit.get("cat_spec") else None
    cont = spec.cont_positions if spec else list(range(X.shape[1]))
    errs = []
    rows = sa[:, s]
    for d in range(X.shape[1]):
        v = X[rows, d]
        v = v[~np.isnan(v)]
        if len(v) < 3:
            continue
        if spec is not None and d in spec.positions:
            c = spec.positions.index(d)
            pmf = W @ np.asarray(fit["cat_probs"][c])
            obs = np.bincount(v.astype(int), minlength=len(pmf)) / len(v)
            errs.append(np.abs(pmf - obs).sum())
            continue
        allv = ref_X[:, d]
        allv = allv[~np.isnan(allv)]
        lo, hi = np.percentile(allv, [0.5, 99.5])
        edges = np.linspace(lo, hi, 31)
        centers = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        Xg = np.full((len(centers), len(cont)), np.nan)
        Xg[:, cont.index(d)] = centers
        dens = W @ np.exp(np.array([np.atleast_1d(_single_component_logpdf(Xg, p, True)).ravel() for p in params]))
        hist = np.histogram(np.clip(v, lo, hi), bins=edges)[0] / len(v)
        errs.append(np.abs(hist - dens * width).sum())
    return float(np.mean(errs)) if errs else np.nan


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def diagnostics(entry):
    fit = entry["res"]["fit"]
    params = fit["component_params"]
    sds = []
    for mu, D, G in params:
        D = np.atleast_2d(np.asarray(D, dtype=float))
        D = D if D.shape[0] == len(mu) else D.T
        sds.append(np.sqrt(np.diag(np.asarray(G) + D @ D.T)))
    sds = np.array(sds)
    obj = fit.get("objectives")
    mono = None
    if obj is not None and len(obj) > 1:
        rel = np.diff(obj) / np.maximum(np.abs(np.asarray(obj[:-1])), 1.0)
        mono = float(rel.min())
    return {"overshoot": fit.get("overshoot_stopped") is not None,
            "obj_min_rel_diff": mono,
            "n_iter": len(obj) if obj is not None else None,
            "min_comp_sd_ratio": float((sds / np.nanmedian(sds, axis=0)[None]).min()),
            "min_comp_weight_max": float(np.asarray(fit["weights"]).max(axis=0).min())}


def analyze(out_dir, n_boot):
    import csv
    warnings.filterwarnings("ignore")
    rows_out = []
    for dataset in ("od", "full"):
        d = out_dir / dataset
        if not d.exists():
            continue
        ss = build_dataset(dataset)
        names = list(ss.sample_names)
        p_idx, b_idx, r_idx = names.index("P/LP"), names.index("B/LB"), names.index(RPV)
        pairs = wd.resolve_weight_divergence({"lambda": 1.0, "groups": GROUPS}, names, len(names))["pairs"]
        X_all, sa_all = ss.scores, ss.sample_assignments.astype(bool)
        rpv_rows = np.where(sa_all[:, r_idx])[0]
        lr_by_arm = {}
        for arm in ARMS:
            lr_rb, lr_rp = [], []
            for b in range(n_boot):
                e, entries = load_best(d, arm, b)
                if e is None:
                    continue
                fit = e["res"]["fit"]
                W = np.asarray(fit["weights"])
                tr, trs = e["train_raw"], e["train_sa"].astype(bool)
                va, vas = e["val_raw"], e["val_sa"].astype(bool)
                rec = {"dataset": dataset, "arm": arm, "bootstrap": b, "restart": e["mode"],
                       "val_ll_std": e["res"]["val_ll"],
                       "oob_ll_row": ll_per_row(fit, va, vas),
                       "oob_ll_rpv_row": ll_per_row(fit, va, vas, r_idx),
                       "n_oob_rpv": int(vas[:, r_idx].sum()), "n_train_rpv": int(trs[:, r_idx].sum()),
                       "rpv_l1_train": marginal_l1(fit, tr, trs, r_idx, X_all),
                       "rpv_l1_oob": marginal_l1(fit, va, vas, r_idx, X_all),
                       "plp_l1_train": marginal_l1(fit, tr, trs, p_idx, X_all),
                       "blb_l1_train": marginal_l1(fit, tr, trs, b_idx, X_all),
                       "R_W": wd.divergence(W, pairs),
                       "W_rpv": np.round(W[r_idx], 3).tolist()}
                lfp, lfb = log_mix(fit, va, W[p_idx]), log_mix(fit, va, W[b_idx])
                llr = lfp - lfb
                rec["oob_auroc_plp_blb"] = auroc(llr[vas[:, p_idx]], llr[vas[:, b_idx]])
                rec.update(diagnostics(e))
                rec["any_restart_overshoot"] = any(diagnostics(x)["overshoot"] for x in entries)
                mins = [diagnostics(x)["obj_min_rel_diff"] for x in entries]
                mins = [m for m in mins if m is not None]
                rec["worst_restart_obj_rel_diff"] = min(mins) if mins else None
                rows_out.append(rec)
                Xr = X_all[rpv_rows]
                lr_rb.append(log_mix(fit, Xr, W[r_idx]) - log_mix(fit, Xr, W[b_idx]))
                lr_rp.append(log_mix(fit, Xr, W[r_idx]) - log_mix(fit, Xr, W[p_idx]))
            if lr_rb:
                lr_by_arm[arm] = (np.array(lr_rb), np.array(lr_rp))
        for arm, (rb, rp) in lr_by_arm.items():
            for r in rows_out:
                if r["dataset"] == dataset and r["arm"] == arm:
                    r["rpv_vs_blb_lr_sd_mean"] = float(np.mean(np.std(rb, axis=0)))
                    r["rpv_vs_blb_lr_median"] = float(np.median(np.median(rb, axis=0)))
                    r["rpv_vs_plp_lr_sd_mean"] = float(np.mean(np.std(rp, axis=0)))
                    r["rpv_vs_plp_lr_median"] = float(np.median(np.median(rp, axis=0)))
        with open(out_dir / f"rpv_lr_{dataset}.pkl", "wb") as fh:
            pickle.dump({"lr_by_arm": lr_by_arm, "rpv_rows": rpv_rows, "ids": ss.ids[rpv_rows]}, fh)
    if not rows_out:
        print("nothing to analyze")
        return
    keys = sorted({k for r in rows_out for k in r})
    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", out_dir / "metrics.csv")
    metrics = [("oob_ll_row", +1), ("oob_ll_rpv_row", +1), ("rpv_l1_train", -1), ("rpv_l1_oob", -1),
               ("plp_l1_train", -1), ("blb_l1_train", -1), ("oob_auroc_plp_blb", +1), ("R_W", +1)]
    for dataset in ("od", "full"):
        by = {(r["arm"], r["bootstrap"]): r for r in rows_out if r["dataset"] == dataset}
        if not by:
            continue
        boots = sorted({b for (_, b) in by})
        print(f"\n== {dataset}: paired diff vs baseline (mean ± SE over bootstraps; arrow = better direction)")
        for arm in ARMS[1:]:
            print(f"  {arm}:")
            for m, sign in metrics:
                dvals = [by[(arm, b)][m] - by[("baseline", b)][m] for b in boots
                         if (arm, b) in by and ("baseline", b) in by
                         and np.isfinite(by[(arm, b)][m]) and np.isfinite(by[("baseline", b)][m])]
                if dvals:
                    dvals = np.array(dvals)
                    print(f"    {m:22s} {dvals.mean():+.4f} ± {dvals.std(ddof=1) / np.sqrt(len(dvals)) if len(dvals) > 1 else 0:.4f}"
                          f"  ({'↑' if sign > 0 else '↓'} better, n={len(dvals)})")
        print(f"  health (per arm): overshoot rate / worst objective rel diff across all restarts / LR SD")
        for arm in ARMS:
            rs = [by[(arm, b)] for b in boots if (arm, b) in by]
            if rs:
                ov = np.mean([r["any_restart_overshoot"] for r in rs])
                worst = [r["worst_restart_obj_rel_diff"] for r in rs if r["worst_restart_obj_rel_diff"] is not None]
                print(f"    {arm:12s} overshoot={ov:.2f}  worst_obj_rel_diff={min(worst) if worst else 'n/a'}  "
                      f"RPV-vs-B LR SD={rs[0].get('rpv_vs_blb_lr_sd_mean', np.nan):.2f}  "
                      f"RPV-vs-P LR SD={rs[0].get('rpv_vs_plp_lr_sd_mean', np.nan):.2f}")


def figures(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import pilot_tp53_od_fit as P
    warnings.filterwarnings("ignore")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dataset, b in (("od", "single"), ("od", 0), ("full", 0)):
        d = out_dir / dataset
        if not d.exists():
            continue
        ss = build_dataset(dataset)
        for arm in ARMS:
            e, _ = load_best(d, arm, b)
            if e is None:
                continue
            fit = e["res"]["fit"]
            P.plot_marginals(ss, fit, len(fit["component_params"]), f"{e['mode']} [{arm}]",
                             fig_dir / f"marginals_{dataset}_b{b}_{arm}.png")
    csv_path = out_dir / "metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        metrics = [("oob_ll_row", "held-out LL / row ↑"), ("oob_ll_rpv_row", "held-out RPV LL / row ↑"),
                   ("rpv_l1_train", "RPV marginal L1 (train) ↓"), ("rpv_l1_oob", "RPV marginal L1 (OOB) ↓"),
                   ("blb_l1_train", "B/LB marginal L1 (train) ↓"), ("oob_auroc_plp_blb", "OOB P/LP vs B/LB AUROC ↑"),
                   ("R_W", "achieved R(W)")]
        for dataset in df.dataset.unique():
            sub = df[df.dataset == dataset]
            fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.8))
            for ax, (m, title) in zip(axes, metrics):
                t = sub.pivot_table(index="bootstrap", columns="arm", values=m)
                arms = [a for a in ARMS if a in t]
                offs = np.linspace(-0.15, 0.15, len(t))
                for i, a in enumerate(arms):
                    ax.scatter(i + offs, t[a], s=10, alpha=0.6)
                    ax.errorbar(i + 0.3, t[a].mean(), yerr=t[a].std(ddof=1) / np.sqrt(t[a].notna().sum()),
                                fmt="D", ms=5, capsize=3, color="k")
                for oi, bi in enumerate(t.index):
                    ax.plot(np.arange(len(arms)) + offs[oi], [t.loc[bi, a] for a in arms], color="gray", lw=0.3)
                ax.set_xticks(range(len(arms)))
                ax.set_xticklabels(arms, rotation=45, fontsize=7)
                ax.set_title(title, fontsize=8, fontweight="bold")
                ax.grid(lw=0.2, alpha=0.4)
            fig.suptitle(f"TP53 {dataset}: weight-divergence / dedicated-RPV arms vs baseline "
                         f"(K={K}, {sub.bootstrap.nunique()} paired bootstraps)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(fig_dir / f"metrics_{dataset}.png", dpi=140)
            plt.close(fig)
    print("wrote", fig_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "analyze", "figures"))
    ap.add_argument("--dataset", choices=("od", "full"), default="od")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-em-iters", type=int, default=MAX_EM_ITERS,
                    help="override only for smoke tests")
    args = ap.parse_args()
    if args.mode == "run":
        run(args.out, args.dataset, args.n_boot, args.workers, args.max_em_iters)
    elif args.mode == "analyze":
        analyze(args.out, args.n_boot)
    else:
        figures(args.out)


if __name__ == "__main__":
    main()
