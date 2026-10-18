"""Does modelling TP53's KawOligo as a CATEGORICAL dimension beat the
production jittered-continuous treatment?

Configs, all on the same tp53_kato_pca2 10-dim TP53 data, K=6, num_fits=3
(gaussian/q1/q2 restart cycling as in production), best restart by val_ll,
10 paired bootstraps (identical train/OOB rows across configs):
  jitter   : production -- KawOligo mapped {0, .5, 1} + N(0, 0.1) jitter
  cat_tau3 : KawOligo categorical (cfusn/categorical.py), prior tau=3
  cat_tau1 : same, tau=1
  dropped  : KawOligo column removed (continuous dims only)

Metrics (per bootstrap; compare paired differences vs jitter):
  M2 (primary) held-out log p(KawOligo level | continuous dims, class) on OOB
     rows with KawOligo observed, plus Brier score. Jitter: the joint
     density integrated over the level bins (-inf,.25), [.25,.75), [.75,inf)
     on a fine grid; dropped: smoothed training class frequencies.
  M1 held-out log-likelihood of the CONTINUOUS dims only (KawOligo masked),
     per OOB row -- same space for every config.
  M3 total-variation distance between predicted per-class KawOligo pmf and
     observed OOB (and training) per-class frequencies.
  M4 OOB AUROC of log LR (P/LP vs B/LB), continuous-dim marginal L1 fit
     error (training rows), convergence flags.

KawOligo is ordinal; the categorical model here is nominal (see
categorical.py).

Usage (throttled; the server is shared):
    source activate excalibr
    taskset -c 0-7 python tests/cfusn_simulations/sim_kawoligo_categorical_tp53.py run
    taskset -c 0-7 python tests/cfusn_simulations/sim_kawoligo_categorical_tp53.py analyze
"""
import os

# One BLAS thread per worker: N_WORKERS single-threaded fits inside the
# taskset core budget, instead of one fit spreading over every core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import copy
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.fit_utils.cfusn import categorical as cat
from src.assay_calibration.fit_utils.cfusn.density_utils import (
    get_likelihood, _single_component_logpdf,
)
from src.assay_calibration.multivariate_data import tp53
from src.assay_calibration.multivariate_data import redundancy_collapse as rc

DEFAULT_OUT = _REPO_ROOT / "tests" / "cfusn_simulations" / "results" / "kawoligo_categorical_tp53"
K = 6
NUM_FITS = 3
N_BOOT = 10
MAX_EM_ITERS = 3000
N_WORKERS = 8
CONFIGS = ("jitter", "cat_tau3", "cat_tau1", "dropped")
KAW = "KawOligo"
JITTER_BINS = (-np.inf, 0.25, 0.75, np.inf)


# ── data / jobs ──────────────────────────────────────────────────────────────

def build_ms(mode):
    ms = tp53.build_tp53_multiscoreset(kawoligo_mode=mode)
    rc.apply_preset({"TP53": ms}, "tp53_kato_pca2", verbose=False)
    return ms


def _common_job_kwargs(b):
    return dict(num_fits=NUM_FITS, bootstrap=True, check_monotonic=False, master_seed=0,
                bootstrap_seed=b, max_em_iters=MAX_EM_ITERS, verbose_overlap=False)


def _raw(job, key):
    """Undo generate_fit_jobs's standardization of an observation matrix."""
    X = job[key]
    if X is None:
        return None
    return X * job["kwargs"]["_scale_std"] + job["kwargs"]["_scale_mean"]


def _drop_kaw_job(job, kaw_pos):
    """The categorical job with KawOligo removed -- identical rows otherwise
    (rows left with no observed dim are dropped from train and OOB)."""
    j = copy.deepcopy(job)
    keep_cols = [c for c in range(j["train_observations"].shape[1]) if c != kaw_pos]
    for obs_key, sa_key in (("train_observations", "train_sample_assignments"),
                            ("val_observations", "val_sample_assignments")):
        X = j[obs_key][:, keep_cols]
        ok = ~np.all(np.isnan(X), axis=1)
        j[obs_key] = X[ok]
        j[sa_key] = j[sa_key][ok]
        if obs_key == "val_observations" and j.get("val_variant_indices") is not None:
            j["val_variant_indices"] = np.asarray(j["val_variant_indices"])[ok]
    kw = j["kwargs"]
    kw.pop("cat_spec", None)
    kw.pop("cat_tau", None)
    kw["_scale_mean"] = np.asarray(kw["_scale_mean"])[keep_cols]
    kw["_scale_std"] = np.asarray(kw["_scale_std"])[keep_cols]
    j["calibrated_dims"] = [d for i, d in enumerate(j["calibrated_dims"]) if i != kaw_pos]
    return j


def make_jobs(b, ms_jit, ms_cat):
    jit = Fit(ms_jit).generate_fit_jobs([K], **_common_job_kwargs(b))
    cat3 = Fit(ms_cat).generate_fit_jobs([K], cat_tau=3.0, **_common_job_kwargs(b))
    cat1 = Fit(ms_cat).generate_fit_jobs([K], cat_tau=1.0, **_common_job_kwargs(b))
    # Pairing: same OOB variants and same training missingness patterns.
    for a, c in zip(jit, cat3):
        assert np.array_equal(a["val_variant_indices"], c["val_variant_indices"])
        assert np.array_equal(np.isnan(a["train_observations"]), np.isnan(c["train_observations"]))
    kaw_pos = cat3[0]["kwargs"]["cat_spec"]["positions"][0]
    dropped = [_drop_kaw_job(j, kaw_pos) for j in cat3]
    return {"jitter": jit, "cat_tau3": cat3, "cat_tau1": cat1, "dropped": dropped}


def _fit_one(path, job):
    if path.exists():  # finished meanwhile by a concurrent runner
        return str(path), 0.0, None
    t0 = time.time()
    res = Fit.execute_fit_job(job)
    out = {"res": res, "seconds": time.time() - t0,
           "train_raw": _raw(job, "train_observations"),
           "train_sa": job["train_sample_assignments"],
           "val_raw": _raw(job, "val_observations"),
           "val_sa": job["val_sample_assignments"],
           "val_variant_indices": job.get("val_variant_indices"),
           "fit_kwargs": {k: v for k, v in job["kwargs"].items() if k in ("cat_spec", "cat_tau", "latent_q", "force_gaussian")}}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    tmp.rename(path)
    return str(path), out["seconds"], res["val_ll"] if res else None


def run(out_dir, n_boot, n_workers=N_WORKERS, reverse=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("building TP53 multiscoresets (jitter / categorical)...", flush=True)
    ms_jit, ms_cat = build_ms("jitter"), build_ms("categorical")
    todo = []
    for b in range(n_boot):
        jobs = make_jobs(b, ms_jit, ms_cat)
        for config, js in jobs.items():
            for job in js:
                path = out_dir / f"{config}_b{b}_f{job['fit_idx']}.pkl"
                if not path.exists():
                    todo.append((path, job))
    if reverse:  # a second runner working from the other end of the queue
        todo = todo[::-1]
    print(f"{len(todo)} fits to run ({n_workers} workers)", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_fit_one, p, j): p for p, j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                path, secs, vll = fut.result()
                print(f"[{i}/{len(todo)}] {Path(path).name}: {secs:.0f}s val_ll={vll}", flush=True)
            except Exception as e:  # keep going; a failed fit just stays missing
                print(f"[{i}/{len(todo)}] {p.name}: FAILED {e!r}", flush=True)
    print("ALL DONE", flush=True)


# ── metrics ──────────────────────────────────────────────────────────────────

def _comp_logpdfs(X, params):
    return np.array([np.atleast_1d(_single_component_logpdf(X, p, True)).ravel() for p in params])


def _logsumexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _row_class(sa):
    return np.argmax(sa, axis=1)


def _level_probs_given_cont(entry, config, X_raw, cls, kaw_col):
    """(n, 3) predicted p(level | continuous dims, class) for each row."""
    fit = entry["res"]["fit"]
    params, W = fit["component_params"], np.asarray(fit["weights"])
    n = len(X_raw)
    if config.startswith("cat"):
        spec = cat.CatSpec.from_dict(fit["cat_spec"])
        Xc, _, _ = cat.split_observations(X_raw, spec)
        lp = _comp_logpdfs(Xc, params) + np.log(W[cls].T)          # (K, n)
        r = np.exp(lp - _logsumexp(lp))
        return r.T @ fit["cat_probs"][0]
    # jitter: integrate the joint density over each level's bin on a grid
    grid = np.linspace(-6.0, 7.0, 2601)
    out = np.zeros((n, 3))
    for i in range(n):
        Xg = np.repeat(X_raw[i:i + 1], len(grid), axis=0)
        Xg[:, kaw_col] = grid
        lp = _comp_logpdfs(Xg, params) + np.log(W[cls[i]])[:, None]   # (K, G)
        dens = np.exp(_logsumexp(lp))                                  # (G,)
        mass = []
        for lo, hi in zip(JITTER_BINS[:-1], JITTER_BINS[1:]):
            m = (grid >= lo) & (grid < hi)
            mass.append(np.trapz(dens[m], grid[m]))
        mass = np.asarray(mass)
        out[i] = mass / mass.sum()
    return out


def _class_pmf(entry, config, kaw_col, n_classes):
    """(S, 3) predicted marginal KawOligo pmf per class."""
    fit = entry["res"]["fit"]
    W = np.asarray(fit["weights"])
    if config.startswith("cat"):
        return W @ fit["cat_probs"][0]
    params = fit["component_params"]
    p_all = len(params[0][0])
    grid = np.linspace(-6.0, 7.0, 2601)
    Xg = np.full((len(grid), p_all), np.nan)
    Xg[:, kaw_col] = grid
    dens_k = np.exp(_comp_logpdfs(Xg, params))                        # (K, G)
    comp_mass = []
    for lo, hi in zip(JITTER_BINS[:-1], JITTER_BINS[1:]):
        m = (grid >= lo) & (grid < hi)
        comp_mass.append(np.trapz(dens_k[:, m], grid[m], axis=1))
    comp_mass = np.array(comp_mass).T                                  # (K, 3)
    comp_mass /= comp_mass.sum(axis=1, keepdims=True)
    return W @ comp_mass


def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _log_lr(entry, config, X_raw):
    fit = entry["res"]["fit"]
    params, W = fit["component_params"], np.asarray(fit["weights"])
    if config.startswith("cat"):
        spec = cat.CatSpec.from_dict(fit["cat_spec"])
        Xc, labels, _ = cat.split_observations(X_raw, spec)
        lp = _comp_logpdfs(Xc, params) + cat.cat_log_pdfs(labels, fit["cat_probs"])
    else:
        lp = _comp_logpdfs(X_raw, params)
    lf_p = _logsumexp(lp + np.log(W[0])[:, None])
    lf_b = _logsumexp(lp + np.log(W[1])[:, None])
    return lf_p - lf_b


def _cont_marginal_l1(entry, config, X_raw, cls, cont_cols, n_classes):
    """Mean over (continuous dim, class) of the L1 distance between the
    training-row histogram and the fitted marginal density."""
    fit = entry["res"]["fit"]
    params, W = fit["component_params"], np.asarray(fit["weights"])
    p_model = len(params[0][0])
    errs = []
    for j, col in enumerate(cont_cols):
        model_col = j if config.startswith("cat") or config == "dropped" else col
        vals_all = X_raw[:, col]
        fin = vals_all[~np.isnan(vals_all)]
        if fin.size < 10:
            continue
        lo, hi = np.percentile(fin, [0.5, 99.5])
        edges = np.linspace(lo, hi, 21)
        centers = 0.5 * (edges[:-1] + edges[1:])
        Xg = np.full((len(centers), p_model), np.nan)
        Xg[:, model_col] = centers
        dens_k = np.exp(_comp_logpdfs(Xg, params))
        for s in range(n_classes):
            v = vals_all[(cls == s) & ~np.isnan(vals_all)]
            if v.size < 10:
                continue
            h, _ = np.histogram(v, bins=edges, density=True)
            errs.append(np.sum(np.abs(h - W[s] @ dens_k) * np.diff(edges)))
    return float(np.mean(errs)) if errs else np.nan


def _install_single_row_fix():
    """ANALYSIS-ONLY patch (production density_utils is untouched): the
    multivariate _single_component_logpdf returns a 0-d scalar for ONE fully
    observed row (the NaN-marginal paths return shape (1,)), and
    log_joint_densities' log_w[:, None] + log_pdfs then broadcasts to (K, K),
    so get_likelihood scores a single-row sample as ~sum over components'
    log-pdfs. Hit here whenever a bootstrap's OOB split holds exactly one
    Synonymous row with every continuous dim observed (b3/b6/b9 for the
    categorical + dropped configs; jitter's row has KawOligo NaN, so it took
    the marginal path and was unaffected). execute_fit_job's stored val_ll --
    and hence restart selection -- inherited the bug, so analyze() re-scores
    every restart with the fixed density before selecting."""
    from src.assay_calibration.fit_utils.cfusn import density_utils as du
    if getattr(du._single_component_logpdf, "_single_row_fixed", False):
        return
    orig = du._single_component_logpdf

    def fixed(x, params, multivariate=False):
        out = orig(x, params, multivariate)
        if multivariate:
            out = np.asarray(out, dtype=float).reshape(np.atleast_2d(x).shape[0])
        return out
    fixed._single_row_fixed = True
    du._single_component_logpdf = fixed


def _val_ll_fixed(e):
    """Per-row validation LL on the stored raw OOB rows (raw units: a constant
    Jacobian offset vs execute_fit_job's standardized val_ll, identical across
    one config's restarts, so restart selection is unaffected by the units)."""
    fit = e["res"]["fit"]
    X, extra = e["val_raw"], None
    if fit.get("cat_spec"):
        spec = cat.CatSpec.from_dict(fit["cat_spec"])
        X, labels, _ = cat.split_observations(X, spec)
        extra = cat.cat_log_pdfs(labels, [np.asarray(P) for P in fit["cat_probs"]])
    return get_likelihood(X, e["val_sa"], fit["component_params"], np.asarray(fit["weights"]),
                          multivariate=True, extra_log_pdfs=extra) / len(X)


def _best(entries):
    ok = [e for e in entries if e["res"] and e["res"]["val_ll"] is not None and np.isfinite(e["res"]["val_ll"])]
    for e in ok:
        e["val_ll_fixed"] = _val_ll_fixed(e)
    return max(ok, key=lambda e: e["val_ll_fixed"]) if ok else None


def analyze(out_dir, n_boot):
    import csv
    _install_single_row_fix()
    rows = []
    for b in range(n_boot):
        best = {}
        for config in CONFIGS:
            entries = []
            for f in range(NUM_FITS):
                p = out_dir / f"{config}_b{b}_f{f}.pkl"
                if p.exists():
                    with open(p, "rb") as fh:
                        entries.append(pickle.load(fh))
            best[config] = _best(entries)
        if any(v is None for v in best.values()):
            print(f"b{b}: incomplete, skipping")
            continue

        # Shared reference: the categorical config's raw (full-width) data.
        ref = best["cat_tau3"]
        kaw_col = ref["res"]["fit"]["cat_spec"]["positions"][0]
        cont_cols = ref["res"]["fit"]["cat_spec"]["cont_positions"]
        val_raw, val_cls = ref["val_raw"], _row_class(ref["val_sa"])
        tr_raw, tr_cls = ref["train_raw"], _row_class(ref["train_sa"])
        S = ref["val_sa"].shape[1]
        has_cont_val = ~np.all(np.isnan(val_raw[:, cont_cols]), axis=1)
        kaw_val = ~np.isnan(val_raw[:, kaw_col]) & has_cont_val
        lev_val = val_raw[kaw_val, kaw_col].astype(int)
        jit_raw = best["jitter"]["val_raw"]
        assert np.array_equal(np.isnan(jit_raw), np.isnan(val_raw))

        tr_lab = tr_raw[:, kaw_col]
        cls_freq = np.zeros((S, 3))
        for s in range(S):
            v = tr_lab[(tr_cls == s) & ~np.isnan(tr_lab)].astype(int)
            cls_freq[s] = (np.bincount(v, minlength=3) + 0.5) / (len(v) + 1.5)

        for config in CONFIGS:
            e = best[config]
            fit = e["res"]["fit"]
            rec = {"bootstrap": b, "config": config, "val_ll_selected": e["val_ll_fixed"],
                   "val_ll_stored": e["res"]["val_ll"],
                   "restart_selected": ("gaussian" if e["fit_kwargs"].get("force_gaussian")
                                        else f"q{e['fit_kwargs'].get('latent_q')}"),
                   "overshoot": fit.get("overshoot_stopped") is not None,
                   "gamma_rejected": (fit.get("guard_stats") or {}).get("gamma_rejected", 0),
                   "n_oob_kaw": int(kaw_val.sum())}
            # M2
            if config == "dropped":
                probs = cls_freq[val_cls[kaw_val]]
            elif config == "jitter":
                probs = _level_probs_given_cont(e, config, jit_raw[kaw_val], val_cls[kaw_val], kaw_col)
            else:
                probs = _level_probs_given_cont(e, config, val_raw[kaw_val], val_cls[kaw_val], kaw_col)
            p_obs = np.clip(probs[np.arange(len(lev_val)), lev_val], 1e-300, None)
            onehot = np.eye(3)[lev_val]
            rec["M2_logscore"] = float(np.mean(np.log(p_obs))) if len(lev_val) else np.nan
            rec["M2_brier"] = float(np.mean(np.sum((probs - onehot) ** 2, axis=1))) if len(lev_val) else np.nan
            # M1: continuous dims only, KawOligo masked
            params, W = fit["component_params"], np.asarray(fit["weights"])
            vsa = ref["val_sa"][has_cont_val]
            if config == "jitter":
                Xv = jit_raw[has_cont_val].copy()
                Xv[:, kaw_col] = np.nan
            else:
                Xv = val_raw[has_cont_val][:, cont_cols]
            rec["M1_cont_ll"] = get_likelihood(Xv, vsa, params, W, multivariate=True) / len(Xv)
            # M3
            if config != "dropped":
                pmf = _class_pmf(e, config, kaw_col, S)
                for split, raw, cl in (("oob", val_raw, val_cls), ("train", tr_raw, tr_cls)):
                    tvs, wts = [], []
                    for s in range(S):
                        v = raw[(cl == s) & ~np.isnan(raw[:, kaw_col]), kaw_col].astype(int)
                        if len(v) == 0:
                            continue
                        obs_f = np.bincount(v, minlength=3) / len(v)
                        tvs.append(0.5 * np.abs(pmf[s] - obs_f).sum())
                        wts.append(len(v))
                    rec[f"M3_tv_{split}"] = float(np.average(tvs, weights=wts)) if tvs else np.nan
            # M4
            Xlr = jit_raw if config == "jitter" else (val_raw if config.startswith("cat") else val_raw[:, cont_cols])
            if config == "dropped":
                Xlr = Xlr[has_cont_val]
                cls_lr = val_cls[has_cont_val]
            else:
                cls_lr = val_cls
            llr = _log_lr(e, config, Xlr)
            rec["M4_auroc_plp_blb"] = float(_auroc(llr[cls_lr == 0], llr[cls_lr == 1]))
            tr_X = best["jitter"]["train_raw"] if config == "jitter" else tr_raw
            rec["M4_cont_marginal_l1"] = _cont_marginal_l1(e, config, tr_X, tr_cls, cont_cols, S)
            rows.append(rec)
        print(f"b{b}: done", flush=True)

    if not rows:
        print("no complete bootstraps")
        return
    csv_path = out_dir / "metrics.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    metrics = ["M2_logscore", "M2_brier", "M1_cont_ll", "M3_tv_oob", "M3_tv_train",
               "M4_auroc_plp_blb", "M4_cont_marginal_l1"]
    by = {(r["bootstrap"], r["config"]): r for r in rows}
    boots = sorted({r["bootstrap"] for r in rows})
    print(f"\nPaired differences vs jitter (mean ± SE over {len(boots)} bootstraps; "
          f"higher is better for logscore/ll/auroc, lower for brier/tv/l1):")
    for config in ("cat_tau3", "cat_tau1", "dropped"):
        print(f"  {config}:")
        for m in metrics:
            d = [by[(b, config)].get(m, np.nan) - by[(b, "jitter")].get(m, np.nan) for b in boots]
            d = np.array([x for x in d if np.isfinite(x)])
            if d.size:
                print(f"    {m:22s} {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else 0:.4f} (n={d.size})")
    print("\nPer-config means:")
    for config in CONFIGS:
        vals = {m: np.nanmean([by[(b, config)].get(m, np.nan) for b in boots]) for m in metrics}
        print(f"  {config:9s} " + "  ".join(f"{m}={v:.4f}" for m, v in vals.items()))


def main():
    global MAX_EM_ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "analyze"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-em-iters", type=int, default=MAX_EM_ITERS,
                    help="override only for smoke tests; the comparison uses the default")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--reverse", action="store_true",
                    help="run the remaining fits in reverse order (for a concurrent second runner)")
    args = ap.parse_args()
    MAX_EM_ITERS = args.max_em_iters
    if args.mode == "run":
        run(args.out, args.n_boot, n_workers=args.workers, reverse=args.reverse)
    else:
        analyze(args.out, args.n_boot)


if __name__ == "__main__":
    main()
