"""Paper simulation (Phase 4): the consequence-aware strata model and the standard refits against exact truth.

Data come from the consequence-cell generative model (analysis/consequence_cells_sim/generate.py; Gaussian score
classes, exact per-consequence priors and LRs) with exact sample counts and one paper-only option, blb_mis_shift (truly
benign missense score at -delta instead of 0, as when benign missense are mildly hypomorphic but tolerated; CanVIG
Box 2D). Every replicate goes through the paper's own pipeline:

    strata fit      csq_analysis.strata_uni strata + csq_analysis.fits.fit_records (config per cell, default S1_m5),
                    Gaussian components, then every paper arm (csq_analysis.calibrate.calibrate_arm) under both mappings
    criteria        csq_analysis.criteria on the strata fits (gamma_c, d_gamma, tv_D, tv_B, provisional flags)
    standard refits std_pool and std_misctl (missense P/LP and B/LB; gnomAD missense prior) plus std_misctl_bref (same
                    fits, B/LB-only benign reference), as run_igvf_batch calibrates them; first STD_REPS replicates
                    only (they cost ~4x the strata fit per bootstrap)
Each calibration is scored on 10k held-out population draws of the consequence it calibrates (pooled arms on every
consequence): prior error, |log LR| error, posterior overstatement, points vs truth points under the same mapping, and
label-level rates (pathogenic with benign evidence, pathogenic with pathogenic evidence, benign with pathogenic evidence,
benign with benign evidence). Truth also gives the true gamma_c, d_gamma and tv_D (mech_overlap.sim_truth) for the
criteria.

Experiments (design(); frozen with the git hash in bundle.json):
    SA  pooling cost vs true d_gamma / tv_D, both directions: missense detected share x PTV LoF share x PTV location
    SB  benign reference: benign-missense shift x number of missense B/LB
    SC  PTV prior: gnomAD PTV count x true PTV prior
    SD  bootstrap scheme / min stratum x dataset size (strata fit config per cell)
    SF  minimum n: missense P/LP x missense B/LB counts

    python -m csq_analysis.sim make <out_dir> [--reps 20] [--B 50] [--R 8] [--experiments SA,SB,...]
    python -m csq_analysis.sim run <out_dir> --index 0 --per-index 8 --workers 16      (hpc/submit_csq.sh, MODULE)
    python -m csq_analysis.sim collect <out_dir>
    python -m csq_analysis.sim smoke [--B 4 --R 2]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
MAPPINGS = ("tavtigian", "lp_anchored")
B_DEFAULT, R_DEFAULT, REPS = 50, 8, 20
STRATA_ARMS = ("pool", "mis", "mis_Ball", "mis_Bmis_syn", "mis_gall", "ptv_poolprior", "ptv_gptv", "ptv_gmisptv",
               "ptv_gall")
STD_ARMS = ("std_pool", "std_misctl", "std_misctl_bref")
STD_REPS = 10
EXPERIMENTS = ("SA", "SB", "SC", "SD", "SF")


def _eval_csqs(arm):
    if arm in ("pool", "std_pool"):
        return ("missense", "PTV", "synonymous")
    return ("PTV",) if arm.startswith("ptv") else ("missense",)


# ── design ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _mis_mech(g):
    """Missense mechanism shares with measured share g in a fitness assay (stability:activity kept at 0.53:0.47)."""
    st = round(g * 0.53, 4)
    return dict(stability=st, activity=round(g - st, 4), other=round(1 - g, 4))


def _cell(exp, size="mid", **over):
    from analysis.consequence_cells_sim import generate as G
    return G.make_cell(exp, size, "realistic", assay="fitness", **over)


def _blb(n_mis, n_syn=55):
    return dict(n_blb=n_syn + n_mis, blb_mix=dict(synonymous=n_syn / (n_syn + n_mis), missense=n_mis / (n_syn + n_mis),
                                                  PTV=0.0))


def design(experiments=EXPERIMENTS):
    cells = []
    if "SA" in experiments:
        for size, g, tau, loc in itertools.product(("mid", "large"), (0.4, 0.6, 0.8, 0.95), (0.5, 0.8, 0.95), (1.0, 1.5)):
            cells.append(_cell("SA", size, kappa=3.5, mis_mech=_mis_mech(g), tau_ptv=tau, ptv_loc=loc))
        for g, tau in itertools.product((0.4, 0.6, 0.8, 0.95), (0.5, 0.8, 0.95)):          # weaker separation
            cells.append(_cell("SA", "mid", kappa=2.0, mis_mech=_mis_mech(g), tau_ptv=tau))
        for size, g in itertools.product(("mid", "large"), (0.6, 0.95)):                   # small PTV shift
            cells.append(_cell("SA", size, kappa=3.5, mis_mech=_mis_mech(g), tau_ptv=0.95, ptv_loc=1.25))
    if "SB" in experiments:
        for d, nb, g in itertools.product((0.0, 0.3, 0.6), (3, 10, 30), (0.6, 0.95)):
            cells.append(_cell("SB", "mid", mis_mech=_mis_mech(g), blb_mis_shift=d, **_blb(nb)))
    if "SC" in experiments:
        for npt, pi in itertools.product((5, 20, 80), (0.5, 0.9)):
            cells.append(_cell("SC", "mid", mis_mech=_mis_mech(0.8), pi_prot=dict(PTV=pi),
                               pop_share=dict(synonymous=0.22, PTV=npt / 500, missense=0.78 - npt / 500)))
    if "SD" in experiments:
        for size, g, cfg in itertools.product(("small", "mid", "large"), (0.6, 0.95),
                                              ("S1_m5", "S0_m5", "S2n5_m1", "S1_m1")):
            cells.append(_cell("SD", size, mis_mech=_mis_mech(g), fit_config=cfg, std=cfg == "S1_m5"))
    if "SF" in experiments:
        for nm, nb, g in itertools.product((3, 5, 10, 20, 40), (3, 10, 30), (0.6, 0.95)):
            cells.append(_cell("SF", "mid", mis_mech=_mis_mech(g), n_plp=nm + 20,
                               plp_mix=dict(synonymous=0.0, missense=nm / (nm + 20), PTV=20 / (nm + 20)), **_blb(nb)))
    seen, out = set(), []
    for c in cells:
        if c["cell_id"] not in seen:
            seen.add(c["cell_id"]); out.append(c)
    return out


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _split_exact(rng, n, mix):
    """Deterministic consequence counts (largest remainder), replacing generate._split's multinomial draw."""
    from analysis.consequence_cells_sim.generate import CONSEQUENCES
    ks = [k for k in CONSEQUENCES if mix.get(k, 0) > 0]
    p = np.array([mix[k] for k in ks], float)
    p = p / p.sum()
    raw = n * p
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt))[:int(n - cnt.sum())]:
        cnt[i] += 1
    return dict(zip(ks, cnt))


def make_data(c, rep):
    """generate.make_data with exact sample counts, plus the benign-missense shift (rows and evaluation draws)."""
    from analysis.consequence_cells_sim import generate as G
    orig = G._split
    G._split = _split_exact
    try:
        d = G.make_data(G.rng_for(c, rep), c)
    finally:
        G._split = orig
    dl = float(c.get("blb_mis_shift", 0.0))
    if dl:
        sel = (d["csq"] == "missense") & ~d["path"]
        d["score"] = np.where(sel, d["score"] - dl, d["score"])
        e = d["eval"]["missense"]
        e["x"] = np.where(~e["is_path"], e["x"] - dl, e["x"])
    return d


def to_rows(d, c, ds="sim"):
    """Replicate rows in csq_analysis.annotate format (one row per variant; strata keys use plp/blb/pop/syn, csq,
    the SpliceAI flag in splice-measuring assays, and no AF proxy)."""
    m = d["mask"]
    n = len(d["score"])
    return pd.DataFrame(dict(dataset=ds, variant_id=[f"v{i}" for i in range(n)], score=d["score"].astype(float),
                             csq=d["csq"], plp=m[:, 0], blb=m[:, 1], pop=m[:, 2], syn=m[:, 3], vus=False,
                             spliceai_max=np.where(d["hi"], 0.5, 0.0), gnomad_maf=0.0,
                             detects_splice=c["assay"] == "fitness", path=d["path"]))


# ── truth ────────────────────────────────────────────────────────────────────────────────────────────────────────

def true_log_lr(x, c, csq):
    """generate.true_log_lr, with benign missense at -blb_mis_shift."""
    from scipy.stats import norm
    from analysis.consequence_cells_sim import generate as G
    dl = float(c.get("blb_mis_shift", 0.0))
    if not dl or csq != "missense":
        return G.true_log_lr(x, c, csq)
    t = G.truth(c, csq)

    def mix(shares, nor_loc):
        lc = G._comp_logpdf(x, c)
        lc["nor"] = norm.logpdf(x, nor_loc, 1)
        terms = [np.log(v) + lc[k] for k, v in shares.items() if v > 0]
        return np.logaddexp.reduce(np.vstack(terms), axis=0)
    return mix(t["P"], 0.0) - mix(t["B"], -dl)


def truth_row(c):
    from scipy.stats import norm
    from analysis.consequence_cells_sim import generate as G
    from analysis.consequence_cells_sim.mech_overlap import sim_truth
    k = c["kappa"]
    grid = np.linspace(-k * max(c.get("ptv_loc", 1.0), 1.0) - 6, 6, 2001)
    r = {f"true_pi__{x}": G.truth(c, x)["pi"] for x in G.CONSEQUENCES} | sim_truth(c, grid)
    dl = float(c.get("blb_mis_shift", 0.0))
    r["true_tv_B"] = float(2 * norm.cdf(dl / 2) - 1)        # TV of N(-dl, 1) vs N(0, 1)
    return r


# ── evaluation ───────────────────────────────────────────────────────────────────────────────────────────────────

def evaluate(row, p, csq, res, ev, c, mapping):
    from analysis.calibration_quality.run import (conservative_lr, lr_bounds, points_of, posterior, posterior_metrics,
                                                  truth_points)
    from analysis.consequence_cells_sim import generate as G
    pi = G.truth(c, csq)["pi"]
    x, is_path = ev[csq]["x"], ev[csq]["is_path"]
    tl = true_log_lr(x, c, csq)
    prior = float(res["prior"])
    row[f"{p}_prior"] = prior
    row[f"{p}_prior_err"] = prior - pi
    l5, l50, l95 = lr_bounds(res, x)
    err = np.abs(l50 - tl)
    row[f"{p}_llr_mae"] = float(np.nanmean(err))
    tail = np.abs(tl) > np.log(2)
    row[f"{p}_llr_mae_tail"] = float(np.nanmean(err[tail])) if tail.any() else np.nan
    lr_c = conservative_lr(l5, l95)
    pm = posterior_metrics(posterior(lr_c, prior), posterior(np.exp(tl), pi), lr_c)
    for k in ("post_mae", "post_over", "post_over_mean"):
        row[f"{p}_{k}"] = pm[k]
    est = points_of(res, x)
    tru = truth_points(tl, pi, mapping)
    row[f"{p}_pts_over"] = float(((est > np.maximum(tru, 0)) | (est < np.minimum(tru, 0))).mean())
    row[f"{p}_pts_wrong"] = float((((est > 0) & (tru < 0)) | ((est < 0) & (tru > 0))).mean())
    row[f"{p}_pts_det"] = float((est != 0).mean())
    row[f"{p}_truth_pts_det"] = float((tru != 0).mean())
    row[f"{p}_pts_mean_abs_err"] = float(np.abs(est - tru).mean())
    for lab, sel in (("path", is_path), ("ben", ~is_path)):
        row[f"{p}_{lab}_n"] = int(sel.sum())
        row[f"{p}_{lab}_benev"] = float((est[sel] < 0).mean()) if sel.any() else np.nan
        row[f"{p}_{lab}_pathev"] = float((est[sel] > 0).mean()) if sel.any() else np.nan


# ── standard refits ──────────────────────────────────────────────────────────────────────────────────────────────

def _std_masks(d):
    """{arm: (fit mask, calibration mask, benign_method)}; the missense refits fit with gnomAD all and calibrate with
    gnomAD missense, as hpc/prepare.py + run_igvf_batch --population-type gnomAD_missense_nsSNV do."""
    m = d["mask"].astype(bool)
    mis = d["csq"] == "missense"
    fit_ctl = m.copy()
    fit_ctl[~mis, 0] = False
    fit_ctl[~mis, 1] = False
    cal_ctl = fit_ctl.copy()
    cal_ctl[~mis, 2] = False
    return {"std_pool": (m, m, "avg"), "std_misctl": (fit_ctl, cal_ctl, "avg"),
            "std_misctl_bref": (fit_ctl, cal_ctl, "benign")}


def _std_calibrate(ss, fits, seeds, K, mapping, bm):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    import tempfile
    lg = logging.getLogger("csqsim")
    lg.setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(dataset_csv="", dataset_name="sim", output_dir=tmp, components=[K], benign_method=bm,
                             use_median_prior=True, use_2c_equation=False, point_values=list(range(1, 9)), n_jobs=1,
                             acmg_mapping_method=mapping)
        try:
            return process_component_fits(fits=fits, scoreset=ss, config=cfg, n_c=K, logger=lg,
                                          bootstrap_seeds=np.asarray(seeds))
        except Exception:
            return None


# ── one replicate ────────────────────────────────────────────────────────────────────────────────────────────────

def run_cell(c, rep, B=B_DEFAULT, R=R_DEFAULT):
    from analysis.consequence_cells_sim import generate as G
    from analysis.consequence_cells_sim.run import fit_arm, scoreset
    from csq_analysis import criteria as CR
    from csq_analysis.calibrate import calibrate_arm
    from csq_analysis.fits import FIT_CONFIGS, fit_records, layout_rows
    t0 = time.time()
    row = {k: (v if not isinstance(v, dict) else json.dumps(v, sort_keys=True)) for k, v in c.items()}
    row.update(rep=int(rep), B=B, R=R, **truth_row(c))
    K = G.n_components(c)
    cfgname = c.get("fit_config", "S1_m5")
    try:
        d = make_data(c, rep)
        ev = d["eval"]
        rows = to_rows(d, c)
        mis, ptv = rows.csq.eq("missense"), rows.csq.eq("PTV")
        row.update(n_plp_mis=int((rows.plp & mis).sum()), n_plp_ptv=int((rows.plp & ptv).sum()),
                   n_blb_mis=int((rows.blb & mis).sum()), n_blb=int(rows.blb.sum()),
                   n_pop_mis=int((rows["pop"] & mis).sum()), n_pop_ptv=int((rows["pop"] & ptv).sum()))
        # strata fit, criteria and paper arms
        us, kw = layout_rows("sim", rows, FIT_CONFIGS[cfgname])
        kw = dict(kw, force_gaussian=True)
        recs = fit_records(us, kw, K, list(range(B)), R, "sim", cfgname)
        row["strata_n_fits"] = sum(1 for r in recs.values() if f"{K}c" in r)
        row["n_strata"] = len(us.names)
        S = CR.setup(us, "avg")
        per = [CR.one_fit(r[f"{K}c"]["fit"], S) for r in recs.values() if f"{K}c" in r]
        if per:
            cr = CR.summarise(per)
            row.update({f"est_{k}": v for k, v in cr.items()})
            row.update({f"est_{k}": v for k, v in CR.flags(cr).items()})
            row["est_estimator"] = S["method"]
        st = dict(K=K, benign_method="avg", liberal_monotonicity=True, clinvar_release="2025")
        for mapping in MAPPINGS:
            pool_prior = None
            for arm in STRATA_ARMS:
                mp = pool_prior if arm == "ptv_poolprior" else None
                if arm == "ptv_poolprior" and mp is None:
                    row[f"{arm}__{mapping}_ok"] = False
                    continue
                try:
                    cal, res = calibrate_arm(us, recs, arm, mapping, st, manual_prior=mp, n_jobs=1, return_res=True)
                except Exception as e:
                    cal, res = dict(status=f"error: {repr(e)[:200]}"), None
                ok = res is not None
                row[f"{arm}__{mapping}_ok"] = ok
                if not ok:
                    continue
                if arm == "pool":
                    pool_prior = float(res["prior"])
                for csq in _eval_csqs(arm):
                    evaluate(row, f"{arm}__{mapping}__{csq}", csq, res, ev, c, mapping)
        # standard refits
        if c.get("std", True) and rep < STD_REPS:
            sc = d["score"].astype(float)
            fitted = {}
            for arm, (mf, mc, bm) in _std_masks(d).items():
                key = mf.tobytes()
                if key not in fitted:
                    ss_f, keep = scoreset(sc, mf)
                    fitted[key] = (keep, *fit_arm(ss_f, K, B, R, True))
                keep, fits, seeds = fitted[key]
                row[f"{arm}_n_fits"] = len(fits)
                if not fits:
                    continue
                ss_c, keep_c = scoreset(sc, mc)
                for mapping in MAPPINGS:
                    res = _std_calibrate(ss_c, fits, seeds, K, mapping, bm)
                    row[f"{arm}__{mapping}_ok"] = res is not None
                    if res is None:
                        continue
                    for csq in _eval_csqs(arm):
                        evaluate(row, f"{arm}__{mapping}__{csq}", csq, res, ev, c, mapping)
        row["status"] = "ok"
    except Exception as e:  # recorded, not raised
        import traceback
        row["status"] = f"error: {type(e).__name__}: {e} | {traceback.format_exc()[-400:]}"[:700]
    row["seconds"] = round(time.time() - t0, 1)
    return row


# ── task files ───────────────────────────────────────────────────────────────────────────────────────────────────

def _task_id(t):
    return f"{t['cell']['cell_id']}__rep{t['rep']}"


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return None


def make(out_dir, reps=REPS, B=B_DEFAULT, R=R_DEFAULT, experiments=EXPERIMENTS):
    cells = design(experiments)
    tasks = [dict(cell=c, rep=r, B=B, R=R) for r in range(reps) for c in cells]   # rep-major: early reps first
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks))
    n = pd.Series([c["experiment"] for c in cells]).value_counts().to_dict()
    (out / "bundle.json").write_text(json.dumps(dict(module="csq_analysis.sim", git=_git_hash(), reps=reps, B=B, R=R,
                                                     experiments=list(experiments), cells_per_experiment=n,
                                                     made=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"{len(tasks)} tasks ({len(cells)} cells x {reps} reps; {n}) -> {out / 'tasks.json'}")


def _run_one(out_dir, t):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    f = Path(out_dir) / "runs" / f"{_task_id(t)}.csv"
    if f.exists():
        return
    row = run_cell(t["cell"], t["rep"], t["B"], t["R"])
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    pd.DataFrame([row]).to_csv(tmp, index=False)
    tmp.rename(f)


def run(out_dir, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(out_dir) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    Parallel(n_jobs=workers or 1, verbose=5)(delayed(_run_one)(out_dir, t) for t in mine)


def collect(out_dir):
    out = Path(out_dir)
    fs = sorted((out / "runs").glob("*.csv"))
    n = len(json.loads((out / "tasks.json").read_text()))
    print(f"{len(fs)} of {n} tasks finished")
    if fs:
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        df.to_csv(out / "results.csv.gz", index=False)
        print(f"  {len(df)} rows ({int((df.status == 'ok').sum())} ok) -> {out / 'results.csv.gz'}")


def smoke(B=4, R=2, out=None):
    """One cell per experiment, one replicate, tiny budget: checks the whole pipeline end to end."""
    cells = {}
    for c in design():
        cells.setdefault(c["experiment"], c)
    rows = []
    for c in cells.values():
        r = run_cell(c, 0, B, R)
        print(f"{c['cell_id']}: {r['status'][:300]} ({r['seconds']} s)")
        rows.append(r)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--reps", type=int, default=REPS)
    m.add_argument("--B", type=int, default=B_DEFAULT)
    m.add_argument("--R", type=int, default=R_DEFAULT)
    m.add_argument("--experiments", default=",".join(EXPERIMENTS))
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, required=True)
    r.add_argument("--workers", type=int, default=16)
    c = sub.add_parser("collect")
    c.add_argument("out_dir")
    s = sub.add_parser("smoke")
    s.add_argument("--B", type=int, default=4)
    s.add_argument("--R", type=int, default=2)
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.reps, a.B, a.R, a.experiments.split(","))
    elif a.cmd == "run":
        run(a.out_dir, a.index, a.per_index, a.workers)
    elif a.cmd == "collect":
        collect(a.out_dir)
    else:
        smoke(a.B, a.R, a.out)


if __name__ == "__main__":
    main()
