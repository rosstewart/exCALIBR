#!/usr/bin/env python3
"""P-KNN (Lin & Brandes 2026) on every ExCALIBR-MV performance evaluation.
Driven by the analysis notebook's "pknn" section
(analyze_mv_pipeline_output.run_pknn_analyses -> run_all); each stage skips
outputs already on disk.

Stages:
    prep        build each evaluated dataset (same builder, rows and canonical
                config as the paper's results tables), run the MV analysis once
                for its gene prior and in-sample points -> prep/<key>.npz
    run         leave-one-out P-KNN per dataset and setting -> <key>/<setting>.csv
    genomewide  the authors' genome-wide setup on our predictors, gene held out
    concordance our predictor scores vs dbNSFP v5.2's
    vus         P-KNN points for the paper's ClinVar VUS
    selftest    wrapper fidelity against the authors' serial bootstrap

    nice -n 19 python -m mv_analysis.comparison_methods.pknn_runs <stage> [--workers N]

Settings (functional / combined use the MV gene prior; predictors 0.0441):
    main       N_calibration 10, regularization = labelled set (TP53 protocol)
    gnomadreg  N_calibration 10, regularization = the gene's gnomAD variants
    N5/N25     N_calibration sweep, labelled-set regularization
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import io
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mv_analysis import config  # noqa: E402

OUT = Path(config.PKNN_DIR)
GENOME_WIDE_PRIOR = 0.0441
SETTINGS = {
    "main": dict(n_cal=10, reg="labelled"),
    "gnomadreg": dict(n_cal=10, reg="gnomad"),
    "N5": dict(n_cal=5, reg="labelled"),
    "N25": dict(n_cal=25, reg="labelled"),
}
TABLES = {
    "functional": Path(config.PAPER_FUNCTIONAL_DIR) / "functional_all_results_table.csv",
    "predictor": Path(config.PAPER_PREDICTOR_DIR) / "predictor_results_table.csv",
    "combined": Path(config.PAPER_COMBINED_DIR) / "combined_results_table.csv",
}
EVIDENCE_DIRECTION = "evidence_direction (>=1 / <=-1)"


def _single_threaded():
    """Pool workers forked from a process that already loaded NumPy inherit
    its full BLAS/OpenMP thread pools (128 threads each); cap them at one."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def datasets():
    """(key, panel, gene_set, gene, cluster_idx, canonical config) for every
    row of the paper's per-gene results tables; the canonical config is the
    best MV row by (mcc, determinate), as in make_panel_performance.per_gene."""
    out = []
    for panel, path in TABLES.items():
        d = pd.read_csv(path)
        d = d[(d["threshold"] == EVIDENCE_DIRECTION) & d["method"].str.startswith("MV")]
        if "gene_set" not in d.columns:
            d = d.assign(gene_set=panel)
        best = d.sort_values(["mcc", "determinate"], ascending=False).groupby(["gene_set", "gene", "cluster"]).head(1)
        for r in best.itertuples():
            if not config.paper_includes(r.gene_set):
                continue
            cl = 0 if r.cluster == "gene" else int(str(r.cluster).replace("cluster", "")) - 1
            key = f"{r.gene_set}__{r.gene}" + (f"__c{cl}" if cl else "")
            out.append(dict(key=key, panel=panel, gene_set=r.gene_set, gene=r.gene, cluster_idx=cl,
                            config=r.config, n_pathogenic=int(r.n_pathogenic), n_benign=int(r.n_benign),
                            mv_mcc=float(r.mcc)))
    return out


def _prep_one(ds):
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from mv_analysis.report import _both_modality_mask, _eval_labels, _uv_nonconflicting_points
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    warnings.filterwarnings("ignore")
    key, gs, gene, cl = ds["key"], ds["gene_set"], ds["gene"], ds["cluster_idx"]
    rj = config.PAPER_RESULTS_JSON
    t0 = time.time()
    dataset_name = config.list_gene_cluster_variants(gene, gs, results_json=rj)[cl]
    ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=rj)
    an = build_gene_set_analysis(ms, gene.lower(), rj, dataset_name=dataset_name, gene_set=gs)
    with fast_results_json(rj), contextlib.redirect_stdout(io.StringIO()):
        an.run(partial_pattern_mode="trust_global", n_jobs=4, **DEFAULT_RUN_KWARGS)
    r = an.results[ds["config"]]
    X = np.asarray(ms.scores, float)
    sa = np.asarray(ms.sample_assignments, bool)
    eval_mask, eval_y = _eval_labels(ms, an.p_idx, an.b_idx, gene_set=gs)
    plp = sa[:, an.p_idx] if an.p_idx is not None else np.zeros(len(X), bool)
    blb = sa[:, an.b_idx] if an.b_idx is not None else np.zeros(len(X), bool)
    measured = ~np.isnan(X).all(axis=1)
    labelled = (plp | blb) & measured
    gnomad = (sa[:, an.g_idx] & measured) if an.g_idx is not None else np.zeros(len(X), bool)
    # population variants on the evaluation footing of the controls (combined:
    # measured by both evidence types), and the independent baseline's points
    # for every row, for the evidence-strength distributions
    gnomad_eval = gnomad & _both_modality_mask(ms) if gs.startswith("combined") else gnomad
    uv = _uv_nonconflicting_points(gene.lower() if gs == "labelseq" else gene.upper(), gs, ms)  # as mv_cockpit
    uv_points = np.full(len(X), np.nan) if uv is None else np.nan_to_num(np.asarray(uv, float))
    prior = GENOME_WIDE_PRIOR if gs == "predictor" else float(r["median_prior"])
    np.savez_compressed(
        OUT / "prep" / f"{key}.npz", X=X, plp=plp, blb=blb, labelled=labelled, eval_mask=eval_mask,
        gnomad=gnomad, gnomad_eval=gnomad_eval, uv_points=uv_points, variants=np.asarray([str(v) for v in ms.kept_variants]),
        dims=np.asarray([str(n) for n in ms.dataset_names]), mv_points=np.asarray(r["points"], float),
        sample_names=np.asarray([str(n) for n in ms.sample_names]), sa=sa)
    meta = dict(ds, dataset_name=dataset_name, prior=prior, mv_median_prior=float(r["median_prior"]),
                n_labelled=int(labelled.sum()), n_eval=int(eval_mask.sum()), n_gnomad=int(gnomad.sum()),
                n_dims=int(X.shape[1]), eval_P=int(eval_y.sum()), eval_B=int((eval_y == 0).sum()),
                seconds=round(time.time() - t0, 1))
    json.dump(meta, open(OUT / "prep" / f"{key}.json", "w"), indent=1)
    return meta


def stage_prep(args):
    (OUT / "prep").mkdir(parents=True, exist_ok=True)
    todo = [d for d in datasets() if (not args.keys or d["key"] in args.keys)
            and (args.force or not (OUT / "prep" / f"{d['key']}.json").exists())]
    print(f"prep: {len(todo)} datasets", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_prep_one, d): d["key"] for d in todo}
        for f in as_completed(futs):
            try:
                m = f.result()
                print(f"  {m['key']}: labelled {m['n_labelled']} eval {m['n_eval']} (P {m['eval_P']}/B {m['eval_B']}) "
                      f"gnomAD {m['n_gnomad']} dims {m['n_dims']} prior {m['prior']:.3f} [{m['seconds']}s]", flush=True)
            except Exception:
                print(f"  {futs[f]} FAILED\n{traceback.format_exc()}", flush=True)


def excalibr_c(prior):
    """The LR base ExCALIBR/MV uses at this prior (Tavtigian-rule fit), so
    P-KNN posteriors become points on the same scale as the MV and UV points
    they are compared with. P-KNN's own base (max over the four single
    posterior thresholds, P_KNN_CPU.get_logbase) diverges from it at high
    priors (0.22: 1724 vs 105; 0.37: 6.8e5 vs 804) and is kept as
    points_pknn_logbase for a sensitivity check."""
    _single_threaded()
    from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
    return float(get_tavtigian_constant(prior))


def load_prep(key):
    z = np.load(OUT / "prep" / f"{key}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}, json.load(open(OUT / "prep" / f"{key}.json"))


def _run_one(key, setting, n_jobs, folds):
    from mv_analysis.comparison_methods import pknn
    warnings.filterwarnings("ignore")
    d, meta = load_prep(key)
    s = SETTINGS[setting]
    lab = d["labelled"]
    X, y = d["X"][lab], d["plp"][lab].astype(int)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return key, setting, "skipped: one class", 0.0
    reg = "labelled" if s["reg"] == "labelled" else d["X"][d["gnomad"]]
    if s["reg"] == "gnomad" and len(reg) == 0:
        return key, setting, "skipped: no gnomAD rows", 0.0
    query = d["eval_mask"][lab]
    t0 = time.time()
    post, drops = pknn.leave_one_out(X, y, meta["prior"], s["n_cal"], reg=reg, query_mask=query,
                                     n_jobs=n_jobs, folds=folds)
    p_path, p_ben, pts, pts_own = (np.full(len(y), np.nan) for _ in range(4))
    p_path[query], p_ben[query], pts[query] = pknn.to_points(post[query], meta["prior"], logbase=excalibr_c(meta["prior"]))
    pts_own[query] = pknn.to_points(post[query], meta["prior"])[2]
    out = pd.DataFrame(dict(variant=d["variants"][lab], label=y, query=query, P_path=p_path, P_ben=p_ben,
                            points=pts, points_pknn_logbase=pts_own,
                            fold_scheme="LOO" if folds is None else f"{folds}-fold",
                            n_cal=min(s["n_cal"], len(y) - 2), reg=s["reg"], prior=meta["prior"]))
    (OUT / key).mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / key / f"{setting}.csv", index=False)
    np.save(OUT / key / f"{setting}_post.npy", post[query].astype(np.float32))
    return key, setting, f"ok ({int(query.sum())} scored, max dropped cols {int(drops.max()) if len(drops) else 0})", \
        time.time() - t0


def stage_run(args):
    keys = args.keys or sorted(p.stem for p in (OUT / "prep").glob("*.json") if config.paper_includes(p.stem))
    jobs = [(k, s) for s in args.settings for k in keys if args.force or not (OUT / k / f"{s}.csv").exists()]
    # largest first so the long LOO runs start early
    size = {k: json.load(open(OUT / "prep" / f"{k}.json"))["n_labelled"] for k in keys}
    jobs.sort(key=lambda j: -size[j[0]])
    print(f"run: {len(jobs)} jobs, {args.workers} workers x {args.inner} inner", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_run_one, k, s, args.inner, args.folds): (k, s) for k, s in jobs}
        for f in as_completed(futs):
            try:
                k, s, msg, sec = f.result()
                print(f"  {k} {s}: {msg} [{sec / 60:.1f} min]", flush=True)
            except Exception:
                print(f"  {futs[f]} FAILED\n{traceback.format_exc()}", flush=True)


GW_DIR = Path("/data/ross/assay_calibration/pknn")
GW_TOOLS = {"REVEL": "REVEL_score", "AlphaMissense": "AlphaMissense_score", "MutPred2": "MutPred2_score"}
GW_N_CAL = 100
GW_LOGBASE = 1124  # Pejaver et al. 2022 c at prior 0.0441, as in the authors' genome-wide notebooks


def _gw_tables():
    cal = pd.read_csv(GW_DIR / "calibration_data_dbNSFP52.csv", low_memory=False)
    reg = pd.read_csv(GW_DIR / "regularization_data_dbNSFP52.csv", low_memory=False)
    for df in (cal, reg):
        df[list(GW_TOOLS.values())] = df[list(GW_TOOLS.values())].apply(pd.to_numeric, errors="coerce")
    cal = cal[cal[list(GW_TOOLS.values())].notna().any(axis=1)].reset_index(drop=True)
    reg = reg[reg[list(GW_TOOLS.values())].notna().any(axis=1)].reset_index(drop=True)
    return cal, reg


def stage_genomewide(args):
    """P-KNN's published genome-wide setup on our three predictors: the
    authors' default ClinVar 2019 calibration and gnomAD regularization sets
    (dbNSFP v5.2 scores), N_calibration 100, 3% regularization, prior 0.0441,
    LR base 1124. Leave-one-gene-out: the evaluated gene's variants are
    removed from the calibration set before scoring its labelled rows."""
    _single_threaded()
    from mv_analysis.comparison_methods import pknn
    warnings.filterwarnings("ignore")
    cal, reg = _gw_tables()
    keys = args.keys or sorted(p.stem for p in (OUT / "prep").glob("predictor__*.json"))
    for key in keys:
        out_csv = OUT / key / "genomewide.csv"
        if out_csv.exists() and not args.force:
            continue
        d, meta = load_prep(key)
        dims = list(d["dims"])
        cols = [GW_TOOLS[n] for n in dims]
        e = d["eval_mask"]
        q = d["X"][e]
        keep = cal["genename"].astype(str).str.split(";").apply(lambda g: meta["gene"] not in g).to_numpy()
        cal_g = cal[keep]
        assert not (cal_g["genename"].astype(str).str.split(";").apply(lambda g: meta["gene"] in g)).any()
        t0 = time.time()
        post, drops = pknn.bootstrap_posteriors(cal_g[cols].to_numpy(float), cal_g["ClinVar_annotation"].to_numpy(int),
                                                q, reg[cols].to_numpy(float), GENOME_WIDE_PRIOR, GW_N_CAL,
                                                n_jobs=args.inner)
        p_path, p_ben, pts = pknn.to_points(post, GENOME_WIDE_PRIOR, logbase=GW_LOGBASE)
        (OUT / key).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(variant=d["variants"][e], label=d["plp"][e].astype(int), query=True, P_path=p_path,
                          P_ben=p_ben, points=pts, fold_scheme="leave-one-gene-out", n_cal=GW_N_CAL,
                          reg="gnomAD (genome-wide)", prior=GENOME_WIDE_PRIOR)).to_csv(out_csv, index=False)
        np.save(OUT / key / "genomewide_post.npy", post.astype(np.float32))
        print(f"  {key}: {e.sum()} scored, calibration {len(cal_g)} (removed {int((~keep).sum())} {meta['gene']}), "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)


def stage_genomewide_combined(args):
    """Genome-wide P-KNN evidence from the three predictors alone for the
    combined models' evaluation rows (variants with functional and predictor
    scores), exactly as stage_genomewide (same calibration and regularization
    sets, N_calibration, prior and LR base, the evaluated gene's labelled
    variants removed from the calibration set). Combined controls take their
    labels from the functional data, so not all of them are predictor-model
    controls; scoring every row keeps one procedure. The comparison adds these
    points to the functional evidence (run_pknn_comparison)."""
    _single_threaded()
    from mv_analysis.comparison_methods import pknn
    warnings.filterwarnings("ignore")
    cal, reg = _gw_tables()
    keys = args.keys or sorted(p.stem for p in (OUT / "prep").glob("combined__*.json") if config.paper_includes(p.stem))
    for key in keys:
        out_csv = OUT / key / "genomewide_predictors.csv"
        if out_csv.exists() and not args.force:
            continue
        d, meta = load_prep(key)
        dims = [str(x) for x in d["dims"]]
        idx = [i for i, n in enumerate(dims) if n in GW_TOOLS]
        cols = [GW_TOOLS[dims[i]] for i in idx]
        e = d["eval_mask"]
        q = d["X"][e][:, idx]
        keep = cal["genename"].astype(str).str.split(";").apply(lambda g: meta["gene"] not in g).to_numpy()
        cal_g = cal[keep]
        t0 = time.time()
        post, drops = pknn.bootstrap_posteriors(cal_g[cols].to_numpy(float), cal_g["ClinVar_annotation"].to_numpy(int),
                                                q, reg[cols].to_numpy(float), GENOME_WIDE_PRIOR, GW_N_CAL,
                                                n_jobs=args.inner)
        p_path, p_ben, pts = pknn.to_points(post, GENOME_WIDE_PRIOR, logbase=GW_LOGBASE)
        (OUT / key).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(variant=d["variants"][e], label=d["plp"][e].astype(int), points=pts,
                          fold_scheme="leave-one-gene-out", n_cal=GW_N_CAL, reg="gnomAD (genome-wide)",
                          prior=GENOME_WIDE_PRIOR)).to_csv(out_csv, index=False)
        print(f"  {key}: {e.sum()} scored on {'+'.join(cols)}, calibration {len(cal_g)} "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)


def stage_genomewide_ingene(args):
    """Genome-wide P-KNN as published, the evaluated gene's labelled variants
    left in the calibration set (in-sample, like the other baselines of the
    pooled performance table), for the predictor models' evaluation rows and,
    from the three predictors alone, the combined models' evaluation rows.
    Settings as stage_genomewide. Writes <key>/genomewide_ingene.csv."""
    _single_threaded()
    from mv_analysis.comparison_methods import pknn
    warnings.filterwarnings("ignore")
    cal, reg = _gw_tables()
    keys = args.keys or sorted(p.stem for p in (OUT / "prep").glob("*.json")
                               if p.stem.startswith(("predictor__", "combined__")) and config.paper_includes(p.stem))
    for key in keys:
        out_csv = OUT / key / "genomewide_ingene.csv"
        if out_csv.exists() and not args.force:
            continue
        d, meta = load_prep(key)
        dims = [str(x) for x in d["dims"]]
        idx = [i for i, n in enumerate(dims) if n in GW_TOOLS]
        cols = [GW_TOOLS[dims[i]] for i in idx]
        e = d["eval_mask"]
        t0 = time.time()
        post, drops = pknn.bootstrap_posteriors(cal[cols].to_numpy(float), cal["ClinVar_annotation"].to_numpy(int),
                                                d["X"][e][:, idx], reg[cols].to_numpy(float), GENOME_WIDE_PRIOR,
                                                GW_N_CAL, n_jobs=args.inner)
        p_path, p_ben, pts = pknn.to_points(post, GENOME_WIDE_PRIOR, logbase=GW_LOGBASE)
        (OUT / key).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(variant=d["variants"][e], label=d["plp"][e].astype(int), points=pts,
                          fold_scheme="in-sample (gene in calibration set)", n_cal=GW_N_CAL,
                          reg="gnomAD (genome-wide)", prior=GENOME_WIDE_PRIOR)).to_csv(out_csv, index=False)
        print(f"  {key}: {e.sum()} scored on {'+'.join(cols)}, calibration {len(cal)} (gene included) "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)


def stage_concordance(args):
    """Do our raw predictor scores match dbNSFP v5.2's (the genome-wide
    calibration/regularization sets' source)? Without shared variant keys,
    variants are matched on two tools' scores (rounded to 3 decimals, unique
    keys only) within the gene, and the third tool is compared on the
    matches: n, median |difference| and Pearson r, per held-out tool."""
    warnings.filterwarnings("ignore")
    cal, reg = _gw_tables()
    both = pd.concat([cal, reg], ignore_index=True).rename(columns={v: k for k, v in GW_TOOLS.items()})
    rows = []
    for key in sorted(p.stem for p in (OUT / "prep").glob("predictor__*.json")):
        d, meta = load_prep(key)
        ours = pd.DataFrame(d["X"], columns=list(d["dims"]))
        g = both[both["genename"].astype(str).str.split(";").apply(lambda x: meta["gene"] in x)]
        for other in GW_TOOLS:
            pair = [t for t in GW_TOOLS if t != other]
            o = ours.dropna(subset=pair + [other]).copy()
            t = g.dropna(subset=pair + [other]).copy()
            for df in (o, t):
                df["k"] = df[pair[0]].round(3).astype(str) + "_" + df[pair[1]].round(3).astype(str)
            mm = o.drop_duplicates("k", keep=False).merge(t.drop_duplicates("k", keep=False), on="k",
                                                           suffixes=("_ours", "_dbnsfp"))
            x, y = mm[f"{other}_ours"], mm[f"{other}_dbnsfp"]
            rows.append(dict(key=key, gene=meta["gene"], tool=other, matched_on="+".join(pair), n=len(mm),
                             median_abs_diff=float(np.median(np.abs(x - y))) if len(mm) else np.nan,
                             pearson_r=float(np.corrcoef(x, y)[0, 1]) if len(mm) > 2 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "genomewide_score_concordance.csv", index=False)
    print(out.to_string(index=False))


def _vus_capture():
    """Run analysis/compute_clinvar_vus_points_by_panel.py's panel builders
    with score_gene swapped for a recorder: the same VUS rows (same builders,
    dimension matching, key matching and combined both-modality restriction)
    as the paper's VUS reclassification, without rerunning MV."""
    sys.path.insert(0, str(REPO / "analysis"))
    import compute_clinvar_vus_points_by_panel as V
    from mv_analysis import report
    captured = []

    def _record(gene, gene_set, ms, dataset_name, vus_keys, variant_index=None):
        variant_index = variant_index or {kv: i for i, kv in enumerate(ms.kept_variants)}
        both_mod = report._both_modality_mask(ms) if gene_set == "combined" else None
        idx, keys = [], []
        for v in vus_keys:
            i = variant_index.get(v)
            if i is None or (both_mod is not None and not both_mod[i]):
                continue
            idx.append(i)
            keys.append(str(v))
        captured.append(dict(gene=gene, gene_set=gene_set, dims=[str(n) for n in ms.dataset_names],
                             X=np.asarray(ms.scores, float)[idx], keys=keys))
        print(f"  captured {gene_set} {gene}: {len(keys)} VUS", flush=True)
        return []

    V.score_gene = _record
    tmp = OUT / "vus" / "_unused.csv"
    V.run_functional_panel(tmp)
    V.run_predictor_panel(tmp)
    V.run_combined_panel(tmp)
    return captured


def stage_vus(args):
    """P-KNN points for the paper's true ClinVar VUS. Functional and combined:
    the per-gene main setting (N_calibration 10) calibrated on every labelled
    variant of the gene, regularization = the labelled set (VUS are scored,
    not used). Predictors:
    the genome-wide setting (gene held out) and the per-gene main setting."""
    from mv_analysis.comparison_methods import pknn
    warnings.filterwarnings("ignore")
    (OUT / "vus").mkdir(parents=True, exist_ok=True)
    cap = _vus_capture()
    cal, reg = _gw_tables() if any(c["gene_set"] == "predictor-mv" for c in cap) else (None, None)
    rows = []
    for c in cap:
        if not c["keys"]:
            continue
        gs = "predictor" if c["gene_set"] == "predictor-mv" else c["gene_set"]
        key = f"{gs}__{c['gene']}"
        if not (OUT / "prep" / f"{key}.json").exists():
            print(f"  no prep for {key}", flush=True)
            continue
        d, meta = load_prep(key)
        dims = list(d["dims"])
        Q = c["X"][:, [c["dims"].index(n) for n in dims]]
        keep = ~np.isnan(Q).all(axis=1)
        Q, keys = Q[keep], np.asarray(c["keys"])[keep]
        lab = d["labelled"]
        y = d["plp"][lab].astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        settings = [("main", None)] + ([("genomewide", None)] if gs == "predictor" else [])
        for setting, _ in settings:
            t0 = time.time()
            if setting == "main":
                post, _ = pknn.fit_and_score(d["X"][lab], y, Q, meta["prior"], 10, n_jobs=args.inner)
                pts = pknn.to_points(post, meta["prior"], logbase=excalibr_c(meta["prior"]))[2]
            else:
                cols = [GW_TOOLS[n] for n in dims]
                k = cal["genename"].astype(str).str.split(";").apply(lambda g: meta["gene"] not in g).to_numpy()
                post, _ = pknn.bootstrap_posteriors(cal[k][cols].to_numpy(float), cal[k]["ClinVar_annotation"].to_numpy(int),
                                                    Q, reg[cols].to_numpy(float), GENOME_WIDE_PRIOR, GW_N_CAL,
                                                    n_jobs=args.inner)
                pts = pknn.to_points(post, GENOME_WIDE_PRIOR, logbase=GW_LOGBASE)[2]
            rows += [dict(gene=c["gene"], gene_set=c["gene_set"], variant_key=k_, setting=setting, pknn_points=p_)
                     for k_, p_ in zip(keys, pts)]
            print(f"  {key} {setting}: {len(keys)} VUS [{(time.time() - t0) / 60:.1f} min]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "vus" / "pknn_vus_points.csv", index=False)


def _paper_oob_splits(ms, seeds, K):
    """{bootstrap seed: out-of-bag row indices} for the paper fits, regenerated
    with the same job generator and seeds (Fit.generate_fit_jobs, master seed
    0). Verified against the stored fits: recomputing each fit's validation
    log-likelihood on the regenerated split reproduces its stored val_ll."""
    from src.assay_calibration.fit_utils.fit import Fit
    out = {}
    for b in seeds:
        job = Fit(ms).generate_fit_jobs([K], num_fits=1, bootstrap=True, bootstrap_seed=int(b), check_monotonic=False,
                                        master_seed=0, max_em_iters=1, verbose_overlap=False)[0]
        v = job.get("val_variant_indices")
        out[int(b)] = np.asarray([] if v is None else v, int)
    return out


def stage_selftest(args):
    """Our bounded, seeded bootstrap vs the authors' serial loop (their serial
    branch needs n_query == n_labelled, so the labelled set is its own query
    set). MI noise is the only allowed difference."""
    from mv_analysis.comparison_methods import pknn
    rng = np.random.default_rng(1)
    n = 120
    y = (rng.random(n) < 0.4).astype(int)
    X = np.column_stack([y + rng.normal(0, 0.8, n), y + rng.normal(0, 1.2, n), rng.normal(0, 1, n)])
    X[rng.random(X.shape) < 0.15] = np.nan
    X[np.isnan(X).all(axis=1), 0] = 0.0
    ours, _ = pknn.bootstrap_posteriors(X, y, X, X, 0.2, 10, n_boot=20, n_jobs=4)
    np.random.seed(0)
    theirs = pknn._pk.get_bootstrap_KNN_score(X, X, X, y, Pprior=0.2, n_calibration_in_window=10,
                                              n_bootstrap=20, parallel=False)
    diff = np.abs(ours - theirs)
    print(f"max |ours - theirs| = {diff.max():.4g}; mean = {diff.mean():.4g}; "
          f"identical cells {np.mean(diff < 1e-12):.3f}")
    a = pknn.to_points(ours, 0.2)[2]
    b = pknn.to_points(theirs, 0.2)[2]
    print(f"points: max diff {np.abs(a - b).max():.4g}; same evidence direction {np.mean(np.sign(np.trunc(a)) == np.sign(np.trunc(b))):.3f}")


def run_all(workers=4, inner=8, settings=("main", "gnomadreg", "N5", "N25")):
    """Every stage the paper's P-KNN comparison needs, skipping finished
    outputs (called from the analysis notebook's "pknn" section)."""
    args = argparse.Namespace(workers=workers, inner=inner, keys=None, settings=list(settings), folds=None,
                              force=False)
    OUT.mkdir(parents=True, exist_ok=True)
    stage_prep(args)
    stage_run(args)
    stage_genomewide(args)
    stage_genomewide_combined(args)
    stage_genomewide_ingene(args)
    if not (OUT / "genomewide_score_concordance.csv").exists():
        stage_concordance(args)
    if not (OUT / "vus" / "pknn_vus_points.csv").exists():
        stage_vus(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "run", "selftest", "genomewide", "genomewide-combined", "genomewide-ingene", "concordance", "vus"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--settings", nargs="*", default=["main"])
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"prep": stage_prep, "run": stage_run, "selftest": stage_selftest, "genomewide": stage_genomewide,
     "genomewide-combined": stage_genomewide_combined, "genomewide-ingene": stage_genomewide_ingene,
     "concordance": stage_concordance,
     "vus": stage_vus}[args.stage](args)


if __name__ == "__main__":
    main()
