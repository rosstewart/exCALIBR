"""Step 0: empirical anchors for the consequence-cell simulation, from existing real-data outputs (no new fits).

Three tables, written to OUT:
  priors.csv    per dataset x consequence: the production prior estimator applied to the large labeled-scale cell
                fits, with the population sample composed from ONE consequence's gnomAD cells at a time. The
                pathogenic density is either all P/LP cells ("pool_P") or that consequence's own P/LP cells ("own_P").
                Median over up to N_SEEDS bootstraps. Extreme priors are kept (production would discard a fit whose
                prior saturates; for PTVs that is common, and frac_ge_0p99 records it).
  evidence.csv  per dataset x consequence x sample group: share of variants with pathogenic / benign evidence in the
                manuscript calibration (OOB points where available), split by SpliceAI max DS > 0.2, plus score
                separation (kappa proxy) between P/LP-missense with pathogenic evidence and the benign controls.
  summary.csv   the anchors used to set simulation defaults, by assay class (SGE / VAMP-seq / other).
Caveat: the priors come from the estimator the simulation tests; they set realistic magnitudes, not ground truth.

Run: python -m analysis.consequence_cells_sim.anchors [--n-jobs 24]
"""
import argparse, glob, gzip, json, logging, os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed

REPO = "/home/rcstewart/exCALIBR"
sys.path.insert(0, REPO)
A = "/data/ross/assay_calibration"
H = A + "/missense_only_calibrations/cells/hpc_large_labeled/labeled"
TSV = A + "/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
MAIN = A + "/exc_pp_clinvar2025_calib"
CFG = REPO + "/src/igvf_configs/dataset_configs_aug_2026.json"
OUT = A + "/consequence_cells_sim/step0"
N_SEEDS = 100
SAI_COLS = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]
PTV = {"stop_gained", "frameshift_variant"}
EXCLUDE = ("SFPQ_IGVF",)


def simple_csq(c):
    c = str(c).split("|")[0]
    if c == "synonymous_variant":
        return "synonymous"
    if c == "missense_variant":
        return "missense"
    if c in PTV:
        return "PTV"
    return "other"


def _cfg(ds):
    c = json.load(open(CFG)).get(ds)
    if c is None:
        return None
    c = c if isinstance(c, dict) else {"n_c": c[0], "benign_method": c[1]}
    nc = c["n_c"] if c["n_c"] in ("2c", "3c") else "3c"
    return nc, c.get("benign_method", "avg")


def _meta():
    import analysis.config as acfg
    m = pd.read_csv(acfg.CURATED_DATASET_METADATA_CSV)
    m["assay_class"] = np.where(m.vamp_sge.eq("SGE"), "SGE", np.where(m.vamp_sge.eq("VAMP-seq"), "VAMP-seq", "other"))
    return m.set_index("dataset")[["gene", "assay_type", "assay_class", "vamp_sge"]]


# ----------------------------------------------------------------------------------------------------------------
# priors from the cell fits
# ----------------------------------------------------------------------------------------------------------------
def _avg(W, n, sel):
    if not sel.any():
        return None
    w = (W[sel] * n[sel, None]).sum(0) / n[sel].sum()
    return w / w.sum()


def priors_one(ds):
    from analysis.membership_cells.cells import Cells
    from src.assay_calibration.fit_utils.point_ranges import get_fit_prior
    warnings.filterwarnings("ignore")
    cf = _cfg(ds)
    if cf is None or not os.path.exists(f"{H}/cells/{ds}.npz"):
        return []
    nc, bm = cf
    cells = Cells.load(f"{H}/cells/{ds}.npz")
    t = cells.table.reset_index(drop=True)
    mask = np.array([list(m) for m in t["mask"]], bool)
    n = t["n"].to_numpy(float)
    csq = t["consequence"].map(simple_csq).to_numpy()
    sai_tag = t["consequence"].str.contains(r"\|SpliceAI").to_numpy()
    recs = []
    for f in sorted(glob.glob(f"{H}/raw/{ds}__{nc}__*.json.gz")):
        for seed, r in json.load(gzip.open(f)).items():
            rr = r.get(nc)
            if rr and "fit" in rr:
                recs.append((int(seed), rr["fit"]))
    recs = sorted(recs)[:N_SEEDS]
    rows = []
    for c in ("synonymous", "missense", "PTV"):
        for splice_cells in ("all", "noSAI"):
            pop_sel = mask[:, 2] & (csq == c) & ((~sai_tag) if splice_cells == "noSAI" else True)
            if not pop_sel.any():
                continue
            pop_rows = np.isin(cells.cell_of_row, np.where(pop_sel)[0])
            scores = cells.scores[pop_rows]
            sa = np.zeros((len(scores), 4), bool); sa[:, 2] = True
            for pmode in ("pool_P", "own_P"):
                psel = mask[:, 0] & ((csq == c) if pmode == "own_P" else True)
                pri = []
                for seed, fit in recs:
                    W = np.asarray(fit["weights"], float)
                    wP, wB, wG, wS = _avg(W, n, psel), _avg(W, n, mask[:, 1]), _avg(W, n, pop_sel), _avg(W, n, mask[:, 3])
                    if wP is None or (wB is None and wS is None):
                        continue
                    ws = [wP, wB if wB is not None else wS, wG, wS if wS is not None else wB]
                    try:
                        p = get_fit_prior({"fit": {"component_params": fit["component_params"], "weights": ws}},
                                          scores, bm if (wB is not None and wS is not None) else "benign",
                                          pathogenic_idx=0, benign_idx=1, gnomad_idx=2, synonymous_idx=3,
                                          sample_assignments=sa, keep_extreme_prior=True)
                        p = p[0] if isinstance(p, tuple) else p
                    except Exception:
                        p = np.nan
                    pri.append(p)
                pri = np.asarray(pri, float)
                ok = np.isfinite(pri)  # extremes kept (keep_extreme_prior): PTV priors often saturate near 1
                rows.append(dict(dataset=ds, consequence=c, pop_cells=splice_cells, p_mode=pmode, n_pop=int(pop_rows.sum()),
                                 n_plp_c=int(n[mask[:, 0] & (csq == c)].sum()), n_seeds=len(pri), n_valid=int(ok.sum()), frac_ge_0p99=float((pri[ok] >= 0.99).mean()) if ok.any() else np.nan,
                                 prior_median=float(np.median(pri[ok])) if ok.any() else np.nan,
                                 prior_q25=float(np.percentile(pri[ok], 25)) if ok.any() else np.nan,
                                 prior_q75=float(np.percentile(pri[ok], 75)) if ok.any() else np.nan))
    return rows


# ----------------------------------------------------------------------------------------------------------------
# evidence-based anchors from the manuscript calibration
# ----------------------------------------------------------------------------------------------------------------
def evidence_one(ds, df_ds):
    from analysis.membership_cells.cells import load_scoreset, _rows
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    cf = _cfg(ds)
    if cf is None:
        return [], None
    nc, bm = cf
    vf = f"{MAIN}/{ds}/{ds}_{nc}_{bm}_variants.csv"
    if not os.path.exists(vf):
        return [], None
    v = pd.read_csv(vf)
    d = df_ds.copy(); d["Dataset"] = ds
    ss = load_scoreset(ds, d)
    ids, csq, sc, sa, names, sai = _rows(ss)
    m = pd.DataFrame({"variant_id": ids, "csq": [simple_csq(c) for c in csq], "sai": sai})
    v = v.merge(m, on="variant_id", how="left")
    pts = v["oob_points"] if "oob_points" in v.columns and v["oob_points"].notna().any() else v["standard_points"]
    s = v["sample"].astype(str)
    grp = np.where(s.str.contains("Pathogenic"), "P/LP", np.where(s.str.contains("Benign"), "B/LB",
                   np.where(s.str.contains("Synonymous|gnomAD|population", regex=True), "pop/syn", "other")))
    e = pd.DataFrame({"dataset": ds, "csq": v["csq"].fillna("unknown"), "grp": grp, "hi_sai": v["sai"] > 0.2,
                      "path": pts.to_numpy() > 0, "ben": pts.to_numpy() < 0, "score": v["score"]})
    rows = []
    for (c, g, h), x in e.groupby(["csq", "grp", "hi_sai"]):
        rows.append(dict(dataset=ds, consequence=c, group=g, sai_gt_0p2=bool(h), n=len(x),
                         frac_path=x.path.mean(), frac_ben=x.ben.mean()))
    # kappa proxy: |median(P/LP missense with pathogenic evidence) - mean(benign controls)| / sd(benign controls)
    ben = e[(e.grp == "B/LB") | ((e.grp == "pop/syn") & (e.csq == "synonymous"))].score
    pm = e[(e.grp == "P/LP") & (e.csq == "missense") & e.path].score
    kap = float(abs(pm.median() - ben.mean()) / ben.std()) if len(pm) >= 3 and len(ben) >= 5 and ben.std() > 0 else np.nan
    return rows, dict(dataset=ds, kappa_proxy=kap, n_plp_mis_path=len(pm), n_benign_ref=len(ben))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-jobs", type=int, default=24); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = _meta()
    ds_cells = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{H}/cells/*.npz"))
    ds_cells = [d for d in ds_cells if d not in EXCLUDE]
    pr = Parallel(n_jobs=a.n_jobs)(delayed(priors_one)(d) for d in ds_cells)
    pr = pd.DataFrame([r for x in pr for r in x])
    pr = pr.join(meta, on="dataset")
    pr.to_csv(f"{OUT}/priors.csv", index=False)
    print("priors:", pr.dataset.nunique(), "datasets", flush=True)

    ds_main = sorted(d for d in os.listdir(MAIN) if os.path.isdir(f"{MAIN}/{d}") and d not in EXCLUDE)
    base = {d: d.replace("_clinvar_2018", "") for d in ds_main}
    want = set(base.values())
    df = pd.concat([c[c["Dataset"].isin(want)] for c in pd.read_csv(TSV, sep="\t", low_memory=False, chunksize=200000)])
    g = dict(tuple(df.groupby("Dataset")))
    res = Parallel(n_jobs=a.n_jobs)(delayed(evidence_one)(d, g[base[d]]) for d in ds_main if base[d] in g)
    ev = pd.DataFrame([r for x, _ in res for r in x]).join(meta, on="dataset")
    kp = pd.DataFrame([k for _, k in res if k]).join(meta, on="dataset")
    ev.to_csv(f"{OUT}/evidence.csv", index=False); kp.to_csv(f"{OUT}/kappa.csv", index=False)
    print("evidence:", ev.dataset.nunique(), "datasets", flush=True)
    summarize(pr, ev, kp)


def summarize(pr, ev, kp):
    out = []
    ms = set(_meta().index)  # manuscript datasets
    p = pr[pr.dataset.isin(ms) & pr.pop_cells.eq("noSAI")]
    for (c, pm, ac), x in p.groupby(["consequence", "p_mode", "assay_class"]):
        y = x.prior_median.dropna()
        out.append(dict(anchor=f"prior_{c}_{pm}", assay_class=ac, n_datasets=len(y), median=y.median(),
                        q25=y.quantile(.25), q75=y.quantile(.75)))
    e = ev[ev.dataset.isin(ms)]
    def share(sel, col="frac_path"):
        x = e[sel]
        return (x[col] * x.n).sum() / max(x.n.sum(), 1), int(x.n.sum())
    for ac in ("SGE", "VAMP-seq", "other"):
        a = e.assay_class.eq(ac)
        for nm, sel in [("plp_missense_path_share (gamma proxy)", a & e.consequence.eq("missense") & e.group.eq("P/LP")),
                        ("plp_PTV_path_share (tau_PTV proxy)", a & e.consequence.eq("PTV") & e.group.eq("P/LP")),
                        ("blb_path_share (false-positive floor)", a & e.group.eq("B/LB")),
                        ("syn_path_share SAI<=0.2", a & e.consequence.eq("synonymous") & ~e.sai_gt_0p2),
                        ("syn_path_share SAI>0.2", a & e.consequence.eq("synonymous") & e.sai_gt_0p2),
                        ("mis_pop_path_share SAI<=0.2", a & e.consequence.eq("missense") & e.group.eq("pop/syn") & ~e.sai_gt_0p2),
                        ("mis_pop_path_share SAI>0.2", a & e.consequence.eq("missense") & e.group.eq("pop/syn") & e.sai_gt_0p2)]:
            v, nn = share(sel)
            out.append(dict(anchor=nm, assay_class=ac, n_variants=nn, median=v))
        k = kp[kp.dataset.isin(ms) & kp.assay_class.eq(ac)].kappa_proxy.dropna()
        out.append(dict(anchor="kappa_proxy", assay_class=ac, n_datasets=len(k), median=k.median(), q25=k.quantile(.25),
                        q75=k.quantile(.75)))
    # control composition (labeled cells): consequence shares of P/LP and B/LB, from the evidence table
    for g in ("P/LP", "B/LB"):
        x = e[e.group.eq(g)].groupby("consequence").n.sum()
        for c, v in (x / x.sum()).items():
            out.append(dict(anchor=f"{g}_consequence_share_{c}", assay_class="all", n_variants=int(x.sum()), median=v))
    # paired same-gene: missense prior VAMP-seq vs SGE/cell fitness in the same gene
    pm = p[p.consequence.eq("missense") & p.p_mode.eq("own_P")]
    for gene, x in pm.groupby("gene"):
        v = x[x.assay_class.eq("VAMP-seq")].prior_median.dropna()
        f = x[(x.assay_class.eq("SGE")) | x.assay_type.eq("cell fitness")].prior_median.dropna()
        if len(v) and len(f):
            out.append(dict(anchor=f"paired_missense_prior_ratio_VAMP_over_fitness_{gene}", assay_class="paired",
                            n_datasets=len(v) + len(f), median=v.median() / f.median()))
    s = pd.DataFrame(out); s.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 220); print(s.round(4).to_string())


if __name__ == "__main__":
    main()
