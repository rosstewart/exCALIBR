"""External validation of the TP53 reduced-penetrance classification.

Joint calibration classifies every TP53 variant as benign-like, intermediate,
P/LP-like or RPV-like (tp53_rpv_report.py; Supplementary Section sec:rpv).
The calls are checked here against evidence the model never sees:

  germline   NCI TP53 Database germline release (TP53DB_GermlineDownload_r21):
             per carrier family its classification (classic Li-Fraumeni, LFS;
             Li-Fraumeni-like or Chompret criteria; family history of cancer
             only, FH; no family history, noFH) and each tumour's age at
             diagnosis
  somatic    the same database's TCGA/ICGC/GENIE somatic count per variant
  gnomAD     presence in gnomAD v4 (tp53_annotated_variants.csv)
  ClinVar    current TP53 records with a low-penetrance or risk-allele
             classification (NCBI E-utilities, cached in
             clinvar_tp53_low_penetrance.json)

Calls are out of bag: a variant in any training sample (P/LP, B/LB,
gnomAD, synonymous or RPV) is classified only by the bootstrap fits whose
samples left it out (production_oob's per-split masks for tp53__TP53), so
that its own label, and the family evidence behind that label, cannot shape
its call; variants in no sample were never seen by any fit and use all of
them. The classification rule is score_rpv_penetrance's (median over the
fits of each likelihood ratio, the RPV sample's thresholds); with every fit
it reproduces the stored in-sample calls, which are reported alongside.

Every comparison excludes the high-confidence and curated reference-cohort
RPVs (tp53.load_reference_cohorts) the reduced-penetrance sample is drawn
from, all of them, including those left out of the sample as ClinVar B/LB,
since the list was selected partly on the same database's germline-to-
somatic ratio. The low-confidence RPVs, which are
not in the sample, are reported separately as a held-out set. Tests count
each variant once (per-variant share of LFS families, median age, somatic
count; Mann-Whitney U), because families and tumours of one variant are not
independent; family- and tumour-level totals are reported alongside.

Outputs (config.PAPER_TP53_RPV_DIR):
  rpv_validation_by_call.csv       one row per call: variants, carriers'
                                   families, % LFS / FH-or-none families,
                                   median age, median somatic count, % in
                                   gnomAD, ClinVar P/LP and B/LB counts
  rpv_validation_variants.csv      per-variant evidence, out-of-bag call
                                   ("call") and in-sample call ("rpv_class")
  rpv_oob_matrices.npz             cached per-fit RPV likelihood ratios
  rpv_validation_heldout.csv       the held-out low-confidence RPVs
  rpv_validation_clinvar.csv       ClinVar low-penetrance records and calls
  rpv_validation_metrics.csv       deeper validation: per metric and class
                                   (penetrance, A; signal vs benign, B)
  rpv_validation_metrics_{insample,clinvar_plp}.csv  sensitivity analyses
  rpv_validation_incidence.csv     age-specific cancer incidence among
                                   non-proband carriers (Kaplan-Meier)
  rpv_validation_burden.csv        case-control burden (GermlineFrequency)
  rpv_validation_facts.json        numbers the manuscript macros read

Usage:
  python mv_analysis/tp53_rpv_validation.py
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mv_analysis import config  # noqa: E402
from src.assay_calibration.multivariate_data import tp53  # noqa: E402

GERMLINE_DB = "/data/ross/assay_calibration/TP53/TP53DB_GermlineDownload_r21.csv"
CALLS = (("plp_like", "P/LP-like"), ("low_pen_rpv", "RPV-like"),
         ("intermediate", "Intermediate"), ("benign_like", "Benign-like"))
CLINVAR_QUERIES = ('TP53[gene] AND "low penetrance"', 'TP53[gene] AND "risk allele"',
                   'TP53[gene] AND "risk factor"[clinsig]')
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def _calls(config_name):
    f = Path(config.PAPER_TP53_RPV_DIR) / f"TP53_{config_name}_RPV_scores.csv"
    return pd.read_csv(f, index_col=0).rename_axis("Variant").reset_index()


def _oob_matrices(config_name):
    """Per-fit log LR of the RPV sample against benign and against P/LP, the
    RPV thresholds and the out-of-bag masks (cached in rpv_oob_matrices.npz)."""
    import contextlib
    import io
    from mv_analysis.production_oob import OUT as OOB_DIR
    from mv_analysis.tp53_rpv_report import TP53_RPV_ROLE
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    cache = Path(config.PAPER_TP53_RPV_DIR) / "rpv_oob_matrices.npz"
    mv = np.load(OOB_DIR / "tp53__TP53" / "mv.npz")
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        if np.array_equal(c["oob"], mv["oob"]):
            return {k: c[k] for k in c.files}
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_multiscoresets_for_gene_set("tp53")["TP53"]
        an = build_gene_set_analysis(ms, "tp53", config.PAPER_RESULTS_JSON, dataset_name="TP53_tp53_mv",
                                     auxiliary_pathogenic_indices=[TP53_RPV_ROLE])
        an.configs = [config_name]
        an.run(partial_pattern_mode="trust_global", path_percentile=5, min_valid_boots=1,
               reestimate_marginal_weights=False, enforce_marginal_monotonicity=False,
               liberal_marginal_monotonicity=False, aux_path_percentile=50, aux_ben_percentile=50)
    r = an.results[config_name]
    ar = r["aux_results"][TP53_RPV_ROLE]
    # the out-of-bag masks are per fit of the same run: its fits must be these
    assert np.allclose(np.asarray(r["lr_matrix"], float), mv["L"], equal_nan=True), "fits differ from production_oob's"
    d = dict(variants=np.asarray(ms._variants_kept, dtype=object), vs_b=np.asarray(ar["lr_matrix"], float),
             vs_p=np.asarray(ar["vsp_results"]["lr_matrix"], float), tau_p=np.asarray(ar["tau_p_log"], float),
             tau_b=np.asarray(ar["tau_b_log"], float), point_values=np.asarray(an.point_values),
             oob=mv["oob"], in_sample=np.asarray(ms._sample_assignments, bool).any(axis=1))
    np.savez_compressed(cache, **d)
    return d


def _classify(d, use, rule="points", min_vs_b=1):
    """score_rpv_penetrance's rule on the fits marked in use (fits x variants).
    rule (the split of variants with pathogenic evidence against benign):
      points     RPV-like when the RPV-vs-P/LP evidence is >= +1 point, else
                 P/LP-like (the paper's rule)
      zero       RPV-like when the median RPV-vs-P/LP log LR is > 0, else P/LP-like
      symmetric  RPV-like at >= +1 point, P/LP-like at <= -1 point, otherwise
                 intermediate
    min_vs_b: points against the benign sample required for P/LP-like or
    RPV-like (the paper uses 1); variants with pathogenic evidence below it
    are intermediate."""
    def med(m):
        x = np.where(use, m, np.nan)
        with np.errstate(all="ignore"):
            return np.nanmedian(x, axis=0) if use.any() else np.full(m.shape[1], np.nan)

    def pts(x):
        p = np.zeros(len(x), int)
        for pv in d["point_values"]:
            p[x >= d["tau_p"][pv - 1]] = pv
        for pv in d["point_values"]:
            p[x <= d["tau_b"][pv - 1]] = -pv
        return p
    b, p = med(d["vs_b"]), med(d["vs_p"])
    rp, vp = pts(b), pts(p)
    rp = np.where((rp >= 1) & (rp < min_vs_b), 0, rp)  # below the vs-benign bar: intermediate
    c = np.full(len(b), "intermediate", dtype=object)
    c[rp <= -1] = "benign_like"
    if rule == "points":
        c[(rp >= 1) & (vp <= 0)] = "plp_like"
        c[(rp >= 1) & (vp > 0)] = "low_pen_rpv"
    elif rule == "zero":
        c[(rp >= 1) & ~(p > 0)] = "plp_like"
        c[(rp >= 1) & (p > 0)] = "low_pen_rpv"
    elif rule == "symmetric":
        c[(rp >= 1) & (vp <= -1)] = "plp_like"
        c[(rp >= 1) & (vp >= 1)] = "low_pen_rpv"
    else:
        raise ValueError(rule)
    c[np.isnan(b)] = "never_out_of_bag"
    return c


def oob_calls(config_name, rule="points", min_vs_b=1):
    """DataFrame Variant, call (out of bag), call_all_fits."""
    d = _oob_matrices(config_name)
    use = np.where(d["in_sample"][None, :], d["oob"], True)
    return pd.DataFrame(dict(Variant=d["variants"], call=_classify(d, use, rule, min_vs_b),
                             call_all_fits=_classify(d, np.ones_like(d["oob"]), rule, min_vs_b)))


def _rpv_sets():
    rpvs, _ = tp53.load_reference_cohorts(tp53.DEFAULT_COHORTS_PATH)
    train = set(rpvs.loc[rpvs["confidence"] != "low", "Variant"])
    return train, set(rpvs.loc[rpvs["confidence"] == "low", "Variant"]) - train


def _germline():
    g = pd.read_csv(GERMLINE_DB, low_memory=False)
    g = g[g["Effect"] == "missense"].assign(Variant=lambda d: d["ProtDescription"].str.replace("p.", "", regex=False))
    fam = g.drop_duplicates(["Variant", "Family_ID"])
    per = fam.groupby("Variant").agg(n_families=("Family_ID", "size"),
                                     share_lfs=("Class", lambda s: (s == "LFS").mean()),
                                     share_fh_or_none=("Class", lambda s: s.isin(["FH", "noFH"]).mean()))
    per = per.join(g.groupby("Variant")["Age_at_diagnosis"].median().rename("median_age"))
    per = per.join(g.drop_duplicates("Variant").set_index("Variant")["TCGA_ICGC_GENIE_count"].rename("somatic_count"))
    return g, fam, per


def fetch_clinvar(cache):
    """ClinVar TP53 records matching CLINVAR_QUERIES (cached; [] when offline)."""
    if cache.exists():
        return json.loads(cache.read_text())

    def get(u, **p):
        time.sleep(1.0)  # E-utilities allow 3 requests/s without a key
        return json.load(urllib.request.urlopen(EUTILS + u + "?" + urllib.parse.urlencode(p), timeout=60))
    try:
        ids = set()
        for q in CLINVAR_QUERIES:
            ids |= set(get("esearch.fcgi", db="clinvar", term=q, retmax=500, retmode="json")["esearchresult"]["idlist"])
        s = get("esummary.fcgi", db="clinvar", id=",".join(sorted(ids)), retmode="json")["result"]
    except Exception as e:  # offline: the check is skipped, not faked
        print(f"  [rpv-validation] ClinVar fetch failed ({e}); skipped")
        return []
    out = [dict(uid=i, title=s[i].get("title", ""),
                classification=s[i].get("germline_classification", {}).get("description"),
                protein_change=s[i].get("protein_change", ""), fetched=time.strftime("%Y-%m-%d"))
           for i in s["uids"]]
    cache.write_text(json.dumps(out, indent=1))
    return out


def _mwu(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(stats.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else float("nan")


def _by_call(o, g, col):
    rows = []
    for k, label in CALLS:
        s = o[o[col] == k]
        f = s[s["n_families"] > 0]
        t = g[g["Variant"].isin(s["Variant"])]["Age_at_diagnosis"].dropna()
        rows.append(dict(call=label, variants=len(s), variants_in_germline_db=len(f),
                         families=int(f["n_families"].sum()),
                         pct_families_lfs=100 * np.average(f["share_lfs"], weights=f["n_families"]) if len(f) else np.nan,
                         pct_families_fh_or_none=100 * np.average(f["share_fh_or_none"], weights=f["n_families"]) if len(f) else np.nan,
                         tumours_with_age=len(t), median_age_tumours=t.median(),
                         median_age_variants=s["median_age"].median(),
                         median_somatic_count=s["somatic_count"].median(),
                         pct_in_gnomad=100 * (s["gnomADv4_AC"].fillna(0) > 0).mean(),
                         clinvar_plp=int((s["ClinVar_category"] == "P/LP").sum()),
                         clinvar_blb=int((s["ClinVar_category"] == "B/LB").sum())))
    return pd.DataFrame(rows)


def _compare(o, col, prefix=""):
    """RPV-like vs P/LP-like among variants with carriers in the germline database."""
    L = o[(o[col] == "low_pen_rpv") & (o["n_families"] > 0)]
    P = o[(o[col] == "plp_like") & (o["n_families"] > 0)]
    w = lambda x: float(100 * np.average(x["share_lfs"], weights=x["n_families"])) if len(x) else float("nan")
    f = dict(lowpen_variants_in_db=len(L), plp_variants_in_db=len(P),
             lowpen_families=int(L["n_families"].sum()), plp_families=int(P["n_families"].sum()),
             lowpen_pct_lfs=w(L), plp_pct_lfs=w(P), p_lfs=_mwu(L["share_lfs"], P["share_lfs"]),
             lowpen_age=float(L["median_age"].median()), plp_age=float(P["median_age"].median()),
             lowpen_age_n=int(L["median_age"].notna().sum()), plp_age_n=int(P["median_age"].notna().sum()),
             p_age=_mwu(L["median_age"], P["median_age"]),
             lowpen_somatic=float(L["somatic_count"].median()), plp_somatic=float(P["somatic_count"].median()),
             p_somatic=_mwu(L["somatic_count"], P["somatic_count"]))
    return {prefix + k: x for k, x in f.items()}


def run(config_name=None):
    config_name = config_name or config.FIXED_MV_CONFIGS[("TP53", "tp53")]
    out = Path(config.PAPER_TP53_RPV_DIR)
    calls = _calls(config_name)
    oc = oob_calls(config_name)
    calls = calls.merge(oc, on="Variant", how="left")
    assert (calls["call_all_fits"] == calls["rpv_class"]).all(), "all-fit calls do not reproduce the stored calls"
    train, heldout = _rpv_sets()
    g, fam, per = _germline()
    ann = pd.read_csv(config.TP53_ANNOTATED_VARIANTS_PATH, usecols=["Variant", "ClinVar_category", "gnomADv4_AC"],
                      low_memory=False)
    v = calls[["Variant", "call", "rpv_class", "rpv_points", "vsp_points"]].merge(ann, on="Variant", how="left")
    v = v.join(per, on="Variant")
    v["set"] = np.where(v["Variant"].isin(train), "RPV list",
                        np.where(v["Variant"].isin(heldout), "held-out RPV", "other"))
    v.to_csv(out / "rpv_validation_variants.csv", index=False)
    o = v[(v["set"] == "other") & (v["call"] != "never_out_of_bag")]

    by_call = _by_call(o, g, "call")
    by_call.to_csv(out / "rpv_validation_by_call.csv", index=False)
    _by_call(v[v["set"] == "other"], g, "rpv_class").to_csv(out / "rpv_validation_by_call_insample.csv", index=False)
    r = by_call.set_index("call")
    facts = dict(config=config_name, n_rpv_list=int((v["set"] == "RPV list").sum()),
                 n_heldout=int((v["set"] == "held-out RPV").sum()),
                 n_never_oob=int((v["call"] == "never_out_of_bag").sum()),
                 n_changed_oob=int(((v["call"] != v["rpv_class"]) & (v["call"] != "never_out_of_bag")).sum()),
                 lowpen_pct_gnomad=float(r.loc["RPV-like", "pct_in_gnomad"]),
                 plp_pct_gnomad=float(r.loc["P/LP-like", "pct_in_gnomad"]))
    facts.update(_compare(o, "call"))
    facts.update(_compare(v[v["set"] == "other"], "rpv_class", "insample_"))
    # stricter: among ClinVar P/LP variants only, does the RPV-like call mark milder families?
    facts.update(_compare(o[o["ClinVar_category"] == "P/LP"], "call", "plponly_"))
    h = v[v["set"] == "held-out RPV"]
    h[["Variant", "call", "rpv_class", "rpv_points", "vsp_points", "n_families", "share_lfs"]].to_csv(
        out / "rpv_validation_heldout.csv", index=False)
    facts["heldout_counts"] = {lab: int((h["call"] == k).sum()) for k, lab in CALLS}

    cv = pd.DataFrame(fetch_clinvar(out / "clinvar_tp53_low_penetrance.json"))
    if len(cv):
        cv["Variant"] = cv["title"].str.extract(r"\(p\.([A-Z][a-z]{2}\d+[A-Z][a-z]{2})\)")[0].map(_three_to_one)
        cv = cv.merge(v[["Variant", "call", "set"]], on="Variant", how="left")
        cv.to_csv(out / "rpv_validation_clinvar.csv", index=False)
        m = cv.dropna(subset=["call"])
        facts["clinvar_records_missense"] = len(m)
        facts["clinvar_in_rpv_list"] = int(m["set"].isin(["RPV list", "held-out RPV"]).sum())
        facts["clinvar_outside"] = m.loc[m["set"] == "other", ["Variant", "call"]].values.tolist()
    # deeper validation (A: penetrance vs P/LP-like; B: signal vs benign-like)
    t, kmt, w, kmd = deep_validation(v, g, "call")
    t.to_csv(out / "rpv_validation_metrics.csv", index=False)
    kmt.to_csv(out / "rpv_validation_incidence.csv", index=False)
    # non-proband carriers for the incidence curves, labelled by call (or the RPV list)
    lab = w["call"].where(w["set"] != "RPV list", "rpv_list")
    kmd.assign(call=kmd["Variant"].map(lab)).to_csv(out / "rpv_validation_km_carriers.csv", index=False)
    deep_validation(v, g, "rpv_class")[0].to_csv(out / "rpv_validation_metrics_insample.csv", index=False)
    plp = v.set_index("Variant")["ClinVar_category"].eq("P/LP")
    deep_validation(v, g, "call", restrict=plp)[0].to_csv(out / "rpv_validation_metrics_clinvar_plp.csv", index=False)
    burden = case_control_burden(pd.read_csv(fetch_tp53db() / "GermlineFrequencyDownload_r21.csv"),
                                 w.reset_index(), "call")
    burden.to_csv(out / "rpv_validation_burden.csv", index=False)
    rl = t[t["group"] == "RPV-like"].set_index("key")
    il = t[t["group"] == "Intermediate"].set_index("key")
    for key in rl.index:
        g3 = t[t["key"] == key].set_index("group")["estimate"]
        facts[f"m_{key}"] = {"plp": float(g3["P/LP-like"]), "rpv": float(g3["RPV-like"]),
                             "benign": float(g3["Benign-like"]),
                             "p_vs_plp": float(rl.loc[key, "p_rpv_vs_plp"]),
                             "p_vs_plp_holm": float(rl.loc[key, "p_rpv_vs_plp_holm"]),
                             "p_vs_benign": float(rl.loc[key, "p_rpv_vs_benign"]),
                             "p_vs_benign_holm": float(rl.loc[key, "p_rpv_vs_benign_holm"]),
                             "rpv_variants": int(rl.loc[key, "variants"]),
                             "intermediate": float(g3["Intermediate"]),
                             "int_variants": int(il.loc[key, "variants"]),
                             "p_int_vs_benign_holm": float(il.loc[key, "p_int_vs_benign_holm"]),
                             "p_int_vs_rpv_holm": float(il.loc[key, "p_int_vs_rpv_holm"])}
    b = burden.set_index("group")
    facts["burden"] = {k: dict(or_=float(b.loc[k, "mh_or"]), lo=float(b.loc[k, "ci_lo"]), hi=float(b.loc[k, "ci_hi"]),
                               case=int(b.loc[k, "case_carriers"]), control=int(b.loc[k, "control_carriers"]))
                       for k in ("P/LP-like", "RPV-like", "Benign-like")}
    k = kmt.set_index("group")
    facts["incidence_by_50"] = {g_: float(k.loc[g_, "incidence_by_50"]) for g_ in ("P/LP-like", "RPV-like", "Benign-like")}
    facts["incidence_carriers"] = {g_: int(k.loc[g_, "carriers"]) for g_ in ("P/LP-like", "RPV-like", "Benign-like")}
    fc = fortuno_concordance(v)
    if fc is not None:
        fc.to_csv(out / "rpv_validation_fortuno.csv", index=False)
        sc = fc.dropna(subset=["call"])
        facts["fortuno"] = {"listed": len(fc), "scored": len(sc),
                            **{lab: int((sc["call"] == k).sum()) for k, lab in CALLS}}
        vc = v[(v["set"] == "other") & v["ClinVar_category"].isin(["VUS", "Conflicting"]) & (v["call"] == "low_pen_rpv")]
        facts["fortuno"]["our_rpv_like_vus"] = len(vc)
        facts["fortuno"]["our_rpv_like_vus_listed"] = int(vc["Variant"].isin(fc["Variant"]).sum())
        facts["fortuno"].update(fortuno_two_way(v, fc, w))
    st = tumour_strata(w)
    st.to_csv(out / "rpv_validation_tumour_strata.csv", index=False)
    facts["strata"] = {f"{r_.stratum}_{r_.metric}": {"plp": r_.plp, "rpv": r_.rpv, "benign": r_.benign,
                                                     "p": r_.p_rpv_vs_benign, "n_rpv": r_.n_rpv, "n_benign": r_.n_benign}
                       for r_ in st.itertuples()}
    (out / "rpv_validation_facts.json").write_text(json.dumps(facts, indent=1))
    print(by_call.round(1).to_string(index=False))
    print(t.round(4).to_string(index=False))
    print(burden.round(3).to_string(index=False))
    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in facts.items()}, indent=1))
    return facts


# ── deeper validation: penetrance (A) and pathogenic signal (B) ─────────────
#
# Pre-specified metrics, all reported. (A) RPV-like vs P/LP-like: lower
# penetrance; (B) RPV-like vs benign-like: some pathogenic signal. Proportions
# are pooled over a class's variants (sum of numerators / sum of
# denominators); CIs resample variants within a class (cluster bootstrap) and
# p-values permute class labels across the two classes' variants, so a
# variant's many carriers or tumours count as one unit. Holm within each
# contrast (RPV-like vs P/LP-like; RPV-like vs benign-like) over all metrics.
# Functional columns of the database (TransactivationClass, DNEclass, *nWT)
# are not used: they are, or derive from, the model's own inputs.

TP53DB_DIR = Path("/data/ross/assay_calibration/TP53/tp53db_r21")
TP53DB_URL = "https://tp53.cancer.gov/static/data/"
TP53DB_FILES = ("GermlineFrequencyDownload_r21.csv", "MutationView_r21.csv", "TumorVariantDownload_r21.csv",
                "GermlineRefDownload_r21.csv")
N_BOOT, N_PERM = 2000, 5000
CORE_LFS_MORPH = ("adrenal cortical", "choroid plexus", "sarcoma")
CORE_LFS_TOPO = ("BRAIN", "SOFT TISSUES", "BONES (other)", "BONES (limbs)", "ADRENAL GLAND")


def fetch_tp53db():
    """Download the NCI TP53 Database r21 files once (TP53DB_DIR)."""
    TP53DB_DIR.mkdir(parents=True, exist_ok=True)
    for f in TP53DB_FILES:
        if not (TP53DB_DIR / f).exists():
            urllib.request.urlretrieve(TP53DB_URL + f, TP53DB_DIR / f)
    return TP53DB_DIR


def _prot(s):
    return s.str.replace("p.", "", regex=False)


def _pooled(num, den, labels, cls, rng, n_boot=N_BOOT):
    """Pooled proportion of class cls and its variant-cluster bootstrap CI."""
    m = labels == cls
    n, d = num[m], den[m]
    est = n.sum() / d.sum() if d.sum() > 0 else np.nan
    if m.sum() < 2 or d.sum() == 0:
        return est, np.nan, np.nan
    idx = rng.integers(0, m.sum(), size=(n_boot, m.sum()))
    with np.errstate(all="ignore"):
        bs = n[idx].sum(1) / d[idx].sum(1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi


def _perm_p(num, den, labels, a, b, rng, n_perm=N_PERM):
    """Two-sided permutation p for the difference in pooled proportions of
    classes a and b, permuting labels across the two classes' variants."""
    m = np.isin(labels, [a, b]) & (den > 0)
    n, d, lab = num[m], den[m], labels[m] == a
    if lab.sum() < 2 or (~lab).sum() < 2:
        return np.nan
    obs = n[lab].sum() / d[lab].sum() - n[~lab].sum() / d[~lab].sum()
    perm = np.array([rng.permutation(lab) for _ in range(n_perm)])
    with np.errstate(all="ignore"):
        diff = (perm * n).sum(1) / (perm * d).sum(1) - ((~perm) * n).sum(1) / ((~perm) * d).sum(1)
    return float((np.sum(np.abs(diff) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))


def _germline_units(g):
    """Per-variant numerators/denominators from the germline family data."""
    ind = g.dropna(subset=["Individual_ID"]).copy()
    first_age = ind.groupby("Individual_ID")["Age_at_diagnosis"].min()
    n_tum = ind.dropna(subset=["Tumor_ID"]).groupby("Individual_ID")["Tumor_ID"].nunique()
    i = ind.drop_duplicates("Individual_ID").set_index("Individual_ID")
    i = i.assign(first_age=first_age, n_tumours=n_tum.reindex(i.index).fillna(0))
    carrier = i["Germline_carrier"].isin(["Confirmed", "Obligatory"])
    rel = i["FamilyCase_group"].notna() & (i["FamilyCase_group"] != "proband")
    relatives = i[carrier & rel & i["Unaffected"].notna()]
    affected = i[i["Unaffected"] == False]  # noqa: E712
    fam = g.drop_duplicates(["Variant", "Family_ID"])
    tum = g.dropna(subset=["Tumor_ID"]).drop_duplicates(["Individual_ID", "Tumor_ID"])
    morph = tum["Morphology"].fillna("").str.lower()
    core = (morph.str.contains("|".join(CORE_LFS_MORPH)) | tum["Short_topo"].isin(CORE_LFS_TOPO)
            | ((tum["Short_topo"] == "BREAST") & (tum["Age_at_diagnosis"] < 50)))
    mode = fam.dropna(subset=["Mode_of_inheritance"])
    u = pd.DataFrame({
        "lfs_n": fam.groupby("Variant")["Class"].apply(lambda s: (s == "LFS").sum()),
        "lfs_d": fam.groupby("Variant").size(),
        "relaff_n": relatives.groupby("Variant")["Unaffected"].apply(lambda s: (s == False).sum()),  # noqa: E712
        "relaff_d": relatives.groupby("Variant").size(),
        "multi_n": affected.groupby("Variant")["n_tumours"].apply(lambda s: (s >= 2).sum()),
        "multi_d": affected.groupby("Variant").size(),
        "denovo_n": mode.groupby("Variant")["Mode_of_inheritance"].apply(lambda s: s.str.contains("de novo").sum()),
        "denovo_d": mode.groupby("Variant").size(),
        "child_n": tum.dropna(subset=["Age_at_diagnosis"]).groupby("Variant")["Age_at_diagnosis"].apply(lambda s: (s < 18).sum()),
        "child_d": tum.dropna(subset=["Age_at_diagnosis"]).groupby("Variant").size(),
        "core_n": core.groupby(tum["Variant"]).sum(),
        "core_d": tum.groupby("Variant").size(),
    })
    # relatives for the age-specific incidence: event = first cancer, censored at current age
    km = relatives.assign(time=np.where(relatives["Unaffected"] == False, relatives["first_age"],  # noqa: E712
                                        relatives["Age"]),
                          event=(relatives["Unaffected"] == False).astype(int)).dropna(subset=["time"])  # noqa: E712
    return u.fillna(0), km[["Variant", "time", "event"]]


def _somatic_units(tv, mv):
    """Per-variant mutability-adjusted somatic count and second-hit rate."""
    m = mv[mv["Effect"] == "missense"].assign(Variant=lambda d: _prot(d["ProtDescription"]))
    m = m.dropna(subset=["Mut_rateAA"]).drop_duplicates("Variant").set_index("Variant")
    t = tv[(tv["Effect"] == "missense") & (tv["Exclude_analysis"] != True)  # noqa: E712
           & (tv["Sample_source"] != "cell-line")].assign(Variant=lambda d: _prot(d["ProtDescription"]))
    n_tp53 = tv.groupby("Sample_ID").size()
    s = t.drop_duplicates(["Variant", "Sample_ID"]).assign(multi=lambda d: d["Sample_ID"].map(n_tp53) >= 2)
    lfs_tissue = s["Short_topo"].isin(CORE_LFS_TOPO + ("BREAST",))
    return pd.DataFrame({
        "som_n": m["Somatic_count"], "som_d": m["Mut_rateAA"],
        "hit2_n": s.groupby("Variant")["multi"].sum(), "hit2_d": s.groupby("Variant").size(),
        "lfstis_n": lfs_tissue.groupby(s["Variant"]).sum(), "lfstis_d": s.groupby("Variant").size(),
        "polymorphism": mv.assign(Variant=lambda d: _prot(d["ProtDescription"])).groupby("Variant")["Polymorphism"]
        .apply(lambda x: (x == "validated").any()),
    })


def _km_incidence(km, ages=(30, 50, 70)):
    """Kaplan-Meier cumulative incidence at the given ages."""
    out = {}
    if km.empty:
        return {a: np.nan for a in ages}
    t, e = km["time"].to_numpy(float), km["event"].to_numpy(int)
    times = np.sort(np.unique(t[e == 1]))
    surv, s = {}, 1.0
    for x in times:
        at_risk = (t >= x).sum()
        s *= 1 - ((t == x) & (e == 1)).sum() / at_risk
        surv[x] = s
    for a in ages:
        prior = [v for x, v in surv.items() if x <= a]
        out[a] = 1 - (prior[-1] if prior else 1.0)
    return out


def case_control_burden(freq, v, col):
    """Mantel-Haenszel OR of carrying a class's variant, cases vs controls,
    over the GermlineFrequency studies (validated polymorphisms excluded from
    the classes, reported as their own row)."""
    from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
    f = freq.assign(Variant=_prot(freq["ProtDescription"]))
    # classes over the variants outside the reference and held-out RPV lists, as in every other test
    lab = (v[v["set"] == "other"] if "set" in v.columns else v).set_index("Variant")[col]
    rows = []
    groups = [(lab_, f["Variant"].map(lab).eq(k) & ~f["Variant"].isin(v.loc[v["polymorphism"], "Variant"]))
              for k, lab_ in CALLS] + [("Validated polymorphisms", f["Variant"].isin(v.loc[v["polymorphism"], "Variant"]))]
    for label, sel in groups:
        tables = []
        for (nc, nn), s in f[sel].groupby(["Total_Cases", "Total_Controls"]):
            a, b = int(s["N_Cases"].sum()), int(s["N_Controls"].sum())
            tables.append(np.array([[a, nc - a], [b, nn - b]], float))
        a = sum(t[0, 0] for t in tables) if tables else 0
        b = sum(t[1, 0] for t in tables) if tables else 0
        if tables and a + b > 0:
            st = StratifiedTable([t + 0.5 if (t == 0).any() else t for t in tables])
            orr, (lo, hi) = st.oddsratio_pooled, st.oddsratio_pooled_confint()
            pooled = np.sum(tables, axis=0)
            p = float(stats.fisher_exact(pooled)[1])
        else:
            orr = lo = hi = p = np.nan
        rows.append(dict(group=label, variants=int(f.loc[sel, "Variant"].nunique()), case_carriers=a,
                         control_carriers=b, mh_or=orr, ci_lo=lo, ci_hi=hi, fisher_p_pooled=p))
    return pd.DataFrame(rows)


FORTUNO_S3 = Path("/data/ross/assay_calibration/TP53/fortuno2025_hggadv_mmc2.xlsx")


def fortuno_concordance(v, col="call"):
    """Calls for the 106 ClinVar VUS/conflicting variants that Fortuno et al.
    2025 (HGG Adv, Table S3; random forest on Kato, Giacomelli, Kotler, Funk,
    BayesDel, AlphaMissense, aGVGD, immune fitness and allele frequency)
    predict as reduced penetrance. Shares functional inputs with the joint
    model, so agreement is consistency, not independent validation."""
    import re
    if not FORTUNO_S3.exists():
        return None
    d = pd.read_excel(FORTUNO_S3, sheet_name="Supplemental Table 3", header=1)
    aa = {k: x for k, x in _AA.items()}

    def one(p):  # p.G262S or p.(Gly262Ser) -> G262S
        p = str(p).strip()
        m = re.match(r"(?:p\.)?\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$", p)
        if m:
            return aa.get(m.group(1), "?") + m.group(2) + aa.get(m.group(3), "?")
        m = re.match(r"(?:p\.)?\(?([A-Z]\d+[A-Z])\)?$", p)
        return m.group(1) if m else None
    d["Variant"] = d.iloc[:, 1].map(one)
    lab = v.set_index("Variant")
    d["call"] = d["Variant"].map(lab[col])
    d["set"] = d["Variant"].map(lab["set"])
    return d[["Variant", "call", "set"]]


def fortuno_two_way(v, fc, w, seed=0):
    """Agreement in both directions on the closest analogue of the variants
    Fortuno et al. scored: ClinVar VUS/conflicting missense variants outside
    the RPV list with a Kotler or Funk score (their 590). Also tests whether
    the variants RPV-like here but not predicted by them differ in tumour
    measures (w: deep_validation's per-variant table) from those both call."""
    a = pd.read_csv(config.TP53_ANNOTATED_VARIANTS_PATH, usecols=["Variant", "Kotler_RFS", "Funk_RFS"],
                    low_memory=False).drop_duplicates("Variant").set_index("Variant")
    has = a["Kotler_RFS"].notna() | a["Funk_RFS"].notna()
    u = v[(v["set"] == "other") & v["ClinVar_category"].isin(["VUS", "Conflicting"])
          & v["Variant"].map(has).fillna(False).astype(bool) & (v["call"] != "never_out_of_bag")]
    ours, theirs = u["call"] == "low_pen_rpv", u["Variant"].isin(set(fc["Variant"].dropna()))
    out = {"universe": len(u), "theirs": int(theirs.sum()), "ours": int(ours.sum()),
           "both": int((ours & theirs).sum()), "ours_only": int((ours & ~theirs).sum()),
           "theirs_only": int((~ours & theirs).sum())}
    rng = np.random.default_rng(seed)
    grp = pd.Series(np.where(theirs, "both", "ours_only"), index=u["Variant"])[ours.to_numpy()]
    x = w.loc[w.index.intersection(grp.index)]
    x = x[~x["polymorphism"]]
    lab = grp.reindex(x.index).to_numpy()
    for m in ("som", "hit2"):
        n, d = x[m + "_n"].fillna(0).to_numpy(float), x[m + "_d"].fillna(0).to_numpy(float)
        out[f"{m}_both"] = float(_pooled(n, d, lab, "both", rng)[0])
        out[f"{m}_ours_only"] = float(_pooled(n, d, lab, "ours_only", rng)[0])
        out[f"p_{m}_both_vs_ours_only"] = _perm_p(n, d, lab, "both", "ours_only", rng)
    return out


DBD = (94, 292)


def tumour_strata(w, seed=0):
    """Tumour measures (somatic count per mutability, second TP53 mutation),
    RPV-like vs benign-like, within strata: the DNA-binding domain, its ClinVar
    VUS/conflicting variants, its variants without a ClinVar record, and the
    rest of the protein. Tests whether the contrast is carried by domain or by
    ClinVar status, both of which track tumour counts."""
    rng = np.random.default_rng(seed)
    o = w[(w["set"] == "other") & (w["call"] != "never_out_of_bag") & ~w["polymorphism"]]
    pos = pd.Series(o.index.str.extract(r"(\d+)")[0].astype(float).to_numpy(), index=o.index)
    dbd = pos.between(*DBD)
    strata = {"dbd": dbd, "dbd_vus": dbd & o["ClinVar_category"].isin(["VUS", "Conflicting"]),
              "dbd_noclinvar": dbd & o["ClinVar_category"].isna(), "outside": ~dbd}
    rows = []
    for name, m in strata.items():
        x = o[m]
        lab = x["call"].to_numpy()
        for metric in ("som", "hit2"):
            n, d = x[metric + "_n"].fillna(0).to_numpy(float), x[metric + "_d"].fillna(0).to_numpy(float)
            est = {c: _pooled(n, d, lab, c, rng)[0] for c in ("plp_like", "low_pen_rpv", "benign_like")}
            rows.append(dict(stratum=name, metric=metric, plp=float(est["plp_like"]), rpv=float(est["low_pen_rpv"]),
                             benign=float(est["benign_like"]),
                             p_rpv_vs_benign=_perm_p(n, d, lab, "low_pen_rpv", "benign_like", rng),
                             n_rpv=int(((lab == "low_pen_rpv") & (d > 0)).sum()),
                             n_benign=int(((lab == "benign_like") & (d > 0)).sum())))
    return pd.DataFrame(rows)


def deep_validation(v, g, col="call", restrict=None, seed=0):
    """Metric table (A and B) for the call column col over variants outside
    the RPV list (restrict: optional extra mask)."""
    rng = np.random.default_rng(seed)
    rng_int = np.random.default_rng(seed + 1)
    gu, km = _germline_units(g)
    db = fetch_tp53db()
    su = _somatic_units(pd.read_csv(db / "TumorVariantDownload_r21.csv", low_memory=False),
                        pd.read_csv(db / "MutationView_r21.csv", low_memory=False))
    w = v.set_index("Variant").join(gu).join(su.drop(columns="polymorphism"))
    w["polymorphism"] = su["polymorphism"].reindex(w.index).fillna(False).astype(bool)
    o = w[(w["set"] == "other") & (w[col] != "never_out_of_bag") & ~w["polymorphism"]]
    if restrict is not None:
        o = o[restrict.reindex(o.index).fillna(False)]
    refs = {"Known RPVs (reference list)": w[w["set"] == "RPV list"],
            "Validated polymorphisms": w[w["polymorphism"]]}
    metrics = (("A", "lfs", "Families meeting classic LFS criteria"),
               ("A", "relaff", "Non-proband carriers affected"),
               ("A", "multi", "Affected carriers with multiple primaries"),
               ("A", "denovo", "Families with a de novo variant"),
               ("A", "child", "Tumours diagnosed before age 18"),
               ("A", "core", "Tumours of the core LFS spectrum"),
               ("B", "som", "Somatic count per unit mutability"),
               ("B", "hit2", "Tumours with a second TP53 mutation"),
               ("B", "lfstis", "Somatic occurrences in LFS-spectrum tissues"))
    rows = []
    labels = o[col].to_numpy()
    for q, key, name in metrics:
        num, den = o[f"{key}_n"].fillna(0).to_numpy(float), o[f"{key}_d"].fillna(0).to_numpy(float)
        p_vs_plp = _perm_p(num, den, labels, "low_pen_rpv", "plp_like", rng)
        p_vs_ben = _perm_p(num, den, labels, "low_pen_rpv", "benign_like", rng)
        # intermediate against benign-like and RPV-like, on its own stream so
        # the RPV-like contrasts above draw the same permutations as before
        p_int_ben = _perm_p(num, den, labels, "intermediate", "benign_like", rng_int)
        p_int_rpv = _perm_p(num, den, labels, "intermediate", "low_pen_rpv", rng_int)
        for k, lab in CALLS:
            est, lo, hi = _pooled(num, den, labels, k, rng)
            m = (labels == k) & (den > 0)
            rows.append(dict(question=q, metric=name, key=key, group=lab, estimate=est, ci_lo=lo, ci_hi=hi,
                             variants=int(m.sum()), denominator=float(den[labels == k].sum()),
                             p_rpv_vs_plp=p_vs_plp if k == "low_pen_rpv" else np.nan,
                             p_rpv_vs_benign=p_vs_ben if k == "low_pen_rpv" else np.nan,
                             p_int_vs_benign=p_int_ben if k == "intermediate" else np.nan,
                             p_int_vs_rpv=p_int_rpv if k == "intermediate" else np.nan))
        for lab, r in refs.items():
            n_, d_ = r[f"{key}_n"].fillna(0).sum(), r[f"{key}_d"].fillna(0).sum()
            rows.append(dict(question=q, metric=name, key=key, group=lab, estimate=n_ / d_ if d_ else np.nan,
                             variants=int((r[f"{key}_d"].fillna(0) > 0).sum()), denominator=float(d_)))
    t = pd.DataFrame(rows)
    # Holm within each contrast, over all nine metrics (both contrasts are read
    # for every metric, not only for the question it was planned under)
    for pcol in ("p_rpv_vs_plp", "p_rpv_vs_benign", "p_int_vs_benign", "p_int_vs_rpv"):
        m = t[pcol].notna()
        ps = t.loc[m, pcol].to_numpy()
        order = np.argsort(ps)
        adj = np.empty_like(ps)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
            adj[i] = running
        t.loc[m, pcol + "_holm"] = adj
    # age-specific incidence among non-proband carriers (KM), per class and reference
    kmr = []
    keep = set(o.index)
    for k, lab in CALLS:
        vv = [x for x in o.index[o[col] == k]]
        kmr.append(dict(group=lab, carriers=int(km["Variant"].isin(vv).sum()),
                        **{f"incidence_by_{a}": x for a, x in _km_incidence(km[km["Variant"].isin(vv)]).items()}))
    for lab, r in refs.items():
        kmr.append(dict(group=lab, carriers=int(km["Variant"].isin(r.index).sum()),
                        **{f"incidence_by_{a}": x for a, x in _km_incidence(km[km["Variant"].isin(r.index)]).items()}))
    return t, pd.DataFrame(kmr), w, km[km["Variant"].isin(keep | set(refs["Known RPVs (reference list)"].index))]


_AA = dict(Ala="A", Arg="R", Asn="N", Asp="D", Cys="C", Gln="Q", Glu="E", Gly="G", His="H", Ile="I", Leu="L",
           Lys="K", Met="M", Phe="F", Pro="P", Ser="S", Thr="T", Trp="W", Tyr="Y", Val="V")


def _three_to_one(p):
    if not isinstance(p, str):
        return None
    return _AA.get(p[:3], "?") + p[3:-3] + _AA.get(p[-3:], "?")


if __name__ == "__main__":
    run()
