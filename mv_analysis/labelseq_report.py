#!/usr/bin/env python3
"""LABEL-seq per-variant evidence, Brnich regions, assay attribution and
sankeys -- for the LABEL-seq manuscript, NOT the ExCALIBR-MV paper (LABEL-seq
is held back from it; nothing here writes into the paper's outputs).

Composes the multivariate pipeline's own pieces:
  production_oob._mv_one             out-of-bag joint-model points, NU genes included
  variant_tables.model_columns       points, LR+ percentiles, prior, posterior, ExCALIBR points
  review_assay_contribution.score_masked + review_decomposition.shapley
                                     what each readout contributes (readouts
                                     marginalized out of the stored fits)
  brnich_regions                     Brnich evidence from synonymous z-score regions

Stages (python -m mv_analysis.labelseq_report <stage> ...):
  oob          out-of-bag joint-model fits per bootstrap split, all 17 genes
               (exact splits from the run's job pickles, per condition)
  attribution  every readout subset of every gene, scored on all measured
               variants; per-variant Shapley terms and leave-one-out changes
               of the combined log LR+
  table        labelseq_variant_table.tsv.gz (one row per protein variant),
               labelseq_input_with_evidence.tsv.gz (the LABEL-seq input with
               those columns), brnich_readouts.csv, checks.json, README.md
  validation   brnich_validation.csv: full-readout Brnich counts and OddsPaths
               against the earlier per-gene LABELseq_brnich.csv
  sankeys      sankeys/<scope>/... and fig_attribution_<scope>.png, scopes
               non_nu (11 genes with P/LP controls) and all (17 genes)
  all          everything in order

Run settings (environment, defaults = the archived tau 0.1 final run, whose
LABEL-seq fits used ClinVar 2026 controls):
  MV_LABELSEQ_FITS_DIR        merged fits (bootstrap_results.json.gz, conditions.json)
  MV_LABELSEQ_PKNN_DIR        P-KNN prep of that run (model rows, configs)
  MV_LABELSEQ_OUTPUT_DIR      where everything is written
  MV_LABELSEQ_CLINVAR_RELEASE the fits' ClinVar release: control labels and the VUS definition
They are set here, before mv_analysis.config is imported, so this must run as
its own process (mv_cockpit's labelseq-report shells out to it).
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LABELSEQ_FITS_DIR = os.environ.get(
    "MV_LABELSEQ_FITS_DIR", "/data/ross/assay_calibration/multivariate/jobs_final_tau_100b_3f_101826/final")
LABELSEQ_PKNN_DIR = os.environ.get("MV_LABELSEQ_PKNN_DIR", "/data/ross/assay_calibration/multivariate/pknn_tau01_102626")
LABELSEQ_OUTPUT_DIR = Path(os.environ.get("MV_LABELSEQ_OUTPUT_DIR", "/data/ross/assay_calibration/labelseq/mv_outputs"))
LABELSEQ_CLINVAR_RELEASE = os.environ.get("MV_LABELSEQ_CLINVAR_RELEASE", "2026")

if "mv_analysis.config" in sys.modules:
    raise RuntimeError("mv_analysis.labelseq_report sets the run's paths before mv_analysis.config is imported; "
                       "run it as its own process (python -m mv_analysis.labelseq_report ...)")
os.environ.update({
    "MV_PAPER_FITS_DIR": LABELSEQ_FITS_DIR,
    "MV_PAPER_RESULTS_JSON": f"{LABELSEQ_FITS_DIR}/bootstrap_results.json.gz",
    "MV_PKNN_DIR": LABELSEQ_PKNN_DIR,
    "MV_PAPER_EXCLUDE_GENE_SETS": "",
    "MV_PAPER_FIGURES_DIR": str(LABELSEQ_OUTPUT_DIR / "pipeline"),
    "MV_REVIEW_DIR": str(LABELSEQ_OUTPUT_DIR / "review"),
    "MV_PRODUCTION_OOB_DIR": str(LABELSEQ_OUTPUT_DIR / "production_oob"),
    "MV_CLINVAR_DEFAULT_RELEASE": LABELSEQ_CLINVAR_RELEASE,
})
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import functools  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mv_analysis import config  # noqa: E402

ARCHIVE_FIGURES = Path(LABELSEQ_FITS_DIR) / "figures"  # the full-panel paper outputs (with LABEL-seq), archived
ARCHIVE_OOB = Path(LABELSEQ_PKNN_DIR) / "production_oob"
OLD_BRNICH_CSV = "/data/ross/assay_calibration/labelseq/LABELseq_brnich.csv"
ATTR_DIR = LABELSEQ_OUTPUT_DIR / "attribution"
WIDE = LABELSEQ_OUTPUT_DIR / "labelseq_variant_table.tsv.gz"
LONG = LABELSEQ_OUTPUT_DIR / "labelseq_input_with_evidence.tsv.gz"
ROWS_TAG = "allrows"  # score_masked cache tag: the model's rows plus every other measured variant

# Brnich settings: (abnormal rule, controls) -> column stem; main first. DMSO
# abundance is always the untreated abundance in genes without one.
BRNICH_SETTINGS = [("all", "brnich"), ("missense", "brnich_missensecontrols")]  # (controls, column prefix)
VUS_COL = f"is_vus_{LABELSEQ_CLINVAR_RELEASE}"  # VUS = the fits' ClinVar release
BRNICH_MAIN = "brnich"  # per-readout Brnich, strongest readout per gene, controls of all variant types


# ---------------------------------------------------------------- models

@functools.lru_cache(maxsize=1)
def labelseq_models():
    """[(key, prep meta)] for every LABEL-seq model in the run's P-KNN prep."""
    out = []
    for p in sorted((Path(config.PKNN_DIR) / "prep").glob("labelseq__*.json")):
        m = json.load(open(p))
        out.append((m["key"], m))
    return out


def is_nu(meta):
    """Negative-unlabelled model: no P/LP controls."""
    return int(meta["eval_P"]) == 0


def _split_fits_dir(meta):
    """The condition directory a model's fits came from (conditions.json
    dataset_source), whose job pickles hold its exact bootstrap splits."""
    f = Path(LABELSEQ_FITS_DIR) / "conditions.json"
    src = json.load(open(f)).get("dataset_source", {}).get(meta["dataset_name"]) if f.exists() else None
    if src is None:
        return None
    p = Path(src)
    return str(p if p.is_absolute() else Path(LABELSEQ_FITS_DIR).parent / src)


# ---------------------------------------------------------------- stage: oob

def stage_oob(workers=8):
    """production_oob._mv_one for every LABEL-seq model (NU included), into
    this report's own production_oob dir, with each model's exact bootstrap
    splits from its condition's job pickles. The models the paper run already
    evaluated must have the archived per-fit log LR+, priors and seeds; their
    out-of-bag masks are compared and reported (the paper run regenerated its
    splits, which does not reproduce RET's: the pickles' validation rows
    match the model's rows exactly, the regenerated ones do not)."""
    from mv_analysis import production_oob as PO
    PO.OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(k, _split_fits_dir(m)) for k, m in labelseq_models() if not (PO.OUT / k / "mv.npz").exists()]
    print(f"[oob] {len(tasks)} models to fit-score", flush=True)
    if tasks:
        PO._pool(PO._mv_one, tasks, workers, "mv")
    report = {}
    for k, m in labelseq_models():
        f = PO.OUT / k / "mv.npz"
        if not f.exists():
            raise RuntimeError(f"[oob] {k}: no mv.npz")
        a = np.load(f)
        old = ARCHIVE_OOB / k / "mv.npz"
        if old.exists():
            b = np.load(old)
            same = (np.array_equal(a["seeds"], b["seeds"]) and np.allclose(a["L"], b["L"], equal_nan=True)
                    and np.allclose(a["priors"], b["priors"]))
            if not same:
                raise AssertionError(f"[oob] {k}: per-fit log LR+ / priors / seeds differ from the archived mv.npz")
            report[k] = "identical to archived" if np.array_equal(a["oob"], b["oob"]) else \
                f"same fits; out-of-bag masks differ from archived in {(a['oob'] != b['oob']).mean():.1%} of entries " \
                "(archived splits regenerated; these from the job pickles)"
        else:
            report[k] = f"new ({int(a['oob'].shape[0])} splits; NU)" if is_nu(m) else "new"
    for k, v in report.items():
        print(f"  {k}: {v}", flush=True)
    (LABELSEQ_OUTPUT_DIR / "oob_check.json").write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------- stage: attribution

@functools.lru_cache(maxsize=2)
def _scored_ms(key):
    """The model's MultiScoreset plus every other measured variant (the rows
    variant_tables.model_columns scores, in the same order)."""
    from mv_analysis import variant_tables as V
    from mv_analysis.comparison_methods import pknn_runs as RP
    d, m = RP.load_prep(key)
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                              results_json=config.PAPER_RESULTS_JSON)
        assert [str(v) for v in ms.kept_variants] == list(d["variants"]), f"{key}: model rows differ from the prep"
        ids_x, X_x = V._extra_rows(m, ms)
    return V._ExtendedMS(ms, ids_x, X_x) if len(ids_x) else ms


def _combined_log_lr(L):
    from analysis.all_variant_evidence import _combine_directional_log_lr
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return _combine_directional_log_lr(np.nanpercentile(L, 5, axis=0), np.nanpercentile(L, 95, axis=0))


def _subset_job(key, S, masked):
    from mv_analysis import review_assay_contribution as A
    warnings.filterwarnings("ignore")
    try:
        L, _, _ = A.score_masked(key, masked, ms=_scored_ms(key), rows_tag=ROWS_TAG)
        return key, S, _combined_log_lr(L)
    except Exception as e:  # noqa: BLE001 -- reported, never silently dropped
        return key, S, f"FAILED {type(e).__name__}: {e}"


def stage_attribution(workers=12):
    """Per model and readout subset S (readouts outside S marginalized out of
    the stored fits): v(S) = the combined log LR+ (5th-percentile bound
    floored at 0 or 95th capped at 0, whichever is farther from 0) of every
    scored variant; v({}) = 0. Exact Shapley terms (sum to v(all)) and
    leave-one-out changes v(all) - v(all but r) -> attribution/<key>.npz."""
    from mv_analysis import review_decomposition as D
    ATTR_DIR.mkdir(parents=True, exist_ok=True)
    # warm the all-measured-variant MultiScoreset cache once (workers would race to build it)
    with contextlib.redirect_stdout(io.StringIO()):
        config.build_multiscoresets_for_gene_set("labelseq", genes=[m["gene"] for _, m in labelseq_models()],
                                                 regularization_type="all_assayed")
    spec = {}
    for k, m in labelseq_models():
        if (ATTR_DIR / f"{k}.npz").exists():
            continue
        names = [str(n) for n in np.load(Path(config.PKNN_DIR) / "prep" / f"{k}.npz")["dims"]]
        spec[k] = names
    jobs = [(k, S, tuple(n for i, n in enumerate(names) if not S >> i & 1))
            for k, names in spec.items() for S in range(1, 1 << len(names))]
    jobs.sort(key=lambda j: (-len(spec[j[0]]), j[0], j[1]))
    print(f"[attribution] {len(spec)} models, {len(jobs)} readout subsets", flush=True)
    res = {k: {} for k in spec}
    failed = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_subset_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            k, S, v = f.result()
            if isinstance(v, str):
                failed.append((k, S, v))
                print(f"  {k} subset {S}: {v}", flush=True)
            else:
                res[k][S] = v
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} subsets", flush=True)
    if failed:
        raise RuntimeError(f"[attribution] {len(failed)} subsets failed")
    for k, names in spec.items():
        n, full = len(names), (1 << len(names)) - 1
        nr = len(res[k][full])
        v = np.zeros((1 << n, nr))
        for S in range(1, 1 << n):
            v[S] = np.nan_to_num(res[k][S])  # no readout of S measured: no evidence
        phi = D.shapley(v, n)
        assert np.allclose(phi.sum(0), v[full]), f"{k}: Shapley terms do not sum to the combined log LR+"
        loo = np.stack([v[full] - v[full & ~(1 << i)] for i in range(n)])
        np.savez_compressed(ATTR_DIR / f"{k}.npz", readouts=np.asarray(names), v_full=res[k][full], phi=phi, loo=loo,
                            variants=np.asarray([str(x) for x in _scored_ms(k).kept_variants]))
        print(f"  {k}: {n} readouts, {nr} variants -> {ATTR_DIR / (k + '.npz')}", flush=True)


# ---------------------------------------------------------------- stage: table

def _consequence(pkey):
    s = str(pkey)
    if s.endswith("="):
        return "synonymous"
    if "Ter" in s or s.endswith("*"):
        return "nonsense"
    return "missense"


def _dicts(names, M, decimals=4):
    """JSON dict per row {name: value} over the non-missing entries of M (rows x names)."""
    out = []
    for row in np.asarray(M, float):
        d = {n: round(float(x), decimals) for n, x in zip(names, row) if np.isfinite(x)}
        out.append(json.dumps(d) if d else None)
    return out


def _labels(df_in):
    """Per protein variant (input rows aggregated): ClinVar significance and
    review status for 2025 and 2026 (distinct values joined by ' | '), VUS in
    each release (any of its nucleotide variants), the full mapped_hgvs_p and
    the input's consequence."""
    from mv_analysis.variant_tables import labelseq_protein_key
    d = df_in.assign(protein_variant=df_in["hgvs_p"].map(labelseq_protein_key))

    def j(s):
        v = sorted({str(x) for x in s.dropna()})
        return " | ".join(v) if v else None
    agg = {c: j for c in ("clinvar_sig_2025", "clinvar_star_2025", "clinvar_sig_2026", "clinvar_star_2026",
                          "simplified_consequence") if c in d.columns}
    agg["hgvs_p"] = "first"
    out = d.groupby("protein_variant").agg(agg).rename(columns={"hgvs_p": "mapped_hgvs_p",
                                                                 "simplified_consequence": "input_consequence"})
    # VUS: any of the protein variant's nucleotide variants "Uncertain significance"
    for rel in ("2025", "2026"):
        out[f"is_vus_{rel}"] = d.assign(v=d[f"clinvar_sig_{rel}"] == "Uncertain significance") \
            .groupby("protein_variant")["v"].any()
    return out


def _gene_base(key, meta):
    """One model's rows without the Brnich columns, plus what Brnich needs."""
    from mv_analysis import brnich_regions as B
    from mv_analysis import variant_tables as V
    df, st = V.model_columns(key, meta, extra=True)
    ms, keep = st["ms"], st["keep"]
    names = [B.readout_name(n) for n in ms.dataset_names]
    X = np.asarray(ms.scores, float)[keep]
    pk = [V.labelseq_protein_key(v) for v in df.index]

    def sample(s):  # a model without that sample (NU: no P/LP) has no such column
        c = f"sample_{s}"
        return df[c].to_numpy(bool) if c in df.columns else np.zeros(len(df), bool)
    syn, plp, blb = sample("synonymous"), sample("PLP"), sample("BLB")
    out = pd.DataFrame({"gene": meta["gene"].upper(), "is_nu": is_nu(meta), "protein_variant": pk})
    out["consequence"] = [_consequence(p) for p in pk]
    out["fit_clinvar_release"] = LABELSEQ_CLINVAR_RELEASE
    out["fit_control_label"] = np.where(plp, "P/LP", np.where(blb, "B/LB", None))
    for s in ("PLP", "BLB", "gnomAD", "synonymous"):
        out[f"sample_{s}"] = sample(s)
    out["mv_fitted"] = df["fitted"].to_numpy(bool)
    out["scores"] = _dicts(names, X)
    out["syn_z"] = _dicts(names, B.synonymous_z(X, syn))
    uv = {c.split("__", 1)[1]: df[c].to_numpy(float) for c in df.columns if c.startswith("excalibr_points__")}
    uv_names = [B.readout_name(n) for n in uv]
    out["excalibr_points"] = _dicts(uv_names, np.column_stack(list(uv.values())) if uv else np.empty((len(df), 0)), 0)
    out["excalibr_combined_points"] = df["excalibr_nonconflicting_points"].to_numpy(float) \
        if "excalibr_nonconflicting_points" in df.columns else np.nan
    out["mv_points"] = df["points"].to_numpy(float)
    out["mv_points_median_lr"] = df["points_median_lr"].to_numpy(float)
    out["mv_points_oob"] = df["points_out_of_bag"].to_numpy(float)
    out["mv_n_oob_fits"] = df["n_out_of_bag_fits"].to_numpy(float)
    out["mv_evidence_code"] = df["evidence_code"].to_numpy()
    for p in ("p5", "p50", "p95"):
        out[f"mv_log_lr_{p}"] = df[f"log_lr_plus_{p}"].to_numpy(float)
    out["mv_log_lr_combined"] = df["log_lr_combined"].to_numpy(float)
    out["mv_lr_plus"] = df["lr_plus"].to_numpy(float)
    out["mv_prior_median"] = df["prior"].to_numpy(float)
    out["mv_posterior"] = df["posterior"].to_numpy(float)
    # readout attribution (stage_attribution), aligned on the scored rows
    a = np.load(ATTR_DIR / f"{key}.npz")
    assert list(a["variants"]) == [str(v) for v in ms.kept_variants], f"{key}: attribution rows differ"
    assert [B.readout_name(n) for n in a["readouts"]] == names, f"{key}: attribution readouts differ"
    vf = np.asarray(a["v_full"], float)[keep]
    assert np.allclose(vf, out["mv_log_lr_combined"], equal_nan=True), \
        f"{key}: attribution's full-model log LR+ differs from the table's"
    phi, loo = np.asarray(a["phi"])[:, keep].T, np.asarray(a["loo"])[:, keep].T
    tot = np.abs(phi).sum(1, keepdims=True)
    share = np.where(tot > 0, np.abs(phi) / np.where(tot > 0, tot, 1), np.nan)
    measured = ~np.isnan(X)
    out["mv_shapley"] = _dicts(names, np.where(measured, phi, np.nan))
    out["mv_shapley_share"] = _dicts(names, np.where(measured, share, np.nan))
    out["mv_loo"] = _dicts(names, np.where(measured, loo, np.nan))
    return out, dict(X=X, names=names, syn=syn, plp=plp, blb=blb)


def _base_cache_file(key, meta):
    """Cache file for _gene_base, keyed by everything it reads: the code
    (variant_tables, production_oob, _gene_base itself), the prep, the
    out-of-bag fits, the attribution, the stored fits and the gene's
    univariate calibrations. Any change gives a new file name."""
    import hashlib
    import inspect
    from mv_analysis import production_oob as PO
    from mv_analysis import variant_tables as V
    h = hashlib.md5(inspect.getsource(_gene_base).encode())
    for mod in (V, PO):
        h.update(Path(mod.__file__).read_bytes())
    files = [Path(config.PKNN_DIR) / "prep" / f"{key}{ext}" for ext in (".npz", ".json")] + \
        [PO.OUT / key / "mv.npz", ATTR_DIR / f"{key}.npz", Path(config.PAPER_RESULTS_JSON)] + \
        sorted(Path(config.LABELSEQ_UV_CALIB_DIR).glob(f"{meta['gene'].lower()}_*/*"))
    for f in files:
        st = f.stat()
        h.update(f"{f}|{st.st_size}|{st.st_mtime}".encode())
    return LABELSEQ_OUTPUT_DIR / "cache" / "table" / f"{key}_{h.hexdigest()}.pkl"


def gene_rows(key, meta):
    """The wide table's rows for one model (one row per scored protein
    variant): the cached base columns plus every Brnich setting."""
    import pickle
    from mv_analysis import brnich_regions as B
    f = _base_cache_file(key, meta)
    if f.exists():
        out, st = pickle.loads(f.read_bytes())
    else:
        out, st = _gene_base(key, meta)
        f.parent.mkdir(parents=True, exist_ok=True)
        for old in f.parent.glob(f"{key}_*.pkl"):
            old.unlink()
        tmp = f.with_name(f.name + f".tmp{os.getpid()}")
        tmp.write_bytes(pickle.dumps((out, st)))
        os.replace(tmp, f)
    out = out.copy()
    X, names, syn, plp, blb = st["X"], st["names"], st["syn"], st["plp"], st["blb"]
    # Brnich evidence, every setting
    mis = np.array([c == "missense" for c in out["consequence"]])
    pats = []
    for ctrl, pre in BRNICH_SETTINGS:
        keep = mis if ctrl == "missense" else True  # missense: both control classes missense only
        ev, pt = B.per_readout_brnich(X, names, syn, plp & keep, blb & keep, pseudocount="if_zero")
        for c in ("points", "code", "readout", "bin", "oddspath", "readout_points"):
            out[f"{pre}_{c}"] = ev[c].to_numpy()
        pats.append(pt.assign(gene=meta["gene"].upper(), controls=ctrl))
    return out, pd.concat(pats, ignore_index=True)


README = """# LABEL-seq per-variant functional evidence

## Important Notes

1. **ClinVar release.** The controls (P/LP, B/LB) the models were calibrated
   on are from ClinVar {rel}, and the VUS in the sankeys are ClinVar {rel} VUS
   (`is_vus_{rel}`). Both the 2025 and 2026 ClinVar labels are in the table.
2. **Genes without P/LP controls** (ARAF, ERBB2, GRB2, KSR1, KSR2, SOS2;
   `is_nu`): We can still assign evidence; however, positive points should be
   interpreted as functional abnormality rather than pathogenicity.

## Files

- `labelseq_variant_table.tsv.gz`: one row per gene and protein variant.
- `labelseq_input_with_evidence.tsv.gz`: the LABEL-seq input (one row per
  variant, assay and treatment) with the columns below appended.
- `brnich_readouts.csv`: per gene, controls setting and readout: control
  counts in each bin, OddsPaths, the points each bin gives, and the readout's
  rank.
- `archive_multireadout_brnich_112326/`: the earlier multi-readout Brnich
  (its columns, sankeys and comparison with `LABELseq_brnich.csv`).
- `sankeys/<scope>/<source>/<sample>_<source>_to_mv.png`: flows from another
  method to ExCALIBR-MV, each with its flows as CSV; all figures summarised in
  `sankeys/summary.csv`.
  - scope: `non_nu` (the 11 genes with P/LP controls) or `all` (17 genes).
  - source: `excalibr` or `brnich`.
  - sample: `vus` (ClinVar {rel} VUS), `plp`, `blb`, `blb_nonsyn` (B/LB
    without synonymous variants).
  - P/LP and B/LB use in-sample ExCALIBR-MV points (each control helped fit
    the model), as do ExCALIBR and Brnich. `<source>_oob/` repeats the
    control figures with out-of-bag ExCALIBR-MV points; ExCALIBR and Brnich
    stay in-sample there (no out-of-bag version of them exists yet).
- `fig_attribution_*.png`: per gene, each readout's mean signed share of the
  ExCALIBR-MV evidence (right of 0: toward pathogenic; left: toward benign),
  for VUS with pathogenic and with benign evidence and for P/LP and B/LB
  controls with evidence in their own direction. Readouts ordered by how many
  genes have them; activity in reds, abundance in blues (KRAS's extra
  abundance readouts in purples), interaction in green.
- `figures_non_nu/<source>_to_mv_panels.png`: ExCALIBR or main Brnich ->
  ExCALIBR-MV (in-sample) for the 11 genes with P/LP controls; panels a P/LP,
  b B/LB, c VUS.
- `checks.json`, `oob_check.json`: internal consistency checks.

## Columns

Points: positive = pathogenic (PS3), negative = benign (BS3); Supporting 1,
Moderate 2, Strong 4, Very Strong 8. Dictionary columns are JSON keyed by
readout (e.g. `abundance_HSP90i`), measured readouts only. Logs are natural
logs.

| column | meaning |
|---|---|
| `gene`, `protein_variant`, `consequence` | gene; protein change (e.g. `p.Ala334=`); synonymous / missense / nonsense |
| `mv_points` | ExCALIBR-MV points: 5th percentile of log LR+ over 100 bootstrap fits against the pathogenic thresholds, 95th percentile against the benign thresholds (Tavtigian et al.), at the gene's prior |
| `mv_points_median_lr` | the same from the median LR+ |
| `mv_points_oob`, `mv_n_oob_fits` | the same from only the fits that left the variant out, and how many; differs from `mv_points` only for controls |
| `mv_evidence_code` | PS3 / BS3 strength of `mv_points` |
| `excalibr_combined_points` | ExCALIBR, readouts calibrated separately and combined: the strongest when they agree in direction, 0 when they conflict |
| `excalibr_points` | {{readout: ExCALIBR points}} |
| `mv_log_lr_p5`, `mv_log_lr_p50`, `mv_log_lr_p95` | 5th percentile, median and 95th percentile of log LR+ over the fits |
| `mv_log_lr_combined` | the log LR+ `mv_points` is based on: the 5th percentile if above 0, the 95th if below 0, otherwise 0 |
| `mv_lr_plus` | `mv_log_lr_combined` as an LR+ (nudged by at most 0.002 on the log scale for the few variants exactly at a threshold, to agree with `mv_points`) |
| `mv_prior_median` | the gene's prior probability of pathogenicity (median over the fits) |
| `mv_posterior` | posterior from `mv_lr_plus` and `mv_prior_median` |
| `mv_shapley_share` | {{readout: share}}: abs(contribution) / sum of abs(contributions) |
| `mv_shapley` | {{readout: contribution}} to `mv_log_lr_combined` (+ toward pathogenic, - toward benign), averaged over all combinations of the other readouts, which are removed from the fitted model (no refit); sums to `mv_log_lr_combined` |
| `mv_loo` | {{readout: change}}: `mv_log_lr_combined` minus its value without that readout |
| `brnich_points`, `brnich_code` | Brnich et al. (2020) evidence from the gene's strongest readout the variant was measured in (`Not measured`: in no usable readout; `Not computable`: the gene has no P/LP controls). Each readout on its own: z-scored against the gene's synonymous variants; normal abs(z) <= 2, abnormal abs(z) >= 3, otherwise intermediate (no evidence); an OddsPath for each bin from the controls measured in that readout, +1 pseudocount only to a zero count (one misclassified variant per class when necessary, as the IGVF Pillar Project); at most Supporting with 10 or fewer controls; abnormal gives only pathogenic and normal only benign evidence. Readouts are ranked per gene by the evidence their OddsPaths reach (readouts with 11 or more controls first, then abnormal points + abs(normal points), then the number of variants the readout gives evidence to, then the OddsPaths), following Calhoun et al. (2026); all readouts are eligible (`brnich_readouts.csv`) |
| `brnich_readout`, `brnich_bin`, `brnich_oddspath` | the readout used, the variant's bin in it and that bin's OddsPath |
| `brnich_readout_points` | {{readout: points}} from every usable readout the variant was measured in |
| `brnich_missensecontrols_...` | the same with P/LP and B/LB controls restricted to missense variants (its own readout ranking) |
| `is_nu` | gene has no P/LP controls |
| `fit_clinvar_release` | ClinVar release of the fits' controls |
| `fit_control_label` | P/LP or B/LB if the variant was a control in the fits |
| `is_vus_2025`, `is_vus_2026` | a nucleotide variant of the protein variant is ClinVar "Uncertain significance" in that release |
| `clinvar_sig_2025`, `clinvar_star_2025`, `clinvar_sig_2026`, `clinvar_star_2026` | ClinVar significance and review status (different values across the variant's nucleotide variants joined by ` | `) |
| `sample_PLP`, `sample_BLB`, `sample_gnomAD`, `sample_synonymous` | the groups of variants ExCALIBR-MV was fitted to |
| `mv_fitted` | in at least one of those groups |
| `mapped_hgvs_p`, `input_consequence` | protein change with its RefSeq protein; consequence as given in the input |
| `scores` | {{readout: score}} as used in the fits (activity on a log10 scale) |
| `syn_z` | {{readout: z-score}} against the gene's synonymous variants |
"""


def column_order(cols):
    """The tables' column order: identity, points, LR+ / prior / posterior,
    readout contributions, Brnich (main setting first), then labels, samples
    and data. Columns not listed keep their order at the end."""
    brnich = [f"{pre}_{c}" for _, pre in BRNICH_SETTINGS
              for c in ("points", "code", "readout", "bin", "oddspath", "readout_points")]
    order = (["gene", "protein_variant", "consequence"]
             + ["mv_points", "mv_points_median_lr", "mv_points_oob", "mv_n_oob_fits", "mv_evidence_code",
                "excalibr_combined_points", "excalibr_points"]
             + ["mv_log_lr_p5", "mv_log_lr_p50", "mv_log_lr_p95", "mv_log_lr_combined", "mv_lr_plus",
                "mv_prior_median", "mv_posterior"]
             + ["mv_shapley_share", "mv_shapley", "mv_loo"]
             + brnich
             + ["is_nu", "fit_clinvar_release", "fit_control_label", "is_vus_2025", "is_vus_2026", "clinvar_sig_2025", "clinvar_star_2025", "clinvar_sig_2026",
                "clinvar_star_2026", "sample_PLP", "sample_BLB", "sample_gnomAD",
                "sample_synonymous", "mv_fitted", "mapped_hgvs_p", "input_consequence", "scores", "syn_z"])
    cols = list(cols)
    return [c for c in order if c in cols] + [c for c in cols if c not in order]


def _gene_rows_job(key, meta):
    warnings.filterwarnings("ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        return key, gene_rows(key, meta)


def _labelseq_input_cached():
    """build_labelseq_dataframe, cached on disk by its source file and the
    code that builds it (not the rest of labelseq.py)."""
    import hashlib
    import inspect
    import pickle
    from src.assay_calibration.multivariate_data import labelseq as LS
    src = Path(LS.DEFAULT_DATA_PATH)
    st = src.stat()
    h = hashlib.md5(f"{src}|{st.st_size}|{st.st_mtime}|fill={LS.fill_annotation_default()}".encode())
    for fn in (LS.build_labelseq_dataframe, LS.fill_nucleotide_annotation, LS.fill_annotation_default,
               LS._parse_first_hgvs_c):
        h.update(inspect.getsource(fn).encode())
    h.update(repr((LS.VEP_CONSEQUENCE_MAPPING, LS._NUCLEOTIDE_ANNOTATION_PREFIXES,
                   LS._NUCLEOTIDE_ANNOTATION_COLUMNS)).encode())
    f = LABELSEQ_OUTPUT_DIR / "cache" / f"labelseq_input_{h.hexdigest()}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    with contextlib.redirect_stdout(io.StringIO()):
        df = LS.build_labelseq_dataframe()
    f.parent.mkdir(parents=True, exist_ok=True)
    for old in f.parent.glob("labelseq_input_*.pkl"):
        old.unlink()
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(df))
    os.replace(tmp, f)
    return df


GZIP = {"method": "gzip", "compresslevel": 6}  # near level 9's size, about twice as fast
WIDE_PKL = LABELSEQ_OUTPUT_DIR / "cache" / "wide.pkl"  # the table stage's wide table, for write_long_table
LONG_STAMP = LABELSEQ_OUTPUT_DIR / "cache" / "long_from_wide.md5"  # which wide table the input table was built from


def write_long_table():
    """labelseq_input_with_evidence.tsv.gz: the LABEL-seq input with the wide
    table's per-variant columns appended (the input keeps its own gene,
    ClinVar and consequence columns). Built by the bundle stage from the last
    table stage, so column changes don't rewrite these 1.4M rows."""
    from mv_analysis import variant_tables as V
    V._labelseq_input = functools.lru_cache(maxsize=1)(_labelseq_input_cached)
    wide = pd.read_pickle(WIDE_PKL)
    longs = []
    for key, m in labelseq_models():
        r = wide[wide["gene"] == m["gene"].upper()]
        df_in, _, pkey = V.input_table(m["gene"], {"labelseq"})
        drop = list(dict.fromkeys(["gene", "mapped_hgvs_p", "input_consequence"] + \
            [c for c in r.columns if c in df_in.columns and c != "protein_variant"]))
        longs.append(df_in.assign(protein_variant=pkey.to_numpy())
                     .merge(r.drop(columns=drop), on="protein_variant", how="left", validate="many_to_one"))
        assert len(longs[-1]) == len(df_in), f"{key}: the input-joined table changed the input's row count"
    pd.concat(longs, ignore_index=True).to_csv(LONG, sep="\t", index=False, compression=GZIP)
    LONG_STAMP.write_text(_wide_digest())
    print(f"[bundle] input-joined -> {LONG}", flush=True)


def _wide_digest():
    import hashlib
    return hashlib.md5(WIDE_PKL.read_bytes()).hexdigest()


def stage_table(workers=6):
    from mv_analysis import variant_tables as V
    warnings.filterwarnings("ignore")
    V._labelseq_input = functools.lru_cache(maxsize=1)(_labelseq_input_cached)
    LABELSEQ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # per-model scoring in parallel (each re-scores ~10k variants with its stored fits)
    done = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_gene_rows_job, k, m) for k, m in labelseq_models()]
        for f in as_completed(futs):
            k, res = f.result()
            done[k] = res
            print(f"[table] {k}: {len(res[0]):,} variants scored ({len(done)}/{len(futs)})", flush=True)
    rows, pats, checks = [], [], {}
    for key, m in labelseq_models():
        r, p = done[key]
        df_in, _, _ = V.input_table(m["gene"], {"labelseq"})
        lab = _labels(df_in)
        r = r.join(lab, on="protein_variant")
        for rel in ("2025", "2026"):
            r[f"is_vus_{rel}"] = r[f"is_vus_{rel}"].fillna(False).astype(bool)
        r = r[column_order(r.columns)]
        rows.append(r)
        pats.append(p)
        checks[key] = _check_against_archive(key, m, r)
    wide = pd.concat(rows, ignore_index=True)
    wide = wide[column_order(wide.columns)]
    wide.to_csv(WIDE, sep="\t", index=False, compression=GZIP)
    WIDE_PKL.parent.mkdir(parents=True, exist_ok=True)
    wide.to_pickle(WIDE_PKL)
    pats = pd.concat(pats, ignore_index=True)
    pats[["gene", "controls"] + [c for c in pats.columns if c not in ("gene", "controls")]] \
        .to_csv(LABELSEQ_OUTPUT_DIR / "brnich_readouts.csv", index=False)
    print(f"[table] {len(wide):,} variants -> {WIDE}", flush=True)
    (LABELSEQ_OUTPUT_DIR / "checks.json").write_text(json.dumps(checks, indent=1))
    (LABELSEQ_OUTPUT_DIR / "README.md").write_text(_readme())
    bad = {k: v for k, v in checks.items() if any(isinstance(x, dict) and x.get("mismatch") for x in v.values())}
    print(f"[table] checks: {len(checks) - len(bad)}/{len(checks)} models agree with the archive"
          + (f"; MISMATCHES: {json.dumps(bad)}" if bad else ""), flush=True)
    return wide


def _check_against_archive(key, meta, r):
    """The archived full-panel outputs (strength_points.csv: in-sample joint
    and independent points of the controls; the VUS reclassification CSV)
    against this table."""
    from mv_analysis.variant_tables import labelseq_protein_key
    out = {}
    sp = pd.read_csv(ARCHIVE_FIGURES / "strength" / "strength_points.csv", low_memory=False)
    sp = sp[(sp["key"] == key) & sp["sample"].isin(["P/LP", "B/LB"])].copy()
    sp["protein_variant"] = sp["variant"].map(labelseq_protein_key)
    t = r.set_index("protein_variant")
    if len(sp):
        j = sp.join(t[["mv_points", "excalibr_combined_points"]], on="protein_variant")
        out["controls_mv_points"] = dict(n=len(j), mismatch=int((j["joint_points"] != j["mv_points"]).sum()))
        out["controls_excalibr_points"] = dict(n=len(j), mismatch=int(
            (j["indep_points"] != j["excalibr_combined_points"].fillna(0)).sum()))
    v = pd.read_csv(ARCHIVE_FIGURES / "vus_reclassification" / "functional_true_clinvar_vus_points.csv")
    v = v[(v["gene_set"] == "labelseq") & (v["gene"] == meta["gene"].lower())]
    if len(v):
        j = v.join(t[["mv_points", "excalibr_combined_points"]], on="variant_key", rsuffix="_t")
        out["vus_mv_points"] = dict(n=len(j), missing=int(j["mv_points"].isna().sum()),
                                    mismatch=int(((j["our_points"] != j["mv_points"]) & j["mv_points"].notna()).sum()))
        out["vus_excalibr_points"] = dict(n=len(j), mismatch=int(
            ((j["indep_points"] != j["excalibr_combined_points"].fillna(0)) & j["mv_points"].notna()).sum()))
    return out


# ---------------------------------------------------------------- stage: validation

def stage_validation():
    """ARCHIVED with the multi-readout Brnich (needs its brnich_readouts
    column; last output in archive_multireadout_brnich_112326/).
    Full-readout Brnich (every readout of the gene's main-setting set
    measured) per gene, both region rules and both control sets, with the
    earlier CSV's always-+1 pseudocount and the manuscript's +1-if-zero, next
    to that CSV's values."""
    from mv_analysis import brnich_regions as B
    from mv_analysis.mv_cockpit import BRNICH_TO_EXCALIBR
    wide = pd.read_csv(WIDE, sep="\t", low_memory=False)
    old = pd.read_csv(OLD_BRNICH_CSV)
    old["gene"] = old["gene"].replace(BRNICH_TO_EXCALIBR)
    old = old.set_index("gene")
    recs = []
    for gene, g in wide.groupby("gene"):
        z = g["syn_z"].map(lambda s: json.loads(s) if isinstance(s, str) else {})
        dims = sorted({r for s in g["brnich_readouts"].dropna() for r in s.split(";")})
        Z = np.array([[d.get(r, np.nan) for r in dims] for d in z], float)
        full = ~np.isnan(Z).any(1) if len(dims) else np.zeros(len(g), bool)
        plp, blb = g["sample_PLP"].to_numpy(bool), g["sample_BLB"].to_numpy(bool)
        mis = (g["consequence"] == "missense").to_numpy()
        for rule in ("all", "any"):
            reg = B.regions(Z[full], rule) if full.any() else np.array([], object)
            for ctrl in ("all", "missense"):
                keep = mis if ctrl == "missense" else np.ones(len(g), bool)
                p_, b = plp & keep, blb & keep
                for pc in ("always", "if_zero"):
                    st = B.pattern_oddspath(reg[(p_ | b)[full]], p_[full][(p_ | b)[full]], b[full][(p_ | b)[full]],
                                            int((reg == "abnormal").sum()), int((reg == "normal").sum()), pc)
                    rec = dict(gene=gene, readouts=";".join(dims), n_readouts=len(dims), rule=rule, controls=ctrl,
                               pseudocount=pc, **{k: st[k] for k in ("n_plp", "n_blb", "plp_normal", "blb_normal",
                                                                     "plp_abnormal", "blb_abnormal",
                                                                     "oddspath_abnormal", "oddspath_normal")})
                    if gene in old.index:
                        o = old.loc[gene]
                        sfx = "missense" if ctrl == "missense" else "all"
                        rec.update(old_mode=o["mode"], old_n_path=o["n_path"],
                                   old_n_benign=o["n_benign_missense" if ctrl == "missense" else "n_benign_all"],
                                   old_n_path_inside=o["n_path_inside"], old_n_path_outside=o["n_path_outside"],
                                   old_oddspath_abnormal=o[f"op_outside_{sfx}"], old_oddspath_normal=o[f"op_inside_{sfx}"],
                                   old_code_abnormal=o[f"evidence_outside_{sfx}"], old_code_normal=o[f"evidence_inside_{sfx}"])
                    recs.append(rec)
    out = pd.DataFrame(recs)
    out.to_csv(LABELSEQ_OUTPUT_DIR / "brnich_validation.csv", index=False)
    print(f"[validation] -> {LABELSEQ_OUTPUT_DIR / 'brnich_validation.csv'}", flush=True)
    return out


# ---------------------------------------------------------------- stage: sankeys

TIERS = [  # (label, lo, hi), most pathogenic first
    ("Very Strong (+8)", 8, 8), ("Strong (+4 to +7)", 4, 7), ("Moderate (+2, +3)", 2, 3), ("Supporting (+1)", 1, 1),
    ("No evidence (0)", 0, 0),
    ("Supporting (-1)", -1, -1), ("Moderate (-2, -3)", -3, -2), ("Strong (-4 to -7)", -7, -4), ("Very Strong (-8)", -8, -8)]
NOT_COMPUTABLE = "Not computable"
NOT_MEASURED = "Not measured"
SAMPLES = {  # name -> (label, row filter, controls?)
    "vus": ("VUS", lambda w: w[VUS_COL], False),
    "plp": ("P/LP", lambda w: w["sample_PLP"], True),
    "blb": ("B/LB", lambda w: w["sample_BLB"], True),
    "blb_nonsyn": ("B/LB non-synonymous", lambda w: w["sample_BLB"] & (w["consequence"] != "synonymous"), True),
}
SOURCES = {  # name -> (header, column, NaN means)
    "excalibr": ("ExCALIBR", "excalibr_combined_points", "no evidence"),
    "brnich": ("Brnich et al.", f"{BRNICH_MAIN}_points", NOT_COMPUTABLE),
}


def _tier_colors():
    from mv_analysis.sankey_plot_flat import STRENGTH_COLOR
    c = {t[0]: STRENGTH_COLOR.get(t[1], "#e0e0e0") for t in TIERS}
    c["No evidence (0)"] = "#e0e0e0"
    c[NOT_COMPUTABLE] = "#ffffff"
    c[NOT_MEASURED] = "#f4f4f4"
    return c


def to_tier(points, nan_as, nan_labels=None):
    """Tier label per value; a missing value is `nan_as` (or, with
    `nan_labels`, that row's label, e.g. a Brnich code "Not measured")."""
    p = pd.Series(points, dtype=float)
    out = pd.Series(nan_as if nan_as == NOT_COMPUTABLE else "No evidence (0)", index=p.index, dtype=object)
    if nan_labels is not None:
        lab = pd.Series(nan_labels, index=p.index)
        ok = p.isna() & lab.isin([NOT_COMPUTABLE, NOT_MEASURED])
        out[ok] = lab[ok]
    q = p.round().clip(-8, 8)
    for name, lo, hi in TIERS:
        out[(q >= lo) & (q <= hi)] = name
    return out


def _source_tier(sub, col, nan_as):
    """Tiers of a sankey source column; Brnich's missing values keep their
    code (Not computable / Not measured)."""
    code = col[: -len("_points")] + "_code" if col.endswith("_points") else None
    if nan_as == NOT_COMPUTABLE and code in sub.columns:
        return to_tier(sub[col], nan_as, sub[code])
    return to_tier(sub[col] if nan_as == NOT_COMPUTABLE else sub[col].fillna(0), nan_as)


def _draw_sankey(ax, src_tier, tgt_tier, left, right):
    """One sankey on `ax`; returns (flow counts, tier order)."""
    from mv_analysis.sankey_plot_flat import plot_categorical_sankey
    order = [t[0] for t in TIERS] + [NOT_MEASURED, NOT_COMPUTABLE]
    flow = (pd.DataFrame({"source": src_tier.to_numpy(), "target": tgt_tier.to_numpy()})
            .groupby(["source", "target"]).size().reset_index(name="count"))
    colors = _tier_colors()
    plot_categorical_sankey(flow, order, order, colors, colors, ax=ax, source_fontsize=9, target_fontsize=9,
                            show_target_counts=True, gap_frac=0.012, min_frac=0.015, gradient=True)
    for x, h in ((0.0, left), (1.0, right)):
        ax.text(x, -0.035, h, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11,
                fontweight="bold")
    return flow, order


def sankey_figure(src_tier, tgt_tier, left, right, out_stem, exts=("pdf", "png"), flows=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    flow, order = _draw_sankey(ax, src_tier, tgt_tier, left, right)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if flows:
        (flow.pivot_table(index="source", columns="target", values="count", fill_value=0)
         .reindex(index=order, columns=order, fill_value=0).rename_axis(index=left, columns=right)
         .to_csv(f"{out_stem}_flows.csv"))


FIG_DIR = LABELSEQ_OUTPUT_DIR / "figures_non_nu"
FIG_PANELS = (("plp", "P/LP"), ("blb", "B/LB"), ("vus", "VUS"))


def _fit_xlim_to_labels(fig, ax, pad=0.01):
    """Shrink the sankey's fixed x margins to the extent of its labels."""
    for _ in range(3):  # label widths in data units change with the limits
        fig.canvas.draw()
        r, inv = fig.canvas.get_renderer(), ax.transData.inverted()
        xs = [0.0, 1.0]
        for t in ax.texts:
            bb = t.get_window_extent(r)
            xs += [inv.transform((bb.x0, bb.y0))[0], inv.transform((bb.x1, bb.y1))[0]]
        ax.set_xlim(min(xs) - pad, max(xs) + pad)


def stage_figures():
    """figures_non_nu/<source>_to_mv_panels.png: ExCALIBR or main Brnich ->
    ExCALIBR-MV (in-sample) for the 11 genes with P/LP controls, panels a P/LP,
    b B/LB, c VUS; each panel letter at its panel's left edge, level with its
    title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wide = pd.read_csv(WIDE, sep="\t", low_memory=False)
    w = wide[~wide["is_nu"]]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("*.png"):
        old.unlink()
    for src in ("excalibr", "brnich"):
        left, col, nan_as = SOURCES[src]
        fig, axes = plt.subplots(1, len(FIG_PANELS), figsize=(6.5 * len(FIG_PANELS), 7.5))
        for ax, (smp, title) in zip(axes, FIG_PANELS):
            sub = w[SAMPLES[smp][1](w).astype(bool)]
            _draw_sankey(ax, _source_tier(sub, col, nan_as), to_tier(sub["mv_points"], "no evidence"), left,
                         "ExCALIBR-MV")
            ax.set_title(f"{title} (n = {len(sub):,})", fontsize=13, fontweight="bold", pad=10)
        for ax in axes:
            _fit_xlim_to_labels(fig, ax)
        fig.tight_layout(w_pad=1.0)
        fig.canvas.draw()
        r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
        for ax, letter in zip(axes, "abc"):
            x0 = ax.get_tightbbox(r).x0
            tb = ax.title.get_window_extent(r)
            fx, fy = inv.transform((x0, (tb.y0 + tb.y1) / 2))
            fig.text(fx, fy, letter, ha="left", va="center", fontsize=18, fontweight="bold")
        fig.savefig(FIG_DIR / f"{src}_to_mv_panels.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"[figures] -> {FIG_DIR}", flush=True)


def _summary(src, tgt):
    s, t = pd.Series(src, dtype=float), pd.Series(tgt, dtype=float)
    return dict(n=len(s), source_not_computable=int(s.isna().sum()),
                source_any=int((s.fillna(0) != 0).sum()), target_any=int((t != 0).sum()),
                source_pathogenic=int((s > 0).sum()), target_pathogenic=int((t > 0).sum()),
                source_benign=int((s < 0).sum()), target_benign=int((t < 0).sum()),
                source_moderate_plus=int((s.abs() >= 2).sum()), target_moderate_plus=int((t.abs() >= 2).sum()),
                gained=int(((s.fillna(0) == 0) & (t != 0)).sum()), lost=int(((s.fillna(0) != 0) & (t == 0)).sum()),
                direction_flip=int((s * t < 0).sum()))


def stage_sankeys(sources=None):
    """Every sankey (or only those of `sources`, keys of SOURCES; summary.csv
    keeps the other sources' rows) and, unless `sources` leaves it out, the
    attribution figures ("attribution")."""
    wide = pd.read_csv(WIDE, sep="\t", low_memory=False)
    out_dir = LABELSEQ_OUTPUT_DIR / "sankeys"
    run = dict(SOURCES) if not sources else {k: v for k, v in SOURCES.items() if k in sources}
    figures = not sources or "attribution" in sources
    summ = []
    for scope in ("non_nu", "all"):
        w = wide[~wide["is_nu"]] if scope == "non_nu" else wide
        for sname, (slabel, filt, controls) in SAMPLES.items():
            sub = w[filt(w).astype(bool)]
            for mv_col, mv_tag in (("mv_points", ""), ("mv_points_oob", "_oob")):
                if mv_tag and not controls:
                    continue  # VUS are never in a sample: out-of-bag = in-sample
                for src, (left, col, nan_as) in run.items():
                    stem = out_dir / scope / (f"{src}{mv_tag}") / f"{sname}_{src}_to_mv{mv_tag}"
                    s = sub[col] if nan_as == NOT_COMPUTABLE else sub[col].fillna(0)
                    sankey_figure(_source_tier(sub, col, nan_as), to_tier(sub[mv_col], "no evidence"), left,
                                  "ExCALIBR-MV" + (" (out-of-bag)" if mv_tag else ""), stem)
                    summ.append(dict(scope=scope, sample=slabel, source=left, target_points=mv_col,
                                     **_summary(s, sub[mv_col])))
        if figures:
            _attribution_figure(w, LABELSEQ_OUTPUT_DIR / f"fig_attribution_{scope}", readout_order(wide))
    summ = pd.DataFrame(summ)
    f = out_dir / "summary.csv"
    if sources and f.exists():
        keep = pd.read_csv(f)
        labels = {v[0] for v in run.values()}
        summ = pd.concat([keep[~keep["source"].isin(labels) & keep["source"].isin([v[0] for v in SOURCES.values()])],
                          summ], ignore_index=True)
    summ.to_csv(f, index=False)
    print(f"[sankeys] {len(summ)} figures -> {out_dir}", flush=True)


# Readout colours by family: activity reds, abundance blues (KRAS's extra
# abundance readouts purples), interaction greens; the untreated readout
# darkest, treatments lighter. Unlisted readouts get a shade of their family.
READOUT_COLORS = {"activity_No_treatment": "#cb181d", "activity_SerumStarve": "#fc9272",
                  "abundance_No_treatment": "#08519c", "abundance_HSP90i": "#4292c6", "abundance_DMSO": "#9ecae1",
                  "abundance_CIAR": "#54278f", "abundance_LZTR1ko": "#8c6bb1", "abundance_LZTR1koCIAR": "#bcbddc",
                  "interaction_LZTR1": "#31a354"}
_FAMILY_CMAPS = {"activity": "Reds", "abundance": "Blues", "interaction": "Greens"}


def readout_order(w):
    """Readouts by how many genes measure them, then by how many variants."""
    genes, variants = {}, {}
    for gene, v in zip(w["gene"], w["mv_shapley"]):
        if isinstance(v, str):
            for r in json.loads(v):
                genes.setdefault(r, set()).add(gene)
                variants[r] = variants.get(r, 0) + 1
    return sorted(genes, key=lambda r: (-len(genes[r]), -variants[r], r))


def readout_colors(readouts):
    import matplotlib
    out, extra = {}, {}
    for r in readouts:
        if r in READOUT_COLORS:
            out[r] = READOUT_COLORS[r]
        else:
            extra.setdefault(r.split("_", 1)[0], []).append(r)
    for fam, rs in extra.items():
        cmap = matplotlib.colormaps[_FAMILY_CMAPS.get(fam, "Greys")]
        for r, x in zip(rs, np.linspace(0.75, 0.35, len(rs))):
            out[r] = matplotlib.colors.to_hex(cmap(x))
    return out


def _signed_shares(col):
    """Per variant {readout: signed share}: Shapley term / sum of abs(terms)
    over the measured readouts (+ toward pathogenic, - toward benign; the
    absolute shares sum to 1)."""
    out = []
    for v in col.dropna():
        d = json.loads(v)
        tot = sum(abs(x) for x in d.values())
        if tot > 0:
            out.append({r: x / tot for r, x in d.items()})
    return out


def _attribution_figure(w, stem, readouts=None):
    """Per gene: each readout's mean signed share of the joint-model evidence
    (Shapley term / sum of abs(terms); right = toward pathogenic, left =
    toward benign), for VUS with pathogenic and with benign evidence and for
    P/LP and B/LB controls with evidence in their own direction. Readouts
    pushing against a variant's call show up on the other side of 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = w["mv_points"]
    groups = [("VUS, pathogenic evidence", w[VUS_COL] & (pts > 0)),
              ("VUS, benign evidence", w[VUS_COL] & (pts < 0)),
              ("P/LP, pathogenic evidence", w["sample_PLP"] & (pts > 0)),
              ("B/LB, benign evidence", w["sample_BLB"] & (pts < 0))]
    genes = sorted(w["gene"].unique())
    readouts = readouts or readout_order(w)  # most common first, stacked nearest 0
    col = readout_colors(readouts)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 0.35 * len(genes) + 1.8), sharey=True)
    for ax, (title, mask) in zip(axes, groups):
        g = w[mask.astype(bool)]
        for i, gene in enumerate(genes):
            sh = _signed_shares(g.loc[g["gene"] == gene, "mv_shapley"])
            if not sh:
                continue
            mean = pd.DataFrame(sh).fillna(0).mean()
            right = left = 0.0
            for r in readouts:
                x = float(mean.get(r, 0.0))
                if x > 0:
                    ax.barh(i, x, left=right, color=col[r], edgecolor="white", lw=0.5)
                    right += x
                elif x < 0:
                    ax.barh(i, x, left=left, color=col[r], edgecolor="white", lw=0.5)
                    left += x
            ax.text(1.04, i, f"n={len(sh)}", va="center", fontsize=7)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.05, 1.3)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel("mean signed share of the evidence\n(< 0 toward benign, > 0 toward pathogenic)", fontsize=8)
    axes[0].set_yticks(range(len(genes)))
    axes[0].set_yticklabels(genes, fontsize=8)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[r]) for r in readouts]
    fig.legend(handles, readouts, loc="lower center", ncol=len(readouts), fontsize=8, frameon=False,
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

# The main download: the two tables and the in-sample ExCALIBR / main Brnich
# -> ExCALIBR-MV sankeys for VUS, P/LP and B/LB. Its README is README with this
# Files section.
MAIN_SOURCES, MAIN_SAMPLES = ("excalibr", "brnich"), ("vus", "plp", "blb")
README_MAIN_FILES = """## Files

- `labelseq_variant_table.tsv.gz`: one row per gene and protein variant.
- `labelseq_input_with_evidence.tsv.gz`: the LABEL-seq input (one row per
  variant, assay and treatment) with the columns below appended.
- `sankeys/<scope>/<source>/<sample>_<source>_to_mv.png` (and `.pdf`; flows
  as `_flows.csv`): flows from ExCALIBR or Brnich to ExCALIBR-MV.
  - scope: `all` (17 genes) or `non_nu` (the 11 genes with P/LP controls).
  - source: `excalibr`, or `brnich` (main Brnich setting).
  - sample: `vus` (ClinVar {rel} VUS), `plp`, `blb`.
  - All three methods use in-sample points: P/LP and B/LB controls helped
    calibrate each method.
- `figures_non_nu/<source>_to_mv_panels.png`: ExCALIBR or main Brnich ->
  ExCALIBR-MV (in-sample) for the 11 genes with P/LP controls; panels a P/LP,
  b B/LB, c VUS.

"""


def _readme(main=False):
    text = README.format(fits=LABELSEQ_FITS_DIR, pknn=LABELSEQ_PKNN_DIR, rel=LABELSEQ_CLINVAR_RELEASE)
    if main:
        a, b = text.index("## Files"), text.index("## Columns")
        text = text[:a] + README_MAIN_FILES.format(rel=LABELSEQ_CLINVAR_RELEASE) + text[b:]
    return text


def stage_bundle():
    """The two downloads, beside the output directory:
    labelseq_mv_results_main_<date>.tar.gz (the tables, the in-sample
    ExCALIBR / main Brnich sankeys for VUS, P/LP and B/LB, and README_main.md
    as README.md) and labelseq_mv_results_full_<date>.tar.gz (every output)."""
    import datetime
    import tarfile
    out = LABELSEQ_OUTPUT_DIR
    if not LONG.exists() or not LONG_STAMP.exists() or LONG_STAMP.read_text() != _wide_digest():
        write_long_table()
    (out / "README.md").write_text(_readme())
    (out / "README_main.md").write_text(_readme(main=True))
    tables = ["labelseq_variant_table.tsv.gz", "labelseq_input_with_evidence.tsv.gz"]
    figures = sorted(str(f.relative_to(out)) for f in FIG_DIR.glob("*.png"))
    sankeys = sorted(f.relative_to(out) for scope in ("all", "non_nu") for src in MAIN_SOURCES
                     for smp in MAIN_SAMPLES for f in (out / "sankeys" / scope / src).glob(f"{smp}_{src}_to_mv*"))
    full = ["README.md"] + tables + ["brnich_readouts.csv", "checks.json", "oob_check.json"] + \
        sorted(f.name for f in out.glob("fig_attribution_*")) + ["sankeys", FIG_DIR.name]
    tag = datetime.date.today().strftime("%m%d%y")
    for name, members in (("main", [("README_main.md", "README.md")] + [(t, t) for t in tables + list(map(str, sankeys)) + figures]),
                          ("full", [(m, m) for m in full])):
        f = out.parent / f"labelseq_mv_results_{name}_{tag}.tar.gz"
        tmp = f.with_name(f.name + ".tmp")
        with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
            for src, arc in members:
                tar.add(out / src, arcname=arc)
        os.replace(tmp, f)
        print(f"[bundle] {name}: {f} ({f.stat().st_size / 1e6:.0f} MB, {len(members)} entries)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["oob", "attribution", "table", "sankeys", "figures", "bundle", "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sources", nargs="*", choices=list(SOURCES) + ["attribution"],
                    help="sankeys: only these sankey sources (and/or the attribution figures)")
    a = ap.parse_args()
    LABELSEQ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.stage in ("oob", "all"):
        stage_oob(a.workers)
    if a.stage in ("attribution", "all"):
        stage_attribution(a.workers)
    if a.stage in ("table", "all"):
        stage_table(max(1, a.workers // 2))
    if a.stage in ("sankeys", "all"):
        stage_sankeys(a.sources)
    if a.stage in ("figures", "all"):
        stage_figures()
    if a.stage in ("bundle", "all"):
        stage_bundle()


if __name__ == "__main__":
    main()
