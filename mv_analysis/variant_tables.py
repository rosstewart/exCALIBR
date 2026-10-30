"""Per-variant joint-calibration evidence, appended to each gene's input table.

The multivariate counterpart of the standard pipeline's dataframe_with_points
(analysis/all_variant_evidence.py): every row of the input table that held
the gene's functional measurements is kept unchanged, and the evidence each
paper model gives the row's variant is appended. One table per gene, with one
block of columns per model the gene is in (functional, combined, predictor;
BRCA2's second group of assays as functional_group2 / combined_group2):

  mv_<model>_fitted             the variant is in a sample the model was fitted to
                                (control, population, synonymous or RPV variant)
  mv_<model>_scored             the variant is scored by the model: every
                                measured variant (predictor models: every
                                variant in Chen et al.'s per-variant table;
                                combined: those with both evidence types),
                                by the same stored fits
  mv_<model>_points             evidence points (the paper's production rule)
  mv_<model>_evidence_code      ACMG code for the points (functional PS3/BS3,
                                predictor PP3/BP4, combined Pathogenic/Benign)
  mv_<model>_lr_plus            one LR+ per variant, as the standard tables:
                                the 5th-percentile log LR+ over bootstrap fits
                                floored at 0 or the 95th capped at 0, whichever
                                is farther from 0, then kept inside the band
                                its points imply
  mv_<model>_log_lr_plus_p5/p95 the bootstrap bounds themselves
  mv_<model>_prior              the model's prior for the gene (median over
                                bootstrap fits)
  mv_<model>_posterior          posterior from lr_plus and prior
  mv_<model>_points_out_of_bag  points from the fits whose samples did not contain
                                the variant (the standard points for unfitted ones)
  mv_<model>_n_out_of_bag_fits  number of those fits
  mv_<model>_sample_<name>      the model's samples the variant belongs to
  mv_<model>_uv_nonconflicting_points  each source calibrated alone, combined
                                by the nonconflicting rule (the comparator)
  mv_<model>_uv_points__<source>      each source's own univariate points

TP53 adds the reduced-penetrance analysis (mv_functional_rpv_*): the median
log LR of the RPV sample against benign and against P/LP, their points, the
in-sample and out-of-bag classes, and membership of the reference lists.

Variants scored by a predictor or combined model but absent from the input
table are appended as rows of their own (mv_row_source), with input columns
empty; mv_protein_variant gives every row's protein change. Rows scored by no
model have empty evidence. Requires the P-KNN prep (MV_PKNN_DIR) and the production
out-of-bag outputs (production_oob.py), as the comparison tables do.

Published names (public_name): mv_* -> excalibrmv_*. Each source calibrated
alone is named by who calibrated it: excalibr_* (assays) or chen_* (predictors).

Outputs: PAPER_FIGURES_DIR/variant_tables/<GENE>_mv_variant_evidence.tsv.gz
and README.md (column definitions).

Usage:
  python -m mv_analysis.variant_tables [--genes TP53 BRCA2 ...]
"""
import argparse
import ast
import contextlib
import functools
import io
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config

OUT_DIR = Path(config.PAPER_FIGURES_DIR) / "variant_tables"
INTEGRATED = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
CLASS_NAMES = {"plp_like": "P/LP-like", "low_pen_rpv": "RPV-like", "intermediate": "Intermediate",
               "benign_like": "Benign-like", "never_out_of_bag": "never out of bag"}


def _paper_keys():
    """{gene: [(key, meta)]} for every evaluated paper model, from the P-KNN prep."""
    out = defaultdict(list)
    for p in sorted((Path(config.PKNN_DIR) / "prep").glob("*.json")):
        m = json.load(open(p))
        if config.paper_includes(m["key"]):
            gene = "FGFR" if m["gene_set"] == "fgfr" else m["gene"].upper()
            out[gene].append((m["key"], m))
    return out


def _model_name(meta):
    return meta["panel"] + (f"_group{meta['cluster_idx'] + 1}" if meta["cluster_idx"] else "")


def _evidence_code(points, panel):
    from analysis.all_variant_evidence import POINTS_TO_EVIDENCE_CODE
    code = pd.Series(points).map(lambda p: POINTS_TO_EVIDENCE_CODE.get(int(p)) if pd.notna(p) else None)
    if panel == "predictor":
        return code.str.replace("PS3", "PP3").str.replace("BS3", "BP4").to_numpy()
    if panel == "combined":
        return code.str.replace("PS3", "Pathogenic").str.replace("BS3", "Benign").to_numpy()
    return code.to_numpy()


def _uv_columns(gene, gene_set, ms):
    """{column suffix: per-row points} for each source calibrated alone and
    their nonconflicting aggregate, named by who calibrated the source:
    `excalibr` for assays (each calibrated by ExCALIBR), `chen` for predictors
    (Chen et al.'s gene-specific calibrations, config.PREDICTOR_UV_SOURCE)."""
    from analysis.plot_common import display_name
    from mv_analysis import uv_sources
    from mv_analysis.report import _UV_DISPATCH_GENE_SET, _uv_nonconflicting_points
    g = gene.lower() if gene_set == "labelseq" else gene.upper()
    pred = "chen" if config.PREDICTOR_UV_SOURCE == "chen" else "excalibr"
    cols = {}
    with contextlib.redirect_stdout(io.StringIO()):
        agg = _uv_nonconflicting_points(g, gene_set, ms)
        if gene_set.startswith("combined"):
            parts = [("excalibr", uv_sources.load_combined_uv_points(g, ms)),
                     (pred, uv_sources.load_predictor_mv_uv_points(g, ms))]
            agg_name = "excalibr" if pred == "excalibr" else "excalibr_chen"
        elif gene_set == "predictor":
            parts = [(pred, uv_sources.load_uv_points(g, ms, _UV_DISPATCH_GENE_SET[gene_set]))]
            agg_name = pred
        else:
            parts = [("excalibr", uv_sources.load_uv_points(g, ms, _UV_DISPATCH_GENE_SET.get(gene_set, gene_set)))]
            agg_name = "excalibr"
    if agg is not None:
        cols[f"{agg_name}_nonconflicting_points"] = np.asarray(agg, float)
    for source, part in parts:
        if part is None:
            continue
        for name, row in zip(part[0], part[1]):
            cols[f"{source}_points__{display_name(name)}"] = np.asarray(row, float)
    return cols


SAMPLE_LABELS = {"P/LP": "PLP", "Pathogenic/Likely Pathogenic": "PLP", "B/LB": "BLB",
                 "Benign/Likely Benign": "BLB", "population": "gnomAD", "gnomAD": "gnomAD",
                 "Synonymous": "synonymous", "synonymous": "synonymous", "RPV": "RPV"}
PREDICTOR_COLUMNS = {"REVEL": "REVEL_score", "MutPred2": "MutPred2_score", "AlphaMissense": "AlphaMissense_score"}


class _ExtendedMS:
    """A model's MultiScoreset with further rows appended that belong to no
    sample: they are scored by the stored fits but take no part in the prior
    or any other estimate. Everything else is read from the model's ms."""

    def __init__(self, ms, ids, X):
        self._base = ms
        sa = np.asarray(ms.sample_assignments)
        self.scores = np.vstack([np.asarray(ms.scores, float), X])
        self.sample_assignments = np.vstack([sa, np.zeros((len(X), sa.shape[1]), sa.dtype)])
        self._variants_kept = list(ms._variants_kept) + list(ids)
        self.kept_variants = list(ms.kept_variants) + list(ids)
        self.n_variants = len(self.scores)

    def __getattr__(self, name):
        return getattr(self._base, name)


@functools.lru_cache(maxsize=1)
def _chen():
    cols = ["gene_symbol", "protein_variant"] + list(PREDICTOR_COLUMNS.values())
    return pd.read_csv(config.CHEN_PREDICTOR_TABLE, usecols=cols)


def _chen_scores(gene, names):
    """(protein ids, scores in the order of `names`) for every variant of
    `gene` in Chen et al.'s per-variant table."""
    c = _chen()
    c = c[c["gene_symbol"].astype(str).str.upper() == gene.upper()].drop_duplicates("protein_variant")
    return c["protein_variant"].astype(str).to_numpy(), c[[PREDICTOR_COLUMNS[n] for n in names]].to_numpy(float)


def _all_rows_ms(m, names):
    """The model's data with every measured variant kept (None when the
    model already holds them all: TP53's functional panel)."""
    gs, gene, cl, rj = m["gene_set"], m["gene"], m["cluster_idx"], config.PAPER_RESULTS_JSON
    if gs == "tp53":
        return None
    if gs == "integrated":
        def build():
            return config.build_multiscoresets_for_gene_set("integrated", genes=[gene],
                                                            regularization_type="all_assayed").get(gene)
    elif gs == "tsc2":
        from src.assay_calibration.data_utils.dataset import BasicMultiScoreset
        from src.assay_calibration.multivariate_data import tsc2

        def build():
            return BasicMultiScoreset(tsc2.build_tsc2_functional_scoresets(keep_unlabelled=True),
                                      dataset_names=tsc2.DATASET_NAMES, sample_names=tsc2.SAMPLE_NAMES)
    elif gs == "fgfr":
        from unittest import mock
        from src.assay_calibration.multivariate_data import fgfr

        orig = fgfr.Scoreset

        class _AllAssayed:  # fgfr builds its Scoresets without unlabelled rows; keep them
            @staticmethod
            def from_mavedb(*a, **k):
                k["regularization_type"] = "all_assayed"
                return orig.from_mavedb(*a, **k)

        def build():
            with mock.patch.object(fgfr, "Scoreset", _AllAssayed):
                return next(iter(config.build_multiscoresets_for_gene_set("fgfr").values()))
    elif gs == "combined":
        from src.assay_calibration.multivariate_data.combined import (
            DEFAULT_INTEGRATED_DATAFRAME, build_combined_multiscoreset, build_functional_scoresets,
            get_functionally_assayed_protein_variants)
        from src.assay_calibration.multivariate_data.common import (functional_datasets_for_gene,
                                                                    resolve_clinvar_release)

        def build():  # as analysis/compute_clinvar_vus_points_by_panel.py's combined panel
            df = _integrated() if DEFAULT_INTEGRATED_DATAFRAME == INTEGRATED else \
                pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
            datasets = functional_datasets_for_gene(df, gene)
            fs = build_functional_scoresets(df, gene, datasets, clinvar_release=resolve_clinvar_release(gene),
                                            regularization_type="all_assayed")
            return build_combined_multiscoreset(
                gene, fs, datasets, config.PREDICTOR_RAW_DATA_DIR,
                functionally_assayed_variants=get_functionally_assayed_protein_variants(df, gene, datasets),
                require_both_modalities=False)
    else:
        return None
    from src.assay_calibration.multivariate_data.redundancy_collapse import select_dims
    with contextlib.redirect_stdout(io.StringIO()):
        full = build()
        have = [str(n) for n in full.dataset_names]
        if not set(names) <= set(have):
            return config.restrict_ms_to_stored_fit(full, gene, gs, cluster_idx=cl, results_json=rj, rebuild=build)
    # the model's own dimensions by name: with every variant kept, the
    # overlap ranking that orders a gene's groups of assays can change
    if have != names:
        select_dims(full, [have.index(n) for n in names])
    return full


def _extra_rows(m, ms):
    """(ids, scores) of the variants the model does not hold, in the model's
    dimensions and native id format. Scores of variants the model does hold
    must agree with the model's, or the extra rows would be on another scale."""
    names = [str(n) for n in ms.dataset_names]
    have = {str(v): i for i, v in enumerate(ms.kept_variants)}
    X0 = np.asarray(ms.scores, float)
    if m["gene_set"] == "predictor":
        ids, X = _chen_scores(m["gene"], names)
    else:
        full = _all_rows_ms(m, names)
        if full is None:
            return [], None
        if m["gene_set"] == "combined" and [str(n) for n in full.dataset_names] != names:
            # the rebuild cannot reproduce the model's dimensions (TP53: its
            # readouts collapsed by redundancy_collapse); the model already
            # holds every measured variant, so only its predictor dimensions
            # need filling below
            print(f"  {m['key']}: all-variant rebuild has other dimensions; extending the model's own rows")
            full = ms
        assert [str(n) for n in full.dataset_names] == names, f"{m['key']}: dimensions differ from the model's"
        ids, X = list(full.kept_variants), np.asarray(full.scores, float).copy()
        if m["gene_set"] == "combined":
            # predictor dimensions for every variant from Chen's table; only
            # variants with both evidence types are scored (the intersection)
            pdims = [j for j, n in enumerate(names) if n in PREDICTOR_COLUMNS]
            fdims = [j for j in range(len(names)) if j not in pdims]
            cid, cX = _chen_scores(m["gene"], [names[j] for j in pdims])
            cpos = {c: i for i, c in enumerate(cid)}
            for i, v in enumerate(ids):
                k = cpos.get(str(v))
                if k is not None:
                    X[i, pdims] = np.where(np.isnan(X[i, pdims]), cX[k], X[i, pdims])
            both = ~np.isnan(X[:, pdims]).all(1) & ~np.isnan(X[:, fdims]).all(1)
            ids, X = [v for v, b in zip(ids, both) if b], X[both]
    shared = [(i, have[str(v)]) for i, v in enumerate(ids) if str(v) in have]
    if shared:
        a, b = X[[i for i, _ in shared]], X0[[j for _, j in shared]]
        same = np.isnan(a) | np.isnan(b) | np.isclose(a, b, rtol=1e-6, atol=1e-9)
        ok = same.all(1)
        # a protein change can come from several nucleotide variants: assays
        # at the protein level average them (keeping every measured variant
        # can add some) and REVEL scores each, so a few shared variants
        # differ; the scale must be the same
        assert ok.mean() >= 0.9, f"{m['key']}: {(~ok).sum()}/{len(ok)} shared variants score differently"
        if not ok.all():
            print(f"  {m['key']}: {(~ok).sum()}/{len(ok)} model variants average further nucleotide variants "
                  "when all are kept (model values reported)")
    # new rows: variants the model lacks, and (combined) model variants that
    # lack predictor scores in the model but have them in Chen's table; the
    # latter are scored as copies, the model's own row is left as it is
    gained = np.zeros(len(ids), bool)
    if m["gene_set"] == "combined":
        pdims = [j for j, n in enumerate(names) if n in PREDICTOR_COLUMNS]
        for i, v in enumerate(ids):
            j = have.get(str(v))
            gained[i] = j is not None and np.isnan(X0[j, pdims]).all() and not np.isnan(X[i, pdims]).all()
    new = [i for i, v in enumerate(ids) if (str(v) not in have or gained[i]) and not np.isnan(X[i]).all()]
    keep, seen = [], set()
    for i in new:
        if str(ids[i]) not in seen:
            seen.add(str(ids[i]))
            keep.append(i)
    return [ids[i] for i in keep], X[keep]


def _score_extended(ms, m):
    """(per-fit log LR+, points) over all rows of `ms`, from the paper's run."""
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    rj = config.PAPER_RESULTS_JSON
    an = build_gene_set_analysis(ms, m["gene"].lower(), rj, dataset_name=m["dataset_name"], gene_set=m["gene_set"])
    with fast_results_json(rj), contextlib.redirect_stdout(io.StringIO()):
        an.run(partial_pattern_mode="trust_global", n_jobs=4, **DEFAULT_RUN_KWARGS)
    r = an.results[m["config"]]
    return np.asarray(r["lr_matrix"], float), np.asarray(r["points"], float)


def _out_of_bag_points(mv, points, check=True):
    """(points, number of fits) per model row from the fits whose samples did
    not contain it, by production_oob's rule (5th/95th percentile of log LR+
    over those fits, thresholds at their median prior). A row in no sample is
    out of bag in every fit; its points are then the standard points."""
    from mv_analysis.production_oob import _points
    L, oob, priors, pv = mv["L"], mv["oob"].astype(bool), mv["priors"], mv["point_values"]
    out, n = np.full(L.shape[1], np.nan), oob.sum(0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for j in np.flatnonzero(n):
            lv = L[oob[:, j], j]
            out[j] = _points(np.nanpercentile(lv, 5), np.nanpercentile(lv, 95), float(np.median(priors[oob[:, j]])), pv)
    every = n == L.shape[0]
    assert not check or np.array_equal(np.nan_to_num(out[every]), np.nan_to_num(points[every])), \
        "variants out of bag in every fit must keep their standard points"
    return out, n


def model_columns(key, meta):
    """Per-variant evidence of one model: DataFrame indexed by the model's
    variant id, with columns named without the mv_<model>_ prefix."""
    from analysis.all_variant_evidence import (POINT_VALUES, _combine_directional_log_lr,
                                               _posterior_from_log_lr, _reconcile_log_lr_with_points)
    from mv_analysis.comparison_methods import pknn_runs as RP
    from mv_analysis.production_oob import OUT as OOB_DIR, _log_c
    d, m = RP.load_prep(key)
    mv = np.load(OOB_DIR / key / "mv.npz")
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                              results_json=config.PAPER_RESULTS_JSON)
    variants = np.asarray(d["variants"], dtype=object)
    assert [str(v) for v in ms.kept_variants] == list(variants), f"{key}: model rows differ from the prep"
    assert mv["L"].shape[1] == len(variants), f"{key}: out-of-bag fits differ from the prep"
    points = np.asarray(d["mv_points"], float)
    assert np.array_equal(np.nan_to_num(points), np.nan_to_num(mv["insample_points"])), f"{key}: points differ"
    L = mv["L"]
    fitted = np.asarray(d["sa"], bool).any(1)  # in a sample the model was fitted to
    oob_pts, n_oob = _out_of_bag_points(mv, points)
    # every other measured variant (predictor models: every variant Chen et
    # al. scored), scored by the same stored fits
    ids_x, X_x = _extra_rows(m, ms)
    if len(ids_x):
        ms = _ExtendedMS(ms, ids_x, X_x)
        L_all, pts_all = _score_extended(ms, m)
        n = len(variants)
        assert np.allclose(L_all[:, :n], L, equal_nan=True), f"{key}: extended scoring changes the model's LRs"
        assert np.array_equal(np.nan_to_num(pts_all[:n]), np.nan_to_num(points)), f"{key}: extended scoring changes points"
        L, points = L_all, pts_all
        # a new variant is in no sample, so every fit is out of bag: its
        # standard points. A copy of a model variant (combined: predictor
        # scores added) keeps that variant's samples and out-of-bag fits
        orig = {str(v): i for i, v in enumerate(variants)}
        src = np.array([orig.get(str(v), -1) for v in ids_x])
        sa_x = np.zeros((len(ids_x), d["sa"].shape[1]), bool)
        sa_x[src >= 0] = np.asarray(d["sa"], bool)[src[src >= 0]]
        oob_x, nx = pts_all[n:].astype(float).copy(), np.full(len(ids_x), L.shape[0])
        if (src >= 0).any():
            cp = np.flatnonzero(src >= 0)
            sub = dict(L=L_all[:, n + cp], oob=mv["oob"][:, src[cp]], priors=mv["priors"],
                       point_values=mv["point_values"])
            oob_x[cp], nx[cp] = _out_of_bag_points(sub, pts_all[n + cp], check=False)
        variants = np.concatenate([variants, np.asarray([str(v) for v in ids_x], dtype=object)])
        fitted = np.r_[fitted, sa_x.any(1)]
        oob_pts, n_oob = np.r_[oob_pts, oob_x], np.r_[n_oob, nx]
    else:
        sa_x = np.zeros((0, d["sa"].shape[1]), bool)
    prior = float(m["mv_median_prior"])  # the model's own (m["prior"] is P-KNN's: 0.0441 for predictors)
    lc = _log_c(round(prior, 4))
    tau_p = np.array([pv / 8 * lc for pv in POINT_VALUES])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        p5, p95 = np.nanpercentile(L, 5, axis=0), np.nanpercentile(L, 95, axis=0)
    log_lr = _reconcile_log_lr_with_points(_combine_directional_log_lr(p5, p95), points, tau_p, -tau_p)
    df = pd.DataFrame({"fitted": fitted, "scored": True, "points": points,
                       "evidence_code": _evidence_code(points, m["panel"]),
                       "lr_plus": np.exp(log_lr), "log_lr_plus_p5": p5, "log_lr_plus_p95": p95,
                       "prior": prior, "posterior": _posterior_from_log_lr(log_lr, prior),
                       "points_out_of_bag": oob_pts, "n_out_of_bag_fits": n_oob}, index=variants)
    oob_f = Path(config.PKNN_DIR) / "production_oob" / f"{key}.csv"
    if oob_f.exists():
        o = pd.read_csv(oob_f).set_index("variant")
        o = o[o["n_oob"] > 0]
        a, b = df.loc[o.index, "points_out_of_bag"].to_numpy(float), o["MV"].to_numpy(float)
        assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b)), f"{key}: out-of-bag points differ from production_oob's"
    sa = np.vstack([np.asarray(d["sa"], bool), sa_x])
    for j, name in enumerate(d["sample_names"]):
        label = SAMPLE_LABELS.get(str(name), str(name))
        if m["gene"].upper() == "TP53" and label == "sample_4":
            label = "RPV"  # TP53's combined model carries S_R unnamed (the functional model's 22 RPV variants)
        df[f"sample_{label}"] = sa[:, j]
    for c, v in _uv_columns(m["gene"], m["gene_set"], ms).items():
        df[c] = v
    # combined models carry synthetic predictor-only duplicates of assayed
    # controls (combined.split_predictor_clinvar_leakage); they are not variants
    keep = ~df.index.astype(str).str.endswith("__predictor_dup")
    if m["gene_set"] == "combined":
        # fitted on the union, but evidence is reported (and evaluated) only
        # for variants with both evidence types
        from mv_analysis.report import _both_modality_mask
        keep &= np.asarray(_both_modality_mask(ms), bool)
    return df[keep]


def tp53_rpv_columns():
    """TP53 reduced-penetrance results per variant (index: protein change)."""
    from mv_analysis import tp53_rpv_validation as T
    cfg = config.FIXED_MV_CONFIGS[("TP53", "tp53")]
    d = T._oob_matrices(cfg)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        vs_b, vs_p = np.nanmedian(d["vs_b"], axis=0), np.nanmedian(d["vs_p"], axis=0)

    def pts(x):
        p = np.zeros(len(x), int)
        for pv in d["point_values"]:
            p[x >= d["tau_p"][pv - 1]] = pv
        for pv in d["point_values"]:
            p[x <= d["tau_b"][pv - 1]] = -pv
        return p
    oc = T.oob_calls(cfg).set_index("Variant")
    train, held = T._rpv_sets()
    idx = pd.Index(d["variants"])
    df = pd.DataFrame({"rpv_vs_benign_log_lr": vs_b, "rpv_vs_plp_log_lr": vs_p,
                       "rpv_vs_benign_points": pts(vs_b), "rpv_vs_plp_points": pts(vs_p),
                       "rpv_class": oc["call_all_fits"].reindex(idx).map(CLASS_NAMES).to_numpy(),
                       "rpv_class_out_of_bag": oc["call"].reindex(idx).map(CLASS_NAMES).to_numpy(),
                       "rpv_reference_list": np.where(idx.isin(train), "fitted (S_R)", np.where(
                           idx.isin(held), "held out (low confidence)", ""))}, index=idx)
    # S_R excludes ClinVar B/LB (tp53.build_tp53_dataframe): mark those reference variants
    v = pd.read_csv(Path(config.PAPER_TP53_RPV_DIR) / "rpv_validation_variants.csv").set_index("Variant")
    df.loc[df.index.isin(train) & (v["ClinVar_category"].reindex(df.index) == "B/LB").to_numpy(),
           "rpv_reference_list"] = "reference list, ClinVar B/LB (not fitted)"
    return df


def _norm_tuple(s):
    """'(\\'ASPA\\', \\'17\\', \\'3476163\\', \\'ACT\\', \\'GCA\\')' -> 'ASPA:17:3476163:ACT:GCA'."""
    try:
        t = ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return str(s)
    return ":".join(_norm_field(x) for x in (t if isinstance(t, tuple) else (t,)))


def _norm_field(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(x)
    except (TypeError, ValueError):
        return str(x)


def _protein_short(aa_ref, aa_pos, aa_alt):
    from analysis.all_variant_evidence import _protein_variant
    return _protein_variant(aa_ref, aa_pos, aa_alt)


@functools.lru_cache(maxsize=1)
def _integrated():
    return pd.read_csv(INTEGRATED, sep="\t", low_memory=False)


def input_table(gene, gene_sets):
    """(input DataFrame, genomic key Series, protein key Series) for a gene:
    the table its functional measurements were read from."""
    if "tp53" in gene_sets:
        from src.assay_calibration.multivariate_data import tp53
        df = pd.read_csv(tp53.DEFAULT_VARIANTS_PATH)
        return df, pd.Series(None, index=df.index, dtype=object), df["Variant"].astype(str)
    if "tsc2" in gene_sets:
        from src.assay_calibration.multivariate_data import tsc2
        df = pd.read_csv(tsc2.DEFAULT_MAVE_CSV)
        return df, pd.Series(None, index=df.index, dtype=object), df["hgvs_pro"].map(tsc2.protein_id)
    if "fgfr" in gene_sets:
        from src.assay_calibration.multivariate_data import fgfr
        df = fgfr.load_fgfr_dataframe()
    else:
        df = _integrated()
        df = df[df["Gene"].astype(str).str.upper() == gene].reset_index(drop=True)
    # genomic key as the Scoreset builds it (FGFR's MaveDB columns are renamed
    # to these on ingestion: data_utils/dataset.py's MaveDB column map)
    gcols = ["Gene", "Chrom", "hg38_start", "ref_allele", "alt_allele"] if "hg38_start" in df.columns else \
        ["gene_symbol", "mapped_hgvs_g_chromosome", "mapped_hgvs_g_start", "mapped_hgvs_g_ref", "mapped_hgvs_g_alt"]
    gkey = df[gcols].astype(object).apply(lambda r: ":".join(_norm_field(x) for x in r), axis=1)
    pkey = pd.Series([_protein_short(a, b, c) for a, b, c in zip(df.get("aa_ref"), df.get("aa_pos"), df.get("aa_alt"))],
                     index=df.index) if "aa_ref" in df.columns else pd.Series(None, index=df.index, dtype=object)
    return df, gkey, pkey


def gene_table(gene, models):
    """The gene's input table with every model's columns appended, plus one
    row per protein change scored by a predictor or combined model but absent
    from the input table (input columns empty, mv_row_source says which)."""
    gene_sets = {m["gene_set"] for _, m in models}
    df, gkey, pkey = input_table(gene, gene_sets)
    out = df.copy()
    out["mv_row_source"] = "input table"
    out["mv_protein_variant"] = pkey.to_numpy()
    extra = {}  # model prefix -> its columns for protein changes not in the input
    in_input = set(pkey.dropna().astype(str))
    for key, m in sorted(models, key=lambda km: (["functional", "combined", "predictor"].index(km[1]["panel"]),
                                                   km[1]["cluster_idx"])):
        from mv_analysis.production_oob import OUT as OOB_DIR
        if not (OOB_DIR / key / "mv.npz").exists():
            print(f"  {gene} {key}: SKIPPED, no out-of-bag fits yet (production_oob.py mv)")
            continue
        cols = model_columns(key, m)
        genomic = m["gene_set"] in ("integrated", "fgfr")
        cols.index = [_norm_tuple(v) for v in cols.index] if genomic else cols.index.astype(str)
        cols = cols[~cols.index.duplicated()]
        j = (gkey if genomic else pkey).map(lambda k: k if k in cols.index else None)
        block = cols.reindex(j.to_numpy())
        block.index = out.index
        block["fitted"] = block["fitted"].fillna(False).astype(bool)
        block["scored"] = block["scored"].fillna(False).astype(bool)
        n_in, n_sc = int(block["fitted"].sum()), int(block["scored"].sum())
        assert n_in > 0, f"{gene} {key}: no input row matched the model's variants"
        prefix = f"mv_{_model_name(m)}_"
        out = out.join(block.add_prefix(prefix))
        n_extra = 0
        if not genomic and m["gene_set"] != "tp53":
            x = cols[~cols.index.isin(in_input)]
            if len(x):
                extra[prefix] = x
            n_extra = len(x)
        print(f"  {gene} {key}: {int(cols['fitted'].sum())} fitted variants, {len(cols)} scored; "
              f"input rows {n_in} fitted, {n_sc} scored, of {len(df)}; {n_extra} scored variants not in the input")
        if m["gene_set"] == "tp53":
            r = tp53_rpv_columns()
            rb = r.reindex(pkey.to_numpy())
            rb.index = out.index
            out = out.join(rb.add_prefix("mv_functional_"))
    if extra:
        ids = sorted(set().union(*[set(x.index) for x in extra.values()]))
        add = pd.DataFrame({"mv_row_source": "scored only (not in the input table)", "mv_protein_variant": ids})
        for prefix, x in extra.items():
            blk = x.reindex(ids).reset_index(drop=True)
            for c in ("fitted", "scored"):
                blk[c] = blk[c].fillna(False).astype(bool)
            add = add.join(blk.add_prefix(prefix))
        for c in out.columns:  # models whose variants are all in the input still get their flags
            if c.endswith(("_fitted", "_scored")) and c not in add.columns:
                add[c] = False
        out = pd.concat([out, add], ignore_index=True)
    else:
        out = out.drop(columns="mv_row_source")  # every row is an input row
    if "Variant" in df.columns and out["mv_protein_variant"].equals(out["Variant"].astype(object).where(out["Variant"].notna())):
        out = out.drop(columns="mv_protein_variant")  # TP53: the input's own "Variant"
    return out.rename(columns=public_name)


def public_name(c):
    """Published column name: joint-calibration columns `excalibrmv_*`."""
    return "excalibrmv_" + c[len("mv_"):] if c.startswith("mv_") else c


README = """# Joint-calibration (ExCALIBR-MV) evidence per variant

One file per gene: `<GENE>_mv_variant_evidence.tsv.gz` (tab-separated), and
`<GENE>_mv_variant_evidence_functional.tsv.gz` with only the functional
(assay) columns: no predictor or combined columns and no scored-only rows. Each is
the gene's input table, unchanged, with `excalibrmv_*` columns appended; rows are never
dropped. Input tables: TP53 `tp53_annotated_variants.csv`; TSC2 the Biar et al.
MAVE table; FGFR1-4 `fgfr_annotated.tsv.gz`; all other genes the rows of
`integrated_variant_effect_dataset_pp_final.tsv.gz` (one row per assay x
variant, so a variant's evidence repeats across its assays' rows).

## Row columns

| column | meaning |
|---|---|
| `excalibrmv_row_source` | only in tables with such rows: `input table`, or `scored only (not in the input table)`, a protein change scored by a predictor model (from Chen et al.'s per-variant table) that the input table lacks; its input columns are empty |
| `excalibrmv_protein_variant` | protein change (e.g. `R175H`); omitted for TP53, whose `Variant` column is already that |

## Per model: `excalibrmv_<model>_<column>`

`<model>` is `functional`, `combined` or `predictor` (`functional_group2`,
`combined_group2`: BRCA2's second group of assays).

| column | meaning |
|---|---|
| `fitted` | the variant is in one of the samples the model was fitted to (see `sample_*`); false for every other measured variant, which the model only scores |
| `scored` | the model gives the variant evidence: every measured variant; predictor models every variant in Chen et al.'s table; combined models only variants with both functional and predictor scores |
| `points` | evidence points, the paper's rule: 5th-percentile log LR+ over bootstrap fits against the pathogenic thresholds, 95th against the benign ones, at the model's prior |
| `evidence_code` | ACMG code for the points: PS3/BS3 (functional), PP3/BP4 (predictor), Pathogenic/Benign (combined) |
| `lr_plus` | one LR+ per variant, as in the standard ExCALIBR tables: the 5th-percentile bound floored at 1 or the 95th capped at 1, whichever is farther from 1, kept inside the band its points imply |
| `log_lr_plus_p5`, `log_lr_plus_p95` | natural-log LR+ at the 5th and 95th percentiles over bootstrap fits |
| `prior` | the model's prior for the gene (median over bootstrap fits) |
| `posterior` | probability of pathogenicity from `lr_plus` and `prior` |
| `points_out_of_bag` | points from only the bootstrap fits whose samples did not contain the variant, at their median prior (the paper's out-of-bag evaluation of controls). A variant that is not `fitted` is out of bag in every fit, so this equals `points` |
| `n_out_of_bag_fits` | number of those fits (all of them for unfitted variants) |
| `sample_PLP`, `sample_BLB`, `sample_gnomAD`, `sample_synonymous`, `sample_RPV` | the model's samples the variant belongs to: ClinVar P/LP, ClinVar B/LB, gnomAD, synonymous, and TP53's reduced-penetrance sample S_R. Each model has its own: functional and combined models take labels from their functional data (TP53: the input table's `ClinVar_category`, gnomAD v4), predictor models from Chen et al.'s labelled sets (ClinVar January 2025, their own filtering), so a few variants differ between `functional` and `predictor` |
| `excalibr_nonconflicting_points` (functional), `chen_nonconflicting_points` (predictor), `excalibr_chen_nonconflicting_points` (combined) | the comparator: each source calibrated alone, combined by the nonconflicting rule; empty when no source measured the variant |
| `excalibr_points__<assay>` | that assay's own ExCALIBR calibration (univariate) applied to its score |
| `chen_points__<predictor>` | that predictor's gene-specific calibration by Chen et al. (domain-aggregate where they published none, e.g. MutPred2 for F9) applied to its score |

## TP53 reduced-penetrance analysis: `excalibrmv_functional_rpv_*`

| column | meaning |
|---|---|
| `rpv_vs_benign_log_lr`, `rpv_vs_plp_log_lr` | median over bootstrap fits of the log likelihood ratio of the reduced-penetrance sample (S_R) against the benign sample and against the P/LP sample |
| `rpv_vs_benign_points`, `rpv_vs_plp_points` | those ratios as points, at the prior of S_R |
| `rpv_class` | P/LP-like, RPV-like, Intermediate or Benign-like, from all fits: Benign-like at <= -1 point against benign; otherwise, with >= +1 point against benign, RPV-like at >= +1 point against P/LP, else P/LP-like; anything else Intermediate |
| `rpv_class_out_of_bag` | the same from the fits whose samples did not contain the variant; used for the external validation |
| `rpv_reference_list` | membership of the curated reduced-penetrance list (Fortuno et al. 2025; Rodriguez et al. 2026): `fitted (S_R)` (these are `sample_RPV`); `reference list, ClinVar B/LB (not fitted)` (on the list but kept out of S_R as ClinVar B/LB); `held out (low confidence)` (met the germline/somatic criterion only at 10 occurrences); empty if not on the list. `sample_RPV` is only the first group |
"""


def run(genes=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    keys = _paper_keys()
    written = []
    for gene in sorted(keys):
        if genes and gene not in {g.upper() for g in genes}:
            continue
        t = gene_table(gene, keys[gene])
        f = OUT_DIR / f"{gene}_mv_variant_evidence.tsv.gz"
        t.to_csv(f, sep="\t", index=False)
        written.append(f)
        # functional-only version: no predictor or combined (predictor-informed) columns
        func = t.loc[t["excalibrmv_row_source"].ne("scored only (not in the input table)")
                     if "excalibrmv_row_source" in t.columns else slice(None)]
        func = func.drop(columns=[c for c in func.columns if c.startswith(("excalibrmv_predictor_", "excalibrmv_combined_",
                                                                         "excalibrmv_row_source"))])
        if any(c.startswith("excalibrmv_functional_") for c in func.columns):
            func.to_csv(OUT_DIR / f"{gene}_mv_variant_evidence_functional.tsv.gz", sep="\t", index=False)
    (OUT_DIR / "README.md").write_text(README)
    (OUT_DIR / "columns.tsv").unlink(missing_ok=True)  # superseded by README.md
    print(f"wrote {len(written)} tables to {OUT_DIR}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="*")
    run(ap.parse_args().genes)
