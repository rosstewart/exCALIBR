"""Do per-assay priors give the independent comparator easier evidence?

Each univariate (independent) calibration sets its evidence thresholds at its
own prior; the joint model uses one prior per gene. A poor assay can
overestimate its prior and so reach evidence too easily (or, the Tavtigian
constant being non-monotonic in the prior, too hard).

  priors.csv        per gene and assay: the assay's own prior (and instability
                    flag), its determinate rate and errors on the paper's
                    evaluation controls, and the joint model's gene prior.
  common_prior.csv  the comparator (nonconflicting rule over the assays)
                    recomputed with every assay's evidence thresholds set at
                    the joint gene prior instead of its own, against the
                    original; per gene MCC, determinate rate, errors.
  rethreshold_check.csv  per assay, agreement between its stored evidence
                    ranges and the ranges rebuilt here from its LR curves at
                    its own prior (the rebuild must reproduce them; assays it
                    does not reproduce keep their original ranges and are
                    flagged).

Evidence ranges at a prior are rebuilt as the standard pipeline does
(pipeline/visualize.py): fit.calculate_score_ranges on the stored 5th/95th
percentile log LR+ curves (<ds>_lr_values.json.gz), then
enforce_monotonicity_point_ranges, extend_points_to_xlims and
enforce_monotonicity_point_ranges again. Functional assays of the
functional and combined panels; predictors (Chen et al.'s published
thresholds, no curves) keep their thresholds.

Usage: python -m mv_analysis.review_priors
"""
import contextlib
import gzip
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

POINT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
_real_json = json


def ranges_at_prior(lr_path, cal, prior):
    """Evidence ranges ({"1": [[lo, hi]], ...}) rebuilt at `prior` from an
    assay's stored LR curves; None when the curves are missing."""
    from src.assay_calibration.fit_utils.fit import calculate_score_ranges
    from src.assay_calibration.fit_utils.point_ranges import (enforce_monotonicity_point_ranges,
                                                              extend_points_to_xlims)
    if not Path(lr_path).exists():
        return None
    lv = _real_json.load(gzip.open(lr_path, "rt"))
    s = np.asarray(lv["score_range"], float)
    lo, hi = np.asarray(lv["log_lr_plus_p5"], float), np.asarray(lv["log_lr_plus_p95"], float)
    ok = np.isfinite(lo) | np.isfinite(hi)
    s, lo, hi = s[ok], lo[ok], hi[ok]
    p, b, _ = calculate_score_ranges(lo, hi, prior, s, POINT_VALUES)
    pr = {**p, **b}
    flipped = bool(cal.get("scoreset_flipped", 0))
    liberal = bool(cal.get("liberal_monotonicity", 1))
    enforce_monotonicity_point_ranges(pr, POINT_VALUES, s, scoreset_flipped=flipped, liberal=liberal)
    extend_points_to_xlims(pr, POINT_VALUES, s, flipped, inf=True)
    enforce_monotonicity_point_ranges(pr, POINT_VALUES, s, scoreset_flipped=flipped, liberal=liberal)
    return {str(k): [list(map(float, r)) for r in v] for k, v in pr.items()}, s


def _classify(scores, ranges):
    from src.assay_calibration.multivariate_analysis.compare_uv_mv_agg import classify_score_from_point_ranges
    return np.array([classify_score_from_point_ranges(float(x), ranges) for x in scores], float)


class _Override:
    """Stand-in for uv_sources' json module: calibration files listed in
    `ranges` load with those evidence ranges instead of their own."""

    def __init__(self):
        self.ranges = {}

    def load(self, f, *a, **k):
        d = _real_json.load(f, *a, **k)
        r = self.ranges.get(str(getattr(f, "name", "")))
        if r is not None and isinstance(d, dict) and "point_ranges" in d:
            d = dict(d, point_ranges=r)
        return d

    def __getattr__(self, name):
        return getattr(_real_json, name)


def _cal_paths(gene, gene_set):
    """{calibration.json path: dataset} of the assays the comparator uses for a model."""
    from mv_analysis import uv_sources as U
    dc = U._dataset_configs()
    out = {}
    names = list(config.TP53_UV_COLUMNS) if gene_set == "tp53" else U._discover_exc_pp_datasets(gene)
    for ds in names:
        p = U._calibration_json_path_for_dataset(ds, config.EXC_PP_CLINVAR2025_UV_CALIB, dc)
        if p is not None and Path(p).exists():
            out[str(p)] = ds
    if gene_set == "tp53":
        for dim, ds in config.TP53_EXTRA_UV_DIMS.items():
            p = Path(config.TP53_EXTRA_UV_CALIB_DIR) / ds / f"{ds}_3c_benign_calibration.json"
            if p.exists():
                out[str(p)] = ds
    return out


def _metrics(pts, y):
    pts = np.nan_to_num(np.asarray(pts, float))
    tp, fn = int(((pts >= 1) & y).sum()), int(((pts <= -1) & y).sum())
    fp, tn = int(((pts >= 1) & ~y).sum()), int(((pts <= -1) & ~y).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, determinate=tp + tn + fp + fn, total=len(pts),
                det_rate=(tp + tn + fp + fn) / len(pts), errors=fp + fn,
                mcc=(tp * tn - fp * fn) / den if den else np.nan)


def run():
    from mv_analysis import uv_sources as U
    from mv_analysis.report import _uv_nonconflicting_points
    warnings.filterwarnings("ignore")
    out = R.out_dir("priors")
    ov = _Override()
    U.json = ov  # uv_sources reads calibrations through this module-level name
    pri, comm, chk = [], [], []
    try:
        for key, m in R.paper_keys():
            if m["panel"] not in ("functional", "combined") or m["gene_set"] in ("fgfr", "tsc2"):
                continue
            d, _ = R.prep(key)
            ev = np.asarray(d["eval_mask"], bool)
            y = np.asarray(d["plp"], bool)[ev]
            gene = m["gene"].upper()
            gprior = float(m["mv_median_prior"])
            with contextlib.redirect_stdout(io.StringIO()):
                ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                                      results_json=config.PAPER_RESULTS_JSON)
            # per-assay priors and the rebuilt ranges at the assay's own and the gene prior
            own, at_gene = {}, {}
            for path, ds in _cal_paths(gene, m["gene_set"]).items():
                cal = _real_json.load(open(path))
                lr = path.replace("_calibration.json", "_lr_values.json.gz")
                r_own = ranges_at_prior(lr, cal, cal["prior"]) if np.isfinite(cal.get("prior", np.nan)) else None
                if r_own is None:
                    chk.append(dict(key=key, dataset=ds, reproduced=np.nan, note="no LR curves"))
                    continue
                agree = float(np.mean(_classify(r_own[1], r_own[0]) == _classify(r_own[1], cal["point_ranges"])))
                chk.append(dict(key=key, dataset=ds, reproduced=agree))
                if agree >= 0.97:
                    own[path] = r_own[0]
                    at_gene[path] = ranges_at_prior(lr, cal, gprior)[0]
                pri.append(dict(key=key, panel=m["panel"], gene=gene, dataset=ds, assay_prior=cal["prior"],
                                prior_unstable=cal.get("prior_unstable"), joint_gene_prior=gprior,
                                rebuilt=agree >= 0.97))
            # per-assay control metrics with their own ranges (original files)
            ov.ranges = {}
            with contextlib.redirect_stdout(io.StringIO()):
                gs = "combined" if m["gene_set"] == "combined" else m["gene_set"]
                parts = (U.load_combined_uv_points(gene, ms) if gs == "combined"
                         else U.load_uv_points(gene, ms, {"integrated": "integrated-functional"}.get(gs, gs)))
                orig = _uv_nonconflicting_points(gene, m["gene_set"], ms)
            if parts is not None:
                for name, row in zip(parts[0], parts[1]):
                    mm = _metrics(np.asarray(row, float)[ev], y)
                    for r in pri:
                        if r["key"] == key and r["dataset"] == name:
                            r.update({f"assay_{k}": v for k, v in mm.items() if k in ("det_rate", "errors", "FP", "mcc")})
            # comparator rebuilt at each assay's own prior (must equal the original) and at the gene prior
            for tag, rr in (("own_prior_rebuilt", own), ("gene_prior", at_gene)):
                ov.ranges = rr
                with contextlib.redirect_stdout(io.StringIO()):
                    pts = _uv_nonconflicting_points(gene, m["gene_set"], ms)
                comm.append(dict(key=key, panel=m["panel"], gene=gene, version=tag,
                                 n_assays_rebuilt=len(rr), **_metrics(np.asarray(pts, float)[ev], y)))
            ov.ranges = {}
            comm.append(dict(key=key, panel=m["panel"], gene=gene, version="original",
                             n_assays_rebuilt=0, **_metrics(np.asarray(orig, float)[ev], y)))
            print(key, flush=True)
    finally:
        U.json = _real_json
    pd.DataFrame(pri).to_csv(out / "priors.csv", index=False)
    comm = pd.DataFrame(comm)
    comm.to_csv(out / "common_prior.csv", index=False)
    pd.DataFrame(chk).to_csv(out / "rethreshold_check.csv", index=False)
    piv = comm.pivot_table(index=["panel", "key"], columns="version", values=["det_rate", "mcc", "errors"])
    print(piv.round(3).to_string())
    return comm


if __name__ == "__main__":
    run()
