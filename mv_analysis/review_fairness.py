"""How much of the independent comparator's edge comes from advantages the
joint model does not have?

The comparator's univariate calibrations are rebuilt with the standard
pipeline (run_igvf_batch.py, from the stored bootstrap fits, exactly the
paper's settings otherwise), in variants:
  full        the paper's calibrations (1000 bootstraps; reproduces
              exc_pp_clinvar2025_calib exactly: the check)
  b100        only the first 100 bootstraps of each assay (the joint model's
              bootstrap budget; the univariate fits keep their best-of-100
              restarts, which the stored fits cannot undo)
  geneprior   each assay's evidence thresholds at the joint model's gene
              prior (--manual-prior) instead of its own
  p<p>        the pathogenic percentile p (benign 100-p) for p in 10..25 (the
              pipeline needs p < 100 - p, so not 50), to
              compare with the joint model at the same percentile
Each variant's calibrations replace the paper's for the comparator
(MV_EXC_PP_UV_CALIB), which is then recomputed on the paper's evaluation
controls with the paper's own code (report._uv_nonconflicting_points).
TP53's readouts calibrated on the joint model's scores (TP53_EXTRA_UV_CALIB_DIR)
and TSC2/FGFR (their own sources) keep their calibrations in every variant.

Outputs (review/fairness/): <variant>/ (calibrations), comparator.csv (per
model and variant: MCC, determinate rate, errors), asymmetry.csv.

Usage: python -m mv_analysis.review_fairness [--variants full b100 geneprior p25 ...] [--jobs N]
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

FITS = "/data/ross/assay_calibration/pp_final_clinvar2025_bootstraps/final_clinvar2025_bootstrap_results.json.gz"
OUT = R.out_dir("fairness")
VARIANTS = ("full", "b100", "geneprior", "p10", "p15", "p20", "p25")  # the pipeline needs p < 100 - p


def comparator_datasets():
    """{dataset: joint-model prior of the model whose comparator uses it}."""
    from mv_analysis import review_priors as P
    ds_prior = {}
    for key, m in R.paper_keys():
        if m["panel"] not in ("functional", "combined") or m["gene_set"] in ("fgfr", "tsc2"):
            continue
        dims = [str(x) for x in R.prep(key)[0]["dims"]]
        for path, ds in P._cal_paths(m["gene"].upper(), m["gene_set"]).items():
            if not path.startswith(config.EXC_PP_CLINVAR2025_UV_CALIB):
                continue  # TP53's own-score readouts live elsewhere
            used = any(ds.startswith(d) or d.startswith(ds.replace("_clinvar_2018", "")) for d in dims) \
                or m["gene_set"] == "tp53"
            # the functional model's prior wins over the combined one's
            if used and (ds not in ds_prior or m["panel"] == "functional"):
                ds_prior[ds] = float(m["mv_median_prior"])
    return ds_prior


def _batch(datasets, outdir, fits=FITS, extra=(), jobs=8):
    cmd = [sys.executable, str(Path(config._ROOT) / "run_igvf_batch.py"),
           "--dataset", config.INTEGRATED_VARIANT_EFFECT_DATASET_PATH, "--precomputed-fits", fits,
           "--dataset-configs", config.DATASET_CONFIGS_PATH, "--clinvar-release", "2025",
           "--datasets", *datasets, "--output-dir", str(outdir), "--n-jobs-inner", str(jobs), "--skip-existing",
           *extra]
    with open(Path(outdir).with_suffix(".log"), "a") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))


def _subset_fits(datasets, n=100):
    f = OUT / f"fits_first{n}bootstraps.json.gz"
    if f.exists():
        return f
    d = json.load(gzip.open(FITS, "rt"))
    sub = {ds: {s: d[ds][s] for s in sorted(d[ds], key=int)[:n]} for ds in datasets if ds in d}
    with gzip.open(f, "wt") as g:
        json.dump(sub, g)
    return f


def build(variant, ds_prior, jobs=8):
    out = OUT / variant
    out.mkdir(exist_ok=True)
    datasets = sorted(ds_prior)
    if variant == "full":
        _batch(datasets, out, jobs=jobs)
    elif variant == "b100":
        _batch(datasets, out, fits=str(_subset_fits(datasets)), jobs=jobs)
    elif variant == "geneprior":
        by_prior = defaultdict(list)
        for ds, p in ds_prior.items():
            by_prior[round(p, 6)].append(ds)
        for p, dss in by_prior.items():
            _batch(sorted(dss), out, extra=("--manual-prior", str(p)), jobs=jobs)
    elif variant.startswith("p"):
        _batch(datasets, out, extra=("--pathogenic-percentile", variant[1:]), jobs=jobs)
    return out


def check_full():
    """The rebuilt full calibrations must equal the paper's."""
    bad = []
    for d in (OUT / "full").iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("*_calibration.json"):
            ref = Path(config.EXC_PP_CLINVAR2025_UV_CALIB) / d.name / f.name
            a, b = json.load(open(f)), json.load(open(ref)) if ref.exists() else None
            if b is None or a["point_ranges"] != b["point_ranges"] or a["prior"] != b["prior"]:
                bad.append(d.name)
    return bad


def _comparator_here():
    """Comparator metrics with the calibrations config points at (run in a subprocess)."""
    from mv_analysis.report import _uv_nonconflicting_points
    from mv_analysis.review_priors import _metrics
    rows = []
    for key, m in R.paper_keys():
        if m["panel"] not in ("functional", "combined"):
            continue
        d, _ = R.prep(key)
        ev = np.asarray(d["eval_mask"], bool)
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                                  results_json=config.PAPER_RESULTS_JSON)
            g = m["gene"].lower() if m["gene_set"] == "labelseq" else m["gene"].upper()
            pts = _uv_nonconflicting_points(g, m["gene_set"], ms)
        if pts is None:
            continue
        rows.append(dict(key=key, panel=m["panel"], gene=m["gene"],
                         **_metrics(np.asarray(pts, float)[ev], np.asarray(d["plp"], bool)[ev])))
    return pd.DataFrame(rows)


def comparator(variant):
    calib = config.EXC_PP_CLINVAR2025_UV_CALIB if variant == "paper" else str(OUT / variant)
    f = OUT / f"comparator_{variant}.csv"
    code = ("import warnings; warnings.filterwarnings('ignore'); "
            "from mv_analysis import review_fairness as F; "
            f"F._comparator_here().to_csv({str(f)!r}, index=False)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=config._ROOT,
                   env=dict(os.environ, MV_EXC_PP_UV_CALIB=calib, PYTHONPATH=config._ROOT))
    return pd.read_csv(f).assign(variant=variant)


def asymmetry():
    rows = [
        ("information", "one readout per calibration", "all readouts of the gene jointly"),
        ("bootstraps", "1,000", "100 (paper run)"),
        ("restarts per bootstrap", "100 (best by held-out likelihood)", "3 (Gaussian, MSN, CFUSN)"),
        ("missing readouts", "none (each readout fit on its own variants)", "marginalized per pattern"),
        ("dimensions", "1", "2-13 (TP53 functional 10 after collapse)"),
        ("prior", "one per readout (its own)", "one per gene"),
        ("percentile", "5th / 95th", "5th / 95th"),
        ("model choice", "per readout, by the pipeline's rule", "per gene, by MCC on the controls"),
    ]
    t = pd.DataFrame(rows, columns=["aspect", "independent (univariate)", "joint"])
    t.to_csv(OUT / "asymmetry.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    ds_prior = comparator_datasets()
    json.dump(ds_prior, open(OUT / "datasets_and_gene_priors.json", "w"), indent=1)
    print(f"{len(ds_prior)} comparator datasets", flush=True)
    for v in args.variants:
        build(v, ds_prior, args.jobs)
        print(f"built {v}", flush=True)
        if v == "full":
            bad = check_full()
            assert not bad, f"full rebuild differs from the paper's calibrations: {bad}"
    res = [comparator("paper")] + [comparator(v) for v in args.variants]
    res = pd.concat(res, ignore_index=True)
    res.to_csv(OUT / "comparator.csv", index=False)
    asymmetry()
    piv = res.pivot_table(index=["panel", "key"], columns="variant", values=["mcc", "det_rate"])
    print(piv.round(3).to_string())
    print(res.groupby(["panel", "variant"])[["mcc", "det_rate", "errors"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
