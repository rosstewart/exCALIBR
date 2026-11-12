"""Per-bootstrap priors of the manuscript calibrations, recomputed with production code.

The pipeline saves only the 5th/50th/95th percentiles of the bootstrap priors (lr_values priors_pct). This reruns
production's own process_component_fits (src/assay_calibration/pipeline/visualize.py) on the saved bootstrap fits
and the production-built Scoreset (analysis.legacy_fits.load_scoreset_and_fits), in memory only (nothing in the
pipeline output is written or changed), and keeps the per-bootstrap priors only if the recomputation reproduces the
saved calibration exactly: the median prior, the share of bootstraps kept, the saved prior percentiles and every
evidence point range. Datasets that do not reproduce are reported and get no priors.

    python -m analysis.calibration_quality.recompute_priors [--out DIR] [--n-jobs 8] [--inner-jobs 8]
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

TOL = 1e-9


def _ranges_equal(a, b):
    ka = {int(k): v for k, v in a.items()}
    kb = {int(k): v for k, v in b.items()}
    for k in set(ka) | set(kb):
        ra, rb = ka.get(k, []) or [], kb.get(k, []) or []
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not np.allclose(np.asarray(x, float), np.asarray(y, float), rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
    return True


def recompute(dataset, cal, inner_jobs=8):
    from analysis import config as cfg
    from analysis.legacy_fits import load_scoreset_and_fits
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    import gzip
    # The saved component key names the files ("3c_avg"); benign_method is what production actually used (an avg
    # run on a dataset without a synonymous sample is saved under the avg name with benign_method "benign").
    n_c, _, file_bm = str(cal["n_c"]).partition("_")
    bm = cal["benign_method"]
    scoreset, _, fits, _, _, _, _ = load_scoreset_and_fits(dataset, n_c=n_c, benign_method=file_bm or bm)
    with tempfile.TemporaryDirectory() as tmp:
        pc = PipelineConfig(dataset_csv="", dataset_name=dataset, output_dir=tmp, components=[int(n_c[0])],
                            use_median_prior=True, use_2c_equation=False, benign_method=bm,
                            liberal_monotonicity=bool(cal.get("liberal_monotonicity", 1)),
                            clinvar_release="2018" if cal.get("clinvar_2018") else "2025",
                            point_values=[1, 2, 3, 4, 5, 6, 7, 8], n_jobs=inner_jobs)
        lg = logging.getLogger(f"recompute.{dataset}")
        lg.setLevel(logging.ERROR)
        new = process_component_fits(fits=list(fits), scoreset=scoreset, config=pc, n_c=int(n_c[0]), logger=lg,
                                     bootstrap_seeds=np.asarray(fits.seeds))
    with gzip.open(Path(cfg.OUTPUT_DIR) / dataset / f"{dataset}_{cal['n_c']}_lr_values.json.gz", "rt") as fh:
        saved_pct = json.load(fh).get("priors_pct")
    pri = np.asarray(new["priors"], float)
    deg = np.asarray(new.get("prior_degenerate") if new.get("prior_degenerate") is not None
                     else np.zeros(len(pri), bool), bool)
    use = pri[~deg] if len(deg) == len(pri) and not deg.all() else pri
    pct = np.nanpercentile(use, [5, 50, 95]).tolist()
    checks = dict(
        prior=abs(float(new["prior"]) - float(cal["prior"])) < TOL,
        pct_kept=abs(float(new["pct_bootstraps_kept"] or 1) - float(cal["pct_bootstraps_kept"] or 1)) < TOL,
        priors_pct=saved_pct is not None and np.allclose(pct, saved_pct, rtol=0, atol=TOL),
        point_ranges=_ranges_equal(new["point_ranges"], cal["point_ranges"]),
    )
    seeds = np.asarray(new.get("valid_bootstrap_seeds", fits.seeds)).tolist()
    return dict(dataset=dataset, n_c=cal["n_c"], match=all(checks.values()), checks=checks,
                priors=pri.tolist(), prior_degenerate=deg.tolist(), seeds=seeds,
                n_fits=len(fits), saved_prior=cal["prior"], new_prior=float(new["prior"]),
                saved_pct=saved_pct, new_pct=pct)


def main():
    from joblib import Parallel, delayed
    from analysis import config as cfg
    from analysis.calibration_quality.diagnostics import real_datasets
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(cfg.CALIBRATION_QUALITY_DIR).parent / "real_diagnostics"))
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--inner-jobs", type=int, default=8)
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cals = real_datasets()
    todo = [d for d in cals if not a.datasets or d in a.datasets]

    def one(d):
        try:
            return recompute(d, cals[d], a.inner_jobs)
        except Exception as e:
            return dict(dataset=d, match=False, error=f"{type(e).__name__}: {e}"[:300])
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(d) for d in todo)
    path = out / "recomputed_priors.json"
    old = json.loads(path.read_text()) if path.exists() and a.datasets else {}
    old.update({r["dataset"]: r for r in res})
    path.write_text(json.dumps(old, default=float))
    for r in res:
        if not r["match"]:
            print("NO MATCH", r["dataset"], r.get("error") or r["checks"], r.get("saved_prior"), r.get("new_prior"))
    print(f"{sum(r['match'] for r in res)}/{len(res)} reproduce production exactly -> {path}")


if __name__ == "__main__":
    main()
