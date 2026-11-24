"""Evidence impact of the prior's uncertainty, for the real calibrations, with production code.

For each calibration, production's process_component_fits is rerun in memory (nothing in the pipeline output is
written) with the prior fixed (manual_prior) at four priors: moved by its predicted mean angular error in each
direction (`typ`: sin^2(theta +- E|error|), a typical error) and at the two ends of its predicted 90% range (`q90`;
diagnostics.predict), all clipped to [0.001, 0.999], on the bootstraps production kept (recomputed_priors.json `seeds`; a fixed prior would
otherwise keep them all), and the gnomAD (population) variants' points are compared with production's own
standard_points. Reported, for each pair: the share of gnomAD variants whose evidence is unchanged at both ends,
the share at each end, and the mean |change in points|. A rerun fixed at the
fitted prior must reproduce production's points (`baseline_agree`); datasets where it does not are flagged.

    python -m analysis.calibration_quality.prior_impact [--out DIR] [--n-jobs 5] [--inner-jobs 8] [--datasets ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _calibrate(dataset, cal, scoreset, fits, seeds, prior, inner_jobs):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    n_c = str(cal["n_c"]).split("_")[0]
    with tempfile.TemporaryDirectory() as tmp:
        pc = PipelineConfig(dataset_csv="", dataset_name=dataset, output_dir=tmp, components=[int(n_c[0])],
                            use_median_prior=True, use_2c_equation=False, benign_method=cal["benign_method"],
                            liberal_monotonicity=bool(cal.get("liberal_monotonicity", 1)),
                            clinvar_release="2018" if cal.get("clinvar_2018") else "2025",
                            point_values=[1, 2, 3, 4, 5, 6, 7, 8], n_jobs=inner_jobs, manual_prior=float(prior))
        lg = logging.getLogger(f"impact.{dataset}")
        lg.setLevel(logging.ERROR)
        return process_component_fits(fits=fits, scoreset=scoreset, config=pc, n_c=int(n_c[0]), logger=lg,
                                      bootstrap_seeds=np.asarray(seeds))


def impact(dataset, cal, kept_seeds, priors, inner_jobs=8):
    """`priors`: {"typ": (lo, hi), "q90": (lo, hi)}."""
    from analysis import config as cfg
    from analysis.legacy_fits import load_scoreset_and_fits
    from src.assay_calibration.fit_utils.fit import assign_points
    n_c, _, file_bm = str(cal["n_c"]).partition("_")
    scoreset, _, fits, _, _, _, _ = load_scoreset_and_fits(dataset, n_c=n_c, benign_method=file_bm or cal["benign_method"])
    keep = set(int(s) for s in kept_seeds)
    pairs = [(f, s) for f, s in zip(fits, fits.seeds) if int(s) in keep]
    fk, sk = [f for f, _ in pairs], [s for _, s in pairs]
    v = pd.read_csv(Path(cfg.OUTPUT_DIR) / dataset / f"{dataset}_{cal['n_c']}_variants.csv",
                    usecols=["score", "sample", "standard_points"])
    g = v[v["sample"].fillna("").astype(str).str.contains("population", regex=False)]
    x, ref = g["score"].to_numpy(float), g["standard_points"].fillna(0).to_numpy(int)

    def pts(prior):
        c = _calibrate(dataset, cal, scoreset, fk, sk, float(np.clip(prior, 0.001, 0.999)), inner_jobs)
        return None if c is None else np.asarray(assign_points(x, {int(k): r for k, r in c["point_ranges"].items()}),
                                                 int)
    base = pts(cal["prior"])
    out = dict(dataset=dataset, n_gnomad=int(len(x)), share_with_evidence=float((ref != 0).mean()),
               baseline_agree=float((base == ref).mean()) if base is not None else np.nan)
    ev = ref != 0
    for kind, (lo, hi) in priors.items():
        res = {tag: pts(p) for tag, p in (("lo", lo), ("hi", hi))}
        for tag, p in res.items():
            out[f"{kind}_prior_{tag}"] = float(np.clip(lo if tag == "lo" else hi, 0.001, 0.999))
            out[f"{kind}_unchanged_{tag}"] = float((p == ref).mean()) if p is not None else np.nan
            out[f"{kind}_mean_abs_change_{tag}"] = float(np.abs(p - ref).mean()) if p is not None else np.nan
        both = (res["lo"] == ref) & (res["hi"] == ref) if all(p is not None for p in res.values()) else None
        out[f"{kind}_unchanged"] = float(both.mean()) if both is not None else np.nan
        out[f"{kind}_unchanged_among_evidence"] = float(both[ev].mean()) if both is not None and ev.any() else np.nan
    return out


def main():
    from joblib import Parallel, delayed
    from analysis import config as cfg
    from analysis.calibration_quality.diagnostics import real_datasets
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(cfg.CALIBRATION_QUALITY_DIR).parent / "real_diagnostics"))
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--inner-jobs", type=int, default=8)
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    tab = pd.read_csv(out / "real_diagnostics.csv")
    rp = json.loads((out / "recomputed_priors.json").read_text())
    cals = real_datasets()
    todo = [r for _, r in tab.iterrows() if (not a.datasets or r.dataset in a.datasets)
            and np.isfinite(r.prior_lo_q90) and rp.get(r.dataset, {}).get("match")]

    def one(r):
        try:
            th, em = np.arcsin(np.sqrt(r.prior)), r.ang_err_mean
            typ = (np.sin(max(th - em, 0)) ** 2, np.sin(min(th + em, np.pi / 2)) ** 2)
            return impact(r.dataset, cals[r.dataset], rp[r.dataset]["seeds"],
                          {"typ": typ, "q90": (r.prior_lo_q90, r.prior_hi_q90)}, a.inner_jobs)
        except Exception as e:
            return dict(dataset=r.dataset, error=f"{type(e).__name__}: {e}"[:300])
    res = pd.DataFrame(Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(r) for r in todo))
    path = out / "prior_impact.csv"
    if a.datasets and path.exists():
        old = pd.read_csv(path)
        res = pd.concat([old[~old.dataset.isin(res.dataset)], res], ignore_index=True)
    res.to_csv(path, index=False)
    print(res.drop(columns=[c for c in ("error",) if c in res and res[c].isna().all()]).round(3).to_string(index=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
