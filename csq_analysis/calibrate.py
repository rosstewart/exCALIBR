"""Calibrate paper arms from strata fits: compose -> production process_component_fits, under both mappings.

For each (fit config, dataset, arm, mapping) writes calib/<config>/<arm>/<mapping>/<dataset>.json with the prior
(median and 5/95%), point ranges, the gamma_hat diagnostic, the composite sizes and the fits kept. In-bag only here
by default (pilots); --oob adds every control's out-of-bag points (production _compute_oob_evidence on the fitted
rows with the fits' stored splits), stored under "oob" in the same JSON.

Per-dataset settings follow the production calibration: K (dataset config), benign_method (production, unless the
arm fixes it), liberal_monotonicity; mappings tavtigian (clinician-facing) and lp_anchored (internal).

    python -m csq_analysis.calibrate run <fit_dir> --configs S1_m5 --arms pool,mis [--mappings tavtigian,lp_anchored]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np

from csq_analysis import strata_uni as U

MAPPINGS = ("tavtigian", "lp_anchored")


def production_settings(ds):
    from analysis import config as acfg
    from csq_analysis import config as C
    K, bm = C.dataset_k(ds)
    f = glob.glob(f"{acfg.OUTPUT_DIR}/{ds}/*_calibration.json")
    cal = json.load(open(f[0])) if f else {}
    return dict(K=K, benign_method=cal.get("benign_method", bm),
                liberal_monotonicity=bool(cal.get("liberal_monotonicity", 1)),
                clinvar_release="2018" if (cal.get("clinvar_2018") or ds.endswith("_clinvar_2018")) else "2025")


def _benign_method(arm, roles, prod_bm):
    bm = U.ARMS[arm][1] or prod_bm
    has_b, has_s = 1 in roles, 3 in roles
    if bm == "avg" and not has_s:
        bm = "benign"
    if bm == "avg" and not has_b:
        bm = "synonymous"
    if bm == "benign" and not has_b and has_s:
        bm = "synonymous"
    if bm == "synonymous" and not has_s and has_b:
        bm = "benign"
    return bm


def _oob(us, res, records, K, cfg, lg, kw=None):
    """Out-of-bag points of every control row (P/LP xor B/LB) under this arm's calibration.

    Fit records store their OOB rows as positions among the FITTED rows (us.fitted), so production
    _compute_oob_evidence runs unchanged on a scoreset of the fitted rows, with the splits restricted to control
    rows. OOB-ness is a property of the fit, not the arm: every arm composed from one fit shares its splits, so a
    control is held out of the shared components and of every composite in the same bootstraps.
    Returns {"rows": positions in us.rows, "points": OOB points, "n_oob": OOB bootstraps used}."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.pipeline.variant_evidence import _compute_oob_evidence
    fit_rows = np.flatnonzero(us.fitted)
    r = us.rows.iloc[fit_rows]
    ctrl = (r.plp ^ r.blb).to_numpy()
    fss = BasicScoreset(r.score.to_numpy(float), us.onehot, sample_names=list(us.names))
    splits = {}
    for s, rec in records.items():
        x = (rec or {}).get(f"{K}c") or {}
        if "fit" in x:
            val = x.get("val_variant_indices")
            if not val:
                from csq_analysis.fits import regenerate_val
                val = regenerate_val(us, kw or {}, K, s)
            splits[int(s)] = {"val_variant_indices": [int(v) for v in val if ctrl[v]]}
    out = _compute_oob_evidence(fss, res, splits, cfg, logger=lg)
    pos = {int(k.rsplit("_", 1)[1]): v for k, v in out.items()}
    keys = sorted(pos)
    return dict(rows=[int(fit_rows[k]) for k in keys], points=[int(pos[k]["points"]) for k in keys],
                n_oob=[int(pos[k]["n_oob"]) for k in keys])


def calibrate_arm(us, records, arm, mapping, settings, manual_prior=None, n_jobs=8, oob=False, kw=None):
    """records: {seed: {"<K>c": record}} from csq_analysis.fits. Returns a compact calibration dict, plus the
    controls' out-of-bag points when oob (fits must carry val_variant_indices)."""
    import tempfile
    from src.assay_calibration.fit_utils.strata import compose_weights
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    K = settings["K"]
    ss, idx, roles = U.arm_scoreset(us, arm)
    if 0 not in roles:
        return dict(status="no P/LP composite")
    M = us.composition(arm)[roles]
    fits, seeds = [], []
    for s, rec in sorted(records.items(), key=lambda kv: int(kv[0])):
        r = rec.get(f"{K}c") if rec else None
        if not r or "fit" not in r:
            continue
        W = np.asarray(r["fit"]["weights"], float)
        fit = {k: v for k, v in r["fit"].items() if k not in ("weights", "val_variant_indices", "train_variant_indices")}
        fit["weights"] = compose_weights(W, M)
        fits.append({"fit": fit})
        seeds.append(int(s))
    if not fits:
        return dict(status="no fits")
    bm = _benign_method(arm, roles, settings["benign_method"])
    lg = logging.getLogger(f"csq.{us.dataset}.{arm}")
    lg.setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(dataset_csv="", dataset_name=us.dataset, output_dir=tmp, components=[K], benign_method=bm,
                             use_median_prior=True, use_2c_equation=False, point_values=list(range(1, 9)),
                             n_jobs=n_jobs, acmg_mapping_method=mapping, manual_prior=manual_prior,
                             liberal_monotonicity=settings["liberal_monotonicity"],
                             clinvar_release=settings["clinvar_release"],
                             keep_extreme_prior_fits=bool(U.arm_options(arm).get("keep_extreme")))
        res = process_component_fits(fits=fits, scoreset=ss, config=cfg, n_c=K, logger=lg,
                                     bootstrap_seeds=np.asarray(seeds))
        oob_pts = _oob(us, res, records, K, cfg, lg, kw) if (oob and res is not None) else None
    if res is None:
        return dict(status="no valid priors", dataset=us.dataset, arm=arm, mapping=mapping, n_fits=len(fits))
    pri = np.asarray(res.get("priors"), float) if res.get("priors") is not None else np.array([])
    pri = pri[np.isfinite(pri)]
    return dict(status="ok", dataset=us.dataset, arm=arm, mapping=mapping, K=K, benign_method=bm,
                manual_prior=manual_prior,
                n_composite={U.ROLES[i]: int(M[j].sum()) for j, i in enumerate(roles)},
                prior=float(res["prior"]), prior_p5=float(np.percentile(pri, 5)) if len(pri) else None,
                prior_p95=float(np.percentile(pri, 95)) if len(pri) else None,
                frac_prior_extreme=float(np.mean((pri <= 1e-3) | (pri >= 1 - 1e-3))) if len(pri) else None,
                point_ranges={str(k): v for k, v in res["point_ranges"].items()},
                pct_bootstraps_kept=res.get("pct_bootstraps_kept"), n_fits=len(fits),
                **{k: res.get(k) for k in res if str(k).startswith("gamma_hat_diagnostic")},
                **({"oob": oob_pts} if oob_pts is not None else {}))


def _one(fit_dir, cfgname, ds, arms, mappings, n_jobs, oob=False):
    from csq_analysis.fits import load_fits, layout, FIT_CONFIGS, config_of, sub_of
    out = Path(fit_dir) / "calib" / cfgname
    def done(a, m):
        f = out / a / m / f"{ds}.json"
        if not f.exists():
            return False
        r = json.loads(f.read_text())
        return not oob or "oob" in r or r.get("status") != "ok"      # an in-bag-only file is redone for --oob
    todo = [(a, m) for a in arms for m in mappings if not done(a, m)]
    if not todo:
        return
    records, meta = load_fits(fit_dir, cfgname, ds)
    us, kw = layout(fit_dir, ds, FIT_CONFIGS[config_of(cfgname)], sub_of(cfgname))
    assert us.names == meta["names"], f"{ds}/{cfgname}: strata changed since the fits were made"
    st = production_settings(ds)
    # arms whose prior is fixed at another arm's prior run after it
    todo.sort(key=lambda am: bool(U.arm_options(am[0]).get("manual_prior_from")))
    for a, m in todo:
        src = U.arm_options(a).get("manual_prior_from")
        mp = None
        if src:
            base = load(fit_dir, cfgname, src, m, ds)
            if base is None:
                base = calibrate_arm(us, records, src, m, st, n_jobs=n_jobs, oob=oob, kw=kw)
                (out / src / m).mkdir(parents=True, exist_ok=True)
                (out / src / m / f"{ds}.json").write_text(json.dumps(base))
            mp = base.get("prior") if base.get("status") == "ok" else None
        try:
            r = (calibrate_arm(us, records, a, m, st, manual_prior=mp, n_jobs=n_jobs, oob=oob, kw=kw) if (mp or not src)
                 else dict(status=f"no {src} prior", dataset=ds, arm=a, mapping=m))
        except Exception as e:
            r = dict(status=f"error: {repr(e)[:300]}", dataset=ds, arm=a, mapping=m)
        f = out / a / m / f"{ds}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r))


def run(fit_dir, configs, arms, mappings, workers=8, inner=8, datasets=None, oob=False):
    from joblib import Parallel, delayed
    bundle = json.loads((Path(fit_dir) / "bundle.json").read_text())
    dss = datasets or bundle["datasets"]
    configs = [c for c in configs if any((Path(fit_dir) / "fits" / c).glob("*.json.gz"))] or configs
    jobs = [(c, d) for c in configs for d in dss]
    Parallel(n_jobs=workers, verbose=5)(delayed(_one)(fit_dir, c, d, arms, mappings, inner, oob) for c, d in jobs)


def load(fit_dir, cfgname, arm, mapping, ds):
    f = Path(fit_dir) / "calib" / cfgname / arm / mapping / f"{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("fit_dir")
    r.add_argument("--configs", default="S1_m5")
    r.add_argument("--arms", default=",".join(U.ARMS))
    r.add_argument("--mappings", default=",".join(MAPPINGS))
    r.add_argument("--datasets", default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--inner", type=int, default=8)
    r.add_argument("--oob", action="store_true", help="also out-of-bag points of every control (final runs)")
    a = ap.parse_args()
    run(a.fit_dir, a.configs.split(","), a.arms.split(","), a.mappings.split(","), a.workers, a.inner,
        a.datasets.split(",") if a.datasets else None, a.oob)


if __name__ == "__main__":
    main()
