"""Pooling criteria per dataset from strata fits: per-consequence gamma_hat, delta-gamma, TV_D and benign-side TV.

For every bootstrap fit of one (fit label, dataset), with the dataset's production benign reference f_N (B/LB and/or
synonymous composites of the pooled arm, blended per the production benign_method):

    gamma_c   P(M=1 | Y=1) of the P/LP of consequence c (missense, PTV, ..., "all" = pooled P/LP), the production
              subtraction_grid estimator: subtraction when the benign reference has >= REF_MIN_N variants, else the
              min-ratio over a grid spanning the reference sample's scores (point_ranges.resolve_subtraction_grid).
              One reference and one estimator choice for every consequence, so delta-gamma compares like with like.
    d_gamma   gamma_missense - gamma_PTV (< 0: missense detected less often -> pooled benign evidence for P/LP
              missense; > 0: PTVs detected less often, e.g. dominant-negative readouts)
    tv_D      1/2 int |f_D,PTV - f_D,missense|, f_D,c unmixed from f_P,c with the same gamma_c (_unmix_density)
    tv_P      1/2 int |f_P,PTV - f_P,missense| (raw P/LP densities, no unmixing)
    tv_B      1/2 int |f_B/LB missense - f_N| (benign-side difference; CanVIG Rec 6 / Box 2D)

Reported as bootstrap medians with 5/95% quantiles (descriptive only: the bootstrap interval undercovers, so flag
rules use the medians; thresholds are provisional until set in simulation).

    python -m csq_analysis.criteria run <fit_dir> --labels S1_m5_R20 [--workers 16] [--datasets ...]
writes <fit_dir>/criteria/<label>.csv (one row per dataset).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csq_analysis import strata_uni as U

REF_MIN_N = 10
MIN_N = 5                    # P/LP of a consequence needed for its gamma_c
GRID_N = 2001
CONSEQUENCES = ("missense", "PTV", "synonymous", "splice", "intronic")
DG_FLAG, TV_FLAG = 0.1, 0.2  # provisional (set in simulation S-E)


def _mix(x, params, w):
    from src.assay_calibration.fit_utils.cfusn.density_utils import joint_densities
    return joint_densities(np.asarray(x, float), params, np.asarray(w, float)).sum(axis=0)


def _comp(us, role):
    """(S,) stratum counts of the strata whose rows satisfy role(f) (all-or-nothing by construction)."""
    fit = us.fitted
    rep = us.rows[fit].groupby(us.stratum[fit]).head(1)
    st = us.stratum[fit][us.rows[fit].index.get_indexer(rep.index)]
    m = np.zeros(len(us.names))
    for f, c in zip(rep.to_dict("records"), st):
        if role(f):
            m[int(c)] = us.counts[int(c)]
    return m


def _is_plp(c):
    return lambda f: bool(f["plp"]) and not f["blb"] and (c == "all" or f["csq"] == c)


def _blb_mis(f):
    return bool(f["blb"]) and not f["plp"] and f["csq"] == "missense"


def setup(us, benign_method):
    """Everything per dataset that does not depend on the bootstrap fit."""
    from src.assay_calibration.fit_utils.point_ranges import resolve_subtraction_grid
    ss, _, roles = U.arm_scoreset(us, "pool")
    M = us.composition("pool")
    bi = roles.index(1) if 1 in roles else None
    si = roles.index(3) if 3 in roles else None
    gi = roles.index(2) if 2 in roles else None
    bm = benign_method
    if bm == "avg" and si is None:
        bm = "benign"
    if bm in ("avg", "benign") and bi is None and si is not None:
        bm = "synonymous"
    if bm == "synonymous" and si is None and bi is not None:
        bm = "benign"
    method, pts, n_ref = resolve_subtraction_grid(ss.scores, ss.sample_assignments, bm, bi, si, ref_min_n=REF_MIN_N)
    r = us.rows
    sc = r.score.to_numpy(float)
    fitted = us.fitted
    pop = sc[fitted & r["pop"].to_numpy()]
    cs = {}
    for c in ("all",) + CONSEQUENCES:
        sel = fitted & r.plp.to_numpy() & ~r.blb.to_numpy() & ((r.csq == c).to_numpy() if c != "all" else True)
        if sel.sum() >= MIN_N:
            cs[c] = dict(m=_comp(us, _is_plp(c)), scores=sc[sel], n=int(sel.sum()))
    mb = _comp(us, _blb_mis)
    return dict(M=M, bi=1 if bi is not None else None, si=3 if si is not None else None, bm=bm, method=method,
                pts=pts, n_ref=n_ref, pop=pop, cs=cs, m_blb_mis=mb, n_blb_mis=int(mb.sum()))


def one_fit(fit, S):
    from src.assay_calibration.fit_utils.point_ranges import (_PATHOMECHANISM_MATCH_FRAC_FLOOR,
                                                              _get_benign_reference_weights, _unmix_density,
                                                              apply_pathomechanism_correction)
    from src.assay_calibration.fit_utils.strata import compose_weights
    params = fit["component_params"]
    W = np.asarray(fit["weights"], float)
    lo, hi = fit["xlims"]
    grid = np.linspace(lo, hi, GRID_N)
    wc = compose_weights(W, S["M"])                     # rows P, B, g, s of the pooled arm
    w_N = np.asarray(_get_benign_reference_weights([wc[0], wc[1], wc[2], wc[3]], S["bm"], S["bi"], S["si"]), float)
    fN = _mix(grid, params, w_N)
    out, fD, fP = {}, {}, {}
    for c, d in S["cs"].items():
        wP = compose_weights(W, d["m"][None, :])[0]
        lab = S["pts"] if S["method"] == "boundary" else d["scores"]
        try:
            _, g = apply_pathomechanism_correction(S["method"], np.asarray(lab, float), S["pop"], params, wP, w_N)
        except Exception:
            g = np.nan
        g = float(g) if g is not None and np.isfinite(g) else np.nan
        out[f"gamma_{c}"] = g
        p = _mix(grid, params, wP)
        fP[c] = p / np.trapz(p, grid)
        if np.isfinite(g) and _PATHOMECHANISM_MATCH_FRAC_FLOOR <= g <= 1:
            f = np.clip(_unmix_density(p, 1 - g, fN) if g < 1 else p, 0, None)
            z = np.trapz(f, grid)
            fD[c] = f / z if z > 0 else None
    tv = lambda a, b: 0.5 * float(np.trapz(np.abs(a - b), grid)) if (a is not None and b is not None) else np.nan  # noqa: E731
    out["tv_D"] = tv(fD.get("PTV"), fD.get("missense"))
    out["tv_P"] = tv(fP.get("PTV"), fP.get("missense"))
    if S["n_blb_mis"] >= MIN_N:
        b = _mix(grid, params, compose_weights(W, S["m_blb_mis"][None, :])[0])
        out["tv_B"] = tv(b / np.trapz(b, grid), fN / np.trapz(fN, grid))
    else:
        out["tv_B"] = np.nan
    out["d_gamma"] = out.get("gamma_missense", np.nan) - out.get("gamma_PTV", np.nan)
    return out


def summarise(rows):
    df = pd.DataFrame(rows)
    r = {"n_boot": len(df)}
    for k in df.columns:
        a = df[k].to_numpy(float)
        a = a[np.isfinite(a)]
        r[f"{k}_med"] = float(np.median(a)) if len(a) else np.nan
        r[f"{k}_q05"] = float(np.quantile(a, 0.05)) if len(a) else np.nan
        r[f"{k}_q95"] = float(np.quantile(a, 0.95)) if len(a) else np.nan
    return r


def flags(r, dg=DG_FLAG, tv=TV_FLAG):
    """Provisional pooling flags on bootstrap medians (thresholds to be set in simulation)."""
    d, t = r.get("d_gamma_med", np.nan), r.get("tv_D_med", np.nan)
    return dict(flag_dgamma=bool(np.isfinite(d) and abs(d) > dg), flag_tvD=bool(np.isfinite(t) and t > tv),
                flag=bool((np.isfinite(d) and abs(d) > dg) or (np.isfinite(t) and t > tv)),
                testable=bool(np.isfinite(d) or np.isfinite(t)))


def dataset_criteria(fit_dir, lab, ds):
    from csq_analysis.calibrate import production_settings
    from csq_analysis.fits import FIT_CONFIGS, config_of, layout, load_fits, sub_of
    records, meta = load_fits(fit_dir, lab, ds)
    if not records:
        return dict(dataset=ds, label=lab, status="no fits")
    us, _ = layout(fit_dir, ds, FIT_CONFIGS[config_of(lab)], sub_of(lab))
    assert us.names == meta["names"], f"{ds}/{lab}: strata changed since the fits were made"
    st = production_settings(ds)
    S = setup(us, st["benign_method"])
    K = st["K"]
    per = [one_fit(rec[f"{K}c"]["fit"], S) for _, rec in sorted(records.items(), key=lambda kv: int(kv[0]))
           if rec and f"{K}c" in rec and "fit" in rec[f"{K}c"]]
    if not per:
        return dict(dataset=ds, label=lab, status="no fits")
    r = dict(dataset=ds, label=lab, status="ok", K=K, benign_method=S["bm"], estimator=S["method"],
             ref_n=S["n_ref"] if np.isfinite(S["n_ref"]) else None, n_blb_mis=S["n_blb_mis"],
             **{f"n_plp_{c}": d["n"] for c, d in S["cs"].items()}, **summarise(per))
    return r | flags(r)


def run(fit_dir, labels, workers=16, datasets=None):
    from joblib import Parallel, delayed
    bundle = json.loads((Path(fit_dir) / "bundle.json").read_text())
    dss = datasets or bundle["datasets"]
    out = Path(fit_dir) / "criteria"
    out.mkdir(exist_ok=True)
    for lab in labels:
        rows = Parallel(n_jobs=workers, verbose=2)(delayed(_safe)(fit_dir, lab, d) for d in dss)
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{lab}.csv", index=False)
        print(f"{lab}: {int((df.status == 'ok').sum())}/{len(df)} datasets -> {out / f'{lab}.csv'}")


def _safe(fit_dir, lab, ds):
    try:
        return dataset_criteria(fit_dir, lab, ds)
    except Exception as e:
        return dict(dataset=ds, label=lab, status=f"error: {repr(e)[:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("fit_dir")
    r.add_argument("--labels", required=True)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--datasets", default=None)
    a = ap.parse_args()
    run(a.fit_dir, a.labels.split(","), a.workers, a.datasets.split(",") if a.datasets else None)


if __name__ == "__main__":
    main()
