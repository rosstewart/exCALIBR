"""Pilot comparisons for the decision checkpoint (in-bag; HPC pilot 100 bootstraps, R20 unless stated).

    configs     per fit config (bootstrap scheme S0/S1/S2, min stratum): controls by consequence under the main
                strategies, with the standard refits alongside (metrics summaries)
    stability   per fit config and arm: bootstrap prior spread, share of bootstraps kept, valid calibrations, and the
                pooled strata prior vs the standard pooled refit's prior (strata ~ ExCALIBR check)
    restarts    12-dataset panel at R8 / R20 / R100: best log-likelihood per bootstrap seed (same seeds, so a pure
                restart effect), prior and evidence changes
    downsample  missense controls subsampled to n (5 reps, 15 datasets, B50 x R8): held-out vs retained missense
                controls by n, mis arms vs pool
    flags       criteria flags per fit config

    python -m csq_analysis.pilot all <fit_dir> [--std <calib_std_dir>]
writes <fit_dir>/pilot/*.csv and prints the headline tables.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

R20 = ["S1_m5_R20", "S0_m5_R20", "S2n3_m1_R20", "S2n5_m1_R20", "S2n10_m1_R20", "S1_m1_R20", "S1_m10_R20"]
RESTARTS = ["S1_m5_R8", "S1_m5_R20", "S1_m5_R100"]
MAPPINGS = ("tavtigian", "lp_anchored")
MAIN = ["pool", "mis:mis", "mis:mis_Ball", "mis:mis_Bmis_syn", "routed", "mis:mis|flag", "routed|flag"]


def _cal(fit_dir, lab, arm, mapping, ds):
    f = Path(fit_dir) / "calib" / lab / arm / mapping / f"{ds}.json"
    return json.loads(f.read_text()) if f.exists() else None


def configs(fit_dir, min_n=3):
    out = []
    for lab in R20:
        for m in MAPPINGS:
            f = Path(fit_dir) / "metrics" / f"summary_{lab}_{m}_min{min_n}_inbag.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f)
            out.append(s.assign(label=lab, mapping=m))
    for m in MAPPINGS:
        f = Path(fit_dir) / "metrics" / f"summary_std_{m}_min{min_n}_inbag.csv"
        if f.exists():
            out.append(pd.read_csv(f).assign(label="std", mapping=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _std_prior(std_dir, ds, mapping="tavtigian"):
    f = glob.glob(f"{std_dir}/std_pool_{mapping}/{ds}/*_calibration.json")
    return json.load(open(f[0])).get("prior") if f else None


def stability(fit_dir, std_dir=None, arms=("pool", "mis", "mis_Ball", "ptv_gptv", "ptv_poolprior")):
    rows = []
    dss = json.loads((Path(fit_dir) / "bundle.json").read_text())["datasets"]
    for lab in R20:
        for arm in arms:
            for ds in dss:
                c = _cal(fit_dir, lab, arm, "tavtigian", ds)
                if c is None:
                    continue
                r = dict(label=lab, arm=arm, dataset=ds, status=c.get("status"))
                if c.get("status") == "ok":
                    r.update(prior=c["prior"], width=(c["prior_p95"] or np.nan) - (c["prior_p5"] or np.nan),
                             kept=c.get("pct_bootstraps_kept"), n_fits=c.get("n_fits"),
                             n_pathogenic_levels=sum(1 for k, v in c["point_ranges"].items() if int(k) > 0 and v),
                             n_benign_levels=sum(1 for k, v in c["point_ranges"].items() if int(k) < 0 and v))
                    if arm == "pool" and std_dir:
                        sp = _std_prior(std_dir, ds)
                        r["std_pool_prior"] = sp
                rows.append(r)
    t = pd.DataFrame(rows)
    ok = t[t.status.eq("ok")]
    g = ok.groupby(["label", "arm"]).agg(n_ok=("dataset", "size"), prior_width_med=("width", "median"),
                                         prior_width_q90=("width", lambda x: x.quantile(0.9)),
                                         kept_med=("kept", "median"), kept_min=("kept", "min"))
    g["n_failed"] = t[~t.status.eq("ok")].groupby(["label", "arm"]).size().reindex(g.index).fillna(0).astype(int)
    if std_dir:
        p = ok[ok.arm.eq("pool")].dropna(subset=["std_pool_prior"])
        d = (p.prior - p.std_pool_prior).abs().groupby(p.label)
        g = g.join(pd.DataFrame({"pool_vs_std_prior_absdiff_med": d.median(),
                                 "pool_vs_std_prior_absdiff_q90": d.quantile(0.9)}).assign(arm="pool")
                   .set_index("arm", append=True), how="left")
    return t, g.reset_index()


def restarts(fit_dir):
    from csq_analysis.fits import load_fits
    rows = []
    for ds in sorted({Path(f).name.split("__")[0] for f in glob.glob(f"{fit_dir}/fits/S1_m5_R100/*.json.gz")}):
        ll = {}
        for lab in RESTARTS:
            try:
                recs, _ = load_fits(fit_dir, lab, ds)
            except FileNotFoundError:
                continue
            ll[lab] = {int(s): next((v["val_ll"] for v in r.values() if isinstance(v, dict) and "val_ll" in v), np.nan)
                       for s, r in recs.items() if r and "error" not in r}
        common = sorted(set.intersection(*[set(v) for v in ll.values()])) if len(ll) == 3 else []
        if not common:
            continue
        a = {lab: np.array([ll[lab][s] for s in common]) for lab in ll}
        r = dict(dataset=ds, n_seeds=len(common))
        for lo in ("S1_m5_R8", "S1_m5_R20"):
            d = a["S1_m5_R100"] - a[lo]
            r[f"{lo[-3:].strip('_')}_ll_gap_med"] = float(np.median(d))
            r[f"{lo[-3:].strip('_')}_ll_gap_max"] = float(np.max(d))
            r[f"{lo[-3:].strip('_')}_share_within_0.5"] = float(np.mean(d < 0.5))
        for arm in ("pool", "mis"):
            for lab in RESTARTS:
                c = _cal(fit_dir, lab, arm, "tavtigian", ds)
                r[f"{arm}_prior_{lab.split('_')[-1]}"] = c.get("prior") if c and c.get("status") == "ok" else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def restart_points(fit_dir, min_n=3):
    """Controls of the 12 panel datasets under pool and mis:mis at R8 / R20 / R100 (metrics points tables)."""
    out = []
    for lab in RESTARTS:
        for m in MAPPINGS:
            f = Path(fit_dir) / "metrics" / f"points_{lab}_{m}_min{min_n}_inbag.pkl"
            if f.exists():
                out.append((lab, m, pd.read_pickle(f)))
    if not out:
        return pd.DataFrame()
    panel = set.intersection(*[set(t.dataset) for _, _, t in out])
    rows = []
    for lab, m, t in out:
        t = t[t.dataset.isin(panel) & (t.plp ^ t.blb)]
        for s in ("pool", "mis:mis", "routed"):
            if s not in t:
                continue
            for cs, sel in (("all", np.ones(len(t), bool)), ("missense", t.csq.eq("missense").to_numpy())):
                x = t[sel]
                rows.append(dict(label=lab, mapping=m, strategy=s, controls=cs, n_datasets=len(panel),
                                 TP=int((x.plp & (x[s] > 0)).sum()), FN=int((x.plp & (x[s] < 0)).sum()),
                                 FP=int((x.blb & (x[s] > 0)).sum()), TN=int((x.blb & (x[s] < 0)).sum())))
    return pd.DataFrame(rows)


def downsample(fit_dir, min_n=1):
    """Held-out (label removed before fitting) and retained missense controls, per (n, rep, dataset, arm)."""
    from csq_analysis.downsample import PANEL, parse_tag, subsample
    from csq_analysis.fits import ROWS_FILE
    from csq_analysis.metrics import points
    rows = pd.read_pickle(Path(fit_dir) / ROWS_FILE)
    rows = rows[rows.dataset.isin(PANEL)]
    strats = {"pool": {}, "mis:mis": {"missense": "mis"}, "mis:mis_Ball": {"missense": "mis_Ball"},
              "mis:mis_Bmis_syn": {"missense": "mis_Bmis_syn"}, "mis:mis_gall": {"missense": "mis_gall"}}
    out = []
    labs = sorted(Path(p).name for p in glob.glob(f"{fit_dir}/calib/S1_m5_R8__n*"))
    for lab in labs:
        sub = parse_tag(lab.split("__")[1])
        held = pd.concat([subsample(r, ds, sub["n"], sub["rep"])[["heldout_plp", "heldout_blb"]]
                          for ds, r in rows.groupby("dataset", sort=False)])
        r2 = rows.assign(heldout=(held.heldout_plp | held.heldout_blb).reindex(rows.index).fillna(False))
        for m in MAPPINGS:
            # original labels: retained controls keep theirs, held-out controls are scored with the label the fit
            # never saw; min_n=1 so a mis arm scores missense whenever its calibration succeeded
            t = points(fit_dir, lab, m, rows.drop(columns=[c for c in ("heldout",) if c in rows]), min_n, strats)
            if t.empty:
                continue
            h = r2[(r2.plp ^ r2.blb) | r2.vus].groupby("dataset", sort=False).heldout.apply(list)
            t["heldout"] = np.concatenate([h[d] for d in t.dataset.unique()])
            x = t[t.csq.eq("missense") & (t.plp ^ t.blb)]
            for (ds, ho), g in x.groupby(["dataset", "heldout"]):
                for s in strats:
                    out.append(dict(label=lab, n=sub["n"], rep=sub["rep"], mapping=m, dataset=ds,
                                    controls="held-out" if ho else "retained", strategy=s,
                                    n_plp=int(g.plp.sum()), n_blb=int(g.blb.sum()),
                                    TP=int((g.plp & (g[s] > 0)).sum()), FN=int((g.plp & (g[s] < 0)).sum()),
                                    FP=int((g.blb & (g[s] > 0)).sum()), TN=int((g.blb & (g[s] < 0)).sum())))
    return pd.DataFrame(out)


def flags(fit_dir):
    out = []
    for lab in R20:
        f = Path(fit_dir) / "criteria" / f"{lab}.csv"
        if f.exists():
            c = pd.read_csv(f)
            out.append(c.assign(label=lab))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "configs", "stability", "restarts", "downsample", "flags"])
    ap.add_argument("fit_dir")
    ap.add_argument("--std", default=None)
    a = ap.parse_args()
    out = Path(a.fit_dir) / "pilot"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    todo = ["configs", "stability", "restarts", "downsample", "flags"] if a.cmd == "all" else [a.cmd]
    if "configs" in todo:
        c = configs(a.fit_dir)
        c.to_csv(out / "configs.csv", index=False)
        if not c.empty:
            v = c[c.controls.isin(["all", "missense", "PTV"]) & c.strategy.isin(MAIN + ["mis:std_misctl", "mis:std_mis"])]
            print(v[["mapping", "label", "controls", "strategy", "TP", "FN", "FP", "TN", "mcc", "determinate"]]
                  .round(3).to_string(index=False))
    if "stability" in todo:
        t, g = stability(a.fit_dir, a.std)
        t.to_csv(out / "stability_by_dataset.csv", index=False)
        g.to_csv(out / "stability.csv", index=False)
        print(g.round(3).to_string(index=False))
    if "restarts" in todo:
        r = restarts(a.fit_dir)
        r.to_csv(out / "restarts.csv", index=False)
        print(r.round(3).to_string(index=False))
        rp = restart_points(a.fit_dir)
        rp.to_csv(out / "restart_points.csv", index=False)
        print(rp.to_string(index=False))
    if "downsample" in todo:
        d = downsample(a.fit_dir)
        d.to_csv(out / "downsample.csv", index=False)
        if not d.empty:
            s = d.groupby(["mapping", "controls", "n", "strategy"])[["n_plp", "n_blb", "TP", "FN", "FP", "TN"]].sum()
            print(s.to_string())
    if "flags" in todo:
        f = flags(a.fit_dir)
        f.to_csv(out / "flags.csv", index=False)
        if not f.empty:
            print(f[f.flag.astype(bool)].groupby("label").dataset.apply(lambda x: ", ".join(sorted(x))).to_string())


if __name__ == "__main__":
    main()
