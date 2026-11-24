"""Missense-only calibration sensitivity: compare three large-preset runs.

Runs (same dataframe, configs, seeds and pillar-project settings):
  baseline     -- production samples (P/LP and B/LB of every variant type,
                  production population sample and benign reference).
  missense     -- P/LP and B/LB restricted to missense, gnomAD missense
                  population, synonymous sample kept in the fit, benign
                  reference forced to missense B/LB (PU mode if none).
  missense_ptv -- missense + a fifth sample of all assayed PTVs (stop_gained
                  of any length + frameshift) used only to fit the shared
                  components, never in lr+ or the prior.
  {missense,missense_ptv}_basecfg -- the same fits as missense / missense_ptv,
                  calibrated with the baseline per-dataset benign_method
                  (avg / synonymous / benign) and the pipeline's normal
                  fallback (no B/LB -> synonymous) instead of a forced B/LB
                  reference.

Every run is scored on the SAME variants: missense P/LP and B/LB controls
(out-of-bootstrap points, in-bag fallback where a control was never OOB),
and missense VUS. The missense flag comes from a default-kwargs Scoreset
(group counts as missense if any member row is missense, the same gate
Scoreset(missense_controls_only=True) uses).

Usage:
  python -m analysis.missense_sensitivity.compare_runs \
      --root /data/ross/assay_calibration/missense_only_calibrations \
      --out  /data/ross/assay_calibration/missense_only_calibrations/analysis \
      [--runs baseline missense missense_ptv missense_basecfg missense_ptv_basecfg]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
import pandas as pd

RUNS = ["baseline", "missense", "missense_ptv"]
# Synthetic runs: pathogenic points from the first calib dir, benign points
# from the second (per variant, in-bag and OOB separately).
COMBINED = {"baseline_plpmis": ("baseline_plpmis_raw", "baseline"),
            "baseline_plpmis_v2": ("baseline_plpmis_v2_raw", "baseline")}
PLP = "Pathogenic/Likely Pathogenic"
BLB = "Benign/Likely Benign"


def _find_outputs(calib_dir: Path, dataset: str):
    d = calib_dir / dataset
    if not d.is_dir():
        return None, None
    cals = sorted(d.glob(f"{dataset}_*_calibration.json"))
    if not cals:
        return None, None
    if len(cals) > 1:
        print(f"  WARNING {dataset}: {len(cals)} calibrations in {d}, using {cals[0].name}")
    cal = cals[0]
    var = Path(str(cal).replace("_calibration.json", "_variants.csv"))
    return cal, (var if var.exists() else None)


def _missense_map(dataset: str, df_ds: pd.DataFrame) -> pd.DataFrame:
    """variant_id -> is_missense / is_ptv for a default-kwargs Scoreset."""
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids

    release = "2018" if dataset.endswith("_clinvar_2018") else "2025"
    pcfg = PipelineConfig(dataset_csv="", dataset_name=dataset, output_dir="/tmp",
                          clinvar_release=release)
    ss = load_dataset_from_df(df_ds, pcfg)
    ids = _get_variant_ids(ss)
    mis, ptv = [], []
    for all_idx, (_, variants) in enumerate(ss.get_variants_by_id().items()):
        if ss._keep_mask[all_idx]:
            mis.append(any(getattr(v, "is_missense", False) for v in variants))
            ptv.append(any(getattr(v, "is_ptv", False) for v in variants))
    assert len(ids) == len(mis)
    return pd.DataFrame({"variant_id": ids, "is_missense": mis, "is_ptv": ptv})


def _combine_points(df_path: pd.DataFrame, df_ben: pd.DataFrame) -> pd.DataFrame:
    """Pathogenic evidence from df_path, benign evidence from df_ben. A variant
    with pathogenic points in df_path keeps them (counted in 'conflict' when
    df_ben gave it benign points)."""
    out = df_ben.copy()
    m = df_path.set_index("variant_id")
    conflicts = 0
    for col in ["standard_points", "oob_points"]:
        if col not in out.columns or col not in m.columns:
            continue
        pp = m[col].reindex(out["variant_id"]).astype(float).values
        bb = out[col].astype(float).values
        comb = np.where(pp > 0, pp, np.where(bb < 0, bb, np.where(np.isnan(bb), np.nan, 0.0)))
        if col == "oob_points":
            conflicts = int(((pp > 0) & (bb < 0)).sum())
        out[col] = comb
    out.attrs["conflicts"] = conflicts
    return out


def _points(df: pd.DataFrame) -> pd.Series:
    p = df["standard_points"].astype(float).copy()
    if "oob_points" in df.columns:
        has = df["oob_points"].notna()
        p[has] = df.loc[has, "oob_points"].astype(float)
    return p


def _metrics(tp, fp, tn, fn, ir_p, ir_b):
    det = tp + fp + tn + fn
    tot = det + ir_p + ir_b
    out = dict(n_plp=tp + fn + ir_p, n_blb=tn + fp + ir_b,
               tp=tp, fp=fp, tn=tn, fn=fn, ir_plp=ir_p, ir_blb=ir_b)
    out["determinate_rate"] = det / tot if tot else np.nan
    out["accuracy"] = (tp + tn) / det if det else np.nan
    out["sensitivity_det"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity_det"] = tn / (tn + fp) if (tn + fp) else np.nan
    # Share of ALL controls given correct-direction evidence (penalises IR).
    out["plp_correct_all"] = tp / (tp + fn + ir_p) if (tp + fn + ir_p) else np.nan
    out["blb_correct_all"] = tn / (tn + fp + ir_b) if (tn + fp + ir_b) else np.nan
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    out["mcc"] = (tp * tn - fp * fn) / den if den > 0 else np.nan
    out["dor"] = (tp * tn) / (fp * fn) if (fp > 0 and fn > 0) else np.inf
    out["dor_haldane"] = ((tp + .5) * (tn + .5)) / ((fp + .5) * (fn + .5))
    return out


def _score_dataset(df: pd.DataFrame, plp_ids, blb_ids, vus_ids):
    pts = _points(df).set_axis(df["variant_id"].values)
    p = pts.reindex(plp_ids).dropna()
    b = pts.reindex(blb_ids).dropna()
    v = df.set_index("variant_id")["standard_points"].reindex(vus_ids).dropna().astype(float)
    m = _metrics(int((p > 0).sum()), int((b > 0).sum()), int((b < 0).sum()),
                 int((p < 0).sum()), int((p == 0).sum()), int((b == 0).sum()))
    m["plp_strong"] = float((p >= 4).mean()) if len(p) else np.nan
    m["blb_strong"] = float((b <= -4).mean()) if len(b) else np.nan
    m["plp_mean_pts"] = float(p.mean()) if len(p) else np.nan
    m["blb_mean_pts"] = float(b.mean()) if len(b) else np.nan
    m["n_vus"] = len(v)
    m["vus_path"] = float((v > 0).mean()) if len(v) else np.nan
    m["vus_benign"] = float((v < 0).mean()) if len(v) else np.nan
    m["vus_det"] = float((v != 0).mean()) if len(v) else np.nan
    m["vus_path_strong"] = float((v >= 4).mean()) if len(v) else np.nan
    m["vus_path_pts_mean"] = float(v.clip(lower=0).mean()) if len(v) else np.nan
    m["n_missing_plp"] = len(plp_ids) - len(p)
    m["n_missing_blb"] = len(blb_ids) - len(b)
    return m, p, b, v


def _thresholds(cal: dict):
    """Lowest score reaching +1 / +4 (abnormal side) and -1 / -4: summarised as
    the number of non-empty pathogenic and benign point levels."""
    pr = cal.get("point_ranges") or {}
    levels = []
    for k, v in pr.items():
        try:
            k = int(float(k))
        except (TypeError, ValueError):
            continue
        if v:
            levels.append(k)
    return (max([l for l in levels if l > 0], default=0),
            min([l for l in levels if l < 0], default=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ross/assay_calibration/missense_only_calibrations")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset-tsv", default=None)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--runs", nargs="+", default=RUNS,
                    help="calib_<run> dirs to compare; the first must be baseline")
    args = ap.parse_args()
    runs = list(args.runs)
    assert runs[0] == "baseline", "first run must be baseline (defines the control set)"
    root = Path(args.root)
    out = Path(args.out or root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    tsv = args.dataset_tsv or _cfg.DATASET_TSV

    calib = {r: root / f"calib_{r}" for r in runs if r not in COMBINED}
    for r in runs:
        if r in COMBINED:
            for src in COMBINED[r]:
                calib.setdefault(src, root / f"calib_{src}")
    datasets = sorted(set.intersection(*[
        {p.name for p in d.iterdir() if p.is_dir() and _find_outputs(d, p.name)[1]}
        for d in calib.values()]))
    print(f"{len(datasets)} datasets calibrated in all {len(runs)} runs")

    df_full = load_master_df(tsv)
    slices = {ds: _filter_dataset_df(df_full, ds, tsv) for ds in datasets}
    del df_full
    from joblib import Parallel, delayed
    maps = Parallel(n_jobs=args.n_jobs, verbose=2)(
        delayed(_missense_map)(ds, slices[ds]) for ds in datasets)
    maps = dict(zip(datasets, maps))

    rows, pooled_pts = [], {r: {"plp": [], "blb": [], "vus": []} for r in runs}
    meta_rows = []
    for ds in datasets:
        mm = maps[ds].set_index("variant_id")
        base_cal, base_var = _find_outputs(calib["baseline"], ds)
        dfb = pd.read_csv(base_var)
        mis = mm["is_missense"].reindex(dfb["variant_id"]).fillna(False).values
        s = dfb["sample"].fillna("")
        plp_ids = dfb.loc[mis & s.str.contains(PLP, regex=False).values, "variant_id"].tolist()
        blb_ids = dfb.loc[mis & s.str.contains(BLB, regex=False).values, "variant_id"].tolist()
        vus_ids = dfb.loc[mis & dfb["is_vus"].fillna(False).astype(bool).values
                          & ~s.str.contains(PLP, regex=False).values
                          & ~s.str.contains(BLB, regex=False).values, "variant_id"].tolist()
        n_nonmis_plp = int((~mis & s.str.contains(PLP, regex=False).values).sum())
        n_nonmis_blb = int((~mis & s.str.contains(BLB, regex=False).values).sum())
        for r in runs:
            n_conflict = np.nan
            if r in COMBINED:
                cal_p, var_p = _find_outputs(calib[COMBINED[r][0]], ds)
                cal = json.load(open(cal_p))
                df = _combine_points(pd.read_csv(var_p),
                                     pd.read_csv(_find_outputs(calib[COMBINED[r][1]], ds)[1]))
                n_conflict = df.attrs["conflicts"]
            else:
                cal_p, var_p = _find_outputs(calib[r], ds)
                cal = json.load(open(cal_p))
                df = pd.read_csv(var_p)
            if r.startswith("missense"):
                # Consistency: the run's own P/LP / B/LB samples should be the
                # baseline missense controls.
                sr = df["sample"].fillna("")
                own_plp = set(df.loc[sr.str.contains(PLP, regex=False), "variant_id"])
                own_blb = set(df.loc[sr.str.contains(BLB, regex=False), "variant_id"])
                if own_plp != set(plp_ids) or own_blb != set(blb_ids):
                    print(f"  NOTE {ds}/{r}: own controls differ from baseline-missense "
                          f"(P/LP {len(own_plp)} vs {len(plp_ids)}, B/LB {len(own_blb)} vs {len(blb_ids)})")
            m, p, b, v = _score_dataset(df, plp_ids, blb_ids, vus_ids)
            pmax, bmin = _thresholds(cal)
            m.update(dataset=ds, run=r, prior=cal.get("prior"),
                     benign_method=cal.get("benign_method"), n_c=cal.get("n_c"),
                     max_path_pts=pmax, max_benign_pts=bmin,
                     uncalibratable=cal.get("uncalibratable_reason"),
                     n_nonmissense_plp_baseline=n_nonmis_plp,
                     n_nonmissense_blb_baseline=n_nonmis_blb,
                     n_oob_conflict=n_conflict)
            rows.append(m)
            pooled_pts[r]["plp"].append(p.values)
            pooled_pts[r]["blb"].append(b.values)
            pooled_pts[r]["vus"].append(v.values)

    res = pd.DataFrame(rows)
    # PU mode in the missense runs = no missense B/LB reference.
    nb = res[res.run == "baseline"].set_index("dataset")["n_blb"]
    res["pu_missense"] = res["dataset"].map(nb == 0)
    res.to_csv(out / "per_dataset_metrics.csv", index=False)

    # ---- pooled control/VUS summaries per run, split by PU status ----------
    def pooled(sub):
        tp, fp, tn, fn, irp, irb = (sub[c].sum() for c in ["tp", "fp", "tn", "fn", "ir_plp", "ir_blb"])
        m = _metrics(int(tp), int(fp), int(tn), int(fn), int(irp), int(irb))
        nv = sub["n_vus"].sum()
        for c in ["vus_path", "vus_benign", "vus_det", "vus_path_strong"]:
            m[c] = (sub[c].fillna(0) * sub["n_vus"]).sum() / nv if nv else np.nan
        m["n_datasets"] = len(sub)
        m["median_prior"] = sub["prior"].median()
        m["median_mcc"] = sub["mcc"].median()
        return m

    summ = []
    for grp, gmask in [("all", slice(None)), ("non_PU", ~res.pu_missense), ("PU_in_missense_runs", res.pu_missense)]:
        g = res[gmask] if not isinstance(gmask, slice) else res
        for r in runs:
            m = pooled(g[g.run == r])
            m.update(group=grp, run=r)
            summ.append(m)
    summ = pd.DataFrame(summ)
    cols = ["group", "run", "n_datasets", "n_plp", "n_blb", "determinate_rate", "accuracy",
            "sensitivity_det", "specificity_det", "plp_correct_all", "blb_correct_all",
            "mcc", "dor", "median_mcc", "median_prior", "vus_det", "vus_path", "vus_benign",
            "vus_path_strong"]
    summ[cols].to_csv(out / "pooled_summary.csv", index=False)

    # ---- paired per-dataset deltas ------------------------------------------
    from scipy.stats import wilcoxon
    piv = {c: res.pivot(index="dataset", columns="run", values=c)
           for c in ["mcc", "accuracy", "determinate_rate", "prior", "vus_path", "vus_benign",
                     "vus_det", "plp_correct_all", "blb_correct_all", "plp_strong", "blb_strong"]}
    pu = res[res.run == "baseline"].set_index("dataset")["pu_missense"]
    paired = []
    for c, pv in piv.items():
        for a, bname in [(x, y) for i, x in enumerate(runs) for y in runs[i + 1:]]:
            for grp, mask in [("non_PU", ~pu), ("PU", pu)]:
                d = (pv.loc[mask[mask].index, bname] - pv.loc[mask[mask].index, a]).dropna()
                if len(d) == 0:
                    continue
                try:
                    pval = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                except ValueError:
                    pval = np.nan
                paired.append(dict(metric=c, comparison=f"{bname} - {a}", group=grp, n=len(d),
                                   median_delta=d.median(), mean_delta=d.mean(),
                                   n_up=int((d > 0).sum()), n_down=int((d < 0).sum()),
                                   wilcoxon_p=pval))
    pd.DataFrame(paired).to_csv(out / "paired_deltas.csv", index=False)

    # ---- pooled evidence distributions --------------------------------------
    lev = list(range(-8, 9))
    dist = []
    for r in runs:
        for k in ["plp", "blb", "vus"]:
            a = np.concatenate(pooled_pts[r][k]) if pooled_pts[r][k] else np.array([])
            a = np.clip(np.round(a), -8, 8)
            cnt = {f"pts_{l}": int((a == l).sum()) for l in lev}
            dist.append(dict(run=r, set=k, n=len(a), **cnt))
    pd.DataFrame(dist).to_csv(out / "pooled_points_distribution.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summ[cols].round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
