"""E7 per-dataset figures: all-P/LP evidence vs missense evidence (post-fit re-weighting and missense-only
refit), each judged against random all-P/LP subsets of the missense size (re-weighted and refit).

Arms (first 100 baseline bootstraps, which share bootstrap composition with the 100-bootstrap refits; point
mapping lp_anchored everywhere):
  all P/LP            production fits                                        prior: gnomAD, all types
  missense, reweight  baseline components, P/LP weights from missense P/LP   prior: gnomAD missense
  missense, refit     refit with missense-only P/LP (plpmis)                 prior: gnomAD missense
  subsets, reweight   baseline components, P/LP weights from 10 random all-P/LP subsets of the missense size
  subsets, refit      3 refits with random all-P/LP subsets of the missense size (plpsub)
  (subset arms use the gnomAD missense prior: they are the count-matched null for the missense arms)

Layout
  Row 0  sample histograms + fitted mixtures; a P/LP column per arm (missense columns show missense P/LP;
         the subset column shows all P/LP with each subset's fitted P/LP density as a thin line)
  Row 1  A  pathogenic evidence: log LR+ 5th percentile (natural log), every arm
         B  benign evidence: log LR+ 95th percentile, every arm
         C  evidence strip: one band per arm / subset, coloured by the points assigned along the score axis
         D  numbers and each rule's call (in-bag points of the dataset's missense P/LP)

Usage:  python -m analysis.missense_sensitivity.plot_stability_datasets [--datasets ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
SENS = ROOT / "sens_popmis_lp"
PLP = "Pathogenic/Likely Pathogenic"
ARM_STYLE = {  # colour, linewidth, linestyle, label
    "all": ("#0b0b0b", 5.0, "-", "all P/LP (prior: gnomAD all)"),   # wide underlay: visible as an outline
    "mis_rw": ("#eb6834", 1.9, "-", "missense, re-weighted"),
    "mis_refit": ("#6250d6", 1.9, "--", "missense-only refit"),
    "sub_rw": ("#2a78d6", 0.9, "-", "random subsets, re-weighted"),
    "sub_refit": ("#1baf7a", 1.3, ":", "random subsets, refit"),
    "cell_all": ("#008300", 2.0, (0, (6, 2)), "all P/LP, membership-cell fit"),
    "cell_mis": ("#e87ba4", 2.2, "-.", "missense, membership-cell fit"),
    "cell_mis_3c": ("#a50f5f", 1.6, (0, (1, 1)), "missense, membership-cell fit (3c)"),
}
CELLS = Path(__import__("os").environ.get("CELLS_DIR", "/data/ross/assay_calibration/missense_only_calibrations/cells/light_c1"))
CELL_SRC = "light, 20 bootstraps" if "light" in str(CELLS) else "HPC medium, 100 bootstraps"

PRIORITY = {
    "TP53_Giacomelli_2018_p53WT_Nutlin3_clinvar_2018": "composition, n large: missense P/LP give more evidence than all P/LP",
    "GCK_Gersing_2024_abundance": "composition by mean points only; highest level unchanged",
    "PTEN_Matreyek_2018_clinvar_2018": "missense at the edge of the subset range",
    "CHEK2_McCarthy-Leo_2024": "n = 7 (count-limited by n), yet subsets stable and missense moves a level",
    "BRCA2_IGVF": "n = 2: subsets unstable and missense outside their range",
    "KCNE1_Muhammad_2024_trafficking_WT_background_DN": "n = 3: subsets span 0-8 points",
    "KCNH2_Kozek_Glazer_2020": "n = 2 but every subset gives the same calls (n rule over-flags)",
    "BRCA2_Sahu_2023_exon13_SGE": "n = 3, stable under subsets (n rule over-flags)",
    "TSC2_IGVF": "n = 61 yet 1 of 10 subsets drops the top level",
    "BAP1_Waters_2024": "n = 11, just above the floor, unstable under subsets",
    "RAD51C_Olvera-León_2024": "n = 8: missense at the top of a wide subset range",
    "PALB2_IGVF": "n = 3: wide subset spread",
    "LDLR_Tabet_2025_uptake": "control: n = 358",
    "TP53_Boettcher_2019_clinvar_2018": "composition: missense P/LP turn benign evidence on for 69% of missense variants",
    "GCK_Gersing_2023_complementation": "missense-only refit collapses pathogenic evidence (top level 5 -> 1); re-weighting and subset refits do not",
    "SGCB_Li_2023": "n = 8: re-weighting deepens benign evidence (mean -1 -> -4) beyond the subsets",
    "DDX3X_Radford_2023": "re-weighting weakens benign evidence (mean -1.9 -> -1.0) beyond the subsets",
    "TP53_Giacomelli_2018_combined_score_clinvar_2018": "re-weighting deepens benign evidence and lowers the top level 8 -> 7",
    "BRCA1_Adamovich_2022_HDR_clinvar_2018": "n = 16: 41% of missense move >= 2 points under re-weighting",
    "FKRP_Ma_2024": "re-weighting raises the top level 7 -> 8",
    "XRCC2_IGVF": "all 6 P/LP non-missense: missense-only refit is NU",
    "PALB2_Boonen_2026_SGE": "all 4 P/LP non-missense: missense-only refit is NU",
}


def _load_cal(d):
    c = sorted(Path(d).glob("*_calibration.json"))
    if not c:
        return None
    return dict(cal=json.load(open(c[0])),
                lr=json.load(gzip.open(str(c[0]).replace("_calibration.json", "_lr_values.json.gz"), "rt")))


def load_arms(ds, kc=None):
    arms = {"all": [_load_cal(SENS / "medium_baseline_popall" / ds)],
            "mis_rw": [_load_cal(SENS / "e1_priority_calibrations" / ds / "missense")],
            "mis_refit": [_load_cal(SENS / "medium_plpmis" / ds)],
            "sub_rw": [_load_cal(d) for d in sorted((SENS / "e1_priority_calibrations" / ds).glob("subset_*"),
                                                    key=lambda p: int(p.name.split("_")[1]))],
            "sub_refit": [_load_cal(SENS / f"medium_plpsub_s{i}" / ds) for i in range(3)]}
    if kc:  # membership-cell fits (light C1 run), at the config K and, for 2c datasets, at 3c too
        arms["cell_all"] = [_load_cal(CELLS / f"calib_cell_all_{kc}" / ds)]
        arms["cell_mis"] = [_load_cal(CELLS / f"calib_cell_mis_{kc}" / ds)]
        if kc == "2c":
            arms["cell_mis_3c"] = [_load_cal(CELLS / "calib_cell_mis_3c" / ds)]
    return {k: [c for c in v if c] for k, v in arms.items()}


def _recs(fits_ds, n_c):
    return [fits_ds[k][n_c] for k in sorted(fits_ds, key=int) if fits_ds[k].get(n_c)] if fits_ds else []


def _reweighted(ds, df_ds, fits_ds, n_c, n_subsets=10):
    """E1 arms re-created (same training rows, same rng as matched_subsample)."""
    from analysis.missense_sensitivity.compare_runs import _missense_map
    from analysis.missense_sensitivity.matched_subsample import training_rows
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release))
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    pc = names.index(PLP)
    sa, scores = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    mis = _missense_map(ds, df_ds)["is_missense"].values
    is_plp = sa[:, pc].astype(bool)
    n_mis = int((is_plp & mis).sum())
    seeds = [k for k in sorted(fits_ds, key=int) if fits_ds[k].get(n_c)]
    recs = [fits_ds[k][n_c] for k in seeds]
    out = dict(names=names, pc=pc, sa=sa, scores=scores, is_plp=is_plp, mis_mask=is_plp & mis, all=recs,
               mis_rw=[], sub_rw=[])
    if n_mis == 0:
        return out
    rows_per_seed = [training_rows(sa, scores, int(b)) for b in seeds]
    rng = np.random.default_rng(12345)
    plp_rows = np.flatnonzero(is_plp)
    members = [("mis_rw", is_plp & mis)]
    for _ in range(n_subsets):
        m = np.zeros(len(scores), bool)
        m[rng.choice(plp_rows, size=n_mis, replace=False)] = True
        members.append(("sub_rw", m))
    for arm, member in members:
        new = []
        for rec, (rows, onehot) in zip(recs, rows_per_seed):
            rec = json.loads(json.dumps(rec))
            rp = rows[onehot[rows, pc] & member[rows]]
            if len(rp):
                rec["fit"]["weights"][pc] = _em_weights(scores[rp], rec["fit"]["component_params"],
                                                        rec["fit"]["weights"][pc]).tolist()
            new.append(rec)
        if arm == "mis_rw":
            out["mis_rw"] = new
        else:
            out["sub_rw"].append(new)
    return out


def _points(cal, x):
    from src.assay_calibration.fit_utils.fit import assign_points
    return assign_points(np.asarray(x, float), {int(k): v for k, v in (cal.get("point_ranges") or {}).items()})


def _rule_text(arms, sc_mis, n_plp, n_mis):
    from analysis.missense_sensitivity.matched_subsample import _score
    st = {k: [_score(c["cal"], sc_mis, []) for c in v] for k, v in arms.items()}
    one = lambda k, key: st[k][0][key] if st.get(k) else np.nan

    def rng(k, key):
        v = [s[key] for s in st.get(k, [])]
        return (min(v), max(v)) if v else (np.nan, np.nan)
    ma, mm, mr = (one(k, "plp_mean_path") for k in ("all", "mis_rw", "mis_refit"))
    la, lm, lr_ = (one(k, "max_path") for k in ("all", "mis_rw", "mis_refit"))
    srw, sref = rng("sub_rw", "plp_mean_path"), rng("sub_refit", "plp_mean_path")
    lrw, lref = rng("sub_rw", "max_path"), rng("sub_refit", "max_path")
    nchg = sum(s["max_path"] != la for s in st.get("sub_rw", []))
    pa, pm, pr = (one(k, "prior") for k in ("all", "mis_rw", "mis_refit"))
    mc, lc, pc = one("cell_mis", "plp_mean_path"), one("cell_mis", "max_path"), one("cell_mis", "prior")
    yn = lambda b: "FLAG" if b else "-"
    out_rw = bool((mm < srw[0] - 1e-9) or (mm > srw[1] + 1e-9))
    return "\n".join([
        f"P/LP: {n_plp} all, {n_mis} missense", "",
        "prior",
        f"  all {pa:.3f}   missense rw {pm:.3f}   refit {pr:.3f}   cell {pc:.3f}", "",
        "mean pathogenic points per missense P/LP",
        f"  all P/LP        {ma:5.2f}",
        f"  missense rw     {mm:5.2f}   subsets rw    {srw[0]:.2f}-{srw[1]:.2f}",
        f"  missense refit  {mr:5.2f}   subsets refit {sref[0]:.2f}-{sref[1]:.2f}",
        f"  missense cell   {mc:5.2f}   ({CELL_SRC})", "",
        "highest pathogenic level",
        f"  all {la:.0f}   missense rw {lm:.0f}   refit {lr_:.0f}   cell {lc:.0f}",
        f"  subsets rw {lrw[0]:.0f}-{lrw[1]:.0f} ({nchg}/{len(st.get('sub_rw', []))} change it)",
        f"  subsets refit {lref[0]:.0f}-{lref[1]:.0f}", "",
        "candidate rules (re-weighting)",
        f"  n < 5                      {yn(n_mis < 5)}",
        f"  n < 10                     {yn(n_mis < 10)}",
        f"  missense moves level >= 1  {yn(abs(lm - la) >= 1)}",
        f"  missense moves mean >= 0.5 {yn(abs(mm - ma) >= 0.5)}",
        f"  missense outside subsets   {yn(out_rw)}",
    ])


def _cell_model(ds, df_ds, kc):
    """Cell-fit records for (ds, K) from CELLS/raw + CELLS/cells: raw per-cell records, the Cells table, and the
    composed cell_mis records in that arm's calibration layout. None if the cell fit is not there."""
    import gzip as _gz
    from analysis.membership_cells.cells import Cells, compose_arm
    raw = sorted((CELLS / "raw").glob(f"{ds}__{kc}__*.json.gz"))
    npz = CELLS / "cells" / f"{ds}.npz"
    if not raw or not npz.exists():
        return None
    cf = {}
    for f in raw:
        cf.update(json.load(_gz.open(f)))
    cells = Cells.load(npz)
    out = dict(cells=cells, raw=[v[kc] for v in cf.values() if v and kc in v])
    for arm, key in (("cell_mis", "mis"), ("cell_all", "all")):
        try:
            comp, layout, _ = compose_arm(cells, cf, df_ds, arm)
        except ValueError:
            comp, layout = {}, []
        out[key] = [v[kc] for v in comp.values() if kc in v]
        out["layout" if key == "mis" else "layout_all"] = layout
    return out


def draw(ds, df_ds, fits_base, fits_mis_refit, fits_sub_refit, cfg_entry, why, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import run_igvf_batch as rib
    from src.assay_calibration.plot_utils.utils import (sample_density, sample_display_name, sample_hist_style,
                                                        _display_dataset, log_thresholds_with_ylim_pad)
    n_c, _, _ = rib.parse_dataset_config(cfg_entry)
    arms = load_arms(ds, n_c if n_c in ("2c", "3c") else "3c")
    if not arms["all"]:
        return None
    A = _reweighted(ds, df_ds, fits_base, n_c)
    names, sa, scores = A["names"], A["sa"], A["scores"]
    sr = np.asarray(arms["all"][0]["lr"]["score_range"])
    n_mis, n_plp = int(A["mis_mask"].sum()), int(A["is_plp"].sum())
    refit_recs = _recs(fits_mis_refit, n_c)
    sub_refit_recs = [r for r in (_recs(f, n_c) for f in fits_sub_refit) if r]

    kc = n_c if n_c in ("2c", "3c") else "3c"
    CM = _cell_model(ds, df_ds, kc)
    cols = []
    for j, n in enumerate(names):
        cols.append(("sample", sample_display_name(n), sa[:, j].astype(bool), j, A["all"], n))
        if n == PLP and CM is not None and CM["all"] and PLP in CM["layout_all"]:
            cols.append(("cell_all", f"P/LP all, cell fit ({kc}, {CELL_SRC})", A["is_plp"],
                         CM["layout_all"].index(PLP), CM["all"], PLP))
        if n == PLP:
            if A["mis_rw"]:
                cols.append(("mis_rw", "P/LP missense, re-weighted", A["mis_mask"], j, A["mis_rw"], PLP))
            if refit_recs or n_mis == 0:
                cols.append(("mis_refit", "P/LP missense, refit", A["mis_mask"], j, refit_recs, PLP))
            if A["sub_rw"] or sub_refit_recs:
                cols.append(("subsets", "P/LP random subsets (missense n)", A["is_plp"], j, None, PLP))
            if CM is not None and CM["mis"] and PLP in CM["layout"]:
                cols.append(("cell_mis", f"P/LP missense, cell fit ({kc}, {CELL_SRC})", A["mis_mask"],
                             CM["layout"].index(PLP), CM["mis"], PLP))
            if CM is not None and CM["raw"]:
                cols.append(("cells", f"P/LP cells, cell fit ({kc})", A["is_plp"], None, CM["raw"], PLP))
    ncol = len(cols)
    fig = plt.figure(figsize=(max(4.2 * ncol, 28), 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3, left=0.04, right=0.99, top=0.9, bottom=0.06)
    g0 = gs[0].subgridspec(1, ncol, wspace=0.28)
    g1 = gs[1].subgridspec(1, 4, width_ratios=[1.25, 1.25, 1.4, 0.95], wspace=0.24)
    xlim = None
    for k, (kind, title, mask, widx, recs, style_name) in enumerate(cols):
        a = fig.add_subplot(g0[0, k])
        x = scores[mask]
        col, alpha = sample_hist_style(style_name, widx)
        if kind in ("mis_rw", "mis_refit", "cell_mis"):
            col, alpha = "#8C3B4A", 0.55
        if len(x) > 1:
            q1, q3 = np.percentile(x, [25, 75]); fd = 2 * (q3 - q1) * len(x) ** (-1 / 3)
            bins = max(10, min(100, int((x.max() - x.min()) / fd)) if fd > 0 else 30)
            sns.histplot(x, bins=bins, stat="density", ax=a, alpha=alpha, color=col)
        elif len(x):
            sns.histplot(x, bins=5, stat="density", ax=a, alpha=alpha, color=col)
        hmax = max([p.get_height() for p in a.patches]) if a.patches else 1.0
        if kind == "subsets":
            for recs_s in A["sub_rw"]:
                d = np.nansum(sample_density(sr, recs_s, widx), axis=1)
                a.plot(sr, np.nanmedian(d, axis=0), color=ARM_STYLE["sub_rw"][0], lw=0.9, alpha=0.7)
            for recs_s in sub_refit_recs:
                d = np.nansum(sample_density(sr, recs_s, 0), axis=1)
                a.plot(sr, np.nanmedian(d, axis=0), color=ARM_STYLE["sub_refit"][0], lw=1.4, ls=":")
            d = np.nansum(sample_density(sr, A["all"], widx), axis=1)
            a.plot(sr, np.nanmedian(d, axis=0), color="black", lw=1.8, label="all P/LP fit")
            a.plot([], [], color=ARM_STYLE["sub_rw"][0], lw=0.9, label=f"{len(A['sub_rw'])} subsets, re-weighted")
            a.plot([], [], color=ARM_STYLE["sub_refit"][0], lw=1.4, ls=":", label=f"{len(sub_refit_recs)} subsets, refit")
            a.legend(fontsize=6.5)
        elif kind == "cells":
            # each P/LP cell's own fitted density (median over bootstraps), shared components
            t = CM["cells"].table
            pal = ["#eb6834", "#2a78d6", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
            k = 0
            for ci, row in t.iterrows():
                if not row["mask"][0]:
                    continue
                col_k = pal[k % len(pal)]
                xc = CM["cells"].scores[CM["cells"].cell_of_row == ci]
                if len(xc) > 1:  # the cell's own data, as a step histogram (density) in its colour
                    a.hist(xc, bins=max(5, min(30, len(xc) // 2)), density=True, histtype="step", color=col_k,
                           lw=1.0, alpha=0.9)
                d = np.nansum(sample_density(sr, recs, ci), axis=1)
                a.plot(sr, np.nanmedian(d, axis=0), color=col_k, lw=1.8 if row.n >= 5 else 1.0,
                       ls="-" if row.n >= 5 else "--", label=f"{row['name']} (n={row.n})")
                k += 1
            a.legend(fontsize=6, loc="upper left")
        elif kind == "mis_refit" and n_mis == 0:
            a.text(0.5, 0.5, "no missense P/LP:\nmissense-only calibration\nruns in NU mode", ha="center",
                   va="center", transform=a.transAxes, fontsize=10)
        elif recs:
            w_idx = 0 if kind == "mis_refit" else widx
            dens = sample_density(sr, recs, w_idx)
            for c in range(dens.shape[1]):
                a.plot(sr, np.nanpercentile(dens[:, c, :], 50, axis=0), color=f"C{c}", ls="--", lw=1, label=f"Comp {c + 1}")
            tot = np.nansum(dens, axis=1)
            p5, p50, p95 = np.percentile(tot, [5, 50, 95], axis=0)
            a.plot(sr, p50, color=ARM_STYLE[kind][0] if kind in ARM_STYLE else "black", lw=1.6)
            a.fill_between(sr, p5, p95, color="gray", alpha=0.3)
            a.legend(fontsize=6.5)
        a.set_title(f"{n_c}: {title}\n(n={int(mask.sum()):,d})", fontsize=9.5)
        a.set_ylim([0, hmax * 1.1]); a.tick_params(labelsize=8); a.grid(lw=0.5, alpha=0.3)
        if xlim is None:
            xlim = a.get_xlim()
        a.set_xlim(xlim)

    aP, aB, aS, aT = (fig.add_subplot(g1[0, i]) for i in range(4))
    order = ["all", "sub_rw", "sub_refit", "cell_all", "mis_refit", "mis_rw", "cell_mis", "cell_mis_3c"]
    ref = np.asarray(arms["all"][0]["lr"]["log_lr_plus_p5"], float)

    def same_as_all(c):
        cur = np.interp(sr, np.asarray(c["lr"]["score_range"]), np.asarray(c["lr"]["log_lr_plus_p5"], float))
        ok = np.isfinite(ref) & np.isfinite(cur)
        # within the plotted range, the curves differ by < 0.05 (median) -> say so in the legend
        return bool(ok.any() and np.nanmedian(np.abs(ref[ok] - cur[ok])) < 0.05)
    pv = list(range(1, 9))
    for ax_, q, ttl in ((aP, "p5", "A  pathogenic evidence: log LR+, 5th percentile"),
                        (aB, "p95", "B  benign evidence: log LR+, 95th percentile")):
        for arm in order:
            col, lw, ls, lab = ARM_STYLE[arm]
            for i, c in enumerate(arms.get(arm, [])):
                tag = " - matches all P/LP" if arm in ("mis_rw", "mis_refit", "cell_all", "cell_mis", "cell_mis_3c") and same_as_all(c) else ""
                ax_.plot(np.asarray(c["lr"]["score_range"]), c["lr"][f"log_lr_plus_{q}"], color=col, lw=lw, ls=ls,
                         alpha=0.45 if arm == "all" else (0.8 if lw < 1.5 else 1),
                         label=(lab + tag) if i == 0 else None, zorder=1 if arm == "all" else (3 if lw > 1.5 else 2))
        for arm, ls in (("all", "--"), ("mis_rw", ":")):
            if arms.get(arm):
                tauP, tauB, _, _ = log_thresholds_with_ylim_pad(arms[arm][0]["cal"]["prior"], pv,
                                                                acmg_mapping_method="lp_anchored")
                for t in (tauP if q == "p5" else tauB):
                    ax_.axhline(t, color=ARM_STYLE[arm][0], ls=ls, lw=0.7, alpha=0.35)
        _, _, top, bot = log_thresholds_with_ylim_pad(arms["all"][0]["cal"]["prior"], pv,
                                                      acmg_mapping_method="lp_anchored")
        ax_.set_ylim(bot, top); ax_.set_xlim(xlim); ax_.grid(lw=0.5, alpha=0.3)
        ax_.axhline(0, color="#52514e", lw=0.8)
        ax_.set_title(ttl + "\nthresholds: all P/LP dashed, missense re-weighted dotted", fontsize=9.5, loc="left")
        ax_.set_xlabel("Score"); ax_.set_ylabel("log LR+ (natural log)")
    aP.legend(fontsize=7.5, loc="best")
    rows_, labs = [], []
    for arm, lab in (("all", "all P/LP"), ("cell_all", "all P/LP, cell fit"), ("mis_rw", "missense, re-weighted"),
                     ("mis_refit", "missense, refit"),
                     ("cell_mis", "missense, cell fit"), ("cell_mis_3c", "missense, cell fit (3c)")):
        if arms.get(arm):
            rows_.append(_points(arms[arm][0]["cal"], sr)); labs.append(lab)
    n_top = len(rows_)
    for i, c in enumerate(arms.get("sub_rw", [])):
        rows_.append(_points(c["cal"], sr)); labs.append(f"subset {i + 1}, re-weighted")
    n_mid = len(rows_)
    for i, c in enumerate(arms.get("sub_refit", [])):
        rows_.append(_points(c["cal"], sr)); labs.append(f"subset {i + 1}, refit")
    cmap = ListedColormap(["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
                           "#f4f4f4",
                           "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"])
    norm = BoundaryNorm(np.arange(-8.5, 9.5, 1), cmap.N)
    im = aS.imshow(np.vstack(rows_), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   extent=[sr[0], sr[-1], len(rows_) - 0.5, -0.5])
    aS.set_yticks(range(len(labs))); aS.set_yticklabels(labs, fontsize=7.5)
    for y in (n_top - 0.5, n_mid - 0.5):
        if 0 < y < len(rows_) - 1:
            aS.axhline(y, color="white", lw=4)
    aS.set_xlim(xlim); aS.set_xlabel("Score")
    aS.set_title("C  evidence strip: points along the score axis", fontsize=9.5, loc="left")
    cb = fig.colorbar(im, ax=aS, ticks=range(-8, 9, 2), fraction=0.04, pad=0.02)
    cb.set_label("points", fontsize=8); cb.ax.tick_params(labelsize=7)
    aT.axis("off")
    aT.text(0, 1, "D\n" + _rule_text(arms, scores[A["mis_mask"]], n_plp, n_mis), va="top", ha="left",
            family="monospace", fontsize=8.5, transform=aT.transAxes)
    fig.suptitle(f"{_display_dataset(ds)} - {why}\npoint mapping LP-anchored (6 points = 0.90 posterior, prior "
                 f"capped at 0.5); first 100 bootstraps", fontsize=13, x=0.04, ha="left", y=0.995)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{_display_dataset(ds)}_stability.png"
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    if CM is not None and CM["raw"]:
        draw_cells(ds, CM, kc, xlim, out)
    return str(fn)


def draw_cells(ds, CM, kc, xlim, out):
    """One panel per cell: the cell's scores (histogram) and its fitted mixture under the shared components
    (components dashed at the median, total median with 5-95% band over bootstraps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.assay_calibration.plot_utils.utils import sample_density, _display_dataset
    cells, recs = CM["cells"], CM["raw"]
    t = cells.table.reset_index(drop=True)
    order = t.sort_values(["n"], ascending=False).index.tolist()
    sr = np.linspace(xlim[0], xlim[1], 600)
    nc = 6
    nr = int(np.ceil(len(order) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.6 * nr + 0.8), squeeze=False)
    for a in axes.flat:
        a.axis("off")
    for k, ci in enumerate(order):
        a = axes.flat[k]; a.axis("on")
        x = cells.scores[cells.cell_of_row == ci]
        if len(x) > 1:
            q1, q3 = np.percentile(x, [25, 75]); fd = 2 * (q3 - q1) * len(x) ** (-1 / 3)
            bins = max(5, min(60, int((x.max() - x.min()) / fd)) if fd > 0 else 15)
        else:
            bins = 5
        sns.histplot(x, bins=bins, stat="density", ax=a, color="#9a9a9a", alpha=0.5)
        hmax = max([p.get_height() for p in a.patches]) if a.patches else 1.0
        dens = sample_density(sr, recs, ci)
        for c in range(dens.shape[1]):
            a.plot(sr, np.nanpercentile(dens[:, c, :], 50, axis=0), color=f"C{c}", ls="--", lw=0.9)
        tot = np.nansum(dens, axis=1)
        p5, p50, p95 = np.percentile(tot, [5, 50, 95], axis=0)
        a.plot(sr, p50, color="black", lw=1.3); a.fill_between(sr, p5, p95, color="gray", alpha=0.3)
        w = np.median([np.asarray(r["fit"]["weights"][ci]) for r in recs], axis=0)
        a.set_title(f"{t.loc[ci, 'name']}\nn={int(t.loc[ci, 'n'])}  w=" + "/".join(f"{v:.2f}" for v in w), fontsize=7.5)
        a.set_xlim(xlim); a.set_ylim(0, max(hmax, np.nanmax(p95) if np.isfinite(np.nanmax(p95)) else 1) * 1.1)
        a.tick_params(labelsize=6.5); a.set_xlabel(""); a.set_ylabel("")
    fig.suptitle(f"{_display_dataset(ds)} - membership cells ({kc}, {CELL_SRC}): each cell's scores and fitted mixture "
                 f"(shared components; w = median mixing weights)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fn = Path(out) / f"{_display_dataset(ds)}_cells.png"
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return str(fn)


def _safe(*a):
    try:
        return draw(*a)
    except Exception as e:
        import traceback
        return f"FAILED {a[0]}: {e!r}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figs_stability_rules" / "per_dataset"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", default="src/igvf_configs/dataset_configs_aug_2026.json")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    cfg = json.load(open(args.config))
    todo = list(args.datasets or PRIORITY)
    fb = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results_first100.json.gz"))
    fm = json.load(gzip.open(ROOT / "explorer_jobs_pp_medium_plpmis" / "bootstrap_results.json.gz"))
    fs = [json.load(gzip.open(ROOT / f"explorer_jobs_pp_medium_plpsub_s{i}" / "bootstrap_results.json.gz"))
          for i in range(3)]
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in todo}
    del df_full
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(_safe)(ds, slices[ds], fb[ds], fm.get(ds), [f.get(ds) for f in fs],
                       cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")), PRIORITY.get(ds, ""), args.out)
        for ds in todo)
    for x in res:
        print(x if x else "skipped (missing calibrations)")


if __name__ == "__main__":
    main()
