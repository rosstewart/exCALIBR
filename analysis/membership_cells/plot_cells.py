"""Per-dataset membership-cell figures: composite samples from ONE cell fit, and their evidence vs standard ExCALIBR.

Row 0  composite samples, each = count-weighted mixture of its cells (histogram of the member variants + composed
       density: components dashed at the median, total with 5-95% band over bootstraps):
         P/LP (all consequences) | P/LP missense | B/LB (all consequences) | gnomAD (all) | gnomAD missense |
         Synonymous | Synonymous without SpliceAI > 0.2 (splice-detecting assays only)
Row 1  A  pathogenic evidence  log LR+ 5th percentile (natural log)
       B  benign evidence      log LR+ 95th percentile
       C  evidence strip (points along the score axis)
       D  priors and points on missense variants
       Arms: cell fit all P/LP (prior: all gnomAD), cell fit missense P/LP (prior: gnomAD missense), cell fit
       missense without splice-risk synonymous (splice assays), standard ExCALIBR all P/LP (first 100 bootstraps,
       prior all gnomAD) and standard missense-only refit (prior gnomAD missense). Mapping lp_anchored throughout.

Usage:  python -m analysis.membership_cells.plot_cells [--cells-dir .../cells/hpc_medium/labeled] [--datasets ...]
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
SAI = "|SpliceAI>0.2"
COMPOSITES = [  # title, predicate(mask, consequence) ; mask = (PLP, BLB, gnomAD, Syn)
    ("P/LP (all consequences)", lambda m, c: m[0]),
    ("P/LP missense", lambda m, c: m[0] and c.startswith("missense_variant")),
    ("B/LB (all consequences)", lambda m, c: m[1]),
    ("gnomAD (all)", lambda m, c: m[2]),
    ("gnomAD missense", lambda m, c: m[2] and c.startswith("missense_variant")),
    ("Synonymous", lambda m, c: m[3]),
    ("Synonymous, SpliceAI > 0.2 removed", lambda m, c: m[3] and not c.endswith(SAI)),
]
HIST = {"P/LP": "#d9a3a8", "B/LB": "#7fb3d5", "gnomAD": "#c8c8c8", "Synonymous": "#9fd4a3"}
SHORT = {"std_all": "standard, all P/LP", "std_mis": "standard, missense refit", "cell_all": "cell, all P/LP",
         "cell_all_nosplsyn": "cell, all P/LP, no splice-syn", "cell_mis": "cell, missense",
         "cell_mis_nosplsyn": "cell, missense, no splice-syn"}
ARMS = [  # key, label, colour, lw, ls
    ("std_all", "standard ExCALIBR, all P/LP", "#0b0b0b", 4.5, "-"),
    ("std_mis", "standard ExCALIBR, missense refit", "#6250d6", 1.9, "--"),
    ("cell_all", "cell fit, all P/LP", "#008300", 2.2, "-"),
    ("cell_all_nosplsyn", "cell fit, all P/LP, splice-risk syn removed", "#005a00", 1.5, ":"),
    ("cell_mis", "cell fit, missense P/LP", "#e87ba4", 2.2, "-"),
    ("cell_mis_nosplsyn", "cell fit, missense, splice-risk syn removed", "#a50f5f", 1.5, ":"),
]


def _load_cal(d):
    c = sorted(Path(d).glob("*_calibration.json"))
    if not c:
        return None
    return dict(cal=json.load(open(c[0])),
                lr=json.load(gzip.open(str(c[0]).replace("_calibration.json", "_lr_values.json.gz"), "rt")))


def _points(cal, x):
    from src.assay_calibration.fit_utils.fit import assign_points
    return assign_points(np.asarray(x, float), {int(k): v for k, v in (cal.get("point_ranges") or {}).items()})


def _mis_scores(cal_dir, ds):
    """Scores of missense variants in a calibration's variants table (for points summaries)."""
    import pandas as pd
    f = sorted((cal_dir / ds).glob(f"{ds}_*_variants.csv"))
    if not f:
        return None
    x = pd.read_csv(f[0], usecols=lambda c: c in {"score", "aa_ref", "aa_alt"})
    a, r = x.aa_alt.astype(str), x.aa_ref.astype(str)
    return x.score[x.aa_alt.notna() & x.aa_ref.notna() & (a != r) & ~a.isin(["*", "Ter", "nan", ""])].values


def draw(ds, kc, cells_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from analysis.membership_cells.cells import Cells
    from src.assay_calibration.plot_utils.utils import sample_density, _display_dataset, log_thresholds_with_ylim_pad
    raw = sorted((cells_dir / "raw").glob(f"{ds}__{kc}__*.json.gz"))
    npz = cells_dir / "cells" / f"{ds}.npz"
    if not raw or not npz.exists():
        return None
    cf = {}
    for f in raw:
        cf.update(json.load(gzip.open(f)))
    recs = [v[kc] for v in cf.values() if v and kc in v]
    cells = Cells.load(npz)
    t = cells.table.reset_index(drop=True)
    W = np.stack([np.asarray(r["fit"]["weights"], float) for r in recs])          # (B, n_cells, K)
    n = t["n"].to_numpy(float)
    detects_splice = bool(cells.meta.get("detects_splice"))
    comps = []
    for title, pred in COMPOSITES:
        if title.endswith("removed") and not detects_splice:
            continue
        sel = np.array([pred(tuple(m), str(c)) for m, c in zip(t["mask"], t["consequence"])])
        if not sel.any():
            continue
        wc = (W[:, sel, :] * n[sel][None, :, None]).sum(1) / n[sel].sum()        # (B, K) composed weights
        rows = np.isin(cells.cell_of_row, np.flatnonzero(sel))
        comps.append((title, wc, cells.scores[rows], int(n[sel].sum()), int(sel.sum())))
    arms = {"std_all": _load_cal(SENS / "medium_baseline_popall" / ds),
            "std_mis": _load_cal(SENS / "medium_plpmis" / ds),
            "cell_all": _load_cal(cells_dir / f"calib_cell_all_{kc}" / ds),
            "cell_mis": _load_cal(cells_dir / f"calib_cell_mis_{kc}" / ds),
            "cell_mis_nosplsyn": _load_cal(cells_dir / f"calib_cell_mis_nosplsyn_{kc}" / ds),
            "cell_all_nosplsyn": _load_cal(cells_dir / f"calib_cell_all_nosplsyn_{kc}" / ds)}
    arms = {k: v for k, v in arms.items() if v}
    if "cell_all" not in arms:
        return None
    sr = np.asarray(arms["cell_all"]["lr"]["score_range"])
    lo, hi = float(np.min(cells.scores)), float(np.max(cells.scores))
    pad = 0.04 * (hi - lo)
    xlim = (lo - pad, hi + pad)

    ncol = len(comps)
    fig = plt.figure(figsize=(max(3.9 * ncol, 26), 12.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.32, left=0.04, right=0.99, top=0.9, bottom=0.06)
    g0 = gs[0].subgridspec(1, ncol, wspace=0.26)
    g1 = gs[1].subgridspec(1, 4, width_ratios=[1.2, 1.2, 1.3, 0.85], wspace=0.34)
    x = np.linspace(xlim[0], xlim[1], 600)
    for k, (title, wc, xs, nv, ncell) in enumerate(comps):
        a = fig.add_subplot(g0[0, k])
        col = next((v for kk, v in HIST.items() if title.startswith(kk)), "#c8c8c8")
        if len(xs) > 1:
            q1, q3 = np.percentile(xs, [25, 75]); fd = 2 * (q3 - q1) * len(xs) ** (-1 / 3)
            bins = max(8, min(80, int((xs.max() - xs.min()) / fd)) if fd > 0 else 30)
            sns.histplot(xs, bins=bins, stat="density", ax=a, color=col, alpha=0.6)
        hmax = max([p.get_height() for p in a.patches]) if a.patches else 1.0
        dens = sample_density(x, [{"fit": dict(r["fit"], weights=[wc[i].tolist()])} for i, r in enumerate(recs)], 0)
        for c in range(dens.shape[1]):
            a.plot(x, np.nanpercentile(dens[:, c, :], 50, axis=0), color=f"C{c}", ls="--", lw=1, label=f"Comp {c + 1}")
        tot = np.nansum(dens, axis=1)
        p5, p50, p95 = np.percentile(tot, [5, 50, 95], axis=0)
        a.plot(x, p50, color="black", lw=1.5); a.fill_between(x, p5, p95, color="gray", alpha=0.3)
        wm = np.median(wc, axis=0)
        a.set_title(f"{kc}: {title}\n(n={nv:,d}; {ncell} cells; w=" + "/".join(f"{v:.2f}" for v in wm) + ")",
                    fontsize=9)
        a.set_xlim(xlim); a.set_ylim(0, hmax * 1.1); a.tick_params(labelsize=7.5); a.grid(lw=0.5, alpha=0.3)
        a.legend(fontsize=6.5)
    aP, aB, aS, aT = (fig.add_subplot(g1[0, i]) for i in range(4))
    pv = list(range(1, 9))
    for ax_, q, ttl in ((aP, "p5", "A  pathogenic evidence: log LR+, 5th percentile"),
                        (aB, "p95", "B  benign evidence: log LR+, 95th percentile")):
        for key, lab, colr, lw, ls in ARMS:
            if key in arms:
                c = arms[key]
                ax_.plot(np.asarray(c["lr"]["score_range"]), c["lr"][f"log_lr_plus_{q}"], color=colr, lw=lw, ls=ls,
                         alpha=0.45 if key == "std_all" else 1, label=lab, zorder=1 if key == "std_all" else 3)
        for key, ls in (("cell_all", "--"), ("cell_mis", ":")):
            if key in arms:
                tauP, tauB, _, _ = log_thresholds_with_ylim_pad(arms[key]["cal"]["prior"], pv,
                                                                acmg_mapping_method="lp_anchored")
                for th in (tauP if q == "p5" else tauB):
                    ax_.axhline(th, color=dict(ARMS_C)[key], ls=ls, lw=0.7, alpha=0.4)
        _, _, top, bot = log_thresholds_with_ylim_pad(arms["cell_all"]["cal"]["prior"], pv,
                                                      acmg_mapping_method="lp_anchored")
        ax_.set_ylim(bot, top); ax_.set_xlim(xlim); ax_.grid(lw=0.5, alpha=0.3); ax_.axhline(0, color="#52514e", lw=0.8)
        ax_.set_title(ttl + "\nthresholds: cell all P/LP dashed, cell missense dotted", fontsize=9.5, loc="left")
        ax_.set_xlabel("Score"); ax_.set_ylabel("log LR+ (natural log)")
    aP.legend(fontsize=7.5, loc="best")
    rows_, labs = [], []
    for key, lab, *_ in ARMS:
        if key in arms:
            rows_.append(_points(arms[key]["cal"], x)); labs.append(SHORT[key])
    cmap = ListedColormap(["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
                           "#f4f4f4", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"])
    im = aS.imshow(np.vstack(rows_), aspect="auto", cmap=cmap, norm=BoundaryNorm(np.arange(-8.5, 9.5, 1), cmap.N),
                   interpolation="nearest", extent=[x[0], x[-1], len(rows_) - 0.5, -0.5])
    aS.set_yticks(range(len(labs))); aS.set_yticklabels(labs, fontsize=8); aS.set_xlim(xlim); aS.set_xlabel("Score")
    aS.set_title("C  evidence strip: points along the score axis", fontsize=9.5, loc="left")
    cb = fig.colorbar(im, ax=aS, ticks=range(-8, 9, 2), fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7)
    # D: priors and points on missense variants (scores from the standard all-P/LP variants table)
    xm = _mis_scores(SENS / "medium_baseline_popall", ds)
    lines = ["D  prior / points on missense variants", ""]
    for key, lab, *_ in ARMS:
        if key not in arms:
            continue
        c = arms[key]["cal"]
        line = f"{SHORT[key]}\n  prior {c['prior']:.3f}"
        if xm is not None and len(xm):
            p = _points(c, xm)
            line += (f"   max level {p.max():+.0f}\n  pathogenic {np.mean(p > 0):.0%} of missense (mean {np.clip(p, 0, None).mean():.2f})"
                     f"\n  benign {np.mean(p < 0):.0%} of missense (mean {np.clip(-p, 0, None).mean():.2f})")
        lines.append(line)
    aT.axis("off")
    aT.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, transform=aT.transAxes)
    fig.suptitle(f"{_display_dataset(ds)} - membership-cell fit ({kc}, 100 bootstraps x 8 restarts; one fit, "
                 f"all composites) vs standard ExCALIBR\npoint mapping LP-anchored (pathogenic: 6 points = 0.90 "
                 f"posterior, prior capped at 0.5; benign: Tavtigian C = 350)", fontsize=12, x=0.04, ha="left", y=0.995)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{_display_dataset(ds)}_cellfit.png"
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return str(fn)


ARMS_C = [(k, c) for k, _, c, _, _ in ARMS]


def _safe(*a):
    try:
        return draw(*a)
    except Exception as e:
        import traceback
        return f"FAILED {a[0]}: {e!r}\n{traceback.format_exc()[-1500:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", default=str(ROOT / "cells" / "hpc_medium" / "labeled"))
    ap.add_argument("--out", default=str(ROOT / "figs_cells"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--config", default="src/igvf_configs/dataset_configs_aug_2026.json")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    cd = Path(args.cells_dir)
    cfg = json.load(open(args.config))
    dss = args.datasets or sorted(p.stem for p in (cd / "cells").glob("*.npz"))
    todo = []
    for ds in dss:
        e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
        kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"
        todo.append((ds, kc if kc in ("2c", "3c") else "3c"))
    res = Parallel(n_jobs=args.n_jobs)(delayed(_safe)(ds, kc, cd, args.out) for ds, kc in todo)
    for r in res:
        print(r if r else "skipped (missing fits or calibrations)")


if __name__ == "__main__":
    main()
