"""Per-dataset figures: baseline calibration vs post-fit missense-only P/LP.

Layout follows plot_scoreset_best_config (row 0 sample fits, row 1 point
assignments, row 2 log LR+), with an extra "P/LP (missense)" column right
after P/LP: the missense P/LP histogram and that sample's density from the
re-weighted fits (reweight_plp_missense.py). Rows 1 and 2 overlay both
calibrations in every column:
  all P/LP      -- the standard baseline calibration (solid);
  missense P/LP -- the calibration of the re-weighted fits (dashed), with
                   its own prior and thresholds.
In the combined scoring the pathogenic points come from the missense P/LP
calibration and the benign points from the standard one.

Usage:
  python -m analysis.missense_sensitivity.plot_plp_missense \
      --root /data/ross/assay_calibration/missense_only_calibrations \
      --out  /data/ross/assay_calibration/missense_only_calibrations/figs_baseline_plpmis
"""
from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PLP = "Pathogenic/Likely Pathogenic"
STYLES = {  # calibration -> (label, LR+ linestyle, row-1 y offset)
    "all": ("all P/LP", "-", -0.15),
    "mis": ("missense P/LP", "--", 0.15),
}
# (pathogenic, median, benign) colours per calibration: the missense P/LP
# calibration is shifted toward orange (pathogenic) and purple (benign).
COLORS = {
    "all": ("red", "black", "blue"),
    "mis": ("#E8731A", "#8C564B", "#7A4FC4"),
}


def _load(calib_dir: Path, ds: str):
    d = calib_dir / ds
    cals = sorted(d.glob(f"{ds}_*_calibration.json"))
    if not cals:
        return None
    cal = json.load(open(cals[0]))
    lr = json.load(gzip.open(str(cals[0]).replace("_calibration.json", "_lr_values.json.gz"), "rt"))
    return dict(cal=cal, lr=lr, comp_key=cals[0].name[len(ds) + 1:-len("_calibration.json")])


def _point_ranges(cal):
    return sorted([(int(k), v) for k, v in (cal.get("point_ranges") or {}).items()])


def _plot_dataset(ds, df_ds, base_fits_ds, mis_fits_ds, base, mis, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from analysis.missense_sensitivity.compare_runs import _missense_map
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from src.assay_calibration.plot_utils.utils import (
        sample_density, sample_display_name, sample_hist_style, log_thresholds_with_ylim_pad,
        reference_config_label, _display_dataset)

    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    pcfg = PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp", clinvar_release=release)
    ss = load_dataset_from_df(df_ds, pcfg)
    scores = np.asarray(ss.scores)
    sa = np.asarray(ss.sample_assignments)
    names = [n for n, c in zip(ss.sample_names, ss.sample_counts) if c > 0]
    mis_flag = _missense_map(ds, df_ds)["is_missense"].values

    nc = base["comp_key"].split("_")[0]
    fits_b = [r[nc] for r in base_fits_ds.values() if isinstance(r, dict) and r.get(nc)]
    fits_m = [r[nc] for r in mis_fits_ds.values() if isinstance(r, dict) and r.get(nc)]
    score_range = np.asarray(base["lr"]["score_range"])
    assert np.allclose(score_range, mis["lr"]["score_range"])

    # columns: (title, score mask, fits, weight index, style name)
    cols = []
    for j, n in enumerate(names):
        cols.append((sample_display_name(n), sa[:, j].astype(bool), fits_b, j, n))
        if n == PLP:
            cols.append(("P/LP (missense)", sa[:, j].astype(bool) & mis_flag, fits_m, j, PLP))
    ncol = len(cols)
    fig, ax = plt.subplots(3, ncol, figsize=(7 * ncol, 18), squeeze=False,
                           gridspec_kw={"hspace": 0.3, "wspace": 0.3})

    for k, (title, mask, fits, widx, style_name) in enumerate(cols):
        a = ax[0, k]
        x = scores[mask]
        if len(x) > 1:
            q1, q3 = np.percentile(x, [25, 75])
            fd = 2 * (q3 - q1) * len(x) ** (-1 / 3)
            bins = max(10, min(100, int((x.max() - x.min()) / fd)) if fd > 0 else 50)
        else:
            bins = 10
        col, alpha = sample_hist_style(style_name, widx)
        if title == "P/LP (missense)":
            col, alpha = "#8C3B4A", 0.6
        if len(x):
            sns.histplot(x, bins=bins, stat="density", ax=a, alpha=alpha, color=col)
        hmax = max([p.get_height() for p in a.patches]) if a.patches else 1.0
        dens = sample_density(score_range, fits, widx)
        for c in range(dens.shape[1]):
            a.plot(score_range, np.nanpercentile(dens[:, c, :], 50, axis=0), color=f"C{c}",
                   linestyle="--", label=f"Comp {c + 1}")
        tot = np.nansum(dens, axis=1)
        pc = np.percentile(tot, [5, 50, 95], axis=0)
        a.plot(score_range, pc[1], color="black", alpha=.5)
        a.fill_between(score_range, pc[0], pc[2], color="gray", alpha=0.3)
        a.set_title(f"{nc}: {title}\n(n={int(mask.sum()):,d})")
        a.set_xlabel("Score"); a.set_ylabel("Density")
        a.set_ylim([0, hmax * 1.1]); a.legend(fontsize=8)
        a.grid(linewidth=0.5, alpha=0.3)
    xlim = ax[0, 0].get_xlim()
    for k in range(1, ncol):
        ax[0, k].set_xlim(xlim)

    # ---- row 1: point assignments, both calibrations ----
    levels = sorted({p for r in (base, mis) for p, _ in _point_ranges(r["cal"])})
    ypos = {p: i for i, p in enumerate(levels)}
    # ---- row 2 thresholds per calibration ----
    pv_all = sorted({abs(p) for p in levels if p != 0}) or [1]
    th = {}
    for key, r in (("all", base), ("mis", mis)):
        th[key] = log_thresholds_with_ylim_pad(r["cal"]["prior"], pv_all)
    ytop = max(th["all"][2], th["mis"][2]); ybot = min(th["all"][3], th["mis"][3])

    for k in range(ncol):
        a1, a2 = ax[1, k], ax[2, k]
        for key, r in (("all", base), ("mis", mis)):
            lab, ls, off = STYLES[key]
            cp, cm, cb = COLORS[key]
            for pval, ranges in _point_ranges(r["cal"]):
                for lo, hi in ranges:
                    x0 = xlim[0] if np.isneginf(lo) else max(lo, xlim[0])
                    x1 = xlim[1] if np.isposinf(hi) else min(hi, xlim[1])
                    # Solid segments: dashes vanish on short point ranges.
                    a1.plot([x0, x1], [ypos[pval] + off] * 2, color=cp if pval > 0 else cb,
                            linestyle="-", alpha=0.9, linewidth=3, solid_capstyle="butt")
            a1.plot([], [], color=cp, linewidth=3, label=f"{lab} (pathogenic)")
            a1.plot([], [], color=cb, linewidth=3, label=f"{lab} (benign)")
            lr = r["lr"]
            for curve, c, nm in ((lr["log_lr_plus_p5"], cp, "5th"),
                                 (lr["log_lr_plus_p50"], cm, "median"),
                                 (lr["log_lr_plus_p95"], cb, "95th")):
                a2.plot(score_range, curve, color=c, linestyle=ls, linewidth=2,
                        alpha=1.0 if key == "all" else 0.85, label=f"{lab} {nm}")
            tauP, tauB, _, _ = th[key]
            for tp, tb in zip(tauP, tauB):
                a2.axhline(tp, color=cp, linestyle=":" if key == "mis" else "--", alpha=0.4)
                a2.axhline(tb, color=cb, linestyle=":" if key == "mis" else "--", alpha=0.4)
        a1.set_ylim(-1, len(levels)); a1.set_xlim(xlim)
        a1.set_yticks(range(len(levels)), labels=[f"{p:+d}" if p else "0" for p in levels])
        a1.set_xlabel("Score"); a1.set_ylabel("Points")
        a1.set_title("Point Assignments", fontsize=11); a1.legend(fontsize=8, loc="best")
        a1.grid(linewidth=0.5, alpha=0.3)
        pa = np.percentile(base["lr"].get("priors_pct", [base["cal"]["prior"]] * 3), 50)
        pm = np.percentile(mis["lr"].get("priors_pct", [mis["cal"]["prior"]] * 3), 50)
        a2.set_title(f"Log LR+\nprior: all P/LP {pa:.3f} | missense P/LP {pm:.3f}", fontsize=11)
        a2.set_xlabel("Score"); a2.set_ylabel("Log LR+")
        a2.set_xlim(xlim); a2.set_ylim([ybot, ytop])
        a2.legend(fontsize=7, loc="best", ncol=2); a2.grid(linewidth=0.5, alpha=0.3)

    label = reference_config_label(base["cal"].get("benign_method"), ss)
    fig.suptitle(f"{_display_dataset(ds)} - {nc} {label}: all P/LP (red/blue) vs missense P/LP (orange/purple)",
                 fontsize=16, y=0.998)
    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{_display_dataset(ds)}_{base['comp_key']}_plp_missense_visualization.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    return str(fn)


def _safe(*a):
    try:
        return _plot_dataset(*a)
    except Exception as e:  # one bad dataset must not stop the batch
        import traceback
        return f"FAILED {a[0]}: {e!r}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ross/assay_calibration/missense_only_calibrations")
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out or root / "figs_baseline_plpmis")
    jobs = root / "explorer_jobs_pp_large_baseline"

    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df

    fb = json.load(gzip.open(jobs / "bootstrap_results.json.gz"))
    fm = json.load(gzip.open(jobs / "bootstrap_results_plp_missense.json.gz"))
    todo = []
    for ds in sorted(args.datasets or fb):
        b, m = _load(root / "calib_baseline", ds), _load(root / "calib_baseline_plpmis_raw", ds)
        if b is None or m is None or ds not in fm:
            print(f"skip {ds}: missing calibration")
            continue
        todo.append((ds, b, m))
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds, _, _ in todo}
    del df_full
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_safe)(ds, slices[ds], fb[ds], fm[ds], b, m, out) for ds, b, m in todo)
    for r in res:
        if r.startswith("FAILED"):
            print(r)
    print(f"{sum(not r.startswith('FAILED') for r in res)}/{len(res)} figures in {out}")


if __name__ == "__main__":
    main()
