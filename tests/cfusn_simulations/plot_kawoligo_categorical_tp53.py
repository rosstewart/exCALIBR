"""Figures for the TP53 KawOligo jitter-vs-categorical comparison.

Reads the per-(config, bootstrap, restart) checkpoints written by
sim_kawoligo_categorical_tp53.py (restart selection = best re-scored val_ll,
as in its analyze mode) and writes to <out>/figures/:

  fig1_kawoligo_data_and_fit.png  -- per sample: the data each model sees
      (jittered values vs integer levels) against each model's fitted
      KawOligo distribution (median over bootstraps, 5-95% range)
  fig2_kawoligo_evidence.png      -- evidence: KawOligo-only marginal log LR+
      per model, and the per-variant ACMG points for every labeled variant
      with KawOligo observed (jitter vs categorical vs KawOligo dropped)
  fig3_kawoligo_metrics.png       -- the held-out comparison metrics
  fig4_kawoligo_per_bootstrap_lr.png -- per-bootstrap log LR+ behind each
      labeled variant's points (shows when one bootstrap's fit drives them)
  kawoligo_variant_points.csv     -- per-variant LR/points table

Points follow the production recipe in a SIMPLIFIED form: log LR+ =
log f_P/LP - log f_B/LB on all observed dims, prior per bootstrap from
estimate_prior_from_class_densities on gnomAD rows (mode 'standard'),
Tavtigian thresholds at the median prior, points from the 5th (pathogenic)
/ 95th (benign) bootstrap percentile of log LR+. No monotonicity pass,
density cap or benign_method mixing -- so these are for comparing the
models with each other, not the production calls.

    taskset -c 0-15 python tests/cfusn_simulations/plot_kawoligo_categorical_tp53.py
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import sim_kawoligo_categorical_tp53 as K  # noqa: E402
from src.assay_calibration.fit_utils.cfusn import categorical as cat  # noqa: E402
from src.assay_calibration.fit_utils.point_ranges import estimate_prior_from_class_densities  # noqa: E402
from src.assay_calibration.multivariate_analysis.mv_calibration import _thresholds  # noqa: E402

LEVELS = ["Tetramer", "Dimer", "Monomer"]
CFG_LABEL = {"jitter": "Jitter (continuous)", "cat_tau3": "Categorical τ=3",
             "cat_tau1": "Categorical τ=1", "dropped": "KawOligo dropped"}
CFG_COLOR = {"jitter": "#d95f02", "cat_tau3": "#1b9e77", "cat_tau1": "#66a61e", "dropped": "#7570b3"}
SAMPLE_COLOR = {"P/LP": "#CA7682", "B/LB": "#1D7AAB", "gnomAD": "#888888", "RPV": "#9B59B6",
                "unlabeled": "#cccccc"}
POINT_VALUES = list(range(1, 9))
GRID = np.linspace(-0.6, 1.6, 441)


def load_best(out_dir, n_boot):
    K._install_single_row_fix()  # no-op on top of the production fix; kept for old checkpoints
    best = {}
    for b in range(n_boot):
        for config in K.CONFIGS:
            entries = [pickle.load(open(out_dir / f"{config}_b{b}_f{f}.pkl", "rb"))
                       for f in range(K.NUM_FITS)
                       if (out_dir / f"{config}_b{b}_f{f}.pkl").exists()]
            best[(config, b)] = K._best(entries)
    return best


def log_f(entry, config, X, w_row, kaw_col):
    """log mixture density of rows X under class weights w_row (all observed dims)."""
    fit = entry["res"]["fit"]
    params = fit["component_params"]
    if config.startswith("cat"):
        spec = cat.CatSpec.from_dict(fit["cat_spec"])
        Xc, labels, has_cont = cat.split_observations(X, spec)
        lp = np.zeros((len(params), len(X)))
        if has_cont.any():
            lp[:, has_cont] = K._comp_logpdfs(Xc[has_cont], params)
        lp += cat.cat_log_pdfs(labels, [np.asarray(P) for P in fit["cat_probs"]])
    else:
        if config == "dropped":
            X = np.delete(X, kaw_col, axis=1)
        ok = ~np.all(np.isnan(X), axis=1)
        lp = np.zeros((len(params), len(X)))
        if ok.any():
            lp[:, ok] = K._comp_logpdfs(X[ok], params)
    with np.errstate(divide="ignore"):
        return K._logsumexp(lp + np.log(np.asarray(w_row))[:, None])


def jitter_marginal(entry, kaw_col, n_dims):
    """(S, G) fitted KawOligo marginal density per class on GRID (jitter fits)."""
    fit = entry["res"]["fit"]
    Xg = np.full((len(GRID), n_dims), np.nan)
    Xg[:, kaw_col] = GRID
    dens_k = np.exp(K._comp_logpdfs(Xg, fit["component_params"]))
    return np.asarray(fit["weights"]) @ dens_k


def row_label(sa_row, names):
    for name in ("P/LP", "B/LB", "RPV", "gnomAD"):
        if name in names and sa_row[names.index(name)]:
            return name
    return "unlabeled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=K.DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=K.N_BOOT)
    args = ap.parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    ms_c, ms_j = K.build_ms("categorical"), K.build_ms("jitter")
    names = list(ms_c.sample_names)
    S = len(names)
    kc = list(ms_c.dataset_names).index(K.KAW)
    D = ms_c.scores.shape[1]
    sa = ms_c.sample_assignments.astype(bool)
    kaw_obs = ~np.isnan(ms_c.scores[:, kc])
    best = load_best(args.out, args.n_boot)
    boots = range(args.n_boot)

    # ── Figure 1: data + fitted distribution per sample ────────────────────
    jit_marg = np.array([jitter_marginal(best[("jitter", b)], kc, D) for b in boots])   # (B, S, G)
    bins = np.array(K.JITTER_BINS[1:-1])
    jit_pmf = np.array([K._class_pmf(best[("jitter", b)], "jitter", kc, S) for b in boots])  # (B, S, 3)
    cat_pmf = {c: np.array([K._class_pmf(best[(c, b)], c, kc, S) for b in boots])
               for c in ("cat_tau3", "cat_tau1")}

    fig, axes = plt.subplots(2, S, figsize=(4.0 * S, 7.0), squeeze=False)
    for s, name in enumerate(names):
        rows = sa[:, s] & kaw_obs
        n = int(rows.sum())
        # top: jitter -- the continuous values the model is fit to
        ax = axes[0, s]
        vals = ms_j.scores[rows, kc]
        if n:
            ax.hist(vals, bins=np.linspace(GRID[0], GRID[-1], 45), density=True,
                    color=SAMPLE_COLOR.get(name, "gray"), alpha=0.35, label=f"jittered data (n={n})")
        med = np.median(jit_marg[:, s], axis=0)
        lo, hi = np.percentile(jit_marg[:, s], [5, 95], axis=0)
        ax.plot(GRID, med, color=CFG_COLOR["jitter"], lw=1.8, label="jitter fit (median)")
        ax.fill_between(GRID, lo, hi, color=CFG_COLOR["jitter"], alpha=0.15, label="5–95% over boots")
        for x in bins:
            ax.axvline(x, color="k", ls=":", lw=0.6)
        for x, lev in zip((0.0, 0.5, 1.0), LEVELS):
            ax.text(x, 1.02, lev, transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
        ax.set_title(f"{name}", fontsize=10, fontweight="bold", color=SAMPLE_COLOR.get(name, "k"), pad=14)
        ax.set_xlabel("KawOligo (jittered code)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_ylim(0, max(5.0, 1.1 * float(np.nanmax(hi))))
        ax.legend(fontsize=6, loc="upper right")
        # bottom: categorical -- the level counts, vs each model's pmf
        ax = axes[1, s]
        counts = np.bincount(ms_c.scores[rows, kc].astype(int), minlength=3)
        freq = counts / max(n, 1)
        x = np.arange(3)
        ax.bar(x, freq, width=0.7, color=SAMPLE_COLOR.get(name, "gray"), alpha=0.35,
               label=f"observed (n={n})")
        for xi, cnt in zip(x, counts):
            ax.text(xi, 0.01, str(cnt), ha="center", va="bottom", fontsize=7)
        for off, (c, pm) in zip((-0.2, 0.0, 0.2), (("jitter", jit_pmf), ("cat_tau3", cat_pmf["cat_tau3"]),
                                                  ("cat_tau1", cat_pmf["cat_tau1"]))):
            med = np.median(pm[:, s], axis=0)
            lo, hi = np.percentile(pm[:, s], [5, 95], axis=0)
            ax.errorbar(x + off, med, yerr=[med - lo, hi - med], fmt="o", ms=4, capsize=2,
                        color=CFG_COLOR[c], label=CFG_LABEL[c] + (" (binned)" if c == "jitter" else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(level | sample)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("TP53 KawOligo — what each model sees and fits, per sample\n"
                 "top: jittered {0, 0.5, 1} + N(0, 0.1²) values and the jitter model's fitted marginal; "
                 "bottom: observed level frequencies vs fitted per-sample pmf (jitter integrated over the dotted bins)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "fig1_kawoligo_data_and_fit.png", dpi=150)
    plt.close(fig)

    # ── Evidence: per-variant log LR+ and points ───────────────────────────
    ev_rows = np.where(kaw_obs)[0]
    pop_rows = np.where(sa[:, names.index("gnomAD")])[0]
    p_idx, b_idx = names.index("P/LP"), names.index("B/LB")
    res = {}
    for config in K.CONFIGS:
        X = ms_j.scores if config == "jitter" else ms_c.scores
        lr_mat, priors = [], []
        for b in boots:
            e = best[(config, b)]
            W = np.asarray(e["res"]["fit"]["weights"])
            lfp = log_f(e, config, X[pop_rows], W[p_idx], kc)
            lfb = log_f(e, config, X[pop_rows], W[b_idx], kc)
            ok = np.isfinite(lfp) & np.isfinite(lfb)
            priors.append(estimate_prior_from_class_densities(np.exp(lfp[ok]), np.exp(lfb[ok]), mode="standard"))
            lr_mat.append(log_f(e, config, X[ev_rows], W[p_idx], kc) - log_f(e, config, X[ev_rows], W[b_idx], kc))
        lr_mat = np.array(lr_mat)
        prior = float(np.nanmedian(priors))
        tau_p, tau_b, _, _ = _thresholds(prior, POINT_VALUES)
        lr5, lr95 = np.nanpercentile(lr_mat, 5, axis=0), np.nanpercentile(lr_mat, 95, axis=0)
        pts = np.zeros(len(ev_rows), dtype=int)
        for pv in POINT_VALUES:
            pts[lr5 >= tau_p[pv - 1]] = pv
        for pv in POINT_VALUES:
            pts[lr95 <= tau_b[pv - 1]] = -pv
        res[config] = dict(lr_mat=lr_mat, lr_med=np.nanmedian(lr_mat, axis=0), lr5=lr5, lr95=lr95, points=pts,
                           prior=prior, tau_p=tau_p, tau_b=tau_b)
        print(f"{config:9s} prior={prior:.3f}  points: {dict(zip(*np.unique(pts, return_counts=True)))}")

    labels = np.array([row_label(sa[r], names) for r in ev_rows])
    levels = ms_c.scores[ev_rows, kc].astype(int)
    var_ids = np.asarray(ms_c._all_ids)
    table = pd.DataFrame({"variant": var_ids[ev_rows],
                          "sample": labels, "KawOligo": [LEVELS[l] for l in levels],
                          "n_cont_dims_observed": (~np.isnan(np.delete(ms_c.scores[ev_rows], kc, axis=1))).sum(1)})
    for config in K.CONFIGS:
        table[f"{config}_logLR_median"] = res[config]["lr_med"]
        table[f"{config}_logLR_p5"] = res[config]["lr5"]
        table[f"{config}_logLR_p95"] = res[config]["lr95"]
        table[f"{config}_points"] = res[config]["points"]
    table.to_csv(fig_dir / "kawoligo_variant_points.csv", index=False)

    # KawOligo-only marginal log LR+
    def marg_lr_cat(config):
        out = []
        for b in boots:
            fit = best[(config, b)]["res"]["fit"]
            W, P = np.asarray(fit["weights"]), np.asarray(fit["cat_probs"][0])
            out.append(np.log(W[p_idx] @ P) - np.log(W[b_idx] @ P))
        return np.array(out)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0:2])
    jm = np.log(jit_marg[:, p_idx]) - np.log(jit_marg[:, b_idx])
    med = np.median(jm, axis=0)
    lo, hi = np.percentile(jm, [5, 95], axis=0)
    ax.plot(GRID, med, color=CFG_COLOR["jitter"], lw=1.8, label="jitter: log f_P/LP(x)/f_B/LB(x), KawOligo alone")
    ax.fill_between(GRID, lo, hi, color=CFG_COLOR["jitter"], alpha=0.15)
    for s_name, yrow in (("P/LP", 1), ("B/LB", -1)):
        r = sa[:, names.index(s_name)] & kaw_obs
        ax.plot(ms_j.scores[r, kc], np.full(r.sum(), yrow * 3.6), "|", ms=10, mew=1.2,
                color=SAMPLE_COLOR[s_name], label=f"{s_name} jittered values")
    for x in bins:
        ax.axvline(x, color="k", ls=":", lw=0.6)
    for c, off in (("cat_tau3", -0.04), ("cat_tau1", 0.04)):
        m = marg_lr_cat(c)
        mm = np.median(m, axis=0)
        ax.errorbar(np.array([0.0, 0.5, 1.0]) + off, mm,
                    yerr=[mm - np.percentile(m, 5, axis=0), np.percentile(m, 95, axis=0) - mm],
                    fmt="s", ms=6, capsize=3, color=CFG_COLOR[c], label=f"{CFG_LABEL[c]}: log P(level|P/LP)/P(level|B/LB)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-4, 4)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel("KawOligo-only log LR+", fontsize=9)
    ax.set_title("Evidence carried by KawOligo alone (median, 5–95% over bootstraps)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    # per-variant log LR+ by level, labeled variants only
    lab = np.isin(labels, ["P/LP", "B/LB", "RPV"])
    ax = fig.add_subplot(gs[0, 2:4])
    rng = np.random.default_rng(0)
    show = ("jitter", "cat_tau3", "dropped")
    for ci, config in enumerate(show):
        for li in range(3):
            for s_name in ("P/LP", "B/LB", "RPV"):
                m = lab & (labels == s_name) & (levels == li)
                if not m.any():
                    continue
                xpos = li * 4 + ci + rng.uniform(-0.25, 0.25, m.sum())
                ax.scatter(xpos, res[config]["lr_med"][m], s=16, color=SAMPLE_COLOR[s_name],
                           edgecolor=CFG_COLOR[config], linewidth=0.9, alpha=0.85)
    short = {"jitter": "jitter", "cat_tau3": "cat τ=3", "dropped": "dropped"}
    ax.set_xticks([li * 4 + ci for li in range(3) for ci in range(len(show))])
    ax.set_xticklabels([short[c] for _ in range(3) for c in show], fontsize=7, rotation=30)
    for t, c in zip(ax.get_xticklabels(), [c for _ in range(3) for c in show]):
        t.set_color(CFG_COLOR[c])
    for li in range(3):
        ax.text(li * 4 + 1, 1.01, LEVELS[li], transform=ax.get_xaxis_transform(), ha="center",
                fontsize=9, fontweight="bold")
    ax.set_yscale("symlog", linthresh=5)
    ax.axhline(0, color="gray", lw=0.8)
    for pv in (1, 4, 8):
        ax.axhline(res["cat_tau3"]["tau_p"][pv - 1], color="red", ls=":", lw=0.6)
        ax.axhline(res["cat_tau3"]["tau_b"][pv - 1], color="blue", ls=":", lw=0.6)
    for s_name in ("P/LP", "B/LB", "RPV"):
        ax.scatter([], [], color=SAMPLE_COLOR[s_name], label=s_name)
    ax.legend(fontsize=7, loc="upper left", title="fill = sample; edge = model", title_fontsize=7)
    ax.set_ylabel("full-row log LR+ (median over bootstraps)", fontsize=9)
    ax.set_title("Labeled variants with KawOligo observed: full-row evidence by level and model\n"
                 "(dotted: ±1/±4/±8-point thresholds at the categorical τ=3 prior)", fontsize=10, fontweight="bold", pad=22)

    # points: paired comparisons + distributions
    for k, (cx, cy) in enumerate((("jitter", "cat_tau3"), ("dropped", "cat_tau3"))):
        ax = fig.add_subplot(gs[1, k])
        for s_name in ("P/LP", "B/LB", "RPV"):
            m = lab & (labels == s_name)
            for li, mk in zip(range(3), ("o", "s", "^")):
                mm = m & (levels == li)
                if mm.any():
                    ax.scatter(res[cx]["points"][mm] + rng.uniform(-0.25, 0.25, mm.sum()),
                               res[cy]["points"][mm] + rng.uniform(-0.25, 0.25, mm.sum()),
                               s=22, marker=mk, color=SAMPLE_COLOR[s_name], alpha=0.8, edgecolor="k", linewidth=0.3)
        ax.plot([-8, 8], [-8, 8], color="gray", lw=0.6, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-8.7, 8.7)
        ax.set_ylim(-8.7, 8.7)
        ax.set_xlabel(f"{CFG_LABEL[cx]} points", fontsize=9)
        ax.set_ylabel(f"{CFG_LABEL[cy]} points", fontsize=9)
        ax.set_title(f"Per-variant points: {CFG_LABEL[cy]} vs {CFG_LABEL[cx]}", fontsize=9, fontweight="bold")
        if k == 0:
            for li, mk in zip(range(3), ("o", "s", "^")):
                ax.scatter([], [], marker=mk, color="w", edgecolor="k", label=LEVELS[li])
            for s_name in ("P/LP", "B/LB", "RPV"):
                ax.scatter([], [], color=SAMPLE_COLOR[s_name], label=s_name)
            ax.legend(fontsize=6, loc="upper left")
    ax = fig.add_subplot(gs[1, 2:4])
    width = 0.2
    groups = ["P/LP", "B/LB", "RPV"]
    for ci, config in enumerate(("jitter", "cat_tau3", "cat_tau1", "dropped")):
        for gi, s_name in enumerate(groups):
            p = res[config]["points"][lab & (labels == s_name)]
            if not len(p):
                continue
            x0 = gi + (ci - 1.5) * width
            correct = (p > 0).mean() if s_name in ("P/LP", "RPV") else (p < 0).mean()
            wrong = (p < 0).mean() if s_name in ("P/LP", "RPV") else (p > 0).mean()
            ax.bar(x0, correct, width, color=CFG_COLOR[config], label=CFG_LABEL[config] if gi == 0 else None)
            ax.bar(x0, -wrong, width, color=CFG_COLOR[config], alpha=0.45, hatch="//")
            ax.text(x0, correct + 0.02, f"{np.mean(np.abs(p)):.1f}", ha="center", fontsize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={int((lab & (labels == g)).sum())})" for g in groups])
    ax.set_ylabel("fraction with correct-direction evidence (up)\n/ wrong-direction evidence (down, hatched)", fontsize=8)
    ax.set_ylim(-0.6, 1.1)
    ax.set_title("Evidence direction per sample (label above bar = mean |points|)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("TP53 KawOligo — evidence given for observed variants (simplified production scoring, "
                 f"{args.n_boot} bootstraps; priors: " +
                 ", ".join(f"{CFG_LABEL[c].split(' (')[0]} {res[c]['prior']:.2f}" for c in K.CONFIGS) + ")",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "fig2_kawoligo_evidence.png", dpi=150)
    plt.close(fig)

    # ── Figure 4: per-bootstrap evidence for the labeled variants ─────────
    order = np.lexsort((levels[lab], labels[lab]))
    lab_idx = np.where(lab)[0][order]
    cfg4 = ("jitter", "cat_tau3", "cat_tau1", "dropped")
    fig, axes = plt.subplots(1, len(cfg4), figsize=(4.2 * len(cfg4), 0.22 * len(lab_idx) + 2.5), sharey=True)
    for ax, config in zip(axes, cfg4):
        M = res[config]["lr_mat"][:, lab_idx].T                      # (variants, boots)
        im = ax.imshow(np.sign(M) * np.log10(1 + np.abs(M)), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}" if abs(M[i, j]) >= 10 else f"{M[i, j]:.1f}",
                        ha="center", va="center", fontsize=4.5)
            ax.text(M.shape[1] - 0.3, i, f" {res[config]['points'][lab_idx[i]]:+d}", va="center",
                    fontsize=7, fontweight="bold", clip_on=False, transform=ax.transData)
        ax.set_xticks(range(M.shape[1]))
        ax.set_xlabel("bootstrap", fontsize=8)
        ax.set_title(f"{CFG_LABEL[config]}\n(right: points)", fontsize=9, fontweight="bold",
                     color=CFG_COLOR[config])
    axes[0].set_yticks(range(len(lab_idx)))
    axes[0].set_yticklabels([f"{var_ids[ev_rows[i]]}  {labels[i]}  {LEVELS[levels[i]]}" for i in lab_idx],
                            fontsize=6)
    for t, i in zip(axes[0].get_yticklabels(), lab_idx):
        t.set_color(SAMPLE_COLOR[labels[i]])
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.04)
    cb.set_label("sign · log10(1 + |log LR+|)", fontsize=8)
    fig.suptitle("Per-bootstrap full-row log LR+ for labeled variants with KawOligo observed "
                 "(points use the 5th/95th percentile over bootstraps)", fontsize=10)
    fig.savefig(fig_dir / "fig4_kawoligo_per_bootstrap_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: metrics ──────────────────────────────────────────────────
    d = pd.read_csv(args.out / "metrics.csv")
    metrics = [("M2_logscore", "M2 log-score ↑\nheld-out log P(level | cont. dims, class)"),
               ("M2_brier", "M2 Brier ↓"),
               ("M3_tv_oob", "M3 TV(pmf, observed) ↓\n(held-out)"),
               ("M1_cont_ll", "M1 continuous-dim\nheld-out LL / row ↑"),
               ("M4_auroc_plp_blb", "M4 P/LP vs B/LB AUROC ↑")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2))
    for ax, (m, title) in zip(axes, metrics):
        t = d.pivot_table(index="bootstrap", columns="config", values=m)
        cfgs = [c for c in K.CONFIGS if c in t and t[c].notna().any()]
        for i, c in enumerate(cfgs):
            ax.scatter(np.full(len(t), i) + np.linspace(-0.15, 0.15, len(t)), t[c], s=12,
                       color=CFG_COLOR[c], alpha=0.6)
            ax.errorbar(i + 0.3, t[c].mean(), yerr=t[c].std(ddof=1) / np.sqrt(t[c].notna().sum()),
                        fmt="D", color=CFG_COLOR[c], ms=6, capsize=3)
        offs = np.linspace(-0.15, 0.15, len(t))
        for oi, bi in enumerate(t.index):  # paired lines, through each bootstrap's own dots
            ax.plot(np.arange(len(cfgs)) + offs[oi], [t.loc[bi, c] for c in cfgs], color="gray", lw=0.3, alpha=0.5)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([CFG_LABEL[c].replace(" (continuous)", "").replace("Categorical", "Cat")
                            for c in cfgs], rotation=30, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(lw=0.2, alpha=0.4)
    fig.suptitle(f"TP53 KawOligo — held-out comparison over {d.bootstrap.nunique()} paired bootstraps "
                 "(dots: bootstraps, lines: pairing, diamond: mean ± SE)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(fig_dir / "fig3_kawoligo_metrics.png", dpi=150)
    plt.close(fig)
    print("wrote", fig_dir)


if __name__ == "__main__":
    main()
