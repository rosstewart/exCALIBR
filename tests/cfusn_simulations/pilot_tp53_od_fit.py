"""PILOT: one TP53 tetramerization-domain-only (residues 319-360) MV fit, for
inspecting per-dimension marginal fits before any domain-specific
calibration run.

Data: the production TP53 MultiScoreset (raw TempSens, Kato 8-promoter panel
collapsed to 2 PCs via the "tp53_kato_pca2" preset -- PCA fit gene-wide,
then projected) with KawOligo as a categorical dim (location model,
EXPERIMENTAL), restricted to variants at residues 319-360. Dims with no or
negligible coverage in the domain are dropped: Funk_RFS / Kotler_RFS
(DNA-binding-domain-only assays, 0 variants here) and log_TempSens (9).

Fit: ONE fit, no bootstrap, first restart of the standard cycling (the
Gaussian restart), K components, standard generate_fit_jobs/execute_fit_job
path. Writes <out>/fit.pkl and <out>/marginals_od_K{K}.png: one row per dim,
one column per sample -- that sample's observed values (histogram; level
frequencies for KawOligo) against the fitted marginal (solid) and each
component's weighted contribution (dashed).

    nice -n 10 taskset -c 0-7 python tests/cfusn_simulations/pilot_tp53_od_fit.py --K 3
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import Fit  # noqa: E402
from src.assay_calibration.fit_utils.cfusn import categorical as cat  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.density_utils import _single_component_logpdf  # noqa: E402
from src.assay_calibration.multivariate_data import tp53  # noqa: E402
from src.assay_calibration.multivariate_data import redundancy_collapse as rc  # noqa: E402

OD_RANGE = (319, 360)
DROP_DIMS = ("Funk_RFS", "Kotler_RFS", "log_TempSens")
SAMPLE_COLOR = {"P/LP": "#CA7682", "B/LB": "#1D7AAB", "gnomAD": "#888888", "RPV": "#9B59B6"}
DEFAULT_OUT = _REPO_ROOT / "tests" / "cfusn_simulations" / "results" / "tp53_od_pilot"


def residue_positions(ids):
    return np.array([int(m.group()) if (m := re.search(r"\d+", str(s))) else -1 for s in ids])


def build_od_scoreset():
    ms = tp53.build_tp53_multiscoreset(kawoligo_mode="categorical")
    rc.apply_preset({"TP53": ms}, "tp53_kato_pca2", verbose=False)
    pos = residue_positions(ms._all_ids)
    rows = (pos >= OD_RANGE[0]) & (pos <= OD_RANGE[1])
    names = list(ms.dataset_names)
    keep = [j for j, n in enumerate(names) if n not in DROP_DIMS]
    return SimpleNamespace(
        scores=ms.scores[rows][:, keep],
        sample_assignments=ms.sample_assignments[rows],
        dataset_names=[names[j] for j in keep],
        sample_names=list(ms.sample_names),
        categorical_dims=dict(ms.categorical_dims),
        ids=np.asarray(ms._all_ids)[rows],
        scoreset_name="TP53_OD",
    )


def fit_once(ss, K, max_em_iters, seed):
    jobs = Fit(ss).generate_fit_jobs([K], num_fits=1, bootstrap=False, check_monotonic=False,
                                     master_seed=seed, bootstrap_seed=0, max_em_iters=max_em_iters,
                                     verbose_overlap=True)
    job = jobs[0]
    mode = "gaussian" if job["kwargs"].get("force_gaussian") else f"q{job['kwargs'].get('latent_q')}"
    print(f"restart mode: {mode}; train rows: {len(job['train_observations'])}; "
          f"calibrated dims: {job.get('calibrated_dims')}", flush=True)
    res = Fit.execute_fit_job(job)
    return job, res, mode


def marginal_logpdf(params, c, d_model, grid, p_cont):
    """Component c's marginal log-density on continuous model column d_model."""
    Xg = np.full((len(grid), p_cont), np.nan)
    Xg[:, d_model] = grid
    return np.atleast_1d(_single_component_logpdf(Xg, params[c], True)).ravel()


def _measured_curve(ss, fit, d, s, grid):
    """Fitted density on dim d for exactly the rows of sample s measured on d."""
    from src.assay_calibration.multivariate_analysis import visualize_fit as vf
    params = cat.wrap_params(list(fit["component_params"]), fit)
    S = np.asarray(fit["weights"]).shape[0]
    sa = ss.sample_assignments[:, :S].astype(bool)
    grids = [np.asarray(grid, dtype=float) if j == d else np.zeros(1) for j in range(ss.scores.shape[1])]
    sa_s = np.zeros_like(sa)
    sa_s[:, s] = sa[:, s]
    out = vf._measured_marginals_for_fit({"component_params": params, "weights": fit["weights"]},
                                         ss.scores, sa_s, grids)
    return np.exp(out[d][s])


def plot_marginals(ss, fit, K, mode, out_png, measured=False):
    """measured=True overlays the fitted mixture reweighted to exactly the
    variants in each histogram (mean component posterior over the rows where
    that dim is measured; visualize_fit._measured_marginals_for_fit), with the
    whole-sample marginal drawn dashed."""
    params, W = fit["component_params"], np.asarray(fit["weights"])
    spec = cat.CatSpec.from_dict(fit["cat_spec"]) if fit.get("cat_spec") else None
    names = ss.dataset_names
    cont_cols = spec.cont_positions if spec else list(range(len(names)))
    cat_cols = spec.positions if spec else []
    S = W.shape[0]
    snames = ss.sample_names[:S]
    sa = ss.sample_assignments[:, :S].astype(bool)
    D = len(names)
    comp_colors = plt.cm.tab10(np.arange(K))
    fig, axes = plt.subplots(D, S, figsize=(3.6 * S, 2.5 * D), squeeze=False)
    for d in range(D):
        col = ss.scores[:, d]
        for s, sname in enumerate(snames):
            ax = axes[d, s]
            color = SAMPLE_COLOR.get(sname, "k")
            obs = col[sa[:, s] & ~np.isnan(col)]
            if d in cat_cols:
                c_idx = cat_cols.index(d)
                levels = spec.level_names[c_idx]
                P = np.asarray(fit["cat_probs"][c_idx])
                x = np.arange(len(levels))
                if len(obs):
                    ax.bar(x, np.bincount(obs.astype(int), minlength=len(levels)) / len(obs), width=0.6,
                           color=color, alpha=0.3, label=f"observed (n={len(obs)})")
                ax.plot(x, W[s] @ P, "o--" if measured else "o-", color="k", lw=1.0 if measured else 1.5,
                        alpha=0.5 if measured else 1.0, label="fit (whole sample)" if measured else "fit")
                if measured and len(obs):
                    ax.plot(x, _measured_curve(ss, fit, d, s, x), "s-", color="C3", lw=1.6,
                            label=f"fit (measured {len(obs)})")
                for c in range(K):
                    ax.plot(x, W[s, c] * P[c], "x--", color=comp_colors[c], lw=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(levels, fontsize=7)
                ax.set_ylim(0, 1.05)
            else:
                d_model = cont_cols.index(d)
                allv = col[~np.isnan(col) & sa.any(axis=1)]
                lo, hi = np.percentile(allv, [0.5, 99.5])
                pad = 0.15 * (hi - lo)
                grid = np.linspace(lo - pad, hi + pad, 400)
                comp = np.array([np.exp(marginal_logpdf(params, c, d_model, grid, len(cont_cols)))
                                 for c in range(K)])
                mix = W[s] @ comp
                if len(obs):
                    ax.hist(obs, bins=np.linspace(grid[0], grid[-1], 30), density=True, color=color,
                            alpha=0.3, label=f"observed (n={len(obs)})")
                ax.plot(grid, mix, color="k", lw=1.0 if measured else 1.5, ls="--" if measured else "-",
                        alpha=0.5 if measured else 1.0, label="fit (whole sample)" if measured else "fit")
                if measured and len(obs):
                    mc = _measured_curve(ss, fit, d, s, grid)
                    ax.plot(grid, mc, color="C3", lw=1.6, label=f"fit (measured {len(obs)})")
                    mix = np.maximum(mix, mc)
                for c in range(K):
                    if W[s, c] > 1e-3:
                        ax.plot(grid, W[s, c] * comp[c], "--", color=comp_colors[c], lw=0.9)
                hist_max = np.histogram(obs, bins=np.linspace(grid[0], grid[-1], 30), density=True)[0].max() if len(obs) else 0
                ax.set_ylim(0, 1.3 * max(hist_max, np.nanmax(mix)) if max(hist_max, np.nanmax(mix)) > 0 else 1)
            if d == 0:
                ax.set_title(f"{sname} (n={int(sa[:, s].sum())})", fontsize=10, fontweight="bold", color=color)
            if s == 0:
                ax.set_ylabel(names[d] + ("\n(categorical)" if d in cat_cols else ""), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")
    handles = [plt.Line2D([], [], color=comp_colors[c], ls="--", label=f"comp {c + 1}") for c in range(K)]
    fig.legend(handles=handles, loc="upper center", ncol=K, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"TP53 tetramerization domain ({OD_RANGE[0]}–{OD_RANGE[1]}) only — single {mode} fit, K={K}, "
                 f"no bootstrap\nsolid: fitted per-sample marginal; dashed: weighted components", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max-em-iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ss = build_od_scoreset()
    sa = ss.sample_assignments.astype(bool)
    print(f"OD variants: {len(ss.scores)}; per sample: "
          + ", ".join(f"{n}={int(sa[:, i].sum())}" for i, n in enumerate(ss.sample_names[:sa.shape[1]])))
    print("dims:", ss.dataset_names)
    job, res, mode = fit_once(ss, args.K, args.max_em_iters, args.seed)
    fit = res["fit"]
    print(f"val_ll={res.get('val_ll')}  converged/iters: {fit.get('n_iter', fit.get('iterations'))}")
    print("weights:\n", np.round(np.asarray(fit["weights"]), 3))
    if fit.get("cat_probs") is not None:
        print("KawOligo pi per component:\n", np.round(np.asarray(fit["cat_probs"][0]), 3))
    with open(args.out / f"fit_od_K{args.K}.pkl", "wb") as fh:
        pickle.dump({"res": res, "dataset_names": ss.dataset_names, "sample_names": ss.sample_names,
                     "mode": mode, "K": args.K}, fh)
    png = args.out / f"marginals_od_K{args.K}.png"
    plot_marginals(ss, fit, args.K, mode, png)
    print("wrote", png)


if __name__ == "__main__":
    main()
