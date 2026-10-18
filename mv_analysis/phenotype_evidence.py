#!/usr/bin/env python3
"""
Disease-phenotype-specific MV evidence visualization, for genes where an
auxiliary PATHOGENIC sample stands in for a distinct disease rather than a
general pathogenic population -- RET (MEN2 vs Hirschsprung disease) and
CARD11 (BENTA vs CADINS) are the SAME mechanism and share this module.

NOT the same concept as TP53's RPV (reduced-penetrance) sample -- RPV is
"reduced penetrance relative to the pathogenic/benign controls," not
"pathogenic evidence for a different disease," and already has its own
custom computation/figures; see mv_analysis/tp53_rpv_report.py instead.

Deliberately uses "phenotype"/"disease group" naming throughout, rather
than this codebase's more generic `auxiliary_pathogenic_indices` term
(unchanged elsewhere -- widely used in production code, not renamed here),
to avoid confusing the two concepts.
"""
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mv_analysis.gene_3d_evidence import (
    _compute_3d_evidence_data, _render_3d_evidence_row, _evidence_style, POINT_CMAP,
)
from mv_analysis.gene_performance_scatter import fast_results_json, RUN_KWARGS
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis

# ---------------------------------------------------------------------------
# RET: MEN2A/MEN2B (activating, pathway-activity-increasing) vs Hirschsprung
# (loss-of-function, pathway-activity-decreasing) germline variants --
# hand-curated variant lists (ported from this session's exploratory RET
# temp-fit script), matched against ms.kept_variants via the same
# protein-HGVS-suffix convention analysis/vus_reclassification.py's
# build_hgvs3_index uses for LABEL-seq genes without genomic-coordinate
# hgvs_c parsing.
# ---------------------------------------------------------------------------
_AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
    "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
    "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter",
}

RET_MEN2B_VARIANTS = ["M918T", "A883F"]
RET_MEN2A_VARIANTS = ["S891A", "E768D", "L790F", "V804L", "K666E", "V804M", "M918V"]
RET_HIRSCHSPRUNG_VARIANTS = ["L963P", "T946I", "R969W", "A877P", "R972G", "R873W", "S891L"]


def _short_to_hgvs3(v: str) -> str:
    ref, pos, alt = v[0], v[1:-1], v[-1]
    return f"p.{_AA_1TO3[ref]}{pos}{_AA_1TO3[alt]}"


def build_hgvs3_index(ms) -> Dict[str, int]:
    """{protein-HGVS suffix -> row index}, matching
    analysis.vus_reclassification.build_hgvs3_index's convention for
    LABEL-seq genes without genomic coordinates."""
    idx = {}
    for i, kv in enumerate(ms.kept_variants):
        s = kv[0] if isinstance(kv, tuple) else kv
        idx[s.split(":", 1)[1] if ":" in s else s] = i
    return idx


def ret_phenotype_masks(ms) -> Dict[str, np.ndarray]:
    """{"MEN2": mask, "Hirschsprung": mask} over ms's rows, from the
    hand-curated variant lists above."""
    hgvs3_index = build_hgvs3_index(ms)
    n = ms.scores.shape[0]

    def _mask(variants):
        mask = np.zeros(n, dtype=bool)
        for v in variants:
            idx = hgvs3_index.get(_short_to_hgvs3(v))
            if idx is not None:
                mask[idx] = True
        return mask

    return {
        "MEN2": _mask(RET_MEN2A_VARIANTS + RET_MEN2B_VARIANTS),
        "Hirschsprung": _mask(RET_HIRSCHSPRUNG_VARIANTS),
    }


def card11_phenotype_masks(ms, benta_role: int = 4, cadins_role: int = 5) -> Dict[str, np.ndarray]:
    """{"BENTA": mask, "CADINS": mask} from the same fixed auxiliary-role
    indices `analyze_card11` already uses (`auxiliary_pathogenic_indices=
    [4, 5]`) -- no hand-typed variant list needed, unlike RET. Unlike RET's
    MEN2/Hirschsprung (a sub-grouping WITHIN the pooled P/LP role), BENTA/
    CADINS are their OWN sample roles, entirely separate from role-0 P/LP --
    see `plot_phenotype_evidence`'s `replace_role=None` for this case."""
    sa = ms._sample_assignments.astype(bool)
    return {"BENTA": sa[:, benta_role].copy(), "CADINS": sa[:, cadins_role].copy()}


def plot_card11_phenotype_evidence(
    ms, results_json: str, config: str = "4c_unc",
    dataset_name: str = "CARD11_card11_mv",
    save_path: Optional[str] = None, run_kwargs=None,
):
    """2D evidence scatter (LOF score x GOF score) for CARD11 -- unlike RET,
    CARD11 only has 2 assay dimensions, so the 3D-only machinery in
    gene_3d_evidence.py doesn't apply. One panel per sample class (P/LP,
    B/LB, gnomAD, Synonymous, BENTA, CADINS), same evidence-point coloring/
    sizing as the 3D plots (`_evidence_style`/`POINT_CMAP`) for visual
    consistency across the paper's figures.
    """
    x_i = [i for i, n in enumerate(ms.dataset_names) if "LOF" in n][0]
    y_i = [i for i, n in enumerate(ms.dataset_names) if "GOF" in n][0]
    x_raw, y_raw = ms.scores[:, x_i], ms.scores[:, y_i]
    complete = ~(np.isnan(x_raw) | np.isnan(y_raw))

    kwargs = {**RUN_KWARGS, **(run_kwargs or {})}
    with fast_results_json(results_json):
        analysis = build_gene_set_analysis(ms, "card11", results_json, dataset_name=dataset_name)
        analysis.run(partial_pattern_mode="trust_global", **kwargs)
    points = np.asarray(analysis.results[config]["points"], dtype=float)
    max_pt = max(analysis.point_values)
    pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)

    sa = ms._sample_assignments.astype(bool)
    panels = [("P/LP", sa[:, 0]), ("B/LB", sa[:, 1]), ("gnomAD", sa[:, 2]), ("Synonymous", sa[:, 3])]
    phenotype_masks = card11_phenotype_masks(ms)
    panels += [(f"CARD11: {name}", mask) for name, mask in phenotype_masks.items()]
    panels = [(name, mask & complete) for name, mask in panels if (mask & complete).sum() > 0]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, mask) in zip(axes, panels):
        face, edge, sizes = _evidence_style(points[mask], pt_norm, max_pt)
        ax.scatter(x_raw[mask], y_raw[mask], c=face, edgecolors=edge, linewidths=0.3, s=sizes)
        ax.set_title(f"{name} (n={mask.sum()})", fontsize=10)
        ax.set_xlabel("LOF score")
    axes[0].set_ylabel("GOF score")
    fig.suptitle("CARD11: MV evidence by disease phenotype", fontsize=12)
    mappable = plt.cm.ScalarMappable(norm=pt_norm, cmap=POINT_CMAP)
    cbar = fig.colorbar(mappable, ax=list(axes), shrink=0.8, pad=0.02)
    cbar.set_label("Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved CARD11 phenotype evidence plot to {save_path}")
    return fig


def plot_phenotype_evidence(
    gene: str, x_dim: str, y_dim: str, z_dim: str, results_json: str,
    phenotype_masks: Dict[str, np.ndarray],
    replace_role: Optional[str] = "P/LP",
    config: str = "4c_unc", z_log10: bool = True,
    elev: float = 20.0, azim: float = -60.0,
    save_path: Optional[str] = None, ms_map=None, run_kwargs=None, dataset_name=None,
):
    """Same rendering as gene_3d_evidence.plot_labelseq_3d_evidence, but with
    the sample-class subplots reorganized by DISEASE PHENOTYPE instead of
    (or in addition to) the standard P/LP/B/LB/gnomAD/Synonymous split --
    this is what actually shows a joint fit separating disease mechanisms
    (fig:mv_ret), not just the pooled P/LP subplot.

    ``replace_role``: if given (RET's case -- MEN2/Hirschsprung variants are
    a sub-grouping WITHIN the pooled P/LP role), that role's single pooled
    subplot is replaced by one subplot per phenotype group (plus a "other"
    subplot for any P/LP rows not covered by a named group, so nothing is
    silently dropped). If None (CARD11's case -- BENTA/CADINS are their OWN
    sample roles, not a subset of P/LP), the phenotype-group subplots are
    simply APPENDED after the standard 4 role subplots.
    """
    data = _compute_3d_evidence_data(
        gene, x_dim, y_dim, z_dim, results_json, config=config, z_log10=z_log10,
        ms_map=ms_map, run_kwargs=run_kwargs, dataset_name=dataset_name,
    )
    complete = ~(np.isnan(data["x_raw"]) | np.isnan(data["y_raw"]) | np.isnan(data["z_raw"]))

    def _phenotype_entries(base_mask=None):
        entries = []
        named_union = np.zeros(complete.shape, dtype=bool)
        for pheno_name, pheno_mask in phenotype_masks.items():
            m = pheno_mask & complete
            if base_mask is not None:
                m = m & base_mask
            named_union |= pheno_mask
            if m.sum() > 0:
                entries.append((f"{gene}: {pheno_name}", m))
        if base_mask is not None:
            remainder = base_mask & ~named_union & complete
            if remainder.sum() > 0:
                entries.append((f"{gene}: other P/LP", remainder))
        return entries

    new_role_masks = []
    if replace_role is not None:
        for name, mask in data["role_masks"]:
            if name == replace_role:
                new_role_masks.extend(_phenotype_entries(base_mask=mask))
            else:
                new_role_masks.append((name, mask))
    else:
        new_role_masks = list(data["role_masks"]) + _phenotype_entries()
    data["role_masks"] = new_role_masks

    n = len(data["role_masks"])
    fig = plt.figure(figsize=(5 * n, 5.5))
    gs = fig.add_gridspec(1, n)
    _render_3d_evidence_row(fig, [gs[0, i] for i in range(n)], data, n, elev, azim)

    fig.suptitle(f"{gene}: MV evidence by disease phenotype", fontsize=12)
    mappable = plt.cm.ScalarMappable(norm=data["pt_norm"], cmap=POINT_CMAP)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved phenotype evidence plot to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# RET phenotype-specific evidence from the canonical fit (MEN2A/B vs
# Hirschsprung vs benign), rendered like the reference interactive HTML
# (pre_092026_em_fixes/experimental_staged_fit/
# ret_stagedinit_men2_hirschsprung_combined_evidence.html) but computed from
# the paper's canonical RET fit instead of that pre-fix staged-init K=6 fit.
# ---------------------------------------------------------------------------
RET_PHENOTYPE_ROLES = {"MEN2": 4, "Hirschsprung": 5}
RET_PHENOTYPE_COLORS = {"MEN2": "#8B0000", "Hirschsprung": "#800080"}
RET_AXIS_LABELS = {"ret_abundance_HSP90i": "Abundance (HSP90i)",
                   "ret_abundance_No_treatment": "Abundance",
                   "ret_activity_No_treatment": "Activity"}


def _with_phenotype_samples(ms, masks: Dict[str, np.ndarray], roles: Dict[str, int]):
    """Shallow copy of `ms` with one extra sample-assignment column per
    phenotype group, at the fixed role index given in `roles` (4, 5, ...),
    so MVCalibrationAnalysis's auxiliary-pathogenic path can address them.
    The phenotype variants also stay in role 0 (P/LP), exactly as in the
    fit itself."""
    import copy
    new = copy.copy(ms)
    n_roles = max(roles.values()) + 1
    sa = np.zeros((ms.scores.shape[0], max(n_roles, ms._sample_assignments.shape[1])), dtype=bool)
    sa[:, :ms._sample_assignments.shape[1]] = ms._sample_assignments.astype(bool)
    for name, role in roles.items():
        sa[:, role] = masks[name]
    new._sample_assignments = sa
    new.sample_counts = sa.sum(axis=0)
    names = list(getattr(ms, "_sample_names", []))
    names += [""] * (sa.shape[1] - len(names))
    for name, role in roles.items():
        names[role] = name
    new._sample_names = names
    return new


def _posthoc_phenotype_weights(fit_inner, X_by_group, n_iter: int = 500, tol: float = 1e-10):
    """Mixing weights of each phenotype group on the fit's FROZEN components:
    iterate the fitter's own weight update (update_steps.get_sample_weights,
    the same conditional-maximization step the EM uses for every sample)
    to convergence, starting from the fit's own P/LP weights."""
    from src.assay_calibration.fit_utils.cfusn.update_steps import get_sample_weights
    if fit_inner.get("cat_spec"):
        raise NotImplementedError("post hoc phenotype weights don't support categorical-dim fits")
    params = []
    for p in fit_inner["component_params"]:
        Delta = np.array(p[1], dtype=float)
        if Delta.ndim == 2 and Delta.shape[1] == 1:
            Delta = Delta.ravel()
        params.append((np.array(p[0], dtype=float), Delta, np.array(p[2], dtype=float)))
    w0 = np.asarray(fit_inner["weights"], dtype=float)[0]
    rows = []
    for X in X_by_group:
        w = w0[None, :].copy()
        ind = np.ones((X.shape[0], 1), dtype=bool)
        for _ in range(n_iter):
            w_new = get_sample_weights(X, ind, params, w, multivariate=True)
            if np.abs(w_new - w).max() < tol:
                w = w_new
                break
            w = w_new
        rows.append(w[0])
    return np.array(rows)


def ret_phenotype_evidence(results_json: str, dataset_name: str = "ret_labelseq_mv",
                           config_name: Optional[str] = None, run_kwargs=None, ms=None,
                           posthoc: bool = False):
    """MEN2A/B-specific and Hirschsprung-specific evidence for RET variants.

    Fits built by hpc/prepare.py with RET's phenotype groups as their own
    samples (roles 4, 5) carry fitted weight rows for them, which are read
    directly. Older fits without those rows (e.g. the 092026 paper run) get
    each group's weights estimated POST HOC on every bootstrap's frozen
    components (_posthoc_phenotype_weights); ``posthoc=True`` forces that
    path even when fitted rows exist. Either way the groups are scored
    through MVCalibrationAnalysis's auxiliary-pathogenic path (each group vs
    the benign reference, with its own prior).

    Returns dict(ms, coords_names, config, points, aux_points{group},
    aux_lr_median{group}, aux_prior{group}, masks).
    """
    from mv_analysis import mv_cockpit as cockpit
    from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
    run_kwargs = dict(run_kwargs or cockpit.DEFAULT_RUN_KWARGS)
    if ms is None:
        ms = build_labelseq_multiscoresets(genes=["ret"])["ret"]
    masks = ret_phenotype_masks(ms)
    if config_name is None:
        config_name = cockpit.pick_best_canonical_config(
            "ret", "labelseq", ms, results_json, dataset_name, run_kwargs)

    ms_aug = _with_phenotype_samples(ms, masks, RET_PHENOTYPE_ROLES)
    analysis = build_gene_set_analysis(
        ms_aug, "ret", results_json, dataset_name=dataset_name, gene_set="labelseq",
        auxiliary_pathogenic_indices=sorted(RET_PHENOTYPE_ROLES.values()))
    analysis.configs = [config_name]

    groups = sorted(RET_PHENOTYPE_ROLES, key=RET_PHENOTYPE_ROLES.get)
    X_by_group = [ms.scores[masks[g]] for g in groups]
    n_aug = n_fitted = 0
    n_rows = max(RET_PHENOTYPE_ROLES.values()) + 1
    for bd in analysis.raw_boots.values():
        entry = bd.get(config_name) if isinstance(bd, dict) else None
        if not entry:
            continue
        inner = entry.get("fit", entry)
        if inner is None or not inner.get("component_params"):
            continue
        w = np.asarray(inner["weights"], dtype=float)
        if w.shape[0] >= n_rows and not posthoc:
            n_fitted += 1
            continue
        extra = _posthoc_phenotype_weights(inner, X_by_group)
        full = np.zeros((max(n_rows, w.shape[0]), w.shape[1]))
        full[:w.shape[0]] = w
        for g, row in zip(groups, extra):
            full[RET_PHENOTYPE_ROLES[g]] = row
        inner["weights"] = full.tolist()
        n_aug += 1
    print(f"  RET phenotype weights: fitted in {n_fitted}, estimated post hoc in {n_aug} "
          f"bootstrap fits ({config_name})")

    analysis.run(partial_pattern_mode="trust_global", **run_kwargs)
    res = analysis.results[config_name]
    aux = res.get("aux_results", {})
    out = dict(ms=ms, masks=masks, config=config_name, dims=list(ms.dataset_names),
               points=np.asarray(res["points"]), aux_points={}, aux_lr_median={}, aux_prior={})
    for g in groups:
        a = aux[RET_PHENOTYPE_ROLES[g]]
        out["aux_points"][g] = np.asarray(a["points"])
        out["aux_lr_median"][g] = np.asarray(a["lr_median"])
        out["aux_prior"][g] = float(a["median_prior"])
    return out


def plot_ret_phenotype_evidence_3d(ev: dict, html_path: Optional[str] = None,
                                   png_path: Optional[str] = None, camera=None,
                                   label_variants: bool = False, title: Optional[str] = None):
    """Slide-36 layout: each variant is drawn once, colored by the evidence it
    receives -- benign evidence (blue scale) if its P/LP-vs-B/LB points are
    negative, otherwise MEN2-specific (red scale) or Hirschsprung-specific
    (purple scale) pathogenic evidence, whichever phenotype gives it more
    points (ties broken by the larger median log LR+), otherwise grey. The
    curated MEN2A/B and Hirschsprung variants are drawn larger and outlined;
    ``label_variants`` also writes each curated variant's name (e.g. M918T)
    next to it, and every point's hover text gives its points under each
    phenotype."""
    import plotly.graph_objects as go
    ms, masks = ev["ms"], ev["masks"]
    X = ms.scores
    complete = ~np.isnan(X).any(axis=1)
    pts = ev["points"]
    pm, ph = ev["aux_points"]["MEN2"], ev["aux_points"]["Hirschsprung"]
    lm, lh = ev["aux_lr_median"]["MEN2"], ev["aux_lr_median"]["Hirschsprung"]
    men2_wins = (pm > ph) | ((pm == ph) & (lm >= lh))
    benign = pts < 0
    men2 = ~benign & men2_wins & (pm > 0)
    hirsch = ~benign & ~men2_wins & (ph > 0)
    none = ~(benign | men2 | hirsch)
    labels = [RET_AXIS_LABELS.get(d, d) for d in ev["dims"]]

    names = np.array([""] * len(X), dtype=object)
    if label_variants:
        idx = build_hgvs3_index(ms)
        for v in RET_MEN2A_VARIANTS + RET_MEN2B_VARIANTS + RET_HIRSCHSPRUNG_VARIANTS:
            i = idx.get(_short_to_hgvs3(v))
            if i is not None:
                names[i] = v
    hover = np.array([f"{n + ': ' if n else ''}P/LP vs B/LB {int(a)}, MEN2A/B {int(b)}, "
                      f"Hirschsprung {int(c)}" for n, a, b, c in zip(names, pts, pm, ph)], dtype=object)

    def _trace(mask, values, scale, cmax, name, size, outline, colorbar_x=None, cbar_title=None):
        m = mask & complete
        marker = dict(size=size, opacity=0.9, line=dict(width=1.2 if outline else 0.3,
                                                       color="#111111" if outline else "#999999"))
        if values is None:
            marker["color"] = "#c8c8c8"
        else:
            marker.update(color=values[m], colorscale=scale, cmin=0, cmax=cmax,
                          showscale=colorbar_x is not None)
            if colorbar_x is not None:
                marker["colorbar"] = dict(title=dict(text=cbar_title), x=colorbar_x, len=0.6)
        labelled = label_variants and outline
        return go.Scatter3d(x=X[m, 0], y=X[m, 1], z=X[m, 2],
                            mode="markers+text" if labelled else "markers",
                            text=names[m] if labelled else None, textposition="top center",
                            textfont=dict(size=11, color="#111111"),
                            hovertext=hover[m], hoverinfo="text",
                            name=f"{name} (n={int(m.sum())})", marker=marker)

    blues = [[0, "#deebf7"], [1, "#08306b"]]
    reds = [[0, "#fee0d2"], [1, RET_PHENOTYPE_COLORS["MEN2"]]]
    purples = [[0, "#f2e6f7"], [1, RET_PHENOTYPE_COLORS["Hirschsprung"]]]
    curated = masks["MEN2"] | masks["Hirschsprung"]
    traces = [
        _trace(none & ~curated, None, None, 8, "No evidence", 4, False),
        _trace(benign & ~curated, -pts, blues, 8, "Benign evidence", 5, False, 1.00, "Benign<br>points"),
        _trace(men2 & ~curated, pm, reds, 8, "MEN2A/B evidence", 6, False, 1.10, "MEN2A/B<br>points"),
        _trace(hirsch & ~curated, ph, purples, 8, "Hirschsprung evidence", 6, False, 1.20,
               "Hirschsprung<br>points"),
        _trace(masks["MEN2"], pm, reds, 8, "MEN2A/B variants", 10, True),
        _trace(masks["Hirschsprung"], ph, purples, 8, "Hirschsprung variants", 10, True),
    ]
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
        legend=dict(x=0.0, y=1.0), margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        title=dict(text=title) if title else None,
        scene_camera=camera or dict(eye=dict(x=1.6, y=1.6, z=0.9)),
    )
    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))
        print(f"wrote {html_path}")
    if png_path:
        _ret_evidence_3d_static(X, complete, dict(none=none, benign=benign, men2=men2, hirsch=hirsch),
                                dict(benign=-pts, men2=pm, hirsch=ph), masks, labels, png_path)
    return fig, dict(benign=int((benign & complete).sum()), men2=int((men2 & complete).sum()),
                     hirschsprung=int((hirsch & complete).sum()), none=int((none & complete).sum()))


def _ret_evidence_3d_static(X, complete, classes, values, masks, labels, png_path,
                            elev=18, azim=-55):
    """Static (matplotlib) counterpart of plot_ret_phenotype_evidence_3d's
    Plotly figure, for the manuscript -- Plotly's image export needs a Chrome
    install this environment does not have. Same classes and colors."""
    with plt.rc_context({"font.size": 7, "font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                         "pdf.fonttype": 42}):
        _ret_evidence_3d_static_body(X, complete, classes, values, masks, labels, png_path,
                                     elev, azim)


def _ret_evidence_3d_static_body(X, complete, classes, values, masks, labels, png_path,
                                 elev, azim):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmaps = {
        "benign": LinearSegmentedColormap.from_list("ben", ["#deebf7", "#08306b"]),
        "men2": LinearSegmentedColormap.from_list("men2", ["#fee0d2", RET_PHENOTYPE_COLORS["MEN2"]]),
        "hirsch": LinearSegmentedColormap.from_list("hir", ["#f2e6f7", RET_PHENOTYPE_COLORS["Hirschsprung"]]),
    }
    norm = Normalize(0, 8)
    curated = masks["MEN2"] | masks["Hirschsprung"]
    fig = plt.figure(figsize=(6.4, 3.8))
    ax = fig.add_axes([-0.02, 0.0, 0.72, 1.0], projection="3d")
    m = classes["none"] & complete & ~curated
    ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=3, c="#c8c8c8", alpha=0.6, linewidths=0)
    for key in ("benign", "men2", "hirsch"):
        m = classes[key] & complete & ~curated
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=6, c=cmaps[key](norm(values[key][m])),
                   alpha=0.85, linewidths=0.15, edgecolors="#555555")
    for g, key in (("MEN2", "men2"), ("Hirschsprung", "hirsch")):
        m = masks[g] & complete
        v = np.clip(values[key][m], 0, 8)
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=28, c=cmaps[key](norm(v)), alpha=1.0,
                   linewidths=0.8, edgecolors="#111111", depthshade=False)
    ax.set_xlabel(labels[0], labelpad=2)
    ax.set_ylabel(labels[1], labelpad=2)
    ax.set_zlabel(labels[2], labelpad=2)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=elev, azim=azim)
    fig.text(0.855, 0.8, "Evidence points", ha="center", fontsize=7)
    for i, (key, title) in enumerate((("benign", "Benign"), ("men2", "MEN2A/B"),
                                      ("hirsch", "Hirsch-\nsprung"))):
        cax = fig.add_axes([0.745 + 0.09 * i, 0.22, 0.014, 0.48])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmaps[key])
        cb = fig.colorbar(sm, cax=cax)
        cax.set_title(title, fontsize=6.5, pad=4)
        cb.ax.tick_params(labelsize=6, length=2)
        if key == "benign":
            cb.set_ticks([0, 2, 4, 6, 8])
            cb.set_ticklabels(["0", "-2", "-4", "-6", "-8"])
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"wrote {png_path}")
