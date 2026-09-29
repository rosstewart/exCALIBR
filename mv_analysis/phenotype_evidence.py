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
