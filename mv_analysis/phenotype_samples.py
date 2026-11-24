"""Phenotype-specific pathogenic groups that are fit as their own auxiliary
samples (e.g. RET's MEN2 / Hirschsprung), defined once for the fitting side
(hpc/prepare.py) and the analysis side (mv_analysis.config), so the data a
stored fit is re-scored on -- and any bootstrap split regenerated from it --
has the samples the fit had.

The group variants also stay in P/LP, so each bootstrap's one-hot resolution
puts a variant in one of its samples. Pathogenic evidence uses the P/LP
sample only; the groups are scored only through auxiliary_pathogenic_indices
(mv_analysis.phenotype_evidence). numpy only, so both sides can import it
cheaply.

To add a gene: list its groups in PHENOTYPE_SAMPLE_GROUPS. CARD11's BENTA /
CADINS and TP53's RPV are built into their gene sets' own data builders, so
they are not listed.
"""
import copy
from typing import Dict

import numpy as np

# Variants matched against ms.kept_variants via the protein-HGVS-suffix
# convention analysis/vus_reclassification.py's build_hgvs3_index uses for
# LABEL-seq genes without genomic-coordinate hgvs_c parsing.
_AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
    "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
    "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter",
}

# RET: MEN2A/MEN2B (activating) vs Hirschsprung (loss-of-function) germline
# variants, hand-curated.
RET_MEN2B_VARIANTS = ["M918T", "A883F"]
RET_MEN2A_VARIANTS = ["S891A", "E768D", "L790F", "V804L", "K666E", "V804M", "M918V"]
RET_HIRSCHSPRUNG_VARIANTS = ["L963P", "T946I", "R969W", "A877P", "R972G", "R873W", "S891L"]

# (gene_set, gene key in lower case) -> {group: (role index, variants)}. Role
# indices follow the four standard samples (P/LP 0, B/LB 1, population 2,
# synonymous 3). Variants are short protein changes ("M918T").
PHENOTYPE_SAMPLE_GROUPS = {
    ("labelseq", "ret"): {
        "MEN2": (4, RET_MEN2A_VARIANTS + RET_MEN2B_VARIANTS),
        "Hirschsprung": (5, RET_HIRSCHSPRUNG_VARIANTS),
    },
}


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


def phenotype_sample_groups(gene_set, gene):
    """{group: (role, variants)} for this gene, or None when it has none."""
    return PHENOTYPE_SAMPLE_GROUPS.get((str(gene_set), str(gene).lower()))


def phenotype_roles(gene_set, gene) -> Dict[str, int]:
    """{group: role index} for this gene ({} when it has none)."""
    return {g: r for g, (r, _) in (phenotype_sample_groups(gene_set, gene) or {}).items()}


def phenotype_masks(ms, gene_set, gene) -> Dict[str, np.ndarray]:
    """{group: row mask} over ms's rows for this gene's phenotype groups."""
    hgvs3_index = build_hgvs3_index(ms)
    n = ms.scores.shape[0]

    def _mask(variants):
        mask = np.zeros(n, dtype=bool)
        for v in variants:
            idx = hgvs3_index.get(_short_to_hgvs3(v))
            if idx is not None:
                mask[idx] = True
        return mask

    return {g: _mask(v) for g, (_, v) in (phenotype_sample_groups(gene_set, gene) or {}).items()}


def _with_phenotype_samples(ms, masks: Dict[str, np.ndarray], roles: Dict[str, int]):
    """Shallow copy of `ms` with one extra sample-assignment column per
    phenotype group, at the fixed role index given in `roles` (4, 5, ...),
    so MVCalibrationAnalysis's auxiliary-pathogenic path can address them.
    The phenotype variants also stay in role 0 (P/LP), exactly as in the
    fit itself."""
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


def with_phenotype_samples(ms, gene_set, gene):
    """The gene's MultiScoreset as its fits see it: each phenotype group in
    PHENOTYPE_SAMPLE_GROUPS as its own sample, the variants also kept in
    P/LP. `ms` itself when the gene has no groups."""
    roles = phenotype_roles(gene_set, gene)
    if not roles:
        return ms
    return _with_phenotype_samples(ms, phenotype_masks(ms, gene_set, gene), roles)
