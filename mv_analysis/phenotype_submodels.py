"""Phenotype sub-models: one small joint calibration per phenotype.

Each sub-model fits one phenotype group against its matched benign reference
and the population, with its own prior -- the joint analogue of a
phenotype-matched univariate calibration. Samples: 0 the phenotype group,
1 matched benign, 2 population (gnomAD), 3 synonymous; all other P/LP are
left out.

Readouts are fixed by biology, never chosen with labels: a gain-of-function
phenotype uses its gain readout alone (BENTA: GOF score; MEN2: activity), a
loss-of-function phenotype uses every readout (CADINS, Hirschsprung). A
one-readout sub-model is exactly a univariate phenotype calibration, which the
multivariate fitter does not fit (it needs >= 2 co-measured readouts), so
BENTA and MEN2 come from the univariate phenotype calibrations (CARD11
gof_benta; RET MEN2 on activity) and only the multi-readout sub-models are
listed in SUBMODELS. UV_SUBMODELS records the univariate ones.
Matched benign: CARD11's cohort GOF-/LOF-benign controls (the controls its
univariate phenotype calibrations used); RET has no cohort benign set, so
B/LB + synonymous.

Shared by hpc/prepare.py (``--gene-set phenotype_sub`` / mv_all pipeline
``phenotype_sub``) and the analysis, so both build identical MultiScoresets.
Dataset keys are ``<gene>_<phenotype>_phenotype_sub_mv``.
"""
import copy
from typing import Dict

import numpy as np

# (gene, phenotype, readout substrings kept; None = every readout)
SUBMODELS = [
    ("CARD11", "CADINS", None),
    ("ret", "Hirschsprung", None),
]
UV_SUBMODELS = [("CARD11", "BENTA", ["GOF"]), ("ret", "MEN2", ["activity"])]
SAMPLE_NAMES = ["phenotype", "matched benign", "population", "synonymous"]


def _card11_matched_benign(ms, phenotype):
    import pandas as pd
    from src.assay_calibration.multivariate_data.card11 import DEFAULT_MASTER_CSV
    df = pd.read_csv(DEFAULT_MASTER_CSV, low_memory=False)
    col = "GOF_ClinSig" if phenotype == "BENTA" else "LOF_ClinSig"
    ok = set(df.loc[df[col].eq("benign"), "hgvs_p"])
    ids = [v[0] if isinstance(v, tuple) else str(v) for v in ms.kept_variants]
    return np.array([v in ok for v in ids])


def _base(gene):
    """(ms, {phenotype: mask})"""
    if gene == "CARD11":
        from src.assay_calibration.multivariate_data.card11 import build_card11_multiscoreset
        ms = build_card11_multiscoreset()
        sa = np.asarray(ms.sample_assignments).astype(bool)
        return ms, {"BENTA": sa[:, 4].copy(), "CADINS": sa[:, 5].copy()}
    from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
    from mv_analysis.phenotype_evidence import ret_phenotype_masks
    ms = build_labelseq_multiscoresets(genes=["ret"])["ret"]
    return ms, ret_phenotype_masks(ms)


def submodel_ms(ms, gene, phenotype, mask, readouts):
    """Copy of ``ms`` restricted to ``readouts`` with samples
    [phenotype, matched benign, population, synonymous]."""
    from src.assay_calibration.multivariate_data.redundancy_collapse import select_dims
    m = copy.deepcopy(ms)
    if readouts is not None:
        dims = [i for i, n in enumerate(ms.dataset_names) if any(r.lower() in n.lower() for r in readouts)]
        if not dims:
            raise ValueError(f"{gene} {phenotype}: no readout matches {readouts} in {list(ms.dataset_names)}")
        select_dims(m, dims)
    old = np.asarray(ms.sample_assignments).astype(bool)
    sa = np.zeros((old.shape[0], 4), dtype=bool)
    sa[:, 0] = mask
    benign = _card11_matched_benign(ms, phenotype) if gene == "CARD11" else old[:, 1]
    sa[:, 1] = benign & ~mask
    sa[:, 2:4] = old[:, 2:4]
    m._sample_assignments = sa
    m.sample_counts = sa.sum(axis=0)
    m._sample_names = list(SAMPLE_NAMES)
    return m


def build_phenotype_submodel_ms_map() -> Dict[str, object]:
    """{"<gene>_<phenotype>": MultiScoreset} for every entry in SUBMODELS."""
    out, bases = {}, {}
    for gene, phenotype, readouts in SUBMODELS:
        if gene not in bases:
            bases[gene] = _base(gene)
        ms, masks = bases[gene]
        out[f"{gene}_{phenotype}"] = submodel_ms(ms, gene, phenotype, masks[phenotype], readouts)
    return out
