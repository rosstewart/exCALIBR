"""
Shared ingestion shapes reused across gene-sets, with each shape's
assumptions stated explicitly rather than re-implemented ad hoc per module.

Two recurring shapes cover every gene-set in this package:

1. **Long-format, one ``Scoreset`` per ``Dataset`` value, grouped by gene**
   (``build_multiscoreset_from_long_dataframe``). Assumption: every
   dimension for a gene lives in the *same* dataframe/schema, distinguished
   only by a ``Dataset`` column value -- i.e. a bare ``Scoreset(df_slice)``
   call is enough to build each dimension. Used by the existing integrated-
   dataframe multivariate pipeline (BRCA1/PTEN/MSH2/pillar-project genes,
   in hpc/prepare.py) and by LABEL-seq. FGFR does NOT fit this shape (its
   3 score dimensions are 3 *columns* on the same rows, not 3 disjoint
   ``Dataset`` values) and builds its MultiScoreset directly instead.

2. **Wide-format, id-keyed, one shared ``sample_assignments`` column**
   (``build_basicmultiscoreset_from_wide_dataframe``). Assumption: every
   dimension for a gene-set is its own *column* in one dataframe, all
   dimensions share exactly one class-membership assignment per variant
   (encoded once, not per-dimension), and variants are aligned by an
   explicit id column. Used by TP53 (16 dims) and CARD11 (2 dims).
"""

from typing import Dict, List, Optional

import pandas as pd

from ..data_utils.dataset import Scoreset, MultiScoreset, BasicMultiScoreset

# Genes whose published/historical calibration used ClinVar 2018 labels
# specifically (not the repo-wide 2026 default) -- shared by every pipeline
# that builds a Scoreset directly from the integrated variant-effect
# dataframe (hpc/prepare.py's integrated multivariate path, and the
# functional side of the "combined" predictor+functional model), so the
# two paths can't silently disagree on which ClinVar release a gene uses.
GENES_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}


def resolve_clinvar_release(gene: str) -> str:
    return "2018" if gene in GENES_2018 else "2026"


def gene_set_dataset_label(gene: str, gene_set: str) -> str:
    """Canonical results-JSON key / save_dir name for one gene under a given
    ``--gene-set`` (hpc/prepare.py's run_multivariate_gene_set). Shared
    between job generation and the analyze drivers so they can't drift
    apart -- MVCalibrationAnalysis looks results up by this exact string.
    """
    return f"{gene}_{gene_set}_mv"


def build_multiscoreset_from_long_dataframe(
    df: pd.DataFrame,
    gene: str,
    datasets: List[str],
    dataset_col: str = "Dataset",
    gene_col: str = "Gene",
    scoreset_kwargs: Optional[Dict] = None,
    min_samples: int = 2,
    min_dims: int = 2,
) -> Optional[MultiScoreset]:
    """One ``Scoreset`` per ``dataset_col`` value for ``gene``, combined into
    a ``MultiScoreset``.

    ``min_samples`` drops a dimension with fewer than that many represented
    sample classes (the existing integrated-dataframe pipeline's default:
    2); ``min_dims`` drops the whole gene if fewer than that many dimensions
    survive. LABEL-seq's notebook applied neither filter -- pass
    ``min_samples=1, min_dims=1`` there to keep every assay/gene the
    notebook did.
    """
    scoreset_kwargs = scoreset_kwargs or {}
    df_gene = df[df[gene_col] == gene] if gene_col in df.columns else df

    scoresets, valid_names = [], []
    for ds_name in datasets:
        ds = Scoreset(df_gene[df_gene[dataset_col] == ds_name], **scoreset_kwargs)
        if sum(1 for _ in ds.samples) < min_samples:
            continue
        scoresets.append(ds)
        valid_names.append(ds_name)

    if len(scoresets) < min_dims:
        return None
    return MultiScoreset(scoresets, dataset_names=valid_names)


def build_basicmultiscoreset_from_wide_dataframe(
    df: pd.DataFrame,
    score_cols: List[str],
    id_col: str,
    sample_assignments_col: str = "sample_assignments",
    dataset_names: Optional[List[str]] = None,
    sample_names: Optional[List[str]] = None,
) -> BasicMultiScoreset:
    """Thin, explicitly-named wrapper around
    ``BasicMultiScoreset.from_dataframe`` for gene-sets where every score
    dimension is a column of one dataframe sharing a single
    ``sample_assignments`` column (TP53, CARD11).
    """
    return BasicMultiScoreset.from_dataframe(
        df,
        score_cols=score_cols,
        id_col=id_col,
        sample_assignments_col=sample_assignments_col,
        dataset_names=dataset_names,
        sample_names=sample_names,
    )
