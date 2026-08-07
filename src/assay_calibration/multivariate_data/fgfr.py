"""
FGFR ingestion: raw annotated variants -> {gene: MultiScoreset}.

Faithful port of process_fgfr.ipynb / analyze_fgfr.ipynb's ingestion cells.
The two notebooks diverged on regularization_type ("all_assayed" in
process_fgfr.ipynb vs. None in analyze_fgfr.ipynb, the value actually used to
produce the current on-disk fits/analysis) -- resolved per the consolidation
plan: regularization_type is always None here, never "all_assayed".
"""

from typing import Dict, List, Optional

import pandas as pd

from ..data_utils.dataset import Scoreset, MultiScoreset

DEFAULT_DATA_PATH = "/data/ross/assay_calibration/FGFR/fgfr_annotated.tsv.gz"
DATASET_COL = "gene_symbol"
COMBINED_DATASET_NAME = "FGFR_combined"
SCORE_COLS = ["Median activation", "Median PemR", "Median FutR"]
FUNC_CLASS_COLS = ["Category activation", "Category LOF", "Category PemR", "Category FutR"]
FUNC_CLASS_COL_PER_DIM = ["Category activation", "Category PemR", "Category FutR"]


def load_fgfr_dataframe(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path, sep="\t")


def _build_one_multiscoreset(gene_df: pd.DataFrame, population_type: str) -> MultiScoreset:
    scoresets = [
        Scoreset.from_mavedb(
            gene_df,
            dataset_col=DATASET_COL,
            score_col=score_col,
            func_class_col=fc_col,
            func_class_cols=FUNC_CLASS_COLS,
            splice_measure=False,
            population_type=population_type,
            regularization_type=None,
        )
        for score_col, fc_col in zip(SCORE_COLS, FUNC_CLASS_COL_PER_DIM)
    ]
    return MultiScoreset(scoresets, dataset_names=SCORE_COLS)


def build_fgfr_multiscoresets(
    df_fgfr: Optional[pd.DataFrame] = None,
    data_path: str = DEFAULT_DATA_PATH,
    genes: Optional[List[str]] = None,
    exclude_genes: Optional[List[str]] = None,
    population_type: str = "gnomAD",
    combine_genes: bool = True,
) -> Dict[str, MultiScoreset]:
    """Build FGFR MultiScoreset(s) (3 dims: activation/PemR/FutR).

    Gene list defaults to every gene_symbol present in the annotated file
    (not the hardcoded 4-gene list analyze_fgfr.ipynb happened to use).

    combine_genes (default True): pool every target gene's P/LP/B/LB/gnomAD/
    Synonymous variants into ONE combined MultiScoreset (key
    "FGFR_combined"), rather than one MultiScoreset per gene. Safe to pool:
    variant identity for this dataset resolves via genomic coordinates
    (Chrom/hg38_start/ref/alt, 100% populated here), and FGFR1-4 sit on
    different chromosomes, so there's no cross-gene ID collision risk.
    Pass combine_genes=False for the original one-MultiScoreset-per-gene
    behavior.

    ``exclude_genes`` is applied here (not by the caller filtering the
    returned dict) because in combine_genes mode the returned dict has one
    "FGFR_combined" key, not gene-named keys -- a post-hoc filter on gene
    name couldn't drop a specific gene out of the pool.
    """
    if df_fgfr is None:
        df_fgfr = load_fgfr_dataframe(data_path)

    target_genes = (
        list(genes) if genes is not None else list(df_fgfr[DATASET_COL].unique())
    )
    if exclude_genes:
        excluded = {g.upper() for g in exclude_genes}
        target_genes = [g for g in target_genes if g.upper() not in excluded]

    if combine_genes:
        df_combined = df_fgfr[df_fgfr[DATASET_COL].isin(target_genes)].copy()
        # from_mavedb requires exactly one Dataset value; pre-setting it
        # (rather than leaving from_mavedb to derive it from dataset_col,
        # which would give one value per gene) is what pools all target
        # genes' rows into a single Scoreset per score dimension.
        df_combined["Dataset"] = COMBINED_DATASET_NAME
        return {COMBINED_DATASET_NAME: _build_one_multiscoreset(df_combined, population_type)}

    gene_ms = {}
    for gene in target_genes:
        gene_df = df_fgfr[df_fgfr[DATASET_COL] == gene]
        gene_ms[gene] = _build_one_multiscoreset(gene_df, population_type)
    return gene_ms
