"""
LABEL-seq ingestion: raw annotated flat file -> {gene: MultiScoreset}.

Faithful port of process_labelseq.ipynb's ingestion cell.
"""

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..data_utils.dataset import MultiScoreset
from .common import build_multiscoreset_from_long_dataframe

DEFAULT_DATA_PATH = "/data/ross/assay_calibration/labelseq/labelseq-annotated-20260529.flat.tsv.gz"

VEP_CONSEQUENCE_MAPPING = {
    "splice_acceptor_variant": "splice_site_variant",
    "splice_region_variant": "splicing_variant",
    "splice_donor_variant": "splice_site_variant",
    "splice_donor_region_variant": "splicing_variant",
    "splice_donor_5th_base_variant": "splicing_variant",
    "splice_polypyrimidine_tract_variant": "splicing_variant",
}


def _parse_first_hgvs_c(x):
    if pd.isna(x):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])
    first = str(x).split("|")[0]
    m = re.search(r"([ACGT])>([ACGT])$", first)
    if m:
        return pd.Series([pd.NA, pd.NA, m.group(1), m.group(2)])
    return pd.Series([pd.NA, pd.NA, np.nan, np.nan])


def build_labelseq_dataframe(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Build the processed LABEL-seq variant dataframe, one row per
    (variant, assay, assay_treatment), with a per-row ``Dataset`` column
    identifying each individual assay.
    """
    df = pd.read_csv(data_path, sep="\t")

    df = df[(df["Mutation"] != "standard") & (df["variant"] != "WT")].copy()

    df["auth_reported_score"] = df["average score"]
    df["clinvar_sig_2026"] = df["clinvar.202601.clinical_significance"]
    df["clinvar_star_2026"] = df["clinvar.202601.review_status"]
    df["Gene"] = df["protein"]
    df["splice_measure"] = "No"
    df["hgvs_p"] = df["mapped_hgvs_p"]

    df["spliceAI_DS_AG"] = df["spliceai.ds_ag"]
    df["spliceAI_DS_AL"] = df["spliceai.ds_al"]
    df["spliceAI_DS_DG"] = df["spliceai.ds_dg"]
    df["spliceAI_DS_DL"] = df["spliceai.ds_dl"]

    df["simplified_consequence"] = (
        df["vep.most_severe_mutational_consequence"]
        .map(lambda x: VEP_CONSEQUENCE_MAPPING.get(x, x))
    )

    df["hgvs_c"] = df["mapped_hgvs_c"]
    df[["Chrom", "hg38_start", "ref_allele", "alt_allele"]] = (
        df["hgvs_c"].apply(_parse_first_hgvs_c)
    )

    parsed = df["variant"].str.extract(
        r"^(?P<aa_ref>[A-Z\*])(?P<aa_pos>\d+)(?P<aa_alt>[A-Z\*\?]+)$"
    )
    df["aa_ref"] = parsed["aa_ref"]
    df["aa_pos"] = pd.to_numeric(parsed["aa_pos"], errors="coerce")
    df["aa_alt"] = parsed["aa_alt"]

    df["gnomad_MAF"] = df["gnomad.v4_1.minor_allele_frequency"]

    df["Dataset"] = (
        df["Gene"].astype(str) + "_" + df["assay"].astype(str) + "_"
        + df["assay_treatment"].astype(str)
    )
    df["mavedb_variant_urn"] = (
        df["Gene"].astype(str) + "_" + df["variant"].astype(str) + "_"
        + df["assay_treatment"].astype(str)
    )
    df["StandardizedClass"] = df["classification_2.5pct"].map(
        {"wt-like": "normal", "low": "abnormal", "high": "abnormal"}
    )
    return df


def build_labelseq_multiscoresets(
    df_labelseq: Optional[pd.DataFrame] = None,
    data_path: str = DEFAULT_DATA_PATH,
) -> Dict[str, MultiScoreset]:
    """Build one MultiScoreset per gene, combining all of that gene's assays
    (each ``Dataset`` value, e.g. ``"braf_abundance_HSP90i"``).
    """
    if df_labelseq is None:
        df_labelseq = build_labelseq_dataframe(data_path)

    gene_to_datasets: Dict[str, list] = {}
    for assay in df_labelseq.Dataset.unique():
        gene = assay.split("_")[0]
        gene_to_datasets.setdefault(gene, []).append(assay)

    gene_ms = {}
    for gene, datasets in gene_to_datasets.items():
        # min_samples=1, min_dims=1: the notebook applied neither filter --
        # every assay (however few sample classes) and every gene (even a
        # single assay) was kept, unlike the integrated-dataframe pipeline's
        # default (min_samples=2, min_dims=2).
        ms = build_multiscoreset_from_long_dataframe(
            df_labelseq, gene, datasets, min_samples=1, min_dims=1,
        )
        if ms is not None:
            gene_ms[gene] = ms
    return gene_ms
