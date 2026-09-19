"""
LABEL-seq ingestion: raw annotated flat file -> {gene: MultiScoreset}.

Faithful port of process_labelseq.ipynb's ingestion cell.
"""

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data_utils.dataset import MultiScoreset
from .common import build_multiscoreset_from_long_dataframe

DEFAULT_DATA_PATH = "/data/ross/assay_calibration/labelseq/labelseq-annotated-20260529.flat.tsv.gz"


def _labelseq_ms_cache_dir(data_path: str, log10_activity: bool,
                            regularization_type: Optional[str] = None) -> Path:
    return Path(data_path).parent / (
        f".labelseq_ms_cache_log10act{log10_activity}_reg{regularization_type}"
    )


def _labelseq_gene_cache_path(cache_dir: Path, gene: str) -> Path:
    return cache_dir / f"{gene}.pkl"

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


def build_labelseq_dataframe(
    data_path: str = DEFAULT_DATA_PATH, log10_activity: bool = True,
) -> pd.DataFrame:
    """Build the processed LABEL-seq variant dataframe, one row per
    (variant, assay, assay_treatment), with a per-row ``Dataset`` column
    identifying each individual assay.

    ``log10_activity``: the "activity" assay's raw scores are strictly
    positive but heavily right-skewed (confirmed across all genes: range
    0.03-45.5), unlike "abundance"/"interaction". When True (default),
    log10-transforms `auth_reported_score` in place for activity rows only
    -- before this, MV bootstrap fits used the raw linear scale.
    """
    df = pd.read_csv(data_path, sep="\t")

    df = df[(df["Mutation"] != "standard") & (df["variant"] != "WT")].copy()

    df["auth_reported_score"] = df["average score"]
    if log10_activity:
        activity_mask = df["assay"] == "activity"
        df.loc[activity_mask, "auth_reported_score"] = np.log10(
            df.loc[activity_mask, "auth_reported_score"])
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
    log10_activity: bool = True,
    genes: Optional[List[str]] = None,
    use_cache: bool = True,
    regularization_type: Optional[str] = None,
) -> Dict[str, MultiScoreset]:
    """Build one MultiScoreset per gene, combining all of that gene's assays
    (each ``Dataset`` value, e.g. ``"braf_abundance_HSP90i"``).

    ``log10_activity``: forwarded to `build_labelseq_dataframe` -- ignored if
    `df_labelseq` is passed in already-built (the caller controls that df's
    own transform in that case).

    ``regularization_type``: forwarded to each gene's `Scoreset(...)`
    construction (see `Scoreset.parse_regularization_type`) -- e.g.
    ``"all_assayed"`` adds a 5th `_sample_assignments` column matching every
    assayed variant regardless of clinical label (a strict superset of the 4
    canonical P/LP/B/LB/gnomAD/Synonymous columns), surfacing the large pool
    of otherwise-dropped unclassified/VUS-like variants. ``None`` (default)
    preserves today's exact 4-class behavior.

    ``genes``: if given (case-insensitive), only builds/returns these genes'
    MultiScoresets -- skips the per-gene Scoreset-construction cost (the
    dominant cost of a from-scratch build, confirmed empirically at ~10
    minutes for all 17 genes together) for every other gene. Ignored (i.e.
    just subsets) when a valid cache hit is used, since that's equally fast
    and keeps the caching logic simple.

    ``use_cache``: cache each gene's MultiScoreset to its OWN pickle file in a
    per-(log10_activity, regularization_type) cache directory next to
    `data_path` (one file per gene, not one combined multi-gene pickle) --
    invalidated whenever the source file's mtime is newer than a given gene's
    cache file. This means a `genes=[...]` call only ever reads the small
    (tens of MB) pickles for the genes actually requested, instead of
    unpickling every gene's data just to discard most of it -- important
    when multiple gene-specific processes run concurrently (loading the old
    combined ~6GB pickle N times over was observed this session to exhaust
    system RAM+swap and trigger the OOM killer). Only used when `df_labelseq`
    isn't passed in explicitly (an explicit df means the caller controls
    ingestion, e.g. a one-off filtered/modified copy) -- pass
    ``use_cache=False`` to force a fresh rebuild (e.g. after a code change
    to the ingestion logic itself, which the mtime check can't detect).
    """
    explicit_df = df_labelseq is not None
    cache_dir = _labelseq_ms_cache_dir(data_path, log10_activity, regularization_type)

    def _cache_fresh(gene: str) -> bool:
        p = _labelseq_gene_cache_path(cache_dir, gene)
        try:
            return p.exists() and p.stat().st_mtime >= Path(data_path).stat().st_mtime
        except OSError:
            return False

    if use_cache and not explicit_df and genes is not None:
        wanted_lower = [g.lower() for g in genes]
        if all(_cache_fresh(g) for g in wanted_lower):
            gene_ms = {}
            for g in wanted_lower:
                with open(_labelseq_gene_cache_path(cache_dir, g), "rb") as f:
                    gene_ms[g] = pickle.load(f)
            return gene_ms

    if df_labelseq is None:
        df_labelseq = build_labelseq_dataframe(data_path, log10_activity=log10_activity)

    gene_to_datasets: Dict[str, list] = {}
    for assay in df_labelseq.Dataset.unique():
        gene = assay.split("_")[0]
        gene_to_datasets.setdefault(gene, []).append(assay)

    build_all = genes is None
    wanted = None if build_all else {g.upper() for g in genes}

    gene_ms = {}
    for gene, datasets in gene_to_datasets.items():
        if not build_all and gene.upper() not in wanted:
            continue
        # min_samples=1, min_dims=1: the notebook applied neither filter --
        # every assay (however few sample classes) and every gene (even a
        # single assay) was kept, unlike the integrated-dataframe pipeline's
        # default (min_samples=2, min_dims=2).
        ms = build_multiscoreset_from_long_dataframe(
            df_labelseq, gene, datasets, min_samples=1, min_dims=1,
            scoreset_kwargs={"regularization_type": regularization_type}
            if regularization_type else None,
        )
        if ms is not None:
            gene_ms[gene] = ms

    if use_cache and not explicit_df:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for gene, ms in gene_ms.items():
                with open(_labelseq_gene_cache_path(cache_dir, gene), "wb") as f:
                    pickle.dump(ms, f)
        except OSError as e:
            print(f"  [labelseq cache] couldn't write to {cache_dir}: {e}")
    return gene_ms
