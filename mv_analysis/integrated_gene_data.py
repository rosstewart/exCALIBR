#!/usr/bin/env python3
"""
Integrated/pillar-project gene-set ingestion: the raw integrated variant-effect
dataframe -> {gene: MultiScoreset}, mirroring `src/assay_calibration/
multivariate_data/labelseq.py`'s pattern for LABEL-seq -- built as a standalone,
NOT-wired-into-hpc/prepare.py module so `regularization_type="all_assayed"` can be
exercised experimentally without touching the production job-generation path.

Reuses hpc/prepare.py's exact gene-grouping/clinvar-release logic
(`_discover_gene_groups`, `resolve_clinvar_release`) and
`build_multiscoreset_from_long_dataframe` (same function LABEL-seq uses), so a
non-regularized build here structurally matches what canonical production runs
were built from -- only the raw dataframe PATH differs from the (stale)
hpc/prepare.py default, which was confirmed this session to no longer exist on
disk; every current caller instead points at `..._pp_final.tsv.gz`.
"""
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.multivariate_data.common import (
    build_multiscoreset_from_long_dataframe, resolve_clinvar_release,
)
from src.assay_calibration.data_utils.dataset import MultiScoreset
from hpc.prepare import _discover_gene_groups

DEFAULT_DATA_PATH = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"

# Same durability lesson as this session's LABEL-seq work: caches must live
# outside /tmp scratchpads, which are subject to periodic cleanup (confirmed
# this session -- an entire multi-hour fit batch was silently wiped).
CACHE_ROOT = Path("/data/ross/assay_calibration/multivariate/experimental_staged_fit/.integrated_ms_cache")


def _cache_dir(regularization_type: Optional[str], population_type: Optional[str]) -> Path:
    return CACHE_ROOT / f"reg{regularization_type}_pop{population_type}"


def build_integrated_multiscoresets(
    df: Optional[pd.DataFrame] = None,
    data_path: str = DEFAULT_DATA_PATH,
    genes: Optional[List[str]] = None,
    regularization_type: Optional[str] = None,
    population_type: Optional[str] = None,
    max_dimensions: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, MultiScoreset]:
    """Build one MultiScoreset per multi-assay gene in the integrated dataframe,
    matching hpc/prepare.py's `_process_multivariate_gene`/`_discover_gene_groups`
    exactly (same gene grouping, same per-gene clinvar_release via
    `resolve_clinvar_release`, same `min_clinvar_star=1`, same default
    `min_samples=2, min_dims=2`) -- only `regularization_type` is new/optional,
    threaded through the same generic `scoreset_kwargs` passthrough
    `build_multiscoreset_from_long_dataframe` already supports.

    Per-gene pickle caching (one small file per gene, not one giant combined
    pickle -- see this session's LABEL-seq fix for why) under a durable
    `/data/ross/...` directory, keyed on (regularization_type, population_type).
    """
    cache_dir = _cache_dir(regularization_type, population_type)

    def _cache_path(gene: str) -> Path:
        return cache_dir / f"{gene}.pkl"

    def _cache_fresh(gene: str) -> bool:
        p = _cache_path(gene)
        try:
            return p.exists() and p.stat().st_mtime >= Path(data_path).stat().st_mtime
        except OSError:
            return False

    explicit_df = df is not None
    if use_cache and not explicit_df and genes is not None:
        wanted = [g.upper() for g in genes]
        if all(_cache_fresh(g) for g in wanted):
            gene_ms = {}
            for g in wanted:
                with open(_cache_path(g), "rb") as f:
                    gene_ms[g] = pickle.load(f)
            return gene_ms

    if df is None:
        print(f"Loading {data_path}...", flush=True)
        df = pd.read_csv(data_path, sep="\t", low_memory=False)

    gene_groups = _discover_gene_groups(df, max_dimensions=max_dimensions)
    if genes is not None:
        wanted = {g.upper() for g in genes}
        gene_groups = {g: ds for g, ds in gene_groups.items() if g.upper() in wanted}

    gene_ms = {}
    for gene, datasets in gene_groups.items():
        df_gene = df[df["Dataset"].isin(datasets)]
        scoreset_kwargs = dict(clinvar_release=resolve_clinvar_release(gene), min_clinvar_star=1)
        if population_type:
            scoreset_kwargs["population_type"] = population_type
        if regularization_type:
            scoreset_kwargs["regularization_type"] = regularization_type
        ms = build_multiscoreset_from_long_dataframe(
            df_gene, gene, datasets, scoreset_kwargs=scoreset_kwargs,
        )
        if ms is not None:
            gene_ms[gene] = ms

    if use_cache and not explicit_df:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for gene, ms in gene_ms.items():
                with open(_cache_path(gene), "wb") as f:
                    pickle.dump(ms, f)
        except OSError as e:
            print(f"  [integrated cache] couldn't write to {cache_dir}: {e}")
    return gene_ms
