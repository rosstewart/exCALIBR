#!/usr/bin/env python3
"""Convert the annotated Domainome (Beltran et al. 2024) score sets into the
integrated-dataset format read by hpc/prepare.py default and run_pipeline.py.

The input is a MaveDB/label-seq style table (one row per genomic realization
of each protein-level variant). The conversion mirrors Scoreset.from_mavedb:
column renames come from LABELSEQ_COLUMN_MAP, Dataset is the per-domain
dataset_name (each domain is a separately normalized aPCA experiment, so
domains are calibrated separately), the score becomes auth_reported_score,
splice_measure is "No" (cDNA assay), and mavedb_variant_urn = Gene_hgvs_p
groups genomic realizations into one protein variant.

Junction-split rows: codons crossing an exon-exon junction are reverse
translated into bracketed multi-site genomic HGVS (g.[a;b]). VEP annotates
the whole span including the intron and calls them splice_acceptor/donor,
so they would be dropped by the VEP splice filter. Their consequence is
recoded from hgvs_p (Ter -> stop_gained, else missense_variant). The VEP
call is kept in vep_consequence_original and junction_split marks them:
the changed bases sit at exon edges, and SpliceAI cannot score them, so a
splice effect of the genomic realization is not assessed. None of these
rows carry ClinVar, gnomAD or SpliceAI annotations, so fits are unaffected.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.data_utils.dataset import LABELSEQ_COLUMN_MAP, Scoreset  # noqa: E402

DEFAULT_INPUT = "/data/ross/assay_calibration/domainome/excalibr_domainome.genomic.mapped.tsv"
DEFAULT_OUTPUT = "/data/ross/assay_calibration/domainome/excalibr_domainome.integrated.tsv.gz"


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    levels = df["assayed_variant_level"].unique()
    if list(levels) != ["protein"]:
        raise ValueError(f"expected protein-level score sets only, got {levels.tolist()}")

    rename = {src: dst for src, dst in LABELSEQ_COLUMN_MAP.items()
              if src in df.columns and dst not in df.columns}
    df = df.rename(columns=rename)
    df = df.rename(columns={"score": "auth_reported_score"})
    df["Dataset"] = df["dataset_name"].astype(str)
    df["splice_measure"] = "No"
    df["mavedb_variant_urn"] = df["Gene"].astype(str) + "_" + df["hgvs_p"].astype(str)

    spans = df["spans_intron"].astype(str).str.lower().eq("true")
    bracket = df["mapped_hgvs_g"].astype(str).str.contains(r"\[", regex=True)
    junction = spans & bracket
    if (spans & ~bracket).any():
        raise ValueError(f"{(spans & ~bracket).sum()} intron-spanning rows without bracketed HGVS")
    df["junction_split"] = junction
    df["vep_consequence_original"] = df["simplified_consequence"]
    is_stop = df["hgvs_p"].astype(str).str.endswith("Ter")
    df.loc[junction, "simplified_consequence"] = np.where(
        is_stop[junction], "stop_gained", "missense_variant")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args()

    raw = pd.read_csv(args.input, sep="\t", low_memory=False)
    df = build(raw)

    j = df[df["junction_split"]]
    print(f"Rows: {len(df)}; datasets: {df['Dataset'].nunique()}")
    print(f"Junction-split rows recoded: {len(j)} "
          f"(ClinVar 2026: {j['clinvar_sig_2026'].notna().sum()}, "
          f"gnomAD: {j['gnomad_MAF'].notna().sum()}, "
          f"SpliceAI: {j['spliceAI_DS_AG'].notna().sum()})")
    print(j.groupby("Dataset")["simplified_consequence"].value_counts().to_string())

    print("\nSample counts per dataset (ClinVar 2026, >=1 star, default splice filters):")
    for name, d in df.groupby("Dataset"):
        s = Scoreset(d, clinvar_release="2026", min_clinvar_star=1)
        counts = dict(zip(s.sample_names, s.sample_counts.tolist()))
        print(f"  {name}: {counts}; protein variants scored={len(s.get_variants_by_id())}; VUS={len(s.vus_scores)}")

    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
