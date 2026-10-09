#!/usr/bin/env python3
"""
Build analysis/manuscript_curated/dataset_metadata.csv (config.
CURATED_DATASET_METADATA_CSV, also config.ASSAY_METHOD_MAP_CSV) from the
assay curation spreadsheet (config.ASSAY_CURATION_CSV; column meanings in
config.ASSAY_CURATION_COLUMN_DEFINITIONS_CSV).

One row per calibrated dataset, keyed by its pipeline name (with the
_clinvar_2018 suffix where it has one). Columns used by Fig. 6, the
VAMP-seq/SGE numbers and Supplementary Data 1:

  dataset, gene, citation, PMID, description, assay_type, model_system,
  model_system_ncbi_taxon, vamp_sge, meta_analysis, IGVF_produced, disease,
  new_functional_classifications_created

Values are lower-cased to the labels the Fig. 6 code expects (e.g. "Cell
fitness" -> "cell fitness", "Vamp-seq" -> "VAMP-seq", "Other Rare Disease"
-> "rare disease", "Cardio" -> "cardio").

    python -m analysis.build_curated_metadata
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from analysis import config as cfg

COLUMNS = ["dataset", "gene", "citation", "PMID", "description", "assay_type", "model_system",
           "model_system_ncbi_taxon", "vamp_sge", "meta_analysis", "IGVF_produced", "disease",
           "new_functional_classifications_created"]

_VAMP_SGE = {"Vamp-seq": "VAMP-seq", "SGE": "SGE", "Other": "other"}
_DISEASE = {"Cancer": "cancer", "Other Rare Disease": "rare disease", "Cardio": "cardio", "Metabolic": "metabolic"}


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def _citation(first_author, year) -> str:
    """'<first author> et al. (<year>)'; 'Unpublished' without a year.
    Curation spells multi-word surnames with underscores (van_Loggerenberg,
    O_Neill) and lists the consortium first for IGVF papers
    ("IGVF, Woo,Casadei, Snyder"), where the first named author is used."""
    if _blank(year):
        return "Unpublished"
    names = [n.strip() for n in str(first_author).split(",") if n.strip()]
    if len(names) > 1 and names[0] == "IGVF":
        names = names[1:]
    name = names[0] if names else "IGVF"
    name = re.sub(r"^O_", "O'", name).replace("_", " ")
    return f"{name} et al. ({int(float(year))})"


def build(curation_csv=None, datasets=None) -> pd.DataFrame:
    cur = pd.read_csv(curation_csv or cfg.ASSAY_CURATION_CSV, dtype=str)
    cur.columns = [c.strip() for c in cur.columns]
    by_name = {r["Dataset Name"].strip(): r for _, r in cur.iterrows()}
    if datasets is None:
        from analysis.discovery import discover_outputs, DEFAULT_EXCLUDED_DATASETS
        tree, _, _ = discover_outputs(Path(cfg.OUTPUT_DIR))
        datasets = sorted(d for d in tree if d not in DEFAULT_EXCLUDED_DATASETS)
    rows, missing = [], []
    for ds in datasets:
        r = by_name.get(ds) if ds in by_name else by_name.get(ds.replace("_clinvar_2018", ""))
        if r is None:
            missing.append(ds)
            continue
        get = lambda col: None if _blank(r.get(col)) else str(r.get(col)).strip()
        pmid = get("PMID")
        meta = get("Primary Score Set or Meta-analysis?")
        newfc = get("New Functional Classifications Created?")
        rows.append({
            "dataset": ds,
            "gene": get("Gene"),
            "citation": _citation(get("First Author"), get("Year of Publication")),
            "PMID": str(int(float(pmid))) if pmid else "UNPUBLISHED",
            "description": get("Experiment Description"),
            "assay_type": (get("Assay Type") or "").lower() or None,
            "model_system": (get("Model System") or "").lower() or None,
            "model_system_ncbi_taxon": get("Model System NCBI Taxon"),
            "vamp_sge": _VAMP_SGE.get(get("Assay Name"), get("Assay Name")),
            "meta_analysis": None if meta is None else meta.lower() == "meta-analysis",
            "IGVF_produced": None if get("IGVF Produced?") is None else get("IGVF Produced?") == "Yes",
            "disease": _DISEASE.get(get("Broad Disease Category"), get("Broad Disease Category")),
            "new_functional_classifications_created": newfc,
        })
    if missing:
        raise SystemExit(f"calibrated datasets missing from {curation_csv or cfg.ASSAY_CURATION_CSV}: {missing}")
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    out = Path(cfg.CURATED_DATASET_METADATA_CSV)
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    for col in ("assay_type", "model_system", "vamp_sge", "meta_analysis", "disease", "IGVF_produced"):
        print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")
