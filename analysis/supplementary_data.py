"""
Supplementary Data 1-3 for the Nature Genetics manuscript, built from
pipeline output (Supplementary Data 4, All of Us, is a collaborator file).

- SD1 `SD1_dataset_summaries.csv`: one row per calibrated dataset, same
  columns/order as `ExCALIBR_latex_src/old_supp_data_1_dataset_summaries.csv`.
- SD2 `SD2_model_fit_visualizations.pdf`: `extended_data_appendix.build_appendix_pdf`.
- SD3 `SD3_oddspath_evidence.csv`: the OddsPath (Brnich et al.) table, keyed
  to the calibrated datasets, with explicit "Not available" rows for datasets
  the OddsPath approach could not evaluate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.plot_common import display_name

from analysis import config as cfg

SD1_COLUMNS = [
    "dataset", "gene", "citation", "PMID", "description", "assay_type", "model_system",
    "model_system_ncbi_taxon", "vamp_sge", "IGVF_produced", "author_assignments_provided", "disease", "n_plp", "n_blb", "n_gnomad", "n_synonymous",
    "n_vus", "n_snv", "n_plp_fit", "n_blb_fit", "n_gnomad_fit", "n_synonymous_fit",
    "range_-8", "range_-7", "range_-6", "range_-5", "range_-4", "range_-3", "range_-2", "range_-1",
    "range_1", "range_2", "range_3", "range_4", "range_5", "range_6", "range_7", "range_8",
    "prior", "yang_dist_plp", "yang_dist_blb", "yang_dist_gnomad", "yang_dist_synonymous",
    "yang_dist_heldout_plp", "yang_dist_heldout_blb", "yang_dist_heldout_gnomad", "yang_dist_heldout_synonymous",
    "relax", "n_c", "benign_method", "clinvar_2018", "scoreset_flipped", "new_functional_classifications_created",
]
SD3_COLUMNS = [
    "Dataset", "Total Controls", "OddsNormal", "OddsAbnormal", "Pathogenic Controls", "Benign Controls",
    "Prior Probability Pathogenic", "Total Assay Abnormal", "True Path in Abnormal", "Total Assay Normal",
    "True Path in Normal", "Pseudocount Details", "Evidence Code Normal", "Evidence Code Abnormal",
]
CURATED_REQUIRED = ["citation", "PMID", "description", "assay_type", "model_system"]
_SAMPLES = {"n_plp": "Pathogenic/Likely Pathogenic", "n_blb": "Benign/Likely Benign",
            "n_gnomad": "population", "n_synonymous": "Synonymous"}
_YANG_SAMPLES = {"yang_dist_plp": "plp", "yang_dist_blb": "blb", "yang_dist_gnomad": "gnomad",
                 "yang_dist_synonymous": "synonymous"}


def _fmt_bound(x) -> str:
    x = float(x)
    return "inf" if x == np.inf else ("-inf" if x == -np.inf else repr(x))


def _fmt_ranges(ranges) -> str:
    return ";".join(f"{_fmt_bound(lo)} {_fmt_bound(hi)}" for lo, hi in (ranges or []))


def _resolved_calibration(dataset: str, tree, model_selections, dataset_configs) -> Dict:
    from analysis.discovery import resolve_component
    comp = resolve_component(dataset, list(tree[dataset].keys()), model_selections, dataset_configs)
    path = next(iter(sorted((Path(cfg.OUTPUT_DIR) / dataset).glob(f"{dataset}_{comp}_calibration.json"))), None)
    if path is None:
        raise FileNotFoundError(f"{dataset}: no calibration json for component {comp}")
    with open(path) as f:
        return json.load(f)


def canonical_yang_rows(yang_df: pd.DataFrame, datasets, tree, model_selections, dataset_configs) -> pd.DataFrame:
    """One Yang-table row per dataset: the row for its resolved component.
    The --with-yang table has a row per fitted component (n_c, e.g. "3c_avg")
    for datasets with several calibrations; rows for other components are dropped."""
    from analysis.discovery import resolve_component
    keep = []
    for ds in datasets:
        sub = yang_df[yang_df["dataset"] == ds]
        if len(sub) > 1 and "n_c" in sub.columns:
            comp = resolve_component(ds, list(tree[ds].keys()), model_selections, dataset_configs)
            sub = sub[sub["n_c"].astype(str) == str(comp)]
            if len(sub) != 1:
                raise ValueError(f"{ds}: expected one Yang row for component {comp}, found {len(sub)}")
        keep.append(sub)
    return pd.concat(keep, ignore_index=True) if keep else yang_df.iloc[:0]


def _yang_cell(row: Optional[pd.Series], sample: str, n_fit: int, interval=("p25", "p75")) -> str:
    """'median (Q1-Q3)' across bootstraps, as in the old SD1; SAMPLE_EXCLUDED when the sample had
    no fit variants, INCOMPUTABLE when too few (1-2) for a distance."""
    if n_fit == 0:
        return "SAMPLE_EXCLUDED"
    if n_fit < 3:
        return "INCOMPUTABLE"
    if row is None:
        return ""
    med = row.get(f"yang_dist_{sample}")
    lo, hi = row.get(f"yang_dist_{sample}_{interval[0]}"), row.get(f"yang_dist_{sample}_{interval[1]}")
    if med is None or pd.isna(med):
        return ""
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"{med:.3f}"
    return f"{med:.3f} ({lo:.3f}-{hi:.3f})"


def _heldout_yang_cell(row: Optional[pd.Series], sample: str, n_fit: int, min_n: int) -> str:
    """Held-out 'median (Q1-Q3)'; SAMPLE_EXCLUDED as for training distances,
    INCOMPUTABLE when the bootstraps' median held-out sample has < min_n variants."""
    if n_fit == 0:
        return "SAMPLE_EXCLUDED"
    if row is None:
        return ""
    n_val = row.get(f"n_val_{sample}")
    if n_val is None or pd.isna(n_val) or n_val < min_n:
        return "INCOMPUTABLE"
    med, lo, hi = (row.get(f"yang_val_{sample}{suf}") for suf in ("", "_p25", "_p75"))
    if med is None or pd.isna(med):
        return ""
    return f"{med:.3f} ({lo:.3f}-{hi:.3f})"


def fit_sample_counts(dataset: str) -> Dict[str, int]:
    """Sample sizes as the pipeline constructs them for fitting (Scoreset from
    load_dataset_from_df with the run's PipelineConfig)."""
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from analysis.discovery import load_master_df

    df_full = load_master_df(str(cfg.DATASET_TSV))
    df_ds = df_full[df_full["Dataset"] == dataset.replace("_clinvar_2018", "")].copy()
    df_ds["Dataset"] = dataset
    pcfg = PipelineConfig(dataset_csv=str(cfg.DATASET_TSV), dataset_name=dataset, output_dir="/tmp",
                          clinvar_release="2018" if "_clinvar_2018" in dataset else "2025")
    scoreset = load_dataset_from_df(df_ds, pcfg)
    counts = dict(zip([s[1] for s in scoreset.samples], (int(c) for c in scoreset.sample_counts)))
    return {f"{k}_fit": counts.get(v, counts.get("gnomAD", 0) if k == "n_gnomad" else 0) for k, v in _SAMPLES.items()}


def build_sd1(datasets: List[str], tree, model_selections, dataset_configs, df_variants: pd.DataFrame,
              df_all_groups: Optional[pd.DataFrame] = None, yang_df: Optional[pd.DataFrame] = None,
              curated_csv: Optional[str] = None, yang_heldout_df: Optional[pd.DataFrame] = None,
              ) -> (pd.DataFrame, List[str]):
    """SD1 rows for `datasets`. Returns (table, problems)."""
    from analysis.plot_common import sample_matches

    problems = []
    curated = pd.read_csv(curated_csv or cfg.CURATED_DATASET_METADATA_CSV, dtype=str).set_index("dataset")
    if yang_df is not None and not yang_df.empty:
        yang_df = canonical_yang_rows(yang_df, [d for d in datasets if d in set(yang_df["dataset"])],
                                      tree, model_selections, dataset_configs)
    yang_idx = yang_df.set_index("dataset") if yang_df is not None and not yang_df.empty else None
    rows = []
    for ds in datasets:
        row = {"dataset": ds}
        meta = curated.loc[ds] if ds in curated.index else None
        if meta is None:
            problems.append(f"{ds}: not in curated metadata CSV")
        for col in ["gene", "citation", "PMID", "description", "assay_type", "model_system",
                    "model_system_ncbi_taxon", "vamp_sge", "IGVF_produced", "disease",
                    "new_functional_classifications_created"]:
            val = meta[col] if meta is not None and col in meta.index else None
            row[col] = "" if val is None or pd.isna(val) else val
        missing_meta = [c for c in CURATED_REQUIRED if not str(row[c]).strip()]
        if missing_meta:
            problems.append(f"{ds}: curated metadata missing {missing_meta}")

        sub = df_variants[df_variants["dataset"] == ds]
        for col, cat in _SAMPLES.items():
            row[col] = int(sample_matches(sub, cat).sum())
        row["n_vus"] = int(sub["is_vus"].fillna(False).astype(bool).sum()) if "is_vus" in sub else 0
        src = df_all_groups[df_all_groups["dataset"] == ds] if df_all_groups is not None else sub
        snv_col = "is_snv" if "is_snv" in src.columns else None
        row["n_snv"] = int(src[snv_col].fillna(False).astype(bool).sum()) if snv_col else ""
        auth = sub["auth_label"].astype(str) if "auth_label" in sub else pd.Series(dtype=str)
        row["author_assignments_provided"] = bool(auth.str.upper().isin(["NORMAL", "ABNORMAL"]).any())
        try:
            row.update(fit_sample_counts(ds))
        except Exception as e:  # noqa: BLE001 -- reported, not hidden
            problems.append(f"{ds}: fit sample counts failed ({e})")
            row.update({f"{k}_fit": "" for k in _SAMPLES})

        cal = _resolved_calibration(ds, tree, model_selections, dataset_configs)
        pr = cal.get("point_ranges", {})
        for i in [*range(-8, 0), *range(1, 9)]:
            row[f"range_{i}"] = _fmt_ranges(pr.get(str(i), []))
        row["prior"] = cal.get("prior")
        row["relax"] = bool(cal.get("relax"))
        row["n_c"] = str(cal.get("n_c", "")).split("_", 1)[0]
        row["benign_method"] = cal.get("benign_method")
        row["clinvar_2018"] = bool(cal.get("clinvar_2018"))
        row["scoreset_flipped"] = bool(cal.get("scoreset_flipped"))

        yrow = yang_idx.loc[ds] if yang_idx is not None and ds in yang_idx.index else None
        if yrow is None:
            problems.append(f"{ds}: no Yang-distance row")
        for col, sample in _YANG_SAMPLES.items():
            n_fit = row.get(f"n_{sample}_fit", "")
            row[col] = _yang_cell(yrow, sample, int(n_fit) if str(n_fit).strip() != "" else 99)
        hrow = (yang_heldout_df.set_index("dataset").loc[ds]
                if yang_heldout_df is not None and ds in set(yang_heldout_df["dataset"]) else None)
        if yang_heldout_df is not None and hrow is None:
            problems.append(f"{ds}: no held-out Yang-distance row")
        for col, sample in _YANG_SAMPLES.items():
            n_fit = row.get(f"n_{sample}_fit", "")
            row[col.replace("yang_dist_", "yang_dist_heldout_")] = _heldout_yang_cell(
                hrow, sample, int(n_fit) if str(n_fit).strip() != "" else 99, cfg.YANG_HELDOUT_MIN_N)
        rows.append(row)
    out = pd.DataFrame(rows)[SD1_COLUMNS]
    out["dataset"] = out["dataset"].map(display_name)  # internal suffixes are not user-facing
    return out, problems


def load_oddspath_csv(op_csv: Optional[str] = None, **read_kw) -> pd.DataFrame:
    """The OddsPath CSV with cfg.OP_SUPERSEDED_BY applied: each superseding
    row (e.g. MSH2_Scott_2022) replaces the row it supersedes and takes that
    row's Dataset name (MSH2_Jia_2021, the name the calibrated dataset
    resolves to)."""
    df_op = pd.read_csv(op_csv or cfg.OP_EVIDENCE_CODES_CSV, **read_kw)
    for old, new in cfg.OP_SUPERSEDED_BY.items():
        if (df_op["Dataset"] == new).any():
            df_op = df_op[df_op["Dataset"] != old].copy()
            row = df_op["Dataset"] == new
            df_op.loc[row, "Dataset"] = old
            # Provenance note in SD3 (no column of its own; Pseudocount
            # Details is the CSV's free-text column).
            note = f"Values from the updated {new} OddsPath calibration (supersedes {old})"
            prev = df_op.loc[row, "Pseudocount Details"].astype(str).str.strip()
            df_op.loc[row, "Pseudocount Details"] = [note if v in ("None", "nan", "") else f"{v}; {note}" for v in prev]
    return df_op.reset_index(drop=True)


def oddspath_resolution(datasets: List[str], op_csv: Optional[str] = None) -> Dict[str, Optional[str]]:
    """calibrated dataset -> its row name in the OddsPath CSV (None = not covered)."""
    from analysis.discovery import resolve_dataset_tsv_name

    df_op = load_oddspath_csv(op_csv)
    return {d: resolve_dataset_tsv_name(d.replace("_clinvar_2018", ""), df_op, cfg.DATASET_TSV) for d in datasets}


def build_sd3(datasets: List[str], op_csv: Optional[str] = None) -> (pd.DataFrame, List[str]):
    df_op = load_oddspath_csv(op_csv, dtype=str)
    resolved = oddspath_resolution(datasets, op_csv)
    rows, problems = [], []
    for ds in datasets:
        name = resolved[ds]
        if name is not None:
            rec = df_op[df_op["Dataset"] == name].iloc[0].to_dict()
            # A side OddsPath cannot evaluate is "Not available", with the
            # reason in its Odds column -- the CSV's own convention (e.g.
            # OddsNormal "No functionally normal controls"). The CSV leaves
            # some such cells empty/"None" (KCNQ4_Zheng_2022_v12_homozygous:
            # its only P/LP control is assay-normal, so OddsAbnormal = 0).
            for side, true_path in (("Normal", None), ("Abnormal", "True Path in Abnormal")):
                code_col, odds_col = f"Evidence Code {side}", f"Odds{side}"
                if str(rec.get(code_col)).strip() in ("None", "nan", ""):
                    rec[code_col] = "Not available"
                    if true_path and str(rec.get(true_path)).strip() == "0":
                        rec[odds_col] = "No pathogenic controls in functionally abnormal class"
            rows.append(rec)
        else:
            rows.append({"Dataset": ds.replace("_clinvar_2018", ""), "OddsNormal": "Not available",
                         "OddsAbnormal": "Not available", "Evidence Code Normal": "Not available",
                         "Evidence Code Abnormal": "Not available"})
    out = pd.DataFrame(rows).reindex(columns=SD3_COLUMNS)
    from analysis.discovery import DEFAULT_EXCLUDED_DATASETS  # excluded on purpose (meta-analyses, SFPQ)
    _excluded = {d.replace("_clinvar_2018", "") for d in DEFAULT_EXCLUDED_DATASETS}
    unused = sorted(set(df_op["Dataset"]) - {v for v in resolved.values() if v} - _excluded)
    if unused:
        problems.append(f"OddsPath CSV rows not matching any calibrated dataset (omitted): {unused}")
    return out, problems
