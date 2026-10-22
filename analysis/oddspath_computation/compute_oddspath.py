#!/usr/bin/env python3
"""
OddsPath (Brnich et al. 2019) for every dataset in our integrated
variant-effect table (config.DATASET_TSV), computed with the collaborators'
method: a port of
igvf-scalable-vus-resolution/notebooks/analysis/OddsPath_calculations.ipynb
(see its README_OddsPath_calculations.md). Fig. 5B and Supplementary Data 3
read the output through config.OP_EVIDENCE_CODES_CSV.

Same as the notebook: CHEK2_Gebbia_2024 confidence-interval flags, splice
handling, the MSH2_Scott_2022 intervals, StandardizedClass from MaveDB's
functional-class category (or from the score intervals where the curation
sheet reports intervals but no classification), ClinVar 2018 for BRCA1,
PTEN, MSH2 and TP53 and 2025 otherwise, Odds_LR with its pseudocounts, the
evidence-code mapping and the KCNQ4_Zheng_2022_v12_homozygous mask.

Differences from the notebook (README.md):
  - A dataset qualifies when at least one of its variants is classed
    functionally normal or abnormal, instead of by the curation sheet's
    "Reported"/"Yes" flags (which leave out CHEK2_Gebbia_2024 and
    F9_Popp_2025_light_chain although both carry MaveDB classes).
  - No hard-coded SFPQ/F9 exclusions: every dataset is computed, then
    analysis.discovery.DEFAULT_EXCLUDED_DATASETS is applied at the end.
  - MSH2: only the MSH2_Scott_2022 row is written (it supersedes the
    MSH2_Jia_2021 intervals; config.OP_SUPERSEDED_BY maps it back to the
    calibrated dataset's name).
  - Odds_LR checks for zero controls before computing the prior (the
    notebook divided first and returned an error row).

    python -m analysis.oddspath_computation.compute_oddspath [--tsv PATH] [--out-dir DIR]

Writes to analysis/oddspath_computation/output/:
  OddsPath_calibrations_recomputed.csv   one row per dataset (collaborator CSV columns)
  excluded.csv                           rows dropped by DEFAULT_EXCLUDED_DATASETS
  not_computable.csv                     datasets without normal/abnormal classes
  comparison_to_collaborator.csv         per-dataset differences from config.OP_EVIDENCE_CODES_CSV_COLLABORATOR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import config as cfg

HERE = Path(__file__).resolve().parent
NOTEBOOK_REPO = HERE / "igvf-scalable-vus-resolution"
INPUT_MAVES_DIR = NOTEBOOK_REPO / "data" / "input" / "maves"
OUT_DIR = HERE / "output"
OUT_CSV = "OddsPath_calibrations_recomputed.csv"

PRIORITY_GENES = ["BRCA1", "PTEN", "MSH2", "TP53"]  # ClinVar 2018 controls
PATHOGENIC_GROUP = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_GROUP = {"Benign", "Likely benign", "Benign/Likely benign"}
CONFLICTS_GROUP = {"Conflicting classifications of pathogenicity"}
COL_HEADER, FUNCTIONAL, ABNORMAL_FUNC = "StandardizedClass", "NORMAL", "ABNORMAL"
COLUMNS = ["Dataset", "Total Controls", "OddsNormal", "OddsAbnormal", "Pathogenic Controls", "Benign Controls",
           "Prior Probability Pathogenic", "Total Assay Abnormal", "True Path in Abnormal", "Total Assay Normal",
           "True Path in Normal", "Pseudocount Details", "Evidence Code Normal", "Evidence Code Abnormal"]


# ---------------------------------------------------------------- variant table

def flag_chek2(pp: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 3: CHEK2_Gebbia_2024 variants with Filter_CI == 1 get Flag '*'."""
    chek2 = pd.read_excel(INPUT_MAVES_DIR / "CHEK2_Gebbia_2024.xlsx", header=0)
    chek2["score"] = chek2["score"].astype(object)
    tmp = pd.merge(pp, chek2, left_on=["hgvs_p", "auth_reported_score"], right_on=["hgvs_pro", "score"], how="left")
    tmp["Flag"] = np.where(tmp["Filter_CI"] == 1, "*", tmp["Flag"])
    return tmp.drop(columns=["aaChange", "hgvs_pro", "type", "score", "error", "LLR", "LLR_strength", "Filter_CI",
                             "Filter_Hypercomplement"])


def _splice_var(series):
    sigs = set(series.dropna())
    if len(sigs) == 0:
        return ""
    if len(sigs) == 1:
        return next(iter(sigs))
    return "Yes"


def flag_splice(pp: pd.DataFrame) -> pd.DataFrame:
    """Notebook cells 4-5: SpliceAI DS >= 0.2 or splice_site_variant, broadcast
    to every variant of the same amino-acid change in amino-acid assays."""
    cond2 = pp[["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]].ge(0.2).any(axis=1)
    cond3 = pp["simplified_consequence"] == "splice_site_variant"
    pp["splice_variant"] = np.where(cond2 | cond3, "Yes", "No")
    pp["Ref_seq_transcript_ID_stripped"] = pp["RefSeq Transcript ID"].str.replace(r"\.\d+$", "", regex=True)
    pp["aa_pos"] = pd.to_numeric(pp["aa_pos"], errors="coerce")
    for c in ("aa_ref", "aa_alt", "Gene", "Ref_seq_transcript_ID_stripped"):
        pp[c] = pp[c].astype(str)
    group_cols_aa = ["Gene", "aa_ref", "aa_pos", "aa_alt", "Ref_seq_transcript_ID_stripped"]
    mask_aa = pp["nucleotide_or_aa"] == "aa"
    pp.loc[mask_aa, "splice_var_amino"] = (
        pp.loc[mask_aa].groupby(group_cols_aa)["splice_variant"].transform(_splice_var))
    return pp


def add_msh2_scott(pp: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 6: MSH2_Jia_2021 again as MSH2_Scott_2022 (intervals set in apply_intervals)."""
    scott = pp[pp["Dataset"] == "MSH2_Jia_2021"].copy()
    scott["mavedb_variant_urn"] = scott["mavedb_variant_urn"] + ":Scott_2022"
    scott["Dataset"] = "MSH2_Scott_2022"
    return pd.concat([pp, scott])


def _class_from_category(category):
    """Notebook cell 7: MaveDB's functional-class category."""
    if pd.isna(category):
        return None
    return {"normal": "Normal", "abnormal": "Abnormal",
            "not_specified": "Indeterminate"}.get(str(category).strip().lower(), None)


def _parse_bound(bound_str):
    bound_str = bound_str.strip().lower()
    if bound_str in ["-inf", "-infinity"]:
        return float("-inf")
    if bound_str in ["inf", "infinity", "+inf"]:
        return float("inf")
    return float(bound_str)


def _score_in_interval(score, interval):
    try:
        left, right = interval[0], interval[-1]
        lower_str, upper_str = interval[1:-1].split(",")
        lower, upper = _parse_bound(lower_str), _parse_bound(upper_str)
        lower_check = score >= lower if left == "[" else score > lower
        upper_check = score <= upper if right == "]" else score < upper
        return lower_check and upper_check
    except Exception:
        return False


def _find_matching_class(row, score):
    """Notebook cell 13: the Class of the first interval (1-9) containing the score."""
    for i in range(1, 10):
        interval_col, class_col = f"Interval {i} Range", f"Interval {i} Class"
        if interval_col in row and pd.notna(row[interval_col]):
            if _score_in_interval(score, row[interval_col]):
                return row[class_col]
    return "Not specified"


def apply_intervals(pp: pd.DataFrame) -> pd.DataFrame:
    """Notebook cells 9-14: the curation sheet's flags; the Scott et al. 2022
    (PMID 36550560) intervals; classes from the score intervals where the
    sheet reports intervals but no classification."""
    mave = pd.read_excel(INPUT_MAVES_DIR / "Supplementary_Data_3.xlsx", sheet_name="Curation", header=0)
    pp = pd.merge(pp, mave[["Dataset Name", "Score Intervals Reported?", "Functional Classification Provided?"]],
                  left_on="Dataset", right_on="Dataset Name", how="left")
    scott = pp["Dataset"] == "MSH2_Scott_2022"
    pp.loc[scott, ["Interval 1 Name", "Interval 1 Range", "Interval 1 Class"]] = ["Normal", "(-inf, 0)", "Normal"]
    pp.loc[scott, ["Interval 2 Name", "Interval 2 Range", "Interval 2 Class"]] = ["Abnormal", "[0.4, inf)", "Abnormal"]
    pp.loc[scott, "Score Intervals Reported?"] = "Reported"
    pp.loc[scott, "Functional Classification Provided?"] = "No"
    mask = (pp["Functional Classification Provided?"].isin(["No"])) & (pp["Score Intervals Reported?"] == "Reported")
    pp.loc[mask, "StandardizedClass"] = pp.loc[mask].apply(
        lambda row: _find_matching_class(row, row["auth_reported_score"]), axis=1)
    return pp


def _summarize_clnsig(series):
    """Notebook cell 16: one ClinVar call per variant across its rows."""
    sigs = set(series.dropna())
    if len(sigs) == 0:
        return np.nan
    has_path = any(v in PATHOGENIC_GROUP for v in sigs)
    has_benign = any(v in BENIGN_GROUP for v in sigs)
    has_conflict = any(v in CONFLICTS_GROUP for v in sigs)
    has_uncertain = "Uncertain significance" in sigs
    if has_conflict or (has_path and has_benign):
        return "has clinvar conflict"
    if has_path:
        if "Likely pathogenic" in sigs or "Pathogenic/Likely pathogenic" in sigs:
            return "Likely pathogenic"
        return "Pathogenic"
    if has_benign:
        if "Likely benign" in sigs or "Benign/Likely benign" in sigs:
            return "Likely benign"
        return "Benign"
    if has_uncertain and len(sigs) == 1:
        return "Uncertain significance"
    if has_uncertain:
        return "VUS/conflict"
    if len(sigs) == 1:
        return next(iter(sigs))
    return "multiple other classifications"


def prepare_variants(tsv: str) -> pd.DataFrame:
    """The notebook's variant table up to cell 20 (one row per MaveDB variant),
    for every dataset (no curation-sheet subsetting)."""
    missing = [f for f in ("CHEK2_Gebbia_2024.xlsx", "Supplementary_Data_3.xlsx")
               if not (INPUT_MAVES_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found under {INPUT_MAVES_DIR}; clone the collaborators' repository first:\n"
            f"    git clone https://github.com/bbi-lab/igvf-scalable-vus-resolution {NOTEBOOK_REPO}")
    pp = pd.read_csv(tsv, sep="\t", low_memory=False)
    pp = flag_chek2(pp)
    pp = flag_splice(pp)
    pp = add_msh2_scott(pp)
    pp["StandardizedClass"] = pp["auth_reported_func_class_category"].map(_class_from_category)
    # Flagged variants (inconsistent mappings) and splice variants, except in
    # datasets curated as detecting splicing effects.
    pp = pp[(pp["Flag"] != "*") & ((pp["splice_var_amino"] != "Yes") | (pp["splice_measure"] == "Yes"))]
    pp = apply_intervals(pp)
    pp["clinvar_18_25"] = np.where(pp["Gene"].isin(PRIORITY_GENES), pp["clinvar_sig_2018"], pp["clinvar_sig_2025"])
    pp["clnsig_group_18_25"] = pp.groupby("mavedb_variant_urn")["clinvar_18_25"].transform(_summarize_clnsig)
    pp["StandardizedClass"] = pp["StandardizedClass"].str.upper().fillna(pp["StandardizedClass"])
    return pp.drop_duplicates("mavedb_variant_urn")


# ---------------------------------------------------------------- OddsPath

def odds_lr(df: pd.DataFrame, clnsig_group: str = "clnsig_group_18_25") -> dict:
    """Notebook cell 19 (OddsPath with +1 pseudocounts)."""
    dataset = df["Dataset"].iloc[0]
    pathogenic = df[df[clnsig_group].isin(PATHOGENIC_GROUP)]
    benign = df[df[clnsig_group].isin(BENIGN_GROUP)]
    pathogenic_controls, benign_controls = len(pathogenic), len(benign)
    total_controls = pathogenic_controls + benign_controls
    if total_controls == 0:
        return {"Dataset": dataset, "Total Controls": 0, "OddsNormal": "No controls", "OddsAbnormal": "No controls",
                "Pathogenic Controls": 0, "Benign Controls": 0, "Prior Probability Pathogenic": 0,
                "Total Assay Abnormal": 0, "True Path in Abnormal": 0, "Total Assay Normal": 0,
                "True Path in Normal": 0, "Pseudocount Details": "No controls"}
    prior_prob = pathogenic_controls / total_controls

    pseudo_reasons = []
    path_abnormal = len(pathogenic[pathogenic[COL_HEADER] == ABNORMAL_FUNC])
    path_normal = len(pathogenic[pathogenic[COL_HEADER] == FUNCTIONAL])
    benign_abnormal = len(benign[benign[COL_HEADER] == ABNORMAL_FUNC])
    benign_normal = len(benign[benign[COL_HEADER] == FUNCTIONAL])
    assay_abnormal_num = len(df[df[COL_HEADER] == ABNORMAL_FUNC])
    assay_normal_num = len(df[df[COL_HEADER] == FUNCTIONAL])
    if benign_abnormal == 0:
        benign_abnormal += 1
        pseudo_reasons.append("Added 1 misclassified benign variant to abnormal function class")
    if path_normal == 0:
        path_normal += 1
        pseudo_reasons.append("Added 1 misclassified pathogenic variant to normal function class")

    base = {"Dataset": dataset, "Total Controls": total_controls, "Pathogenic Controls": pathogenic_controls,
            "Benign Controls": benign_controls, "Prior Probability Pathogenic": prior_prob,
            "Total Assay Abnormal": assay_abnormal_num, "True Path in Abnormal": path_abnormal,
            "Total Assay Normal": assay_normal_num, "True Path in Normal": path_normal}
    for n, why in ((benign_controls, "No benign controls"), (pathogenic_controls, "No pathogenic controls")):
        if n == 0:
            return {**base, "OddsNormal": why, "OddsAbnormal": why, "Pseudocount Details": why}

    if assay_abnormal_num == 0:
        odds_abnormal = "No functionally abnormal controls"
    else:
        odds_abnormal = (path_abnormal / pathogenic_controls) / (benign_abnormal / benign_controls)
    if assay_normal_num == 0:
        odds_normal = "No functionally normal controls"
    else:
        odds_normal = (path_normal / pathogenic_controls) / (benign_normal / benign_controls)
    return {**base,
            "OddsNormal": round(odds_normal, 4) if isinstance(odds_normal, float) else odds_normal,
            "OddsAbnormal": round(odds_abnormal, 4) if isinstance(odds_abnormal, float) else odds_abnormal,
            "Pseudocount Details": "; ".join(pseudo_reasons) if pseudo_reasons else "None"}


def odds_to_evidence(x) -> str:
    """Notebook cell 23: OddsPath -> ACMG/ClinGen evidence strength."""
    if pd.isna(x) or isinstance(x, str):
        return "Not available"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "Not available"
    if x < 0.053:
        return "BS3_strong"
    if x < 0.23:
        return "BS3_moderate"
    if x < 0.48:
        return "BS3_supporting"
    if x <= 2.1:
        return "Indeterminate"
    if x <= 4.3:
        return "PS3_supporting"
    if x <= 18.7:
        return "PS3_moderate"
    if x <= 350:
        return "PS3_strong"
    return "PS3_very_strong"


def compute(tsv: str = None) -> (pd.DataFrame, pd.DataFrame):
    """(one OddsPath row per qualifying dataset, datasets that do not qualify)."""
    pp = prepare_variants(tsv or cfg.DATASET_TSV)
    classed = pp[pp[COL_HEADER].isin([FUNCTIONAL, ABNORMAL_FUNC])].groupby("Dataset").size()
    qualifying = sorted(set(classed.index) - {"MSH2_Jia_2021"})  # MSH2: Scott intervals only
    not_computable = pd.DataFrame({"Dataset": sorted(set(pp["Dataset"]) - set(classed.index)),
                                   "Reason": "no variants classed functionally normal or abnormal"})
    rows = pd.DataFrame([odds_lr(pp[pp["Dataset"] == d]) for d in qualifying])
    for col in ["OddsAbnormal", "OddsNormal", "Prior Probability Pathogenic"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").round(4).combine_first(rows[col])
    rows["Evidence Code Normal"] = rows["OddsNormal"].apply(odds_to_evidence)
    rows["Evidence Code Abnormal"] = rows["OddsAbnormal"].apply(odds_to_evidence)
    # Too few controls for a reliable Abnormal-direction estimate (notebook cell 25).
    rows.loc[rows["Dataset"] == "KCNQ4_Zheng_2022_v12_homozygous", "Evidence Code Abnormal"] = "None"
    return rows[COLUMNS], not_computable


def split_excluded(rows: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """(kept, excluded): DEFAULT_EXCLUDED_DATASETS, matched without the
    pipeline's internal _clinvar_2018 suffix."""
    from analysis.discovery import DEFAULT_EXCLUDED_DATASETS

    excluded = {d.replace("_clinvar_2018", "") for d in DEFAULT_EXCLUDED_DATASETS}
    drop = rows["Dataset"].isin(excluded)
    return rows[~drop].reset_index(drop=True), rows[drop].reset_index(drop=True)


def compare(rows: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    """Long table of every value that differs from the collaborators' CSV,
    plus datasets present in only one of the two."""
    ref = pd.read_csv(reference_csv, dtype=str).set_index("Dataset")
    new = rows.astype(str).set_index("Dataset")
    out = []
    for d in sorted(set(ref.index) | set(new.index)):
        if d not in new.index:
            out.append({"Dataset": d, "Column": "(row)", "Collaborator": "present", "Recomputed": "absent"})
            continue
        if d not in ref.index:
            out.append({"Dataset": d, "Column": "(row)", "Collaborator": "absent", "Recomputed": "present"})
            continue
        for c in COLUMNS[1:]:
            a, b = ref.at[d, c], new.at[d, c]
            try:
                same = np.isclose(float(a), float(b), rtol=0, atol=1e-9)
            except (TypeError, ValueError):
                same = a == b
            if not same:
                out.append({"Dataset": d, "Column": c, "Collaborator": a, "Recomputed": b})
    return pd.DataFrame(out, columns=["Dataset", "Column", "Collaborator", "Recomputed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=None, help="default: config.DATASET_TSV")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, not_computable = compute(args.tsv)
    kept, excluded = split_excluded(rows)
    kept.to_csv(out_dir / OUT_CSV, index=False)
    excluded.to_csv(out_dir / "excluded.csv", index=False)
    not_computable.to_csv(out_dir / "not_computable.csv", index=False)
    print(f"{len(kept)} datasets -> {out_dir / OUT_CSV}")
    print(f"excluded (DEFAULT_EXCLUDED_DATASETS): {excluded['Dataset'].tolist()}")
    print(f"not computable: {len(not_computable)}")

    diff = compare(kept, cfg.OP_EVIDENCE_CODES_CSV_COLLABORATOR)
    diff.to_csv(out_dir / "comparison_to_collaborator.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70):
        print(f"\n{len(diff)} differences from {cfg.OP_EVIDENCE_CODES_CSV_COLLABORATOR}:")
        print(diff.to_string(index=False) if len(diff) else "(none)")
