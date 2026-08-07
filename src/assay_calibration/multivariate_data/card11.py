"""
CARD11 ingestion: raw master CSV -> 6-sample BasicMultiScoreset
(LOF score, GOF score).

Faithful port of prepare_CARD11_data.ipynb's ingestion cells.

``mask_gnomad``/``mask_synonymous`` were used in the notebook but never
defined in any saved cell (confirmed against the raw .ipynb JSON) --
confirmed by the user to be ``gnomAD_AF.notna()`` and
``"Variant class" == "synonymous"``, matching Scoreset's own
``is_gnomAD``/``is_synonymous`` conventions exactly.

Six samples, in fixed column order (0-5): P/LP, B/LB, population,
Synonymous, BENTA (GOF), CADINS (LOF). GOF-BENTA code 0 remaps to 4 and
LOF-CADINS code 0 remaps to 5; both drop their own code 1 (their local
"benign" class collapses into the shared B/LB set carried by the ClinVar
sources instead of being double counted). Merge order
[gof_benta, lof_cadins, gof_clinvar, lof_clinvar] determines which source
"wins" when the same hgvs_p has overlapping non-null scores.
"""

from typing import Optional

import pandas as pd

from ..data_utils.dataset import BasicMultiScoreset
from .common import build_basicmultiscoreset_from_wide_dataframe

DEFAULT_MASTER_CSV = (
    "/data/ross/assay_calibration/CARD11/"
    "CARD11_master_plus_splice_ClinSig_ClinVar_VariantClass.csv"
)

CLINVAR_BENIGN = ["Benign", "Likely benign", "Benign/Likely benign"]
CLINVAR_PATHOGENIC = ["Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"]

SCORE_COLS = ["lof_score", "gof_score"]
DATASET_NAMES = ["LOF score", "GOF score"]
SAMPLE_NAMES = [
    "Pathogenic/Likely Pathogenic", "Benign/Likely Benign",
    "population", "Synonymous", "BENTA (GOF)", "CADINS (LOF)",
]


def _make_sample_df(df, score_col, masks):
    sub = df[["hgvs_p", score_col]].copy()
    sub = sub.rename(columns={score_col: "score"})
    label_series = [
        mask.map({True: str(i), False: None})
        for i, (mask, _name) in enumerate(masks)
    ]
    sub["sample_assignments"] = pd.concat(label_series, axis=1).apply(
        lambda row: ",".join(v for v in row if v is not None), axis=1
    )
    sub = sub[sub["sample_assignments"] != ""]
    return sub.reset_index(drop=True)


def _parse_samples(assignments_str):
    if pd.isna(assignments_str) or str(assignments_str).strip() == "":
        return set()
    return {s.strip() for s in str(assignments_str).split(",")}


def _remap_samples(assignments_str, remap: dict, drop: set):
    samples = _parse_samples(assignments_str)
    result = set()
    for s in samples:
        if s in drop:
            continue
        result.add(remap.get(s, s))
    return result


def build_merged_6sample_dataframe(
    master_csv: str = DEFAULT_MASTER_CSV,
    df_card11: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the merged 6-sample dataframe, equivalent to
    processed_single_df/merged_6sample.csv.gz.
    """
    if df_card11 is None:
        df_card11 = pd.read_csv(master_csv)
    df = df_card11.copy()

    mask_gof_benign = df["GOF_ClinSig"] == "benign"
    mask_gof_benta = df["GOF_ClinSig"] == "BENTA"
    mask_lof_benign = df["LOF_ClinSig"] == "benign"
    mask_lof_cadins = df["LOF_ClinSig"] == "CADINS"
    mask_cv_benign = df["ClinVar classification"].isin(CLINVAR_BENIGN)
    mask_cv_path = df["ClinVar classification"].isin(CLINVAR_PATHOGENIC)
    mask_gnomad = df["gnomAD_AF"].notna()
    mask_synonymous = df["Variant class"] == "synonymous"

    gof_benta = _make_sample_df(df, "GOF_score", [
        (mask_gof_benta, "GOF BENTA"),
        (mask_gof_benign, "GOF benign"),
        (mask_gnomad, "gnomAD present"),
        (mask_synonymous, "synonymous"),
    ])
    lof_cadins = _make_sample_df(df, "LOF_score", [
        (mask_lof_cadins, "LOF CADINS"),
        (mask_lof_benign, "LOF benign"),
        (mask_gnomad, "gnomAD present"),
        (mask_synonymous, "synonymous"),
    ])
    gof_clinvar = _make_sample_df(df, "GOF_score", [
        (mask_cv_path, "ClinVar pathogenic"),
        (mask_cv_benign, "ClinVar benign"),
        (mask_gnomad, "gnomAD present"),
        (mask_synonymous, "synonymous"),
    ])
    lof_clinvar = _make_sample_df(df, "LOF_score", [
        (mask_cv_path, "ClinVar pathogenic"),
        (mask_cv_benign, "ClinVar benign"),
        (mask_gnomad, "gnomAD present"),
        (mask_synonymous, "synonymous"),
    ])

    gof_benta_proc = gof_benta.copy()
    gof_benta_proc["_samples"] = gof_benta_proc["sample_assignments"].apply(
        lambda x: _remap_samples(x, remap={"0": "4"}, drop={"1"})
    )
    lof_cadins_proc = lof_cadins.copy()
    lof_cadins_proc["_samples"] = lof_cadins_proc["sample_assignments"].apply(
        lambda x: _remap_samples(x, remap={"0": "5"}, drop={"1"})
    )
    gof_clinvar_proc = gof_clinvar.copy()
    gof_clinvar_proc["_samples"] = gof_clinvar_proc["sample_assignments"].apply(
        lambda x: _remap_samples(x, remap={}, drop=set())
    )
    lof_clinvar_proc = lof_clinvar.copy()
    lof_clinvar_proc["_samples"] = lof_clinvar_proc["sample_assignments"].apply(
        lambda x: _remap_samples(x, remap={}, drop=set())
    )

    gof_benta_proc["gof_score"] = gof_benta_proc["score"]
    gof_benta_proc["lof_score"] = float("nan")
    lof_cadins_proc["lof_score"] = lof_cadins_proc["score"]
    lof_cadins_proc["gof_score"] = float("nan")
    gof_clinvar_proc["gof_score"] = gof_clinvar_proc["score"]
    gof_clinvar_proc["lof_score"] = float("nan")
    lof_clinvar_proc["lof_score"] = lof_clinvar_proc["score"]
    lof_clinvar_proc["gof_score"] = float("nan")

    combined = pd.concat(
        [gof_benta_proc, lof_cadins_proc, gof_clinvar_proc, lof_clinvar_proc],
        ignore_index=True,
    )[["hgvs_p", "gof_score", "lof_score", "_samples"]]

    def aggregate_group(grp):
        all_samples = set()
        for s in grp["_samples"]:
            all_samples |= s
        sorted_samples = sorted(all_samples, key=lambda x: int(x))
        gof_score = grp["gof_score"].dropna().iloc[0] if grp["gof_score"].notna().any() else None
        lof_score = grp["lof_score"].dropna().iloc[0] if grp["lof_score"].notna().any() else None
        return pd.Series({
            "gof_score": gof_score,
            "lof_score": lof_score,
            "sample_assignments": ",".join(sorted_samples),
        })

    merged_6sample = (
        combined.groupby("hgvs_p", sort=False).apply(aggregate_group).reset_index()
    )[["hgvs_p", "gof_score", "lof_score", "sample_assignments"]]
    return merged_6sample


def build_card11_multiscoreset(
    merged_6sample: Optional[pd.DataFrame] = None,
    **build_kwargs,
) -> BasicMultiScoreset:
    """Build the 2-dimension, 6-sample CARD11 BasicMultiScoreset.

    Uses the shared wide-format helper (common.py) -- same shape as TP53's
    ingestion: both score dimensions are columns of one dataframe sharing a
    single per-variant sample_assignments column.
    """
    if merged_6sample is None:
        merged_6sample = build_merged_6sample_dataframe(**build_kwargs)

    return build_basicmultiscoreset_from_wide_dataframe(
        merged_6sample,
        score_cols=SCORE_COLS,
        id_col="hgvs_p",
        sample_assignments_col="sample_assignments",
        dataset_names=DATASET_NAMES,
        sample_names=SAMPLE_NAMES,
    )
