"""
TSC2 ingestion: Biar et al., Nat Commun 17, 8474 (2026) protein-level MAVE
CSV -> 3-dimension BasicMultiScoreset (VAMP-seq, immune SGE, cliPE).

The CSV is keyed by protein change only (``hgvs_pro``, e.g. "p.SER1232TER"),
with no genomic coordinates, so the panel is protein-level (ids in the short
"A1546T" / "S1232*" / "A1546A" form used by the predictors and by
Scoreset.aa_subs), like TP53 and CARD11. VAMP-seq is the same experiment as
the integrated dataframe's TSC2_IGVF (MaveDB 00001267-0-3, residues 554-757
and 1512-1805), reprocessed to one score per protein change and with nonsense
variants added.

Labels are not taken from the CSV's own ClinVar column (missense only, no
review status). Instead each TSC2 SNV in the integrated dataframe is
classified with the pipeline's own rules (data_utils.dataset.Variant:
ClinVar release, minimum review status) and collapsed onto its protein
change: P/LP (or B/LB) when its labelled SNVs agree, unlabelled when they
conflict. SNVs the splice filter would drop (VEP splice consequence or any
SpliceAI delta >= 0.2, as Scoreset.splicing_filter) do not contribute a
ClinVar label, since a protein-level score says nothing about a splice
effect. gnomAD membership is any SNV with a gnomAD MAF, or the CSV's
gnomAD 4.1 allele count > 0. Synonymous is the CSV's own class.

Four samples, in fixed column order (0-3): P/LP, B/LB, gnomAD, Synonymous.
"""

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data_utils.dataset import BasicMultiScoreset, BasicScoreset, Variant

DEFAULT_MAVE_CSV = "/data/ross/assay_calibration/TSC2/TSC2maves_ToMergeWithAmbry_V2_wDT.csv"
DEFAULT_INTEGRATED_DATAFRAME = (
    "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
)

SCORE_COLS = ["VAMPseq_ave", "immuneSGEscore", "cliPE_score"]
DATASET_NAMES = ["TSC2_Biar_2026_VAMPseq", "TSC2_Biar_2026_immuneSGE", "TSC2_Biar_2026_cliPE"]
SAMPLE_NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous"]

_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "TER": "*",
}
_HGVS_PRO = re.compile(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$")
_SPLICE_CONSEQUENCES = {"splice region", "splice_site_variant", "splice_acceptor_variant", "splice_donor_variant"}
_SPLICEAI_COLS = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]


def protein_id(hgvs_pro: str) -> Optional[str]:
    """"p.SER1232TER" -> "S1232*"; synonymous "p.ASN1233ASN" -> "N1233N"."""
    m = _HGVS_PRO.match(str(hgvs_pro).strip())
    if m is None:
        return None
    ref, pos, alt = _AA3.get(m.group(1).upper()), m.group(2), _AA3.get(m.group(3).upper())
    if ref is None or alt is None:
        return None
    return f"{ref}{int(pos)}{alt}"


def snv_splice_flag(df: pd.DataFrame, spliceai_threshold: Optional[float] = 0.2) -> np.ndarray:
    """Rows Scoreset.splicing_filter would drop for an assay that does not
    detect splice effects (its defaults: VEP splice consequence, or any
    SpliceAI delta >= threshold)."""
    flag = df["simplified_consequence"].astype(str).str.lower().isin(_SPLICE_CONSEQUENCES).to_numpy()
    if spliceai_threshold is not None:
        flag |= df[_SPLICEAI_COLS].ge(spliceai_threshold).any(axis=1).to_numpy()
    return flag


def protein_labels_from_integrated(
    df_integrated: pd.DataFrame,
    clinvar_release: str = "2026",
    min_clinvar_star: int = 1,
    spliceai_threshold: Optional[float] = 0.2,
) -> pd.DataFrame:
    """Per protein change (index: short id): ``plp``/``blb`` (labelled SNVs
    agree), ``conflict`` (they disagree) and ``gnomad`` (any SNV in gnomAD),
    from TSC2's SNVs in the integrated dataframe."""
    df = df_integrated[df_integrated["Gene"] == "TSC2"]
    df = df[df["aa_ref"].notna() & df["aa_pos"].notna() & df["aa_alt"].notna()]
    ids = df["aa_ref"].astype(str) + df["aa_pos"].astype(float).astype(int).astype(str) + df["aa_alt"].astype(str)
    variants = [Variant(r, min_clinvar_star, clinvar_release, "auth_reported_score")
                for r in df.to_dict("records")]
    splice = snv_splice_flag(df, spliceai_threshold)
    snv = pd.DataFrame({
        "id": ids.to_numpy(),
        "p": np.array([v.is_pathogenic for v in variants], dtype=bool) & ~splice,
        "b": np.array([v.is_benign for v in variants], dtype=bool) & ~splice,
        "gnomad": np.array([v.is_gnomAD for v in variants], dtype=bool),
    })
    g = snv.groupby("id").agg(p=("p", "any"), b=("b", "any"), gnomad=("gnomad", "any"))
    return pd.DataFrame({
        "plp": g["p"] & ~g["b"],
        "blb": g["b"] & ~g["p"],
        "conflict": g["p"] & g["b"],
        "gnomad": g["gnomad"],
    })


def build_tsc2_dataframe(
    mave_csv: str = DEFAULT_MAVE_CSV,
    df_integrated: Optional[pd.DataFrame] = None,
    clinvar_release: str = "2026",
    min_clinvar_star: int = 1,
    spliceai_threshold: Optional[float] = 0.2,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per protein change with the three scores, ``id`` and a
    4-class ``sample_assignments`` column (see module docstring)."""
    df = pd.read_csv(mave_csv)
    df["id"] = df["hgvs_pro"].map(protein_id)
    if df["id"].isna().any():
        raise ValueError(f"unparsed hgvs_pro: {df.loc[df['id'].isna(), 'hgvs_pro'].tolist()[:5]}")
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate protein changes: {df.loc[df['id'].duplicated(), 'id'].tolist()[:5]}")
    if df_integrated is None:
        df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
    labels = protein_labels_from_integrated(df_integrated, clinvar_release, min_clinvar_star, spliceai_threshold)
    labels = labels.reindex(df["id"], fill_value=False)

    gnomad_ac = pd.to_numeric(df["gnomAD4.1_joint_AC"], errors="coerce")
    sample_assignments = np.zeros((len(df), len(SAMPLE_NAMES)), dtype=bool)
    sample_assignments[:, 0] = labels["plp"].to_numpy()
    sample_assignments[:, 1] = labels["blb"].to_numpy()
    sample_assignments[:, 2] = labels["gnomad"].to_numpy() | (gnomad_ac > 0).to_numpy()
    sample_assignments[:, 3] = (df["set2"] == "SYN").to_numpy()

    if verbose:
        csv_labelled = df["clinvar_clnsig"].isin([
            "Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic",
            "Benign", "Likely_benign", "Benign/Likely_benign"])
        no_label = csv_labelled & ~sample_assignments[:, :2].any(axis=1)
        print(f"TSC2: P/LP {sample_assignments[:, 0].sum()}, B/LB {sample_assignments[:, 1].sum()}, "
              f"conflicting {int(labels['conflict'].sum())}; {int(no_label.sum())} of "
              f"{int(csv_labelled.sum())} CSV-labelled variants have no pipeline label")

    out = df[["id", "hgvs_pro", "set2"] + SCORE_COLS].copy()
    out["sample_assignments"] = [",".join(map(str, np.flatnonzero(row))) for row in sample_assignments]
    return out


def build_tsc2_functional_scoresets(
    df_tsc2: Optional[pd.DataFrame] = None,
    keep_unlabelled: bool = False,
    **build_kwargs,
) -> List[BasicScoreset]:
    """One BasicScoreset per dataset, keyed on the short protein id -- so
    combined.py can join them with the predictor dimensions, as
    tp53.build_tp53_functional_scoresets. Rows without a score in that
    dataset are left out, and so are rows without any sample (Scoreset's
    default keep mask, as for TSC2_IGVF) unless ``keep_unlabelled`` (for
    scoring VUS, as Scoreset's regularization_type="all_assayed")."""
    if df_tsc2 is None:
        df_tsc2 = build_tsc2_dataframe(**build_kwargs)
    scoresets = []
    for col, name in zip(SCORE_COLS, DATASET_NAMES):
        keep = df_tsc2[col].notna()
        if not keep_unlabelled:
            keep &= df_tsc2["sample_assignments"] != ""
        sub = df_tsc2[keep]
        scoresets.append(BasicScoreset(
            scores=sub[col].to_numpy(dtype=float),
            sample_assignments=sub["sample_assignments"].to_numpy(),
            ids=sub["id"].astype(str).to_numpy(),
            scoreset_name=name,
            sample_names=SAMPLE_NAMES,
        ))
    return scoresets


def build_tsc2_multiscoreset(df_tsc2: Optional[pd.DataFrame] = None, **build_kwargs) -> BasicMultiScoreset:
    """The 3-dimension BasicMultiScoreset for TSC2."""
    return BasicMultiScoreset(
        build_tsc2_functional_scoresets(df_tsc2, **build_kwargs),
        dataset_names=DATASET_NAMES,
        sample_names=SAMPLE_NAMES,
    )


def functionally_assayed_ids(df_tsc2: Optional[pd.DataFrame] = None, **build_kwargs) -> set:
    """Every protein change any of the three assays scored, labelled or not
    (combined.split_predictor_clinvar_leakage's functional side)."""
    if df_tsc2 is None:
        df_tsc2 = build_tsc2_dataframe(**build_kwargs)
    return set(df_tsc2.loc[df_tsc2[SCORE_COLS].notna().any(axis=1), "id"].astype(str))


def build_uv_dataframes(df_tsc2: Optional[pd.DataFrame] = None, **build_kwargs) -> Dict[str, pd.DataFrame]:
    """Per dataset: (id, score, sample_assignments) for the univariate
    pipeline (hpc/prepare.py basicscoreset), labelled rows only."""
    if df_tsc2 is None:
        df_tsc2 = build_tsc2_dataframe(**build_kwargs)
    out = {}
    for col, name in zip(SCORE_COLS, DATASET_NAMES):
        sub = df_tsc2[df_tsc2[col].notna() & (df_tsc2["sample_assignments"] != "")]
        out[name] = (sub[["id", col, "sample_assignments"]]
                     .rename(columns={col: "score"}).reset_index(drop=True))
    return out
