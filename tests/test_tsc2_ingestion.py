"""TSC2 (Biar et al. 2026) ingestion (multivariate_data/tsc2.py) and its
combined-panel switch (multivariate_data/combined.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.assay_calibration.multivariate_data import tsc2  # noqa: E402
from src.assay_calibration.multivariate_data.combined import (  # noqa: E402
    build_functional_scoresets, get_functionally_assayed_protein_variants,
)

STAR1 = "criteria provided, single submitter"


def _snv(aa_ref, aa_pos, aa_alt, sig=np.nan, star=np.nan, gnomad=np.nan, cons="missense_variant", spliceai=0.0):
    return {"Gene": "TSC2", "Dataset": "TSC2_IGVF", "aa_ref": aa_ref, "aa_pos": float(aa_pos), "aa_alt": aa_alt,
            "auth_reported_score": 0.5, "clinvar_sig_2026": sig, "clinvar_star_2026": star,
            "clinvar_sig_2025": sig, "clinvar_star_2025": star, "gnomad_MAF": gnomad,
            "simplified_consequence": cons, "ref_allele": "A", "alt_allele": "G",
            "spliceAI_DS_AG": spliceai, "spliceAI_DS_AL": 0.0, "spliceAI_DS_DG": 0.0, "spliceAI_DS_DL": 0.0}


def _integrated():
    return pd.DataFrame([
        _snv("A", 1546, "T", "Pathogenic", STAR1),                     # P/LP
        _snv("A", 1546, "T", "Likely pathogenic", STAR1),              # second codon, agrees
        _snv("L", 1600, "P", "Pathogenic", STAR1),                     # conflict ...
        _snv("L", 1600, "P", "Benign", STAR1),                         # ... -> unlabelled
        _snv("S", 1232, "S", "Likely benign", STAR1, cons="synonymous_variant"),
        _snv("R", 1700, "W", "Pathogenic", "no assertion criteria provided"),  # 0 star
        _snv("G", 1650, "D", "Pathogenic", STAR1, spliceai=0.5),       # splice-flagged
        _snv("K", 1580, "R", gnomad=1e-5),                             # gnomAD only
    ])


def _mave_csv(tmp_path):
    rows = [
        ("p.ALA1546THR", 0.1, 0.2, np.nan, "MIS"),
        ("p.LEU1600PRO", 0.3, np.nan, np.nan, "MIS"),
        ("p.SER1232SER", 1.0, 1.1, 0.0, "SYN"),
        ("p.SER1232TER", -0.1, np.nan, 0.5, "PTC"),
        ("p.ARG1700TRP", 0.2, np.nan, np.nan, "MIS"),
        ("p.GLY1650ASP", 0.4, np.nan, np.nan, "MIS"),
        ("p.LYS1580ARG", 0.9, np.nan, np.nan, "MIS"),
    ]
    df = pd.DataFrame(rows, columns=["hgvs_pro", "VAMPseq_ave", "immuneSGEscore", "cliPE_score", "set2"])
    df["gnomAD4.1_joint_AC"] = ["-", "-", "-", "3", "-", "-", "-"]
    df["clinvar_clnsig"] = ["Pathogenic", "-", "-", np.nan, "Pathogenic", "Pathogenic", "-"]
    path = tmp_path / "tsc2.csv"
    df.to_csv(path, index=False)
    return path


def _samples(df, pid):
    return df.loc[df["id"] == pid, "sample_assignments"].item()


def test_protein_id():
    assert tsc2.protein_id("p.ALA1546THR") == "A1546T"
    assert tsc2.protein_id("p.SER1232TER") == "S1232*"
    assert tsc2.protein_id("p.ASN1233ASN") == "N1233N"
    assert tsc2.protein_id("p.Xyz12Ala") is None


def test_labels_collapse_to_protein_changes(tmp_path):
    df = tsc2.build_tsc2_dataframe(_mave_csv(tmp_path), df_integrated=_integrated())
    assert _samples(df, "A1546T") == "0"         # both codons P/LP
    assert _samples(df, "L1600P") == ""          # conflicting SNVs
    assert _samples(df, "S1232S") == "1,3"       # B/LB + synonymous
    assert _samples(df, "S1232*") == "2"         # no integrated row; CSV gnomAD AC
    assert _samples(df, "R1700W") == ""          # 0-star label ignored
    assert _samples(df, "G1650D") == ""          # splice-flagged SNV gives no label
    assert _samples(df, "K1580R") == "2"


def test_scoresets_and_uv_frames(tmp_path):
    df = tsc2.build_tsc2_dataframe(_mave_csv(tmp_path), df_integrated=_integrated())
    ms = tsc2.build_tsc2_multiscoreset(df)
    assert list(ms.dataset_names) == tsc2.DATASET_NAMES
    assert list(ms.sample_names) == tsc2.SAMPLE_NAMES
    ss = tsc2.build_tsc2_functional_scoresets(df)
    assert sorted(ss[0]._ids) == ["A1546T", "K1580R", "S1232*", "S1232S"]   # labelled VAMP rows
    assert sorted(ss[2]._ids) == ["S1232*", "S1232S"]
    assert len(tsc2.build_tsc2_functional_scoresets(df, keep_unlabelled=True)[0].scores) == 7
    uv = tsc2.build_uv_dataframes(df)
    assert set(uv) == set(tsc2.DATASET_NAMES)
    assert all((u["sample_assignments"] != "").all() for u in uv.values())
    assert list(uv["TSC2_Biar_2026_immuneSGE"]["id"]) == ["A1546T", "S1232S"]


def test_combined_uses_tsc2_panel(tmp_path, monkeypatch):
    # combined.py builds the panel from tsc2.py's default MAVE CSV
    monkeypatch.setattr(tsc2.build_tsc2_dataframe, "__defaults__",
                        (str(_mave_csv(tmp_path)),) + tsc2.build_tsc2_dataframe.__defaults__[1:])
    integrated = _integrated()
    ss = build_functional_scoresets(integrated, "TSC2", ["TSC2_IGVF"])
    assert [s.scoreset_name for s in ss] == tsc2.DATASET_NAMES
    assayed = get_functionally_assayed_protein_variants(integrated, "TSC2", ["TSC2_IGVF"])
    assert assayed == {"A1546T", "L1600P", "S1232S", "S1232*", "R1700W", "G1650D", "K1580R"}
