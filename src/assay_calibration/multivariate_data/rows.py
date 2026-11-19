"""
Row builder for the combined (functional + predictor) gene set, "v2".

One row per protein change for coding variants, so protein-level
information -- the three predictors and protein-level assays -- is counted
once. The combined model is a calibration of missense variants not predicted
to affect splicing: SNVs with SpliceAI > ``spliceai_threshold`` are left out
of every assay (non-splice assays already drop them in
``Scoreset.splicing_filter``; splice-detecting assays drop them here).
Remaining SNV-level measurements of the same protein change are averaged
per assay. Non-coding SNVs (no protein change) keep their own rows, keyed
"g:<chrom>:<pos>:<ref>:<alt>"; they have no predictor scores.

Labels: one source per row.
  - Rows with a functional score take every membership from the functional
    side only (the gene's ClinVar release, >=1 star), combined over the
    row's entries with the univariate "any" rule (data_utils.dataset.Scoreset:
    P/LP if any member is P/LP, B/LB if any is B/LB, likewise gnomAD and
    synonymous). A row that is both P/LP and B/LB keeps both memberships
    (its own stratum in fit_utils/strata.py, never a calibration stratum).
    gnomAD membership from the predictor CSV is OR-ed in (population
    membership does not depend on the ClinVar release).
  - Predictor-only rows (no functional score) keep the predictor CSV's
    labels (its sidecar ClinVar labels, which agree with ClinVar 2025 on
    every variant both label).

Predictor values: the predictor CSV's value wherever the CSV has the
protein change (so a variant scores the same as in the predictor-only gene
set; the CSVs never contain a tool's own training variants). Variants the
CSV lacks take the integrated dataframe's columns (``REVEL``, ``AM_score``,
``MutPred2``): AlphaMissense and MutPred2 are protein-level; REVEL is per
SNV and is averaged over the row's own SNVs (all low-SpliceAI SNVs of the
protein change when the panel is protein-level, e.g. TP53/TSC2); a tool's
value is blanked for its own training variants (``REVEL_train``,
``MP2_train``). Predictor-only rows use the CSV values.

Every predictor value goes through the same transform as the predictor
CSVs (predictors.transform_predictor_scores, reference = the gene's CSV
column), so all rows share one scale.

Output: a BasicMultiScoreset (predictor dims first, then functional dims)
with ``row_meta`` (a DataFrame aligned with ``kept_variants``) and
``row_builder = "v2"``.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data_utils.dataset import BasicMultiScoreset, BasicScoreset
from .predictors import (
    PREDICTORS, PREDICTOR_DATASET_NAMES, SAMPLE_COLUMNS, load_predictor_data,
    resolve_predictor_transform, transform_predictor_scores,
)

SAMPLE_NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]
SPLICEAI_COLS = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]
# integrated-dataframe predictor columns and training flags, per predictor CSV key
INTEGRATED_PREDICTOR_COLS = {"REVEL": ("REVEL", "REVEL_train"), "MP2": ("MutPred2", "MP2_train"),
                             "AM": ("AM_score", None)}
PER_SNV_PREDICTORS = {"REVEL"}

_PTV = {"stop_gained", "frameshift_variant", "start_lost", "stop_lost"}


def consequence_class(protein_id: Optional[str], simplified: Optional[str] = None) -> str:
    """missense / PTV / synonymous / other, from VEP's simplified consequence
    when given, else from the short protein id ("A119T", "S1232*", "N1233N")."""
    if isinstance(simplified, str) and simplified:
        if simplified == "missense_variant":
            return "missense"
        if simplified == "synonymous_variant":
            return "synonymous"
        if simplified in _PTV:
            return "PTV"
        if protein_id is None:
            return "other"
    if not protein_id or protein_id.startswith("g:"):
        return "other"
    ref, alt = protein_id[0], protein_id[-1]
    if alt == "*" and ref != "*":
        return "PTV"
    if ref == alt:
        return "synonymous"
    return "missense"


def _max_spliceai(rows: List[dict]) -> float:
    vals = [r.get(c) for r in rows for c in SPLICEAI_COLS]
    vals = [float(v) for v in vals if v is not None and pd.notna(v)]
    return max(vals) if vals else np.nan


def _functional_entries(fs, spliceai_threshold: float) -> pd.DataFrame:
    """One record per kept entry of a functional scoreset: row key, score,
    memberships, consequence, SpliceAI, member SNV atoms."""
    name = getattr(fs, "scoreset_name", None)
    sa = np.asarray(fs._sample_assignments, dtype=bool)
    if sa.shape[1] < 4:
        sa = np.hstack([sa, np.zeros((sa.shape[0], 4 - sa.shape[1]), dtype=bool)])
    sa = sa[:, :4]
    scores = np.asarray(fs.scores, dtype=float)
    recs = []
    if hasattr(fs, "get_variants_by_id") and hasattr(fs, "_keep_mask"):
        groups = list(fs.get_variants_by_id().values())
        kept = [g for g, k in zip(groups, np.asarray(fs._keep_mask, dtype=bool)) if k]
        if len(kept) != len(scores):
            raise ValueError(f"{name}: {len(kept)} kept groups vs {len(scores)} scores")
        detects_splice = bool(getattr(fs, "detects_splice", False))
        for k, variants in enumerate(kept):
            rows = [v.row for v in variants]
            v0 = variants[0]
            try:
                pid = str(v0.aa_ref) + str(int(v0.aa_pos)) + str(v0.aa_alt)
                if "nan" in pid.lower():
                    pid = None
            except (TypeError, ValueError):
                pid = None
            atoms = sorted({(str(r.get("Chrom")), int(r["hg38_start"]), str(r.get("ref_allele")), str(r.get("alt_allele")))
                            for r in rows if pd.notna(r.get("hg38_start"))})
            sai = _max_spliceai(rows)
            if detects_splice and np.isfinite(sai) and sai > spliceai_threshold:
                continue  # combined scope: missense not predicted to affect splicing
            csq = consequence_class(pid, rows[0].get("simplified_consequence"))
            key = pid if pid is not None else ("g:" + ":".join(map(str, atoms[0])) if atoms else None)
            if key is None:
                continue
            recs.append(dict(key=key, protein_id=pid, dim=name, score=scores[k], P=sa[k, 0], B=sa[k, 1],
                             g=sa[k, 2], s=sa[k, 3], consequence=csq, spliceai=sai, atoms=tuple(atoms),
                             splice_assay=detects_splice))
    else:  # BasicScoreset panels keyed by protein id (TP53, TSC2)
        ids = np.asarray(fs._ids).astype(str)
        for k, pid in enumerate(ids):
            recs.append(dict(key=pid, protein_id=pid, dim=name, score=scores[k], P=sa[k, 0], B=sa[k, 1],
                             g=sa[k, 2], s=sa[k, 3], consequence=consequence_class(pid), spliceai=np.nan,
                             atoms=(), splice_assay=False))
    return pd.DataFrame(recs)


def _integrated_predictor_table(df_gene: pd.DataFrame, spliceai_threshold: float) -> pd.DataFrame:
    """Per SNV of the gene in the integrated dataframe: protein id, genomic
    atom, the three predictor values and training flags (low-SpliceAI SNVs
    only)."""
    d = df_gene[df_gene["aa_ref"].notna() & df_gene["aa_pos"].notna() & df_gene["aa_alt"].notna()
                & df_gene["hg38_start"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["protein_id", "atom"])
    d["protein_id"] = d["aa_ref"].astype(str) + d["aa_pos"].astype(float).astype(int).astype(str) + d["aa_alt"].astype(str)
    d["atom"] = list(zip(d["Chrom"].astype(str), d["hg38_start"].astype(float).astype(int),
                         d["ref_allele"].astype(str), d["alt_allele"].astype(str)))
    sai = d[[c for c in SPLICEAI_COLS if c in d.columns]].max(axis=1)
    d = d[~(sai > spliceai_threshold)]
    keep = ["protein_id", "atom"]
    for col, tcol in INTEGRATED_PREDICTOR_COLS.values():
        if col in d.columns:
            keep.append(col)
        if tcol is not None and tcol in d.columns:
            keep.append(tcol)
    return d[keep].drop_duplicates("atom")


def _flag(series) -> bool:
    return bool(series.astype(str).str.lower().isin(["true", "1", "1.0"]).any())


def build_combined_rows(
    gene: str,
    functional_scoresets: List,
    predictor_data_dir: str,
    df_integrated: pd.DataFrame,
    predictors: Optional[List[str]] = None,
    predictor_transform: Optional[str] = None,
    spliceai_threshold: float = 0.2,
) -> Optional[BasicMultiScoreset]:
    """The v2 combined multiscoreset for ``gene`` (see module docstring).
    ``functional_scoresets`` are combined.build_functional_scoresets's output
    (labels at the gene's ClinVar release; pass regularization_type=
    "all_assayed" there to keep unlabelled functional rows for scoring VUS).
    Returns None if no predictor CSVs are available for ``gene``."""
    predictors = list(predictors) if predictors is not None else list(PREDICTORS)
    transform = resolve_predictor_transform(predictor_transform)
    pdfs: Dict[str, pd.DataFrame] = load_predictor_data(predictor_data_dir, genes=[gene]).get(gene, {})
    have = [p for p in predictors if p in pdfs]
    if not have:
        return None

    # ── functional side ──────────────────────────────────────────────────
    E = pd.concat([_functional_entries(fs, spliceai_threshold) for fs in functional_scoresets], ignore_index=True)
    E = E[np.isfinite(E["score"].astype(float))]
    func_names = [fs.scoreset_name for fs in functional_scoresets]
    vals = E.groupby(["key", "dim"])["score"].mean().unstack("dim").reindex(columns=func_names)
    n_entries = E.groupby(["key", "dim"]).size().unstack("dim").reindex(columns=func_names).fillna(0).astype(int)
    lab = E.groupby("key")[["P", "B", "g", "s"]].any()
    meta = E.groupby("key").agg(
        protein_id=("protein_id", "first"), consequence=("consequence", "first"),
        spliceai_max=("spliceai", "max"), atoms=("atoms", lambda a: tuple(sorted(set().union(*map(set, a))))),
    )

    # ── predictor values ─────────────────────────────────────────────────
    csv = {}
    for p in have:
        d = pdfs[p].dropna(subset=["protein_variant", "score"]).copy()
        d["protein_variant"] = d["protein_variant"].astype(str)
        onehot = (d["sample_assignments"].astype(str).str.get_dummies(sep=",")
                  .reindex(columns=SAMPLE_COLUMNS, fill_value=0).astype(bool))
        d["P"], d["B"], d["g"] = onehot["0"].to_numpy(), onehot["1"].to_numpy(), onehot["2"].to_numpy()
        csv[p] = d.drop_duplicates("protein_variant").set_index("protein_variant")

    pt = _integrated_predictor_table(df_integrated[df_integrated["Gene"] == gene], spliceai_threshold)
    by_pid = dict(tuple(pt.groupby("protein_id"))) if len(pt) else {}

    func_keys = list(vals.index)
    pred_vals = {p: np.full(len(func_keys), np.nan) for p in have}
    blanked = {p: 0 for p in have}
    source = {p: np.array(["none"] * len(func_keys), dtype=object) for p in have}
    for i, key in enumerate(func_keys):
        pid = meta.at[key, "protein_id"]
        if pid is None or meta.at[key, "consequence"] != "missense":
            continue  # predictors only describe missense changes
        snvs = by_pid.get(pid)
        if snvs is not None and meta.at[key, "atoms"]:
            own = snvs[snvs["atom"].isin(set(meta.at[key, "atoms"]))]
            if len(own):
                snvs = own
        for p in have:
            # The predictor CSV first, so every variant it has scores the same
            # as in the predictor-only gene set (it never contains the tool's
            # training variants); the integrated dataframe only fills variants
            # the CSV lacks, with training variants blanked.
            if pid in csv[p].index:
                pred_vals[p][i] = float(csv[p].at[pid, "score"])
                source[p][i] = "csv"
                continue
            col, tcol = INTEGRATED_PREDICTOR_COLS[p]
            if snvs is not None and col in snvs.columns:
                if tcol is not None and tcol in snvs.columns and _flag(snvs[tcol]):
                    blanked[p] += 1
                    source[p][i] = "training-blanked"
                    continue
                v = pd.to_numeric(snvs[col], errors="coerce").dropna()
                if len(v):
                    pred_vals[p][i] = float(v.mean()) if p in PER_SNV_PREDICTORS else float(v.iloc[0])
                    source[p][i] = "integrated"
    for p in have:
        ref = pdfs[p]["score"].to_numpy(dtype=float)
        pred_vals[p] = transform_predictor_scores(pred_vals[p], ref, transform)

    # gnomAD membership from the CSVs (release-independent) for functional rows
    csv_g = set().union(*[set(csv[p].index[csv[p]["g"]]) for p in have])
    lab["g"] = lab["g"] | np.array([meta.at[k, "protein_id"] in csv_g for k in lab.index])

    # ── predictor-only rows ──────────────────────────────────────────────
    func_pids = set(meta["protein_id"].dropna())
    po_ids = sorted(set().union(*[set(csv[p].index) for p in have]) - func_pids)
    po = pd.DataFrame(index=po_ids)
    for p in have:
        c = csv[p].reindex(po_ids)
        ref = pdfs[p]["score"].to_numpy(dtype=float)
        po[p] = transform_predictor_scores(c["score"].to_numpy(dtype=float), ref, transform)
        for col in ("P", "B", "g"):
            po[f"{col}_{p}"] = c[col].fillna(False).astype(bool).to_numpy()
    po_lab = pd.DataFrame({col: np.any([po[f"{col}_{p}"] for p in have], axis=0) for col in ("P", "B", "g")},
                          index=po_ids)
    po_lab["s"] = False
    po_lab = po_lab[po_lab[["P", "B", "g"]].any(axis=1)]
    po = po.loc[po_lab.index]

    # ── assemble ─────────────────────────────────────────────────────────
    all_keys = func_keys + list(po.index)
    lab_all = pd.concat([lab.loc[func_keys], po_lab])[["P", "B", "g", "s"]]
    sa_all = lab_all.to_numpy(dtype=bool)
    pos = {k: i for i, k in enumerate(all_keys)}

    pred_names = [PREDICTOR_DATASET_NAMES[p] for p in have]
    scoresets = []
    for p, nm in zip(have, pred_names):
        v = np.concatenate([pred_vals[p], po[p].to_numpy(dtype=float)])
        ok = np.isfinite(v)
        scoresets.append(BasicScoreset(scores=v[ok], sample_assignments=sa_all[ok], ids=np.array(all_keys)[ok],
                                       scoreset_name=nm))
    for nm in func_names:
        v = vals[nm].to_numpy(dtype=float)
        ok = np.isfinite(v)
        idx = [pos[k] for k in np.array(func_keys)[ok]]
        scoresets.append(BasicScoreset(scores=v[ok], sample_assignments=sa_all[idx], ids=np.array(func_keys)[ok],
                                       scoreset_name=nm))
    ms = BasicMultiScoreset(scoresets, dataset_names=pred_names + func_names, sample_names=SAMPLE_NAMES)

    cats = {s.scoreset_name: s.categorical_levels for s in functional_scoresets
            if getattr(s, "categorical_levels", None)}
    if cats:
        ms.categorical_dims = cats

    # row metadata aligned with kept_variants
    kept = list(ms.kept_variants)
    rm = pd.DataFrame(index=pd.Index(kept, name="row_key"))
    is_func = rm.index.isin(func_keys)
    rm["protein_id"] = [meta.at[k, "protein_id"] if k in meta.index else k for k in kept]
    rm["consequence"] = [meta.at[k, "consequence"] if k in meta.index else "missense" for k in kept]
    rm["spliceai_max"] = [meta.at[k, "spliceai_max"] if k in meta.index else np.nan for k in kept]
    rm["label_source"] = np.where(is_func, "functional", "predictor")
    nent = n_entries.reindex(kept).fillna(0).astype(int)
    rm["max_entries_per_assay"] = nent.max(axis=1).to_numpy()
    for p in have:
        s = pd.Series(source[p], index=func_keys)
        rm[f"src_{PREDICTOR_DATASET_NAMES[p]}"] = [s.get(k, "csv") if k in s.index else "csv" for k in kept]
    ms.row_meta = rm
    ms.row_builder = "v2"
    ms.row_build_report = dict(
        gene=gene, n_rows=len(kept), n_functional_rows=int(is_func.sum()), n_predictor_only=int((~is_func).sum()),
        training_blanked={PREDICTOR_DATASET_NAMES[p]: int(blanked[p]) for p in have},
        both_PB=int((sa_all[:, 0] & sa_all[:, 1]).sum()),
        averaged_cells=int((n_entries.to_numpy() > 1).sum()),
    )
    return ms
