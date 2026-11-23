"""Per-row annotations of each dataset's production scoreset (consequence, SpliceAI, gnomAD AF, sample membership).

Rows follow the production Scoreset exactly (same filters as analysis.discovery.load_all_variants), keyed by the
production variant id (pipeline.variant_evidence._get_variant_ids), so they join 1:1 onto production variant tables.

    python -m csq_analysis.annotate [--workers 16]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from csq_analysis import config as C

SAI_COLS = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]
STD = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]


def _num(row, k):
    v = row.get(k)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def scoreset_rows(ds, df_ds):
    """DataFrame, one row per kept scoreset row: variant_id, score, simplified_consequence, csq (paper class),
    spliceai_max, gnomad_maf, membership flags plp/blb/pop/syn, detects_splice."""
    from analysis.membership_cells.cells import load_scoreset
    from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids, _get_variant_is_vus
    # all_assayed extra sample: every scored row is kept (assayed-only rows too); the standard memberships are
    # unchanged (Scoreset extra samples are fit-only columns)
    ss = load_scoreset(ds, df_ds, extra_sample="all_assayed")
    ids = _get_variant_ids(ss)
    vus = _get_variant_is_vus(ss)
    raw, sai, maf = [], [], []
    for all_idx, (_, variants) in enumerate(ss.get_variants_by_id().items()):
        if not ss._keep_mask[all_idx]:
            continue
        row = variants[0].row
        c = row.get("simplified_consequence")
        raw.append(str(c) if c is not None and c == c else "unknown")
        s = [_num(row, k) for k in SAI_COLS]
        s = [x for x in s if x == x]
        sai.append(max(s) if s else np.nan)
        maf.append(_num(row, "gnomad_MAF"))
    sa = np.asarray(ss._sample_assignments, bool)
    names = list(ss.sample_names)
    col = {n: names.index(n) for n in STD if n in names}
    flag = {k: (sa[:, col[n]] if n in col else np.zeros(len(ids), bool))
            for k, n in zip(("plp", "blb", "pop", "syn"), STD)}
    assert len(ids) == len(raw) == sa.shape[0], (ds, len(ids), len(raw), sa.shape)
    out = pd.DataFrame(dict(dataset=ds, variant_id=ids, score=np.asarray(ss.scores, float), simplified_consequence=raw,
                            csq=[C.fine_class(x) for x in raw], spliceai_max=sai, gnomad_maf=maf, **flag))
    out["vus"] = np.asarray(vus, bool) if vus is not None else False
    out["detects_splice"] = bool(getattr(ss, "detects_splice", False))
    return out


def _one(ds, df_ds):
    try:
        return scoreset_rows(ds, df_ds)
    except Exception as e:  # report, don't hide
        return pd.DataFrame(dict(dataset=[ds], error=[repr(e)[:300]]))


def build(workers=16):
    from joblib import Parallel, delayed
    from analysis import config as acfg
    from analysis.discovery import _filter_dataset_df, load_master_df
    dss = C.canonical_datasets()
    df_full = load_master_df(acfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, acfg.DATASET_TSV) for ds in dss}
    del df_full
    res = Parallel(n_jobs=workers, verbose=5)(delayed(_one)(ds, slices[ds]) for ds in dss)
    rows = pd.concat(res, ignore_index=True)
    bad = rows[rows.get("error").notna()] if "error" in rows else rows.iloc[:0]
    if len(bad):
        print("FAILED:", bad[["dataset", "error"]].to_string())
    rows = rows[rows.get("error").isna()].drop(columns="error") if "error" in rows else rows
    C.ANNOT.mkdir(parents=True, exist_ok=True)
    out = C.ANNOT / "rows.pkl"
    rows.to_pickle(out)
    print(f"{rows.dataset.nunique()} datasets, {len(rows)} rows -> {out}")
    return rows


def load_rows():
    return pd.read_pickle(C.ANNOT / "rows.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    build(ap.parse_args().workers)


if __name__ == "__main__":
    main()
