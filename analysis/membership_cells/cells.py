"""Membership-cell fits: every variant in exactly one disjoint cell (membership pattern x simplified consequence), one
shared set of components, a mixing-weight vector per cell; any standard sample is composed afterwards.

Why: in the ECM each training row's responsibilities sum to 1, so a variant's total pull on the components is
fixed by its row weight; its sample only decides how that pull is split (r_vk = w_{s(v)k} phi_k(x_v) / f_s(x_v)).
With disjoint cells nothing is counted twice and no random one-hot assignment is needed; every cell keeps its
leverage on the components while having its own mixing weights. A union U of cells is a mixture of mixtures, so
its weights are exactly  w_U = sum_c m_{U,c} w_c / sum_c m_{U,c}  (m_{U,c} = members of U in cell c).

Scales
  labeled  cells over the labeled samples only (rows in P/LP, B/LB, population or Synonymous)
  assay    labeled cells plus unlabeled assayed variants ('assayed-only|<consequence>'), via extra_sample='all_assayed'

Composition takes membership from the calibration-time scoreset itself (built with the arm's flags), so missing
samples anywhere in the order, NU/PU modes and missense-only P/LP are handled by the same rule as the pipeline:
weights cover the non-empty calibration samples, in order (pipeline/visualize.py, sample-index detection).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STD = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]
SHORT = {"Pathogenic/Likely Pathogenic": "PLP", "Benign/Likely Benign": "BLB", "population": "gnomAD",
         "Synonymous": "Syn"}
SAI_TAG = "|SpliceAI>0.2"
ARMS = {
    # name: (PipelineConfig flags for the calibration scoreset == run_igvf_batch flags)
    "cell_all": dict(),
    "cell_mis": dict(missense_plp_only=True, population_type="gnomAD_missense_nsSNV"),
    # as cell_mis, but SpliceAI-flagged synonymous cells (splice-detecting assays only) are left out of the
    # Synonymous sample's composed weights, so splice-disrupting synonymous variants do not shape the benign
    # reference (memo D7). Calibration flags are cell_mis's.
    "cell_mis_nosplsyn": dict(missense_plp_only=True, population_type="gnomAD_missense_nsSNV"),
}
ARM_CLI = {
    "cell_all": [],
    "cell_mis": ["--missense-plp-only", "--population-type", "gnomAD_missense_nsSNV"],
    "cell_mis_nosplsyn": ["--missense-plp-only", "--population-type", "gnomAD_missense_nsSNV"],
}
# per arm: {calibration sample name: predicate on cell name -> drop that cell from the sample's composition}
ARM_EXCLUDE = {"cell_mis_nosplsyn": {"Synonymous": lambda cell: cell.endswith(SAI_TAG)}}
SAI_COLS = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]


def _release(ds):
    return "2018" if ds.endswith("_clinvar_2018") else "2025"


def load_scoreset(ds, df_ds, **flags):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    return load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                      clinvar_release=_release(ds), **flags))


def _rows(ss):
    """Per kept row: variant id, consequence, score, multi-label membership over the scoreset's samples."""
    from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids
    ids = _get_variant_ids(ss)
    csq, sai = [], []
    for all_idx, (_, variants) in enumerate(ss.get_variants_by_id().items()):
        if ss._keep_mask[all_idx]:
            row = variants[0].row
            c = row.get("simplified_consequence")
            csq.append(str(c) if c is not None and c == c else "unknown")
            v = [float(row.get(k)) for k in SAI_COLS if row.get(k) is not None]
            v = [x for x in v if np.isfinite(x)]
            sai.append(max(v) if v else np.nan)
    ss_sai = np.asarray(sai, float)
    sa = np.asarray(ss._sample_assignments, bool)
    sc = np.asarray(ss.scores, float)
    assert len(ids) == len(csq) == sa.shape[0] == len(sc), (len(ids), len(csq), sa.shape, len(sc))
    return np.asarray(ids), np.asarray(csq), sc, sa, list(ss.sample_names), ss_sai


@dataclass
class Cells:
    dataset: str
    scale: str
    scores: np.ndarray            # (n,)
    cell_of_row: np.ndarray       # (n,) int
    variant_ids: np.ndarray       # (n,)
    table: pd.DataFrame           # one row per cell: name, mask (tuple of STD membership), consequence, n
    meta: dict = field(default_factory=dict)

    @property
    def onehot(self):
        m = np.zeros((len(self.scores), len(self.table)), bool)
        m[np.arange(len(self.scores)), self.cell_of_row] = True
        return m

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, scores=self.scores, cell_of_row=self.cell_of_row, variant_ids=self.variant_ids,
                            table=self.table.to_json(), meta=json.dumps(self.meta), dataset=self.dataset,
                            scale=self.scale)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        t = pd.read_json(str(z["table"]))
        t["mask"] = t["mask"].apply(tuple)
        return cls(str(z["dataset"]), str(z["scale"]), z["scores"], z["cell_of_row"], z["variant_ids"], t,
                   json.loads(str(z["meta"])))


def build_cells(ds, df_ds, scale="labeled"):
    """Disjoint cells keyed by (membership over P/LP, B/LB, population, Synonymous; consequence)."""
    flags = dict(extra_sample="all_assayed") if scale == "assay" else {}
    ss = load_scoreset(ds, df_ds, **flags)
    ids, csq, sc, sa, names, sai = _rows(ss)
    # splice-detecting assays keep splice-affecting rows; give SpliceAI > 0.2 rows their own cells
    detects_splice = bool(getattr(ss, "detects_splice", False))
    if detects_splice:
        csq = np.array([c + SAI_TAG if (s == s and s > 0.2) else c for c, s in zip(csq, sai)])
    col = {n: names.index(n) for n in STD if n in names}
    mask = np.stack([sa[:, col[n]] if n in col else np.zeros(len(sc), bool) for n in STD], axis=1)
    keep = np.isfinite(sc)
    if scale == "labeled":
        keep &= mask.any(axis=1)
    keys = [(tuple(bool(b) for b in m), c) for m, c in zip(mask[keep], csq[keep])]
    uniq = sorted(set(keys), key=lambda k: (k[0][::-1], k[1]))
    index = {k: i for i, k in enumerate(uniq)}
    cell_of_row = np.array([index[k] for k in keys], int)
    tab = pd.DataFrame({"mask": [k[0] for k in uniq], "consequence": [k[1] for k in uniq]})
    tab["name"] = [("+".join(SHORT[n] for n, b in zip(STD, m) if b) or "assayed-only") + "|" + c
                   for m, c in zip(tab["mask"], tab["consequence"])]
    tab["n"] = np.bincount(cell_of_row, minlength=len(tab))
    return Cells(ds, scale, sc[keep], cell_of_row, ids[keep], tab,
                 meta=dict(n_rows=int(keep.sum()), n_cells=len(tab), detects_splice=detects_splice))


def fit_cells(cells, K, seeds, R=8, master_seed=0):
    """Production univariate fitter on the cell layout; one record per seed in the stored-record shape."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    ss = BasicScoreset(cells.scores, cells.onehot, sample_names=list(cells.table["name"]))
    fitter = Fit(ss)
    out = {}
    for b in seeds:
        try:
            models, best, best_val_ll = fitter.run([K], bootstrap=True, bootstrap_seed=int(b), num_fits=R,
                                                   master_seed=master_seed, check_monotonic=True, core_limit=1,
                                                   verbose=False)
        except Exception as e:  # a failed bootstrap is dropped, as in production
            out[str(b)] = {"error": repr(e)[:300]}
            continue
        fit = {k: v for k, v in models[best].items() if k not in ("history", "likelihoods")}
        fit = json.loads(json.dumps(fit, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        out[str(b)] = {f"{K}c": dict(dataset_name=cells.dataset, bootstrap_seed=int(b), num_components=K,
                                     fit_idx=int(best), fit=fit, val_ll=float(best_val_ll), n_cap_hits=0,
                                     calibrated_dims=None)}
    return out


def layout_and_membership(cells, calib_ss, exclude=None):
    """Non-empty calibration samples (in order) and, for each, its member count per cell.
    exclude: {sample name: predicate(cell name)} -> those cells are dropped from that sample's composition."""
    ids, _, sc, sa, names, _ = _rows(calib_ss)
    counts = np.asarray(calib_ss.sample_counts)
    layout = [i for i, c in enumerate(counts) if c > 0]
    pos = {v: i for i, v in enumerate(cells.variant_ids)}
    M = np.zeros((len(layout), len(cells.table)), float)
    for r, s in enumerate(layout):
        member_ids = ids[sa[:, s] & np.isfinite(sc)]
        miss = [v for v in member_ids if v not in pos]
        if miss:
            raise ValueError(f"{cells.dataset}: {len(miss)} member(s) of calibration sample {names[s]!r} have no "
                             f"cell (e.g. {miss[:3]}); build cells at a scale that contains them")
        np.add.at(M[r], cells.cell_of_row[[pos[v] for v in member_ids]], 1.0)
        for name, pred in (exclude or {}).items():
            if names[s] == name:
                M[r, [pred(n) for n in cells.table["name"]]] = 0.0
        if M[r].sum() == 0:
            raise ValueError(f"{cells.dataset}: non-empty calibration sample {names[s]!r} has no member cells")
    return [names[s] for s in layout], M


def compose(record, M):
    """Standard-layout record from a cell record: row s = sum_c M[s,c] w_c / sum_c M[s,c]."""
    rec = json.loads(json.dumps(record))
    W = np.asarray(rec["fit"]["weights"], float)                 # (n_cells, K)
    if W.shape[0] != M.shape[1]:
        raise ValueError(f"cell weights have {W.shape[0]} rows but {M.shape[1]} cells")
    rec["fit"]["weights"] = ((M @ W) / M.sum(axis=1, keepdims=True)).tolist()
    return rec


def compose_arm(cells, cell_fits, df_ds, arm):
    """{seed: {Kc: composed record}} for one arm, plus the layout and its membership matrix.
    Raises ValueError('skip: ...') when the arm cannot be calibrated (e.g. empty population)."""
    calib_ss = load_scoreset(cells.dataset, df_ds, **ARMS[arm])
    if arm in ARM_EXCLUDE and not cells.meta.get("detects_splice"):
        raise ValueError(f"skip: {arm}: assay does not detect splicing (SpliceAI-flagged rows already removed)")
    names, M = layout_and_membership(cells, calib_ss, ARM_EXCLUDE.get(arm))
    if "population" not in names and "gnomAD" not in names:
        raise ValueError(f"skip: {arm}: empty population sample")
    out = {}
    for seed, rec in cell_fits.items():
        if not rec or "error" in rec:
            continue
        out[seed] = {k: compose(v, M) for k, v in rec.items()}
    return out, names, M
