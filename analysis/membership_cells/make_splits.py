"""OOB splits (val_variant_indices per bootstrap seed) for any calibration scoreset, by variant id.

run_igvf_batch --oob needs, per dataset and seed, the positions (in the CALIBRATION-time scoreset) of the variants
held out of that bootstrap's fit. Two sources:
  cells     the membership-cell fit's own bootstrap: makeOneHot is the identity on the one-hot cell layout and
            sample_specific_bootstrap(onehot, seed) gives the held-out rows (exactly Fit.run with master_seed 0).
            Cells of 1 row are always in training, so their variants never get OOB evidence.
  remap     an existing splits pickle built on another scoreset (e.g. default population) -> remapped by variant id
            onto the target scoreset (e.g. gnomAD-missense population). Held-out variants absent from the target are
            dropped; target variants never held out get no OOB evidence.
Output: {dataset: {seed: {"val_variant_indices": int array}}}  (the index-based format the pipeline reads).

Usage:
  python -m analysis.membership_cells.make_splits cells <cells_dir> <arm> <out.pkl> [--datasets ...]
  python -m analysis.membership_cells.make_splits remap <splits.pkl> <out.pkl> --source-flags '{}' \
      --target-flags '{"population_type": "gnomAD_missense_nsSNV", "missense_plp_only": true}' [--seeds 100]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def _calib_ids(ds, df_ds, flags):
    from analysis.membership_cells.cells import load_scoreset, _rows
    ids = _rows(load_scoreset(ds, df_ds, **flags))[0]
    return ids, {v: i for i, v in enumerate(ids)}


def cell_splits(ds, df_ds, cells_dir, arm):
    from analysis.membership_cells.cells import Cells, ARMS
    from src.assay_calibration.fit_utils.fit import sample_specific_bootstrap
    cells_dir = Path(cells_dir)
    cells = Cells.load(cells_dir / "cells" / f"{ds}.npz")
    seeds = set()
    for f in (cells_dir / "raw").glob(f"{ds}__*.json.gz"):
        seeds |= {int(k) for k, v in json.load(gzip.open(f)).items() if v and "error" not in v}
    _, pos = _calib_ids(ds, df_ds, ARMS[arm])
    onehot = cells.onehot
    out = {}
    for b in sorted(seeds):
        _, va = sample_specific_bootstrap(onehot, b)
        idx = [pos[v] for v in cells.variant_ids[va] if v in pos]
        out[b] = {"val_variant_indices": np.asarray(sorted(set(idx)), int)}
    return out


def remap_splits(ds, df_ds, splits_ds, source_flags, target_flags, seeds=None):
    src_ids, _ = _calib_ids(ds, df_ds, source_flags)
    _, tgt_pos = _calib_ids(ds, df_ds, target_flags)
    out = {}
    for b, sp in splits_ds.items():
        if seeds is not None and int(b) >= seeds:
            continue
        v = np.asarray(sp["val_variant_indices"], int)
        idx = [tgt_pos[src_ids[i]] for i in v if i < len(src_ids) and src_ids[i] in tgt_pos]
        out[int(b)] = {"val_variant_indices": np.asarray(sorted(set(idx)), int)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells"); c.add_argument("cells_dir"); c.add_argument("arm"); c.add_argument("out")
    c.add_argument("--datasets", nargs="*", default=None)
    r = sub.add_parser("remap"); r.add_argument("splits"); r.add_argument("out")
    r.add_argument("--source-flags", default="{}"); r.add_argument("--target-flags", default="{}")
    r.add_argument("--seeds", type=int, default=None); r.add_argument("--datasets", nargs="*", default=None)
    for p in (c, r):
        p.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    if a.cmd == "cells":
        dss = a.datasets or sorted(p.stem for p in (Path(a.cells_dir) / "cells").glob("*.npz"))
    else:
        src = pickle.load(open(a.splits, "rb"))
        dss = a.datasets or sorted(src)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in dss}
    del df_full

    def one(ds):
        try:
            if a.cmd == "cells":
                return ds, cell_splits(ds, slices[ds], a.cells_dir, a.arm)
            return ds, remap_splits(ds, slices[ds], src[ds], json.loads(a.source_flags), json.loads(a.target_flags),
                                    a.seeds)
        except Exception as e:
            print(f"FAILED {ds}: {e!r}")
            return ds, None
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(ds) for ds in dss)
    out = {ds: s for ds, s in res if s}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(a.out, "wb"))
    print(f"{len(out)} datasets -> {a.out}")


if __name__ == "__main__":
    main()
