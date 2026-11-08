"""C1/C2 light membership-cell fits (local gauge; treat with a grain of salt), compose, write fits files.

For each dataset and K (config n_c, plus 3c when n_c = 2): build cells, fit B bootstraps x R restarts with the
production fitter on the cell layout, save the raw cell fits, then compose each arm (cells.ARMS) into the standard
layout of that arm's calibration scoreset and write fits_<arm>_<K>c.json.gz for run_igvf_batch --precomputed-fits.

Usage:
  python -m analysis.membership_cells.run_light [--scale labeled|assay] [--B 20] [--R 8] [--datasets ...] [--n-jobs 60]
Calibrate afterwards with the printed commands (or the scratchpad launcher).
"""
from __future__ import annotations

import argparse
import gzip
import json
import traceback
from pathlib import Path

ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations/cells")


def _cfg_nc(cfg, ds):
    e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", ""))
    if e is None:
        return None
    return int((e["n_c"] if isinstance(e, dict) else e[0])[0])


def build(ds, df_ds, scale, out):
    from analysis.membership_cells.cells import build_cells
    f = out / "cells" / f"{ds}.npz"
    if not f.exists():
        build_cells(ds, df_ds, scale).save(f)
    return str(f)


def task(ds, K, seeds, R, out):
    """One (dataset, K, seed block): fit and save raw cell fits (cells built beforehand)."""
    from analysis.membership_cells.cells import Cells, fit_cells
    f = out / "raw" / f"{ds}__{K}c__{seeds[0]}-{seeds[-1]}.json.gz"
    if f.exists():
        return str(f)
    try:
        cells = Cells.load(out / "cells" / f"{ds}.npz")
        fits = fit_cells(cells, K, seeds, R=R)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        json.dump(fits, gzip.open(tmp, "wt"))
        tmp.rename(f)
        return str(f)
    except Exception as e:
        return f"FAILED {ds} {K}c: {e!r}\n{traceback.format_exc()[-800:]}"


def compose_all(out, slices):
    """Merge raw blocks per (dataset, K), compose every arm, write fits_<arm>_<K>c.json.gz; report skips."""
    from analysis.membership_cells.cells import Cells, ARMS, compose_arm
    merged = {}
    for f in sorted((out / "raw").glob("*.json.gz")):
        ds, kc, _ = f.name[:-len(".json.gz")].split("__")
        merged.setdefault((ds, kc), {}).update(json.load(gzip.open(f)))
    files, report = {}, []
    for (ds, kc), cf in sorted(merged.items()):
        cells = Cells.load(out / "cells" / f"{ds}.npz")
        n_ok = sum(1 for v in cf.values() if v and "error" not in v)
        for arm in ARMS:
            try:
                fits, names, M = compose_arm(cells, cf, slices[ds], arm)
            except ValueError as e:
                report.append(dict(dataset=ds, K=kc, arm=arm, status=str(e)))
                continue
            files.setdefault((arm, kc), {})[ds] = fits
            report.append(dict(dataset=ds, K=kc, arm=arm, status="ok", n_boot=n_ok, layout="|".join(names),
                               members="|".join(str(int(x)) for x in M.sum(1))))
    for (arm, kc), d in files.items():
        json.dump(d, gzip.open(out / f"fits_{arm}_{kc}.json.gz", "wt"))
    import pandas as pd
    rep = pd.DataFrame(report)
    rep.to_csv(out / "compose_report.csv", index=False)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="labeled", choices=["labeled", "assay"])
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--block", type=int, default=5, help="bootstraps per task")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="src/igvf_configs/dataset_configs_aug_2026.json")
    ap.add_argument("--compose-only", action="store_true")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.plot_stability_datasets import PRIORITY
    out = Path(args.out or ROOT / f"light_{'c1' if args.scale == 'labeled' else 'c2'}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(args.config))
    dss = list(args.datasets or PRIORITY)
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in dss}
    del df_full
    if not args.compose_only:
        tasks = []
        for ds in dss:
            nc = _cfg_nc(cfg, ds)
            if nc is None:
                continue
            for K in sorted({min(nc, 3), 3}):
                for s0 in range(0, args.B, args.block):
                    tasks.append((ds, K, list(range(s0, min(s0 + args.block, args.B)))))
        Parallel(n_jobs=min(args.n_jobs, len(dss)))(delayed(build)(ds, slices[ds], args.scale, out) for ds in dss)
        print(f"{len(tasks)} tasks")
        res = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(task)(ds, K, seeds, args.R, out) for ds, K, seeds in tasks)
        for r in res:
            if r.startswith("FAILED"):
                print(r)
    rep = compose_all(out, slices)
    print(rep.groupby(["arm", "status"]).size().to_string())
    print(rep[rep.status != "ok"].to_string(index=False))


if __name__ == "__main__":
    main()
