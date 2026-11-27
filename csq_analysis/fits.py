"""Strata-model bootstrap fits for the consequence paper: task bundles (make) and workers (run).

One task = one (fit configuration, dataset, block of bootstrap seeds). Fits use the production univariate fitter
(Fit.run, check_monotonic, master_seed 0, the dataset's production K) on the strata one-hot of csq_analysis.strata_uni.
Each task writes fits/<config>/<dataset>__<s0>-<s1>.json.gz = {seed: {"<K>c": record}} (production record shape, so
composition and calibration reuse production code), and the stratum layout once per (config, dataset) under strata/.

Fit configurations (plan F1-F4, F7):
    S1_m5     stratum bootstrap, thin strata (< 5) merged into a same-role sibling (default)
    S1_m1     stratum bootstrap, no merging
    S1_m10    stratum bootstrap, merge below 10
    S0_m5     composite (ExCALIBR-equivalent) resampling of the pooled composites
    S2n{3,5,10}_m1   no fit merging; strata below n resampled jointly with their same-role sibling
Restart budget is per bundle (--R); the restart pilot uses a dataset panel (--datasets).

    python -m csq_analysis.fits make <out> --configs S1_m5,S0_m5 --B 100 --R 20 [--datasets a,b] [--per-task 10]
    python -m csq_analysis.fits merge <out> <bundle_a> <bundle_b> ...     (one job dir for several bundles)
    python -m csq_analysis.fits run <out> --index 0 --per-index 16 --workers 16
Outputs are keyed by label = <config>_R<restarts>, so restart variants of one config never collide.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np

FIT_CONFIGS = {
    "S1_m5": dict(min_stratum=5),
    "S1_m1": dict(min_stratum=1),
    "S1_m10": dict(min_stratum=10),
    "S0_m5": dict(min_stratum=5, scheme="composite"),
    "S2n3_m1": dict(min_stratum=1, scheme="groups", n_min=3),
    "S2n5_m1": dict(min_stratum=1, scheme="groups", n_min=5),
    "S2n10_m1": dict(min_stratum=1, scheme="groups", n_min=10),
}
ROWS_FILE = "rows.pkl"


def label(cfg, R, sub=None):
    """Output name of a (fit config, restart count[, missense subsample]) task, e.g. S1_m5_R20 or S1_m5_R8__n10r2."""
    from csq_analysis.downsample import sub_tag
    return f"{cfg}_R{int(R)}" + (f"__{sub_tag(sub)}" if sub else "")


def config_of(lab):
    """Fit configuration name of an output label (labels without an _R suffix are bare config names)."""
    base, _, r = lab.split("__")[0].rpartition("_R")
    return base if base in FIT_CONFIGS and r.isdigit() else lab


def sub_of(lab):
    """Missense subsample {"n", "rep"} of an output label, or None."""
    from csq_analysis.downsample import parse_tag
    parts = lab.split("__")
    return parse_tag(parts[1]) if len(parts) > 1 else None


def _rows(out_dir, ds, sub=None):
    import pandas as pd
    r = pd.read_pickle(Path(out_dir) / ROWS_FILE)
    r = r[r.dataset == ds]
    if sub:
        from csq_analysis.downsample import subsample
        r = subsample(r, ds, sub["n"], sub["rep"])
    return r


def layout(out_dir, ds, cfg, sub=None):
    """Strata of `ds` under fit config `cfg` (missense controls subsampled per `sub`), plus resampling kwargs."""
    return layout_rows(ds, _rows(out_dir, ds, sub), cfg)


def layout_rows(ds, rows, cfg):
    """layout() on rows already in memory (csq_analysis.annotate format; also csq_analysis.sim replicates)."""
    from csq_analysis import strata_uni as U
    us = U.build(ds, rows, min_stratum=cfg["min_stratum"])
    kw = {}
    scheme = cfg.get("scheme")
    fitted = us.fitted
    if scheme == "composite":
        M = us.composition("pool")
        inc = M[:, us.stratum[fitted]] > 0                     # (4, n_fitted): row in composite r
        kw = dict(uni_bootstrap_scheme="composite", bootstrap_assignments=inc.T)
    elif scheme == "groups":
        g = U.build(ds, rows, min_stratum=cfg["n_min"])        # same rows/order; merging only for resampling
        kw = dict(uni_bootstrap_scheme="groups", bootstrap_groups=g.stratum[fitted])
    return us, kw


def _strata_meta(us):
    return dict(dataset=us.dataset, names=us.names, counts=us.counts.tolist(), merges=us.merges,
                fitted_variant_ids=us.rows.variant_id[us.fitted].tolist())


def make(out_dir, configs, B, R, datasets=None, per_task=10, rows_src=None, subs=None):
    from csq_analysis import config as C
    import shutil
    out = Path(out_dir)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    src = Path(rows_src) if rows_src else C.ANNOT / "rows.pkl"
    shutil.copy(src, out / ROWS_FILE)
    dss = datasets or C.canonical_datasets()
    tasks = []
    for cfg in configs:
        if cfg not in FIT_CONFIGS:
            raise ValueError(f"unknown fit config {cfg}")
        for ds in dss:
            K, _ = C.dataset_k(ds)
            for sub in (subs or [None]):
                for s0 in range(0, B, per_task):
                    t = dict(config=cfg, label=label(cfg, R, sub), dataset=ds, K=K,
                             seeds=list(range(s0, min(B, s0 + per_task))), R=R)
                    if sub:
                        t["sub"] = sub
                    tasks.append(t)
    (out / "tasks.json").write_text(json.dumps(tasks))
    (out / "bundle.json").write_text(json.dumps(dict(configs=configs, B=B, R=R, datasets=dss, per_task=per_task,
                                                     fit_configs={c: FIT_CONFIGS[c] for c in configs}, subs=subs)))
    print(f"{len(tasks)} tasks ({len(configs)} configs x {len(dss)} datasets x {len(subs or [None])} subsamples x "
          f"{-(-B // per_task)} seed blocks, R={R}) "
          f"-> {out / 'tasks.json'}")


def _run_one(out_dir, t):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    out = Path(out_dir)
    ds, cfgname, K, seeds = t["dataset"], t["config"], int(t["K"]), t["seeds"]
    lab = t.get("label", cfgname)
    f = out / "fits" / lab / f"{ds}__{seeds[0]}-{seeds[-1]}.json.gz"
    if f.exists():
        return "skip"
    us, kw = layout(out, ds, FIT_CONFIGS[cfgname], t.get("sub"))
    meta = out / "strata" / lab / f"{ds}.json"
    if not meta.exists():
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(_strata_meta(us)))
    rec = fit_records(us, kw, K, seeds, int(t["R"]), ds, cfgname)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(rec, fh)
    tmp.rename(f)
    return "ok"


def fit_records(us, kw, K, seeds, R, ds, cfgname):
    """{seed: {"<K>c": record}} of one strata layout: one best-of-R fit per bootstrap seed (failed bootstraps kept as
    {"error"}), with each bootstrap's out-of-bag rows (positions among the fitted rows)."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    ss = BasicScoreset(us.rows.score.to_numpy(float)[us.fitted], us.onehot, sample_names=list(us.names))
    fitter = Fit(ss)
    fit_kw = {k: v for k, v in kw.items() if k != "force_gaussian"}
    rec = {}
    for b in seeds:
        try:
            models, best, val_ll = fitter.run([K], bootstrap=True, bootstrap_seed=int(b), num_fits=int(R),
                                              master_seed=0, check_monotonic=True, core_limit=1, verbose=False, **kw)
        except Exception as e:      # a failed bootstrap is dropped, as in production
            rec[str(b)] = {"error": repr(e)[:300]}
            continue
        if best is None:
            rec[str(b)] = {"error": "no valid restart"}
            continue
        fit = {k: v for k, v in models[best].items() if k not in ("history", "likelihoods")}
        fit = json.loads(json.dumps(fit, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        # out-of-bag rows of this bootstrap (positions among the FITTED rows, i.e. strata.json fitted_variant_ids):
        # Fit.run keeps them only in its internal job dicts, so regenerate the (deterministic) split once
        job = fitter.generate_fit_jobs([K], bootstrap=True, bootstrap_seed=int(b), num_fits=1, master_seed=0,
                                       check_monotonic=True, **fit_kw)[0]
        val = job.get("val_variant_indices")
        rec[str(b)] = {f"{K}c": dict(dataset_name=ds, bootstrap_seed=int(b), num_components=K, fit_idx=int(best),
                                     fit=fit, val_ll=float(val_ll), strata_config=cfgname, restarts=int(R),
                                     val_variant_indices=(np.asarray(val, int).tolist() if val is not None else []))}
    return rec


def run(out_dir, index, per_index, workers):
    from joblib import Parallel, delayed
    tasks = json.loads((Path(out_dir) / "tasks.json").read_text())
    mine = tasks[index * per_index:(index + 1) * per_index]
    Parallel(n_jobs=workers or 1, verbose=5)(delayed(_run_one)(out_dir, t) for t in mine)


def regenerate_val(us, kw, K, seed):
    """OOB rows (positions among fitted rows) of one bootstrap seed, regenerated exactly as _run_one stores them;
    for fit records made before val_variant_indices were stored."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    ss = BasicScoreset(us.rows.score.to_numpy(float)[us.fitted], us.onehot, sample_names=list(us.names))
    job = Fit(ss).generate_fit_jobs([int(K)], bootstrap=True, bootstrap_seed=int(seed), num_fits=1, master_seed=0,
                                    check_monotonic=True, **kw)[0]
    return np.asarray(job.get("val_variant_indices"), int).tolist()


def load_fits(out_dir, lab, ds):
    """{seed: record} of every finished block of (output label, dataset), plus the strata metadata."""
    out = Path(out_dir)
    recs = {}
    for f in sorted((out / "fits" / lab).glob(f"{ds}__*.json.gz")):
        with gzip.open(f, "rt") as fh:
            recs.update(json.load(fh))
    meta = json.loads((out / "strata" / lab / f"{ds}.json").read_text())
    return recs, meta


def merge(out_dir, sources):
    """One bundle from several made bundles (task order kept, source by source). Every task carries its own label
    and restart count, so bundles with different R or datasets share one fits/ tree without collisions."""
    import shutil
    out = Path(out_dir)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    tasks, groups, dss = [], [], []
    for src in sources:
        src = Path(src)
        t = json.loads((src / "tasks.json").read_text())
        for x in t:
            x.setdefault("label", label(x["config"], x["R"], x.get("sub")))
        tasks += t
        b = json.loads((src / "bundle.json").read_text())
        groups.append(dict(source=src.name, configs=b["configs"], B=b["B"], R=b["R"], datasets=b["datasets"],
                           n_tasks=len(t), labels=sorted({x["label"] for x in t})))
        dss += [d for d in b["datasets"] if d not in dss]
        if not (out / ROWS_FILE).exists():
            shutil.copy(src / ROWS_FILE, out / ROWS_FILE)
    labels = [x["label"] for x in tasks]
    keys = [(x["label"], x["dataset"], x["seeds"][0]) for x in tasks]
    assert len(keys) == len(set(keys)), "duplicate (label, dataset, seed block) across sources"
    (out / "tasks.json").write_text(json.dumps(tasks))
    (out / "bundle.json").write_text(json.dumps(dict(groups=groups, datasets=dss, labels=sorted(set(labels)),
                                                     fit_configs=FIT_CONFIGS)))
    print(f"{len(tasks)} tasks from {len(sources)} bundles -> {out}")
    for g in groups:
        print(f"  {g['source']}: {g['n_tasks']} tasks, labels {g['labels']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--configs", default="S1_m5")
    m.add_argument("--B", type=int, default=100)
    m.add_argument("--R", type=int, default=20)
    m.add_argument("--datasets", default=None)
    m.add_argument("--per-task", type=int, default=10)
    g = sub.add_parser("merge")
    g.add_argument("out_dir")
    g.add_argument("sources", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--per-index", type=int, required=True)
    r.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out_dir, a.configs.split(","), a.B, a.R, a.datasets.split(",") if a.datasets else None, a.per_task)
    elif a.cmd == "merge":
        merge(a.out_dir, a.sources)
    else:
        run(a.out_dir, a.index, a.per_index, a.workers)


if __name__ == "__main__":
    main()
