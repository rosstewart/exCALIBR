"""One-off: manual vs. automatic model selection (reviewer comments 33-34).

NOT part of the manuscript pipeline (analyze_pipeline_output.py does not read
anything this writes). Answers: how much does manual model configuration
change performance relative to a fully automatic configuration?

  manual : the current src/igvf_configs/dataset_configs_aug_2026.json entry
           for each dataset, every setting included (n_c, benign_method,
           liberal_monotonicity, ...).
  auto   : {"n_c": f"{conservative_k}c", "benign_method": "avg"}, no
           overrides (liberal_monotonicity / postprocess at pipeline
           defaults). conservative_k is the held-out-LL rule of
           src/assay_calibration/pipeline/model_selection.bootstrap_paired_test
           (k=3 only if the 5th percentile of paired val-LL(3c) - val-LL(2c)
           over bootstraps is > 0). "avg" falls back to benign/synonymous by
           itself when one of those samples is missing (pipeline/visualize.py).

Both arms are calibrated from the SAME light refits (hpc/prepare.py
--preset medium: 100 bootstraps x 8 restarts, both 2c and 3c; plus 4c for
the one dataset whose manual config is 4c), so the comparison isolates the
selection itself from fit depth. Bootstrap seeds 0..99 under the default
master seed reproduce production's bootstrap composition for those seeds, so
the production splits pickle serves for OOB evidence.

Only datasets whose configuration differs between arms are recalibrated;
for the all-dataset pool, unchanged datasets contribute their production
matrix to both arms.

Usage (source activate excalibr; everything under nice -n 19):
  # 0. fits: jobs already prepared under ROOT/light_fits_{2c3c,234c}
  #    (submit with hpc/submit_array.sh or hpc/run_local_array.sh)
  python analysis/oneoff_auto_selection_ablation.py merge
  python analysis/oneoff_auto_selection_ablation.py configs
  python analysis/oneoff_auto_selection_ablation.py run --dry-run
  python analysis/oneoff_auto_selection_ablation.py run --arm manual auto
  python analysis/oneoff_auto_selection_ablation.py compare
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import config  # noqa: E402

ROOT = Path("/data/ross/assay_calibration/auto_selection")
LIGHT_FIT_DIRS = [ROOT / "light_fits_2c3c", ROOT / "light_fits_234c"]
LIGHT_FITS = ROOT / "light_bootstrap_results.json.gz"
MANUAL_CFG_SRC = Path(config.DATASET_CONFIGS)
CFG = {"manual": ROOT / "cfg_manual.json", "auto": ROOT / "cfg_auto.json"}
SELECTION_CSV = ROOT / "selection_vs_manual.csv"
CHANGED_JSON = ROOT / "changed_datasets.json"
ARM_OUT = {"manual": ROOT / "manual", "auto": ROOT / "auto"}


def _entry_parts(entry):
    """(n_c, benign_method, overrides) of a dataset-config entry (dict or list form)."""
    if isinstance(entry, dict):
        return (str(entry["n_c"]), entry.get("benign_method", "avg"),
                {k: v for k, v in entry.items() if k not in ("n_c", "benign_method")})
    over = entry[2] if len(entry) > 2 and isinstance(entry[2], dict) else {}
    return str(entry[0]), (entry[1] if len(entry) > 1 else "avg"), over


# ---------------------------------------------------------------------------
def cmd_merge(args):
    merged = {}
    for d in LIGHT_FIT_DIRS:
        agg = d / "bootstrap_results.json.gz"
        if not agg.exists() or args.reaggregate:
            subprocess.run(["nice", "-n", "19", sys.executable, str(_REPO_ROOT / "hpc" / "aggregate_results.py"),
                            str(d), str(agg)], check=True)
        with gzip.open(agg, "rt") as f:
            part = json.load(f)
        overlap = set(part) & set(merged)
        if overlap:
            raise SystemExit(f"datasets in more than one light-fit dir: {sorted(overlap)}")
        merged.update(part)
    with gzip.open(LIGHT_FITS, "wt") as f:
        json.dump(merged, f)
    print(f"wrote {LIGHT_FITS}: {len(merged)} datasets")
    for ds, boots in sorted(merged.items()):
        comps = {}
        for b in boots.values():
            for k in b:
                comps[k] = comps.get(k, 0) + 1
        print(f"  {ds}: {len(boots)} bootstraps, per-component {comps}")


# ---------------------------------------------------------------------------
def cmd_configs(args):
    from src.assay_calibration.pipeline.model_selection import bootstrap_paired_test

    with gzip.open(LIGHT_FITS, "rt") as f:
        light = json.load(f)
    manual_all = json.loads(MANUAL_CFG_SRC.read_text())
    datasets = json.loads((ROOT / "datasets_all.json").read_text())

    manual_cfg, auto_cfg, rows = {}, {}, []
    for ds in datasets:
        if ds not in light:
            print(f"  WARNING: no light fits for {ds}; skipped")
            continue
        m_nc, m_benign, m_over = _entry_parts(manual_all[ds])
        t = bootstrap_paired_test(light[ds], k_range=[2, 3], verbose=False)
        a_nc = f"{t['conservative_k']}c"
        manual_cfg[ds] = manual_all[ds]
        auto_cfg[ds] = {"n_c": a_nc, "benign_method": "avg"}
        rows.append({
            "dataset": ds, "manual_n_c": m_nc, "manual_benign_method": m_benign,
            "manual_overrides": json.dumps(m_over) if m_over else "",
            "auto_n_c": a_nc, "wilcoxon_selected_k": t["selected_k"], "wilcoxon_p": t["p_value"],
            "valll_diff_mean": t["mean_diff"], "valll_diff_median": t["median_diff"],
            "valll_diff_p5": t["fifth_percentile"], "n_paired_bootstraps": t["n_samples"],
            "n_c_differs": m_nc != a_nc, "benign_method_differs": m_benign != "avg",
            "overrides_differ": bool(m_over),
        })
    sel = pd.DataFrame(rows)
    sel["any_differs"] = sel[["n_c_differs", "benign_method_differs", "overrides_differ"]].any(axis=1)

    CFG["manual"].write_text(json.dumps(manual_cfg, indent=2, ensure_ascii=False))
    CFG["auto"].write_text(json.dumps(auto_cfg, indent=2, ensure_ascii=False))
    sel.to_csv(SELECTION_CSV, index=False)
    changed = sel.loc[sel["any_differs"], "dataset"].tolist()
    CHANGED_JSON.write_text(json.dumps(changed, indent=1, ensure_ascii=False))

    print(f"{len(sel)} datasets; differ from manual: n_c {int(sel.n_c_differs.sum())}, "
          f"benign_method {int(sel.benign_method_differs.sum())}, overrides {int(sel.overrides_differ.sum())}, "
          f"any {len(changed)}")
    print(pd.crosstab(sel["manual_n_c"], sel["auto_n_c"], rmargins=True))
    print(f"wrote {CFG['manual']}, {CFG['auto']}, {SELECTION_CSV}, {CHANGED_JSON}")


# ---------------------------------------------------------------------------
def _arm_command(arm, datasets, n_jobs_inner):
    return [
        "nice", "-n", "19", sys.executable, str(_REPO_ROOT / "run_igvf_batch.py"),
        "--dataset", str(config.DATASET_TSV),
        "--precomputed-fits", str(LIGHT_FITS),
        "--dataset-configs", str(CFG[arm]),
        "--output-dir", str(ARM_OUT[arm]),
        "--clinvar-release", "2025",
        "--oob", "--splits-file", str(config.SPLITS_FILE),
        "--n-jobs-inner", str(n_jobs_inner),
        "--skip-existing",
        "--datasets", *datasets,
    ]


def cmd_run(args):
    changed = json.loads(CHANGED_JSON.read_text())
    if args.datasets:
        changed = [d for d in changed if d in set(args.datasets)]
    procs = []
    for arm in args.arm:
        cmd = _arm_command(arm, changed, args.n_jobs_inner)
        print(f"[{arm}] {len(changed)} datasets\n  " + " ".join(cmd[:18]) + " ...")
        if args.dry_run:
            continue
        ARM_OUT[arm].mkdir(parents=True, exist_ok=True)
        log = open(ROOT / f"run_{arm}.log", "a")
        env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1"}
        procs.append((arm, subprocess.Popen(cmd, cwd=str(_REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)))
    for arm, p in procs:
        rc = p.wait()
        print(f"[{arm}] exit {rc} (log: {ROOT / f'run_{arm}.log'})")


# ---------------------------------------------------------------------------
def _per_dataset_matrices(output_dir, dataset_configs, datasets, label):
    """{dataset: confusion df or None} for one ExCALIBR-shaped output tree (OOB)."""
    from analysis.confusion import build_confusion_matrix
    from analysis.discovery import discover_outputs, load_all_variants

    tree, ms, cal = discover_outputs(Path(output_dir))
    df = load_all_variants(tree=tree, model_selections=ms, dataset_configs=dataset_configs,
                           methods_filter=None, datasets_filter=list(datasets), calibrations=cal,
                           min_controls=0)
    out = {}
    for ds in datasets:
        sub = df[df["dataset"] == ds] if not df.empty else df
        if not sub.empty and sub["method"].nunique() > 1:
            sub = sub[sub["method"] == sorted(sub["method"].unique())[0]]
        out[ds] = build_confusion_matrix(sub, use_oob=True, label=f"{ds}/{label}") if not sub.empty else None
    return out, df


def _pooled_row(name, mats, scope):
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics
    mats = [m for m in mats if m is not None]
    agg = sum(mats[1:], mats[0].copy())
    met = compute_classification_metrics(agg)
    return {"scope": scope, "arm": name, "n_datasets": len(mats),
            **{k: met[k] for k in ("mcc", "accuracy", "coverage", "sensitivity", "specificity",
                                   "TP", "TN", "FP", "FN", "uncertain", "total")}}


def _paired(a_mats, b_mats, metric):
    from scipy.stats import wilcoxon
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics
    a, b = [], []
    for x, y in zip(a_mats, b_mats):
        if x is None or y is None:
            continue
        mx, my = compute_classification_metrics(x)[metric], compute_classification_metrics(y)[metric]
        if np.isfinite(mx) and np.isfinite(my):
            a.append(mx); b.append(my)
    a, b = np.array(a), np.array(b)
    d = b - a
    p = wilcoxon(d).pvalue if len(d) and np.any(d != 0) else np.nan
    return {"metric": metric, "n": len(d), "median_a": np.median(a) if len(a) else np.nan,
            "median_b": np.median(b) if len(b) else np.nan, "median_diff": np.median(d) if len(d) else np.nan,
            "n_b_better": int((d > 0).sum()), "n_b_worse": int((d < 0).sum()), "p_wilcoxon": p}


def cmd_compare(args):
    from analysis.plot_common import display_name

    sel = pd.read_csv(SELECTION_CSV)
    changed = json.loads(CHANGED_JSON.read_text())
    prod_cfg = json.loads(MANUAL_CFG_SRC.read_text())
    man_cfg = json.loads(CFG["manual"].read_text())
    auto_cfg = json.loads(CFG["auto"].read_text())

    prod, prod_df = _per_dataset_matrices(config.OUTPUT_DIR, prod_cfg, sel["dataset"].tolist(), "production")
    datasets = sorted(d for d in prod if prod[d] is not None)
    changed = [d for d in changed if d in datasets]
    man, _ = _per_dataset_matrices(ARM_OUT["manual"], man_cfg, changed, "manual_light")
    auto, _ = _per_dataset_matrices(ARM_OUT["auto"], auto_cfg, changed, "auto_light")
    missing = [d for d in changed if man[d] is None or auto[d] is None]
    if missing:
        print(f"  WARNING: {len(missing)} changed datasets missing a light-arm output, dropped: {missing}")
    changed = [d for d in changed if d not in missing]
    identical = [d for d in changed if man[d].equals(auto[d])]

    rows = []
    # Changed datasets only: the paired light-fit comparison (plus production for reference).
    rows.append(_pooled_row("production (manual, full fits)", [prod[d] for d in changed], "changed"))
    rows.append(_pooled_row("manual (light fits)", [man[d] for d in changed], "changed"))
    rows.append(_pooled_row("automatic (light fits)", [auto[d] for d in changed], "changed"))
    # All datasets: unchanged datasets take their production matrix in both arms.
    man_all = [man[d] if d in changed else prod[d] for d in datasets]
    auto_all = [auto[d] if d in changed else prod[d] for d in datasets]
    rows.append(_pooled_row("production (manual, full fits)", [prod[d] for d in datasets], "all"))
    rows.append(_pooled_row("manual", man_all, "all"))
    rows.append(_pooled_row("automatic", auto_all, "all"))

    # Existing ablations (already in tab:ablation_baseline_performance), recomputed here
    # against production on matched datasets so all four sit side by side.
    for name, out_dir in (("skew-locked", config.SKEW_LOCKED_OUTPUT_DIR),
                          (f"prior fixed 0.1", config.MANUAL_PRIOR_OUTPUT_DIR)):
        if not out_dir or not Path(out_dir).exists():
            print(f"  SKIP {name}: {out_dir} missing")
            continue
        other, _ = _per_dataset_matrices(out_dir, prod_cfg, datasets, name)
        matched = [d for d in datasets if other[d] is not None]
        rows.append(_pooled_row("production (manual, full fits)", [prod[d] for d in matched], f"{name} matched"))
        rows.append(_pooled_row(name, [other[d] for d in matched], f"{name} matched"))

    pooled = pd.DataFrame(rows)
    paired = pd.DataFrame([
        {"comparison": "auto vs manual (light, changed datasets)", **_paired([man[d] for d in changed],
                                                                             [auto[d] for d in changed], m)}
        for m in ("mcc", "accuracy", "coverage")
    ] + [
        {"comparison": "manual light vs production (changed datasets; fit-depth check)",
         **_paired([prod[d] for d in changed], [man[d] for d in changed], m)}
        for m in ("mcc", "accuracy", "coverage")
    ])

    from src.assay_calibration.plot_utils.utils import compute_classification_metrics
    per_ds = []
    for d in changed:
        r = sel.set_index("dataset").loc[d].to_dict()
        row = {"dataset": display_name(d), "manual_config": f"{r['manual_n_c']}/{r['manual_benign_method']}"
               + (f" {r['manual_overrides']}" if isinstance(r["manual_overrides"], str) and r["manual_overrides"] else ""),
               "auto_config": f"{r['auto_n_c']}/avg", "outputs_identical": d in identical}
        for tag, mats in (("prod", prod), ("manual", man), ("auto", auto)):
            met = compute_classification_metrics(mats[d])
            for k in ("mcc", "accuracy", "coverage", "FP", "FN"):
                row[f"{tag}_{k}"] = met[k]
        per_ds.append(row)
    per_ds = pd.DataFrame(per_ds)

    sel_out = sel.copy()
    sel_out["dataset"] = sel_out["dataset"].map(display_name)
    sel_out.to_csv(ROOT / "selection_vs_manual_display.csv", index=False)
    pooled.to_csv(ROOT / "pooled_metrics.csv", index=False)
    paired.to_csv(ROOT / "paired_tests.csv", index=False)
    per_ds.to_csv(ROOT / "per_dataset_metrics.csv", index=False)

    n = len(sel)
    md = [
        "# Manual vs. automatic model selection (one-off)", "",
        f"- Datasets: {n}; automatic n_c differs from manual for {int(sel.n_c_differs.sum())}; "
        f"manual benign_method is not avg for {int(sel.benign_method_differs.sum())}; "
        f"manual has non-default overrides for {int(sel.overrides_differ.sum())}; "
        f"any configuration difference: {int(sel.any_differs.sum())}.",
        f"- Of the {len(changed)} recalibrated datasets, {len(identical)} gave identical OOB confusion "
        f"matrices under both configurations (e.g. avg falls back to benign with no Synonymous sample).",
        "", "## Pooled OOB metrics", "", "```\n" + pooled.to_string(index=False, float_format="%.4f") + "\n```",
        "", "## Paired per-dataset tests (Wilcoxon signed-rank; b - a)", "",
        "```\n" + paired.to_string(index=False, float_format="%.4f") + "\n```",
        "", "## Per-dataset (recalibrated datasets)", "", "```\n" + per_ds.to_string(index=False, float_format="%.3f") + "\n```",
    ]
    (ROOT / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:8]))
    print(pooled.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"wrote {ROOT}/summary.md, pooled_metrics.csv, paired_tests.csv, per_dataset_metrics.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("merge", help="aggregate light fits and merge into one file")
    pm.add_argument("--reaggregate", action="store_true")
    pm.set_defaults(func=cmd_merge)
    sub.add_parser("configs", help="held-out-LL selection + manual/auto config JSONs").set_defaults(func=cmd_configs)
    pr = sub.add_parser("run", help="run_igvf_batch.py for each arm on changed datasets")
    pr.add_argument("--arm", nargs="+", choices=["manual", "auto"], default=["manual", "auto"])
    pr.add_argument("--n-jobs-inner", type=int, default=32,
                    help="per-arm CPU budget (arms run concurrently; mind the 4096 proc+thread cap)")
    pr.add_argument("--datasets", nargs="+", default=None, help="restrict to these (internal) names")
    pr.add_argument("--dry-run", action="store_true")
    pr.set_defaults(func=cmd_run)
    sub.add_parser("compare", help="pooled + paired metrics, manual vs auto").set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
