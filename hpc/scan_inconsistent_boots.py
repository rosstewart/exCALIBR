"""SCAN (analysis only; production defaults unchanged): how often the opt-in
``drop_inconsistent_boots`` check fires across the paper run, and how each
dataset/config's metrics change with it on vs off.

    nice -n 10 taskset -c 0-23 python hpc/scan_inconsistent_boots.py --workers 6
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, gzip, json, sys, warnings, logging, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p_ in (REPO, REPO / "mv_analysis"):
    sys.path.insert(0, str(p_))

OUT = Path("/data/ross/assay_calibration/multivariate/inconsistent_boot_scan_101426")
RULE = "margin"
RUN_KW = dict(path_percentile=5, min_valid_boots=1, reestimate_marginal_weights=False,
              enforce_marginal_monotonicity=False, liberal_marginal_monotonicity=False,
              partial_pattern_mode="trust_global", n_jobs=4)
AUX = {"TP53_tp53_mv": [4], "CARD11_card11_mv": [4, 5]}


def parse(key):
    for suf, gs in (("_labelseq_mv", "labelseq"), ("_predictors_mv", "predictor"), ("_combined_fgfr_mv", "fgfr"),
                    ("_combined_mv", "combined"), ("_card11_mv", "card11"), ("_tsc2_mv", "tsc2"), ("_tp53_mv", "tp53"),
                    ("_mv_clinvar_2018", "integrated"), ("_mv", "integrated")):
        if key.endswith(suf):
            return gs, key[: -len(suf)]
    return None, None


def scan_one(key):
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import config as mv_config
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.multivariate_analysis.report_gene import confusion_and_metrics
    gs, gene = parse(key)
    rows = []
    try:
        ms = mv_config.build_ms_for_gene_cluster(gs, gene, results_json=mv_config.PAPER_RESULTS_JSON)
        ds = ("TP53_tp53_mv" if gs == "tp53" else
              mv_config.canonical_dataset_name(gene, gs, results_json=mv_config.PAPER_RESULTS_JSON))
        if ds != key:
            return [{"dataset": key, "error": f"name resolves to {ds}"}]
        for drop in (False, True):
            an = build_gene_set_analysis(ms, gene, mv_config.PAPER_RESULTS_JSON, dataset_name=key,
                                         auxiliary_pathogenic_indices=AUX.get(key), gene_set=gs)
            kw = dict(RUN_KW)
            if AUX.get(key):
                kw.update(aux_path_percentile=50, aux_ben_percentile=50)
            an.run(drop_inconsistent_boots=drop, inconsistency_rule=RULE, **kw)
            chk = getattr(an, "inconsistent_boot_check", {})
            for cfg in an.configs:
                if an.results.get(cfg) is None:
                    rows.append({"dataset": key, "config": cfg, "drop": drop, "error": "no result"})
                    continue
                _, m = confusion_and_metrics(an, cfg)
                c = chk.get(cfg, {})
                rows.append({"dataset": key, "gene_set": gs, "config": cfg, "drop": drop,
                             "n_valid": int(an.results[cfg].get("n_valid", 0) or 0),
                             "dropped": c.get("dropped", 0), "median_rho": c.get("median_rho"),
                             **{k: m[k] for k in ("TP", "FN", "TN", "FP", "coverage", "mcc")}})
    except Exception as e:                       # stale fit / unbuildable dataset: record, keep scanning
        rows.append({"dataset": key, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return rows


def main():
    import config as mv_config
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rule", choices=("margin", "strict"), default="margin")
    args = ap.parse_args()
    global OUT, RULE
    RULE = args.rule
    if RULE != "margin":
        OUT = OUT.parent / f"{OUT.name}_{RULE}"
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(json.load(gzip.open(mv_config.PAPER_RESULTS_JSON, "rt")).keys())
    print(f"{len(keys)} datasets", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_one, k): k for k in keys}
        for f in as_completed(futs):
            r = f.result()
            rows += r
            print(f"  {futs[f]} ({time.time() - t0:.0f}s): " +
                  "; ".join(f"{x.get('config')} drop={x.get('drop')} dropped={x.get('dropped')} mcc={x.get('mcc', float('nan')):.3f} cov={x.get('coverage', float('nan')):.3f}"
                            if "error" not in x else f"ERROR {x['error'][:90]}" for x in r), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "scan_rows.csv", index=False)
    df = pd.DataFrame(rows)
    ok = df[df.get("error").isna()] if "error" in df else df
    piv = ok.pivot_table(index=["dataset", "config"], columns="drop",
                         values=["mcc", "coverage", "dropped", "n_valid"], aggfunc="first")
    piv.columns = [f"{a}_{'on' if b else 'off'}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["d_mcc"] = piv["mcc_on"] - piv["mcc_off"]
    piv["d_cov"] = piv["coverage_on"] - piv["coverage_off"]
    piv.to_csv(OUT / "scan_summary.csv", index=False)
    pd.set_option("display.width", 220)
    fired = piv[piv["dropped_on"] > 0].sort_values("dropped_on", ascending=False)
    print(f"\ncheck fired on {len(fired)} of {len(piv)} dataset/config pairs")
    print(fired[["dataset", "config", "n_valid_off", "dropped_on", "mcc_off", "mcc_on", "coverage_off", "coverage_on"]]
          .round(3).to_string(index=False))
    if "error" in df:
        print("\nerrors:", df[df["error"].notna()][["dataset", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
