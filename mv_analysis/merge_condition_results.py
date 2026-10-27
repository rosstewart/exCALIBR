"""Assemble the final MV results from a multi-condition run (hpc/prepare.py
--pathogenic-share), taking each panel's datasets from the condition chosen
for that panel.

    python mv_analysis/merge_condition_results.py <root> \
        --functional tau_0.1 --predictor tau_0.1 --combined tau_0.1 [--phenotype tau_0.1] \
        [--override araf_labelseq_mv=tau_0 ...]

--override takes one dataset from another condition, e.g. the genes without
P/LP, whose correct weighted fit is the unweighted one (pathogenic-share
factor 1) when a run's weighted condition mis-assigned their sample roles.

reads <root>/<condition>/bootstrap_results.json.gz (hpc/aggregate_results.py
<root> --per-condition) and writes <root>/final/bootstrap_results.json.gz,
plus a link to <root>/logs (MethNumRestarts reads the array logs) and
final/conditions.json recording the choice. Then point the analysis at it:
MV_PAPER_FITS_DIR=<root>/final python mv_analysis/analyze_mv_pipeline_output.py.
"""
import argparse
import gzip
import json
import os
from pathlib import Path


def panel_of(dataset_key: str) -> str:
    if dataset_key.endswith("_phenotype_sub_mv"):
        return "phenotype"
    if dataset_key.endswith("_combined_fgfr_mv"):  # pooled FGFR is a functional gene set
        return "functional"
    if "_predictors_mv" in dataset_key:
        return "predictor"
    if "_combined_mv" in dataset_key:
        return "combined"
    return "functional"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--functional", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--combined", required=True)
    ap.add_argument("--phenotype", default=None, help="default: the functional choice")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=COND",
                    help="take this dataset from COND instead of its panel's condition (repeatable)")
    args = ap.parse_args()
    root = Path(args.root)
    choice = {"functional": args.functional, "predictor": args.predictor, "combined": args.combined,
              "phenotype": args.phenotype or args.functional}
    override = dict(o.split("=", 1) for o in args.override)

    loaded, merged, source = {}, {}, {}
    for cond in sorted(set(choice.values()) | set(override.values())):
        with gzip.open(root / cond / "bootstrap_results.json.gz", "rt") as f:
            loaded[cond] = json.load(f)
    for cond, results in loaded.items():
        for key, boots in results.items():
            if override.get(key, choice[panel_of(key)]) == cond:
                merged[key] = boots
                source[key] = cond

    out = root / "final"
    out.mkdir(exist_ok=True)
    with gzip.open(out / "bootstrap_results.json.gz", "wt") as f:
        json.dump(merged, f)
    missing = set(override) - set(merged)
    assert not missing, f"--override datasets not found: {sorted(missing)}"
    (out / "conditions.json").write_text(json.dumps({"panel_choice": choice, "overrides": override,
                                                     "dataset_source": source}, indent=1))
    if (root / "logs").exists() and not (out / "logs").exists():
        os.symlink(root / "logs", out / "logs")
    print(f"{len(merged)} datasets -> {out / 'bootstrap_results.json.gz'}")
    for panel, cond in choice.items():
        print(f"  {panel}: {cond} ({sum(panel_of(k) == panel for k in merged)} datasets)")


if __name__ == "__main__":
    main()
