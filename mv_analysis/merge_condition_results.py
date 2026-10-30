"""Assemble the final MV results from a multi-condition run (hpc/prepare.py
--pathogenic-share), taking each panel's datasets from the condition chosen
for that panel.

    python mv_analysis/merge_condition_results.py <root> \
        --functional tau_0.1 --predictor tau_0.1 --combined tau_0.1 [--phenotype tau_0.1] \
        [--override araf_labelseq_mv=tau_0 ...] [--supplement <run dir> ...] [--replace <run dir> ...]

--override takes one dataset from another condition, e.g. the genes without
P/LP, whose correct weighted fit is the unweighted one (pathogenic-share
factor 1) when a run's weighted condition mis-assigned their sample roles.

--supplement adds bootstraps from a later completion run (its own
bootstrap_results.json.gz, same condition and settings) to datasets of the
merge, only at bootstrap indices the merge lacks; an index present in both
is an error, so a completion run never replaces an existing fit.

--replace takes every dataset of a refit run (<run dir>/bootstrap_results.json.gz)
wholesale, replacing the merged dataset of the same name or adding a new one,
e.g. a model refit on new data (TSC2 on the Biar et al. panel) or split into
clusters (BRCA2 combined).

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
    if "_combined_mv" in dataset_key or "_combined_cluster" in dataset_key:
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
    ap.add_argument("--supplement", action="append", default=[], metavar="DIR",
                    help="completion run whose bootstraps fill missing indices of the merged datasets (repeatable)")
    ap.add_argument("--replace", action="append", default=[], metavar="DIR[:KEY,KEY]",
                    help="refit run whose datasets replace (or are added to) the merge wholesale; "
                         "':KEY,KEY' takes only those datasets, e.g. a finished one from a run still going (repeatable)")
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

    added = {}
    for sup in args.supplement:
        with gzip.open(Path(sup) / "bootstrap_results.json.gz", "rt") as f:
            extra = json.load(f)
        for key, boots in extra.items():
            assert key in merged, f"--supplement {sup}: {key} is not in the merge"
            clash = set(boots) & set(merged[key])
            assert not clash, f"--supplement {sup}: {key} already has bootstraps {sorted(clash, key=int)[:5]}"
            merged[key] = {**merged[key], **boots}
            added.setdefault(str(sup), {})[key] = sorted(boots, key=int)
            print(f"  {key}: +{len(boots)} bootstraps from {sup} ({len(merged[key])} total)")

    replaced = {}
    for spec in args.replace:
        rep, _, only = spec.partition(":")
        with gzip.open(Path(rep) / "bootstrap_results.json.gz", "rt") as f:
            refit = json.load(f)
        if only:
            missing = set(only.split(",")) - set(refit)
            assert not missing, f"--replace {rep}: {sorted(missing)} not in that run"
            refit = {k: refit[k] for k in only.split(",")}
        for key, boots in refit.items():
            print(f"  {key}: {'replaced' if key in merged else 'added'} from {rep} ({len(boots)} bootstraps)")
            merged[key] = boots
            source[key] = str(rep)
            replaced.setdefault(str(rep), []).append(key)

    out = root / "final"
    out.mkdir(exist_ok=True)
    with gzip.open(out / "bootstrap_results.json.gz", "wt") as f:
        json.dump(merged, f)
    missing = set(override) - set(merged)
    assert not missing, f"--override datasets not found: {sorted(missing)}"
    (out / "conditions.json").write_text(json.dumps({"panel_choice": choice, "overrides": override,
                                                     "dataset_source": source, "supplements": added,
                                                     "replacements": replaced}, indent=1))
    if (root / "logs").exists() and not (out / "logs").exists():
        os.symlink(root / "logs", out / "logs")
    print(f"{len(merged)} datasets -> {out / 'bootstrap_results.json.gz'}")
    for panel, cond in choice.items():
        print(f"  {panel}: {cond} ({sum(panel_of(k) == panel for k in merged)} datasets)")


if __name__ == "__main__":
    main()
