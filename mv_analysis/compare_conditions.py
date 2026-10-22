"""Compare MV results across fit runs/conditions panel by panel, from the
results tables analyze_mv_pipeline_output.py writes under each run's figures
directory.

    python mv_analysis/compare_conditions.py \
        paper=/data/ross/assay_calibration/multivariate/jobs_all_100b_3f_092026 \
        tau_0.1=<root>/tau_0.1 tau_0=<root>/tau_0 tau_0.2=<root>/tau_0.2 \
        [--out <dir>]

Per panel (functional, predictor, combined): pooled MCC, median per-gene MCC
(genes with both classes), determinate rate, errors, correct P/LP and B/LB
calls, and genes better / worse than the first run listed. MV rows use each
gene's canonical config (config.select_canonical_mv_rows, TP53 fixed at 4c).
Writes condition_comparison_panels.csv and condition_comparison_genes.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mv_analysis import config  # noqa: E402
from mv_analysis.figures.scripts.make_panel_performance import PANELS  # noqa: E402

EVIDENCE = "evidence_direction (>=1 / <=-1)"


def _mcc(tp, tn, fp, fn):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else np.nan


def mv_rows(run_dir, panel):
    path = Path(run_dir) / "figures" / PANELS[panel][0]
    if not path.exists() or path.stat().st_size < 10:  # absent or empty (no fits for this panel)
        return None
    t = pd.read_csv(path)
    t = t[(t["threshold"] == EVIDENCE) & t["method"].str.startswith("MV")]
    return config.select_canonical_mv_rows(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=run_dir (first is the reference)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    panels, genes = [], []
    for panel in PANELS:
        tabs = {lab: mv_rows(d, panel) for lab, d in runs}
        ref_lab = runs[0][0]
        ref = tabs[ref_lab]
        for lab, t in tabs.items():
            if t is None:
                continue
            s = t[["TP", "TN", "FP", "FN", "determinate", "total"]].sum()
            both = t[(t["n_pathogenic"] > 0) & (t["n_benign"] > 0)]
            row = dict(panel=panel, run=lab, genes=len(t), pooled_mcc=_mcc(s.TP, s.TN, s.FP, s.FN),
                       median_mcc=both["mcc"].median(), det_rate=s.determinate / s.total,
                       errors=int(s.FP + s.FN), correct_plp=int(s.TP), correct_blb=int(s.TN))
            if ref is not None and lab != ref_lab:
                keys = [k for k in ("gene", "gene_set", "cluster") if k in both.columns and k in ref.columns]
                m = both.merge(ref, on=keys, suffixes=("", "_ref"))
                d = m["mcc"] - m["mcc_ref"]
                row.update(better=int((d > 0.005).sum()), worse=int((d < -0.005).sum()),
                           worse_genes=",".join(sorted(m.loc[d < -0.005, "gene"].astype(str))))
            panels.append(row)
            for _, r in t.iterrows():
                genes.append(dict(panel=panel, run=lab, gene=r["gene"], gene_set=r.get("gene_set", panel),
                                  cluster=r["cluster"], config=r["config"], mcc=r["mcc"],
                                  det_rate=r["determinate"] / r["total"] if r["total"] else np.nan,
                                  errors=int(r["FP"] + r["FN"]), TP=int(r["TP"]), TN=int(r["TN"]),
                                  n_pathogenic=int(r["n_pathogenic"]), n_benign=int(r["n_benign"])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P, G = pd.DataFrame(panels), pd.DataFrame(genes)
    P.to_csv(out / "condition_comparison_panels.csv", index=False)
    G.to_csv(out / "condition_comparison_genes.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
