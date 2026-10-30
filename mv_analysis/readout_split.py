"""Joint vs independent calibration of functional assays, split by how many
of a gene's readouts measured each control variant (one, or two or more).

A variant measured by a single readout has nothing to combine: joint
calibration scores it by that readout's density within the joint model
(shared components, one gene-level prior), independent calibration by that
readout's own ExCALIBR calibration. The split separates where combining
readouts can help (two or more) from where joint calibration can at best
match independent calibration (one).

Per-variant joint points come from the P-KNN/out-of-bag prep (the paper's
canonical configuration, asserted against the per-gene results table);
independent points are recomputed from the current univariate calibrations
(report._uv_nonconflicting_points), as the prep can predate them.

Outputs (PAPER_FUNCTIONAL_DIR/readout_split/):
  readout_split_pooled.csv    pooled MCC, determinate rate, errors per group
                              and method, with the paired bootstrap interval
                              of the MCC and determinate-rate differences
                              (variants resampled within gene)
  readout_split_by_gene.csv   per gene and group, counts and metrics
  readout_split_facts.json    numbers for the manuscript macros

Usage:
  python -m mv_analysis.readout_split
"""
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis.baselines import EVIDENCE_DIRECTION

N_BOOT = 2000
GROUPS = (("single", "one readout"), ("multi", "two or more readouts"))


def _counts(y, p):
    p = np.nan_to_num(np.asarray(p, float))
    y = np.asarray(y, bool)
    return dict(tp=int(((p >= 1) & y).sum()), fn=int(((p <= -1) & y).sum()),
                fp=int(((p >= 1) & ~y).sum()), tn=int(((p <= -1) & ~y).sum()), n=len(p))


def _metrics(c):
    tp, tn, fp, fn = (float(c[k]) for k in ("tp", "tn", "fp", "fn"))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(mcc=(tp * tn - fp * fn) / den if den > 0 else np.nan,
                det=(tp + tn + fp + fn) / c["n"] if c["n"] else np.nan, errors=int(fp + fn))


def variant_table():
    """One row per functional control variant: gene, single (one readout),
    label, joint and independent points."""
    from mv_analysis.comparison_methods import pknn_runs as RP
    from mv_analysis.report import _uv_nonconflicting_points
    warnings.filterwarnings("ignore")
    comp = pd.read_csv(Path(config.PAPER_PRODUCTION_OOB_DIR) / "production_oob_comparison.csv")
    keys = comp.loc[comp["panel"] == "functional", "key"].unique()
    res = pd.read_csv(Path(config.PAPER_FUNCTIONAL_DIR) / "functional_all_results_table.csv")
    res = res[res["threshold"] == EVIDENCE_DIRECTION]
    rows = []
    for key in keys:
        d, m = RP.load_prep(key)
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                                  results_json=config.PAPER_RESULTS_JSON)
            gene = m["gene"].lower() if m["gene_set"] == "labelseq" else m["gene"].upper()
            uv = _uv_nonconflicting_points(gene, m["gene_set"], ms)
        assert [str(v) for v in ms.kept_variants] == list(d["variants"]), f"{key}: rows differ from prep"
        ev = d["eval_mask"]
        joint, indep = np.nan_to_num(d["mv_points"]), np.nan_to_num(np.asarray(uv, float)) if uv is not None else None
        if indep is None:
            continue
        # joint points must be the paper's: determinate count matches the canonical results row
        cl = "gene" if m["cluster_idx"] == 0 else f"cluster{m['cluster_idx'] + 1}"
        mv = config.select_canonical_mv_rows(res[res["method"].str.startswith("MV") & (res["cluster"] == cl)
                                                 & (res["gene"].astype(str).str.upper() == str(m["gene"]).upper())])
        if len(mv):
            assert int(mv["determinate"].iloc[0]) == int((joint[ev] != 0).sum()), f"{key}: joint points differ from paper"
        nobs = (~np.isnan(d["X"])).sum(1)
        rows.append(pd.DataFrame({"gene": m["gene"], "key": key, "single": nobs[ev] == 1,
                                  "y": d["plp"][ev].astype(bool), "joint": joint[ev], "indep": indep[ev]}))
    return pd.concat(rows, ignore_index=True)


def run(seed=0):
    out = Path(config.PAPER_FUNCTIONAL_DIR) / "readout_split"
    out.mkdir(parents=True, exist_ok=True)
    v = variant_table()
    rng = np.random.default_rng(seed)
    pooled, facts = [], {"n_controls": len(v)}
    for tag, label in GROUPS:
        x = v[v["single"] == (tag == "single")]
        y, j, u = x["y"].to_numpy(), x["joint"].to_numpy(), x["indep"].to_numpy()
        mj, mu = _metrics(_counts(y, j)), _metrics(_counts(y, u))
        idx = [np.flatnonzero(x["gene"].to_numpy() == g) for g in x["gene"].unique()]
        dm, dd = [], []
        for _ in range(N_BOOT):
            b = np.concatenate([rng.choice(i, len(i)) for i in idx])
            a, c = _metrics(_counts(y[b], j[b])), _metrics(_counts(y[b], u[b]))
            dm.append(a["mcc"] - c["mcc"])
            dd.append(a["det"] - c["det"])
        r = dict(group=label, n=len(x), n_genes=x["gene"].nunique(),
                 joint_mcc=mj["mcc"], indep_mcc=mu["mcc"], joint_det=mj["det"], indep_det=mu["det"],
                 joint_errors=mj["errors"], indep_errors=mu["errors"],
                 diff_mcc=mj["mcc"] - mu["mcc"], diff_mcc_lo=np.percentile(dm, 2.5), diff_mcc_hi=np.percentile(dm, 97.5),
                 diff_det=mj["det"] - mu["det"], diff_det_lo=np.percentile(dd, 2.5), diff_det_hi=np.percentile(dd, 97.5))
        pooled.append(r)
        facts[tag] = {k: (float(val) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val)
                      for k, val in r.items()}
    pd.DataFrame(pooled).to_csv(out / "readout_split_pooled.csv", index=False)
    # mixed rule: independent evidence for single-readout variants, joint for the rest
    mixed = np.where(v["single"], v["indep"], v["joint"])
    facts["mixed"] = _metrics(_counts(v["y"], mixed))
    facts["all_joint"] = _metrics(_counts(v["y"], v["joint"]))
    facts["all_indep"] = _metrics(_counts(v["y"], v["indep"]))
    by = []
    for (g, single), x in v.groupby(["gene", "single"]):
        cj, cu = _counts(x["y"], x["joint"]), _counts(x["y"], x["indep"])
        by.append(dict(gene=g, group="one readout" if single else "two or more readouts", n=len(x),
                       n_plp=int(x["y"].sum()), n_blb=int((~x["y"]).sum()),
                       **{f"joint_{k}": val for k, val in _metrics(cj).items()},
                       **{f"indep_{k}": val for k, val in _metrics(cu).items()}))
    by = pd.DataFrame(by)
    by.to_csv(out / "readout_split_by_gene.csv", index=False)
    share = v.groupby("gene")["single"].mean()
    facts["pct_single"] = float(100 * v["single"].mean())
    facts["n_single"] = int(v["single"].sum())
    facts["genes_mostly_single"] = sorted(share[share >= 0.8].index.tolist())
    (out / "readout_split_facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print(pd.DataFrame(pooled).round(3).to_string(index=False))
    print(json.dumps({k: facts[k] for k in ("mixed", "all_joint", "all_indep", "pct_single", "genes_mostly_single")},
                     indent=1, default=float))
    return facts


if __name__ == "__main__":
    run()
