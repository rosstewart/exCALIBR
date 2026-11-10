"""acmgscaler arm of the calibration-quality simulation (local; needs Rscript and config.ACMGSCALER_DIR).

Each replicate's data are regenerated from its cell seed (analysis.calibration_quality.generate.rng_for), so this
runs after, and independently of, the HPC grid. acmgscaler is fit to the P/LP and B/LB controls only, with the
fixed prior used in the manuscript (0.10); the held-out evaluation draws are passed as unlabeled rows, so every one
gets a call, and its points (on the ACMG/AMP levels +-1/2/4/8) are compared with the truth floored to the same
levels. Cells with fewer than 10 P/LP or 10 B/LB controls are recorded as not calibratable (acmgscaler's own
minimum).

    python -m analysis.calibration_quality.acmgscaler_arm <out_dir> [--reps 30] [--n-jobs 32]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.calibration_quality import generate as G
from analysis.calibration_quality.run import (MAPPINGS, compare, posterior, posterior_histograms,
                                             posterior_metrics, to_tier, truth_points)

FIXED_PRIOR = 0.10
N_EVAL_ACMG = 3000  # per class: acmgscaler scores every row, so a smaller draw keeps the R call fast


def run_one(cell, rep, prior=FIXED_PRIOR):
    from analysis.comparison_methods import run_acmgscaler
    t0 = time.time()
    row = {**cell, "rep": int(rep), "acmg_prior": prior, "acmg_prior_err": prior - cell["pi"]}
    if cell["n_plp"] < 10 or cell["n_blb"] < 10:
        row["acmg_status"] = "not calibratable (<10 P/LP or B/LB)"
        return row
    d = G.make_data(G.rng_for(cell, rep), cell)
    plp, blb = d["samples"][0], d["samples"][1]
    ev = {k: x[:N_EVAL_ACMG] for k, x in d["eval"].items()}
    scores = np.concatenate([plp, blb] + list(ev.values()))
    sample = ["Pathogenic/Likely Pathogenic"] * len(plp) + ["Benign/Likely Benign"] * len(blb) + \
             [""] * sum(len(x) for x in ev.values())
    try:
        out = run_acmgscaler(pd.DataFrame({"score": scores, "sample": sample}), prior=prior)
        pts = out["acmgscaler_points"].to_numpy()[len(plp) + len(blb):].astype(int)
        o = 0
        for k, x in ev.items():
            p = pts[o:o + len(x)]
            o += len(x)
            for mapping, pre in MAPPINGS.items():  # truth under each threshold mapping, floored to the levels
                ref = to_tier(truth_points(G.true_log_lr(x, cell), cell["pi"], mapping))
                for m, v in compare(p, ref).items():
                    row[f"{pre}acmg_vs_truth_{k}_tier_{m}"] = v
        # Posteriors (exact; no evidence-level mapping) from the likelihood ratio acmgscaler itself uses for
        # evidence, unmodified (add_evidence_levels.R): the upper end of its interval where its median LR is below
        # 1, else the lower end; with its fixed prior 0.10.
        lr = out["acmgscaler_lr"].to_numpy(float)[len(plp) + len(blb):]
        lo = out["acmgscaler_lr_lower"].to_numpy(float)[len(plp) + len(blb):]
        hi = out["acmgscaler_lr_upper"].to_numpy(float)[len(plp) + len(blb):]
        lr_c = np.where(lr < 1, hi, lo)
        o = 0
        for k, x in ev.items():
            sl = slice(o, o + len(x))
            o += len(x)
            true_post = posterior(np.exp(G.true_log_lr(x, cell)), cell["pi"])
            is_path = d["eval_pop_is_path"][:len(x)] if k == "population" else None
            post_c = posterior(lr_c[sl], prior)
            for m, v in posterior_metrics(post_c, true_post, lr_c[sl], is_path).items():
                row[f"acmg_{k}_{m}"] = v
            for m, v in posterior_histograms(post_c, true_post if k != "population" else None, is_path).items():
                row[f"acmg_{k}_{m}"] = v
            row[f"acmg_{k}_post_mae_central"] = float(np.abs(posterior(lr[sl], prior) - true_post).mean())
        row["acmg_status"] = "ok"
    except Exception as e:
        row["acmg_status"] = f"error: {type(e).__name__}: {e}"[:300]
    row["acmg_seconds"] = round(time.time() - t0, 1)
    return row


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=32)
    a = ap.parse_args()
    cells = G.design()
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(run_one)(c, r) for r in range(a.reps) for c in cells)
    out = Path(a.out_dir) / "acmgscaler_rows.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
