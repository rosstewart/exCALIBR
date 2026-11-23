"""Sparse-missense regime: refits with the missense P/LP and missense B/LB controls subsampled to n.

For each (dataset, rep) one random permutation of the missense P/LP rows and one of the missense B/LB rows are drawn
(seeded by dataset and rep), and the first n of each are kept, so the n levels of a rep are nested (n=3 is inside
n=5, ...). B/LB missense keep min(n, available). Rows whose label is dropped stay in the scoreset: they keep any other
membership (gnomAD) or become unfitted assayed rows, and are flagged heldout_plp / heldout_blb, so they serve as
held-out missense controls the fit never saw. P/LP+B/LB conflicts and every other consequence are untouched.

A subsample is a task field sub = {"n": n, "rep": rep}; csq_analysis.fits applies it when building the strata, and the
output label carries it (S1_m5_R8__n10r2).

    python -m csq_analysis.downsample make <out> [--configs S1_m5] [--B 50] [--R 8] [--n 3,5,10,20,40] [--reps 5]
then run the bundle with python -m csq_analysis.fits run (hpc/submit_csq.sh).
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np

# one dataset per gene, each with >= 40 missense P/LP (B/LB missense is the scarce side: 1-56)
PANEL = [
    "LDLR_Tabet_2025_abundance", "GCK_Gersing_2024_abundance", "G6PD_IGVF", "VHL_Buckley_2024",
    "KCNH2_O_Neill_2024_surface_expression", "F9_Popp_2025_light_chain",
    "TP53_Giacomelli_2018_combined_score_clinvar_2018", "DDX3X_Radford_2023", "PTEN_Mighell_2018_clinvar_2018",
    "OTC_Lo_2023", "BRCA1_Findlay_2018_clinvar_2018", "BRCA2_Huang_2025_SGE", "TSC2_IGVF",
    "MSH2_Jia_2021_clinvar_2018", "ASPA_Grønbæk-Thygesen_2024_abundance",
]
LEVELS = (3, 5, 10, 20, 40)
REPS = 5


def _perm(ds, rep, what, n):
    seed = zlib.crc32(f"{ds}|{rep}|{what}".encode()) & 0x7FFFFFFF
    return np.random.RandomState(seed).permutation(n)


def subsample(rows, ds, n, rep):
    """rows of one dataset (csq_analysis.annotate rows, original order) with all but n missense P/LP and n missense
    B/LB labels removed. Row order and count are unchanged."""
    r = rows.copy()
    r["heldout_plp"] = False
    r["heldout_blb"] = False
    mis = r.csq.eq("missense").to_numpy()
    for flag, other, what in (("plp", "blb", "P"), ("blb", "plp", "B")):
        pool = np.flatnonzero(mis & r[flag].to_numpy() & ~r[other].to_numpy())
        drop = pool[_perm(ds, rep, what, len(pool))[int(n):]]
        col = r.columns.get_loc(flag)
        r.iloc[drop, col] = False
        r.iloc[drop, r.columns.get_loc(f"heldout_{flag}")] = True
    return r


def sub_tag(sub):
    return f"n{int(sub['n'])}r{int(sub['rep'])}"


def parse_tag(tag):
    n, rep = tag[1:].split("r")
    return dict(n=int(n), rep=int(rep))


def make(out_dir, configs=("S1_m5",), B=50, R=8, levels=LEVELS, reps=REPS, datasets=None, per_task=10):
    from csq_analysis import fits
    subs = [dict(n=int(n), rep=int(k)) for n in levels for k in range(reps)]
    fits.make(out_dir, list(configs), B, R, datasets or PANEL, per_task, subs=subs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out_dir")
    m.add_argument("--configs", default="S1_m5")
    m.add_argument("--B", type=int, default=50)
    m.add_argument("--R", type=int, default=8)
    m.add_argument("--n", default=",".join(map(str, LEVELS)))
    m.add_argument("--reps", type=int, default=REPS)
    m.add_argument("--datasets", default=None)
    m.add_argument("--per-task", type=int, default=10)
    a = ap.parse_args()
    make(a.out_dir, a.configs.split(","), a.B, a.R, [int(x) for x in a.n.split(",")], a.reps,
         a.datasets.split(",") if a.datasets else None, a.per_task)


if __name__ == "__main__":
    main()
