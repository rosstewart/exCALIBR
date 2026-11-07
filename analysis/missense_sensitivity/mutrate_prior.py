"""E8: consequence-specific, mutation-rate-weighted (de novo) missense priors on the baseline fits.

The population is every assayed missense variant of the dataset, each weighted by
the rate at which it arises de novo. The fitted P/LP and benign-reference
densities are unchanged, so for each baseline bootstrap fit the prior is the
weighted label-shift EM (see af_prior.weighted_prior_em) over those variants.

Mutation rates: gnomAD v2 (Karczewski 2020) per-trinucleotide SNV rates
(mutation_rate_methylation_bins.txt.bgz), CpG transitions at methylation level
--cpg-level (default 2 = high, typical of coding-exon CpGs; 0 as a sensitivity).
Sequence context from hg38.

Weight of a missense variant:
  * nucleotide-level datasets (labelled "nt" and mostly single-nucleotide rows): the rate of that exact nucleotide change (SNV: its
    context rate; MNV: the MNV rate below);
  * amino-acid-level datasets (one library codon per protein change): the rate of the
    PROTEIN change from the reference codon = sum of the rates of every SNV in the
    codon that produces it, plus the MNV rate of every adjacent double change that
    does; split equally among the dataset's rows for that change.
MNVs are mostly single multinucleotide mutation events, far more frequent than two
independent SNVs. They are modelled as a rate per adjacent-pair double change of
rho * mu_bar / 9 (mu_bar = mean per-site SNV rate), so rho is the MNV event rate
per adjacent pair relative to the SNV rate per site. rho = 0 drops MNV-only
changes; reported at several rho (ASSUMPTION: pin rho to Kaplanis 2019 /
Besenbacher 2016 before use).

Variants reported per dataset (median over bootstraps):
  uniform        every assayed missense row weight 1
  uniform_snv    SNV-reachable protein changes only, weight 1
  mu_rho0        mutation-rate weights, MNVs excluded
  mu_rho<r>      mutation-rate weights, MNVs at rho = r
  mu_rho0_cpg0   rho = 0 with CpG transitions at methylation level 0

Usage:  python -m analysis.missense_sensitivity.mutrate_prior [--n-jobs 32]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/data/ross/assay_calibration/missense_only_calibrations")
MUT_TABLE = Path("/data/ross/assay_calibration/reference/mutation_rate_methylation_bins.txt.bgz")
HG38 = Path("/data/dbs/reference_genome/hg38.fa.gz")
WINDOWS = ROOT / "analysis_v3" / "e8_hg38_windows.pkl"
PLP, BLB, SYN = "Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "Synonymous"
RHOS = [0.01, 0.03, 0.1]

_COMP = str.maketrans("ACGT", "TGCA")
_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AAS[16 * i + 4 * j + k] for i, a in enumerate(_BASES) for j, b in enumerate(_BASES)
         for k, c in enumerate(_BASES)}
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
       "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "*"}


def rc(s):
    return s.translate(_COMP)[::-1]


def load_mu():
    t = pd.read_csv(MUT_TABLE, sep="\t", compression="gzip")
    mu = {(r.context, r.alt, int(r.methylation_level)): float(r.mu_snp) for r in t.itertuples()}
    lvl0 = t[t.methylation_level == 0]
    mu_bar = float(lvl0.groupby("context").mu_snp.sum().mean())  # mean per-site SNV rate
    return mu, mu_bar


def snv_rate(mu, ctx, alt, cpg_level):
    """Rate of ctx[1] -> alt (genomic orientation, ctx = 3 bases)."""
    if len(ctx) != 3 or "N" in ctx or alt == ctx[1]:
        return np.nan
    if ctx[1] in "GT":
        ctx, alt = rc(ctx), rc(alt)
    lvl = cpg_level if (ctx[1] == "C" and ctx[2] == "G" and alt == "T") else 0
    return mu.get((ctx, alt, lvl), mu.get((ctx, alt, 0), np.nan))


def build_windows(regions, pad=20):
    """{chrom: [(start, end, seq)]} for the requested (chrom, lo, hi) regions (1-based, inclusive)."""
    want = {}
    for c, lo, hi in regions:
        want.setdefault(str(c), []).append((max(1, lo - pad), hi + pad))
    out, chrom, buf = {}, None, []

    def _flush():
        if chrom and chrom in want:
            s = "".join(buf).upper()
            out[chrom] = [(lo, hi, s[lo - 1:hi]) for lo, hi in want[chrom]]
    with gzip.open(HG38, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                _flush()
                name = line[1:].split()[0]
                chrom = name[3:] if name.startswith("chr") else name
                buf = []
                if chrom not in want:
                    chrom = None
            elif chrom:
                buf.append(line.rstrip())
    _flush()
    return out


class Genome:
    def __init__(self, windows):
        self.w = windows

    def base(self, chrom, pos):
        for lo, hi, s in self.w.get(str(chrom), []):
            if lo <= pos <= hi:
                return s[pos - lo]
        return "N"

    def ctx(self, chrom, pos):
        return "".join(self.base(chrom, p) for p in (pos - 1, pos, pos + 1))


def _strand(rows):
    """+1 if genomic ref == transcript ref for single-nt rows, else -1 (majority)."""
    v = [1 if r["ref_allele"] == r["transcript_ref"] else -1 for r in rows
         if isinstance(r["ref_allele"], str) and len(r["ref_allele"]) == 1
         and isinstance(r["transcript_ref"], str) and len(r["transcript_ref"]) == 1]
    return 1 if (np.mean(v) if v else 1) >= 0 else -1


def _codon_positions(r, strand):
    tp = int(r["transcript_pos"])
    phase = (tp - 1) % 3
    if strand == 1:
        g0 = int(r["hg38_start"]) - phase
        return [g0, g0 + 1, g0 + 2]
    g0 = int(r["hg38_end"]) + phase
    return [g0, g0 - 1, g0 - 2]


def protein_change_rates(genome, mu, mu_bar, chrom, gpos, strand, rhos, cpg_level):
    """Reference codon and {rho: {aa: rate}} for every protein change from it."""
    gb = [genome.base(chrom, p) for p in gpos]
    tb = gb if strand == 1 else [b.translate(_COMP) for b in gb]
    ref_codon = "".join(tb)
    if ref_codon not in CODON:
        return None, None
    snv = {}
    for i, p in enumerate(gpos):
        ctx = genome.ctx(chrom, p)
        for alt in "ACGT":
            if alt == gb[i]:
                continue
            c = list(tb); c[i] = alt if strand == 1 else alt.translate(_COMP)
            aa = CODON["".join(c)]
            snv[aa] = snv.get(aa, 0.0) + snv_rate(mu, ctx, alt, cpg_level)
    mnv = {}
    for i in (0, 1):
        for a in "ACGT":
            for b in "ACGT":
                if a == tb[i] or b == tb[i + 1]:
                    continue
                c = list(tb); c[i] = a; c[i + 1] = b
                aa = CODON["".join(c)]
                mnv[aa] = mnv.get(aa, 0.0) + mu_bar / 9
    out = {}
    for rho in rhos:
        d = dict(snv)
        if rho > 0:
            for aa, m in mnv.items():
                d[aa] = d.get(aa, 0.0) + rho * m
        out[rho] = d
    return ref_codon, out


def _aa1(r):
    a = str(r.get("aa_alt", ""))
    return AA3.get(a, a) if len(a) == 3 else a


def missense_weights(rows, genome, mu, mu_bar, nt_level, rhos, cpg_level):
    """{rho: weights} over `rows` (missense population rows), plus diagnostics."""
    strand = _strand(rows)
    chrom = str(rows[0]["Chrom"])
    n = len(rows)
    W = {rho: np.full(n, np.nan) for rho in rhos}
    snv_reach = np.zeros(n, bool)
    bad = 0
    key = [(int(r["aa_pos"]) if np.isfinite(float(r["aa_pos"])) else -1, _aa1(r)) for r in rows]
    counts = pd.Series(key).value_counts().to_dict()
    cache = {}
    ref_ok = []
    for i, r in enumerate(rows):
        ra = str(r["ref_allele"])
        if len(ra) == 1:
            ref_ok.append(genome.base(chrom, int(r["hg38_start"])) == ra)
        try:
            gpos = _codon_positions(r, strand)
        except (TypeError, ValueError):
            bad += 1
            continue
        ck = tuple(gpos)
        if ck not in cache:
            cache[ck] = protein_change_rates(genome, mu, mu_bar, chrom, gpos, strand, rhos, cpg_level)
        ref_codon, rates = cache[ck]
        aa_ref = str(r.get("aa_ref", ""))
        aa_ref = AA3.get(aa_ref, aa_ref) if len(aa_ref) == 3 else aa_ref
        if ref_codon is None or CODON[ref_codon] != aa_ref:
            bad += 1
            continue
        aa = key[i][1]
        snv_reach[i] = rates[0.0].get(aa, 0.0) > 0
        ref, alt = str(r["ref_allele"]), str(r["alt_allele"])
        diffs = [j for j in range(min(len(ref), len(alt))) if ref[j] != alt[j]]
        for rho in rhos:
            if nt_level and len(ref) == len(alt) and len(diffs) == 1:
                W[rho][i] = snv_rate(mu, genome.ctx(chrom, int(r["hg38_start"]) + diffs[0]), alt[diffs[0]],
                                     cpg_level)
            elif nt_level and len(diffs) == 2 and diffs[1] - diffs[0] == 1:
                W[rho][i] = rho * mu_bar / 9
            elif nt_level:
                W[rho][i] = 0.0
            else:
                W[rho][i] = rates[rho].get(aa, 0.0) / counts[key[i]]
    return W, snv_reach, bad / max(n, 1), strand, (float(np.mean(ref_ok)) if ref_ok else np.nan)


def _task(ds, df_ds, fits_ds, cal, windows, cpg_level):
    from analysis.missense_sensitivity.af_prior import weighted_prior_em
    from src.assay_calibration.fit_utils.cfusn import density_utils
    from src.assay_calibration.fit_utils.point_ranges import _get_benign_reference_weights
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    release = "2018" if ds.endswith("_clinvar_2018") else "2025"
    base = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                      clinvar_release=release))
    names = [n for n, c in zip(base.sample_names, base.sample_counts) if c > 0]
    if PLP not in names or not ({BLB, SYN} & set(names)):
        return dict(dataset=ds, status="PU/NU skipped")
    ss = load_dataset_from_df(df_ds, PipelineConfig(dataset_csv="", dataset_name=ds, output_dir="/tmp",
                                                    clinvar_release=release, population_type="all_variants"))
    pop_rows, pop_scores = [], []
    for all_idx, (_, variants) in enumerate(ss.get_variants_by_id().items()):
        if not ss._keep_mask[all_idx]:
            continue
        v = variants[0]
        if getattr(v, "is_missense", False) and np.isfinite(float(v.assay_score)):
            pop_rows.append(v.row); pop_scores.append(float(v.assay_score))
    if len(pop_rows) < 10:
        return dict(dataset=ds, status="too few missense")
    x = np.asarray(pop_scores)
    mu, mu_bar = load_mu()
    genome = Genome(windows)
    # nucleotide-level = the rows are mostly single-nucleotide changes (SGE-like); codon
    # libraries labelled "nt" (e.g. randomised codons) are treated as amino-acid level
    nt_level = (str(df_ds["nucleotide_or_aa"].iloc[0]) == "nt"
                and np.mean([len(str(r["ref_allele"])) == 1 for r in pop_rows]) >= 0.5)
    rhos = [0.0] + RHOS
    W, reach, bad, strand, ref_match = missense_weights(pop_rows, genome, mu, mu_bar, nt_level, rhos, cpg_level)
    W0c, _, _, _, _ = missense_weights(pop_rows, genome, mu, mu_bar, nt_level, [0.0], 0)
    ok = np.isfinite(W[0.0])
    variants = {"uniform": np.ones(len(x)), "uniform_snv": (reach & ok).astype(float),
                "mu_rho0": np.nan_to_num(W[0.0]), "mu_rho0_cpg0": np.nan_to_num(W0c[0.0])}
    for rho in RHOS:
        variants[f"mu_rho{rho:g}"] = np.nan_to_num(W[rho])
    idx = dict(p=names.index(PLP), b=names.index(BLB) if BLB in names else None,
               s=names.index(SYN) if SYN in names else None)
    bm = cal.get("benign_method", "avg")
    if idx["s"] is None and bm in ("avg", "synonymous"):
        bm = "benign"
    if idx["b"] is None and bm in ("avg", "benign"):
        bm = "synonymous"
    b_idx = idx["s"] if (bm == "synonymous" or idx["b"] is None) else idx["b"]
    est = {k: [] for k in variants}
    for rec in fits_ds.values():
        m = next(iter(rec.values()))["fit"]
        params, weights = m["component_params"], m["weights"]
        try:
            f_p = density_utils.joint_densities(x, params, weights[idx["p"]]).sum(axis=0)
            w_b = _get_benign_reference_weights(weights, bm, b_idx, idx["s"])
            f_b = density_utils.joint_densities(x, params, w_b).sum(axis=0)
        except Exception:
            continue
        for k, a in variants.items():
            est[k].append(weighted_prior_em(f_p, f_b, a))
    out = dict(dataset=ds, status="ok", nt_level=nt_level, strand=strand, benign_method=bm,
               prior_production=cal.get("prior"), n_missense=len(x), frac_unmapped=bad, ref_match=ref_match,
               frac_snv_reachable=float(reach[ok].mean()) if ok.any() else np.nan)
    for k, a in variants.items():
        v = np.asarray(est[k], float)
        out[f"prior_{k}"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        out[f"neff_{k}"] = float(a.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--cpg-level", type=int, default=2)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "analysis_v3" / "e8_mutrate_prior.csv"))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from analysis import config as _cfg
    from analysis.discovery import load_master_df, _filter_dataset_df
    from analysis.missense_sensitivity.compare_runs import _find_outputs
    fits = json.load(gzip.open(ROOT / "explorer_jobs_pp_large_baseline" / "bootstrap_results.json.gz"))
    datasets = sorted(args.datasets or fits)
    cals = {}
    for ds in datasets:
        c, _ = _find_outputs(ROOT / "calib_baseline", ds)
        if c is not None:
            cals[ds] = json.load(open(c))
    df_full = load_master_df(_cfg.DATASET_TSV)
    slices = {ds: _filter_dataset_df(df_full, ds, _cfg.DATASET_TSV) for ds in cals}
    del df_full
    if WINDOWS.exists():
        windows = pickle.load(open(WINDOWS, "rb"))
    else:
        regions = []
        for d in slices.values():
            for c, g in d.groupby("Chrom"):
                s = pd.to_numeric(g.hg38_start, errors="coerce").dropna()
                e = pd.to_numeric(g.hg38_end, errors="coerce").dropna()
                if len(s):
                    regions.append((str(c), int(s.min()), int(e.max())))
        windows = build_windows(regions)
        pickle.dump(windows, open(WINDOWS, "wb"))
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_task)(ds, slices[ds], fits[ds], cals[ds], windows, args.cpg_level) for ds in cals)
    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    ok = df[df.status == "ok"]
    pd.set_option("display.width", 250)
    print(f"{len(ok)} datasets (skipped: {(df.status != 'ok').sum()})")
    cols = [c for c in ok.columns if c.startswith("prior_")]
    print(ok[cols].describe(percentiles=[.25, .5, .75]).loc[["count", "25%", "50%", "75%"]].round(4).to_string())
    print(ok[["frac_unmapped", "frac_snv_reachable"] + [c for c in ok.columns if c.startswith("neff_")]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
