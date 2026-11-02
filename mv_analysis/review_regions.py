"""Region-specific calibration of assays by coverage (stage 1: no refit).

When a gene's assays each measure only part of the protein, one gene-wide
joint model is fit on a matrix with large blocks of missing readouts. This
maps, per gene, which readouts measured which residues, segments the protein
into regions with a near-constant readout set, and stratifies the current
gene-wide models' out-of-bag results by region and by the number of readouts
missing per control, to show where missingness costs evidence and which genes
merit region-specific joint fits (stage 2, cluster refits).

Coverage: from each readout's own input rows (the integrated dataframe, or
TP53's / TSC2's input tables), the number of variants it scored at each
residue. A readout covers a residue when it scored at least COVER_FRAC of the
variants scored there by the most complete readout. Adjacent residues with the
same covering set form a segment; segments shorter than MIN_SEG residues are
merged into the longer neighbour. Pfam domains (Chen et al.'s per-variant
table) are recorded per segment.

Outputs (review/region_calibration/): coverage_<model>.csv (residue x
readout counts), segments.csv, stratified.csv (per model and segment: controls,
joint out-of-bag and comparator MCC / determinate rate / errors),
by_missing.csv (per model and number of readouts measured), and
fig_coverage_<model>.png.

Usage: python -m mv_analysis.review_regions
"""
import functools
import re
import warnings

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("region_calibration")
COVER_FRAC = 0.5
MIN_SEG = 10
TP53_COLS = {"AIP1nWT_PC1": "WAF1nWT", "AIP1nWT_PC2": "WAF1nWT", "log_TempSens": "TempSens"}
TSC2_COLS = {"TSC2_Biar_2026_VAMPseq": "VAMPseq_ave", "TSC2_Biar_2026_immuneSGE": "immuneSGEscore",
             "TSC2_Biar_2026_cliPE": "cliPE_score"}


@functools.lru_cache(maxsize=1)
def _integrated():
    cols = ["Dataset", "Gene", "Chrom", "hg38_start", "ref_allele", "alt_allele", "aa_ref", "aa_pos", "aa_alt",
            "auth_reported_score"]
    return pd.read_csv(config.INTEGRATED_VARIANT_EFFECT_DATASET_PATH, sep="\t", usecols=cols, low_memory=False)


@functools.lru_cache(maxsize=1)
def _pfam():
    c = pd.read_csv(config.CHEN_PREDICTOR_TABLE, usecols=["gene_symbol", "protein_variant", "domain"], low_memory=False)
    c["pos"] = pd.to_numeric(c["protein_variant"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    c = c.dropna(subset=["pos"])
    return c.groupby(["gene_symbol", "pos"])["domain"].agg(lambda s: s.mode().iloc[0]).rename("domain")


def _norm(*xs):
    out = []
    for x in xs:
        try:
            f = float(x)
            out.append(str(int(f)) if f == int(f) else str(x))
        except (TypeError, ValueError):
            out.append(str(x))
    return ":".join(out)


def coverage(key, m):
    """(residue x readout counts DataFrame, {model row: residue})."""
    d, _ = R.prep(key)
    dims = [str(x) for x in d["dims"]]
    variants = [str(v) for v in d["variants"]]
    if m["gene_set"] == "tp53":
        t = pd.read_csv(config.TP53_ANNOTATED_VARIANTS_PATH)
        t = t[t["variant_class"] == "missense"]
        cov = pd.DataFrame({n: t.groupby("position")[TP53_COLS.get(n, n)].apply(lambda s: s.notna().sum())
                            for n in dims})
        pos = {v: int(re.search(r"\d+", v).group()) for v in variants if re.search(r"\d+", v)}
        return cov, pos
    if m["gene_set"] == "tsc2":
        from src.assay_calibration.multivariate_data import tsc2
        t = pd.read_csv(tsc2.DEFAULT_MAVE_CSV)
        t["pos"] = pd.to_numeric(t["hgvs_pro"].map(tsc2.protein_id).str.extract(r"(\d+)")[0], errors="coerce")
        cov = pd.DataFrame({n: t.groupby("pos")[c].apply(lambda s: s.notna().sum()) for n, c in TSC2_COLS.items()
                            if n in dims})
        pos = {v: int(re.search(r"\d+", v).group()) for v in variants if re.search(r"\d+", v)}
        return cov, pos
    df = _integrated()
    df = df[df["Gene"].astype(str).str.upper() == m["gene"].upper()]
    df = df[df["auth_reported_score"].notna() & df["aa_pos"].notna()]
    cov = {}
    for n in dims:
        sub = df[df["Dataset"].astype(str).str.startswith(n)]
        cov[n] = sub.groupby("aa_pos").size()
    cov = pd.DataFrame(cov).fillna(0)
    key_pos = dict(zip((_norm(*r) for r in df[["Gene", "Chrom", "hg38_start", "ref_allele", "alt_allele"]]
                        .itertuples(index=False)), df["aa_pos"]))
    import ast
    pos = {}
    for v in variants:
        try:
            k = _norm(*ast.literal_eval(v))
        except (ValueError, SyntaxError):
            continue
        if k in key_pos:
            pos[v] = int(key_pos[k])
    return cov, pos


def segments(cov, gene):
    """Segments of residues with the same covering readout set."""
    cov = cov.sort_index()
    covered = cov.ge(COVER_FRAC * cov.max(axis=1), axis=0) & (cov > 0)
    pats = covered.apply(lambda r: tuple(c for c in cov.columns if r[c]), axis=1)
    segs, start, cur = [], None, None
    for p, pat in pats.items():
        if pat != cur:
            if cur is not None:
                segs.append([start, prev, cur])
            start, cur = p, pat
        prev = p
    if cur is not None:
        segs.append([start, prev, cur])
    # merge short segments into the longer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b, pat) in enumerate(segs):
            if b - a + 1 < MIN_SEG:
                left = segs[i - 1] if i > 0 else None
                right = segs[i + 1] if i + 1 < len(segs) else None
                tgt = max([x for x in (left, right) if x], key=lambda s: s[1] - s[0])
                tgt[0], tgt[1] = min(tgt[0], a), max(tgt[1], b)
                segs.pop(i)
                changed = True
                break
    # neighbours left with the same readout set are one region
    merged = [segs[0]] if segs else []
    for a, b, pat in segs[1:]:
        if pat == merged[-1][2]:
            merged[-1][1] = b
        else:
            merged.append([a, b, pat])
    segs = merged
    pf = _pfam()
    out = []
    for i, (a, b, pat) in enumerate(segs):
        doms = [pf.get((gene, float(p))) for p in range(int(a), int(b) + 1)]
        doms = pd.Series([x for x in doms if isinstance(x, str) and x != "no_pfam"])
        out.append(dict(segment=i, start=int(a), end=int(b), readouts=";".join(pat), n_readouts=len(pat),
                        pfam=";".join(doms.value_counts().index[:3]) if len(doms) else ""))
    return pd.DataFrame(out)


def _metrics(pts, y):
    from mv_analysis.review_priors import _metrics as M
    return M(pts, y)


def run():
    warnings.filterwarnings("ignore")
    sp = R.strength_points()
    segs_all, strat, miss = [], [], []
    for key, m in R.paper_keys("functional"):
        if m["gene_set"] == "fgfr":
            continue
        d, _ = R.prep(key)
        cov, pos = coverage(key, m)
        if cov.empty:
            continue
        cov.to_csv(OUT / f"coverage_{key}.csv")
        sg = segments(cov, m["gene"].upper())
        sg.insert(0, "key", key)
        segs_all.append(sg)
        o = R.oob_points(key, 5)
        o = o[o["n_oob"] > 0]
        s = sp[sp["key"] == key].drop_duplicates("variant").set_index("variant")
        o["indep"] = o["variant"].map(s["indep_points"]) if len(s) else np.nan
        o["pos"] = o["variant"].map(pos)
        X = np.asarray(d["X"], float)
        o["n_measured"] = (~np.isnan(X[o["row"].to_numpy()])).sum(1)
        y = o["label"].to_numpy() == 1

        def seg_of(p):
            if not np.isfinite(p):
                return np.nan
            hit = sg[(sg["start"] <= p) & (sg["end"] >= p)]
            return hit["segment"].iloc[0] if len(hit) else np.nan
        o["segment"] = o["pos"].map(seg_of)
        for sid, x in o.groupby("segment"):
            yy = x["label"].to_numpy() == 1
            r = sg[sg["segment"] == sid].iloc[0]
            strat.append(dict(key=key, gene=m["gene"], segment=int(sid), start=r["start"], end=r["end"],
                              readouts=r["readouts"], n_controls=len(x), n_plp=int(yy.sum()),
                              n_blb=int((~yy).sum()), mean_measured=x["n_measured"].mean(),
                              **{f"joint_{k}": v for k, v in _metrics(x["points"], yy).items()
                                 if k in ("mcc", "det_rate", "errors")},
                              **{f"indep_{k}": v for k, v in _metrics(x["indep"], yy).items()
                                 if k in ("mcc", "det_rate", "errors")}))
        for nm, x in o.groupby("n_measured"):
            yy = x["label"].to_numpy() == 1
            miss.append(dict(key=key, gene=m["gene"], n_measured=int(nm), n_readouts=X.shape[1],
                             n_controls=len(x),
                             **{f"joint_{k}": v for k, v in _metrics(x["points"], yy).items()
                                if k in ("mcc", "det_rate", "errors")},
                             **{f"indep_{k}": v for k, v in _metrics(x["indep"], yy).items()
                                if k in ("mcc", "det_rate", "errors")}))
        _figure(key, cov, sg)
        print(key, len(sg), "segments", flush=True)
    pd.concat(segs_all).to_csv(OUT / "segments.csv", index=False)
    pd.DataFrame(strat).to_csv(OUT / "stratified.csv", index=False)
    pd.DataFrame(miss).to_csv(OUT / "by_missing.csv", index=False)


def _figure(key, cov, sg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = cov.sort_index()
    frac = cov.div(cov.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(cov.columns) + 1.2))
    ax.imshow(frac.T.to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=1,
              extent=[cov.index.min(), cov.index.max(), len(cov.columns), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(cov.columns)) + 0.5)
    ax.set_yticklabels([c[:40] for c in cov.columns], fontsize=6)
    for _, r in sg.iterrows():
        ax.axvline(r["start"], color="#d55e00", lw=0.6)
    ax.set_xlabel("residue")
    ax.set_title(f"{key}: readout coverage (fraction of the best-covered readout), segments in orange", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_coverage_{key}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
