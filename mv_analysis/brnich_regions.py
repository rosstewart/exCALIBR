"""Brnich et al. (2020) functional evidence from synonymous z-score regions,
per variant, for any MultiScoreset.

Each readout is z-scored against the gene's synonymous variants (mean and
SD of the synonymous sample). Over a set of readouts, a variant is in the
functionally NORMAL region when |z| <= 2 in every one of them, and in the
ABNORMAL region by one of two rules:
  "all"  |z| >= 3 in every readout
  "any"  |z| >= 3 in at least one readout (LABEL-seq main setting)
Anything else is intermediate (no evidence).

Readout subsets: a variant is judged on the readouts it was measured in, and
that set of readouts (its pattern) gets its own OddsPath, from the controls
measured in all of them, judged on those readouts only. A smaller pattern
can therefore carry weaker evidence than the full set.

OddsPath per pattern and class: the share of P/LP controls in the class over
the share of B/LB controls in it, with the ExCALIBR manuscript's pseudocount
rule by default (+1 benign in the abnormal class only when it has none, +1
pathogenic in the normal class only when it has none;
analysis.oddspath_computation.compute_oddspath.odds_path_from_counts).
Evidence strength by odds_to_evidence (Brnich/Tavtigian cut-offs), points by
the cockpit's map (+-1, 2, 4, 8); each class only in its own direction (an
abnormal class with OddsPath below the pathogenic cut-offs is Indeterminate,
never benign evidence, and vice versa). With 10 or fewer controls for the
pattern, evidence is capped at Supporting (Brnich et al. 2020). Not computable (NaN) when the pattern has no
P/LP or no B/LB controls, e.g. negative-unlabelled genes.
"""
import numpy as np
import pandas as pd

NORMAL_Z, ABNORMAL_Z = 2.0, 3.0
# Brnich et al. (2020): assays with 10 or fewer validation controls (P/LP +
# B/LB) give at most PS3_supporting / BS3_supporting.
MAX_CONTROLS_SUPPORTING = 10

# LABEL-seq readouts used for the Brnich regions: untreated abundance, HSP90i
# abundance and untreated activity (interaction, SerumStarve and the KRAS
# treatment assays left out). abundance_DMSO stands in for untreated
# abundance in genes without an untreated abundance readout (main setting).
LABELSEQ_READOUTS = ("abundance_No_treatment", "abundance_HSP90i", "activity_No_treatment")
LABELSEQ_DMSO = "abundance_DMSO"


def readout_name(dataset_name):
    """'braf_abundance_HSP90i' -> 'abundance_HSP90i'."""
    s = str(dataset_name)
    return s.split("_", 1)[1] if "_" in s else s


def labelseq_brnich_dims(readouts, include_dmso=True):
    """Indices into `readouts` (readout names, e.g. "abundance_HSP90i") used
    for the LABEL-seq Brnich regions: whichever of LABELSEQ_READOUTS the gene
    has; with include_dmso (the main setting), DMSO abundance is used as the
    untreated abundance when the gene has no untreated abundance readout
    (never alongside it)."""
    want = set(LABELSEQ_READOUTS)
    if include_dmso and "abundance_No_treatment" not in readouts:
        want.add(LABELSEQ_DMSO)
    return [j for j, n in enumerate(readouts) if n in want]


def synonymous_z(X, synonymous):
    """z-scores of every column of X against the synonymous rows (population
    SD); NaN for a column with fewer than two synonymous values or no spread."""
    X = np.asarray(X, float)
    Z = np.full(X.shape, np.nan)
    syn = np.asarray(synonymous, bool)
    for j in range(X.shape[1]):
        s = X[syn, j]
        s = s[~np.isnan(s)]
        if len(s) >= 2 and s.std() > 0:
            Z[:, j] = (X[:, j] - s.mean()) / s.std()
    return Z


def regions(Z, rule="all"):
    """Per row: "normal", "abnormal", "intermediate", or None (no readout)."""
    Z = np.asarray(Z, float)
    obs = ~np.isnan(Z)
    a = np.abs(np.where(obs, Z, 0.0))
    has = obs.any(1)
    normal = has & np.where(obs, a <= NORMAL_Z, True).all(1)
    if rule == "all":
        abnormal = has & np.where(obs, a >= ABNORMAL_Z, True).all(1)
    elif rule == "any":
        abnormal = has & (obs & (a >= ABNORMAL_Z)).any(1)
    else:
        raise ValueError(f"rule must be 'all' or 'any', not {rule!r}")
    out = np.where(normal, "normal", np.where(abnormal, "abnormal", "intermediate")).astype(object)
    out[~has] = None
    return out


def _counts(reg, plp, blb):
    return dict(n_plp=int(plp.sum()), n_blb=int(blb.sum()),
                plp_abnormal=int((plp & (reg == "abnormal")).sum()), plp_normal=int((plp & (reg == "normal")).sum()),
                blb_abnormal=int((blb & (reg == "abnormal")).sum()), blb_normal=int((blb & (reg == "normal")).sum()))


def computable_counts(stats):
    """Both control classes present (an OddsPath can be computed)."""
    return stats["n_plp"] > 0 and stats["n_blb"] > 0


def pattern_oddspath(reg, plp, blb, n_abnormal, n_normal, pseudocount="if_zero"):
    """{counts..., oddspath_abnormal, oddspath_normal, pseudocounts} for one
    pattern's controls (`reg` their regions); n_abnormal / n_normal: all
    variants of the pattern in each class (a class with none has no OddsPath,
    as compute_oddspath.odds_lr)."""
    from analysis.oddspath_computation.compute_oddspath import odds_path_from_counts
    c = _counts(reg, plp, blb)
    op_a, op_n, notes = odds_path_from_counts(c["plp_abnormal"], c["plp_normal"], c["blb_abnormal"],
                                              c["blb_normal"], c["n_plp"], c["n_blb"], pseudocount)
    return dict(c, oddspath_abnormal=op_a if n_abnormal else None, oddspath_normal=op_n if n_normal else None,
                pseudocounts="; ".join(notes))


def brnich_evidence(X, names, synonymous, plp, blb, rule="all", pseudocount="if_zero"):
    """Per-row Brnich evidence over the readouts in X (columns `names`).

    synonymous / plp / blb: boolean row masks (the z reference and the
    controls; blb may already be restricted, e.g. to missense). Returns
    (per-row DataFrame: readouts, region, oddspath, code, points;
     per-pattern DataFrame: counts and both OddsPaths)."""
    from analysis.oddspath_computation.compute_oddspath import odds_to_evidence
    from mv_analysis.mv_cockpit import _EVIDENCE_POINTS
    X = np.asarray(X, float)
    plp, blb = np.asarray(plp, bool), np.asarray(blb, bool)
    Z = synonymous_z(X, synonymous)
    obs = ~np.isnan(Z)
    pat = [tuple(names[j] for j in np.flatnonzero(o)) for o in obs]
    n = len(X)
    out = pd.DataFrame({"readouts": [";".join(p) if p else None for p in pat], "region": None,
                        "oddspath": np.nan, "code": None, "points": np.nan})
    pats = []
    for p in sorted({p for p in pat if p}, key=lambda t: (-len(t), t)):
        cols = [names.index(r) for r in p]
        cover = obs[:, cols].all(1)  # measured in every readout of the pattern
        reg = regions(Z[np.ix_(cover, cols)], rule)
        full = np.full(n, None, dtype=object)
        full[cover] = reg
        stats = pattern_oddspath(full[cover & (plp | blb)], plp[cover & (plp | blb)], blb[cover & (plp | blb)],
                                 int((full == "abnormal").sum()), int((full == "normal").sum()), pseudocount)
        pats.append(dict(readouts=";".join(p), n_readouts=len(p), n_variants_with_pattern=sum(q == p for q in pat),
                         n_measured=int(cover.sum()), **stats))
        mine = np.array([q == p for q in pat])
        r = full[mine]
        op = np.where(r == "abnormal", stats["oddspath_abnormal"] if stats["oddspath_abnormal"] is not None else np.nan,
                      np.where(r == "normal", stats["oddspath_normal"] if stats["oddspath_normal"] is not None
                               else np.nan, np.nan)).astype(float)
        computable = stats["n_plp"] > 0 and stats["n_blb"] > 0
        code = np.where(r == "intermediate", "Intermediate",
                        [odds_to_evidence(x) for x in op]).astype(object)
        # each class gives evidence only in its own direction: an abnormal
        # readout is never benign evidence (OddsPath < 1 there, e.g. no P/LP
        # control abnormal, is no evidence), a normal one never pathogenic
        wrong = ((r == "abnormal") & np.array([str(c).startswith("BS3") for c in code])) | \
                ((r == "normal") & np.array([str(c).startswith("PS3") for c in code]))
        code[wrong] = "Indeterminate"
        # 10 or fewer controls for this readout combination: Supporting at most
        capped = stats["n_plp"] + stats["n_blb"] <= MAX_CONTROLS_SUPPORTING
        if capped:
            code = np.array([("PS3_supporting" if str(c).startswith("PS3") else
                              "BS3_supporting" if str(c).startswith("BS3") else c) for c in code], dtype=object)
        pats[-1]["capped_supporting"] = bool(capped and computable_counts(stats))
        pts = np.array([0.0 if c in ("Intermediate", "Indeterminate") else
                        float(_EVIDENCE_POINTS[c]) if c in _EVIDENCE_POINTS else np.nan for c in code])
        if not computable:
            code[:] = "Not computable"
            pts[:] = np.nan
            op[:] = np.nan
        out.loc[mine, "region"] = r
        out.loc[mine, "oddspath"] = op
        out.loc[mine, "code"] = code
        out.loc[mine, "points"] = pts
    return out, pd.DataFrame(pats)


# ---------------------------------------------------------------------------
# Per-readout Brnich with the strongest readout per gene (LABEL-seq main
# method; replaces the multi-readout regions above, which are kept for the
# archived outputs). Following Calhoun et al. (2026, Genome Medicine): assess
# each readout on its own and use the best-performing one, judged by the
# evidence its OddsPaths reach; variants not measured in it fall back to the
# next-ranked readout they were measured in.
# ---------------------------------------------------------------------------
def _bin_points(op, direction, n_controls):
    """Points one bin gives: odds_to_evidence of its OddsPath, only in its own
    direction ("abnormal" pathogenic, "normal" benign), Supporting at most
    with MAX_CONTROLS_SUPPORTING or fewer controls."""
    from analysis.oddspath_computation.compute_oddspath import odds_to_evidence
    from mv_analysis.mv_cockpit import _EVIDENCE_POINTS
    if op is None or not np.isfinite(op):
        return 0
    code = odds_to_evidence(op)
    pts = _EVIDENCE_POINTS.get(code, 0)
    if (direction == "abnormal" and pts < 0) or (direction == "normal" and pts > 0):
        return 0
    return int(np.sign(pts)) if n_controls <= MAX_CONTROLS_SUPPORTING else int(pts)


def per_readout_brnich(X, names, synonymous, plp, blb, pseudocount="if_zero"):
    """Brnich evidence from each readout on its own, applied per variant from
    the gene's strongest readout it was measured in.

    Each readout: z-scores against the synonymous rows, bins normal / abnormal
    / intermediate (one readout, so the any / all rules coincide), and its own
    OddsPaths from the controls measured in it (brnich_evidence on that
    column). Readouts are ranked by: (1) 11 or more controls, (2) the evidence
    strength their bins reach (abnormal points + |normal points|), (3) the
    number of variants the readout gives evidence to, (4) the OddsPaths
    themselves (log OddsPath of each bin in its own direction), (5) number of
    controls. A readout without both P/LP and B/LB controls is not
    used.

    Returns (per-row DataFrame: readout, bin, oddspath, code, points,
    readout_points {readout: points} over every usable readout the row was
    measured in; per-readout DataFrame: counts, OddsPaths, bin points, rank)."""
    import json
    X = np.asarray(X, float)
    n = len(X)
    per, stats = {}, []
    for j, name in enumerate(names):
        ev, pt = brnich_evidence(X[:, [j]], [name], synonymous, plp, blb, rule="any", pseudocount=pseudocount)
        if not len(pt):
            continue
        s = pt.iloc[0].to_dict()
        n_ctrl = int(s["n_plp"] + s["n_blb"])
        usable = s["n_plp"] > 0 and s["n_blb"] > 0
        op_a, op_n = s.get("oddspath_abnormal"), s.get("oddspath_normal")
        op_a = None if op_a is None or pd.isna(op_a) else float(op_a)
        op_n = None if op_n is None or pd.isna(op_n) else float(op_n)
        pa, pn = (_bin_points(op_a, "abnormal", n_ctrl), _bin_points(op_n, "normal", n_ctrl)) if usable else (0, 0)
        lo = (np.log(op_a) if op_a and op_a > 1 else 0.0) + (-np.log(op_n) if op_n and 0 < op_n < 1 else 0.0)
        n_ev = int((ev["points"].fillna(0) != 0).sum()) if usable else 0
        stats.append(dict(readout=name, n_measured=int((~np.isnan(X[:, j])).sum()), n_plp=int(s["n_plp"]),
                          n_blb=int(s["n_blb"]), plp_abnormal=s["plp_abnormal"], plp_normal=s["plp_normal"],
                          blb_abnormal=s["blb_abnormal"], blb_normal=s["blb_normal"], oddspath_abnormal=op_a,
                          oddspath_normal=op_n, points_abnormal=pa, points_normal=pn, n_with_evidence=n_ev,
                          usable=bool(usable),
                          capped_supporting=bool(usable and n_ctrl <= MAX_CONTROLS_SUPPORTING),
                          _key=(usable, n_ctrl > MAX_CONTROLS_SUPPORTING, pa + abs(pn), n_ev, lo, n_ctrl)))
        per[name] = ev
    st = pd.DataFrame(stats)
    out = pd.DataFrame({"readout": None, "bin": None, "oddspath": np.nan, "code": None, "points": np.nan,
                        "readout_points": None}, index=range(n))
    if st.empty or not st["usable"].any():
        out["code"] = "Not computable"
        return out, st.drop(columns="_key", errors="ignore")
    order = sorted(range(len(st)), key=lambda i: st.loc[i, "_key"], reverse=True)
    st = st.loc[order].reset_index(drop=True)
    st["rank"] = np.where(st["usable"], np.arange(1, len(st) + 1), np.nan)
    ranked = [r for r, u in zip(st["readout"], st["usable"]) if u]
    chosen = np.full(n, None, dtype=object)
    for r in ranked:
        measured = ~np.isnan(X[:, names.index(r)])
        take = measured & (chosen == None)  # noqa: E711
        chosen[take] = r
    for r in ranked:
        rows = chosen == r
        if rows.any():
            ev = per[r]
            out.loc[rows, "readout"] = r
            out.loc[rows, "bin"] = ev.loc[rows, "region"].to_numpy()
            out.loc[rows, "oddspath"] = ev.loc[rows, "oddspath"].to_numpy()
            out.loc[rows, "code"] = ev.loc[rows, "code"].to_numpy()
            out.loc[rows, "points"] = ev.loc[rows, "points"].to_numpy()
    out.loc[chosen == None, "code"] = "Not measured"  # noqa: E711
    pts = {r: per[r]["points"].to_numpy(float) for r in ranked}
    out["readout_points"] = [json.dumps({r: int(pts[r][i]) for r in ranked if np.isfinite(pts[r][i])}) or None
                             for i in range(n)]
    return out, st.drop(columns="_key")
