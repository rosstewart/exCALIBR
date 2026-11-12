"""Consequence-aware simulation: generative model, true per-consequence priors and LRs, and the design grid.

Per variant (see the schematic, analysis/consequence_cells_sim/schematic.py):
    consequence c in {synonymous, missense, PTV}
    SpliceAI flag   hi ~ Bernoulli(h_c)          (max of DS_AG/AL/DG/DL >= 0.2, the production filter; only the
                                                  threshold matters here)
    splice disrupt  d  ~ Bernoulli(d_hi_c if hi else d_lo)        (d_lo: disruption the filter cannot see)
    splice-pathogenic   sp ~ d * Bernoulli(q_sp)  (default q_sp = 1: a disruption the assay detects is pathogenic;
                                                  q_sp < 1 is a sensitivity slice for tolerated disruptions)
    protein-pathogenic  pp ~ Bernoulli(pi_prot_c), with mechanism m (missense: stability / activity / other;
                        PTV: LoF (tau_ptv) / partial; synonymous: regulatory)
    pathogenic = sp or pp
    functionally abnormal in this assay:
        fitness (SGE-like, endogenous locus, measures splicing):   d or (pp and m in {stability, activity, LoF})
        abundance (VAMP-like, cDNA, no splicing):                   pp and m in {stability, LoF}
      (hypomorph slice: missense m=partial and PTV partial score as intermediate)
    score: abnormal N(-kappa, 1); intermediate N(-kappa/2, 1); normal N(0, 1)
Production splice handling: abundance assays drop rows with hi (SpliceAI > 0.2) before fitting; fitness assays keep
them (the cells arm gives them their own cells). Truth is defined on the kept population of each consequence.

Every probability that enters the truth is a finite enumeration over (hi, d, sp, pp, m), so the true prior pi_c, the
true abnormal / intermediate shares of the pathogenic and benign densities, and LR_c(x) are exact.

Parameter defaults: DEFAULTS below are anchored on step 0 (anchors.py; step0/summary.csv). defaults.json in step0/,
if present, overrides them (for re-anchoring without code changes).
"""
from __future__ import annotations

import itertools
import json
import os
import zlib

import numpy as np
import scipy.stats as sps

NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous"]
CONSEQUENCES = ("synonymous", "missense", "PTV")
CSQ_LABEL = {"synonymous": "synonymous_variant", "missense": "missense_variant", "PTV": "stop_gained"}
N_EVAL = 10_000
ANCHOR_FILE = "/data/ross/assay_calibration/consequence_cells_sim/step0/defaults.json"

DEFAULTS = dict(
    # Anchored on step 0 (anchors.py -> step0/summary.csv, 2026-11-10); comments give the anchor.
    kappa=3.5, assay="fitness", hypomorph=False, shape="gauss",
    # SpliceAI max DS > 0.2 share among assayed variants (splice-measuring manuscript datasets: syn 2.1%, mis 3.4%)
    h=dict(synonymous=0.021, missense=0.034, PTV=0.03),
    # P(disruption | max DS >= 0.2), per consequence: excess pathogenic-evidence rate of population/synonymous rows
    # in splice-measuring manuscript datasets, (a - b) / (1 - b) with a, b the rates at DS >= / < 0.2:
    # synonymous 0.158 vs 0.014 (n = 203) -> 0.146; missense 0.305 vs 0.097 (n = 269) -> 0.230; PTV n = 11, so the
    # missense value is used (PTVs are mostly abnormal through truncation anyway)
    d_hi=dict(synonymous=0.146, missense=0.23, PTV=0.23),
    d_lo=0.0,       # synonymous DS < 0.2: 1.4% pathogenic evidence, the B/LB false-positive floor -> no leakage
    q_sp=1.0,       # a disruption the assay detects is pathogenic (q_sp = 0.5 slice: tolerated disruptions)
    eta_b=0.0,      # share of B/LB that are truly pathogenic (uniform label noise; slice 0.05 / 0.15)
    # B/LB synonymous labels are blind to splicing ("synonymous, so protein intact, so benign"):
    # P(labelled LB | splice-pathogenic synonymous) / P(labelled LB | benign synonymous). Anchor: in splice-measuring
    # assays B/LB synonymous have DS >= 0.2 as often as population synonymous (2.6% vs 2.1%) and score abnormal at
    # DS >= 0.2 as often (18.8% vs 15.8%), so labels do not screen out splice variants -> 1 (slice 0: clean labels)
    rho_syn_blb=1.0,
    # protein-mechanism pathogenic share: missense = SGE median missense prior 0.090 (IQR 0.040-0.136; Pejaver 2022
    # 0.0441 for missense in disease genes sits near q25); PTV priors saturate near 1 in most datasets
    pi_prot=dict(synonymous=0.0, missense=0.09, PTV=0.90),
    # missense mechanisms: P/LP missense pathogenic-evidence share is 0.79 in SGE and 0.42 in VAMP-seq (ratio 0.54);
    # gamma_fitness = stability + activity = 0.85, gamma_abundance = stability = 0.45 (ratio 0.53)
    mis_mech=dict(stability=0.45, activity=0.40, other=0.15),
    tau_ptv=0.95,   # P/LP PTV pathogenic-evidence share: SGE 0.95 (VAMP-seq 0.80)
    nu=0.3,         # hypomorph slice only
    # population consequence shares: median over the large-cell datasets (missense 0.75, synonymous 0.22, PTV 0.02)
    pop_share=dict(synonymous=0.22, missense=0.75, PTV=0.03),
    # control mixes among the modelled consequences: P/LP PTV 0.31, synonymous 0.001; B/LB synonymous 0.91
    plp_mix=dict(synonymous=0.002, missense=0.688, PTV=0.31),
    blb_mix=dict(synonymous=0.91, missense=0.09, PTV=0.0),
    syn_overlap=0.5,  # share of the synonymous sample that is also in the population sample
    n_plp=40, n_blb=60, n_pop=500, n_syn=150,
)
# population sizes: median 530 (IQR 158-708, max 2707) gnomAD rows per dataset in the cell tables
SIZES = {"small": (10, 15, 150, 50), "mid": (40, 60, 500, 150), "large": (160, 240, 2000, 600)}
MIXES = {"balanced": (0.50, 0.10), "realistic": (0.91, 0.31), "extreme": (0.97, 0.50)}  # (B/LB syn share, P/LP PTV)
KAPPAS = (2.0, 3.5, 5.0)  # kappa proxy (biased high: P/LP with evidence) medians SGE 5.6, VAMP 6.7, other 4.2
PI_MIS = (0.0441, 0.09, 0.136)  # Pejaver 2022; SGE median; SGE q75


def _anchored_defaults():
    d = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(ANCHOR_FILE):
        for k, v in json.load(open(ANCHOR_FILE)).items():
            d[k] = v
    return d


# ── cells of the design ──────────────────────────────────────────────────────────────────────────────────────────

def make_cell(experiment, size="mid", mix="realistic", **over):
    c = _anchored_defaults()
    c["n_plp"], c["n_blb"], c["n_pop"], c["n_syn"] = SIZES[size]
    syn_b, ptv_p = MIXES[mix]
    c["blb_mix"] = dict(synonymous=syn_b, missense=1 - syn_b, PTV=0.0)
    rest = 1 - ptv_p - c["plp_mix"]["synonymous"]
    c["plp_mix"] = dict(synonymous=c["plp_mix"]["synonymous"], missense=rest, PTV=ptv_p)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k] = {**c[k], **v}
        else:
            c[k] = v
    c.update(experiment=experiment, size=size, mix=mix)
    c["cell_id"] = cell_id(c, over)
    return c


def cell_id(c, over):
    s = f"{c['experiment']}__{c['assay']}_{c['size']}_{c['mix']}_k{c['kappa']:g}"
    for k in sorted(over):
        if k in ("kappa", "assay"):
            continue
        v = over[k]
        s += f"_{k}" + ("-".join(f"{kk}{vv:g}" for kk, vv in sorted(v.items())) if isinstance(v, dict) else f"{v:g}"
                        if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{v}")
    return s


def design():
    cells = []
    for kappa, size, mix, assay in itertools.product(KAPPAS, SIZES, MIXES, ("fitness", "abundance")):
        cells.append(make_cell("core", size, mix, kappa=kappa, assay=assay))
    for assay in ("fitness", "abundance"):
        sl = lambda **o: cells.append(make_cell("slice", "mid", "realistic", assay=assay, **o))
        for p in PI_MIS:
            if p != DEFAULTS["pi_prot"]["missense"]:
                sl(pi_prot=dict(missense=p))
        sl(tau_ptv=0.8)                           # VAMP-seq PTV evidence share
        if assay == "fitness":
            sl(q_sp=0.5)                          # tolerated disruptions: abnormal in the assay but benign
        for dlo in (0.005, 0.02):                 # disruption the SpliceAI filter misses
            sl(d_lo=dlo)
        for eb in (0.05, 0.15):                   # uniform B/LB label noise
            sl(eta_b=eb)
        if assay == "fitness":
            sl(rho_syn_blb=0.0)                   # clean synonymous B/LB labels (default: splice-blind)
            # missense measured share in fitness assays (default 0.85): BRCA/TP53-like 0.95, VHL/DDX3X-like 0.6
            # (real SGE P/LP missense evidence shares 0.22-1.00, median 0.89); stability:activity kept at 0.45:0.40
            for g in (0.95, 0.6):
                st = round(g * 0.45 / 0.85, 4)
                sl(mis_mech=dict(stability=st, activity=round(g - st, 4), other=round(1 - g, 4)))
        sl(hypomorph=True, nu=0.3)
        sl(n_blb=0)                               # PU-like: no B/LB
    sl_mis = make_cell("misfit", "mid", "realistic", assay="fitness", shape="skew")
    cells.append(sl_mis)
    seen, out = set(), []
    for c in cells:
        if c["cell_id"] not in seen:
            seen.add(c["cell_id"]); out.append(c)
    return out


def rng_for(cell, rep):
    return np.random.default_rng([zlib.crc32(cell["cell_id"].encode()), int(rep)])


# ── exact truth by enumeration ───────────────────────────────────────────────────────────────────────────────────

def _mechanisms(c, csq):
    """(mechanism, probability given protein-pathogenic) and its score class per assay type."""
    if csq == "missense":
        mm = dict(c["mis_mech"])
        if c.get("hypomorph"):
            mm = {k: v * (1 - c["nu"]) for k, v in mm.items()}; mm["partial"] = c["nu"]
        return mm
    if csq == "PTV":
        return {"LoF": c["tau_ptv"], "partial": 1 - c["tau_ptv"]}
    return {"regulatory": 1.0}


def _protein_class(m, assay, hypomorph):
    """Score class from the protein mechanism alone: 'abn', 'int' (intermediate) or 'nor'."""
    if m == "LoF" or (m == "stability"):
        return "abn"
    if m == "activity":
        return "abn" if assay == "fitness" else "nor"
    if m == "partial":
        return "int" if hypomorph else "nor"
    return "nor"


def enumerate_states(c, csq):
    """All latent states (prob, hi, pathogenic, score class) for a consequence, before filtering."""
    h, out = c["h"][csq], []
    mech = _mechanisms(c, csq)
    for hi in (True, False):
        ph = h if hi else 1 - h
        pd_ = (c["d_hi"][csq] if isinstance(c["d_hi"], dict) else c["d_hi"]) if hi else c["d_lo"]
        for d in (True, False):
            p_d = pd_ if d else 1 - pd_
            for sp in ((True, False) if d else (False,)):
                p_sp = (c["q_sp"] if sp else 1 - c["q_sp"]) if d else 1.0
                for pp in (True, False):
                    p_pp = c["pi_prot"][csq] if pp else 1 - c["pi_prot"][csq]
                    for m, p_m in (mech.items() if pp else [(None, 1.0)]):
                        cls = _protein_class(m, c["assay"], c.get("hypomorph")) if pp else "nor"
                        if c["assay"] == "fitness" and d:
                            cls = "abn"                  # splicing is measured: disruption scores as LoF
                        out.append((ph * p_d * p_sp * p_pp * p_m, hi, sp or pp, cls))
    return out


def kept(c, hi):
    return not (c["assay"] == "abundance" and hi)     # production removes SpliceAI > 0.2 in non-splice assays


def truth(c, csq):
    """Exact prior and class-conditional score-class shares in the kept population of one consequence."""
    st = [s for s in enumerate_states(c, csq) if kept(c, s[1])]
    tot = sum(s[0] for s in st)
    pp = sum(s[0] for s in st if s[2])
    sh = lambda path, cls: sum(s[0] for s in st if s[2] == path and s[3] == cls)
    pi = pp / tot
    P = {k: (sh(True, k) / pp if pp > 0 else 0.0) for k in ("abn", "int", "nor")}
    B = {k: sh(False, k) / (tot - pp) for k in ("abn", "int", "nor")}
    return dict(pi=pi, P=P, B=B)


def _comp_logpdf(x, c):
    k = c["kappa"]
    if c.get("shape") == "skew":   # misfit slice: standardized skew-normal components (a = 4), same locations
        a = 4.0
        dl = a / np.sqrt(1 + a * a); mu, sd = dl * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * dl * dl / np.pi)
        f = lambda loc: sps.skewnorm.logpdf(x, a, loc=loc - mu / sd, scale=1 / sd)
        return dict(abn=f(-k), int=f(-k / 2), nor=f(0.0))
    return dict(abn=sps.norm.logpdf(x, -k, 1), int=sps.norm.logpdf(x, -k / 2, 1), nor=sps.norm.logpdf(x, 0, 1))


def _mix_logpdf(x, c, shares):
    lc = _comp_logpdf(x, c)
    terms = [np.log(v) + lc[k] for k, v in shares.items() if v > 0]
    if not terms:            # no pathogenic variants in this class (pi_c = 0): LR is 1 and the posterior is 0
        return lc["nor"]
    return np.logaddexp.reduce(np.vstack(terms), axis=0)


def true_log_lr(x, c, csq):
    t = truth(c, csq)
    return _mix_logpdf(x, c, t["P"]) - _mix_logpdf(x, c, t["B"])


def true_component_params(c):
    """True components in production (skew, loc, scale) format, abnormal first."""
    k = c["kappa"]
    comps = [(0.0, -k, 1.0)] + ([(0.0, -k / 2, 1.0)] if c.get("hypomorph") else []) + [(0.0, 0.0, 1.0)]
    return comps


def n_components(c):
    return 3 if c.get("hypomorph") else 2


# ── sampling ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _draw(rng, c, csq, n, path=None, path_weight=None):
    """n kept variants of one consequence, optionally conditioned on true pathogenicity (exact state distribution).
    path_weight: instead of conditioning, keep all states and down-weight pathogenic ones by this factor (label
    blindness: B/LB synonymous with rho_syn_blb). Returns score, hi, pathogenic, score class."""
    st = [s for s in enumerate_states(c, csq) if kept(c, s[1]) and (path is None or s[2] == path) and s[0] > 0]
    if n == 0 or not st:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, "<U3")
    p = np.array([s[0] * (path_weight if (path_weight is not None and s[2]) else 1.0) for s in st]); p = p / p.sum()
    idx = rng.choice(len(st), size=n, p=p)
    hi = np.array([st[i][1] for i in idx]); pth = np.array([st[i][2] for i in idx]); cls = np.array([st[i][3] for i in idx])
    k = c["kappa"]
    if c.get("shape") == "skew":
        a = 4.0; dl = a / np.sqrt(1 + a * a); mu, sd = dl * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * dl * dl / np.pi)
        z = (sps.skewnorm.rvs(a, size=n, random_state=rng) - mu) / sd
    else:
        z = rng.standard_normal(n)
    loc = np.where(cls == "abn", -k, np.where(cls == "int", -k / 2, 0.0))
    return loc + z, hi, pth, cls


def _split(rng, n, mix):
    ks = [k for k in CONSEQUENCES if mix.get(k, 0) > 0]
    p = np.array([mix[k] for k in ks]); p = p / p.sum()
    cnt = rng.multinomial(n, p) if n else np.zeros(len(ks), int)
    return dict(zip(ks, cnt))


def make_data(rng, c):
    """Rows of one replicate with multi-label membership (P/LP, B/LB, population, Synonymous), consequence,
    SpliceAI flag and (for diagnostics only) the true status; plus per-consequence population evaluation draws."""
    rows = dict(score=[], mask=[], csq=[], hi=[], path=[])

    def add(sc, hi, pth, csq, m):
        rows["score"].append(sc); rows["hi"].append(hi); rows["path"].append(pth)
        rows["csq"].append(np.full(len(sc), csq)); rows["mask"].append(np.tile(m, (len(sc), 1)))

    for csq, n in _split(rng, c["n_plp"], c["plp_mix"]).items():
        s, hi, pth, _ = _draw(rng, c, csq, n, path=True); add(s, hi, pth, csq, [1, 0, 0, 0])
    for csq, n in _split(rng, c["n_blb"], c["blb_mix"]).items():
        n_noise = rng.binomial(n, c.get("eta_b", 0.0)) if n else 0
        if csq == "synonymous" and c.get("rho_syn_blb", 0.0) > 0:   # splice-blind synonymous labels
            s, hi, pth, _ = _draw(rng, c, csq, n - n_noise, path_weight=c["rho_syn_blb"])
        else:
            s, hi, pth, _ = _draw(rng, c, csq, n - n_noise, path=False)
        add(s, hi, pth, csq, [0, 1, 0, 0])
        if n_noise:
            s, hi, pth, _ = _draw(rng, c, csq, n_noise, path=True); add(s, hi, pth, csq, [0, 1, 0, 0])
    n_ov = int(round(c["syn_overlap"] * c["n_syn"]))
    for csq, n in _split(rng, c["n_pop"], c["pop_share"]).items():
        s, hi, pth, _ = _draw(rng, c, csq, n)
        m = np.zeros((n, 4), int); m[:, 2] = 1
        if csq == "synonymous":
            m[:min(n_ov, n), 3] = 1                       # shared population + synonymous-sample rows
            n_ov -= min(n_ov, n)
        rows["score"].append(s); rows["hi"].append(hi); rows["path"].append(pth)
        rows["csq"].append(np.full(n, csq)); rows["mask"].append(m)
    n_syn_only = c["n_syn"] - int(round(c["syn_overlap"] * c["n_syn"])) + n_ov
    s, hi, pth, _ = _draw(rng, c, "synonymous", n_syn_only); add(s, hi, pth, "synonymous", [0, 0, 0, 1])
    data = {k: np.concatenate(v) for k, v in rows.items() if k != "mask"}
    data["mask"] = np.vstack(rows["mask"]).astype(bool)
    ev = {}
    for csq in CONSEQUENCES:
        s, hi, pth, _ = _draw(rng, c, csq, N_EVAL)
        ev[csq] = dict(x=s, is_path=pth)
    data["eval"] = ev
    return data
