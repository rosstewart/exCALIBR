"""Calibration-quality simulation: design grid and data generation.

Generative model (one assay, low score = abnormal, unit-variance normals):
    benign variants                      N(0, 1)
    pathogenic, measured mechanism       N(-kappa, 1)   (a fraction gamma of pathogenic variants)
    pathogenic, unmeasured mechanism     N(0, 1)        (score like benign variants)
so p1 = gamma N(-kappa, 1) + (1 - gamma) N(0, 1) and p0 = N(0, 1). This is the prior simulation's assay A1 with
gamma = the population M1 fraction (gamma = 1 reads every pathogenic variant, like A3 with uncorrelated
mechanisms), so a two-component mixture is the true model family.

Samples: P/LP (pathogenic with the mechanism mix lab_gamma, plus a fraction eps_p of benign variants), B/LB (benign
plus a fraction eta_b of pathogenic variants), population (prior pi) and synonymous (benign).

Cells: a crossed core grid (controls per class x population/synonymous richness x separation x gamma x prior) and
one-factor slices (asymmetric controls, mislabeled controls, P/LP mechanism-mix bias). Every cell has a stable id;
replicate r of a cell draws from default_rng([crc32(cell id), r]), so results do not depend on the grid order or on
which worker runs them, and any replicate can be regenerated (the acmgscaler arm does this).
"""
from __future__ import annotations

import itertools
import zlib

import numpy as np
import scipy.stats as sps

NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous"]
RICHNESS = {"low": (50, 0), "mid": (200, 100), "high": (1000, 500)}  # (population, synonymous)
CORE_N = (5, 10, 20, 40, 80)
CORE_KAPPA = (1.5, 2.5, 4.0)
CORE_GAMMA = (0.25, 0.5, 1.0)
CORE_PI = (0.10, 0.25)
SLICE_N = (5, 20, 80)
SLICE_KAPPA = (2.5, 4.0)
SLICE_GAMMA = (0.25, 1.0)
N_EVAL = 10_000  # held-out draws per evaluation class


def _cell(experiment, n_plp, n_blb, richness, kappa, gamma, pi, lab_gamma=None, eps_p=0.0, eta_b=0.0,
          n_pop=None, n_syn=None, shape="gauss", shape_par=0.0):
    pop, syn = RICHNESS[richness]
    c = dict(experiment=experiment, n_plp=int(n_plp), n_blb=int(n_blb), richness=richness,
             n_pop=int(pop if n_pop is None else n_pop), n_syn=int(syn if n_syn is None else n_syn),
             kappa=float(kappa), gamma=float(gamma), lab_gamma=float(gamma if lab_gamma is None else lab_gamma),
             pi=float(pi), eps_p=float(eps_p), eta_b=float(eta_b), shape=shape, shape_par=float(shape_par))
    c["cell_id"] = cell_id(c)
    return c


def cell_id(c):
    s = (f"{c['experiment']}__plp{c['n_plp']}_blb{c['n_blb']}_pop{c['n_pop']}_syn{c['n_syn']}"
         f"__k{c['kappa']:g}_g{c['gamma']:g}_lg{c['lab_gamma']:g}_pi{c['pi']:g}"
         f"_ep{c['eps_p']:g}_eb{c['eta_b']:g}")
    if c.get("shape", "gauss") != "gauss":  # Gaussian cells keep their original ids (and so their seeds)
        s += f"__{c['shape']}{c['shape_par']:g}"
    return s


# Deliberately misspecified truths (fit with skew-locked Gaussian components, so the model cannot match them):
#   skew     every class's scores are skew-normal with shape parameter a = shape_par, standardized to mean 0 and
#            variance 1 before shifting, so means and variances match the Gaussian cells
#   bimodal  the measured-mechanism abnormal class is an equal mixture of N(-kappa -+ shape_par, 1)
#   heavy    every class's scores are Student t with shape_par degrees of freedom (scale 1) around its mean
MISFIT = {"skew": (2.0, 4.0, 8.0, 16.0), "bimodal": (1.0, 2.0, 3.0, 4.0, 6.0), "heavy": (1.5, 2.0, 3.0, 5.0)}
MISFIT_N = (10, 40, 160)


def _skew_params(a):
    d = a / np.sqrt(1 + a * a)
    scale = 1 / np.sqrt(1 - 2 * d * d / np.pi)
    return a, -scale * d * np.sqrt(2 / np.pi), scale


def _draw_normal(rng, n, c):
    if c.get("shape") == "heavy":
        return sps.t.rvs(c["shape_par"], 0.0, 1.0, size=n, random_state=rng)
    if c.get("shape") == "skew":
        a, loc, sc = _skew_params(c["shape_par"])
        return sps.skewnorm.rvs(a, loc, sc, size=n, random_state=rng)
    return rng.normal(0.0, 1.0, n)


def _draw_abnormal(rng, n, c):
    k = c["kappa"]
    if c.get("shape") == "heavy":
        return sps.t.rvs(c["shape_par"], -k, 1.0, size=n, random_state=rng)
    if c.get("shape") == "skew":
        a, loc, sc = _skew_params(c["shape_par"])
        return sps.skewnorm.rvs(a, loc - k, sc, size=n, random_state=rng)
    if c.get("shape") == "bimodal":
        side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        return rng.normal(-k + side * c["shape_par"], 1.0, n)
    return rng.normal(-k, 1.0, n)


def _pdf_normal(x, c):
    if c.get("shape") == "heavy":
        return sps.t.pdf(x, c["shape_par"], 0.0, 1.0)
    if c.get("shape") == "skew":
        a, loc, sc = _skew_params(c["shape_par"])
        return sps.skewnorm.pdf(x, a, loc, sc)
    return sps.norm.pdf(x)


def _pdf_abnormal(x, c):
    k = c["kappa"]
    if c.get("shape") == "heavy":
        return sps.t.pdf(x, c["shape_par"], -k, 1.0)
    if c.get("shape") == "skew":
        a, loc, sc = _skew_params(c["shape_par"])
        return sps.skewnorm.pdf(x, a, loc - k, sc)
    if c.get("shape") == "bimodal":
        d = c["shape_par"]
        return 0.5 * sps.norm.pdf(x, -k - d) + 0.5 * sps.norm.pdf(x, -k + d)
    return sps.norm.pdf(x, -k)


def design():
    """All cells: core crossed grid plus the one-factor slices."""
    cells = [_cell("core", n, n, r, k, g, pi)
             for n, r, k, g, pi in itertools.product(CORE_N, RICHNESS, CORE_KAPPA, CORE_GAMMA, CORE_PI)]
    for n, k, g in itertools.product(SLICE_N, SLICE_KAPPA, SLICE_GAMMA):
        # Asymmetric controls: few P/LP with many B/LB and the reverse, no B/LB (synonymous reference), and no
        # benign reference at all (PU: no B/LB, no synonymous).
        cells += [_cell("asym_plp_few", max(n // 4, 2), n, "mid", k, g, 0.25),
                  _cell("asym_blb_few", n, max(n // 4, 2), "mid", k, g, 0.25),
                  _cell("no_blb", n, 0, "mid", k, g, 0.25),
                  _cell("pu", n, 0, "mid", k, g, 0.25, n_syn=0)]
        for e in (0.05, 0.10, 0.20):
            cells += [_cell("noise_plp", n, n, "mid", k, g, 0.25, eps_p=e),
                      _cell("noise_blb", n, n, "mid", k, g, 0.25, eta_b=e)]
        # Priors outside the core grid (shows where a fixed prior, e.g. acmgscaler's 0.10, fails).
        cells += [_cell("prior", n, n, "mid", k, g, pi) for pi in (0.05, 0.40)]
        if g < 1:  # mechanism-mix bias only exists when some mechanisms are unmeasured
            for lg in (0.0, 0.5, 0.75, 1.0):
                if lg != g:
                    cells.append(_cell("mech_bias", n, n, "mid", k, g, 0.25, lab_gamma=lg))
    for (shape, pars), n, g in itertools.product(MISFIT.items(), MISFIT_N, (1.0, 0.5)):
        cells += [_cell("misfit", n, n, "mid", 4.0, g, 0.25, shape=shape, shape_par=a) for a in pars]
    seen, out = set(), []
    for c in cells:  # drop duplicate ids (e.g. asymmetric cells that coincide at small n)
        if c["cell_id"] not in seen:
            seen.add(c["cell_id"])
            out.append(c)
    return out


def rng_for(cell, rep):
    return np.random.default_rng([zlib.crc32(cell["cell_id"].encode()), int(rep)])


def _pathogenic(rng, n, gamma, c):
    measured = rng.random(n) < gamma
    return np.where(measured, _draw_abnormal(rng, n, c), _draw_normal(rng, n, c))


def _mixed(rng, n, frac_path, gamma, c):
    is_path = rng.random(n) < frac_path
    return np.where(is_path, _pathogenic(rng, n, gamma, c), _draw_normal(rng, n, c)), is_path


def make_data(rng, c):
    """One replicate: the four samples (scores) and held-out evaluation draws. For Gaussian cells the random
    draws are exactly those of the original generator (same calls in the same order), so replicates regenerate."""
    plp_benign = rng.random(c["n_plp"]) < c["eps_p"]
    plp = np.where(plp_benign, _draw_normal(rng, c["n_plp"], c), _pathogenic(rng, c["n_plp"], c["lab_gamma"], c))
    blb, _ = _mixed(rng, c["n_blb"], c["eta_b"], c["gamma"], c)
    pop, _ = _mixed(rng, c["n_pop"], c["pi"], c["gamma"], c)
    syn = _draw_normal(rng, c["n_syn"], c)
    ev_path = _pathogenic(rng, N_EVAL, c["gamma"], c)
    ev_benign = _draw_normal(rng, N_EVAL, c)
    ev_pop, ev_pop_is_path = _mixed(rng, N_EVAL, c["pi"], c["gamma"], c)
    return dict(samples=[plp, blb, pop, syn], eval=dict(pathogenic=ev_path, benign=ev_benign, population=ev_pop),
                eval_pop_is_path=ev_pop_is_path)


def true_log_lr(x, c):
    """log lr+(x) = log p1(x) / p0(x) for the population's pathogenic variants (mechanism mix gamma)."""
    g, k = c["gamma"], c["kappa"]
    if c.get("shape", "gauss") == "gauss":
        # p1/p0 = g exp(-k x - k^2/2) + (1 - g), computed stably
        return np.logaddexp(np.log(g) - k * x - k * k / 2, np.log1p(-g) if g < 1 else -np.inf)
    with np.errstate(divide="ignore"):
        ratio = np.log(_pdf_abnormal(x, c)) - np.log(_pdf_normal(x, c))
    return np.logaddexp(np.log(g) + ratio, np.log1p(-g) if g < 1 else -np.inf)


def true_component_params(c):
    """The true components in production (skew, loc, scale) format: measured-pathogenic, then normal."""
    return [(0.0, -c["kappa"], 1.0), (0.0, 0.0, 1.0)]


def true_weights(c):
    """True per-sample mixing proportions [abnormal, normal] for the four samples (P/LP, B/LB, population,
    synonymous)."""
    plp_abn = (1 - c["eps_p"]) * c["lab_gamma"]
    blb_abn = c["eta_b"] * c["gamma"]
    pop_abn = c["pi"] * c["gamma"]
    return np.array([[plp_abn, 1 - plp_abn], [blb_abn, 1 - blb_abn], [pop_abn, 1 - pop_abn], [0.0, 1.0]])


def density(x, c, which):
    """True score density of a class: 'benign' or 'pathogenic' (population mechanism mix)."""
    if which == "benign":
        return _pdf_normal(x, c)
    return c["gamma"] * _pdf_abnormal(x, c) + (1 - c["gamma"]) * _pdf_normal(x, c)
