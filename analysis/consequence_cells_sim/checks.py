"""Verification checks for the consequence-cell simulation (plan: Verification 1-3).

    python -m analysis.consequence_cells_sim.checks [--null]

1. Each simulated sample's consequence mix and pathogenic share match the cell settings; true densities integrate
   to 1; the exact prior matches large-sample draws.
2. Composition: count-weighted cell weights (true per-cell abnormal shares) reproduce the analytic mixture of the
   member rows to 1e-12.
3. (--null) Null case: every consequence with the same prior and the same mechanism -> pooled ~ cells ~ truth.
"""
import argparse

import numpy as np

from analysis.consequence_cells_sim import generate as G
from analysis.consequence_cells_sim import run as R


def check_samples():
    for assay in ("fitness", "abundance"):
        c = G.make_cell("check", "large", "realistic", assay=assay, n_plp=4000, n_blb=4000, n_pop=40000, n_syn=8000)
        d = G.make_data(G.rng_for(c, 0), c)
        m = d["mask"]
        for s, mix in ((0, c["plp_mix"]), (1, c["blb_mix"])):
            rows = m[:, s] & ~m[:, 2]
            got = {k: float((d["csq"][rows] == k).mean()) for k in G.CONSEQUENCES}
            want = {k: v for k, v in mix.items()}
            if assay == "abundance" and s == 0:     # no visible-or-kept pathogenic synonymous: the mix renormalises
                want = {k: (v if k != "synonymous" else 0.0) for k, v in want.items()}
                t = sum(want.values()); want = {k: v / t for k, v in want.items()}
            assert all(abs(got[k] - want[k]) < 0.02 for k in G.CONSEQUENCES), (assay, s, got, want)
            if s == 0:
                assert d["path"][rows].all()
            else:  # B/LB: only splice-blind synonymous labels may carry pathogenic variants (rho_syn_blb)
                assert not d["path"][rows & (d["csq"] != "synonymous")].any()
        pop = m[:, 2]
        for k in G.CONSEQUENCES:
            sel = pop & (d["csq"] == k)
            t = G.truth(c, k)
            se = np.sqrt(max(t["pi"] * (1 - t["pi"]), 1e-6) / max(sel.sum(), 1))
            assert abs(d["path"][sel].mean() - t["pi"]) < 5 * se + 1e-3, (assay, k, d["path"][sel].mean(), t["pi"])
            x = np.linspace(-15, 10, 20001)
            for sh in (t["P"], t["B"]):
                if sum(sh.values()) > 0:
                    assert abs(np.trapz(np.exp(G._mix_logpdf(x, c, sh)), x) - 1) < 1e-6
    print("1. samples, priors and densities: ok")


def check_compose():
    c = G.make_cell("check", "mid", "realistic", assay="fitness")
    d = G.make_data(G.rng_for(c, 1), c)
    names, cor = R.cell_keys(d, c["assay"])
    # a per-cell weight vector: the share of abnormal score class among the cell's rows (any known per-cell weights)
    rng = np.random.default_rng(0)
    W = rng.dirichlet([1, 1], size=len(names))
    for kind, csq in (("pooled", None), ("cell_mis", None), ("cell_c", "PTV"), ("nu_c", "missense")):
        m = R.layout(d, kind, csq)
        ss, keep = R.scoreset(d["score"], m)
        M = R.membership(m, keep, cor, len(names))
        comp = R.compose_fits([{"fit": {"component_params": None, "weights": W, "xlims": None}}], M)[0]["fit"]["weights"]
        cols = [s for s in range(4) if m[keep, s].any()]
        for r, s in enumerate(cols):
            rows = keep & m[:, s]
            direct = W[cor[rows]].mean(0)      # mixture of the member rows' cell weights
            assert np.allclose(comp[r], direct, atol=1e-12), (kind, s)
    print("2. composition equals the member-row mixture to 1e-12: ok")


def check_null(B=20, R_=4):
    c = G.make_cell("null", "large", "realistic", assay="fitness", d_hi=0.0, h=dict(synonymous=0, missense=0, PTV=0),
                    pi_prot=dict(synonymous=0.2, missense=0.2, PTV=0.2), mis_mech=dict(stability=1.0, activity=0.0, other=0.0),
                    tau_ptv=1.0, plp_mix=dict(synonymous=0.3, missense=0.4, PTV=0.3),
                    # the benign reference must be clean: no synonymous sample (it would be 20% pathogenic here) and
                    # no splice-blind B/LB synonymous labels
                    n_syn=0, rho_syn_blb=0.0)
    c["mis_mech"] = dict(stability=1.0, activity=0.0, other=0.0)
    # synonymous protein mechanism is 'regulatory' (normal); make it measured for the null by treating it as LoF
    orig = G._mechanisms
    G._mechanisms = lambda cc, csq: {"LoF": 1.0} if csq in ("synonymous", "PTV") else orig(cc, csq)
    try:
        row = R.run_cell(c, 0, B=B, R=R_)
    finally:
        G._mechanisms = orig
    print("3. null case:", row["status"][:200])
    for arm in ("pooled", "cell_all", "cell_missense", "true_cell_missense"):
        k = f"{arm}__missense_prior_err"
        print(f"   {arm:20s} prior err {row.get(k, np.nan):+.3f}   post_mae {row.get(f'{arm}__missense_post_mae', np.nan):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--null", action="store_true"); a = ap.parse_args()
    check_samples(); check_compose()
    if a.null:
        check_null()
