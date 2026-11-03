"""Build the prior-simulation report page (HTML with inline SVG) from the outputs
of simulate_priors.py and test_priors.py.

Run:
    python -m analysis.prior_simulation.report [--out PATH]
"""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from analysis.prior_simulation.simulate_priors import (
    ASSAYS, CLASSES, DEFAULT_OUT_DIR, POP_M1_FRAC, class_density,
)

A1, A2, A3 = ASSAYS
SHORT = dict(zip(ASSAYS, ["A1", "A2", "A3"]))
LONG = dict(zip(ASSAYS, ["M1 only", "M2 only", "M1 + M2"]))
CSS_VAR = dict(zip(ASSAYS, ["--a1", "--a2", "--a3"]))
DETECTED = dict(zip(ASSAYS, [POP_M1_FRAC, 1 - POP_M1_FRAC, 1.0]))
from analysis.prior_simulation.summary import KEYS, key_numbers, mean_ci, sel, summarize  # noqa: F401


# ── SVG helpers ──────────────────────────────────────────────────────────────

def lin(v, a, b, c, d):
    return c + (v - a) / (b - a) * (d - c)


def fmt_tick(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != 0 else "0"


def panel(title, xlabel, xlim, ylim, xticks, yticks, series, w=300, h=230, ref=None, identity=False,
          expected=None, ylabel="Estimated prior", xticklabels=None):
    """One chart panel. series: list of dicts with assay, xs, (m, lo, hi) arrays, arm."""
    ml, mr, mt, mb = 46, 12, 30, 40
    pw, ph = w - ml - mr, h - mt - mb
    X = lambda v: ml + lin(v, *xlim, 0, pw)
    Y = lambda v: mt + lin(v, *ylim, ph, 0)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{ml}" y="16" class="ptitle">{html.escape(title)}</text>']
    for t in yticks:
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{ml - 6}" y="{Y(t) + 3.5:.1f}" class="tick" text-anchor="end">{fmt_tick(t)}</text>')
    for i, t in enumerate(xticks):
        lab = xticklabels[i] if xticklabels is not None else fmt_tick(t)
        out.append(f'<text x="{X(t):.1f}" y="{mt + ph + 15}" class="tick" text-anchor="middle">{lab}</text>')
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" class="axis"/>')
    out.append(f'<text x="{ml + pw / 2:.1f}" y="{h - 6}" class="alabel" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text transform="translate(12 {mt + ph / 2:.1f}) rotate(-90)" class="alabel" '
               f'text-anchor="middle">{html.escape(ylabel)}</text>')
    if identity:
        lo, hi = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
        out.append(f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" class="truth"/>')
    if ref is not None:
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" class="truth"/>')
    if expected:
        for assay, xs, ys in expected:
            pts = " ".join(f"{X(x):.1f},{Y(min(max(y, ylim[0]), ylim[1])):.1f}" for x, y in zip(xs, ys) if np.isfinite(y))
            out.append(f'<polyline points="{pts}" class="expect" style="stroke:var({CSS_VAR[assay]})"/>')
    span = xlim[1] - xlim[0]
    for s in series:
        off = (ASSAYS.index(s["assay"]) - 1) * 0.028 * span + (-0.009 if s["arm"] == "oracle" else 0.009) * span
        col = f"var({CSS_VAR[s['assay']]})"
        for x, m, lo, hi, n, tip in zip(s["xs"], s["m"], s["lo"], s["hi"], s["n"], s["tips"]):
            if not np.isfinite(m):
                continue
            cx = X(x + off)
            clip = lambda v: min(max(v, ylim[0]), ylim[1])
            out.append(f'<g class="pt"><title>{html.escape(tip)}</title>')
            if np.isfinite(lo):
                out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(clip(lo)):.1f}" y2="{Y(clip(hi)):.1f}" '
                           f'style="stroke:{col}" class="err"/>')
            fill = col if s["arm"] == "oracle" else "var(--surface)"
            out.append(f'<circle cx="{cx:.1f}" cy="{Y(clip(m)):.1f}" r="4" style="fill:{fill};stroke:{col}" '
                       f'class="dot"/><circle cx="{cx:.1f}" cy="{Y(clip(m)):.1f}" r="10" class="hit"/></g>')
    out.append("</svg>")
    return "\n".join(out)


def series_for(summ, xcol, label_fn):
    out = []
    for assay in ASSAYS:
        s = summ[summ.assay == assay].sort_values(xcol)
        for arm, p in (("oracle", "o"), ("fit", "f")):
            tips = [f"{SHORT[assay]} ({LONG[assay]}), {'true components' if arm == 'oracle' else 'ExCALIBR fit'}, "
                    f"{label_fn(r)}: {r[p + '_m']:.3f} [{r[p + '_lo']:.3f}, {r[p + '_hi']:.3f}], n = {int(r[p + '_n'])}"
                    for _, r in s.iterrows()]
            out.append({"assay": assay, "arm": arm, "xs": s[xcol].to_numpy(), "m": s[p + "_m"].to_numpy(),
                        "lo": s[p + "_lo"].to_numpy(), "hi": s[p + "_hi"].to_numpy(),
                        "n": s[p + "_n"].to_numpy(), "tips": tips})
    return out


def expected_for(summ, xcol):
    return [(a, summ[summ.assay == a].sort_values(xcol)[xcol].to_numpy(),
             summ[summ.assay == a].sort_values(xcol)["expected"].to_numpy()) for a in ASSAYS]


# ── diagram ──────────────────────────────────────────────────────────────────

def density_path(x, y, X, Y):
    return "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x, y))


def diagram(base, n_o, n_f):
    """Data generation -> assays -> estimate, with each assay's estimates overlaid."""
    W, H = 1040, 560
    o = [f'<svg viewBox="0 0 {W} {H}" class="diagram" role="img" '
         f'aria-label="Simulation design: variants, latent effects, three assays, prior estimates">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']

    def box(x, y, w, h, title, lines, cls="node"):
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>')
        o.append(f'<text x="{x + 12}" y="{y + 22}" class="ntitle">{html.escape(title)}</text>')
        for i, ln in enumerate(lines):
            o.append(f'<text x="{x + 12}" y="{y + 42 + 17 * i}" class="nbody">{ln}</text>')

    def arrow(x1, y1, x2, y2):
        o.append(f'<path d="M{x1},{y1} C{(x1 + x2) / 2},{y1} {(x1 + x2) / 2},{y2} {x2},{y2}" '
                 f'class="flow" marker-end="url(#arr)"/>')

    # Column 1: population and samples.
    box(16, 20, 230, 132, "Variants in the gene", [
        "benign: probability P(Y=0)",
        "pathogenic: probability P(Y=1)",
        "&#160;&#160;via M1: 25% of pathogenic",
        "&#160;&#160;via M2: 75% of pathogenic",
        "true P(Y=1) = 0.25 (varied)",
    ])
    box(16, 176, 230, 196, "Samples drawn", [
        "<tspan class='b'>P/LP</tspan>: pathogenic (100)",
        "&#160;&#160;+ label noise ε, mechanism bias",
        "<tspan class='b'>B/LB</tspan>: benign (100)",
        "&#160;&#160;+ label noise δ",
        "<tspan class='b'>gnomAD</tspan>: pathogenic w.p. P(Y=1) (500)",
        "<tspan class='b'>Synonymous</tspan>: benign (200)",
        "sparse: 8/1/50/20 and 3/0/10/6",
        f"{n_o} datasets per setting ({n_f} fitted)",
    ])
    arrow(131, 152, 131, 176)
    # Column 2: latent effects.
    box(282, 176, 210, 196, "Latent effect (e1, e2)", [
        "e ~ N(μ, Σ), Σ = [[1, ρ], [ρ, 1]]",
        "benign: μ = (0, 0)",
        "via M1: μ = (−Δ, −ρΔ)",
        "via M2: μ = (−ρΔ, −Δ)",
        "shift Δ = 4 SD (1.5, 2.5)",
        "correlation ρ = 0 (0.3, 0.6)",
        "same variants read by",
        "all three assays",
    ])
    arrow(246, 274, 282, 274)

    # Column 3: assays with class densities and estimates.
    x0, colw, rowh, y0 = 560, 464, 166, 20
    xs = np.linspace(-8, 3.5, 300)
    for i, assay in enumerate(ASSAYS):
        y = y0 + i * (rowh + 14)
        o.append(f'<rect x="{x0}" y="{y}" width="{colw}" height="{rowh}" rx="6" class="node"/>')
        o.append(f'<text x="{x0 + 12}" y="{y + 22}" class="ntitle" style="fill:var({CSS_VAR[assay]})">'
                 f'{SHORT[assay]}: assay reads {LONG[assay]}</text>')
        score = {"A1 (M1 only)": "score = e1", "A2 (M2 only)": "score = e2", "A3 (M1+M2)": "score = min(e1, e2)"}[assay]
        o.append(f'<text x="{x0 + colw - 12}" y="{y + 22}" class="nbody" text-anchor="end">{score}</text>')
        arrow(492, 274, x0, y + rowh / 2)
        # Mini density plot.
        dx, dy, dw, dh = x0 + 14, y + 34, 190, 96
        X = lambda v: dx + lin(v, -8, 3.5, 0, dw)
        dens = {c: class_density(xs, c, assay, 4.0, 0.0) for c in CLASSES}
        Y = lambda v: dy + lin(v, 0, 0.45, dh, 0)
        o.append(f'<line x1="{dx}" x2="{dx + dw}" y1="{dy + dh}" y2="{dy + dh}" class="axis"/>')
        for c, cls in (("B", "dB"), ("M1", "dM1"), ("M2", "dM2")):
            o.append(f'<path d="{density_path(xs, dens[c], X, Y)}" class="{cls}"/>')
        o.append(f'<text x="{X(0):.1f}" y="{dy + dh + 13}" class="tick" text-anchor="middle">normal</text>')
        o.append(f'<text x="{X(-4):.1f}" y="{dy + dh + 13}" class="tick" text-anchor="middle">abnormal</text>')
        # Estimate strip: 0 to 0.5.
        sx, sw, sy = x0 + 238, 210, y + 92
        S = lambda v: sx + lin(v, 0, 0.5, 0, sw)
        o.append(f'<text x="{sx}" y="{y + 50}" class="nbody">estimated prior (true 0.25)</text>')
        o.append(f'<line x1="{sx}" x2="{sx + sw}" y1="{sy}" y2="{sy}" class="axis"/>')
        for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
            o.append(f'<line x1="{S(t):.1f}" x2="{S(t):.1f}" y1="{sy}" y2="{sy + 4}" class="axis"/>'
                     f'<text x="{S(t):.1f}" y="{sy + 16}" class="tick" text-anchor="middle">{fmt_tick(t)}</text>')
        o.append(f'<line x1="{S(0.25):.1f}" x2="{S(0.25):.1f}" y1="{sy - 34}" y2="{sy}" class="truth"/>')
        ab = 0.25 * DETECTED[assay]
        o.append(f'<g class="pt"><title>Fraction of gnomAD variants that are pathogenic and score abnormal: '
                 f'{ab:.3f}</title><path d="M{S(ab):.1f},{sy - 6} l5,6 l-5,6 l-5,-6 z" class="abn"/></g>')
        r = base[base.assay == assay].iloc[0]
        col = f"var({CSS_VAR[assay]})"
        for arm, p, yy in (("true components", "o", sy - 26), ("ExCALIBR fit", "f", sy - 14)):
            fill = col if p == "o" else "var(--surface)"
            o.append(f'<g class="pt"><title>{arm}: {r[p + "_m"]:.3f} [{r[p + "_lo"]:.3f}, {r[p + "_hi"]:.3f}]'
                     f'</title><line x1="{S(r[p + "_lo"]):.1f}" x2="{S(r[p + "_hi"]):.1f}" y1="{yy}" y2="{yy}" '
                     f'style="stroke:{col}" class="err"/><circle cx="{S(r[p + "_m"]):.1f}" cy="{yy}" r="4" '
                     f'style="fill:{fill};stroke:{col}" class="dot"/></g>')
        o.append(f'<text x="{sx}" y="{y + rowh - 12}" class="nbody">abnormal in gnomAD: '
                 f'{100 * ab:.1f}%; estimate {r["o_m"]:.2f} / {r["f_m"]:.2f}</text>')
    # Legend.
    ly = H - 24
    items = [("dB", "benign"), ("dM1", "pathogenic via M1"), ("dM2", "pathogenic via M2")]
    lx = 16
    for cls, lab in items:
        o.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" class="{cls}"/>'
                 f'<text x="{lx + 28}" y="{ly + 4}" class="nbody">{lab}</text>')
        lx += 40 + 7.2 * len(lab)
    o.append(f'<circle cx="{lx + 4}" cy="{ly}" r="4" class="legdot"/><text x="{lx + 14}" y="{ly + 4}" '
             f'class="nbody">true components</text>')
    lx += 130
    o.append(f'<circle cx="{lx + 4}" cy="{ly}" r="4" class="legdot hollow"/><text x="{lx + 14}" y="{ly + 4}" '
             f'class="nbody">ExCALIBR fit (mean, 95% CI)</text>')
    lx += 200
    o.append(f'<path d="M{lx + 4},{ly - 6} l5,6 l-5,6 l-5,-6 z" class="abn"/><text x="{lx + 14}" y="{ly + 4}" '
             f'class="nbody">pathogenic and abnormal</text>')
    o.append("</svg>")
    return "\n".join(o)


# ── page ─────────────────────────────────────────────────────────────────────

def fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def results_table(rows):
    head = ("<tr><th>Setting</th><th>Assay</th><th>Expected</th><th>True components</th>"
            "<th>ExCALIBR fit</th></tr>")
    body = []
    for label, r in rows:
        body.append(f"<tr><td>{label}</td><td><span class='sw' style='background:var({CSS_VAR[r.assay]})'></span>"
                    f"{SHORT[r.assay]} ({LONG[r.assay]})</td><td class='num'>{r.expected:.3f}</td>"
                    f"<td class='num'>{fmt(r.o_m, r.o_lo, r.o_hi)}</td><td class='num'>{fmt(r.f_m, r.f_lo, r.f_hi)}</td></tr>")
    return f"<div class='tw'><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


MACROS_TEX = Path("/data/ross/assay_calibration/exc_pp_clinvar2025_calib/figures/generated/macros.tex")
ED3_TABLE = Path("/data/ross/assay_calibration/exc_pp_clinvar2025_calib/figures/generated/tables/author_performance.tex")
SD1_CSV = Path("/data/ross/assay_calibration/exc_pp_clinvar2025_calib/supplementary_data/SD1_dataset_summaries.csv")
# Datasets whose priors Reviewer 3 cites (comment 6), in the reviewer's order.
REVIEWER_PRIOR_DATASETS = [
    "TARDBP_Bolognesi_Faure_2019", "SCN5A_Glazer_2020", "TPK1_Weile_2017", "ASPA_Grønbæk-Thygesen_2024_abundance",
    "ASPA_Grønbæk-Thygesen_2024_toxicity", "GCK_Gersing_2023_complementation", "GCK_Gersing_2024_abundance",
    "KCNH2_Jiang_2022", "KCNH2_Kozek_Glazer_2020",
]


def sd1_prior_facts(path=SD1_CSV):
    """Revised priors of the reviewer-cited datasets, the 87-dataset summary, and the PU (no benign reference)
    datasets, from Supplementary Data 1."""
    d = pd.read_csv(path)
    p = d.set_index("dataset")["prior"]
    pu = d[(d["n_blb_fit"].fillna(0) == 0) & (d["n_synonymous_fit"].fillna(0) == 0)]
    return {"priors": {k: float(p[k]) for k in REVIEWER_PRIOR_DATASETS}, "n": int(p.notna().sum()),
            "median": float(p.median()), "n20": int((p >= 0.2).sum()), "n30": int((p >= 0.3).sum()),
            "n40": int((p >= 0.4).sum()), "pu": pu[["dataset", "n_plp_fit", "n_gnomad_fit"]].to_dict("records")}


def sample_size_html(ss):
    """Section: one-at-a-time downsampling of the standard controls (test_priors.py --design sample_size)."""
    from analysis.prior_simulation.summary import SAMPLES, sample_size_key_numbers
    k = sample_size_key_numbers(ss)
    arms = (("oracle", "true components"), ("fit", "ExCALIBR fit"))
    metrics = (("bias", "bias_lo", "bias_hi", "Bias (mean − 0.25)", (-0.1, 0.3), [-0.1, 0, 0.1, 0.2, 0.3], 0.0),
               ("sd", "sd_lo", "sd_hi", "SD of the estimate", (0, 0.3), [0, 0.1, 0.2, 0.3], None),
               ("discard", None, None, "Discarded (%)", (0, 100), [0, 25, 50, 75, 100], None))
    charts = []
    for col, lo_c, hi_c, ylabel, ylim, yticks, ref in metrics:
        for sample in SAMPLES:
            g = ss[ss["sample"] == sample]
            sizes = sorted(g["size"].unique(), reverse=True)
            pos = {n: i for i, n in enumerate(sizes)}
            series = []
            for assay in ASSAYS:
                for arm, arm_lab in arms:
                    r = g[(g.assay == assay) & (g.arm == arm)].sort_values("size", ascending=False)
                    scale = 100 if col == "discard" else 1
                    m = r[col].to_numpy() * scale
                    lo = r[lo_c].to_numpy() if lo_c else np.full(len(r), np.nan)
                    hi = r[hi_c].to_numpy() if hi_c else np.full(len(r), np.nan)
                    tips = [f"{SHORT[assay]} ({LONG[assay]}), {arm_lab}, {sample} = {int(x.size)}: "
                            f"bias {x.bias:+.3f}, SD {x.sd:.3f}, RMSE {x.rmse:.3f}, discarded {100 * x.discard:.0f}%, "
                            f"n = {int(x.n)}" for x in r.itertuples()]
                    series.append({"assay": assay, "arm": arm, "xs": np.array([pos[n] for n in r["size"]]),
                                   "m": m, "lo": lo, "hi": hi, "n": r["n"].to_numpy(), "tips": tips})
            charts.append(panel(f"{sample} reduced", f"{sample} variants (others at standard size)",
                                (-0.5, len(sizes) - 0.5), ylim, list(range(len(sizes))), yticks, series,
                                ref=ref, ylabel=ylabel, xticklabels=[str(int(n)) for n in sizes]))
    rows = []
    for assay in ASSAYS:
        for sample in ("base",) + SAMPLES:
            o, raw, f = (k[(assay, arm, sample)] for arm in ("oracle", "oracle_raw", "fit"))
            lab = "standard (100/100/500/200)" if sample == "base" else f"{sample} = {int(o['size'])}"
            rows.append(f"<tr><td>{SHORT[assay]}</td><td>{lab}</td>"
                        + "".join(f"<td class='num'>{v}</td>" for v in (
                            f"{o.bias:+.3f}", f"{o.sd:.3f}", f"{o.rmse:.3f}", f"{100 * o.bias2 / max(o.rmse ** 2, 1e-12):.0f}%",
                            f"{100 * o.discard:.0f}%", f"{raw.bias:+.3f}",
                            f"{f.bias:+.3f}", f"{f.sd:.3f}", f"{100 * f.discard:.0f}%")) + "</tr>")
    table = ("<div class='tw'><table><thead><tr><th>Assay</th><th>Setting</th>"
             "<th>Bias</th><th>SD</th><th>RMSE</th><th>Bias² share</th><th>Discarded</th><th>Bias before discard</th>"
             "<th>Fit bias</th><th>Fit SD</th><th>Fit discarded</th></tr>"
             "<tr><th></th><th></th><th colspan='6'>true components</th><th colspan='3'>ExCALIBR fit</th></tr></thead>"
             "<tbody>" + "".join(rows) + "</tbody></table></div>")
    n_o = int(ss[ss.arm == "oracle_raw"]["n"].max())
    n_f = int(ss[ss.arm == "fit"]["n"].max())
    return f"""<section>
<h2>Sample size: bias, variance, and which sample matters</h2>
<p>Starting from the standard controls (100 P/LP, 100 B/LB, 500 gnomAD, 200 synonymous; true prior 0.25, mean
pathogenic effect −4), one sample at a time is reduced while the other three stay at their standard size. Bias is the
mean of the kept estimates minus the truth; estimates at or below 0.001 or at or above 0.999 are discarded, as in
production, so the discard rate is shown separately, and the table also gives the true-component bias before the
discard rule. {n_o} simulated datasets per setting with the true components, {n_f} with the fit; bars: 95% CIs
(t-interval for the bias, bootstrap for the SD).</p>
{{legend}}
<div class="card charts">{"".join(charts)}</div>
<p>At the smallest size of each sample:</p>
<div class="card">{table}</div>
{{findings}}
</section>"""


def sample_size_findings(ss, reps):
    """Findings for the sample-size section, from the summary and the replicate-level class counts."""
    def row(sample, size, assay, arm):
        g = ss[(ss["sample"] == sample) & (ss["size"] == size) & (ss.assay == assay) & (ss.arm == arm)]
        return g.iloc[0]
    A1_, A3_ = ASSAYS[0], ASSAYS[2]
    sizes = {smp: int(ss[ss["sample"] == smp]["size"].min()) for smp in ("P/LP", "B/LB", "gnomAD", "synonymous")}
    base = {a: row("gnomAD", 500, a, "oracle") for a in ASSAYS}
    g_raw = {a: row("gnomAD", sizes["gnomAD"], a, "oracle_raw") for a in ASSAYS}
    g_kept = row("gnomAD", sizes["gnomAD"], A1_, "oracle")
    g_fit = row("gnomAD", sizes["gnomAD"], A1_, "fit")
    _g = reps[(reps.regime == f"gnomAD = {sizes['gnomAD']}") & (reps.assay == A1_) & reps.oracle.isna()]
    pct_zero = 100 * (_g.oracle_raw <= 0.001).mean()
    p_none = (1 - float(reps.true_prior.iloc[0]) * POP_M1_FRAC) ** sizes["gnomAD"]  # P(no M1 variant in gnomAD)
    p20, p10 = row("P/LP", 20, A1_, "oracle_raw"), row("P/LP", 10, A1_, "oracle_raw")
    p_a3 = max(abs(row("P/LP", n, A3_, "oracle_raw").bias) for n in (20, 10, 5))
    other = ss[ss["sample"].isin(["B/LB", "synonymous"]) & ~ss.is_base & (ss.arm == "oracle_raw")]
    # Stratify the P/LP = 20 M1-only estimates by how many P/LP act through M1.
    a = reps[(reps.regime == "P/LP = 20") & (reps.assay == A1_)]
    strat = a.groupby("n_plp_m1")["oracle_raw"].agg(["size", "mean"])
    n_exp = int(round(20 * POP_M1_FRAC))
    lo_k = int(strat[strat["size"] >= 10].index.min())
    hi_k = int(strat[strat["size"] >= 10].index.max())
    strat_rows = "".join(f"<tr><td class='num'>{int(k)}</td><td class='num'>{int(r['size'])}</td>"
                         f"<td class='num'>{r['mean']:.3f}</td></tr>" for k, r in strat.iterrows() if r["size"] >= 10)
    return f"""<p><b>gnomAD size drives the variance, and the apparent bias comes from discarding.</b> Reducing the
gnomAD sample from 500 to {sizes['gnomAD']} raised the standard deviation of the true-component estimate from
{base[A1_].sd:.3f} to {g_raw[A1_].sd:.3f} for the M1-only assay and from {base[A3_].sd:.3f} to {g_raw[A3_].sd:.3f} for the
combined assay, but before the discard rule the estimate stayed unbiased (bias {g_raw[A1_].bias:+.3f}
[{g_raw[A1_].bias_lo:+.3f}, {g_raw[A1_].bias_hi:+.3f}] and {g_raw[A3_].bias:+.3f}). The rule then removed
{100 * g_kept.discard:.0f}% of the M1-only estimates, {pct_zero:.0f}% of them at zero: with so few gnomAD variants,
many samples contain no abnormally scoring variant ({100 * p_none:.0f}% of samples of {sizes['gnomAD']} contain no M1
variant). Removing these zeros leaves only the high estimates, so the kept estimates averaged
{g_kept.bias:+.3f} above the truth; the fit behaved the same way ({g_fit.bias:+.3f}, {100 * g_fit.discard:.0f}% of
bootstrap iterations discarded).</p>
<p><b>P/LP size drives a real bias, but only for the assay measuring the minority mechanism.</b> With 20 and 10 P/LP
variants, the M1-only estimate was {p20.bias:+.3f} and {p10.bias:+.3f} too high before any discarding, whereas the
combined assay stayed within {p_a3:.3f}. The cause is the chance mechanism mix of a small P/LP sample: the estimate
scales inversely with the fraction of P/LP that act through the measured mechanism. With 20 P/LP, datasets with the
expected {n_exp} M1 variants estimated {strat.loc[n_exp, 'mean']:.2f}, those with {lo_k} estimated
{strat.loc[lo_k, 'mean']:.2f}, and those with {hi_k} estimated {strat.loc[hi_k, 'mean']:.2f}. Because the
relationship is convex, under-representation raises the estimate more than over-representation lowers it. This is
the same effect as a P/LP sample drawn from the wrong mechanism mix, arising here by chance.</p>
<p><b>B/LB and synonymous size hardly matter.</b> Down to {sizes['B/LB']} B/LB or {sizes['synonymous']} synonymous
variants, every true-component bias was within {other.bias.abs().max():.3f} and every standard deviation at most
{other.sd.max():.3f}.</p>
<div class='card'><p>M1-only assay, 20 P/LP: estimate by the number of P/LP acting through M1 (true components, before
the discard rule; counts with at least 10 datasets)</p><div class='tw'><table><thead><tr><th>M1 P/LP</th>
<th>Datasets</th><th>Mean estimate</th></tr></thead><tbody>{strat_rows}</tbody></table></div></div>"""


CALIB_DIR = SD1_CSV.parents[1]


def sd1_bootstrap_kept(path=SD1_CSV, calib_dir=CALIB_DIR):
    """Per dataset: fraction of bootstrap fits kept after the prior discard rule (pct_bootstraps_kept in the
    selected model's calibration JSON), with the dataset's prior and control counts."""
    import glob
    import json
    d = pd.read_csv(path)
    rows = []
    for r in d.itertuples():
        ds = r.dataset if (calib_dir / r.dataset).is_dir() else r.dataset + "_clinvar_2018"
        exact = calib_dir / ds / f"{ds}_{r.n_c}_{r.benign_method}_calibration.json"
        files = [str(exact)] if exact.exists() else sorted(glob.glob(str(calib_dir / ds / f"{ds}_{r.n_c}_*_calibration.json")))
        kept = json.loads(Path(files[0]).read_text()).get("pct_bootstraps_kept") if files else None
        rows.append({"dataset": r.dataset, "kept": np.nan if kept is None else float(kept), "prior": r.prior,
                     "n_plp": r.n_plp_fit, "n_blb": r.n_blb_fit, "n_gnomad": r.n_gnomad_fit, "n_syn": r.n_synonymous_fit})
    return pd.DataFrame(rows)


POLICY_ROWS = [  # (experiment, regime, true prior, label) shown in the policy table
    ("1_clean", "rich", 0.25, "rich controls, 0.25"),
    ("1_clean", "TPK1-like", 0.05, "TPK1-like, 0.05"), ("1_clean", "TPK1-like", 0.25, "TPK1-like, 0.25"),
    ("1_clean", "TPK1-like", 0.40, "TPK1-like, 0.40"),
    ("1_clean", "KCNH2_Kozek-like", 0.05, "KCNH2-like, 0.05"), ("1_clean", "KCNH2_Kozek-like", 0.10, "KCNH2-like, 0.10"),
    ("1_clean", "KCNH2_Kozek-like", 0.25, "KCNH2-like, 0.25"), ("1_clean", "KCNH2_Kozek-like", 0.40, "KCNH2-like, 0.40"),
    ("6_sample_size", "gnomAD = 10", 0.25, "gnomAD = 10, 0.25"), ("6_sample_size", "gnomAD = 20", 0.25, "gnomAD = 20, 0.25"),
    ("6_sample_size", "P/LP = 10", 0.25, "P/LP = 10, 0.25"),
]
POLICY_COLS = (("P0", "Discard (current)"), ("P2a", "Clip [0.001, 0.999]"), ("P3", "Clip + EM-start → 0.10"),
               ("P4", "Fallback 0.10"))


def start_value_html(reps):
    """Sub-section of the rules section: EM started at 0.1 instead of 0.5, under the current discard rule, for the
    sparse-control settings (rich controls are listed only as the largest difference)."""
    from analysis.prior_simulation.summary import start_value_summary
    sv = start_value_summary(reps)
    if sv is None:
        return "<p><i>EM starting value: pending (run predates the 0.1 start).</i></p>"
    def g(reg, tp, a, arm, st):
        x = sv[(sv.regime == reg) & np.isclose(sv.true_prior, tp) & (sv.assay == a) & (sv.arm == arm)
               & np.isclose(sv.start, st)]
        return x.iloc[0] if len(x) else None
    rich = sv[sv.regime == "rich"].pivot_table(index=["true_prior", "assay", "arm"], columns="start", values="mean")
    rich_dev = float((rich[0.5] - rich[0.1]).abs().max())
    allp = sv.pivot_table(index=["experiment", "regime", "true_prior", "assay", "arm"], columns="start",
                          values="mean")
    n_big = int(((allp[0.5] - allp[0.1]).abs() > 0.01).sum())
    body = []
    for reg, lab in (("TPK1-like", "Sparse (8/1/50/20)"), ("KCNH2_Kozek-like", "Very sparse (3/0/10/6)")):
        for a in ASSAYS:
            for tp in (0.05, 0.10, 0.25, 0.40):
                cells = []
                for arm in ("oracle", "fit"):
                    h, q = g(reg, tp, a, arm, 0.5), g(reg, tp, a, arm, 0.1)
                    better = abs(q.bias) < abs(h.bias) - 0.005
                    worse = abs(q.bias) > abs(h.bias) + 0.005
                    style = (" style='background:var(--okbg)'" if better else
                             " style='background:var(--warnbg)'" if worse else "")
                    cells.append(f"<td class='num'>{h.bias:+.3f}<br><span class='sub'>{100 * h.at_start:.0f}% at 0.5"
                                 f"</span></td><td class='num'{style}>{q.bias:+.3f}<br><span class='sub'>"
                                 f"{100 * q.at_start:.0f}% at 0.1</span></td>")
                body.append(f"<tr><td>{lab}</td><td>{SHORT[a]}</td><td class='num'>{tp:.2f}</td>{''.join(cells)}</tr>")
    table = ("<div class='tw'><table><thead><tr><th>Controls</th><th>Assay</th><th>True prior</th>"
             "<th>True comp., start 0.5</th><th>True comp., start 0.1</th><th>Fit, start 0.5</th>"
             "<th>Fit, start 0.1</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    A1 = ASSAYS[0]
    k = {(tp, arm, st): g("KCNH2_Kozek-like", tp, A1, arm, st) for tp in (0.05, 0.25, 0.40)
         for arm in ("oracle", "fit") for st in (0.5, 0.1)}
    return f"""<h3>EM starting value: 0.1 instead of 0.5</h3>
<p>Every replicate was also re-estimated with the prior EM started at 0.1 instead of production's 0.5, using the same
saved densities and the current discard rule (nothing refit). When the prior is identified the start does not matter
(Corollary 1b): with rich controls the mean estimate changed by at most {rich_dev:.1e}. When it is not identified
the estimate is the start (Corollary 1a), and the discard rule keeps it, unlike an estimate of zero. So the start acts
like a per-bootstrap fallback value. Only {n_big} of {len(allp)} setting/assay/arm combinations moved by more than
0.01, all with sparse or very sparse controls, or with 5 or 10 P/LP.</p>
<div class="card"><p>Bias of the dataset prior, EM started at 0.5 (production) or 0.1; the sub-line is the share of
estimates (true components) or bootstrap fits (ExCALIBR fit) still at the start. Shaded: |bias| at least 0.005
lower (green) or higher (amber) with the 0.1 start.</p>{table}</div>
<p>For the very sparse M1-only assay, the 0.1 start lowered the true-component bias from
{k[(0.05, 'oracle', 0.5)].bias:+.3f} to {k[(0.05, 'oracle', 0.1)].bias:+.3f} at a true prior of 0.05 and from
{k[(0.25, 'oracle', 0.5)].bias:+.3f} to {k[(0.25, 'oracle', 0.1)].bias:+.3f} at 0.25, but raised its magnitude
from {k[(0.40, 'oracle', 0.5)].bias:+.3f} to {k[(0.40, 'oracle', 0.1)].bias:+.3f} at 0.40. The ExCALIBR fit moved
less, because the median over bootstrap fits is set by the fits that do have signal: {k[(0.05, 'fit', 0.5)].bias:+.3f}
to {k[(0.05, 'fit', 0.1)].bias:+.3f}, {k[(0.25, 'fit', 0.5)].bias:+.3f} to {k[(0.25, 'fit', 0.1)].bias:+.3f}, and
{k[(0.40, 'fit', 0.5)].bias:+.3f} to {k[(0.40, 'fit', 0.1)].bias:+.3f}. As with the 0.10 fallback, the gain at low
priors comes from assuming the answer, and it reverses when the truth is high. A start of 0.5 has the advantage that
a prior of exactly 0.5 is recognizable as unidentified, which is how the one real dataset at 0.500 is detected.</p>
"""


def policy_html(ps, reps, kept):
    """Section: alternatives to discarding boundary priors (test_priors.py --design policy), applied post hoc to the
    same raw bootstrap priors, plus how often the real datasets hit the discard rule."""
    fit = ps[ps.arm == "fit"]
    orc = ps[ps.arm == "oracle"]
    w = ps.pivot_table(index=["experiment", "regime", "true_prior", "assay", "arm"], columns="policy", values="mean")
    d_keep_clip = float((w.P1 - w.P2a).abs().max())
    d_clip_clip = float((w.P2a - w.P2b).abs().max())
    r = ps.pivot_table(index=["experiment", "regime", "true_prior", "assay", "arm"], columns="policy", values="rmse")
    diff = r[(r.P0 - r.P2a).abs() > 0.005]
    wins = {arm: (int((diff.xs(arm, level="arm").P2a < diff.xs(arm, level="arm").P0).sum()),
                  len(diff.xs(arm, level="arm"))) for arm in ("fit", "oracle")}
    clip_wins_low = diff.xs("fit", level="arm")
    clip_wins_low = clip_wins_low[clip_wins_low.P2a < clip_wins_low.P0].index.get_level_values("true_prior")
    def g(df, e, reg, tp, a, pol):
        x = df[(df.experiment == e) & (df.regime == reg) & np.isclose(df.true_prior, tp) & (df.assay == a)
               & (df.policy == pol)]
        return x.iloc[0] if len(x) else None
    body = []
    for a in (ASSAYS[0], ASSAYS[2]):
        for e, reg, tp, lab in POLICY_ROWS:
            cells = {p: g(fit, e, reg, tp, a, p) for p, _ in POLICY_COLS}
            best = min(c.rmse for c in cells.values())
            tds = []
            for p, _ in POLICY_COLS:
                c = cells[p]
                extra = (f"{100 * c.nan_frac:.0f}% none" if p == "P0" else
                         f"{100 * c.floor:.0f}% at floor" if p == "P2a" else
                         f"{100 * c.fallback:.0f}% fell back" if p == "P4" else "")
                style = " style='background:var(--okbg);font-weight:600'" if c.rmse <= best + 1e-9 else ""
                tds.append(f"<td class='num'{style}>{c.bias:+.3f} / {c.rmse:.3f}"
                           + (f"<br><span class='sub'>{extra}</span>" if extra else "") + "</td>")
            body.append(f"<tr><td>{SHORT[a]}</td><td>{lab}</td>{''.join(tds)}</tr>")
    table = ("<div class='tw'><table><thead><tr><th>Assay</th><th>Setting, true prior</th>"
             + "".join(f"<th>{t}</th>" for _, t in POLICY_COLS)
             + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    # headline numbers
    g10 = {p: g(fit, "6_sample_size", "gnomAD = 10", 0.25, ASSAYS[0], p) for p in ("P0", "P2a", "P4")}
    k05 = {p: g(fit, "1_clean", "KCNH2_Kozek-like", 0.05, ASSAYS[2], p) for p in ("P0", "P2a")}
    k40 = {p: g(fit, "1_clean", "KCNH2_Kozek-like", 0.40, ASSAYS[0], p) for p in ("P0", "P2a", "P3", "P4")}
    k10 = {p: g(fit, "1_clean", "KCNH2_Kozek-like", 0.10, ASSAYS[0], p) for p in ("P2a", "P3")}
    t25 = {p: g(fit, "1_clean", "TPK1-like", 0.25, ASSAYS[0], p) for p in ("P0", "P2a")}
    rich_max = float(fit[(fit.regime == "rich") & (fit.true_prior >= 0.10)].groupby(["true_prior", "assay"])["mean"]
                     .agg(lambda x: x.max() - x.min()).max())
    floor_max = float(fit[fit.policy == "P2a"].floor.max())
    n_fit = int(fit.n.max())
    n_boot = int(reps.loc[reps.fit_n_boot > 0, "fit_n_boot"].max())
    n_orc = int(orc.n.max())
    # real datasets
    k = kept.dropna(subset=["kept"])
    n_all, n_full = len(k), int((k.kept >= 1).sum())
    n10, n50 = int((k.kept < 0.9).sum()), int((k.kept < 0.5).sum())
    lost = k[k.kept < 0.9].sort_values("kept")
    lost_rows = "".join(
        f"<tr><td>{x.dataset}</td><td class='num'>{100 * (1 - x.kept):.0f}%</td><td class='num'>{x.prior:.3f}</td>"
        f"<td class='num'>{int(x.n_plp)}/{int(x.n_blb)}/{int(x.n_gnomad)}/{int(x.n_syn)}</td></tr>"
        for x in lost.itertuples())
    em = k[np.isclose(k.prior, 0.5, atol=1e-6)]
    em_txt = "; ".join(f"{x.dataset} ({int(x.n_plp)} P/LP, prior {x.prior:.3f}, {100 * (1 - x.kept):.0f}% of "
                       "bootstrap fits discarded)" for x in em.itertuples())
    start_html = start_value_html(reps)
    return f"""<section>
<h2>Alternatives to discarding boundary priors</h2>
<p>In production, a bootstrap fit whose prior is at or below 0.001 or at or above 0.999 is dropped, from the prior
median and from the likelihood ratios; a dataset with no remaining fit gets no calibration. The sample-size results
showed that this discard rule, not the estimator, biases sparse priors upward. Each alternative rule below is applied to the same raw
bootstrap priors (nothing is refit), and the dataset prior is the median over that dataset's bootstrap fits. The rules:
<b>discard</b> (current); <b>keep</b> every value; <b>clip</b> to [0.001, 0.999] or to [0.0001, 0.9999];
<b>clip + EM-start → 0.10</b>, which also replaces a bootstrap whose EM stayed at its 0.5 start (no signal) with 0.10;
and a dataset-level <b>fallback</b> to 0.10 when more than half the bootstrap fits are discarded or at the EM start.
{n_fit} fitted datasets per setting with {n_boot} bootstrap fits each, and {n_orc} true-component datasets per setting.
Production is unchanged.</p>
<div class="card"><p>ExCALIBR fit, bias / RMSE of the dataset prior under each rule (shaded: lowest RMSE in the row). Under
<i>discard</i>, “none” is the share of datasets left with no prior; under <i>clip</i>, “at floor” is the share whose
prior is the 0.001 floor. <i>Keep</i> and the [0.0001, 0.9999] clip are not shown because they match the
[0.001, 0.999] clip.</p>{table}</div>
<p><b>The clip bounds do not matter.</b> Keeping every value, clipping to [0.001, 0.999] and clipping to
[0.0001, 0.9999] gave dataset priors within {max(d_keep_clip, d_clip_clip):.4f} of each other in every setting. The
median depends only on how many bootstrap values fall below it, not how far below, so a clipped zero still counts as a
zero. The bound would matter only where a single value is used directly: up to {100 * floor_max:.0f}% of the sparse
datasets had a median prior at the 0.001 floor itself.</p>
<p><b>Keeping or clipping removes the bias but increases the variance, and neither wins on error.</b> At 10 gnomAD
variants, the M1-only bias fell from {g10['P0'].bias:+.3f} (discard; {100 * g10['P0'].nan_frac:.0f}% of datasets with
no prior) to {g10['P2a'].bias:+.3f} (clip), with RMSE {g10['P0'].rmse:.3f} and {g10['P2a'].rmse:.3f}. For the
KCNH2-like controls at a true prior of 0.05, the combined assay's bias fell from {k05['P0'].bias:+.3f} to
{k05['P2a'].bias:+.3f}, and {100 * k05['P0'].nan_frac:.0f}% of datasets had no prior under discarding. But at
higher true priors, discarding had the lower error: TPK1-like at 0.25, M1-only RMSE {t25['P0'].rmse:.3f} (discard)
versus {t25['P2a'].rmse:.3f} (clip); KCNH2-like at 0.40, {k40['P0'].rmse:.3f} versus {k40['P2a'].rmse:.3f}. Over all
fitted settings where the two differ (RMSE difference over 0.005), clipping had the lower error in {wins['fit'][0]} of
{wins['fit'][1]}, all at a true prior of {min(clip_wins_low):.2f}–{max(clip_wins_low):.2f}; with the true components,
in {wins['oracle'][0]} of {wins['oracle'][1]}. Discarding removes the zero estimates, which are right when the true
prior is near zero and wrong otherwise; with rich controls the rules agree to within {rich_max:.3f} at true priors of
0.10 and above.</p>
<p><b>The 0.10 fallback lowers the error by assumption, not by estimation.</b> It had the lowest RMSE in most sparse
settings (10 gnomAD variants, M1-only: {g10['P4'].rmse:.3f}, {100 * g10['P4'].fallback:.0f}% of datasets fell back),
because it pulls uninformative datasets toward 0.10, which is near most of the true priors simulated. When the truth
is higher it is biased downward (KCNH2-like at 0.40, M1-only: {k40['P4'].bias:+.3f}). Its benefit depends on the chosen value, and because
the evidence constant c depends on the prior, that value changes every evidence threshold. Replacing only the
bootstrap fits whose EM stayed at its start with 0.10 pulls the prior toward 0.10 in the same way: for the KCNH2-like M1-only assay it
moved the bias from {k10['P2a'].bias:+.3f} to {k10['P3'].bias:+.3f} at a true prior of 0.10 and from
{k40['P2a'].bias:+.3f} to {k40['P3'].bias:+.3f} at 0.40.</p>
<p><b>In the real datasets the rule matters for few.</b> Of the {n_all} datasets, {n_full} kept every bootstrap fit,
{n10} lost more than 10% and {n50} more than half:</p>
<div class="card"><div class='tw'><table><thead><tr><th>Dataset</th><th>Bootstrap fits discarded</th><th>Prior</th>
<th>P/LP / B/LB / gnomAD / syn.</th></tr></thead><tbody>{lost_rows}</tbody></table></div></div>
<p>One dataset sits exactly at the EM start: {em_txt or "none"}. With a single P/LP variant, the prior is not
identifiable and the reported 0.500 is the EM's starting value; the dataset receives no evidence, so this affects only
the reported prior.</p>
{start_html}
<div class="note"><p><b>Recommendation.</b> No rule is better across settings, so the current rule can stay for
this revision, with its limitation stated: when the controls cannot identify the prior, discarding boundary estimates
biases it upward, and only {n10} of the {n_all} datasets lose more than 10% of their bootstrap fits. If a change is
made later, clipping to [0.001, 0.999] is the smallest one: it removes that bias, keeps every dataset calibrated, and
the clip bound itself does not affect the prior. Separately, datasets whose prior stays at the EM start (such as the
one above) could be flagged as having no identifiable prior, as the PU and NU modes already do, rather than reported as
0.500. Not tested here: clipping also keeps those bootstrap fits in the likelihood-ratio computation, and with
{n_boot} bootstrap fits per dataset the share of datasets with no prior under discarding is higher than with
production's 1,000.</p></div>
</section>"""


def latex_preview(tex, macros):
    """Readable preview of a LaTeX block: macros replaced by their current values, markup stripped."""
    import re
    t = re.sub(r"\\(\w+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), tex)
    t = re.sub(r"\\(\w+)(?![\w{])", lambda m: macros.get(m.group(1), m.group(0)), t)
    t = t.replace("\\excalibr", "ExCALIBR")
    t = re.sub(r"\\textcolor\{red\}\{", "", t)
    t = re.sub(r"\\(textbf|textit|subsection|caption)\{", "", t)
    t = re.sub(r"\\ref\*?\{([^}]*)\}", r"[\1]", t)
    t = re.sub(r"\\(label|includegraphics|input|begin|end)(\[[^]]*\])?\{[^}]*\}(\{[^}]*\})?", "", t)
    t = re.sub(r"\\(centering|,)", " ", t)
    t = t.replace("\\%", "%").replace("\\_", "_").replace("--", "–").replace("~", " ").replace("\\ ", " ")
    t = t.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", t)).strip()


def manuscript_edits_html(macros):
    """Before/after blocks for the manuscript and supplement (analysis/prior_simulation/manuscript_edits.py)."""
    import re
    from analysis.prior_simulation.manuscript_edits import EDITS, check_edits
    status = {i: (ok, msg) for i, ok, msg in check_edits()}
    undefined = sorted({m for _, _, _, _, _, after, _ in EDITS for m in re.findall(r"\\((?:SuppPriorSim|ResDetSub)\w+)", after)
                        if m not in macros})
    cards, group = [], None
    for i, (grp, path, loc, kind, before, after, note) in enumerate(EDITS, 1):
        if grp != group:
            cards.append(f"<h3 class='egroup'>{grp} <span>({path.name})</span></h3>")
            group = grp
        ok, msg = status[i]
        flag = "" if ok else f"<span class='bad'>Anchor not found: {html.escape(msg)}</span>"
        label = "Find" if kind == "replace" else "Insert after (on a new line)"
        repl = "Replace with" if kind == "replace" else "New text"
        cards.append(
            f"<div class='card edit'><header><h3>E{i}. {html.escape(loc)}</h3>{flag}</header>"
            + (f"<p class='enote'>{html.escape(note)}</p>" if note else "")
            + f"<div class='elabel'>{label}</div><pre class='before'>{html.escape(before)}</pre>"
            + f"<div class='elabel'>{repl}<button type='button' data-target='e{i}'>Copy LaTeX</button></div>"
            + f"<pre class='after' id='e{i}'>{html.escape(after)}</pre>"
            + f"<details><summary>Preview with current values</summary><p class='preview'>"
            + html.escape(latex_preview(after, macros)).replace("\n\n", "</p><p class='preview'>")
            + "</p></details></div>")
    warn = ("" if not undefined else
            "<div class='note'><p><b>Macros not yet in macros.tex:</b> " + ", ".join(undefined)
            + ". Rebuild A11h before pasting.</p></div>")
    return f"""<section>
<h2>Manuscript and supplement edits</h2>
<p>Paste-ready LaTeX for <code>papers/ExCALIBR_110126.tex</code> and <code>papers/ExCALIBR_supp_110126.tex</code>,
in order. New text is in <code>\\textcolor{{red}}{{}}</code>; every number is a build macro. Each “Find” anchor was
checked against the current files and occurs exactly once. After pasting, copy both files to
<code>ExCALIBR_latex_src/</code> and run <code>bash scripts/compile_check.sh</code>.</p>
{warn}
{"".join(cards)}
</section>"""


def read_macros(path=MACROS_TEX):
    """{name: rendered value} from the build's macros.tex (colour wrappers stripped)."""
    import re
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*(%.*)?$", line)
        if m:
            v = m.group(2)
            c = re.match(r"\\textcolor\{\w+\}\{(.*)\}$", v)
            out[m.group(1)] = c.group(1) if c else v
    return out


def ed3_both_determinate_dor(path=ED3_TABLE):
    """(ExCALIBR, author) DOR (P vs. B) from ED Table 3's both-determinate columns."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("DOR (P vs. B)"):
            cells = [c.strip(" \\") for c in line.split("&")]
            return cells[3], cells[4]
    return "--", "--"


# One-off check (scratchpad common_gnomad_evidence.py): the build's 61 common
# (MAF >= 1%) gnomAD variants joined to the selected calibrations.
COMMON_GNOMAD = dict(n_calibrated=60, n_benign_only=50, n_indeterminate=9, n_pathogenic=1)


def build(out_dir):
    reps = pd.read_csv(out_dir / "paired_replicates.csv")
    tests = pd.read_csv(out_dir / "paired_tests.csv")
    summ = summarize(reps)
    n_o = int(reps.groupby(KEYS)["oracle"].size().max())
    n_f = int(reps.groupby(KEYS)["fit"].count().max())

    base = sel(summ, experiment="1_clean", regime="rich", true_prior=0.25)
    clean = {reg: sel(summ, experiment="1_clean", regime=reg) for reg in ("rich", "TPK1-like", "KCNH2_Kozek-like")}
    noise_p = sel(summ, experiment="2_noise_plp")
    noise_b = sel(summ, experiment="2_noise_blb")
    mech = sel(summ, experiment="3_mech_bias")
    sep = sel(summ, experiment="4_separation", true_prior=0.25)
    cor4 = sel(summ, experiment="5_correlation", true_prior=0.25, delta=4.0)
    cor25 = sel(summ, experiment="5_correlation", true_prior=0.25, delta=2.5)

    pticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    charts_clean = "".join(panel(
        {"rich": "Rich controls (100 / 100 / 500 / 200)", "TPK1-like": "TPK1-like (8 / 1 / 50 / 20)",
         "KCNH2_Kozek-like": "KCNH2_Kozek-like (3 / 0 / 10 / 6)"}[reg],
        "True prior", (0, 0.45), (0, 0.6), [0, 0.1, 0.2, 0.3, 0.4], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        series_for(df, "true_prior", lambda r: f"true prior {r.true_prior:.2f}"), identity=True)
        for reg, df in clean.items())
    charts_bias = "".join([
        panel("Benign variants labelled P/LP", "Fraction of P/LP that are benign", (0.03, 0.22), (0.1, 0.45),
              [0.05, 0.1, 0.15, 0.2], [0.1, 0.2, 0.3, 0.4], series_for(noise_p, "eps_p", lambda r: f"ε = {r.eps_p:.2f}"),
              ref=0.25, expected=expected_for(noise_p, "eps_p")),
        panel("Pathogenic variants labelled B/LB", "Fraction of B/LB that are pathogenic", (0.03, 0.22), (0.1, 0.45),
              [0.05, 0.1, 0.15, 0.2], [0.1, 0.2, 0.3, 0.4],
              series_for(noise_b, "delta_b", lambda r: f"δ = {r.delta_b:.2f}"), ref=0.25,
              expected=expected_for(noise_b, "delta_b")),
        panel("P/LP mechanism mix (gnomAD: 25% M1)", "Fraction of P/LP acting via M1", (-0.06, 1.06), (0, 0.85),
              [0, 0.25, 0.5, 0.75, 1], [0, 0.2, 0.4, 0.6, 0.8],
              series_for(mech, "lab_m1_frac", lambda r: f"P/LP M1 fraction {r.lab_m1_frac:.2f}"), ref=0.25,
              expected=expected_for(mech, "lab_m1_frac")),
    ])
    charts_fit = "".join([
        panel("Separation (true prior 0.25)", "Abnormal shift Δ (SD)", (1.2, 4.3), (0.1, 0.45), [1.5, 2.5, 4],
              [0.1, 0.2, 0.3, 0.4], series_for(sep, "delta", lambda r: f"Δ = {r.delta:.1f}"), ref=0.25),
        panel("Correlated mechanisms, Δ = 4", "Correlation ρ", (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6],
              [0.1, 0.2, 0.3, 0.4], series_for(cor4, "rho", lambda r: f"ρ = {r.rho:.1f}"), ref=0.25),
        panel("Correlated mechanisms, Δ = 2.5", "Correlation ρ", (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6],
              [0.1, 0.2, 0.3, 0.4], series_for(cor25, "rho", lambda r: f"ρ = {r.rho:.1f}"), ref=0.25),
    ])

    # Numbers quoted in the text (shared with the manuscript build).
    K = key_numbers(reps, tests, summ)
    (b1, b2, b3, bt, n_rich_sig, n_dev, n_dev_sig, fo23, m75, np20, nb20, kc, kc_lo, kc_hi, kc_at05,
     sep15, sep25, o_bias_max, o_bias_nsig, f23_bias_max, fa1_max, o_dev_max, sep15_o_err,
     sep25_25, ws_gap_lo, ws_gap_hi, rho6_gap_max) = (
        K[x] for x in ("b1", "b2", "b3", "bt", "n_rich_sig", "n_dev", "n_dev_sig", "fo23", "m75", "np20", "nb20",
                       "kc", "kc_lo", "kc_hi", "kc_at05", "sep15", "sep25", "o_bias_max", "o_bias_nsig",
                       "f23_bias_max", "fa1_max", "o_dev_max", "sep15_o_err",
                       "sep25_25", "ws_gap_lo", "ws_gap_hi", "rho6_gap_max"))
    t_rows = (
        [(f"Clean, rich, P(Y=1) = {r.true_prior:.2f}", r) for _, r in clean["rich"].iterrows()]
        + [(f"P/LP 20% benign", np20[a]) for a in ASSAYS]
        + [(f"B/LB 20% pathogenic", nb20[a]) for a in ASSAYS]
        + [(f"P/LP 75% M1", m75[a]) for a in ASSAYS]
        + [(f"KCNH2_Kozek-like, P(Y=1) = {r.true_prior:.2f}", r) for _, r in kc.iterrows()]
    )

    resp4a = (
        "Formally, ExCALIBR estimates the prior P(Y=1) from the population sample as the mixing proportion in "
        "p<sub>S<sub>G</sub></sub>(s) = P(Y=1)·p<sub>S<sub>P</sub></sub>(s) + P(Y=0)·p<sub>S<sub>B</sub></sub>(s), where "
        "p<sub>S<sub>P</sub></sub> and p<sub>S<sub>B</sub></sub> are the assay-score densities of the pathogenic and "
        "benign control variants. When an assay measures only some of a gene's disease mechanisms, "
        "p<sub>S<sub>P</sub></sub> is itself a mixture: pathogenic variants acting through a measured mechanism score "
        "abnormally, and those acting through other mechanisms score normally. If the control variants are sampled "
        "without bias, the pathogenic variants in the population follow the same mixture, so the identity still holds "
        "and P(Y=1) remains the fraction of pathogenic variants in the population, not the fraction that score "
        "abnormally. To test this, we simulated a gene with two disease mechanisms (M1 causing 25% and M2 75% of "
        "pathogenic variants) and three assays measuring M1 only, M2 only, or both (Supplementary Section X). "
        "Each variant's effect on each mechanism was drawn from a normal distribution with unit variance and mean "
        "0, except that pathogenic variants had a mean of −4 on the mechanism they act through; the single-mechanism "
        "assays read one effect and the combined assay the lower of the two. With "
        "100 P/LP, 100 B/LB, 500 population and 200 synonymous variants and a true prior of 0.25, the three assays "
        f"estimated priors of {b1.f_m:.2f}, {b2.f_m:.2f} and {b3.f_m:.2f} with the full ExCALIBR fit "
        f"({b1.o_m:.2f}, {b2.o_m:.2f} and {b3.o_m:.2f} with the true component distributions; means of {n_f} and "
        f"{n_o} simulated datasets), although only 6% and 19% of population variants were pathogenic and scored "
        "abnormally in the M1-only and M2-only assays. Across true priors from 0.05 to 0.40, neither estimator was "
        f"biased for any assay (largest mean error {o_bias_max:.3f} with the true components and "
        f"{max(f23_bias_max, fa1_max):.3f} with the fit; none significant, two-sided Wilcoxon signed-rank test, "
        "Holm-corrected). Measuring fewer mechanisms makes the estimate less precise instead, because fewer "
        "pathogenic variants then score differently from benign ones: the standard deviation of the estimate with the "
        f"true components was {K['sd_base'][A1]:.3f} for the M1-only assay and {K['sd_base'][A3]:.3f} for the assay "
        "measuring both mechanisms. The prior therefore tracks the fraction of pathogenic variants, not the fraction "
        "that score abnormally."
    )

    def rng(lo, hi, fmt=".2f"):
        a, b = format(lo, fmt), format(hi, fmt)
        return a if a == b else f"{a}–{b}"

    sd_b, sd_15, u1, gs = K["sd_base"], K["sd_sep15"], K["unid_a1"], K["gap_sep"]
    resp4b = (
        "This argument rests on two assumptions that can fail in practice. First, the prior is identifiable only if "
        "the pathogenic and benign score distributions differ: if p<sub>S<sub>P</sub></sub> = "
        "p<sub>S<sub>B</sub></sub>, every value of P(Y=1) explains the population scores equally well. Weaker "
        "separation costs precision but not accuracy: when the mean effect of pathogenic variants on their mechanism "
        f"was −1.5 instead of −4, the standard deviation of the estimate rose to {sd_15[A1]:.3f} and {sd_15[A3]:.3f} for "
        f"the M1-only and combined assays, yet the mean estimate stayed within {sep15_o_err:.3f} of the truth. When "
        "the distributions do not differ at all, the prior cannot be estimated, and the EM algorithm stays at its "
        "starting value of 0.5: in an assay in which no P/LP control scores abnormally, the mean estimate was "
        f"{u1.o_m:.2f}, and {100 * u1.at05:.0f}% of fitted bootstrap iterations returned exactly 0.5. Too few controls "
        "have a similar effect: with 3 P/LP, 0 B/LB, 10 population and 6 synonymous variants (the counts cited for "
        f"KCNH2_Kozek_Glazer_2020), estimates were {rng(kc_lo, kc_hi)} too high. "
        "Fitting the component distributions adds little bias. At a true prior of 0.25, the ExCALIBR fit matched "
        f"the true-component estimates on the same simulated datasets to within {K['fo23_sep25']:.3f} for the M2-only "
        "and combined assays at every separation. For the M1-only assay, it was higher by "
        f"{gs[(0.25, 4.0)][0]:.2f}, {gs[(0.25, 2.5)][0]:.2f} and {gs[(0.25, 1.5)][0]:.2f} at mean effects of −4, "
        "−2.5 and −1.5, significantly so only at −2.5. Because a fixed excess matters more for a small prior, we "
        f"repeated these settings at a true prior of 0.10, where the M1-only excess was {gs[(0.1, 2.5)][0]:.2f} at "
        f"−2.5 and {gs[(0.1, 1.5)][0]:.2f} at −1.5 ({gs[(0.1, 4.0)][0]:.2f} at −4). "
        "Second, the argument assumes that the control variants are sampled without bias. Mislabelled controls shift "
        f"the prior modestly: from a true 0.25 to "
        f"{rng(min(r.f_m for r in np20.values()), max(r.f_m for r in np20.values()))} when 20% of P/LP variants are "
        f"benign, and to {rng(min(r.f_m for r in nb20.values()), max(r.f_m for r in nb20.values()))} when 20% of B/LB "
        "variants are pathogenic. P/LP controls that are a biased sample of pathogenic variants have larger effects. "
        "When 75% of P/LP variants act through M1 but only 25% of pathogenic population variants do, the M2-only "
        f"assay estimates a prior of {m75[A2].f_m:.2f} and the M1-only assay {m75[A1].f_m:.2f}, while the assay "
        f"measuring both mechanisms is unaffected ({m75[A3].f_m:.2f}). An "
        "inflated prior therefore indicates controls that are too few, poorly separated or unrepresentative, not an "
        "assay that measures only part of a gene's disease mechanisms."
    )

    M = read_macros()
    dor_both = ed3_both_determinate_dor()
    resp_det = (
        "ExCALIBR's lower determinate rate does not come at the expense of correct calls. Of the "
        f"{M['ResDetSubAuthOnlyN']} ClinVar P/LP and B/LB variant effect measurements classified only by the authors, "
        f"author annotations had an accuracy of {M['ResDetSubAuthOnlyAcc']} (MCC {M['ResDetSubAuthOnlyMCC']}), "
        f"mostly because P/LP variants were classified as functionally normal (sensitivity {M['ResDetSubAuthOnlySens']}); "
        f"ExCALIBR left these indeterminate. Conversely, the {M['ResDetSubExcOnlyN']} measurements classified only by "
        f"ExCALIBR had an accuracy of {M['ResDetSubExcOnlyAcc']} (MCC {M['ResDetSubExcOnlyMCC']}). We now report this "
        "breakdown in Supplementary Table X."
    )
    resp_dor = (
        f"The diagnostic odds ratio (DOR) is {M['ResPerfDORExcalibr']} for ExCALIBR vs. {M['ResPerfDORAuthor']} for author "
        "annotations (pathogenic vs. benign calls). When indeterminate calls are counted against each method, ExCALIBR "
        f"remains ahead for both pathogenic (DOR {M['ResPerfDORPathExcalibr']} vs. {M['ResPerfDORPathAuthor']}) and "
        f"benign ({M['ResPerfDORBenExcalibr']} vs. {M['ResPerfDORBenAuthor']}) evidence. Restricted to the variants where "
        f"both methods made a determinate call, the DOR is {dor_both[0]} vs. {dor_both[1]}, and with each variant counted "
        f"once across assays it is {M['ResDedupDORExcalibr']} vs. {M['ResDedupDORAuthor']}."
    )
    cg = COMMON_GNOMAD
    resp_gnomad = (
        "Moreover, ExCALIBR assigns these common variants almost exclusively benign evidence: of the "
        f"{cg['n_calibrated']} that are calibrated, {cg['n_benign_only']} receive only benign evidence, "
        f"{cg['n_indeterminate']} receive no evidence in any assay, and {cg['n_pathogenic']} (a synonymous KCNH2 variant, "
        "p.Leu564=) receives pathogenic evidence in a single assay. The remaining variant is excluded by the authors' "
        "quality flag."
    )
    n_boot, n_restarts = K["n_boot"], K["n_restarts"]
    F = sd1_prior_facts()
    fp = F["priors"]
    tardbp = next(r for r in F["pu"] if r["dataset"] == "TARDBP_Bolognesi_Faure_2019")
    _p01_keys = ("ResPriorFixedNumBothDet", "ResPriorFixedNumFlipped", "ResPriorFixedAccBothDet", "ResPriorDetMedEst",
                 "ResPriorDetMedFixed", "ResCmpAccMedExcalibr", "ResCmpAccMedPriorFixedAll", "ResPriorHighNumDatasets",
                 "ResPriorHighVusStrongEst", "ResPriorHighVusStrongFixed", "ResPriorHighPlpBenignEst",
                 "ResPriorHighPlpBenignFixed", "NumDatasets")
    if not all(k in M for k in _p01_keys):  # numbers come from the manuscript build (A9e); until it has run:
        p01_text = ("Finally, Extended Data Fig. 4 shows that using priors greater than 10% largely does not reduce "
                    "ExCALIBR's accuracy. [Fixed-prior comparison over all datasets pending the A9 rebuild.]")
    else:
        _flip = "none" if M["ResPriorFixedNumFlipped"] == "0" else M["ResPriorFixedNumFlipped"]
        p01_text = (
            f"Finally, we reran ExCALIBR on all {M['NumDatasets']} datasets with the prior fixed at 10%. The two runs "
            f"never assigned evidence in opposite directions: of the {M['ResPriorFixedNumBothDet']} P/LP and B/LB "
            f"variants that received evidence under both priors, {_flip} changed direction, so accuracy on these "
            f"variants was identical ({M['ResPriorFixedAccBothDet']}). The estimated priors changed only the strength "
            "of evidence and whether a variant received any; per dataset, they assigned evidence to slightly more "
            f"control variants (median determinate rate {M['ResPriorDetMedEst']}% vs. {M['ResPriorDetMedFixed']}%) "
            f"at the same accuracy (median {M['ResCmpAccMedExcalibr']} vs. {M['ResCmpAccMedPriorFixedAll']}; "
            f"Extended Data Fig. 4G–I). In the {M['ResPriorHighNumDatasets']} datasets with estimated priors above "
            f"{M['MethTavtigianCMinPrior']}, the estimated priors assigned strong or stronger pathogenic evidence to "
            f"fewer variants of uncertain significance than a 10% prior ({M['ResPriorHighVusStrongEst']}% vs. "
            f"{M['ResPriorHighVusStrongFixed']}%), and fewer P/LP variants received benign evidence "
            f"({M['ResPriorHighPlpBenignEst']} vs. {M['ResPriorHighPlpBenignFixed']}). Higher estimated priors "
            "therefore do not inflate pathogenic evidence or reduce ExCALIBR's accuracy.")
    resp6 = (
        "We are confident that there are no theoretical problems in how we estimate priors if the control variants "
        "are unbiased and the assay provides sufficient separation of controls. Like mentioned in the response to "
        "comment 4, this estimation becomes less precise as separation between controls decreases. This same "
        "analysis (Supplementary Section 3X) shows when priors are inaccurate. Specifically, a small population "
        "sample reduces precision of the estimate, a small P/LP sample can bias it upward for an assay that detects "
        "only a minority of pathogenic variants, and P/LP controls that over-represent (with respect to the "
        "population) mechanisms that the assay does not measure inflate the estimate. We agree that the estimated "
        "priors are not the conventional 10%; it is a dataset-specific estimate."
        "<br><br>"
        "When we look at Figure 2, we can see that 18.4%, 14.6%, 24.9% and 5.5% for BRCA1, G6PD, PTEN, and DDX3X, "
        "respectively, are all quite reasonable estimates. These numbers are also supported in the literature, "
        "although scarcely. For example, in one publication by Clark et al. (2018), PMC6034809, they picked all rare "
        "NAGLU variants from ExAC (at the time it contained 60,000 sequenced exomes) and established that of the 189 "
        "variants in the database, 24 were known to be pathogenic and 41 of the remaining 165 were experimentally "
        "confirmed to have enzymatic activity consistent with pathogenic variants. This sets the prior to "
        "(24+41)/189 = 34.4%. Similarly, Trinidad et al. (2023), PMC10360315, assayed all 230 ARSA missense variants "
        "in gnomAD, among others, and found that 36% of the variants of unknown significance they tested had "
        "activity in the pathogenic range. In autosomal recessive genes such as these, a substantial fraction of "
        "population variants can therefore be pathogenic, so higher priors are not implausible, especially for "
        "disease-enriched domains."
        "<br><br>"
        "The values cited by the reviewer come from the first version of the manuscript. In the revised analysis of "
        f"{F['n']} datasets, the median prior is {F['median']:.3f} ({F['n20']}, {F['n30']} and {F['n40']} datasets at "
        "or above 0.2, 0.3 and 0.4). For the datasets the reviewer lists, the revised priors are "
        f"{fp['TARDBP_Bolognesi_Faure_2019']:.3f} (TARDBP_Bolognesi_Faure_2019), "
        f"{fp['SCN5A_Glazer_2020']:.3f} (SCN5A_Glazer_2020), {fp['TPK1_Weile_2017']:.3f} (TPK1_Weile_2017), "
        f"{fp['ASPA_Grønbæk-Thygesen_2024_abundance']:.3f} and {fp['ASPA_Grønbæk-Thygesen_2024_toxicity']:.3f} "
        "(ASPA_Grønbæk-Thygesen_2024 abundance and toxicity), "
        f"{fp['GCK_Gersing_2023_complementation']:.3f} (GCK_Gersing_2023_complementation), "
        f"{fp['GCK_Gersing_2024_abundance']:.3f} (GCK_Gersing_2024_abundance), "
        f"{fp['KCNH2_Jiang_2022']:.3f} (KCNH2_Jiang_2022) and {fp['KCNH2_Kozek_Glazer_2020']:.3f} "
        f"(KCNH2_Kozek_Glazer_2020). The highest of these, TARDBP_Bolognesi_Faure_2019 "
        f"({fp['TARDBP_Bolognesi_Faure_2019']:.3f}), is the only one of the {F['n']} datasets without any B/LB or "
        f"synonymous variants; it has {tardbp['n_plp_fit']} P/LP and {tardbp['n_gnomad_fit']} population variants. "
        "Its prior is therefore estimated from the minimum ratio of the population to the pathogenic score density "
        "(Supplementary Section 1.7), not with the maximum-likelihood (EM) estimator applied to all other datasets. "
        "Because we lack any benign sample for this dataset, this algorithm rests on the assumption that P/LP "
        "variants are well-separated from functionally normal variants. Assuming the majority of gnomAD variants are "
        "putatively neutral, this assumption does not hold in this dataset because P/LP scores overlap almost "
        "entirely with the population scores, causing inflated prior estimates. This estimator is an upper bound on "
        "the prior and equals the prior only if some scores are attained by pathogenic but essentially no benign "
        "variants. Therefore, it overestimates the prior when pathogenic and benign scores overlap, and the prior is "
        "not identifiable from this dataset’s controls."
        "<br><br>"
        "Each prior is estimated from one assay's scores and controls, so it is a property of the dataset rather "
        "than of the gene alone. Assays of the same gene differ when they measure different properties or contain "
        "different control variants; combining them into a gene-level prior is beyond the scope of this work. When "
        "P/LP controls act through mechanisms the assay does not measure and users have reason to believe the "
        "standard estimate would be biased (see above paragraph and response to comment 4), we provide the option "
        "of a pathomechanism-aware prior. By modeling the P/LP sample as a mixture of mechanism-detected and "
        "mechanism-mismatched variants, we can estimate the probability that a variant is pathogenic through a "
        "mechanism the assay detects, which never exceeds the standard prior and may be closer to the "
        "mechanism-specific pretest probability the reviewer describes (Supplementary Section 1.8). Curating "
        "phenotype-matched control sets would address a similar issue; however, this is not yet feasible at scale "
        "due to a lack of structured phenotype data for ClinVar variants. Additionally, this would limit the "
        "applicability of benign evidence if mechanism-mismatched P/LPs are excluded as benign evidence would rest "
        "on only the subset of mechanisms measured. Moreover, the priors the reviewer lists do not inflate "
        "pathogenic evidence in the Bayesian ACMG/AMP framework. Since the evidence thresholds are set by the "
        f"constant c, which is smallest at a prior of {M['MethTavtigianCMinPrior']} and increases above it, priors "
        "above that value, including every one the reviewer cites (0.300–0.887), make pathogenic evidence harder "
        "(not easier) to reach than a prior at the minimum (Methods). " + p01_text
    )
    # First-version priors of the datasets the reviewer lists (archived response PDF, ExCALIBR_110326).
    v1 = {"TARDBP_Bolognesi_Faure_2019": 0.887, "SCN5A_Glazer_2020": 0.872, "TPK1_Weile_2017": 0.612,
          "ASPA_Grønbæk-Thygesen_2024_abundance": 0.517, "ASPA_Grønbæk-Thygesen_2024_toxicity": 0.461,
          "GCK_Gersing_2023_complementation": 0.422, "GCK_Gersing_2024_abundance": 0.409,
          "KCNH2_Jiang_2022": 0.392, "KCNH2_Kozek_Glazer_2020": 0.300}
    resp52 = (
        f"As explained in our response to comment 12, the {M.get('ResClinvarVUSPathPct', '17')}% of VUS in Figure 3B "
        "received pathogenic evidence of any strength, not a likely pathogenic or pathogenic classification: in the "
        "Bayesian ACMG/AMP point system (Tavtigian et al., 2020), a likely pathogenic classification needs 6 points, "
        "which supporting (1), moderate (2) or strong (4) functional evidence cannot reach alone. The rate is also "
        "close to the 17% of ClinVar missense VUS that scored pathogenic-like in the PTEN assay of Mighell et al. "
        "(2018). We now report the positive predictive value and false discovery rate of ExCALIBR's evidence "
        "(Extended Data Table 3). Whether high priors inflate pathogenic evidence is addressed in our response to "
        "comment 6."
    )
    v1_rows = "".join(
        f"<tr><td>{d}</td><td class='num'>{v:.3f}</td><td class='num'>{fp[d]:.3f} {'↓' if fp[d] < v else '↑'}</td></tr>"
        for d, v in v1.items())
    v1_table = ("<div class='card'><p>Priors of the datasets the reviewer lists, first version (archived response) vs. "
                "revision. First version: median 0.237 across 111 datasets (67 ≥ 0.2, 25 ≥ 0.3, 12 ≥ 0.4); revision: "
                f"median {F['median']:.3f} across {F['n']} ({F['n20']}, {F['n30']}, {F['n40']}). The dataset set also "
                "changed (111 → " + str(F['n']) + "), so the drop in the median is not a method effect alone.</p>"
                "<div class='tw'><table><thead><tr><th>Dataset</th><th>First version</th><th>Revised</th></tr></thead>"
                f"<tbody>{v1_rows}</tbody></table></div></div>")
    resp53 = (
        "Assigning evidence should not be equated with the 10% pathogenic resolution of VUS that the reviewer cites "
        "above (responses to comment 12 and point 52). Most of this evidence is below the 6 points needed for a "
        "likely pathogenic classification and must be combined with other evidence, so pathogenic evidence for "
        f"{M.get('ResClinvarVUSPathPct', '17')}% of VUS can be consistent with fewer than 10% of VUS being classified "
        "as pathogenic, and the PTEN assay of Mighell et al. (2018) itself found 17% of ClinVar missense VUS to score "
        "pathogenic-like. We validate the evidence independently of the calibration in two ways. Every ClinVar P/LP "
        "and B/LB variant receives evidence out of bag, from fits that did not use it, and we now report the "
        "positive predictive value and false discovery rate of this evidence (Extended Data Table 3). In the All of "
        "Us cohort, carriers of variants assigned pathogenic evidence are at increased disease risk (Fig. 5C; "
        "Supplementary Data 4; response to comment 8). Validating individual VUS calls requires clinical "
        "reclassification, which we are pursuing with ClinGen and the VCEPs (response to comment X)."
    )

    pending = "" if n_restarts > 1 else (
        "<div class='note'><p><b>Update pending.</b> The fitted-arm results on this page use 1 restart per "
        "bootstrap. With one restart, most M1-only fits miss the small abnormal cluster and settle on a local optimum "
        "that splits the normal peak, which is what makes the M1-only fit run high. With 20 restarts every fit finds "
        "the cluster and the bias disappears (0.110 vs. 0.109 with the true components, true P(Y=1) = 0.10). A rerun "
        "with 8 restarts is in progress. Do not paste response blocks 1–3 until this page is updated; blocks 4–6 are "
        "final.</p></div>")

    css = """
:root{--bg:#f6f7f9;--surface:#ffffff;--ink:#14171c;--muted:#565d68;--faint:#8a919c;--line:#dfe3e8;--grid:#eceff3;
--a1:#2a78d6;--a2:#eb6834;--a3:#1baf7a;--warn:#9a5b00;--warnbg:#fff4e0;--okbg:#e3f3ea;
--f-display:"Source Serif 4",Georgia,serif;--f-body:"Public Sans",system-ui,sans-serif;--f-mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121418;--surface:#1b1e24;--ink:#eceef1;--muted:#a9b0ba;
--faint:#7a828d;--line:#2d323a;--grid:#262a31;--a1:#3987e5;--a2:#d95926;--a3:#199e70;--warn:#f0b35a;--warnbg:#2e2412;--okbg:#16301f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121418;--surface:#1b1e24;--ink:#eceef1;--muted:#a9b0ba;--faint:#7a828d;--line:#2d323a;
--grid:#262a31;--a1:#3987e5;--a2:#d95926;--a3:#199e70;--warn:#f0b35a;--warnbg:#2e2412;--okbg:#16301f;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.55}
main{max-width:1080px;margin:0 auto;padding-block:32px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:28px}
h1,h2,h3{font-family:var(--f-display);font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2rem;line-height:1.15}h2{font-size:1.35rem}h3{font-size:1.05rem}
p{margin:0;max-width:72ch}.lede{color:var(--muted);font-size:1.02rem}
section{display:grid;gap:12px;min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}
.eyebrow{font-family:var(--f-mono);font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.key{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.key .card{display:grid;gap:6px}.key b{font-family:var(--f-display);font-size:1.02rem}
.scroll{overflow-x:auto}.diagram{width:100%;min-width:760px;height:auto;display:block}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px}
.chart{width:100%;height:auto;display:block}
.node{fill:var(--surface);stroke:var(--line)}.ntitle{font:600 14px var(--f-body);fill:var(--ink)}
.nbody{font:12.5px var(--f-body);fill:var(--muted)}.b{font-weight:600;fill:var(--ink)}
.flow{fill:none;stroke:var(--faint);stroke-width:1.5}.arrowhead{fill:var(--faint)}
.dB{fill:none;stroke:var(--faint);stroke-width:2}.dM1{fill:none;stroke:var(--a1);stroke-width:2;stroke-dasharray:5 3}
.dM2{fill:none;stroke:var(--a2);stroke-width:2;stroke-dasharray:2 3}
.abn{fill:none;stroke:var(--ink);stroke-width:1.5}.legdot{fill:var(--muted);stroke:var(--muted)}
.legdot.hollow{fill:var(--surface)}
.ptitle{font:600 12.5px var(--f-body);fill:var(--ink)}.tick{font:11px var(--f-mono);fill:var(--muted)}
.alabel{font:11.5px var(--f-body);fill:var(--muted)}.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--faint);stroke-width:1}.truth{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4}
.expect{fill:none;stroke-width:1.5;opacity:.55}.err{stroke-width:2;stroke-linecap:round}.dot{stroke-width:2}
.hit{fill:transparent}.pt:hover .dot{stroke-width:3}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.85rem;color:var(--muted);align-items:center}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:.8rem}.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums;font-size:.82rem}
.resp{display:grid;gap:8px}.resp .text{font-size:.95rem;max-width:none}
.resp header{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
button{font:600 .82rem var(--f-body);color:var(--ink);background:var(--bg);border:1px solid var(--line);
border-radius:6px;padding:5px 12px;cursor:pointer}button:hover{border-color:var(--muted)}
button:focus-visible{outline:2px solid var(--a1);outline-offset:2px}
.edit{display:grid;gap:8px}.edit header h3{margin:0}.egroup{margin:18px 0 0}.egroup span{color:var(--muted);font-weight:400;font-size:.85rem}
.elabel{display:flex;justify-content:space-between;align-items:center;gap:10px;font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.edit pre{margin:0;padding:10px 12px;border:1px solid var(--line);border-radius:6px;white-space:pre-wrap;word-break:break-word;
font-family:var(--f-mono);font-size:.78rem;line-height:1.5;max-height:420px;overflow:auto}
.edit pre.before{background:var(--bg);color:var(--muted)}.edit pre.after{background:var(--surface);color:var(--ink)}
.enote{font-size:.88rem;color:var(--muted);margin:0}.preview{font-size:.9rem;max-width:none}.bad{color:var(--warn);font-weight:600;font-size:.85rem}
details summary{cursor:pointer;font-size:.85rem;color:var(--muted)}
.sub{font-size:.75rem;color:var(--muted);font-weight:400}
.note{background:var(--warnbg);border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:6px;padding:12px 14px}
ul{margin:0;padding-left:20px;display:grid;gap:4px;max-width:76ch}code{font-family:var(--f-mono);font-size:.85em}
"""
    legend = ("<div class='legend'>" + "".join(
        f"<span><span class='sw' style='background:var({CSS_VAR[a]})'></span>{SHORT[a]}: {LONG[a]}</span>" for a in ASSAYS)
        + "<span>● true components</span><span>○ ExCALIBR fit</span><span>bars: 95% CI of the mean</span>"
          "<span>dashed: true prior</span><span>faint line: large-sample expectation</span></div>")

    def resp_block(i, title, text):
        return (f"<div class='card resp'><header><h3>{title}</h3><button type='button' id='copy{i}' "
                f"data-target='r{i}'>Copy text</button></header><p class='text' id='r{i}'>{text}</p></div>")

    edits_html = manuscript_edits_html(M)
    from analysis.prior_simulation.summary import load_sample_size, sample_size_summary
    _ss_reps = load_sample_size(out_dir)
    sample_size_section = ""
    if _ss_reps is not None:
        _ss = sample_size_summary(_ss_reps)
        sample_size_section = sample_size_html(_ss).replace("{legend}", legend).replace(
            "{findings}", sample_size_findings(_ss, _ss_reps))
    from analysis.prior_simulation.summary import load_policy, policy_summary
    _pol_reps = load_policy(out_dir)
    policy_section = ""
    if _pol_reps is not None:
        policy_section = policy_html(policy_summary(_pol_reps), _pol_reps, sd1_bootstrap_kept())

    page = f"""<title>Prior Recovery Simulation</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Public+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap">
<style>{css}</style>
<main>
<header style="display:grid;gap:10px">
<span class="eyebrow">ExCALIBR revision · Reviewer 3, comments 4 and 6</span>
<h1>Does an assay that measures one disease mechanism estimate the right prior?</h1>
<p class="lede">A simulated gene has two disease mechanisms and three assays that measure one or both. We know the true
prior, so we can check what ExCALIBR estimates, with and without fitting, and what pushes the estimate up.</p>
</header>

{pending}
<section class="key">
<div class="card"><span class="eyebrow">Clean controls</span><b>All three assays recover the true prior</b>
<p>True 0.25: estimates {b1.f_m:.2f} (M1 only), {b2.f_m:.2f} (M2 only), {b3.f_m:.2f} (both). With the true components there is
no bias at any prior from 0.05 to 0.40 (largest error {o_bias_max:.3f}), and none with the fit (largest
{max(f23_bias_max, fa1_max):.3f}). The fit runs high only for the minority-mechanism assay at weak separation.</p></div>
<div class="card"><span class="eyebrow">What inflates it</span><b>P/LP from the wrong mechanism mix</b>
<p>When P/LP over-represent the mechanism an assay does not measure, the prior rises to {m75['A2 (M2 only)'].f_m:.2f}.
The assay measuring both mechanisms stays at {m75['A3 (M1+M2)'].f_m:.2f}.</p></div>
<div class="card"><span class="eyebrow">Also inflates it</span><b>Too few controls to identify the prior</b>
<p>With 3 P/LP, 0 B/LB and 10 gnomAD variants, estimates are biased upward by {kc_lo:.2f}–{kc_hi:.2f}. Label noise
moves the prior only modestly, as theory predicts.</p></div>
</section>

<section>
<h2>Design and estimates</h2>
<p>Every replicate draws one set of variants. Each variant's latent effect on the two mechanisms is read by all three
assays, so assays are compared on the same data. Right: each assay's class score distributions, and its prior
estimates (rich controls, true prior 0.25) against the fraction of gnomAD variants that are pathogenic and score
abnormally, which is what the reviewer suggests the prior measures.</p>
<div class="card scroll">{diagram(base, n_o, n_f)}</div>
</section>

<section>
<h2>What was done</h2>
<ul>
<li><b>Two estimators.</b> <i>True components</i>: no fit; each sample's density uses the exact class distributions,
with its class weights estimated from the data, and the prior comes from the production prior EM on the gnomAD sample
(<code>estimate_prior_from_class_densities</code>, average of B/LB and synonymous as the benign reference, as in
production). <i>ExCALIBR fit</i>: the production fit (<code>Fit.run</code>, 2 Gaussian components,
{n_restarts} restarts per bootstrap with the best kept by held-out log-likelihood, as in production) and
<code>get_fit_prior</code>, median over {n_boot} bootstraps.</li>
<li><b>Settings.</b> Clean controls at true priors 0.05–0.40 with rich and sparse controls; benign variants in P/LP
and pathogenic variants in B/LB at 5–20%; P/LP mechanism mix from 0% to 100% M1 with gnomAD fixed at 25%; separation
Δ of 1.5, 2.5 and 4 SD; correlation ρ of 0, 0.3 and 0.6.</li>
<li><b>Tests.</b> {n_o} replicate datasets per setting with the true components; the first {n_f} also get the ExCALIBR fit, so fit vs. true components is paired. Two-sided Wilcoxon signed-rank tests, Holm-corrected within each
experiment: bias against the true prior; deviation from the large-sample expectation; fit against true components
(paired); and between assays (paired, on the estimate and on its absolute error).</li>
</ul>
</section>

<section>
<h2>Clean controls: estimate vs. true prior</h2>
{legend}
<div class="card charts">{charts_clean}</div>
<p>With rich controls, the true-component estimates are unbiased for every assay (largest mean error
{o_bias_max:.3f}), and so is the fit (largest {max(f23_bias_max, fa1_max):.3f}; none significant). The M1-only
assay is less
precise than the others (significantly larger absolute error). With TPK1-like controls the M1-only assay is unstable. With KCNH2_Kozek-like controls (3 P/LP, 0 B/LB) every assay
is biased upward at low priors. When the few P/LP variants contain none that score abnormally, the prior EM cannot move
from its starting value of 0.5 ({100 * kc_at05:.0f}% of M1-only bootstraps in this regime).</p>
</section>

<section>
<h2>Biased controls: label noise and mechanism mix</h2>
{legend}
<div class="card charts">{charts_bias}</div>
<p>With the true components, each estimate matches its large-sample expectation to within {o_dev_max:.3f}. Label
noise moves the prior modestly: up to about 0.32 when 20% of P/LP are benign, and down to about 0.16–0.19 when 20% of B/LB are
pathogenic. The mechanism mix of P/LP is what produces large errors, in opposite directions for the two single-mechanism
assays. The assay measuring both mechanisms is unaffected. Where the P/LP contain no variant the assay can detect (0%
M1 for A1, 100% for A2), the prior is unidentifiable and the estimate stays at 0.5.</p>
</section>

<section>
<h2>Fitting: separation and correlated mechanisms</h2>
{legend}
<div class="card charts">{charts_fit}</div>
<p>For the M2-only and combined assays, the fit matches the true-component estimate on the same datasets to within
{K['fo23_sep25']:.3f} at a true prior of 0.25. For the M1-only assay, the fit exceeds it by
{K['gap_sep'][(0.25, 4.0)][0]:.2f}, {K['gap_sep'][(0.25, 2.5)][0]:.2f} and {K['gap_sep'][(0.25, 1.5)][0]:.2f} at Δ = 4, 2.5 and
1.5 (significant only at 2.5); at a true prior of 0.10 the excess is {K['gap_sep'][(0.1, 2.5)][0]:.2f} at Δ = 2.5 and
{K['gap_sep'][(0.1, 1.5)][0]:.2f} at 1.5. With the true components, Δ = 1.5 keeps every assay within {sep15_o_err:.3f} of the
truth at 0.25. Correlation between mechanisms removes the excess (within {rho6_gap_max:.3f} at ρ = 0.6), because M2
variants then shift that assay too.</p>
</section>

{sample_size_section}
{policy_section}

<section>
<h2>Key numbers</h2>
<p>Mean prior estimate [95% CI of the mean]: {n_o} simulated datasets with the true components, {n_f} with the fit.</p>
<div class="card">{results_table(t_rows)}</div>
</section>

<section>
<h2>Response text</h2>
<p>Paste-ready for the response document. Replace "Supplementary Section X" and "Supplementary Table X" with the
references once placed. Blocks 1–2 fill the gaps in the comment 4 draft, block 3 is for comment 6 (with the first-version prior table below it) and blocks 7–8 for summary points 52–53.
Blocks 4–8 take their numbers from the manuscript build.</p>
{resp_block(1, "Comment 4: theory and simulation (replaces “WE COULD WRITE…”)", resp4a)}
{resp_block(2, "Comment 4: the two remaining concerns", resp4b)}
{resp_block(3, "Comment 6: priors are too high", resp6)}
{v1_table}
{resp_block(7, "Point 52: “The reported pathogenic-evidence rate is highly concerning” (VUS rate; replaces the current response)", resp52)}
{resp_block(8, "Point 53: “Assigning pathogenic evidence to 17% of ClinVar VUS requires stronger independent validation” (extends the current response)", resp53)}
{resp_block(4, "Lower determinate rate (ExCALIBR vs. author)", resp_det)}
{resp_block(5, "DOR (continues “…DOR analyses probably best reflect the advances from our paper.”)", resp_dor)}
{resp_block(6, "Common gnomAD variants (continues “…the influence of common variants is negligible.”)", resp_gnomad)}
</section>

{edits_html}

<section>
<h2>Known limitation</h2>
<div class="note"><p>When the labelled P/LP sample contains no variant the assay can detect, the pathogenic and benign
densities on gnomAD are identical. The production prior EM then returns its starting value of 0.5, which the discard
rule (prior ≤ 0.001 or ≥ 0.999) does not catch. This drives the sparse-control inflation above. The supplement
describes it as a limitation; the pipeline is unchanged.</p></div>
</section>

<section>
<p class="lede" style="font-size:.85rem">Code: <code>analysis/prior_simulation/</code> (<code>simulate_priors.py</code>,
<code>test_priors.py</code>, <code>report.py</code>). Data: <code>paired_replicates.csv</code>, <code>paired_tests.csv</code>.</p>
</section>
</main>
<script>
document.querySelectorAll("button[data-target]").forEach(function (btn) {{
  btn.addEventListener("click", function () {{
    var el = document.getElementById(btn.dataset.target);
    var text = el.innerText;
    var label = btn.dataset.label || (btn.dataset.label = btn.textContent);
    function done(msg) {{ btn.textContent = msg; setTimeout(function () {{ btn.textContent = label; }}, 1600); }}
    try {{
      navigator.clipboard.writeText(text).then(function () {{ done("Copied"); }}, function () {{ select(el); done("Selected"); }});
    }} catch (e) {{ select(el); done("Selected"); }}
  }});
}});
function select(el) {{ var r = document.createRange(); r.selectNodeContents(el); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }}
</script>
"""
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "prior_simulation_report.html")
    args = ap.parse_args()
    args.out.write_text(build(args.data_dir))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
