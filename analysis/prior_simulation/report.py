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
          expected=None, ylabel="Estimated prior"):
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
    for t in xticks:
        out.append(f'<text x="{X(t):.1f}" y="{mt + ph + 15}" class="tick" text-anchor="middle">{fmt_tick(t)}</text>')
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
        "p<sub>S<sub>P</sub></sub> and p<sub>S<sub>B</sub></sub> are the "
        "assay-score densities of the pathogenic and benign control variants. When an assay measures only one of a "
        "gene's disease mechanisms, p<sub>S<sub>P</sub></sub> is itself a mixture: pathogenic variants acting through the measured "
        "mechanism score abnormally, and those acting through other mechanisms score normally. If the control variants "
        "are sampled without bias, the pathogenic variants in the population follow the same mixture, so the identity "
        "still holds and P(Y=1) remains the fraction of pathogenic variants in the population, not the fraction that score "
        "abnormally. To test this, we simulated a gene with two disease mechanisms (M1 causing 25% and M2 75% of "
        "pathogenic variants) and three assays measuring M1 only, M2 only, or both (Supplementary Section X). With "
        "100 P/LP, 100 B/LB, 500 population and 200 synonymous variants and a true prior of 0.25, the three assays "
        f"estimated priors of {b1.f_m:.2f}, {b2.f_m:.2f} and {b3.f_m:.2f} with the full ExCALIBR fit "
        f"({b1.o_m:.2f}, {b2.o_m:.2f} and {b3.o_m:.2f} with the true component distributions; means of {n_f} and "
        f"{n_o} simulated datasets), although only 6% and 19% of population variants scored abnormally in the M1-only and M2-only "
        f"assays. Across true priors from 0.05 to 0.40, the estimates with the true component distributions were "
        f"unbiased for every assay (largest mean error {o_bias_max:.3f}; "
        f"{'none' if o_bias_nsig == 0 else o_bias_nsig} significant, two-sided Wilcoxon signed-rank test, "
        "Holm-corrected). The ExCALIBR fit was also unbiased for every assay (largest mean error "
        f"{max(f23_bias_max, fa1_max):.3f}; none significant). Measuring fewer mechanisms therefore reduces the "
        "precision of the estimate, but the prior still tracks the fraction of pathogenic variants, not the fraction "
        "that score abnormally."
    )
    resp4b = (
        "Although this argument is theoretical, two concerns remain. First, an assay could separate pathogenic from "
        "benign variants too poorly for the mixture fit to recover the prior, for example because the likelihood "
        "surface maximized by our EM algorithm becomes flatter. In our simulations, the ExCALIBR fit matched the "
        f"estimates obtained with the true component distributions to within {fo23:.3f} for assays detecting most or "
        "all pathogenic variants, and with the true components the prior stayed within "
        f"{sep15_o_err:.3f} of the truth even at a 1.5 standard-deviation shift. The fit did run high in one case: "
        "an assay that detects only a minority (25%) of pathogenic variants and separates them weakly from benign "
        f"variants. At a 2.5 standard-deviation shift, it estimated {sep25.f_m:.2f} for a true prior of 0.10 and "
        f"{sep25_25.f_m:.2f} for 0.25 ({sep25.o_m:.2f} and {sep25_25.o_m:.2f} with the true components), because the "
        "few abnormal-scoring variants are then hard to distinguish from the tail of the normal-scoring peak. "
        "Second, the argument assumes that the control variants are sampled without bias. Mislabelled controls "
        f"shift the prior modestly and predictably: from a true 0.25 to {min(r.f_m for r in np20.values()):.2f}–"
        f"{max(r.f_m for r in np20.values()):.2f} when 20% of P/LP variants are benign, and to "
        f"{min(r.f_m for r in nb20.values()):.2f}–{max(r.f_m for r in nb20.values()):.2f} when 20% of B/LB variants are "
        "pathogenic. A larger effect arises when P/LP variants act through a different mix of mechanisms than the "
        "pathogenic variants in the population. When 75% of P/LP variants act through M1 but only 25% of pathogenic "
        f"population variants do, the M2-only assay estimates a prior of {m75['A2 (M2 only)'].f_m:.2f} and the M1-only "
        f"assay {m75['A1 (M1 only)'].f_m:.2f}, while the assay measuring both mechanisms is unaffected "
        f"({m75['A3 (M1+M2)'].f_m:.2f}). With the true components, every noise and sampling-bias estimate matched "
        f"its large-sample prediction to within {o_dev_max:.3f}. An inflated prior therefore "
        "indicates control variants that under-represent the mechanism an assay measures, not an assay that measures "
        "only part of a gene's disease mechanisms."
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
    resp6 = (
        "The priors for the four datasets in Figure 2 are 18.4%, 14.6%, 24.9% and 5.5% for BRCA1, G6PD, PTEN and "
        "DDX3X, respectively. Values of this size are also supported by the few studies that functionally "
        "characterized the variants observed in a population database. Clark et al. (2018, PMC6034809) assayed the "
        "enzymatic activity of every NAGLU missense variant of unknown significance in ExAC. Of the 189 NAGLU missense "
        "variants, 24 were known to be pathogenic, and 41 of the 164 assayed variants of unknown significance had "
        "activity consistent with pathogenic alleles (≤15% of wild type), giving a pathogenic fraction of "
        "(24 + 41)/189 = 34.4%. Similarly, Trinidad et al. (2023, Genome Biology 24:172) assayed all 230 ARSA "
        "missense variants in gnomAD, among others, and found that 36% of the variants of unknown significance they "
        "tested had activity in the pathogenic range. In autosomal recessive disease genes such as these, a "
        "substantial fraction of population variants can therefore be pathogenic, and priors of 20%–35% are not "
        "implausible. "
        "As detailed in our response to comment 4, the prior estimated by ExCALIBR is the fraction of pathogenic "
        "variants among the population variants measured by an assay, and our simulations show that it is not the "
        "fraction of population variants with pathogenic-like (abnormal) scores (Supplementary Section X). We agree "
        "that it is not the conventional clinical anchor of 0.10; it is a dataset-specific estimate from the population "
        "sample relative to the P/LP and B/LB controls. The simulations also show when an estimated prior is too high: "
        "when the P/LP controls over-represent mechanisms that the assay does not measure (an estimate of "
        f"{m75['A2 (M2 only)'].f_m:.2f} for a true P(Y=1) of 0.25), and when the controls are too few to identify the "
        "prior. With 3 P/LP, 0 B/LB, 10 population and 6 synonymous variants (the counts cited for "
        f"KCNH2_Kozek_Glazer_2020), estimates were biased upward by {kc_lo:.2f}–{kc_hi:.2f} at true priors of "
        "0.05–0.25. The values cited by the reviewer come from the first version of the manuscript. In the revised "
        "analysis of 87 datasets, the median prior is 0.196 (42, 25 and 11 datasets at or above 0.2, 0.3 and 0.4), and "
        "the priors of TPK1_Weile_2017 and KCNH2_Kozek_Glazer_2020 are 0.352 and 0.167. Moreover, the priors the "
        "reviewer lists do not inflate pathogenic evidence in the Bayesian ACMG/AMP framework: the evidence "
        f"thresholds are set by the constant c, which is smallest at a prior of {M['MethTavtigianCMinPrior']} and "
        "increases above it, so priors above that value, including every one the reviewer cites (0.300–0.887), make "
        "pathogenic evidence harder, not easier, to reach than a prior at the minimum (Methods). Finally, on the "
        f"{M['NumACMGScalerDatasets']} datasets that acmgscaler could also calibrate, fixing the prior at 0.10 gave a "
        f"median per-dataset accuracy of {M['ResCmpAccMedPriorFixed']}, compared with {M['ResCmpAccMedExcalibrAcmgSet']} "
        "with estimated priors (Extended Data Fig. 4D–F), so ExCALIBR's accuracy does not depend on high priors."
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
--a1:#2a78d6;--a2:#eb6834;--a3:#1baf7a;--warn:#9a5b00;--warnbg:#fff4e0;
--f-display:"Source Serif 4",Georgia,serif;--f-body:"Public Sans",system-ui,sans-serif;--f-mono:"JetBrains Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121418;--surface:#1b1e24;--ink:#eceef1;--muted:#a9b0ba;
--faint:#7a828d;--line:#2d323a;--grid:#262a31;--a1:#3987e5;--a2:#d95926;--a3:#199e70;--warn:#f0b35a;--warnbg:#2e2412;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121418;--surface:#1b1e24;--ink:#eceef1;--muted:#a9b0ba;--faint:#7a828d;--line:#2d323a;
--grid:#262a31;--a1:#3987e5;--a2:#d95926;--a3:#199e70;--warn:#f0b35a;--warnbg:#2e2412;color-scheme:dark}
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
<p>For assays detecting most or all pathogenic variants, the fit matches the true-component estimate to within
{fo23:.3f}. The M1-only fit runs high at weaker separation: at Δ = 2.5 it exceeds the true-component estimate by
{ws_gap_lo:.2f}–{ws_gap_hi:.2f} ({sep25.f_m:.2f} vs. {sep25.o_m:.2f} at a true prior of 0.10, {sep25_25.f_m:.2f} vs.
{sep25_25.o_m:.2f} at 0.25). With the true components, even Δ = 1.5 keeps every assay within {sep15_o_err:.3f} of the
truth, so the excess comes from fitting a small abnormal cluster that overlaps the normal peak. Correlation between
mechanisms removes it (within {rho6_gap_max:.3f} at ρ = 0.6), because M2 variants then shift that assay too.</p>
</section>

<section>
<h2>Key numbers</h2>
<p>Mean prior estimate [95% CI of the mean]: {n_o} simulated datasets with the true components, {n_f} with the fit.</p>
<div class="card">{results_table(t_rows)}</div>
</section>

<section>
<h2>Response text</h2>
<p>Paste-ready for the response document. Replace "Supplementary Section X" and "Supplementary Table X" with the
references once placed. Blocks 1–2 fill the gaps in the comment 4 draft and block 3 is for comment 6. Blocks 4–6 take
their numbers from the manuscript build.</p>
{resp_block(1, "Comment 4: theory and simulation (replaces “WE COULD WRITE…”)", resp4a)}
{resp_block(2, "Comment 4: the two remaining concerns", resp4b)}
{resp_block(3, "Comment 6: priors are too high", resp6)}
{resp_block(4, "Lower determinate rate (ExCALIBR vs. author)", resp_det)}
{resp_block(5, "DOR (continues “…DOR analyses probably best reflect the advances from our paper.”)", resp_dor)}
{resp_block(6, "Common gnomAD variants (continues “…the influence of common variants is negligible.”)", resp_gnomad)}
</section>

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
    function done(msg) {{ btn.textContent = msg; setTimeout(function () {{ btn.textContent = "Copy text"; }}, 1600); }}
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
