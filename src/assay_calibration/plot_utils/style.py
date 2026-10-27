"""Shared figure style for the ExCALIBR and ExCALIBR-MV papers: one source
for the sample colors, the evidence-point ramp, the threshold colors and the
font stack, so every figure in the repository uses the same meanings.

Values are those of the ExCALIBR paper figures (analysis/fit_hist_snv_plot.py).
"""

FONT_SANS = ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]

SAMPLE_NAMES = ["ClinVar PLP", "ClinVar BLB", "gnomAD", "Synonymous"]
SAMPLE_COLORS = ["#CA7682", "#1D7AAB", "#A0A0A0", "#6BAA75"]
SAMPLE_ALPHAS = [0.7, 0.5, 0.15, 0.4]
SAMPLE_COLOR = {"P/LP": "#CA7682", "B/LB": "#1D7AAB", "gnomAD": "#A0A0A0", "Synonymous": "#6BAA75"}

# Evidence points, -8 (strongest benign) to +8 (strongest pathogenic); 0 is
# indeterminate.
STRENGTH_COLOR = {
    -8: "#4b91a6", -7: "#5DA3BD", -6: "#6FAACE", -5: "#74ABCE",
    -4: "#7ab5d1", -3: "#99c8dc", -2: "#d0e8f0", -1: "#e4f1f6",
    0: "#e0e0e0",
    1: "#e6b1b8", 2: "#d68f99", 3: "#ca7682", 4: "#b85c6b",
    5: "#B1535F", 6: "#AA4E58", 7: "#A2484F", 8: "#943744",
}

BENIGN_THRESHOLD_COLOR = "#2166AC"
PATHOGENIC_THRESHOLD_COLOR = "#B2182B"

# Evidence floored to the standard levels (+-1, +-2, +-4, +-8), as in the
# ExCALIBR evidence-distribution figures
# (plot_utils.utils.plot_combined_evidence_distributions); drawn left to right
# from strongest pathogenic to strongest benign.
STRENGTH_COLOR_FLOORED = {
    8: "#943744", 4: "#B85C6B", 2: "#D68F99", 1: "#E6B1B8",
    0: "#E0E0E0",
    -1: "#99C8DC", -2: "#7AB5D1", -4: "#4B91A6", -8: "#2E6B7E",
}
FLOORED_POINT_ORDER = [8, 4, 2, 1, 0, -1, -2, -4, -8]

# Confusion matrices (analysis/clingen.py): per-column gradients for the
# benign, indeterminate and pathogenic columns, scaled to each row's maximum.
CONFUSION_BLUE = ["#F0F8FC", "#99C8DC", "#7AB5D1", "#4B91A6", "#2E6B7E"]
CONFUSION_GRAY = ["#F5F5F5", "#CCCCCC", "#999999", "#666666"]
CONFUSION_RED = ["#FCF0F2", "#E6B1B8", "#D68F99", "#B85C6B", "#943744"]
