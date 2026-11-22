"""Build the v2 cluster bundles (data only; the user submits).

    nice -n 19 python hpc/build_v2_bundles.py {bootscheme|standard|standard_n5|kdiag|kdiag_standard} <root>
    nice -n 19 python hpc/build_v2_bundles.py {production|production_standard} <root> <components.json> "<scheme args>"

  bootscheme  Phase 2a: bootstrap-scheme comparison on real data.
              Conditions <root>/<scheme>/<subrun>/tau_0.1; schemes S0 (composite
              x pattern), S1 (stratum), S2n5 / S2n3 (stratum x pattern, nearest
              collapse, n_min 5 / 3); subruns tp53 (TP53), integrated (BRCA1),
              combined (BRCA2, TSC2). 4c, 100 bootstraps, 3 fits, tau 0.1, strata.
  standard    Standard (non-strata) model on the Phase 2a datasets and settings:
              4 composite samples, sample x pattern resampling (the default
              scheme), v2 rows, ClinVar 2025. Condition <root>/standard/<subrun>.
              Same seeds and rows as Phase 2a, so its resamples match S0's.
  standard_n5 Standard model, sample x pattern with patterns < 5 rows merged
              (S2n5's rule on the 4 samples). Condition <root>/standard_n5/<subrun>.
  kdiag_standard  K diagnostic for the standard model (4 samples, sample x pattern),
              plus a strata refit of PALB2 and GCK (<root>/integrated_strata_refit).
  production_standard  Phase 6 for the standard model (same arguments as production).
  production  Phase 6: python hpc/build_v2_bundles.py production <root> <components.json> "<scheme args>"
  kdiag       Phase 3: component diagnostic. Every MV gene set of the paper
              (integrated, tp53, fgfr, tsc2, predictor-mv, combined) at K = 2..6,
              10 bootstraps, 3 fits, tau 0.1, strata, stratum x pattern (n_min 5).

Each bundle's conditions are merged into one job array at <root>/jobs with
condition tags, so results land in <root>/<condition>/<dataset>/.
"""
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = "/home/rcstewart/exCALIBR"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/hpc")
M = Path("/data/ross/assay_calibration/multivariate")
COMMON = ["--constraints", "unc", "--init-strategy", "kmeans", "--num-fits", "3", "--pathogenic-share", "0.1",
          "--n-jobs", "4"]
STRATA = ["--strata"]


def prep(out, pipelines, genes, components, n_boot, extra, strata=True):
    cmd = ["nice", "-n", "19", sys.executable, f"{REPO}/hpc/prepare.py", "mv_all", "--output-dir", str(out),
           "--pipelines", *pipelines, "--components", *map(str, components), "--n-bootstraps", str(n_boot),
           *COMMON, *(STRATA if strata else []), *extra]
    if genes:
        cmd += ["--genes", *genes]
    log = Path(str(out) + ".log")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    if r.returncode:
        raise SystemExit(f"prepare failed for {out}: see {log}")
    print(f"prepared {out}")


def merge(root, labels, n_boot):
    """Collect each condition's merged jobs (<root>/<label>/jobs, tagged
    "tau_0.1") into one array at <root>/jobs, re-tagged "<label>/tau_0.1"."""
    import prepare as P
    all_jobs = []
    for lab in labels:
        jd = root / lab / "jobs"
        for a in sorted(jd.glob("array_*.pkl")):
            with open(a, "rb") as f:
                cj = pickle.load(f)
            for j in cj:
                j["condition"] = f"{lab}/{j.get('condition', 'tau_0.1')}"
            all_jobs.extend(cj)
            a.unlink()
    P._save_manifest(str(root), all_jobs, 50, n_boot, {"condition_order": [f"{l}/tau_0.1" for l in labels]})
    print(f"{root}: {len(all_jobs)} jobs")


BOOTSCHEME_SUBRUNS = {"tp53": (["tp53"], ["TP53"]), "integrated": (["integrated"], ["BRCA1"]),
                      "combined": (["combined"], ["BRCA2", "TSC2"])}


def bootscheme(root):
    schemes = {"S0": ["--bootstrap-scheme", "composite_pattern"],
               "S1": ["--bootstrap-scheme", "stratum"],
               "S2n5": ["--bootstrap-scheme", "stratum_pattern", "--bootstrap-min-cell", "5"],
               "S2n3": ["--bootstrap-scheme", "stratum_pattern", "--bootstrap-min-cell", "3"]}
    labels = []
    for s, extra in schemes.items():
        for sub, (pipes, genes) in BOOTSCHEME_SUBRUNS.items():
            prep(root / s / sub, pipes, genes, [4], 100, extra)
            labels.append(f"{s}/{sub}")
    merge(root, labels, 100)


def standard(root, label="standard", extra=()):
    labels = []
    for sub, (pipes, genes) in BOOTSCHEME_SUBRUNS.items():
        prep(root / label / sub, pipes, genes, [4], 100, list(extra), strata=False)
        labels.append(f"{label}/{sub}")
    merge(root, labels, 100)


def standard_n5(root):
    """Standard model, sample x pattern resampling with patterns of fewer than
    5 rows merged into the most similar pattern of the same sample (the S2n5
    rule applied to the 4 samples). Condition <root>/standard_n5/<subrun>."""
    standard(root, "standard_n5", ["--bootstrap-scheme", "stratum_pattern", "--bootstrap-min-cell", "5"])


ALL_SUBRUNS = {"integrated": ["integrated"], "tp53": ["tp53"], "fgfr": ["fgfr"], "tsc2": ["tsc2"],
               "predictor": ["predictor-mv"], "combined": ["combined"]}


def kdiag(root, strata=True, refit_strata=None):
    """K = 2..6, 10 bootstraps. Strata: stratum x pattern (n_min 5). Standard
    (strata=False): 4 samples, sample x pattern (the default scheme).
    refit_strata: {label: (pipelines, genes)} strata conditions added to the
    same bundle (e.g. genes whose earlier strata K-grid fits were wrong)."""
    s_extra = ["--bootstrap-scheme", "stratum_pattern", "--bootstrap-min-cell", "5"]
    labels = []
    for sub, pipes in ALL_SUBRUNS.items():
        prep(root / sub, pipes, None, [2, 3, 4, 5, 6], 10, s_extra if strata else [], strata=strata)
        labels.append(sub)
    for lab, (pipes, genes) in (refit_strata or {}).items():
        prep(root / lab, pipes, genes, [2, 3, 4, 5, 6], 10, s_extra, strata=True)
        labels.append(lab)
    merge(root, labels, 10)


# PALB2 and GCK: the 2026-11-19 strata K grid was prepared with chunked type
# inference (Chrom int/str mixed), leaving a variant's assays in separate rows.
KDIAG_STRATA_REFIT = {"integrated_strata_refit": (["integrated"], ["PALB2", "GCK"])}


def production(root, components_file, scheme_args, strata=True):
    """Phase 6: every MV gene set of the paper at the per-dataset component
    grids the user chose (components_file) and the bootstrap scheme chosen for
    the model (scheme_args, e.g. "--bootstrap-scheme stratum_pattern
    --bootstrap-min-cell 5", or "" for sample x pattern). strata=False builds
    the standard 4-sample model. 100 bootstraps, 3 fits, tau 0.1."""
    labels = []
    for sub, pipes in ALL_SUBRUNS.items():
        prep(root / sub, pipes, None, [3, 4], 100,
             ["--components-file", str(components_file), *scheme_args.split()], strata=strata)
        labels.append(sub)
    merge(root, labels, 100)


if __name__ == "__main__":
    which = sys.argv[1]
    root = Path(sys.argv[2])
    if which in ("production", "production_standard"):
        production(root, Path(sys.argv[3]), sys.argv[4], strata=which == "production")
    else:
        {"bootscheme": bootscheme, "kdiag": kdiag, "kdiag_standard": lambda r: kdiag(r, strata=False, refit_strata=KDIAG_STRATA_REFIT),
         "standard": standard, "standard_n5": standard_n5}[which](root)
