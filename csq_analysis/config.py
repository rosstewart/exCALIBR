"""Paths, dataset list and per-dataset settings for the consequence paper."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CSQ_ROOT", "/data/ross/assay_calibration/consequence_paper"))
PRELIM = ROOT / "prelim"
ANNOT = ROOT / "annotations"
CACHE = ROOT / "cache"
ASSAY_CONTEXT = ANNOT / "assay_context.csv"

DATA = Path("/data/ross/assay_calibration")
CELLS_LARGE = DATA / "missense_only_calibrations/cells/hpc_large_labeled/labeled"
REAL_OVERLAP = DATA / "consequence_cells_sim/real_overlap"
SD1 = DATA / "exc_pp_clinvar2025_calib/supplementary_data/SD1_dataset_summaries.csv"
DATASET_CONFIGS = REPO / "src/igvf_configs/dataset_configs_aug_2026.json"

# consequence classes (fit_utils/strata.py / multivariate_data/rows.py: start_lost and stop_lost count as PTV there;
# the paper keeps that definition so strata and ExCALIBR-MV agree)
PTV = ("stop_gained", "frameshift_variant", "start_lost", "stop_lost")
FINE = {"missense_variant": "missense", "synonymous_variant": "synonymous",
        "stop_gained": "PTV", "frameshift_variant": "PTV", "start_lost": "PTV", "stop_lost": "PTV",
        "splice_site_variant": "splice", "splicing_variant": "splice", "intron_variant": "intronic",
        "UTR_variant": "UTR", "inframe_indel": "inframe"}
CLASS_ORDER = ["missense", "PTV", "synonymous", "splice", "intronic", "UTR", "inframe", "other"]


def fine_class(simplified_consequence):
    """Paper consequence class of a simplified_consequence value (SpliceAI tag stripped)."""
    c = str(simplified_consequence).split("|")[0]
    return FINE.get(c, "other")


@lru_cache(maxsize=1)
def dataset_configs():
    return json.loads(DATASET_CONFIGS.read_text())


def dataset_k(ds):
    """K and benign_method from the production dataset config (same rule as membership_cells.metrics)."""
    cfg = dataset_configs()
    e = cfg.get(ds) or cfg.get(ds.replace("_clinvar_2018", "")) or {}
    kc = (e.get("n_c") if isinstance(e, dict) else e[0]) or "3c"
    bm = e.get("benign_method", "avg") if isinstance(e, dict) else "avg"
    return int(str(kc)[0]), bm


def canonical_datasets():
    """The 87 manuscript datasets: production calibrations minus DEFAULT_EXCLUDED_DATASETS."""
    from analysis import config as acfg
    from analysis.discovery import DEFAULT_EXCLUDED_DATASETS
    out = Path(acfg.OUTPUT_DIR)
    names = sorted(p.name for p in out.iterdir() if p.is_dir() and any(p.glob("*_calibration.json")))
    cfg = dataset_configs()
    keep = [d for d in names if d not in DEFAULT_EXCLUDED_DATASETS
            and (d in cfg or d.replace("_clinvar_2018", "") in cfg)]
    return keep
