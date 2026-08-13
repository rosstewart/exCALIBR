"""
Serialize/load panel c's (ExCALIBR vs. author confusion) aggregate matrices
to/from a small JSON file.

Panel c (`analysis.figure4.panels.plot_panel_c`) only ever plots the *sum*
of the per-dataset confusion matrices across however many datasets have both
an ExCALIBR and an author call -- it never breaks them out per-dataset. So a
standalone reproduction of just Figure 4 doesn't need to rediscover and
rebuild every dataset in a multi-GB pipeline output tree (see
`analysis.figure4.driver._build_confusion_matrices`) just to recompute this
one panel: computing it once (`analysis.figure4.export_msh2_bundle`, which
does need the full pipeline output) and shipping this file instead lets
`build_figure4(panel_c_data=...)` skip that rebuild entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_CONFUSION_COLUMNS = ["Normal", "IR", "Abnormal"]
_CONFUSION_INDEX = ["BLB", "PLP"]


def save_panel_c_bundle(
    path: str,
    danz_agg: pd.DataFrame,
    auth_agg: pd.DataFrame,
    n_datasets: int,
    vus_pct_danz: Optional[float] = None,
    vus_pct_auth: Optional[float] = None,
) -> None:
    """Write the aggregate ExCALIBR/author confusion matrices (already summed
    across whatever datasets went into them) + pooled VUS-determinate
    percentages to `path` as JSON.

    `n_datasets` is provenance only (how many datasets the sums cover) --
    not used when loading back, just recorded so the bundle documents what
    it represents.
    """
    payload = {
        "danz_agg": danz_agg.loc[_CONFUSION_INDEX, _CONFUSION_COLUMNS].values.tolist(),
        "auth_agg": auth_agg.loc[_CONFUSION_INDEX, _CONFUSION_COLUMNS].values.tolist(),
        "n_datasets": n_datasets,
        "vus_pct_danz": vus_pct_danz,
        "vus_pct_auth": vus_pct_auth,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_panel_c_bundle(
    path: str,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str], Optional[float], Optional[float]]:
    """Load a file written by `save_panel_c_bundle` back into the
    `(danzs_oob, auths_oob, dataset_names, vus_pct_danz, vus_pct_auth)` shape
    `build_figure4` expects -- as single-element lists, since panel c only
    ever sums them anyway (see this module's docstring)."""
    with open(path) as f:
        payload = json.load(f)

    danz_agg = pd.DataFrame(payload["danz_agg"], index=_CONFUSION_INDEX, columns=_CONFUSION_COLUMNS)
    auth_agg = pd.DataFrame(payload["auth_agg"], index=_CONFUSION_INDEX, columns=_CONFUSION_COLUMNS)
    dataset_names = [f"aggregate ({payload.get('n_datasets', '?')} datasets)"]

    return [danz_agg], [auth_agg], dataset_names, payload.get("vus_pct_danz"), payload.get("vus_pct_auth")
