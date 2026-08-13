"""
Serialize/load just the Scoreset attributes `analysis.figure4.panels` actually
reads (`.scores`, `.sample_assignments`, `.sample_counts`, `.snv_scores`), so
a minimal MSH2-only reproduction bundle doesn't need to ship the ~54K-row
master-TSV slice or rerun the pipeline's dataframe-to-Scoreset construction
(splice filtering, ClinVar-release parsing, sample assignment, ...) just to
get back these 4 small arrays for panels a/b/e.

Not applicable to panel c (confusion matrices, auth labels, is_vus, ...),
which needs much more of the Scoreset than this -- see
`analysis.figure4.panel_c_io` for that already-solved, separate problem.

Reuses `Scoreset.to_csv` (scores + sample_assignments, as a comma-joined
string of active sample-column indices per row) for writing, and
`BasicScoreset`'s own comma-string parsing (`validate_sample_assignments`,
which pads to >= 4 columns so column 0-3 always line up with the standard
P/LP-B/LB-gnomAD-Synonymous samples, matching what a real `Scoreset` does)
for reading -- *not* `BasicScoreset.from_csv`, which expects a "scores"
(plural) column and so doesn't actually match what `Scoreset.to_csv` writes
("score", singular); this module's own `load_scoreset_bundle` reads with the
column name `Scoreset.to_csv` actually uses instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assay_calibration.data_utils.dataset import BasicScoreset


def _snv_scores_path(scores_path) -> Path:
    """"name.csv.gz" -> "name_snv.csv.gz"; "name.csv" -> "name_snv.csv"."""
    scores_path = Path(scores_path)
    suffixes = "".join(scores_path.suffixes)
    stem = scores_path.name[: -len(suffixes)] if suffixes else scores_path.stem
    return scores_path.with_name(f"{stem}_snv{suffixes}")


def save_scoreset_bundle(scores_path, scoreset) -> None:
    """Write `scoreset` (anything with `.scores`/`.sample_assignments`/
    `.snv_scores`, e.g. a real Scoreset) to `scores_path` and a sibling
    "<name>_snv<ext>" file. Compression is inferred from a ".gz" suffix."""
    scores_path = Path(scores_path)
    compression = "gzip" if scores_path.suffix == ".gz" else None
    scoreset.to_csv(scores_path, compression=compression)
    pd.DataFrame({"score": np.asarray(scoreset.snv_scores)}).to_csv(
        _snv_scores_path(scores_path), index=False, compression=compression,
    )


def load_scoreset_bundle(scores_path) -> BasicScoreset:
    """Load a bundle written by `save_scoreset_bundle` back into a
    `BasicScoreset` (`.scores`/`.sample_assignments`/`.sample_counts`) with
    `.snv_scores` attached from the sibling file -- the 4 attributes
    `analysis.figure4.panels` actually reads off a Scoreset."""
    # dtype=str: forces the "sample_assignments" column to stay a string
    # column even when every row happens to hold a single-digit value (e.g.
    # every variant in exactly one sample) -- otherwise pandas would infer
    # int64, and BasicScoreset would treat each raw integer as its own
    # distinct "sample identifier" rather than a set of one-hot column
    # indices, silently reshuffling the standard P/LP-B/LB-gnomAD-Synonymous
    # column order.
    df = pd.read_csv(scores_path, dtype={"sample_assignments": str})
    scoreset = BasicScoreset(scores=df["score"].values, sample_assignments=df["sample_assignments"].values)
    scoreset.snv_scores = pd.read_csv(_snv_scores_path(scores_path))["score"].values
    return scoreset
