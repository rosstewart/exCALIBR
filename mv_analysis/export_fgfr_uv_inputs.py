#!/usr/bin/env python3
"""
One-off export: pool FGFR1-4 variants (safe -- different chromosomes, no genomic-ID
collision, same approach as multivariate_data/fgfr.py's combine_genes=True) from
/data/ross/assay_calibration/FGFR/dataframe_processed.csv.gz into one BasicScoreset-
compatible CSV per score dimension, for the user to run univariate ExCALIBR (UV)
calibration on directly.

For each dimension, writes both the raw-scale {dim}_scores.csv.gz (unchanged) and a
log-transformed {dim}_log_scores.csv.gz (natural log of the raw score column).

Note: Scoreset.to_csv() writes a 'score' (singular) column; BasicScoreset.from_csv()
expects 'scores' (plural). Left as 'score' here (matching Scoreset.to_csv()'s actual
output) -- rename before feeding to BasicScoreset.from_csv() if that's the consumer.
"""
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.multivariate_data.common import resolve_clinvar_release

SOURCE_PATH = "/data/ross/assay_calibration/FGFR/dataframe_processed.csv.gz"
OUTPUT_DIR = Path("/data/ross/assay_calibration/FGFR")
DIMENSIONS = {
    "activation": "score_activation",
    "pemr": "score_pemr",
    "futr": "score_futr",
}


def main():
    df = pd.read_csv(SOURCE_PATH, low_memory=False)
    clinvar_release = resolve_clinvar_release("FGFR1")

    for dim, score_col in DIMENSIONS.items():
        df_pooled = df.copy()
        df_pooled["Dataset"] = "FGFR_combined"
        ss = Scoreset(
            df_pooled, score_col=score_col,
            clinvar_release=clinvar_release, min_clinvar_star=1,
            population_type="gnomAD",
        )
        out_path = OUTPUT_DIR / f"{dim}_scores.csv.gz"
        ss.to_csv(out_path, compression="gzip")
        print(f"{dim}: {len(ss.dataframe)} variants -> {out_path}")

        ss_log = Scoreset(
            df_pooled, score_col=score_col,
            clinvar_release=clinvar_release, min_clinvar_star=1,
            population_type="gnomAD", log_transform=True,
        )
        log_out_path = OUTPUT_DIR / f"{dim}_log_scores.csv.gz"
        ss_log.to_csv(log_out_path, compression="gzip")
        print(f"{dim}: {len(ss_log.dataframe)} variants -> {log_out_path}")


if __name__ == "__main__":
    main()
