"""
"All dataset table" — one row per dataset with gene/citation/description/
assay metadata, used for the paper's dataset-summary table.

Ported from the live (uncommented) `process_single_dataset` /
`create_dataset_table_latex` / call-site block in test/plot_MSH2_ex.py
(~lines 777-988) — the large commented-out earlier draft around lines
431-690 was intentionally skipped, only this live version was ported.

Data sources, replacing hardcoded paths:
  - analysis.config.DATASET_TSV               (main variant dataframe: gives
    the Dataset list plus per-dataset 'Assay Type'/'Model_system' lookup —
    plays both roles the source notebook split across two separate
    dataframes, `df` and `df_std`, which are two different dumps of the same
    underlying integrated dataset)
  - analysis.config.DATASET_DESCRIPTIONS_CSV  (dataset_descriptions.csv)
  - analysis.config.DATASET_MEASUREMENTS_CSV  (dataset_measurements.csv) —
    also used to rebuild the gene->disease mapping that the notebook instead
    pickled to dataset_to_disease_assay_model.pkl; since that pickle is
    itself just a groupby of this same measurements CSV, it's rebuilt here
    rather than loaded from a pickle.
  - analysis.config.ASSAY_METHOD_MAP_CSV      (var_effect_measurements_dataset.csv)

Two additional inputs from the source notebook aren't in the four config
constants above and degrade gracefully (printed warning, not a crash) rather
than being re-derived:
  - datasets_to_exclude.pkl -> no config constant; if not found at the
    guessed path next to DATASET_TSV, no datasets are excluded.
  - the `datasets_new_configs` literal list (-> 'author_assignments_provided'
    column) -> replaced with the keys of analysis.config.DATASET_CONFIGS
    (same JSON dataset-configs file used elsewhere in this refactor, e.g.
    analysis.discovery/analysis.legacy_fits), since that's the set of
    datasets that have "new configs" un-pickled data; if the file is
    missing, the column is emitted as all-False.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from analysis import config as cfg


# ---------------------------------------------------------------------------
# Gene / author / year extraction (unchanged)
# ---------------------------------------------------------------------------

def extract_gene_from_dataset(dataset_name):
    """
    Extract gene name from dataset string.
    Handles special case for multi-gene datasets.
    """
    first_part = dataset_name.split('_')[0]

    # Special handling for CALM1_CALM2_CALM3
    if dataset_name.startswith('CALM1_CALM2_CALM3'):
        return 'CALM1/2/3'

    return first_part


def extract_author_from_dataset(dataset_name):
    """
    Extract author name from dataset string.
    Returns None if unpublished.
    Handles special case for multi-gene datasets.
    """
    parts = dataset_name.split('_')
    end = " et al."

    # Check for unpublished
    if 'unpublished' in parts:
        return None

    # Special handling for CALM1_CALM2_CALM3
    if dataset_name.startswith('CALM1_CALM2_CALM3'):
        # Author is at index 3 for this dataset
        return 'Weile'+end

    # Find the year (4 digits starting with 19 or 20)
    for i, part in enumerate(parts):
        if re.match(r'^(19|20)\d{2}$', part):
            # The part before the year is the author
            if i > 0:
                return parts[i-1]+end
            break

    return None


def extract_year_from_dataset(dataset_name):
    """
    Extract publication year from dataset string.
    Returns None if unpublished.

    Examples:
    - BRCA1_Findlay_2018 -> '2018'
    - BARD1_unpublished -> None
    - BRCA2_Sahu_2023_exon13_SGE -> '2023'
    """
    parts = dataset_name.split('_')

    # Check for unpublished
    if 'unpublished' in parts:
        return None

    # Find the year (4 digits starting with 19 or 20)
    for part in parts:
        if re.match(r'^(19|20)\d{2}$', part):
            return part

    return None


# ---------------------------------------------------------------------------
# Per-dataset row construction (unchanged logic; lookups passed in explicitly
# instead of read off module-level globals)
# ---------------------------------------------------------------------------

def process_single_dataset(dataset, dataset_to_desc, df_std_lookup, assay_method_lookup, sort_idx):
    """
    Process a single dataset and return its metadata.

    Parameters
    ----------
    dataset : str
        Dataset name
    dataset_to_desc : dict
        Mapping dataset -> (description, PMID)
    df_std_lookup : dict
        Mapping dataset -> {'assay_type':..., 'model_system':...}
    assay_method_lookup : dict
        Mapping dataset -> {'vamp_sge':..., 'IGVF_produced':...}
    sort_idx : int
        Original sort index for maintaining order

    Returns
    -------
    tuple : (sort_idx, dataset_data_dict or None)
    """

    if dataset == "F9_Popp_2025_model":
        return sort_idx, None

    try:
        desc, pmid = dataset_to_desc.get(dataset, ("", ""))

        gene = extract_gene_from_dataset(dataset)
        author = extract_author_from_dataset(dataset)
        year = extract_year_from_dataset(dataset)

        # Format author/year
        if author and year:
            citation = f"{author} ({year})"
        elif author:
            citation = author
        else:
            citation = "Unpublished"

        # Get assay_type and model_system from df_std
        std_info = df_std_lookup.get(dataset, {})
        assay_type = std_info.get('assay_type', None)
        model_system = std_info.get('model_system', None)

        # Get vamp_sge and IGVF_produced from assay_method_map
        method_info = assay_method_lookup.get(dataset, {})
        vamp_sge = method_info.get('vamp_sge', None)
        IGVF_produced = method_info.get('IGVF_produced', None)

        return sort_idx, {
            'dataset': dataset,
            'gene': gene,
            'citation': citation,
            'description': desc,
            'PMID': pmid,
            'assay_type': assay_type,
            'model_system': model_system,
            'vamp_sge': vamp_sge,
            'IGVF_produced': IGVF_produced
        }

    except Exception as e:
        print(f"Error processing {dataset}: {e}")
        return sort_idx, None


def create_dataset_table_latex(df, dataset_to_desc, df_std_lookup, assay_method_lookup):
    """
    Create a table of all datasets with metadata (name kept from the source
    notebook, though the LaTeX-generation code itself was already commented
    out there — this returns just the DataFrame, as the live version did).

    Parameters
    ----------
    df : DataFrame
        Full dataset DataFrame (source of the Dataset column)
    dataset_to_desc : dict
        Mapping of dataset names to (description, PMID)
    df_std_lookup : dict
        Mapping dataset -> {'assay_type':..., 'model_system':...}
    assay_method_lookup : dict
        Mapping dataset -> {'vamp_sge':..., 'IGVF_produced':...}

    Returns
    -------
    DataFrame : table data
    """

    # Get sorted list of datasets with indices
    datasets_sorted = sorted(df.Dataset.unique())

    print(f"Processing {len(datasets_sorted)} datasets...")

    # Process in parallel but preserve order
    results = [process_single_dataset(
            dataset, dataset_to_desc, df_std_lookup, assay_method_lookup, idx
        )
        for idx, dataset in enumerate(datasets_sorted)]

    # Sort by original index and filter out None
    results_sorted = sorted(results, key=lambda x: x[0])
    table_data = [data for _, data in results_sorted if data is not None]

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Print summary statistics
    print("\n" + "="*80)
    print("DATASET SUMMARY STATISTICS")
    print("="*80)
    print(f"Total datasets: {len(table_df)}")
    print(f"\nAssay types:")
    print(table_df['assay_type'].value_counts().to_string())
    print(f"\nModel systems:")
    print(table_df['model_system'].value_counts().to_string())
    print(f"\nVAMP-seq/SGE:")
    print(table_df['vamp_sge'].value_counts().to_string())
    print(f"\nIGVF produced:")
    print(table_df['IGVF_produced'].value_counts().to_string())
    print(f"\nUnpublished datasets: {(table_df['citation'] == 'Unpublished').sum()}")
    print("="*80)

    return table_df


# ---------------------------------------------------------------------------
# Optional/degraded inputs
# ---------------------------------------------------------------------------

def _load_datasets_to_exclude(dataset_tsv: str) -> list:
    """Best-effort load of datasets_to_exclude.pkl.

    No config constant names this file specifically (see module docstring);
    guess it lives alongside DATASET_TSV, and degrade to an empty exclusion
    list (with a warning) if it's not there.
    """
    import pickle

    guess = Path(dataset_tsv).parent / "datasets_to_exclude.pkl"
    if cfg.warn_if_missing(str(guess), "datasets_to_exclude.pkl (no dedicated config constant)"):
        return []
    with open(guess, "rb") as f:
        datasets_to_exclude = pickle.load(f)
    return list(datasets_to_exclude)


def _load_author_assignments_datasets(dataset_configs_path: Optional[str] = None) -> Optional[list]:
    """Return the list of datasets present in analysis.config.DATASET_CONFIGS.

    Replaces the notebook's hardcoded `datasets_new_configs` literal list
    (used only to build the 'author_assignments_provided' boolean column)
    with the keys of the JSON dataset-configs file already used elsewhere in
    this refactor (analysis.discovery / analysis.legacy_fits). Returns None
    (with a warning) if the JSON file isn't available, so the caller can
    degrade the column to all-False.
    """
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS
    if not dataset_configs_path or not Path(dataset_configs_path).exists():
        print(f"  SKIP author_assignments_provided source: DATASET_CONFIGS not found "
              f"({dataset_configs_path}); column will be all-False")
        return None
    with open(dataset_configs_path) as f:
        dataset_configs = json.load(f)
    return list(dataset_configs.keys())


def _build_gene_to_disease(measurements_csv: str) -> Dict[str, dict]:
    """Rebuild the dataset -> {'Disease':..., 'Assay Type':..., 'Model_system':...}
    mapping that the notebook instead pickled to
    dataset_to_disease_assay_model.pkl — it's just a groupby of this same
    DATASET_MEASUREMENTS_CSV, so it's rebuilt directly rather than loaded
    from a pickle.
    """
    df_measurements = pd.read_csv(measurements_csv)
    cols = ['Disease', 'Assay Type', 'Model_system']

    result = {}
    for ds, g in df_measurements.groupby('Dataset_tag'):
        row = {}
        skip = False
        for c in cols:
            vals = g[c].dropna().unique()
            if len(vals) == 0:
                skip = True
                break
            if len(vals) > 1:
                raise AssertionError(
                    f"Multiple values found where exactly one expected for "
                    f"Dataset_tag={ds}, column={c}: {vals.tolist()}"
                )
            row[c] = vals[0]
        if not skip:
            result[ds] = row

    if 'TSC2_rapgap_unpublished' in result:
        result['TSC2_combined_unpublished'] = result['TSC2_rapgap_unpublished']
        del result['TSC2_rapgap_unpublished']
    if 'TSC2_tuberin_unpublished' in result:
        del result['TSC2_tuberin_unpublished']

    return result


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def build_dataset_table(
    dataset_tsv: Optional[str] = None,
    descriptions_csv: Optional[str] = None,
    measurements_csv: Optional[str] = None,
    assay_method_map_csv: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Build the all-dataset summary table.

    Parameters mirror the notebook's hardcoded paths, defaulting to
    analysis.config constants. Returns the resulting DataFrame; only writes
    to disk if `output_csv` is explicitly given (the notebook's unconditional
    `.to_csv(...)` write is NOT reproduced as a module import/call side
    effect).
    """
    dataset_tsv = dataset_tsv or cfg.DATASET_TSV
    descriptions_csv = descriptions_csv or cfg.DATASET_DESCRIPTIONS_CSV
    measurements_csv = measurements_csv or cfg.DATASET_MEASUREMENTS_CSV
    assay_method_map_csv = assay_method_map_csv or cfg.ASSAY_METHOD_MAP_CSV

    print("Loading main dataframe...")
    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    df_std = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)

    print("Loading dataset descriptions...")
    df_description = pd.read_csv(descriptions_csv)
    dataset_to_desc = {}
    for _, row in df_description.iterrows():
        dataset, desc, pmid = row["Dataset_tag"], row["Experiment Short Description"], row["PMID"]
        dataset_to_desc[dataset] = (desc, pmid)

    print("Loading assay_method_map...")
    assay_method_map = pd.read_csv(assay_method_map_csv)

    # Model system mapping
    model_map = {
        'immortalized human cells': 'immortalized human cells',
        'murine primary cells': 'murine primary cells',
        'yeast': 'yeast',
        'other': 'other',
        'Not Applicable': 'not applicable'
    }

    # Assay type mapping
    assay_type_mapping = {
        'Reporter': 'reporter',
        'Cell fitness': 'cell fitness',
        'Cell Fitness': 'cell fitness',
        'Direct Protein Function': 'direct protein function',
        'Not Applicable (trained predictor)': 'not applicable (trained predictor)'
    }

    print("Building lookup dictionaries...")

    # Create lookup from df_std for assay_type and model_system
    df_std_lookup = (
        df_std
        .groupby("Dataset")
        .first()[["Assay Type", "Model_system"]]
        .to_dict('index')
    )
    for dataset, values in df_std_lookup.items():
        values['assay_type'] = assay_type_mapping.get(values['Assay Type'], values['Assay Type'])
        values['model_system'] = model_map.get(values['Model_system'], values['Model_system'])

    # Create lookup from assay_method_map
    assay_method_lookup = {}
    for _, row in assay_method_map.iterrows():
        dataset = row['Dataset']

        if row['Vamp'] == 'Yes':
            vamp_sge = 'VAMP-seq'
        elif row['SGE'] == 'Yes':
            vamp_sge = 'SGE'
        else:
            vamp_sge = 'other'

        assay_method_lookup[dataset] = {
            'vamp_sge': vamp_sge,
            'IGVF_produced': row['IGVF'] == 'Yes'
        }

    print(f"Loaded lookups for {len(df_std_lookup)} datasets in df_std")
    print(f"Loaded lookups for {len(assay_method_lookup)} datasets in assay_method_map")

    table_df = create_dataset_table_latex(df_std, dataset_to_desc, df_std_lookup, assay_method_lookup)

    cols_to_keep = ['dataset', 'gene', 'citation', 'description', 'PMID', 'assay_type',
                    'model_system', 'vamp_sge', 'IGVF_produced']
    table_df = table_df[cols_to_keep]

    datasets_to_exclude = _load_datasets_to_exclude(dataset_tsv)
    datasets_to_exclude = datasets_to_exclude + ['TSC2_combined_unpublished']
    table_df = table_df[~table_df.dataset.isin(datasets_to_exclude)]
    table_df = pd.concat([table_df, pd.DataFrame(
                                [['TSC2_combined_unpublished', 'TSC2', 'Unpublished', '', '', 'reporter', 'immortalized human cells', 'VAMP-seq', True]],
                                columns=table_df.columns)],
                    ignore_index=True
    )

    author_assignments_datasets = _load_author_assignments_datasets(dataset_configs_path)
    if author_assignments_datasets is not None:
        table_df["author_assignments_provided"] = table_df.dataset.isin(author_assignments_datasets)
    else:
        table_df["author_assignments_provided"] = False

    table_df = table_df.sort_values('dataset')

    gene_to_disease = _build_gene_to_disease(measurements_csv)
    table_df['disease'] = [
        gene_to_disease[d]['Disease'].lower() if d in gene_to_disease else None
        for d in table_df['dataset']
    ]

    if output_csv:
        table_df.to_csv(output_csv)
        print(f"  Saved dataset table to {output_csv}")

    return table_df


if __name__ == "__main__":
    build_dataset_table()
