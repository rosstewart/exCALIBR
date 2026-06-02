import pandas as pd
import numpy as np
from pathlib import Path
from fire import Fire
from functools import reduce
import logging
from io import StringIO
from tqdm import tqdm
import json
from joblib import Parallel, delayed  # noqa: F401

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)


class PillarProjectDataframe:
    def __init__(self, data_path: Path | str):
        self.data_path = Path(data_path)
        self.init_data()

    def init_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        self.dataframe = pd.read_csv(self.data_path)

    def __len__(self):
        return len(self.dataframe)

    def get_unique_clinsigs(self):
        sig_sets = self.dataframe.clinvar_sig.apply(
            lambda li: set(_clean_clinsigs(_tolist(li)))
        ).values
        return reduce(lambda x, y: x.union(y), sig_sets)


def _tolist(value, sep="^"):
    try:
        return value.split(sep)
    except AttributeError:
        if pd.isna(value):
            return [
                np.nan,
            ]
        return [
            value,
        ]


def _clean_clinsigs(values):
    return [v.split(";")[0] if isinstance(v, str) else "nan" for v in values]


func_class_map = {#'BRCA1_Findlay_2018': {'LOF':'Abnormal','FUNC':'Normal','INT':'Indeterminate'},
         'BAP1_Waters_2024': {'depleted':'Abnormal','unchanged':'Normal','enriched':'Not specified'},
         'BRCA2_Hu_2024': {'Abnormal': 'Abnormal', 'Normal': 'Normal', 'Intermediate': 'Indeterminate'},
         'CRX_Shepherdson_2024': {'low_activity': 'Abnormal','non-significant': 'Normal',
                                  'high_activity': 'Not specified'},
         'DDX3X_Radford_2023_cLFC_day15': {'fast depleting': 'Abnormal', 'slow depleting': 'Abnormal',
                                          'unchanged': 'Normal', 'enriched': 'Not specified'},
         'FKRP_Ma_2024': {'damaging_severe': 'Abnormal', 'damaging_mild': 'Abnormal', 
                          'damaging_intermediate': 'Abnormal', 'functional': 'Normal'},
         'JAG1_Gilbert_2024': {'Abnormal': 'Abnormal', 'Likely abnormal': 'Abnormal','Normal': 'Normal'},
         'KCNE1_Muhammad_2024_presence_of_WT': {'Loss': 'Abnormal','Possible':'Abnormal','Partial':'Abnormal',
                                               'Normal':'Normal','Gain':'Not specified','PossibleGain':'Not specified'},
         'KCNE1_Muhammad_2024_absence_of_WT': {'Loss': 'Abnormal','Possible':'Abnormal','Partial':'Abnormal',
                                               'Normal':'Normal','Gain':'Not specified','PossibleGain':'Not specified'},
         'KCNE1_Muhammad_2024_potassium_flux': {'Loss': 'Abnormal','Possible':'Abnormal','Partial':'Abnormal',
                                               'Normal':'Normal','Gain':'Not specified','PossibleGain':'Not specified'},
         'LARGE1_Ma_2024': {'damaging':'Abnormal','functional': 'Normal'},
         'NDUFAF6_Sung_2024': {'abnormal': 'Abnormal','normal':'Normal','uncertain': 'Indeterminate'},
         'OTC_Lo_2023': {'Amorphic':'Abnormal','Unimpaired':'Normal', 'Hypomorphic':'Not specified'},
         'RAD51C_Olvera-León_2024_z_score_D4_D14': {'fast depleted': 'Abnormal','slow depleted': 'Abnormal',
                                                   'unchanged': 'Normal','enriched':'Not specified'},
         'RHO_Wan_2019':{'low': 'Abnormal','very low': 'Abnormal','high': 'Normal','indeterminate': 'Indeterminate'},
         'SCN5A_Glazer_2020': {'LOF':'Abnormal','possiblyLOF':'Abnormal','possiblyWT':'Normal','WT':'Normal',
                              'GOF': 'Not specified','possiblyGOF':'Not specified'},
         'SCN5A_Ma_2024_current_density':{'severe LOF': 'Abnormal', 'moderate LOF': 'Abnormal','normal': 'Normal'},
         'SGCB_Li_2023': {'Non-Functional': 'Abnormal','Functional': 'Normal'},
         'TP53_Fayer_2021_meta': {'Functionally abnormal': 'Abnormal','Functionally normal': 'Normal'},
         'VHL_Buckley_2024': {'LOF1': 'Abnormal','LOF2': 'Abnormal','Neutral': 'Normal', 'Intermediate': 'Indeterminate'},
         'BARD1_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal', 'indeterminate': 'Indeterminate'},
         'PALB2_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal', 'indeterminate': 'Indeterminate'},
         'CTCF_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal', 'indeterminate': 'Indeterminate'},
         'RAD51D_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal', 'indeterminate': 'Indeterminate'},
         'SFPQ_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal', 'indeterminate': 'Indeterminate'},
         'PTEN_Matreyek_2018': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal'},
         'F9_Popp_2025_model': {'WT-like': 'Normal','Loss of function': 'Abnormal'},
         'G6PD_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal'}, 
         'TSC2_rapgap_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal'},
         'TSC2_tuberin_unpublished': {'functionally_abnormal': 'Abnormal', 'functionally_normal': 'Normal'},
         'CARD11_Meitlis_2020_DMSO_no_introns': {'functional': 'Normal', 'not definitive': 'Indeterminate', 
                                                 'likely functional': 'Normal','likely nonfunctional': 'Abnormal', 
                                                 'nonfunctional': 'Abnormal'}}

INTERVAL_CLASS_NORMALIZATION = {
    "normal": "Normal",
    "abnormal": "Abnormal",
    "not specified": "Indeterminate",
}

def classify_by_intervals(score, intervals):
    """
    intervals: list of dicts with keys:
        - 'range': tuple (start, end, start_inclusive, end_inclusive)
        - 'class': raw MaveDB class string
    """
    for interval in intervals:
        lo, hi, lo_inc, hi_inc = interval["range"]

        in_range = (
            (score > lo or (lo_inc and score == lo)) and
            (score < hi or (hi_inc and score == hi))
        )

        if in_range:
            raw = interval["class"]
            if raw is None or str(raw).lower() == "nan":
                return "Indeterminate"

            return INTERVAL_CLASS_NORMALIZATION.get(
                raw.strip().lower(),
                "Indeterminate"
            )

    return "Indeterminate"

import math
import re

def parse_interval_range(range_str):
    """
    Parses strings like:
      '[-0.748, Inf)'
      '(-Inf,-1.328]'
    Returns: (lo, hi, lo_inc, hi_inc)
    """
    if not isinstance(range_str, str):
        return None

    m = re.match(r'([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])', range_str)
    if not m:
        return None

    lo_br, lo, hi, hi_br = m.groups()

    def to_float(x):
        x = x.strip()
        if x.lower() in {"-inf", "-infinity"}:
            return -math.inf
        if x.lower() in {"inf", "infinity"}:
            return math.inf
        return float(x)

    return (
        to_float(lo),
        to_float(hi),
        lo_br == "[",
        hi_br == "]",
    )

def extract_intervals_from_row(row, max_intervals=6):
    intervals = []

    for i in range(1, max_intervals + 1):
        range_key = f"Interval {i} range"
        class_key = f"Interval {i} MaveDB class"

        range_str = row.get(range_key)
        class_val = row.get(class_key)

        parsed = parse_interval_range(range_str)
        if parsed is None:
            continue

        intervals.append({
            "range": parsed,
            "class": class_val
        })

    return intervals



def standardize_auth_label(
    dataset_name,
    func_class,
    score=None,
    intervals=None,
):
    """
    Returns standardized label: Normal / Abnormal / Indeterminate

    Priority:
      1) Dataset-specific func_class_map
      2) Interval-based classification using score
      3) Indeterminate
    """

    if dataset_name in func_class_map:
        dataset_map = func_class_map[dataset_name]
        if func_class in dataset_map:
            return dataset_map[func_class]

    if isinstance(func_class, str):
        norm = func_class.strip().lower()
        if norm in INTERVAL_CLASS_NORMALIZATION:
            return INTERVAL_CLASS_NORMALIZATION[norm]

    if score is not None and intervals is not None:
        return classify_by_intervals(score, intervals)

    return "Indeterminate"



class BasicScoreset:
    def __init__(self, scores, sample_assignments, ids=None, **kwargs):
        self.scores = np.asarray(scores)
        self._sample_assignments = np.asarray(sample_assignments)
        if ids is not None:
            ids = np.asarray(list(ids))
            if len(ids) != len(self.scores):
                raise ValueError(
                    f"ids length {len(ids)} does not match scores length "
                    f"{len(self.scores)}"
                )
        self._ids = ids
        self.validate_inputs()
        self.validate_sample_assignments()
        self.sample_counts = self._sample_assignments.sum(axis=0)
        self.scoreset_name = kwargs.get("scoreset_name", "BasicScoreset")
        _default_names_4 = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous"]
        _default_names_3 = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD"]
        if "sample_names" in kwargs:
            self.sample_names = kwargs["sample_names"]
        elif self.n_samples == 3:
            self.sample_names = _default_names_3
        else:
            self.sample_names = _default_names_4[:self.n_samples]

    def validate_inputs(self):
        n_observations = self.scores.shape[0]
        if self._sample_assignments.shape[0] != n_observations:
            raise ValueError(
                f"Number of observations in scores {n_observations:,d} does not match number of rows in sample_assignments {self._sample_assignments.shape[0]:,d}"
            )

    def validate_sample_assignments(self):
        ndim = self._sample_assignments.ndim
        if ndim == 1:
            sample_vals = np.array(self._sample_assignments)
            if sample_vals.dtype.kind in ('U', 'O'):
                # print(
                #     "Assuming sample_assignments is list of comma-separated strings"
                # )
                # Always split on comma — single values like "2" split fine into ["2"]
                all_keys = sorted({k for v in sample_vals for k in str(v).split(',')})
                key_to_idx = {k: i for i, k in enumerate(all_keys)}
                result = np.zeros((len(sample_vals), len(all_keys)), dtype=bool)
                for row, v in enumerate(sample_vals):
                    for k in str(v).split(','):
                        result[row, key_to_idx[k]] = True
                self._sample_assignments = result
            else:
                # print(
                #     "Assuming sample_assignments is a list of sample identifiers, converting to 2D array."
                # )
                unique_samples = sorted(set(sample_vals))
                self._sample_assignments = np.zeros(
                    (len(sample_vals), len(unique_samples)), dtype=bool
                )
                for sampleNum, sample_id in enumerate(unique_samples):
                    self._sample_assignments[:, sampleNum] = sample_vals == sample_id
        elif ndim != 2:
            raise ValueError(
                f"sample_assignments must be a 1D list of sample ids or 2D array of one-hot vectors, got {ndim} dimensions"
            )

        # Warn if any sample has no exclusive variants — makeOneHot coverage will be
        # infeasible for that sample, causing an error during bootstrap fitting.
        sa = np.array(self._sample_assignments)
        if sa.ndim == 2 and sa.shape[1] > 1:
            shared_mask = sa.sum(axis=1) > 1
            for s in range(sa.shape[1]):
                n_exclusive = (sa[:, s] & ~shared_mask).sum()
                if sa[:, s].any() and n_exclusive == 0:
                    import warnings
                    warnings.warn(
                        f"Sample index {s} has no exclusive variants "
                        f"(all {int(sa[:, s].sum())} variant(s) are shared with other samples). "
                        f"Bootstrap one-hot assignment may be infeasible for this sample.",
                        stacklevel=3,
                    )

    @property
    def sample_assignments(self):
        return self._sample_assignments[:, self.sample_counts > 0]

    @property
    def n_samples(self):
        return self._sample_assignments.shape[1]

    @property
    def ids(self):
        return self._ids

    @property
    def samples(self):
        for sample_index in range(self.n_samples):
            if self.sample_counts[sample_index] > 0:
                yield self.scores[
                    self._sample_assignments[:, sample_index]
                ], self.sample_names[sample_index]


    # @property
    # def samples(self):
    #     """
    #     Iterate over the samples in the scoreset, yielding the sample scores and sample name.
    #     """
    #     for sample_index in range(self.sample_assignments.shape[1]):
    #         sample_scores = self.scores[self.sample_assignments[:, sample_index]]
    #         if len(sample_scores) > 0:
    #             yield sample_scores, f"Sample {sample_index + 1}"

    @classmethod
    def from_csv(cls, csv_path: Path | str, **kwargs):
        """
        Create a BasicScoreset from a CSV file.

        Required columns:
         - scores: The scores for each observation
         - sample_assignments: The ID of the sample to which each observation belongs
            Example:
            scores,sample_assignments
            0.5,1
            0.7,2
            0.3,1

        Parameters
        ----------
        csv_path : Path|str
            The path to the CSV file to create the scoreset from
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "scores" not in df.columns or "sample_assignments" not in df.columns:
            raise ValueError(
                "CSV must contain 'scores' and 'sample_assignments' columns"
            )
        scores = np.array(df["scores"].values)
        sample_assignments = np.array(df["sample_assignments"].values)
        return cls(scores, sample_assignments, **kwargs)


class Scoreset:
    def __init__(self, dataframe: pd.DataFrame, **kwargs):
        """
        Required Arg:
        - dataframe : pd.DataFrame : Pillar Project dataframe

        Optional Arg:
        - min_clinvar_star : int (default 1) : minimum review status (star count) to use Clinical Significance annotations
        - clinvar_release : str in {'2026','2018'} (default '2026') : Clinvar release to use
        """
        self._init_kwargs = dict(kwargs)
        self._init_dataframe(dataframe, **kwargs)

    def to_json(self, output_path: Path | str):
        """
        Save the scoreset to a JSON file.

        Parameters
        ----------
        output_path : Path|str
            The path to save the JSON file to

        Returns
        -------
        None
        """
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_json(output_path, orient="records", lines=True)

    def to_csv(self, output_path: Path | str):
        """
        Save the scoreset to a simple output CSV.

        Parameters
        ----------
        output_path : Path|str
            The path to save the CSV file to

        Returns
        -------
        None
        """
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        rows, cols = np.where(self.sample_assignments)
        sample_assignments_str = [
            ",".join(map(str, cols[rows == i])) 
            for i in range(self.sample_assignments.shape[0])
        ]

        output_df = pd.DataFrame({"score": self.scores, "sample": sample_assignments_str})
        
        output_df.to_csv(output_path, index=False)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, **kwargs):
        """
        Create a Scoreset from a pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to create the scoreset from

        Returns
        -------
        Scoreset
            A Scoreset object initialized with the given dataframe
        """
        return cls(dataframe, **kwargs)

    @classmethod
    def from_json(cls, json_path: Path | str, **kwargs):
        """
        Create a Scoreset from a JSON or JSONL file.
    
        Automatically detects:
          - Standard JSON object or list
          - Newline-delimited JSON (JSONL)
    
        Parameters
        ----------
        json_path : Path|str
            Path to JSON/JSONL input file
    
        Returns
        -------
        Scoreset
            A Scoreset initialized with parsed data
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
    
        with open(json_path, "r") as f:
            text = f.read().strip()
    
        # Try standard JSON first
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Only fallback if this specific error indicates multiple objects
            if "Extra data" not in str(e):
                raise  # If it's a different JSON error, report it normally
    
            # Fallback for JSONL: parse line-by-line
            try:
                lines = [json.loads(line) for line in text.splitlines() if line.strip()]
                data = lines  # list of JSON objects
            except Exception:
                raise ValueError(
                    f"File is neither valid JSON nor JSONL: {json_path}\n"
                    f"Original error: {e}"
                )
    
        # Convert result into DataFrame
        dataframe = pd.DataFrame.from_records(data)
        return cls(dataframe, **kwargs)


    def _init_dataframe(self, dataframe: pd.DataFrame, **kwargs):
        """
        Initialize the scoreset from the dataframe

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to initialize the scoreset from

        Optional Arg:
            - min_clinvar_star : int (default 1) : minimum review status (star count) to use Clinical Significance annotations
            - clinvar_release : str in {'2026','2018'} (default '2026') : Clinvar release to use
        Returns
        -------
        None
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if len(dataframe.Dataset.unique()) != 1:
            raise ValueError("dataframe must contain only one dataset")
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row")
                
        self.score_col = kwargs.get("score_col", "auth_reported_score")
        
        score_col = self.score_col
        
        # ensure numeric conversion
        dataframe = dataframe.assign(
            **{
                score_col: pd.to_numeric(dataframe[score_col], errors="coerce")
            }
        )
        
        # drop NaNs in chosen score column
        dataframe = dataframe.dropna(subset=[score_col])
        
        # (this does nothing by default)
        dataframe = Scoreset.remove_outliers(dataframe, **kwargs)
        
        if not len(dataframe):
            raise ValueError(
                f"dataframe must contain at least one row with a non-NaN {score_col}"
            )
        self.dataframe = dataframe
        self.filter_nonsense = kwargs.get("filter_nonsense",False)
        self.filter_invalid()
        self.splicing_filter(**kwargs)
        min_clinvar_star = kwargs.get("min_clinvar_star",1)
        clinvar_release = kwargs.get("clinvar_release",'2026')
        n_jobs = kwargs.get("n_jobs", 1)
        rows = self.dataframe.to_dict('records')
        if n_jobs == 1:
            self.variants = [
                Variant(row, min_clinvar_star, clinvar_release, score_col)
                for row in rows
            ]
        else:
            self.variants = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(Variant)(row, min_clinvar_star, clinvar_release, score_col)
                for row in rows
            )
        self.synonymous_exclusive = kwargs.get("synonymous_exclusive",False)
        self.score_avg = kwargs.get("score_avg",True)

        # for discordance and downsampling analyses
        self.discordance_pct = kwargs.get("discordance_pct",0.0)
        self.downsample_n_variants = kwargs.get("downsample_n_variants",None)
        self.downsample_seed = kwargs.get("downsample_seed",None)
        
        self._init_matrices(**kwargs)

    def filter_invalid(self):
        self.dataframe = self.dataframe[self.dataframe.Flag != "*"]

        if self.filter_nonsense:
            # print('filtering nonsense variants within 50 aa of termini...')
            mask = (
                # Keep all non-stop_gained variants
                (self.dataframe['simplified_consequence'] != 'stop_gained') |
                # OR keep stop_gained only if position is between 51-349
                ((self.dataframe['simplified_consequence'] == 'stop_gained') &
                 (self.dataframe['aa_pos'] >= 51) &
                 (self.dataframe['aa_pos'] <= 349))
            )
            self.dataframe = self.dataframe[mask]

    def splicing_filter(self, **kwargs):
        self.detects_splice = (
            self.dataframe.loc[:, "splice_measure"].unique()[0] == "Yes"  # type: ignore
        )
        # if assay does not detect effects of splicing, remove likely splicing aberrations
        if not self.detects_splice:
            # Remove VEP (mapped/unmapped) consequences
            self.dataframe = self.dataframe[
                ~self.dataframe.simplified_consequence.str.lower().isin([
                    "splice region",
                    "splice_site_variant",
                    "splice_acceptor_variant",
                    "splice_donor_variant",
                ])
            ]
            
            # Remove SpliceAI scores
            spliceai_cols = ["spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL"]
            mask = self.dataframe[spliceai_cols].lt(0.2) | self.dataframe[spliceai_cols].isna()
            self.dataframe = self.dataframe[mask.all(axis=1)]

    @staticmethod
    def remove_outliers(dataframe, **kwargs):
        """
        Optionally clip the dataframe to remove observations outside a specified percentile range

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to remove outliers from

        Optional Parameters
        -------------------
        - quantile_min : float (default: 0.0)
        - quantile_max : float (default: 1.0)

        Returns
        -------
        pd.DataFrame
            The dataframe with outliers removed (1.5 IQR Rule)
        """
        score_col = kwargs.get("score_col", "auth_reported_score")
        quantile_min = kwargs.get("quantile_min", 0.0)
        quantile_max = kwargs.get("quantile_max", 1.0)
        lowerbound = dataframe[score_col].quantile(quantile_min)
        upperbound = dataframe[score_col].quantile(quantile_max)
        scores = dataframe[score_col]
        include = (scores >= lowerbound) & (scores <= upperbound)
        return dataframe[include]

    def __len__(self):
        return len(self.variants)

    def _init_matrices(self, **kwargs):
        """
        Optional Arguments:
        - population_type : str : one of {'all_variants',
                                  'all_nsSNV',
                                  'all_missense_nsSNV',
                                  'gnomAD',
                                  'gnomAD_nsSNV',
                                  'gnomAD_missense_nsSNV'}
        
        """
        self.has_synomyous = any([variant.is_synonymous for variant in self.variants])
        self.NSamples = 4
        self.sample_names = [
            "Pathogenic/Likely Pathogenic",
            "Benign/Likely Benign",
            "population",
            "Synonymous",
        ]
        variants_by_id = self.get_variants_by_id()
        self.n_variants = len(variants_by_id)
        self._sample_assignments = np.zeros(
            (self.n_variants, self.NSamples), dtype=bool
        )
        self._scores = np.zeros(self.n_variants)
        self._auth_labels = np.empty(self.n_variants, dtype='U50')
        self._snv_scores = []
        self._vus_scores = []
        self._all_scores = []
        self._aa_subs = np.empty(self.n_variants, dtype='U50')
        self._variant_codes = []
        self._ids = []
        self.parse_population_type(**kwargs)
        group_cols = ['Gene','Chrom','hg38_start','ref_allele','alt_allele']
        is_pop = self.is_population_member  # avoid repeated self-lookup in tight loop
        for idx, (_id, variants) in enumerate(variants_by_id.items()):

            self._ids.append(_id)

            v0 = variants[0]
            if self.score_avg:
                v0.assay_score = np.mean([v.assay_score for v in variants])

            score0 = v0.assay_score
            self._scores[idx] = score0
            self._auth_labels[idx] = v0.auth_label
            if v0.is_snv:
                self._snv_scores.append(score0)
            self._all_scores.append(score0)

            if any(v.is_vus for v in variants):
                self._vus_scores.append(score0)

            try:
                self._aa_subs[idx] = v0.aa_ref + str(int(v0.aa_pos)) + v0.aa_alt
            except (ValueError, TypeError):
                self._aa_subs[idx] = None

            # Store ALL unique nucleotide-level group_col tuples for this
            # variant group (not just v0).  Protein-level assays
            # group many nucleotide substitutions under a single protein-level
            # ID; saving all of them allows MultiScoreset to match this entry
            # against any of the underlying nucleotide variants in other assays.
            seen_codes: set = set()
            codes = []
            for v in variants:
                try:
                    code = tuple(getattr(v, g) for g in group_cols)
                    if code not in seen_codes:
                        seen_codes.add(code)
                        codes.append(code)
                except (ValueError, TypeError):
                    pass
            self._variant_codes.append(codes if codes else None)

            if any(v.is_synonymous for v in variants):
                self._sample_assignments[idx, 3] = True
                if self.synonymous_exclusive:
                    continue
            if any(is_pop(v) for v in variants):
                self._sample_assignments[idx, 2] = True
            if any(v.is_pathogenic for v in variants):
                self._sample_assignments[idx, 0] = True
            if any(v.is_benign for v in variants):
                self._sample_assignments[idx, 1] = True
        
        keep_mask = self._sample_assignments.any(axis=1)
        self._keep_mask = keep_mask
        self._scores = self.scores[keep_mask]
        self._sample_assignments = self._sample_assignments[keep_mask]
        self._auth_labels = self._auth_labels[keep_mask]
        self._aa_subs = self._aa_subs[keep_mask]

        
        if self.discordance_pct != 0:
            self._sample_assignments = self.add_label_noise(self._sample_assignments, self.discordance_pct)
        
        if self.downsample_n_variants is not None:
            keep_mask_downsample = self.downsample_controls(self._sample_assignments, self.downsample_n_variants, self.downsample_seed)
            self._keep_mask_downsample = keep_mask_downsample
            self._scores = self._scores[keep_mask_downsample]
            self._sample_assignments = self._sample_assignments[keep_mask_downsample]
            self._auth_labels = self._auth_labels[keep_mask_downsample]
            self._aa_subs = self._aa_subs[keep_mask_downsample]

        self.n_variants = len(self._scores)
        self.sample_counts = self._sample_assignments.sum(axis=0)
        self._snv_scores = np.array(self._snv_scores)
        self._vus_scores = np.array(self._vus_scores)
        self._all_scores = np.array(self._all_scores)

    

    def add_label_noise(self, sample_assignments, discordance_pct):
        """
        Introduces label noise by overwriting a discordance_pct fraction of pathogenic
        (col 0) rows with randomly sampled benign rows, and vice versa. Each class is
        treated independently, so 1% discordance means exactly 1% of P and 1% of B
        are affected. Sampling is done from the original assignments to avoid
        order-dependency. Seed is deterministic based on discordance_pct.
        """
        assert 0 <= discordance_pct <= 0.5, (
            f"discordance_pct must be between 0 and 0.5 inclusive, got {discordance_pct}"
        )
        rng = np.random.default_rng(seed=int(discordance_pct * 1e6))
        assignments = sample_assignments.copy()
    
        path_idx = np.where(assignments[:, 0])[0]
        ben_idx  = np.where(assignments[:, 1])[0]
    
        n_path_flip = int(discordance_pct * len(path_idx))
        n_ben_flip  = int(discordance_pct * len(ben_idx))
    
        # Overwrite P->B: selected pathogenic rows get a randomly sampled benign row
        flip_path = rng.choice(path_idx, size=n_path_flip, replace=False)
        src_ben   = rng.choice(ben_idx,  size=n_path_flip, replace=True)
        assignments[flip_path] = sample_assignments[src_ben]
    
        # Overwrite B->P: selected benign rows get a randomly sampled pathogenic row
        flip_ben  = rng.choice(ben_idx,  size=n_ben_flip,  replace=False)
        src_path  = rng.choice(path_idx, size=n_ben_flip,  replace=True)
        assignments[flip_ben] = sample_assignments[src_path]
    
        return assignments


    def downsample_controls(self, sample_assignments, downsample_n_variants, downsample_seed):
        """
        Downsample pathogenic (col 0) and benign (col 1) variants each to at most
        downsample_n_variants, keeping all non-control rows intact.
        Returns a boolean keep mask over the input arrays.
        Seed is deterministic based on downsample_seed.
        """
        assert downsample_seed is not None, f"downsample_seed must be set to enforce multiple reproducible runs"
        
        rng = np.random.default_rng(seed=int(downsample_seed))
        keep = np.ones(len(sample_assignments), dtype=bool)
    
        for col in (0, 1):          # 0 = pathogenic, 1 = benign
            idx = np.where(sample_assignments[:, col])[0]
            if len(idx) > downsample_n_variants:
                drop = rng.choice(idx, size=len(idx) - downsample_n_variants, replace=False)
                keep[drop] = False
    
        return keep
    
    

    def parse_population_type(self,**kwargs):
        population_type = kwargs.get("population_type",'gnomAD')
        valid_population_types = {'all_variants',
                                  'all_nsSNV',
                                  'all_missense_nsSNV',
                                  'gnomAD',
                                  'gnomAD_nsSNV',
                                  'gnomAD_missense_nsSNV'}
        if population_type not in valid_population_types:
            raise ValueError(f"Invalid population type {population_type}; must be in {valid_population_types}")
        self.population_type = population_type

    def is_population_member(self,variant)->bool:
        if self.population_type == "all_variants":
            return True
        if self.population_type == "all_nsSNV":
            return variant.is_nsSNV
        if self.population_type == "all_missense_nsSNV":
            return variant.simplified_consequence == "missense_variant" and variant.is_nsSNV
        if self.population_type == "gnomAD":
            return variant.is_gnomAD
        if self.population_type == "gnomAD_nsSNV":
            return variant.is_gnomAD and variant.is_nsSNV
        if self.population_type == "gnomAD_missense_nsSNV":
            return variant.is_gnomAD and \
                variant.is_nsSNV and \
                variant.simplified_consequence == "missense_variant"
        return False
        

    def get_variants_by_id(self):
        """
        Iterate over all unique Variant.ID values, returning the variants with that given ID.

        Returns
        -------
        dict
            A dictionary where keys are unique Variant.ID values and values are lists of Variant objects with that ID
        """
        variants_by_id = {}
        for variant in self.variants:
            key = getattr(variant, "mavedb_variant_urn", None) or variant.ID
            variants_by_id.setdefault(key, []).append(variant)
        return variants_by_id

    @property
    def sample_assignments(self):
        return self._sample_assignments[:, self.sample_counts > 0]

    @property
    def n_samples(self):
        return self.sample_assignments.shape[1]

    @property
    def samples(self):
        for sample_index in range(self.NSamples):
            if self.sample_counts[sample_index] > 0:
                yield self.scores[
                    self._sample_assignments[:, sample_index]
                ], self.sample_names[sample_index]

    @property
    def scores(self):
        return self._scores

    @property
    def snv_scores(self):
        return self._snv_scores
        
    @property
    def vus_scores(self):
        return self._vus_scores
    
    @property
    def all_scores(self):
        return self._all_scores
        
    @property
    def aa_subs(self):
        return self._aa_subs
        
    @property
    def variant_codes(self):
        return self._variant_codes

    @property
    def auth_labels(self):
        return self._auth_labels

    @auth_labels.setter
    def auth_labels(self, value):
        self._auth_labels = value

    @property
    def scoreset_name(self):
        return self.dataframe.Dataset.values[0]

    @property
    def keep_mask(self):
        return self._keep_mask

    def __repr__(self):
        out = f"{self.scoreset_name}: {len(self.scores)} total variants\n"
        for sample_scores, sample_name in self.samples:
            out += f"\t{sample_name}: {len(sample_scores)} variants\n"

        return out

    def to_serializable(self):
        return {
            "__type__": "Scoreset",
            "dataframe": self.dataframe.to_dict(orient="split"),
            "kwargs": self._init_kwargs,
        }

    
    @classmethod
    def from_serializable(cls, payload: dict):
        df = pd.DataFrame(**payload["dataframe"])
        return cls(df, **payload["kwargs"])



class Variant:
    def __init__(self, variant_info, min_clinvar_star: int, clinvar_release: str, score_col: str):
        self.min_clinvar_star = min_clinvar_star
        self.clinvar_release = clinvar_release
        self.score_col = score_col
        # Accept both pd.Series and plain dict (dict is faster from to_dict('records'))
        self.row = variant_info if isinstance(variant_info, dict) else dict(variant_info)
        self._init_variant_info(self.row, score_col)

    def _init_variant_info(self, variant_info, score_col: str):
        self.ID = None
        self.simplified_consequence = None
        self.clinvar_star = None
        self.gnomad_MAF = None
        self.assay_score = variant_info[score_col]
        self.transcript_ref = ""
        self.transcript_alt = ""
        self.aa_ref = np.nan
        self.aa_alt = ""
        self.hgvs_p = ""
        self.__dict__.update({str(k): v for k, v in variant_info.items()})
        self.clinvar_sig = variant_info[f"clinvar_sig_{self.clinvar_release}"]
        self.clinvar_star = variant_info[f"clinvar_star_{self.clinvar_release}"]
        self.parse_gnomAD_MAF()
        self.parse_clinvar_sig()
        self.parse_consequences()
        self.assign_nsSNV()
        self.parse_auth_class()

    def assign_nsSNV(self):
        # is the variant a nonsynonymous nsSNV
        self.is_nsSNV = (isinstance(self.transcript_ref,str) and len(self.transcript_ref) == 1) and \
                        (isinstance(self.transcript_alt,str) and len(self.transcript_alt) == 1) and \
                        (isinstance(self.hgvs_p,float) | \
                         (self.aa_ref != self.aa_alt))

    def parse_consequences(self):
        self.is_synonymous = (self.simplified_consequence == "Synonymous") or (
            self.simplified_consequence == "synonymous_variant"
        )
        self.is_missense = self.simplified_consequence == 'missense_variant'
        self.is_snv = len(str(self.ref_allele)) == 1 and len(str(self.alt_allele)) == 1 and str(self.ref_allele) != str(self.alt_allele)

    def parse_clinvar_sig(self):
        self.is_conflicting = (
            self.clinvar_sig == "Conflicting classifications of pathogenicity"
        )
        self.eval_quality()
        
        self.is_benign = self.sufficient_quality and self.clinvar_sig in {
            "Benign",
            "Likely benign",
            "Benign/Likely benign",
        }
        self.is_pathogenic = self.sufficient_quality and self.clinvar_sig in {
            "Pathogenic",
            "Likely pathogenic",
            "Pathogenic/Likely pathogenic",
        }
        self.is_vus = self.sufficient_quality_2026 and self.clinvar_sig_2026 in { # VUS ALWAYS 2026 SINCE NOT CONTROL
            "Uncertain significance",
        }

    def standardize_class(self, c):
        # c: Class
        if c is None:
            return c
        elif pd.isna(c) or (isinstance(c, str) and c.strip().lower() in {"nan", "not specified"}):
            return "Indeterminate"
        else:
            return c.capitalize()
        

    def parse_auth_class(self):
        self.auth_label = self.standardize_class(getattr(self, "StandardizedClass", None))
        if self.auth_label is not None:
            return
    
        func_class = getattr(self, "auth_reported_func_class", None)
    
        score = getattr(self, "score", None)
    
        intervals = extract_intervals_from_row(self.row)
        # print('intervals',intervals)
    
        if func_class is not None and self.Dataset in func_class_map:
            self.auth_label = standardize_auth_label(
                dataset_name=self.Dataset,
                func_class=func_class,
                score=score,
                intervals=intervals if intervals else None
            )
            # print("class",self.auth_label)
        else:
            if score is not None and intervals:
                self.auth_label = classify_by_intervals(score, intervals)
                # print("interval",self.auth_label)
            else:
                self.auth_label = "Indeterminate"
                # print("ind",self.auth_label)


    def eval_quality(self):
        one_star_status = {
            'criteria provided, single submitter',
            'criteria provided, conflicting interpretations',
            'criteria provided, conflicting classifications',

        }
        two_star_statuses = {
            'criteria provided, multiple submitters, no conflicts',

        }
        three_star_statuses = {
            'reviewed by expert panel'
        }
        zero_star_statuses = {
            np.nan,
            'no assertion criteria provided',
            'no assertion provided',
            'no interpretation for the single variant',
            'no classification provided',
            '-'
        }
        if self.min_clinvar_star == 0:
            self.sufficient_quality = True
            self.sufficient_quality_2026 = True
        elif self.min_clinvar_star == 1:
            self.sufficient_quality = self.clinvar_star not in zero_star_statuses
            self.sufficient_quality_2026 = self.clinvar_star_2026 not in zero_star_statuses
        elif self.min_clinvar_star == 2:
            self.sufficient_quality = self.clinvar_star not in zero_star_statuses.union(one_star_status)
            self.sufficient_quality_2026 = self.clinvar_star_2026 not in zero_star_statuses.union(one_star_status)
        elif self.min_clinvar_star == 3:
            self.sufficient_quality = self.clinvar_star not in zero_star_statuses.union(one_star_status).union(two_star_statuses)
            self.sufficient_quality_2026 = self.clinvar_star_2026 not in zero_star_statuses.union(one_star_status).union(two_star_statuses)
        elif self.min_clinvar_star == 4:
            self.sufficient_quality = self.clinvar_star not in zero_star_statuses.union(one_star_status).union(two_star_statuses).union(three_star_statuses)
            self.sufficient_quality_2026 = self.clinvar_star_2026 not in zero_star_statuses.union(one_star_status).union(two_star_statuses).union(three_star_statuses)
        else:
            raise ValueError(f"Invalid min_clinvar_star value {self.min_clinvar_star}")

    def parse_gnomAD_MAF(self):
        """
        It is possible that the MAF is a list of values separated by a semicolon. If so, parse the list and obtain the maximum value.
        """
        self.is_gnomAD = not pd.isna(self.gnomad_MAF)

    @property
    def score(self):
        return self.assay_score

    @staticmethod
    def is_nan(value):
        return pd.isna(value) or value == "nan"


def summarize_datasets(dataframe_path, **kwargs):
    """
    Summarize the datasets in the dataframe at dataframe_path.

    Parameters
    ----------
    dataframe_path : str
        The path to the dataframe containing the dataset

    Keyword Arguments
    -----------------
    - output_file : str|Path
        The path to save the summary to

    Returns
    -------
    None
    """
    output_file = kwargs.get("output_file", None)
    if output_file is not None:
        output_file = Path(output_file)
        # output_file.mkdir(parents=True, exist_ok=True)
        f = open(str(output_file), "w")
    else:
        f = StringIO()
    df = PillarProjectDataframe(dataframe_path)
    for dataset_name, ds_df in df.dataframe.groupby("Dataset"):
        scoreset = Scoreset(
            ds_df,
            missense_only=kwargs.get("missense_only", False),
            synonymous_exclusive=kwargs.get("synonymous_exclusive", False),
            score_avg=kwargs.get("score_avg", False),
            score_col=kwargs.get("score_col", "auth_reported_score"),
            discordance_pct = kwargs.get("discordance_pct",0.0),
            downsample_n_variants = kwargs.get("downsample_n_variants",None),
            downsample_seed = kwargs.get("downsample_seed",None),
            filter_nonsense=kwargs.get("filter_nonsense", False)
        )
        f.write(f"{dataset_name}\n")
        f.write(str(scoreset))
        f.write("\n")
    if isinstance(f, StringIO):
        print(f.getvalue())
    else:
        f.close()


def csv_to_vcf(input_filepath, output_filepath):
    """
    Convert a CSV file to a gzipped VCF file.

    Parameters
    ----------
    input_filepath : str|Path
        The path to the input CSV file.
    output_filepath : str|Path
        The path to the output gzipped VCF file.

    Returns
    -------
    None
    """
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Filter rows with non-null hg38_start
    df = df[df["hg38_start"].notnull()]

    # Open the output file for writing
    with open(output_filepath, "w") as vcf_file:
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write("##source=tsv_to_vcf\n")
        vcf_file.write(
            """##contig=<ID=1,length=248956422,assembly=GRCh38>
##contig=<ID=2,length=242193529,assembly=GRCh38>
##contig=<ID=3,length=198295559,assembly=GRCh38>
##contig=<ID=4,length=190214555,assembly=GRCh38>
##contig=<ID=5,length=181538259,assembly=GRCh38>
##contig=<ID=6,length=170805979,assembly=GRCh38>
##contig=<ID=7,length=159345973,assembly=GRCh38>
##contig=<ID=8,length=145138636,assembly=GRCh38>
##contig=<ID=9,length=138394717,assembly=GRCh38>
##contig=<ID=10,length=133797422,assembly=GRCh38>
##contig=<ID=11,length=135086622,assembly=GRCh38>
##contig=<ID=12,length=133275309,assembly=GRCh38>
##contig=<ID=13,length=114364328,assembly=GRCh38>
##contig=<ID=14,length=107043718,assembly=GRCh38>
##contig=<ID=15,length=101991189,assembly=GRCh38>
##contig=<ID=16,length=90338345,assembly=GRCh38>
##contig=<ID=17,length=83257441,assembly=GRCh38>
##contig=<ID=18,length=80373285,assembly=GRCh38>
##contig=<ID=19,length=58617616,assembly=GRCh38>
##contig=<ID=20,length=64444167,assembly=GRCh38>
##contig=<ID=21,length=46709983,assembly=GRCh38>
##contig=<ID=22,length=50818468,assembly=GRCh38>
##contig=<ID=X,length=156040895,assembly=GRCh38>
##contig=<ID=Y,length=57227415,assembly=GRCh38>
##contig=<ID=M,length=16569,assembly=GRCh38>
"""
        )
        vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        df = df.sort_values(by=["Chrom", "hg38_start"])  # type: ignore
        # Write VCF rows
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vcf_file.write(
                f"{row['Chrom']}\t"
                f"{int(row.hg38_start)}\t"
                f"{row.get('mavedb_variant_urn', row['ID'])}\t"
                f"{row['ref_allele']}\t"
                f"{row['alt_allele']}\t.\t.\t.\n"
            )




class _UnionFind:
    """Path-compressed union-find for hashable elements."""
    def __init__(self):
        self._parent: dict = {}

    def add(self, x):
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x):
        self.add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


class MultiScoreset:
    """Combine D Scoresets into a single (N, D) score matrix for multivariate fitting.

    Observations may be missing in some dimensions (NaN).
    Sample assignments are merged (OR) across assays for each variant.
    """

    group_cols = ['Gene', 'Chrom', 'hg38_start', 'ref_allele', 'alt_allele']

    def __init__(self, scoresets, dataset_names=None, **kwargs):
        if len(scoresets) == 0:
            raise ValueError("scoresets cannot be empty")

        self.scoresets = scoresets
        self.d = len(scoresets)
        self._init_kwargs = dict(kwargs)

        if dataset_names is None:
            dataset_names = [
                getattr(s, 'scoreset_name', f"assay_{i}")
                for i, s in enumerate(scoresets)
            ]

        if len(dataset_names) != self.d:
            raise ValueError("dataset_names must match number of scoresets")

        self.dataset_names = dataset_names
        self._sample_names = kwargs.get("sample_names", [
            "Pathogenic/Likely Pathogenic",
            "Benign/Likely Benign",
            "population",
            "Synonymous",
        ])

        self._build()

    # ──────────────────────────────────────
    # Extract variant keys from a scoreset
    # ──────────────────────────────────────

    def _get_variant_keys(self, scoreset):
        """Extract variant keys for the KEPT variants in a scoreset.

        Uses _variant_codes (pre-filter, len = n_before_keep) indexed by
        _keep_mask to produce keys aligned with scoreset.scores (post-filter).
        """
        if not hasattr(scoreset, '_variant_codes') or not hasattr(scoreset, '_keep_mask'):
            raise ValueError(
                f"{getattr(scoreset, 'scoreset_name', 'scoreset')} missing "
                f"_variant_codes or _keep_mask; ensure it is a Scoreset instance."
            )

        all_codes = scoreset._variant_codes   # list, len = n_before_keep
        mask = scoreset._keep_mask             # bool array, len = n_before_keep

        if len(all_codes) != len(mask):
            raise ValueError(
                f"Length mismatch: _variant_codes ({len(all_codes)}) vs "
                f"_keep_mask ({len(mask)}) in "
                f"{getattr(scoreset, 'scoreset_name', '?')}"
            )

        # Filter to kept variants
        kept_codes = [all_codes[i] for i in range(len(mask)) if mask[i]]

        # Sanity check alignment with scores
        if len(kept_codes) != len(scoreset.scores):
            raise ValueError(
                f"After filtering: {len(kept_codes)} keys but "
                f"{len(scoreset.scores)} scores in "
                f"{getattr(scoreset, 'scoreset_name', '?')}"
            )

        # Return a frozenset of group_col tuples per variant.
        # New format: code is a list of tuples [(gene,chrom,pos,ref,alt), ...]
        # Old format (backward compat): code is a flat list [gene,chrom,pos,ref,alt]
        keys = []
        for code in kept_codes:
            if code is None:
                raise ValueError(
                    f"None variant code in {getattr(scoreset, 'scoreset_name', '?')}; "
                    f"ensure group columns {self.group_cols} are present."
                )
            if code and isinstance(code[0], tuple):
                # New format: list of tuples
                keys.append(frozenset(code))
            else:
                # Old format: flat list of values → single tuple
                keys.append(frozenset([tuple(code)]))

        return keys

    # ──────────────────────────────────────
    # Build the aligned (N, D) matrix
    # ──────────────────────────────────────

    def _build(self):
        variant_keys = []   # per assay: list[frozenset[tuple]]
        scores_list = []
        samples_list = []

        for s in self.scoresets:
            keys = self._get_variant_keys(s)   # list[frozenset[tuple]]
            scores = s.scores

            if hasattr(s, '_sample_assignments'):
                samples = s._sample_assignments
                if samples.shape[1] < 4:
                    pad = np.zeros((samples.shape[0], 4 - samples.shape[1]), dtype=bool)
                    samples = np.hstack([samples, pad])
            else:
                sa = s.sample_assignments
                if sa.shape[1] < 4:
                    pad = np.zeros((sa.shape[0], 4 - sa.shape[1]), dtype=bool)
                    samples = np.hstack([sa, pad])
                else:
                    samples = sa

            if not (len(keys) == len(scores) == samples.shape[0]):
                raise ValueError(
                    f"Inconsistent lengths in {getattr(s, 'scoreset_name', '?')}: "
                    f"keys={len(keys)}, scores={len(scores)}, samples={samples.shape[0]}"
                )

            variant_keys.append(keys)
            scores_list.append(scores)
            samples_list.append(samples)

        # ── Union-find: merge variants that share any group_col tuple ──────────
        # Each individual group_col tuple (gene,chrom,pos,ref,alt) is a node.
        # Within a single assay's variant entry, all tuples in its frozenset
        # represent the same biological entity → union them.
        # Across assays, if any tuple appears in both entries → same variant.
        uf = _UnionFind()

        # First pass: register every atom and union within each frozenset
        for assay_keys in variant_keys:
            for fset in assay_keys:
                atoms = list(fset)
                for a in atoms:
                    uf.add(a)
                for i in range(1, len(atoms)):
                    uf.union(atoms[0], atoms[i])

        # Collect canonical roots and assign a sequential variant index
        # Use a representative atom per biological variant as the canonical key
        root_to_idx: dict = {}
        canonical: list = []   # one representative atom per merged cluster

        def _get_or_create(fset):
            root = uf.find(next(iter(fset)))
            if root not in root_to_idx:
                root_to_idx[root] = len(canonical)
                canonical.append(root)
            return root_to_idx[root]

        n_variants = 0
        assay_row_idx = []   # per assay: list of global variant indices
        for assay_keys in variant_keys:
            row_indices = []
            for fset in assay_keys:
                idx = _get_or_create(fset)
                row_indices.append(idx)
            assay_row_idx.append(row_indices)
            n_variants = max(n_variants, max(row_indices) + 1 if row_indices else 0)

        # ── Allocate and fill matrices ─────────────────────────────────────────
        scores_matrix = np.full((n_variants, self.d), np.nan)
        sample_assignments = np.zeros((n_variants, 4), dtype=bool)

        for assay_i in range(self.d):
            row_indices = assay_row_idx[assay_i]
            scores = scores_list[assay_i]
            samples = samples_list[assay_i]
            for k, idx in enumerate(row_indices):
                scores_matrix[idx, assay_i] = scores[k]
                sample_assignments[idx] |= samples[k]

        # ── Keep only variants with at least one observed score ────────────────
        missing_mask = np.isnan(scores_matrix)
        keep_mask = ~np.all(missing_mask, axis=1)

        # Build _variants_kept: one representative group_col tuple per variant
        # (the canonical atom chosen by union-find).  Stored as a tuple matching
        # group_cols so downstream code (e.g. align_variants) can unpack it.
        all_canonical = canonical   # list of group_col tuples, one per variant
        self._variants_kept = [all_canonical[i] for i in range(n_variants) if keep_mask[i]]

        self._scores_matrix = scores_matrix
        self._missing_mask = missing_mask
        self._keep_mask = keep_mask
        self._scores = scores_matrix[keep_mask]
        self._missing = missing_mask[keep_mask]
        self.variants = all_canonical
        self._sample_assignments = sample_assignments[keep_mask]
        self.n_variants = self._scores.shape[0]

        self._xlims = tuple(
            (np.nanmin(self._scores[:, dim]), np.nanmax(self._scores[:, dim]))
            for dim in range(self.d)
        )
        self.sample_counts = self._sample_assignments.sum(axis=0)

    # ──────────────────────────────────────
    # Scoreset-compatible API
    # ──────────────────────────────────────

    @property
    def scores(self):
        """(N, D) array with NaN for missing."""
        return self._scores

    @property
    def full_scores(self):
        return self._scores_matrix

    @property
    def missing(self):
        return self._missing

    @property
    def keep_mask(self):
        return self._keep_mask

    @property
    def kept_variants(self):
        return self._variants_kept

    @property
    def sample_assignments(self):
        """(N, S) with empty-sample columns dropped."""
        return self._sample_assignments[:, self.sample_counts > 0]

    @property
    def n_samples(self):
        return (self.sample_counts > 0).sum()

    @property
    def scoreset_name(self):
        return " + ".join(self.dataset_names)

    @property
    def sample_names(self):
        return [
            self._sample_names[i]
            for i in range(min(len(self._sample_names), len(self.sample_counts)))
            if self.sample_counts[i] > 0
        ]

    @property
    def xlims(self):
        """Per-dimension (min, max) tuples."""
        return self._xlims

    @property
    def n_assays(self):
        return self.d

    @property
    def is_multivariate(self):
        return True

    @property
    def samples(self):
        """Iterate over (scores, sample_name) pairs.
        Scores are (n_sample, D) arrays with NaN for missing dims."""
        for si in range(min(len(self._sample_names), self._sample_assignments.shape[1])):
            if self.sample_counts[si] > 0:
                mask = self._sample_assignments[:, si]
                yield self._scores[mask], self._sample_names[si]

    @property
    def pathogenic_mask(self):
        return self._sample_assignments[:, 0]

    @property
    def benign_mask(self):
        return self._sample_assignments[:, 1]

    @property
    def population_mask(self):
        return self._sample_assignments[:, 2]

    @property
    def synonymous_mask(self):
        return self._sample_assignments[:, 3]

    def __repr__(self):
        out = f"MultiScoreset: {self.scoreset_name}\n"
        out += f"  {self.n_variants} variants, {self.d} assays\n"
        for dim in range(self.d):
            n_obs = (~np.isnan(self._scores[:, dim])).sum()
            out += f"  Dim {dim} ({self.dataset_names[dim]}): {n_obs}/{self.n_variants} observed\n"
        miss_pct = self._missing.mean() * 100
        out += f"  Overall: {miss_pct:.1f}% missing entries\n"
        for scores, name in self.samples:
            out += f"  {name}: {len(scores)} variants\n"
        return out

    def __len__(self):
        return self.n_variants


class BasicMultiScoreset:
    """Combine D BasicScoresets into a single (N, D) score matrix using
    explicit per-variant IDs to align across assays.

    Each input BasicScoreset must have its ``ids`` attribute populated.
    Variants are aligned by ID (typically a protein-variant string). The
    union of IDs across all inputs is taken; for each ID, missing
    dimensions are filled with NaN. Sample assignments are merged (OR).

    Exposes the same interface as MultiScoreset so it is a drop-in for
    Fit() and the analysis scripts.
    """

    def __init__(self, scoresets, dataset_names=None, **kwargs):
        if len(scoresets) == 0:
            raise ValueError("scoresets cannot be empty")
        for i, s in enumerate(scoresets):
            if getattr(s, "_ids", None) is None:
                raise ValueError(
                    f"BasicScoreset {i} ({getattr(s, 'scoreset_name', '?')}) "
                    f"has no ids; pass ids= when constructing each "
                    f"BasicScoreset before combining."
                )

        self.scoresets = scoresets
        self.d = len(scoresets)
        self._init_kwargs = dict(kwargs)

        if dataset_names is None:
            dataset_names = [
                getattr(s, "scoreset_name", f"assay_{i}")
                for i, s in enumerate(scoresets)
            ]
        if len(dataset_names) != self.d:
            raise ValueError("dataset_names must match number of scoresets")
        self.dataset_names = dataset_names

        s_total = max(s._sample_assignments.shape[1] for s in scoresets)
        self._n_samples_total = s_total

        default = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign",
                   "population", "Synonymous"]
        names = kwargs.get("sample_names", default[:s_total])
        if len(names) < s_total:
            names = list(names) + [
                f"sample_{i}" for i in range(len(names), s_total)
            ]
        self._sample_names = names

        self._build()

    @staticmethod
    def _parse_sample_assignments(sa_vals):
        """Parse 1-D sample_assignments into a 2-D boolean array.

        Accepts the same formats as BasicScoreset.validate_sample_assignments:
          - object/string dtype  → split each value on ',' and one-hot encode
          - numeric dtype        → treat each value as a sample id, one-hot encode
          - already 2-D array   → returned as-is (cast to bool)
        """
        sa_vals = np.asarray(sa_vals)
        if sa_vals.ndim == 2:
            return sa_vals.astype(bool)
        if sa_vals.dtype.kind in ('U', 'O'):
            all_keys = sorted({k for v in sa_vals for k in str(v).split(',')})
            key_to_idx = {k: i for i, k in enumerate(all_keys)}
            result = np.zeros((len(sa_vals), len(all_keys)), dtype=bool)
            for row, v in enumerate(sa_vals):
                for k in str(v).split(','):
                    result[row, key_to_idx[k]] = True
            return result
        # numeric — treat each value as a sample identifier
        unique_samples = sorted(set(sa_vals))
        result = np.zeros((len(sa_vals), len(unique_samples)), dtype=bool)
        for col_i, sid in enumerate(unique_samples):
            result[:, col_i] = sa_vals == sid
        return result

    @classmethod
    def from_dataframe(cls, dataframe, score_cols=None, id_col=None,
                       sample_assignments_col='sample_assignments',
                       dataset_names=None, **kwargs):
        """Create a BasicMultiScoreset directly from a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain score columns and a sample_assignments column.
        score_cols : list of str, optional
            Columns to use as score dimensions.  If None, all numeric columns
            except id_col and sample_assignments_col are used.
        id_col : str, optional
            Column to use as variant identifier.  Defaults to the DataFrame index.
        sample_assignments_col : str
            Column containing sample assignments (comma-separated strings, integer
            sample ids, or a pre-built boolean/int one-hot column is also accepted
            if the column holds array-like objects — but see _parse_sample_assignments
            for supported 1-D formats).
        dataset_names : list of str, optional
            Names for each score dimension.  Defaults to the column names in
            score_cols.
        **kwargs
            sample_names : list of str, optional
                Names for each sample class.  Defaults to the class-level defaults
                (P/LP, B/LB, population, Synonymous); any extra samples beyond the
                defaults are auto-named 'sample_{i}'.
        """
        df = dataframe.copy()

        exclude = {sample_assignments_col}
        if id_col is not None:
            exclude.add(id_col)

        if score_cols is None:
            score_cols = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]

        d = len(score_cols)
        if d == 0:
            raise ValueError("No score columns found or specified.")

        all_ids = df[id_col].tolist() if id_col is not None else list(df.index)
        N = len(df)

        scores_matrix = df[score_cols].to_numpy(dtype=float)

        sample_assignments = cls._parse_sample_assignments(
            df[sample_assignments_col].values
        )
        n_samples_total = sample_assignments.shape[1]

        if dataset_names is None:
            dataset_names = list(score_cols)
        elif len(dataset_names) != d:
            raise ValueError(
                f"dataset_names length {len(dataset_names)} does not match "
                f"number of score columns {d}"
            )

        default_names = [
            "Pathogenic/Likely Pathogenic", "Benign/Likely Benign",
            "population", "Synonymous",
        ]
        sample_names = list(kwargs.get("sample_names", default_names[:n_samples_total]))
        if len(sample_names) < n_samples_total:
            sample_names += [
                f"sample_{i}" for i in range(len(sample_names), n_samples_total)
            ]

        missing_mask = np.isnan(scores_matrix)
        keep_mask = ~np.all(missing_mask, axis=1)

        obj = cls.__new__(cls)
        obj.scoresets = []
        obj.d = d
        obj._init_kwargs = dict(kwargs)
        obj.dataset_names = dataset_names
        obj._n_samples_total = n_samples_total
        obj._sample_names = sample_names

        obj._all_ids = all_ids
        obj._variants_kept = [all_ids[i] for i in range(N) if keep_mask[i]]
        obj._scores_matrix = scores_matrix
        obj._missing_mask = missing_mask
        obj._keep_mask = keep_mask
        obj._scores = scores_matrix[keep_mask]
        obj._missing = missing_mask[keep_mask]
        obj._sample_assignments = sample_assignments[keep_mask]
        obj.n_variants = obj._scores.shape[0]
        obj._xlims = tuple(
            (np.nanmin(obj._scores[:, dim]), np.nanmax(obj._scores[:, dim]))
            for dim in range(d)
        )
        obj.sample_counts = obj._sample_assignments.sum(axis=0)

        return obj

    def _build(self):
        # Union of IDs preserving first-seen order
        seen: dict = {}
        for s in self.scoresets:
            for vid in s._ids:
                if vid not in seen:
                    seen[vid] = len(seen)
        all_ids = list(seen.keys())
        N = len(all_ids)

        scores_matrix = np.full((N, self.d), np.nan)
        sample_assignments = np.zeros((N, self._n_samples_total), dtype=bool)

        for assay_i, s in enumerate(self.scoresets):
            ids = s._ids
            scores = np.asarray(s.scores)
            samples = np.asarray(s._sample_assignments)
            S_s = samples.shape[1]
            for k, vid in enumerate(ids):
                idx = seen[vid]
                scores_matrix[idx, assay_i] = scores[k]
                sample_assignments[idx, :S_s] |= samples[k]

        missing_mask = np.isnan(scores_matrix)
        keep_mask = ~np.all(missing_mask, axis=1)

        self._all_ids = all_ids
        self._variants_kept = [all_ids[i] for i in range(N) if keep_mask[i]]
        self._scores_matrix = scores_matrix
        self._missing_mask = missing_mask
        self._keep_mask = keep_mask
        self._scores = scores_matrix[keep_mask]
        self._missing = missing_mask[keep_mask]
        self._sample_assignments = sample_assignments[keep_mask]
        self.n_variants = self._scores.shape[0]

        self._xlims = tuple(
            (np.nanmin(self._scores[:, dim]), np.nanmax(self._scores[:, dim]))
            for dim in range(self.d)
        )
        self.sample_counts = self._sample_assignments.sum(axis=0)

    # ── Scoreset/MultiScoreset-compatible API ────────────────────────

    @property
    def scores(self):
        return self._scores

    @property
    def full_scores(self):
        return self._scores_matrix

    @property
    def missing(self):
        return self._missing

    @property
    def keep_mask(self):
        return self._keep_mask

    @property
    def kept_variants(self):
        return self._variants_kept

    @property
    def sample_assignments(self):
        return self._sample_assignments[:, self.sample_counts > 0]

    @property
    def n_samples(self):
        return int((self.sample_counts > 0).sum())

    @property
    def scoreset_name(self):
        return " + ".join(self.dataset_names)

    @property
    def sample_names(self):
        return [
            self._sample_names[i]
            for i in range(min(len(self._sample_names), len(self.sample_counts)))
            if self.sample_counts[i] > 0
        ]

    @property
    def xlims(self):
        return self._xlims

    @property
    def n_assays(self):
        return self.d

    @property
    def is_multivariate(self):
        return True

    @property
    def samples(self):
        for si in range(min(len(self._sample_names),
                            self._sample_assignments.shape[1])):
            if self.sample_counts[si] > 0:
                mask = self._sample_assignments[:, si]
                yield self._scores[mask], self._sample_names[si]

    @property
    def pathogenic_mask(self):
        return self._sample_assignments[:, 0]

    @property
    def benign_mask(self):
        if self._sample_assignments.shape[1] > 1:
            return self._sample_assignments[:, 1]
        return np.zeros(self.n_variants, dtype=bool)

    @property
    def population_mask(self):
        if self._sample_assignments.shape[1] > 2:
            return self._sample_assignments[:, 2]
        return np.zeros(self.n_variants, dtype=bool)

    @property
    def synonymous_mask(self):
        if self._sample_assignments.shape[1] > 3:
            return self._sample_assignments[:, 3]
        return np.zeros(self.n_variants, dtype=bool)

    def __repr__(self):
        out = f"BasicMultiScoreset: {self.scoreset_name}\n"
        out += f"  {self.n_variants} variants, {self.d} assays\n"
        for dim in range(self.d):
            n_obs = (~np.isnan(self._scores[:, dim])).sum()
            out += (f"  Dim {dim} ({self.dataset_names[dim]}): "
                    f"{n_obs}/{self.n_variants} observed\n")
        miss_pct = self._missing.mean() * 100
        out += f"  Overall: {miss_pct:.1f}% missing entries\n"
        for scores, name in self.samples:
            out += f"  {name}: {len(scores)} variants\n"
        return out

    def __len__(self):
        return self.n_variants


if __name__ == "__main__":
    Fire()
    # summarize_datasets("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_gold_standard_02_05_25.csv",missense_only=False, synonymous_exclusive=False,output_file="dataset_summary_all_synonymousNonExclusive.txt")