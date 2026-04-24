#!/usr/bin/env python
# coding: utf-8

# In[184]:


import sys
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import seaborn as sns
logging.getLogger('matplotlib').setLevel(logging.ERROR)
sys.path.append("..")
from src.assay_calibration.fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from src.assay_calibration.fit_utils.two_sample import density_utils  # noqa: E402
import importlib
import src.assay_calibration.fit_utils.point_ranges
importlib.reload(src.assay_calibration.fit_utils.point_ranges)
from src.assay_calibration.fit_utils.point_ranges import (enforce_monotonicity_point_ranges, prior_equation_2c, prior_invalid, get_fit_prior, get_bootstrap_score_ranges, remove_insufficient_bootstrap_converage_points, check_thresholds_reached, extend_points_to_xlims, compute_single_fit_log_densities, compute_oob_variant_evidence_with_full_processing)  # noqa: E402
# from src.assay_calibration.data_utils.dataset_clinvar_version_functionality import (
#     Scoreset as Scoreset_clinvar_func,
# )
from src.assay_calibration.data_utils.dataset import (
    Scoreset,
)
from src.assay_calibration.data_utils.legacy_dataset import Scoreset as LegacyScoreset
from src.assay_calibration.fit_utils.utils import serialize_dict  # noqa: E402
from src.assay_calibration.plot_utils.utils import plot_scoreset, plot_scoreset_compare_point_assignments, plot_summary
from src.assay_calibration.fit_utils.model_selection import bootstrap_paired_test
from yang_dist import load_splits_dict
from collections import defaultdict
import pickle
import pandas as pd


def summarize_scoreset(fits,scoreset,save_filepath,use_median_prior,use_2c_equation,n_c,benign_method,oob_only, **kwargs):
    """
    Summarizes a scoreset based on the provided arguments.
    Args:
        args: An object containing the following attributes:
            - fits (List[Dict]) : List of fit results
            - scoreset_name (str) : Name of the scoreset to summarize
            - df (pandas.DataFrame, optional): A pandas DataFrame to be summarized. 
              If provided, this will be used directly.
            - pillar_df_filepath (str, optional): A file path to a CSV file. If `df` 
              is not provided, the CSV file at this path will be read into a pandas 
              DataFrame and used for summarization.
            - use_median_prior (bool). Whether to use median prior or 5-th percentile most conservative bootstrap for each threshold.
            - use_2c_equation (bool). Whether to use prior equation rather than EM for 2c.
    Optional Keyword Args:
        - point_values (List[int], optional): List of point values to assign. Defaults to 
          [1,2,3,4,5,6,7,8].
        - pathogenic_idx (int, optional): Index of the pathogenic component. Defaults to 0.
        - benign_idx (int, optional): Index of the benign component. Defaults to 1.
        - tolerance (float, optional): Tolerance for convergence in prior estimation. Defaults to 1e-4.
        - max_em_steps (int, optional): Maximum number of EM steps for prior estimation. Defaults to 10000.

    Note:
        Either `df` or `pillar_df_filepath` must be provided in `args`. If both are 
        provided, `df` takes precedence.
    """
    priors, prior, point_ranges, score_range, log_fp, log_fb, all_path_ranges, all_ben_ranges, C, variant_to_oob_points = process_fits(fits,scoreset,use_median_prior,use_2c_equation,benign_method,Path(save_filepath), oob_only)
    results = dict(prior=prior,
                   point_ranges=point_ranges,
                   priors=priors,
                   score_range=score_range,
                   log_lr_plus=log_fp - log_fb,
                   C=C,
                   all_path_ranges=all_path_ranges,
                   all_ben_ranges=all_ben_ranges)
    results = serialize_dict(results)
    save_filepath = Path(save_filepath)
    save_filepath.parent.mkdir(exist_ok=True,parents=True)
    # with open(save_filepath,'w') as f:
    #     json.dump(results,f,indent=2)
    # save_filepath_compact = save_filepath.parent / f"{save_filepath.stem}_compact.json"
    # with open(save_filepath_compact,'w') as f:
    #     json.dump({k: results[k] for k in ['prior','point_ranges']},
    #               f,indent=2)
    # scoreset_fit_figure = plot_scoreset(scoreset, results, fits,score_range, use_median_prior, use_2c_equation, n_c, benign_method, C)
    # figure_filepath = save_filepath.parent / f"{save_filepath.stem}_figure_fits.png"
    # scoreset_fit_figure.savefig(figure_filepath,bbox_inches='tight',dpi=300)
    # plt.close(scoreset_fit_figure) 
    # summary_fig = plot_summary(scoreset, fits, results, score_range, log_fp, log_fb, use_median_prior, use_2c_equation, n_c, benign_method, C, dataset)
    
    # summary_figure_filepath = save_filepath.parent / f"{save_filepath.stem}_figure_summary.png"
    # summary_fig.savefig(summary_figure_filepath,bbox_inches='tight',dpi=300)
    # plt.close(summary_fig)

    return scoreset, results, fits, score_range, f"({'equation' if use_2c_equation else 'em'}, {'median' if use_median_prior else '5-percentile'}, {benign_method})", n_c, variant_to_oob_points            


def process_fits(fits, scoreset, use_median_prior, use_2c_equation, benign_method, save_filepath, oob_only, **kwargs)->Tuple[np.ndarray,Dict[int,List[Tuple[float,float]]],np.ndarray,np.ndarray,np.ndarray,List[Dict[int,List[Tuple[float,float]]]],List[Dict[int,List[Tuple[float,float]]]]]:

    pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx = None, None, None, None
    for i,sample_name in enumerate(scoreset.sample_names):
        if scoreset.sample_counts[i] == 0:
            continue
        print(sample_name)
        if sample_name == "Pathogenic/Likely Pathogenic":
            pathogenic_idx = i
        elif sample_name == "Benign/Likely Benign":
            benign_idx = i
        elif sample_name == "gnomAD" or sample_name == "population":
            gnomad_idx = i
        elif sample_name == "Synonymous":
            synonymous_idx = i
        else:
            raise ValueError("invalid sample name")

    print(pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx)
    assert gnomad_idx is not None and (pathogenic_idx is not None or (benign_idx is not None or synonymous_idx is not None))
    if benign_idx is None:
        gnomad_idx -= 1
        if synonymous_idx is not None:
            synonymous_idx -= 1

    if pathogenic_idx is None:
        gnomad_idx -= 1
        if benign_idx is not None:
            benign_idx -= 1
        if synonymous_idx is not None:
            synonymous_idx -= 1

    print("pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx")
    print(pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx)

    log_filepath = save_filepath.parent / f"{save_filepath.stem}.log"
    with open(log_filepath,'w') as log_f:
    
        n_cores = os.cpu_count() or 1
    
        priors_filepath = save_filepath.parent / f"{save_filepath.stem.replace('_median','').replace('_5-percentile','')}_priors.npy"
        if not priors_filepath.exists() or True:
        
            if not use_2c_equation:
                fit_priors = np.array([get_fit_prior(fit, scoreset, benign_method,  pathogenic_idx=pathogenic_idx, benign_idx=benign_idx, 
                                                                                                                  gnomad_idx=gnomad_idx, synonymous_idx=synonymous_idx) for fit in fits])
            else:
                print('computing priors with equation...')
                fit_priors = []
                for fit in fits:
                    
                    if len(fit['fit']['weights']) == 3:
                        w_p = fit['fit']['weights'][pathogenic_idx]
                        w_g = fit['fit']['weights'][gnomad_idx]
                        if benign_idx is not None:
                            w_b = fit['fit']['weights'][benign_idx]
                        else:
                            w_b = fit['fit']['weights'][synonymous_idx]
                    elif len(fit['fit']['weights']) == 4:
                        w_p, w_b, w_g, w_s = fit['fit']['weights']
                    else:
                        raise ValueError(f"Number of samples != 3 or 4: {len(fit['fit']['weights'])}")
                    if benign_method == 'synonymous':
                        fit_priors.append(prior_equation_2c(w_p, w_s, w_g))
                    elif benign_method == 'avg':
                        w_bs = (np.array(w_b)+np.array(w_s))/2
                        fit_priors.append(prior_equation_2c(w_p, w_bs, w_g))
                    else:
                        fit_priors.append(prior_equation_2c(w_p, w_b, w_g))
                fit_priors = np.array(fit_priors)
                
            np.save(priors_filepath, fit_priors)
            
        else:
            print(f"loading priors from cached {'equation' if use_2c_equation else 'em'}")
            fit_priors = np.load(priors_filepath)
        
        # Filter invalid priors (needed for both in-bag and OOB)
        nan_mask = np.isnan(fit_priors)
        invalid_range_mask = (fit_priors <= 0) | (fit_priors >= 1)
        valid_mask = ~(nan_mask | invalid_range_mask)
        
        fit_priors = fit_priors[valid_mask]
        fits = np.array(fits)[valid_mask]
        
        point_values = kwargs.get('point_values',[1,2,3,4,5,6,7,8])
        
        # Compute median prior (needed for both in-bag and OOB)
        prior = np.nanmedian(fit_priors)
        print(f"prior: {prior}")
        
        # Compute score range (needed for both in-bag and OOB)
        observed_scores = scoreset.scores[scoreset._sample_assignments.any(1)]
        score_range = np.linspace(*np.percentile(observed_scores,[0,100]),10000)
    
        # Compute densities (needed for both in-bag and OOB)
        results_fpfb = Parallel(n_jobs=min(len(fits), n_cores), verbose=10)(
            delayed(compute_single_fit_log_densities)(
                fit, prior, score_range, benign_method,
                pathogenic_idx=pathogenic_idx,
                benign_idx=benign_idx,
                gnomad_idx=gnomad_idx,
                synonymous_idx=synonymous_idx
            )
            for fit, prior in zip(fits, fit_priors)
        )
        
        # Unpack results_fpfb
        _log_fp = np.array([r[0] if r[0] is not None else np.full(len(score_range), np.nan) for r in results_fpfb])
        _log_fb = np.array([r[1] if r[1] is not None else np.full(len(score_range), np.nan) for r in results_fpfb])
        
        log_fp = np.full((len(fits),len(score_range)),np.nan)
        log_fb = np.full((len(fits),len(score_range)),np.nan)
        
        # Determine scoreset_flipped (needed for both in-bag and OOB)
        scoreset_flipped = False
        if pathogenic_idx is not None:
            path_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,pathogenic_idx]])
        else:
            path_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,gnomad_idx]])

        if benign_idx is not None or synonymous_idx is not None:
            if benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
                ben_mean_score = (np.mean(scoreset.scores[scoreset._sample_assignments[:,benign_idx]])+np.mean(scoreset.scores[scoreset._sample_assignments[:,synonymous_idx]]))/2
            elif benign_method == 'synonymous' and synonymous_idx is not None:
                ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,synonymous_idx]])
            else:
                ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,benign_idx]]) if benign_idx is not None else np.mean(scoreset.scores[scoreset._sample_assignments[:,synonymous_idx]])
        else:
            ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,gnomad_idx]])
        
        if path_mean_score > ben_mean_score:
            scoreset_flipped = True

        # Manual override for specific dataset
        if scoreset.scoreset_name == "TARDBP_Bolognesi_Faure_2019":
            scoreset_flipped = True

        print(f"scoreset_flipped: {scoreset_flipped}", file=log_f)
        
        # Initialize in-bag outputs
        point_ranges = {}
        ranges_pathogenic = None
        ranges_benign = None
        C = None
        range_subset = None
    
        if not oob_only:
            # ========== IN-BAG PROCESSING ==========
            print("Computing in-bag calibration...")
            
            boot_points_filepath = save_filepath.parent / f"{save_filepath.stem.replace('_median','').replace('_5-percentile','')}_boot_points.npz"
            
            print('getting point ranges for each bootstrap...')
            results = Parallel(
                n_jobs=min(len(fits), n_cores),
                verbose=10
            )(
                delayed(get_bootstrap_score_ranges)(fitIdx, fit, fp, fb, score_range, fit_priors, point_values)
                for fitIdx, (fit, fp, fb) in enumerate(zip(fits, _log_fp, _log_fb))
            )
            
            # Update parent arrays in main process
            ranges_pathogenic, ranges_benign = [], []
            Cs = []
            
            for fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C_val in results:
                log_fp[fitIdx] = log_fp_local
                log_fb[fitIdx] = log_fb_local
                ranges_pathogenic.append({key: np.array(value).reshape(-1) for key, value in ranges_p.items()})
                ranges_benign.append({key: np.array(value).reshape(-1) for key, value in ranges_b.items()})
                Cs.append(C_val)
        
            log_lr_plus = log_fp - log_fb
            nan_counts = np.isnan(log_lr_plus).sum(0)
            range_subset = nan_counts < log_lr_plus.shape[0]
            
            print('log_lr_plus[0]', log_lr_plus[0], file=log_f)
            
            C = np.array([np.nanpercentile(Cs, 5), np.nanpercentile(Cs, 95)])
            print('ranges_p bootstrap 0, score +8:',ranges_pathogenic[0][1], file=log_f)

        
            percent_no_evidence = {point: 0.0 for point in point_values + list(-1*np.array(point_values))} 
            
            print('range_subset true',sum(range_subset), range_subset.shape, file=log_f)
            
            p_xaxis_5percentile_conservative = 5 if not scoreset_flipped else 95
            b_xaxis_5percentile_conservative = 95 if not scoreset_flipped else 5
            
            if prior > 0 and prior < 1:
                
                if use_median_prior:
                    print('using median prior to get unified thresholds...')
                    point_ranges_pathogenic, point_ranges_benign, C = calculate_score_ranges(
                        np.nanpercentile(log_lr_plus[:,range_subset], 5, axis=0),
                        np.nanpercentile(log_lr_plus[:,range_subset], 95, axis=0),
                        prior,
                        score_range[range_subset],
                        point_values
                    )
                    point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
                    if prior <= 0 or prior >= 1:
                        for point in point_ranges:
                            point_ranges[point] = []
                    
                else:
                    # use 5-percentile of most conservative bootstrap thresholds for each point assignment
                    print('using 5-percentile to get conservative thresholds...')
                    
                    p_max = max if not scoreset_flipped else min
                    b_min = min if not scoreset_flipped else max
                    p_inf = -np.inf if not scoreset_flipped else np.inf
                    b_inf = np.inf if not scoreset_flipped else -np.inf
        
                    conservative_thresholds = {}
                    print('boot prior:',np.nanmin(fit_priors), '-', np.nanmax(fit_priors), file=log_f)
                    for point_value in point_values:
                        conservative_thresholds[point_value] = np.nanpercentile([p_max(ranges_p[point_value]) if len(ranges_p[point_value]) > 0 else p_inf for ranges_p in ranges_pathogenic], p_xaxis_5percentile_conservative)
                        print(point_value,'nan bootstrap points:',np.isnan([p_max(ranges_p[point_value]) if len(ranges_p[point_value]) > 0 else p_inf for ranges_p in ranges_pathogenic]).sum(), file=log_f)
                        conservative_thresholds[-1*point_value] = np.nanpercentile([b_min(ranges_b[-1*point_value]) if len(ranges_b[-1*point_value]) > 0 else b_inf for ranges_b in ranges_benign], b_xaxis_5percentile_conservative)
        
                    print('conservative_thresholds',conservative_thresholds, file=log_f)
                    for point_value, threshold in conservative_thresholds.items():
                        assert point_value != 0
                        if np.isnan(threshold) or np.isinf(threshold):
                            point_ranges[point_value] = []
                            continue
                        valid_scores = score_range[range_subset]
                        if (point_value > 0 and not scoreset_flipped) or (point_value < 0 and scoreset_flipped):
                            if abs(point_value) == max(point_values):
                                point_ranges[point_value] = [valid_scores[0], threshold]
                            else:
                                lower_lim = conservative_thresholds[point_value+1] if point_value > 0 else conservative_thresholds[point_value-1]
                                if np.isnan(lower_lim):
                                    point_ranges[point_value] = [valid_scores[0], threshold]
                                else:
                                    point_ranges[point_value] = [lower_lim, threshold]
                        else:
                            if abs(point_value) == max(point_values):
                                point_ranges[point_value] = [threshold, valid_scores[-1]]
                            else:
                                upper_lim = conservative_thresholds[point_value-1] if point_value < 0 else conservative_thresholds[point_value+1]
                                if np.isnan(upper_lim):
                                    point_ranges[point_value] = [threshold, valid_scores[-1]]
                                else:
                                    point_ranges[point_value] = [threshold, upper_lim]
                        
                    for point_value in point_ranges:
                        if len(point_ranges[point_value]) != 0:
                            point_ranges[point_value] = [point_ranges[point_value]]
        
            print(f"point_ranges: {point_ranges}", file=log_f)
    
            print(f"lr path: {np.nanpercentile(log_lr_plus[:,range_subset],5,axis=0)}", file=log_f)
            print(f"lr ben: {np.nanpercentile(log_lr_plus[:,range_subset],95,axis=0)}", file=log_f)
            
            # remove_insufficient_bootstrap_converage_points(point_ranges, percent_no_evidence, point_values)
            # print(f"point_ranges after removing insufficient bootstraps: {point_ranges}", file=log_f)
            
            liberal = False if scoreset.scoreset_name in ["GCK_Gersing_2023_complementation", "DDX3X_Radford_2023_cLFC_day15", "DDX3X_Radford_2023_cLFC_day15_clinvar_2018", "F9_Popp_2025_heavy_chain"] else True
            enforce_monotonicity_point_ranges(point_ranges, point_values, score_range[range_subset], scoreset_flipped=scoreset_flipped, liberal=liberal, log_f=log_f)
            extend_points_to_xlims(point_ranges, point_values, score_range[range_subset], scoreset_flipped)
            enforce_monotonicity_point_ranges(point_ranges, point_values, score_range[range_subset], scoreset_flipped=scoreset_flipped, liberal=liberal, log_f=log_f) # enforce again after extending 
            print(f"point_ranges after enforcing monotonicity: {point_ranges}", file=log_f)
            
        else:
            # OOB-only mode: skip in-bag computation
            print("Skipping in-bag calibration (oob_only=True)")
            # Still need range_subset for OOB
            log_lr_plus = _log_fp - _log_fb
            nan_counts = np.isnan(log_lr_plus).sum(0)
            range_subset = nan_counts < log_lr_plus.shape[0]

            # Set log_fp and log_fb from densities (needed for OOB)
            for fitIdx in range(len(fits)):
                log_fp[fitIdx] = _log_fp[fitIdx]
                log_fb[fitIdx] = _log_fb[fitIdx]

            # ========== OOB PROCESSING (always runs) ==========
            print("Computing OOB calibration...")
            dataset_to_splits = load_splits_dict()
            variant_to_oob_points = compute_oob_variant_evidence_with_full_processing(
                dataset, fits, scoreset, dataset_to_splits, 
                fit_priors, valid_mask, 
                log_fp[:, range_subset], log_fb[:, range_subset],
                score_range[range_subset] if range_subset is not None else score_range,
                point_values, benign_method, n_c,
                scoreset_flipped=scoreset_flipped,
                min_oob_samples=10,
                n_jobs=os.cpu_count()
            )
    
    # Return appropriate values
    if oob_only:
        # Return None for in-bag specific outputs
        return (
            fit_priors, 
            prior, 
            None,  # point_ranges
            score_range[range_subset] if range_subset is not None else score_range,
            log_fp,  # log_fp subset
            log_fb,  # log_fb subset
            None,  # ranges_pathogenic
            None,  # ranges_benign
            None,  # C
            variant_to_oob_points
        )
    else:
        # Return full outputs
        return (
            fit_priors, 
            prior, 
            point_ranges,
            score_range[range_subset],
            log_fp[:,range_subset], 
            log_fb[:,range_subset], 
            ranges_pathogenic, 
            ranges_benign, 
            C, 
            None
        )

    


# In[159]:


import gzip

mode = "point_assignment_replicate_run"

if mode == "point_assignment_comparison":
    results_name = 'initial_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
    
    results_name = 'clinvar_circ_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = {**results, **json.load(f)}

elif mode == "point_assignment_clinvar_2018":
    results_name = 'clinvar_2018_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
        
elif mode == "point_assignment_semifinal_rerun":
    results_name = 'semifinal_run_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
elif mode == "point_assignment_12_04_25_rerun":
    results_name = '12_04_25_rerun_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
        
elif mode == "point_assignment_replicate_run":
    # results_name = 'replicate_run_datasets_results_1000bootstraps_100fits'
    results_name = 'bap1_rad51c_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
        
    # with gzip.open(f"/data/ross/assay_calibration/{results_name.replace('replicate_run','tsc2')}.json.gz", 'rt', encoding='utf-8') as f:
    #     results = {**results, **json.load(f)}


    


# In[160]:


print(len(results), 'datasets\n')
print(*results.keys(),sep='\n')


# In[161]:


dataset_configs = {
    "ASPA_Grønbæk-Thygesen_2024_abundance": ("3c", "avg"),
    "ASPA_Grønbæk-Thygesen_2024_toxicity": ("3c", "avg"),
    "BARD1_unpublished": ("2c", "avg"),
    "CALM1_CALM2_CALM3_Weile_2017": ("3c", "avg"),
    "CARD11_Meitlis_2020_DMSO_no_introns": ("3c", "avg"),
    "CARD11_Meitlis_2020_Ibrutinib_no_introns": ("3c", "avg"),
    "CBS_Sun_2020_high_B6": ("3c", "avg"),
    "CBS_Sun_2020_low_B6": ("2c", "avg"),
    "CHK2_Gebbia_2024": ("2c", "benign"),
    "CHEK2_Gebbia_2024": ("2c", "benign"),
    "CRX_Shepherdson_2024": ("2c", "avg"),
    "CTCF_unpublished": ("2c", "avg"),
    "F9_Popp_2025_carboxy_F9_specific": ("3c", "avg"),
    "F9_Popp_2025_carboxy_gla_motif": ("3c", "avg"),
    "F9_Popp_2025_heavy_chain": ("3c", "benign"),
    "F9_Popp_2025_light_chain": ("3c", "benign"),
    "F9_Popp_2025_strep_2": ("3c", "avg"),
    "FKRP_Ma_2024": ("3c", "avg"),
    "G6PD_unpublished": ("2c", "avg"),
    "GCK_Gersing_2023_complementation": ("3c", "avg"),
    "GCK_Gersing_2024_abundance": ("2c", "avg"),
    "HMBS_van_Loggerenberg_2023_combined": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_ubquitous": ("2c", "avg"),
    "JAG1_Gilbert_2024": ("3c", "avg"),
    "KCNE1_Muhammad_2024_absence_of_WT": ("3c", "avg"),
    "KCNE1_Muhammad_2024_potassium_flux": ("2c", "avg"),
    "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
    "KCNH2_Jiang_2022": ("2c", "avg"),
    "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"),
    "KCNQ4_Zheng_2022_current_homozygous": ("3c", "avg"),
    "KCNQ4_Zheng_2022_v12_homozygous": ("2c", "avg"),
    "LARGE1_Ma_2024": ("3c", "avg"),
    "MSH2_Jia_2021": ("2c", "avg"),
    "NDUFAF6_Sung_2024": ("2c", "avg"),
    "OTC_Lo_2023": ("3c", "avg"),
    "PALB2_unpublished": ("2c", "avg"),
    "PAX6_McDonnell_2024_BLX_geneticin": ("3c", "avg"),
    "PAX6_McDonnell_2024_BLX_no_geneticin": ("3c", "avg"),
    "PAX6_McDonnell_2024_LE9_geneticin": ("2c", "avg"),
    "PAX6_McDonnell_2024_LE9_no_geneticin": ("2c", "avg"),
    "RAD51D_unpublished": ("3c", "avg"),
    "RHO_Wan_2019": ("2c", "avg"),
    "SCN5A_Glazer_2020": ("3c", "avg"),
    "SCN5A_Ma_2024_current_density": ("2c", "avg"),
    "TP53_Boettcher_2019": ("2c", "avg"),
    "TP53_Fortuno_2021_Kato_meta": ("2c", "avg"),
    "TP53_Giacomelli_2018_combined_score": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53WT_Nutlin3": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_Nutlin3": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_etoposide": ("2c", "avg"),
    "TP53_Kato_2003_AIP1nWT": ("3c", "avg"),
    "TP53_Kato_2003_BAXnWT": ("3c", "avg"),
    "TP53_Kato_2003_GADD45nWT": ("2c", "avg"),
    "TP53_Kato_2003_MDM2nWT": ("3c", "avg"),
    "TP53_Kato_2003_NOXAnWT": ("3c", "avg"),
    "TP53_Kato_2003_P53R2nWT": ("2c", "avg"),
    "TP53_Kato_2003_WAF1nWT": ("3c", "avg"),
    "TP53_Kato_2003_h1433snWT": ("3c", "avg"),
    "TPK1_Weile_2017": ("2c", "avg"),
    "TSC2_rapgap_unpublished": ("3c", "avg"),
    "TSC2_tuberin_unpublished": ("3c", "avg"),
    "XRCC2_unpublished": ("2c", "avg"),
    "BAP1_Waters_2024": ("3c", "avg"),
    "BRCA1_Adamovich_2022_Cisplatin": ("2c", "avg"),
    "BRCA1_Adamovich_2022_HDR": ("2c", "avg"),
    "BRCA1_Findlay_2018": ("2c", "avg"),
    "BRCA2_Hu_2024": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_Cisplatin": ("3c", "benign"),
    "BRCA2_Sahu_2023_exon13_Olaparib": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_SGE": ("3c", "benign"),
    "BRCA2_Sahu_2023_exon13_global_score": ("2c", "avg"),
    "BRCA2_Sahu_2025_HDR": ("2c", "avg"),
    "BRCA2_unpublished": ("3c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15": ("2c", "benign"),
    # "PTEN_Matreyek_2018": ("2c", "avg"), # replace with filtered nonsense
    "PTEN_Mighell_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14": ("2c", "avg"),
    "VHL_Buckley_2024": ("2c", "benign"),
    "BAP1_Waters_2024_clinvar_2018": ("3c", "avg"),
    "BRCA1_Adamovich_2022_Cisplatin_clinvar_2018": ("2c", "avg"),
    "BRCA1_Adamovich_2022_HDR_clinvar_2018": ("2c", "avg"),
    "BRCA1_Findlay_2018_clinvar_2018": ("2c", "avg"),
    "BRCA2_Hu_2024_clinvar_2018": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_Cisplatin_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2023_exon13_Olaparib_clinvar_2018": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_SGE_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2023_exon13_global_score_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2025_HDR_clinvar_2018": ("2c", "avg"),
    "BRCA2_unpublished_clinvar_2018": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
    # "PTEN_Matreyek_2018_clinvar_2018": ("2c", "avg"), # replace with filtered nonsense
    "PTEN_Mighell_2018_clinvar_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("2c", "avg"),
    "VHL_Buckley_2024_clinvar_2018": ("2c", "benign"),
    "KCNH2_Kozek_Glazer_2020": ("3c", "avg"),
    "TP53_Fayer_2021_meta": ("2c", "avg"),
    "TARDBP_Bolognesi_Faure_2019": ("3c", "avg"),
    "SGCB_Li_2023": ("2c", "avg"),
    "SFPQ_unpublished": ("2c", "avg")
}

dataset_relax_configs = {
    "BARD1_unpublished": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
    # "FKRP_Ma_2024": ("3c", "avg"),
    # "G6PD_unpublished": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_combined": ("2c", "avg"),
    "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_ubquitous": ("3c", "avg"),
    "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
    # "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"), # KEEP CONSTRAINED 3c avg # only compute lr+ when densities are high enough, extend
    # "LARGE1_Ma_2024": ("3c", "avg"), # KEEP CONSTRAINED
    "PTEN_Matreyek_2018_filtered_nonsense": ("2c", "avg"),
    "PTEN_Matreyek_2018_filtered_nonsense_clinvar_2018": ("2c", "avg"),
    "PTEN_Matreyek_2018": ("2c", "avg"),
    "PTEN_Matreyek_2018_clinvar_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14": ("3c", "avg"), # ENFORCE 10e-3 on normalized densities 0-1 
    "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("3c", "avg"),
    "RAD51D_unpublished": ("2c", "avg"),
    # "TSC2_Calhoun_cliPE_unpublished": ("2c", "avg"),
    # "TSC2_Calhoun_immuneSGE_unpublished": ("2c", "avg"),
    "TSC2_tuberin_unpublished": ("3c", "avg"),
    "VHL_Buckley_2024": ("3c", "avg"),
    "VHL_Buckley_2024_clinvar_2018": ("3c", "avg"),
    # "XRCC2_unpublished": ("2c", "avg"), 
    
    # new 11_22_25 rerun datasets (all relax)
    "CALM1_CALM2_CALM3_Weile_2017": ("3c", "avg"),
    "G6PD_unpublished": ("3c", "avg"),
    "MSH2_Jia_2021_clinvar_2018": ("2c", "avg"),
    "TP53_Boettcher_2019_clinvar_2018": ("2c", "avg"),
    "TP53_Fortuno_2021_Kato_meta_clinvar_2018": ("2c", "avg"),
    "TP53_Giacomelli_2018_combined_score_clinvar_2018": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53WT_Nutlin3_clinvar_2018": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_Nutlin3_clinvar_2018": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_etoposide_clinvar_2018": ("2c", "avg"),
    "TP53_Kato_2003_AIP1nWT_clinvar_2018": ("3c", "avg"),
    "TP53_Kato_2003_BAXnWT_clinvar_2018": ("3c", "avg"),
    "TP53_Kato_2003_GADD45nWT_clinvar_2018": ("2c", "avg"),
    "TP53_Kato_2003_MDM2nWT_clinvar_2018": ("3c", "avg"),
    "TP53_Kato_2003_NOXAnWT_clinvar_2018": ("3c", "avg"),
    "TP53_Kato_2003_P53R2nWT_clinvar_2018": ("2c", "avg"),
    "TP53_Kato_2003_WAF1nWT_clinvar_2018": ("3c", "avg"),
    "TP53_Kato_2003_h1433snWT_clinvar_2018": ("3c", "avg"),
    "TPK1_Weile_2017": ("2c", "avg"),
    "XRCC2_unpublished": ("2c", "avg"),
    "LARGE1_Ma_2024": ("3c", "avg"),
    "FKRP_Ma_2024": ("3c", "avg"),
}

# overwrite conflicting keys with dataset_relax_configs
# prev_dataset_configs = {**dataset_configs, **dataset_relax_configs}
with open('/data/ross/assay_calibration/new_dataset_configs.json','rt') as f:
    prev_dataset_configs = json.load(f)

keep_old_list = set()
with open('/data/ross/assay_calibration/keep_old_datasets.txt','r') as f:
    for line in f:
        keep_old_list.add(line.strip())


# for dataset, config in new_dataset_configs.items():
#     if dataset not in keep_old_list:
#         prev_dataset_configs[dataset] = config

df = pd.read_csv("/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz", sep='\t')


# In[41]:


all_variant_to_oob_points_fixed = {}




lib_datasets = [
        "GCK_Gersing_2023_complementation", 
        "DDX3X_Radford_2023_cLFC_day15", 
        "DDX3X_Radford_2023_cLFC_day15_clinvar_2018", 
        "F9_Popp_2025_heavy_chain"
]


# In[164]:


# all_variant_to_oob_points_semifull_processing = {}
with open("/data/ross/assay_calibration/variant_to_oob_points_full.pkl","rb") as f:
    all_variant_to_oob_points_full_processing = pickle.load(f)


# In[165]:


len(all_variant_to_oob_points_full_processing)


# In[132]:


"TP53_Fayer_2021_meta_clinvar_2018" in results.keys()


# In[185]:


### import importlib
import src.assay_calibration.plot_utils.utils
import src.assay_calibration.fit_utils.point_ranges
import src.assay_calibration.fit_utils.model_selection
importlib.reload(src.assay_calibration.plot_utils.utils)
importlib.reload(src.assay_calibration.fit_utils.point_ranges)
importlib.reload(src.assay_calibration.fit_utils.model_selection)
from src.assay_calibration.fit_utils.point_ranges import extend_points_to_xlims, enforce_monotonicity_point_ranges, get_fit_prior, compute_single_fit_log_densities, compute_oob_variant_evidence_with_full_processing
from src.assay_calibration.plot_utils.utils import plot_scoreset, plot_scoreset_compare_point_assignments, plot_summary
from src.assay_calibration.fit_utils.model_selection import bootstrap_paired_test


# syn_only_datasets = ["TARDBP_Bolognesi_Faure_2019", "SGCB_Li_2023", "SFPQ_unpublished"]#"TARDBP_Bolognesi_Faure_2019"]#("CALM1_CALM2_CALM3_Weile_2017", "CARD11_Meitlis_2020_DMSO_no_introns",
                    # "CARD11_Meitlis_2020_Ibrutinib_no_introns", "SCN5A_Glazer_2020", "KCNH2_Kozek_Glazer_2020")

# datasets_to_rerun = ["DDX3X_Radford_2023_cLFC_day15", "DDX3X_Radford_2023_cLFC_day15_clinvar_2018", "F9_Popp_2025_heavy_chain"]
# datasets_prior_rerun = ["BRCA2_Sahu_2023_exon13_Olaparib", "BRCA2_Sahu_2023_exon13_global_score", "CARD11_Meitlis_2020_Ibrutinib_no_introns"]

auto_dataset_config = {}

# with open("/data/ross/assay_calibration/variant_to_oob_points_simple.pkl","rb") as f:
#     all_variant_to_oob_points_lib = pickle.load(f)



start = False
for dataset in results.keys(): # PRIORITIZE URGENT DATASETS TO RUN FIRST, REST ALPHANUMERIC
    
    # if dataset not in author_reported_list:# and not dataset.endswith("clinvar_2018"):
    #     continue

    # if dataset not in ["GCK_Gersing_2023_complementation"]:#lib_datasets:
    #     continue

    # if dataset != "TSC2_combined_unpublished":
    #     continue

    
    mode = "point_assignment_replicate_run"
    if dataset in keep_old_list:
        if dataset in dataset_relax_configs:
            mode = "point_assignment_semifinal_rerun"
        elif "_clinvar_2018" in dataset:
            mode = "point_assignment_clinvar_2018"
        else:
            mode = "point_assignment_comparison"
            
    # if dataset not in datasets_prior_rerun:
    #     continue

    # if dataset != "KCNH2_Kozek_Glazer_2020" and dataset != "SFPQ_unpublished":
    #     continue
    # print(dataset)
    # if dataset.replace("_NEW_not_clinvar_2018","") not in syn_only_datasets:
    #     continue

    # if dataset != "TP53_Fayer_2021_meta_clinvar_2018":#, "VHL_Buckley_2024", "TP53_Fortuno_2021_Kato_meta", "TSC2_rapgap_unpublished"]:
    #     continue

    # if dataset in all_variant_to_oob_points_full_processing:
    #     continue
    
    # if dataset == 'GCK_Gersing_2023_complementation':
    #     start = True
    #     continue
    # if not start:
    #     continue
    
    
    print(dataset,"start")

    scoresets_dir = None
    # if mode == "point_assignment_comparison":
    #     scoresets_dir = "/data/ross/assay_calibration/scoresets"
    # elif mode == "point_assignment_clinvar_2018":
    #     scoresets_dir = "/data/ross/assay_calibration/scoresets_old_clinvar"
    # elif mode == "point_assignment_semifinal_rerun":
    #     scoresets_dir = "/data/ross/assay_calibration/scoresets_semifinal_run"
    # elif mode == "point_assignment_12_04_25_rerun":
    #     scoresets_dir = "/data/ross/assay_calibration/scoresets_12_04_25_rerun"
    # elif mode == "point_assignment_replicate_run":
    #     scoresets_dir = None
    # else:
    #     raise ValueError("no specified scoreset dir")
    

    # Load dataset once
    clinvar_release = '2018' if 'clinvar_2018' in dataset else '2025'    
    if "not_clinvar_2018" in dataset:
        clinvar_release = '2025'

    # if scoresets_dir is not None:
    #     dataset_f = f"{scoresets_dir}/{dataset}.json"
    #     scoreset = Scoreset.from_json(dataset_f, 
    #                     clinvar_release=clinvar_release,
    #                     min_clinvar_star=1)
    # else:
    #     scoreset = Scoreset(df[df["Dataset"] == dataset],
    #                     clinvar_release=clinvar_release,
    #                     min_clinvar_star=1)
    
    # scoreset = LegacyScoreset.from_json(dataset_f)

    # print(scoreset)
    
    # n_samples = len([s for s in scoreset.samples])

    # if n_samples < 3:
    #     continue
    
    dataset_save = dataset.replace("_NEW_not_clinvar_2018","")

    if dataset_save.replace("_clinvar_2018","") in prev_dataset_configs:
        n_c_options, benign_options = prev_dataset_configs[dataset_save.replace("_clinvar_2018","")]
    else:
        print(f"{dataset_save} not in manually selected configs, selecting 3c avg")
        n_c_options, benign_options = ["3c","avg"]#["2c", "3c"], ["benign", "avg"]

    print("benign_options", benign_options)

    # these may be overwritten by accident
    save_dir = f'/data/ross/assay_calibration/{mode.replace("point_assignment_12_04_25_rerun","point_assignment_semifinal_rerun")}/{dataset_save}'
    os.makedirs(save_dir, exist_ok=True)

    scoresets_dict, summary_dict, scoreset_fits_dict, score_ranges_dict = {},{},{},{}
    figure_filepath = f"{save_dir}/{dataset_save}_point_comparison.png"
    # if os.path.exists(figure_filepath):
    #     print(f'path exists: {figure_filepath}, skipping')
    #     continue

    n_c = f"{bootstrap_paired_test(results[dataset], verbose=False)['conservative_k']}c"
    n_c_auto_selected = n_c
    print(f"model selected n_c: {n_c}")
    # n_c_options = list(set([n_c, n_c_options]))#, benign_options = n_c, "avg"
    auto_dataset_config[dataset] = (n_c, benign_options)
    
    for n_c in [n_c_options]:#['2c','3c'] if "VHL_Buckley" not in dataset else ['2c','3c','4c']:
        assert n_c in ["2c","3c","4c"]
        if n_c[-1] != 'c':
            n_c = str(n_c) + 'c'
        
        if len(results[dataset]) == 0:
            print('dataset empty')
            continue

        if n_c == '4c' and n_c not in results[dataset][list(results[dataset].keys())[0]]:
            # print(results[dataset][list(results[dataset].keys())[0]])
            continue

        boot_results = [results[dataset][key][n_c] for key,val in results[dataset].items()]
        
        # n_count, nn_count = 0,0
        # for b in boot_results:
        #     if b is None:
        #         n_count += 1
        #     else:
        #         nn_count += 1
        
        # print(n_count, nn_count)

        # boot_results = boot_results[:20] # for speed of dev
        
        if scoresets_dir is not None:
            dataset_f = f"{scoresets_dir}/{dataset}.json"
            scoreset = Scoreset.from_json(dataset_f, 
                            clinvar_release=clinvar_release,
                            min_clinvar_star=1)
        else:
            scoreset = Scoreset(df[df["Dataset"] == dataset.replace("_clinvar_2018","")],
                            clinvar_release=clinvar_release,
                            min_clinvar_star=1)

        n_samples = len([s for s in scoreset.samples])

        for benign_method in [benign_options]:#['benign', 'avg']:
            assert benign_method in ["avg","benign","synonymous"]
            use_median_prior, use_2c_equation = True, False
            # for use_median_prior in [True]:
            if True:
            # for use_median_prior, use_2c_equation in zip([False, True, False], [False, False, True]):
                    
                # only do synonymous analysis if synonymous exists and only for one config
                
                experiment_code = f'{dataset_save}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
                
                pkl_filepath = f'{save_dir}/{experiment_code}.pkl'
                if not os.path.exists(pkl_filepath) and benign_method != 'benign' and n_samples != 4:
                    benign_method = "benign"
                    
                if n_c != '2c' and use_2c_equation: # N/A
                    continue
    
                experiment_code = f'{dataset_save}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
                
                save_filepath = f'{save_dir}/{experiment_code}.json'
                pkl_filepath = f'{save_dir}/{experiment_code}.pkl'
        
                # if os.path.exists(pkl_filepath):# or os.path.exists(pkl_filepath.replace("_avg","")):
                #     print(f'{pkl_filepath} exists')
                
                #     with open(pkl_filepath,'rb') as f:
                #         scoreset, indv_summary, fits, score_range, config, n_c = pickle.load(f)

                #     print(dataset, config, "point_ranges:",indv_summary["point_ranges"], pkl_filepath)
                        
                #     if not config.startswith('(e'): # old config, swap
                #         config = f"({'equation' if use_2c_equation else 'em'}, {'median' if use_median_prior else '5-percentile'}, {benign_method})"
                    
                #     if 'C' in indv_summary and indv_summary["point_ranges"] is not None and len(indv_summary["point_ranges"]) != 0:
                #         scoresets_dict[(config,n_c)] = scoreset
                #         summary_dict[(config,n_c)] = indv_summary
                #         scoreset_fits_dict[(config,n_c)] = fits
                #         score_ranges_dict[(config,n_c)] = score_range
                #         continue

                    
                print(f'\n{"-"*80}\nStarting {experiment_code}...\n{"-"*80}\n')
                try:
                
                    scoreset, indv_summary, fits, score_range, config, _, variant_to_oob_points = summarize_scoreset(boot_results,scoreset,save_filepath,use_median_prior,use_2c_equation,n_c,benign_method,oob_only=True)
                    all_variant_to_oob_points_full_processing[dataset_save] = variant_to_oob_points
                    if indv_summary is None or indv_summary["point_ranges"] is None:
                        continue
                    
                    print(dataset_save, variant_to_oob_points)
                    
                    del indv_summary['all_path_ranges']
                    del indv_summary['all_ben_ranges']
                    
                    scoresets_dict[(config,n_c)] = scoreset
                    summary_dict[(config,n_c)] = indv_summary
                    scoreset_fits_dict[(config,n_c)] = fits
                    score_ranges_dict[(config,n_c)] = score_range

                    # if dataset == "TSC2_combined_unpublished":
                    with open(pkl_filepath, 'wb') as f:
                        pickle.dump((scoreset, indv_summary, fits, score_range, config, n_c), f)

                except AssertionError as e:
                    print(e)

    # try:
        
    #     point_comparison_figure = plot_scoreset_compare_point_assignments(dataset_save, scoresets_dict, summary_dict, scoreset_fits_dict, score_ranges_dict, n_samples, n_c_auto_selected)
    #     point_comparison_figure.savefig(figure_filepath,bbox_inches='tight',dpi=300)
    #     plt.close(point_comparison_figure)

    # except Exception as e:
    #     print(e)


# In[125]:


len(all_variant_to_oob_points_semifull_processing.keys())


# In[81]:


with open("/data/ross/assay_calibration/variant_to_oob_points_simple.pkl","wb") as f:
    pickle.dump(all_variant_to_oob_points_fixed, f)


# In[96]:


with open("/data/ross/assay_calibration/variant_to_oob_points_lib.pkl","wb") as f:
    pickle.dump(all_variant_to_oob_points_lib, f)


# In[ ]:


all_variant_to_oob_points_semifull_processing


# In[182]:


len(all_variant_to_oob_points_full_processing)


# In[178]:


all_variant_to_oob_points_full_processing["TSC2_combined_unpublished"]


# In[186]:


with open("/data/ross/assay_calibration/variant_to_oob_points_full.pkl","wb") as f:
    pickle.dump(all_variant_to_oob_points_full_processing, f)


# In[ ]:





# In[126]:


# all_variant_to_oob_points_semifull_processing = {}

# with open("/data/ross/assay_calibration/variant_to_oob_points_simple.pkl","rb") as f:
#     all_variant_to_oob_points_semifull_processing = pickle.load(f)

# with open("/data/ross/assay_calibration/variant_to_oob_points_full.pkl","rb") as f:
#     lib_points = pickle.load(f)

# for dataset in lib_datasets:
#     if dataset not in all_variant_to_oob_points_semifull_processing:
#         continue
#     all_variant_to_oob_points_semifull_processing[dataset] = lib_points[dataset]
#     print(dataset,'lib')

with open("/data/ross/assay_calibration/variant_to_oob_points_semifull.pkl","wb") as f:
    pickle.dump(all_variant_to_oob_points_semifull_processing, f)


# In[ ]:





# In[83]:


# all_variant_to_oob_points_fixed


# In[80]:


len(all_variant_to_oob_points_fixed.keys())


# In[209]:


all_variant_to_oob_points["BRCA1_Findlay_2018"]


# In[174]:


results.keys()


# In[168]:


prev_dataset_configs[dataset_save]


# In[ ]:


auto_dataset_config


# In[135]:


prev_dataset_configs


# In[206]:


match = 0
increase_c = 0
decrease_c = 0

for key in auto_dataset_config:
    auto = auto_dataset_config[key][0]
    man = prev_dataset_configs[key][0]

    if auto == man:
        match += 1
    elif auto > man:
        increase_c += 1
    else:
        decrease_c += 1

match /= len(auto_dataset_config)
increase_c /= len(auto_dataset_config)
decrease_c /= len(auto_dataset_config)

print(f"{100*match:.1f}% match, {100*increase_c:.1f}% 2c->3c, {100*decrease_c:.1f}% 3c->2c")


# In[143]:


with open('/data/ross/assay_calibration/auto_dataset_config.json','wt') as f:
    json.dump(auto_dataset_config, f, indent=2)


# In[ ]:


'done'


# In[ ]:


'done'


# In[ ]:


# results_name = 'initial_datasets_results_1000bootstraps_100fits'
# with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
#     results = json.load(f)

# boot_results = [results["ASPA_Grønbæk-Thygesen_2024_toxicity"][key][n_c] for key,val in results["ASPA_Grønbæk-Thygesen_2024_toxicity"].items()]

# boot_results[0]
# {'dataset_name': 'ASPA_Grønbæk-Thygesen_2024_toxicity',
#  'bootstrap_seed': 0,
#  'num_components': 2,
#  'fit_idx': 54,
#  'fit': {'component_params': [[0.005207834168390946,
#     0.021940890977596272,
#     0.061134479247006705],
#    [1.4598109894471645, 0.5348310182877707, 0.237462765286969]],
#   'weights': [[0.23213910517054026, 0.7678608948294599],
#    [1.0, 1.3911847093481276e-91],
#    [0.7770771791411332, 0.22292282085886678]],
#   'kmeans': 'method_of_moments',
#   'xlims': [-0.1191, 0.9763],
#   'times_submerged': []},
#  'val_ll': 0.4940112234637928}


# In[14]:


import glob
datasets_with_4c = set()
for job_f in glob.glob('/data/ross/assay_calibration/explorer_jobs_semifinal_run/jobs/*.pkl'):
    with open(job_f,'rb') as f:
        jobs = pickle.load(f)

    found_4c = False
    for job in jobs:
        if len(job['jobs_4c']) != 0:
            found_4c = True
            datasets_with_4c.add(job['dataset_name'])
datasets_with_4c


# In[3]:


# import pickle
# from joblib import Parallel, delayed
# import os

# datasets_to_visualize = [
#     'HMBS_van_Loggerenberg_2023_combined',
#     'KCNH2_O_Neill_2024_surface_expression',
#     'TP53',
# ]

# def process_dataset(dataset, results):
#     """Process a single dataset"""
    
#     # Check if dataset should be visualized
#     for prospect_dataset in datasets_to_visualize:
#         if prospect_dataset in dataset:
#             break
#     else:
#         return None  # Skip this dataset
    
#     dataset_f = f"/data/ross/assay_calibration/scoresets/{dataset}.json"
#     scoreset = Scoreset.from_json(dataset_f)
#     n_samples = len([s for s in scoreset.samples])

#     save_dir = f'/data/ross/assay_calibration/point_assignment_comparison/{dataset}'
#     os.makedirs(save_dir, exist_ok=True)

#     scoresets_dict, summary_dict, scoreset_fits_dict, score_ranges_dict = {},{},{},{}
    
#     for n_c in ['2c','3c']:
        
#         boot_results = [results[dataset][key][n_c] for key,val in results[dataset].items()]
        
#         for benign_method in ['benign', 'avg']:
#             use_median_prior, use_2c_equation = True, False
#             if True:
                    
#                 # only do synonymous analysis if synonymous exists and only for one config
#                 if benign_method != 'benign' and (not use_median_prior or use_2c_equation or n_samples != 4):
#                     continue
                    
#                 if n_c == '3c' and use_2c_equation: # N/A
#                     continue
    
#                 experiment_code = f'{dataset}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
                
#                 save_filepath = f'{save_dir}/{experiment_code}.json'
#                 pkl_filepath = f'{save_dir}/{experiment_code}.pkl'
        
#                 if os.path.exists(pkl_filepath):
#                     print(f'{pkl_filepath} exists')
#                     with open(pkl_filepath,'rb') as f:
#                         scoreset, indv_summary, fits, score_range, config, n_c = pickle.load(f)
                        
#                     if not config.startswith('(e'):
#                         config = f"({'equation' if use_2c_equation else 'em'}, {'median' if use_median_prior else '5-percentile'}, {benign_method})"
                    
#                     if 'C' in indv_summary:
#                         scoresets_dict[(config,n_c)] = scoreset
#                         summary_dict[(config,n_c)] = indv_summary
#                         scoreset_fits_dict[(config,n_c)] = fits
#                         score_ranges_dict[(config,n_c)] = score_range
#                         continue
                    
#                 print(f'\n{"-"*80}\nStarting {experiment_code}...\n{"-"*80}\n')
#                 try:
                
#                     scoreset, indv_summary, fits, score_range, config, _ = summarize_scoreset(boot_results,scoreset,save_filepath,use_median_prior,use_2c_equation,n_c,benign_method)
#                     del indv_summary['all_path_ranges']
#                     del indv_summary['all_ben_ranges']
                    
#                     scoresets_dict[(config,n_c)] = scoreset
#                     summary_dict[(config,n_c)] = indv_summary
#                     scoreset_fits_dict[(config,n_c)] = fits
#                     score_ranges_dict[(config,n_c)] = score_range
            
#                     with open(pkl_filepath, 'wb') as f:
#                         pickle.dump((scoreset, indv_summary, fits, score_range, config, n_c), f)

#                 except Exception as e:
#                     print(e)

#     try:
#         point_comparison_figure = plot_scoreset_compare_point_assignments(dataset, scoresets_dict, summary_dict, scoreset_fits_dict, score_ranges_dict, n_samples)
        
#         figure_filepath = f"{save_dir}/{dataset}_point_comparison.png"
#         point_comparison_figure.savefig(figure_filepath,bbox_inches='tight',dpi=300)
#         plt.close(point_comparison_figure)

#     except NotImplementedError as e:
#         print(e)
    
#     return dataset

# # Parallel execution
# datasets_to_process = [d for d in results.keys() 
#                        if any(prospect in d for prospect in datasets_to_visualize)]

# print(f"Processing {len(datasets_to_process)} datasets in parallel...")

# min_cores = 20
# n_cores = min(min_cores, len(datasets_to_process))

# processed = Parallel(n_jobs=n_cores, verbose=10)(
#     delayed(process_dataset)(dataset, results) 
#     for dataset in datasets_to_process
# )

# print(f"\nCompleted {sum(1 for p in processed if p is not None)} datasets")


# In[8]:


# [(-8, []), (-7, []), (-6, []), (-5, []), (-4, []), (-3, []), (-2, [[-0.18066820274493, 1.1121633402781734]]), (-1, [[-0.2190387471559836, -0.18066820274493]]), (1, [[-0.9107317610725785, -0.8994764013786694]]), (2, [[-0.92352194254293, -0.9107317610725785]]), (3, [[-0.9409165893426077, -0.92352194254293]]), (4, [[-0.9961701732945247, -0.9409165893426077]]), (5, [[-3.3065885540987723, -0.9961701732945247]]), (6, []), (7, []), (8, [])]


# In[24]:


# with open('/data/ross/assay_calibration/point_assignment_comparison/test/evidence_data.json','rt') as f:
#     old_data = json.load(f)


# In[44]:


# np.nanpercentile(indv_summary['log_lr_plus'], 5, axis=0)[0]


# In[50]:


# tauP,tauB,_ = list(thresholds_from_prior(indv_summary['prior'],[1,2,3,4,5,6,7,8]))


# In[51]:


# tauP


# In[48]:


# tauB


# In[52]:


# C = 136
# for i in range(1,9):
#     print(f'{i}: {C**(i/8)}')


# In[39]:


# def plot_bootstrap_percentiles(data, x, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 5))
#     else:
#         fig = ax.get_figure()
    
#     x = np.array(x)
#     p = np.nanpercentile(data, [5, 50, 95], axis=0)
#     print(p[0][0])
    
#     ax.plot(x, p[1], 'k-', label='Median', linewidth=2)
#     ax.plot(x, p[0], 'r--', label='5th %ile')
#     ax.plot(x, p[2], 'b--', label='95th %ile')
#     ax.fill_between(x, p[0], p[2], color='gray', alpha=0.3)
#     ax.legend()
#     ax.grid(alpha=0.3)
    
#     return fig, ax
    
# plot_bootstrap_percentiles(np.array(old_data['log_lrPlus']), old_data['scores'])


# In[ ]:




