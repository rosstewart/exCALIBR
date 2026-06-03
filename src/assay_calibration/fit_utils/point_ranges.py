import sys
sys.path.append('..')
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
from .fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from .cfusn import density_utils  # noqa: E402
from ..data_utils.dataset import Scoreset  # noqa: E402
from .utils import serialize_dict  # noqa: E402
from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# logging.getLogger('matplotlib').setLevel(logging.ERROR)

def enforce_monotonicity_point_ranges(point_ranges, point_values, score_range, scoreset_flipped=False, liberal=False, log_f=None):

    
    if liberal:
        # print('enforcing monotonicity in points...',file=log_f)
        for i in point_values:
            point = i # pathogenic
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][-1]] if not scoreset_flipped else [point_ranges[point][0]]
                    
            point = -i # benign
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][0]] if not scoreset_flipped else [point_ranges[point][-1]]

        # make sure none overlap
        for i in point_values:
            for j in point_values:
                if j <= i:
                    continue

                
                point_i = i # pathogenic
                point_j = j # pathogenic

                if len(point_ranges[point_i]) != 0 and len(point_ranges[point_j]) != 0:
                    # 2d array, should be flattened but idk if will break things
                    # if point_j encompasses point_i (in case of sudden spike), remove point_i
                    if point_ranges[point_i][0][0] >= point_ranges[point_j][0][0] and point_ranges[point_i][0][1] <= point_ranges[point_j][0][1]:
                        point_ranges[point_i] = []

                
                point_i = -i # benign
                point_j = -j # benign
                
                if len(point_ranges[point_i]) != 0 and len(point_ranges[point_j]) != 0:
                    # print(f"Checking if {point_j} encompasses {point_i}: "
                          # f"{point_ranges[point_i][0]} vs {point_ranges[point_j][0]}", file=log_f)
                    
                    if point_ranges[point_i][0][0] >= point_ranges[point_j][0][0] and \
                       point_ranges[point_i][0][1] <= point_ranges[point_j][0][1]:
                        print(f"  -> Removing {point_i}", file=log_f)
                        point_ranges[point_i] = []
                
        return
    
    max_path_points = None
    max_ben_points = None
    
    # print('enforcing monotonicity in points (std)...',file=log_f)
    for i in point_values:
        point = i # pathogenic

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point+1]) == 0: # highest evidence at 1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            if l != score_range[0] and h != score_range[-1]: # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_path_points = point

        if max_path_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point + 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_path_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_path_points is None:
                    max_path_points = point
                
        point = -i # benign

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point-1]) == 0: # highest evidence at -1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            if l != score_range[0] and h != score_range[-1]: # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_ben_points = point

        if max_ben_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point - 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_ben_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_ben_points is None:
                    max_ben_points = point



def extend_points_to_xlims(point_ranges, point_values, score_range, scoreset_flipped, log_f=None, inf=False):
    # print('extending points to xlims...',file=log_f)
    left = -np.inf if inf else score_range[0]
    right = np.inf if inf else score_range[-1]
    for i in point_values:
        point = i # pathogenic
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point+j in point_ranges:
                if len(point_ranges[point+j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if not scoreset_flipped else [[point_ranges[point][0][0], right]]
                
        point = -i # benign
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point-j in point_ranges:
                if len(point_ranges[point-j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if scoreset_flipped else [[point_ranges[point][0][0], right]]

    # check if one evidence extends entire range, in case of incorrectly determined flipped scoreset (BAD!)
    for i in point_values:
        pos, neg = i,-i
        if len(point_ranges[pos]) != 0 and point_ranges[pos][0][0] == left and point_ranges[pos][-1][-1] == right:
            point_ranges[pos] = []
            print(pos,'extends the whole score range, removing...')
        if len(point_ranges[neg]) != 0 and point_ranges[neg][0][0] == left and point_ranges[neg][-1][-1] == right:
            point_ranges[neg] = []
            print(neg,'extends the whole score range, removing...')

        
def prior_equation_2c(w_p, w_b, w_g):
    return (w_g[1] - w_b[1]) / (w_p[1] - w_b[1])

def prior_invalid(prior):
    return prior <= 0 or prior >= 1

def get_fit_prior(fit, scoreset_or_scores, benign_method, pathogenic_idx=0, benign_idx=1, gnomad_idx=2, synonymous_idx=3,
                  sample_assignments=None, **kwargs):
    if benign_idx is None:
        benign_idx = synonymous_idx
    if synonymous_idx is None:
        synonymous_idx = benign_idx

    if benign_method == 'synonymous':
        benign_idx = synonymous_idx

    params = fit['fit']['component_params']
    weights = fit['fit']['weights']

    # Accept either a Scoreset object or pre-extracted (scores, sample_assignments) arrays.
    # Passing raw numpy arrays avoids pickling the full Scoreset for every parallel task.
    if sample_assignments is not None:
        scores = scoreset_or_scores
        sa = sample_assignments
    else:
        scores = scoreset_or_scores.scores
        sa = scoreset_or_scores.sample_assignments

    population = scores[sa[:, gnomad_idx]]
    # print(f"population: {len(population)} samples")
    
    pop_density = density_utils.joint_densities(
        population, params, weights[gnomad_idx]
    ).sum(axis=0)
    
    # Compute pathogenic density if available
    pathogenic_density = []
    if pathogenic_idx is not None:
        pathogenic_density = density_utils.joint_densities(
            population, params, weights[pathogenic_idx]
        ).sum(axis=0)
        assert len(pathogenic_density) == len(population)
    
    # Compute benign density if available
    benign_density = []
    if benign_idx is not None and synonymous_idx is not None:
        if benign_method != 'avg':
            benign_density = density_utils.joint_densities(
                population, params, weights[benign_idx]
            ).sum(axis=0)
        else:
            bs_weights = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
            benign_density = density_utils.joint_densities(
                population, params, bs_weights
            ).sum(axis=0)
        assert len(benign_density) == len(population)
    # print(f"benign_density: {benign_density}")
    
    if len(pathogenic_density) != 0 and len(benign_density) != 0:
        mode = 'standard'  # Both labeled classes available
        prior_estimate = 0.5
        # print("standard prior estimation")
    elif len(pathogenic_density) != 0 and len(benign_density) == 0:
        mode = 'positive_unlabeled'  # Only pathogenic available
        prior_estimate = 0.1
        # print("PU prior estimation")
    elif len(pathogenic_density) == 0 and len(benign_density) != 0:
        mode = 'negative_unlabeled'  # Only benign available
        prior_estimate = 0.9
    else:
        raise ValueError("Must have at least one of pathogenic or benign density")

    # default_prior = 0.1
    # if mode == 'negative_unlabeled' or mode == 'positive_unlabeled':
    #     kl_divergence = np.mean(np.abs((benign_density if mode == 'negative_unlabeled' else pathogenic_density) - pop_density) / (pop_density + 1e-10))
    #     if kl_divergence < 0.1:
    #         return default_prior
    
    if mode == 'standard':
        # Standard EM
        converged = False
        em_steps = 0
        max_em_steps = kwargs.get("max_em_steps", 10000)
        tolerance = kwargs.get("tolerance", 1e-6)
        while not converged and em_steps < max_em_steps:
            em_steps += 1
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                posteriors = 1 / (
                    1 + (1 - prior_estimate) / prior_estimate
                    * benign_density / pathogenic_density
                )
            new_prior = np.nanmean(posteriors)
            if abs(new_prior - prior_estimate) < tolerance:
                converged = True
            prior_estimate = new_prior
            if prior_estimate < 0 or prior_estimate > 1:
                break
    else:
        # Mean-matching estimator:
        #   NU: α = 1 - E_{gnomAD}[fb] / E_{benign}[fb]
        #   PU: α = E_{gnomAD}[fp] / E_{path}[fp]
        # Requires only that pathogenic (benign) variants have low benign (path) density.
        # More robust than Blanchard-Recht: does not assume fpop/fbenign ≥ 1-α pointwise,
        # which breaks when fitted model weights don't satisfy the mixture decomposition.
        if mode == 'negative_unlabeled':
            labeled = scores[sa[:, benign_idx]]
            labeled_density = density_utils.joint_densities(
                labeled, params, weights[benign_idx] if benign_method != 'avg'
                else (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
            ).sum(axis=0)
            mean_pop = float(np.nanmean(benign_density))
            mean_lab = float(np.nanmean(labeled_density))
            prior_estimate = float(np.clip(1.0 - mean_pop / mean_lab, 0.001, 0.999)) if mean_lab > 0 else 0.1
        else:  # positive_unlabeled
            labeled = scores[sa[:, pathogenic_idx]]
            labeled_density = density_utils.joint_densities(
                labeled, params, weights[pathogenic_idx]
            ).sum(axis=0)
            mean_pop = float(np.nanmean(pathogenic_density))
            mean_lab = float(np.nanmean(labeled_density))
            prior_estimate = float(np.clip(mean_pop / mean_lab, 0.001, 0.999)) if mean_lab > 0 else 0.1

    if prior_estimate <= 0.001 or prior_estimate >= 0.999:
        return np.nan

    return prior_estimate

def get_bootstrap_score_ranges(fitIdx, fit, fp, fb, score_range, fit_priors, point_values,
                                acmg_mapping_method="tavtigian"):
    fit_xmin, fit_xmax = fit['fit']['xlims']
    mask = (score_range >= fit_xmin) & (score_range <= fit_xmax)# & ((fp > -7.0) | (fb > -7.0)) # add min density check

    # log_fp_local = np.zeros_like(fp)
    # log_fb_local = np.zeros_like(fb)

    # CRITICAL: IGNORE BOOTSTRAPS THAT DON'T SPAN DATA POINT. MARKING 0 WILL CAUSE STRANGE LR+ CURVES AT EXTREMES
    log_fp_local = np.full_like(fp, np.nan, dtype=float)
    log_fb_local = np.full_like(fb, np.nan, dtype=float)

    log_fp_local[mask] = fp[mask]
    log_fb_local[mask] = fb[mask]

    lrP = log_fp_local[mask] - log_fb_local[mask]
    s = score_range[mask]

    if acmg_mapping_method == "continuous":
        from .fit import calculate_classification_ranges
        ranges_p, ranges_b, thresholds = calculate_classification_ranges(
            lrP, lrP, fit_priors[fitIdx], s
        )
        # For continuous, "C" slot carries the LR+ threshold dict (not an int).
        C = thresholds
    else:
        ranges_p, ranges_b, C = calculate_score_ranges(
            lrP, lrP, fit_priors[fitIdx], s, point_values,
            acmg_mapping_method=acmg_mapping_method,
        )
        if C is not None:
            C = int(C)  # tavtigian path returns int; piecewise returns None

    if prior_invalid(fit_priors[fitIdx]):
        log_fp_local = np.full_like(fp, np.nan, dtype=float)
        log_fb_local = np.full_like(fb, np.nan, dtype=float)
        for key in ranges_p:
            ranges_p[key] = []
        for key in ranges_b:
            ranges_b[key] = []
        C = np.nan

    return fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C

def remove_insufficient_bootstrap_converage_points(point_ranges, percent_no_evidence, point_values):

    # P/LP
    for point in point_values:
        if percent_no_evidence[point] > 0.05 and len(point_ranges[point]) > 0:
            if point > 1 : # extend range below
                i = 1
                while point-i != 0:
                    if len(point_ranges[point-i]) > 0:
                        new_range = np.vstack([point_ranges[point-i], point_ranges[point]])[
                                                np.vstack([point_ranges[point-i], point_ranges[point]])[:, 0].argsort()]
                        point_ranges[point-i] = new_range
                        break
                    i += 1
                
            point_ranges[point] = [] # remove strength

    # B/LB
    for point_p in point_values:
        point = -point_p 
        if percent_no_evidence[point] > 0.05 and len(point_ranges[point]) > 0:
            if point < -1 : # extend range below
                i = 1
                while point+i != 0:
                    if len(point_ranges[point+i]) > 0:
                        new_range = np.vstack([point_ranges[point+i], point_ranges[point]])[
                                                np.vstack([point_ranges[point+i], point_ranges[point]])[:, 0].argsort()]
                        point_ranges[point+i] = new_range
                        break
                    i += 1
                
            point_ranges[point] = [] # remove strength



def check_thresholds_reached(lrPlus, tau, point_values, pathogenicOrBenign):
    
    if pathogenicOrBenign == "benign":
        point_values = -1 * np.array(point_values)
    
    reached = {}
    
    for p in point_values:
        if pathogenicOrBenign == "pathogenic":
            # Check if LR+ ever exceeds threshold
            reached[p] = np.any(lrPlus >= tau[abs(p)-1]) # list idx not dict
        else:
            # Check if LR+ ever goes below threshold
            reached[p] = np.any(lrPlus <= tau[abs(p)-1]) # list idx not dict
    
    return reached



def compute_single_fit_log_densities(fit, prior, score_range, benign_method,
                                     pathogenic_idx=0, benign_idx=1, 
                                     gnomad_idx=2, synonymous_idx=3,
                                     log_density_threshold=-7.0):
    """
    Compute log pathogenic and benign densities for a single fit.
    
    When both densities fall below log_density_threshold, sets both to the 
    threshold value so that log_fp - log_fb ≈ 0 (LR+ ≈ 1, neutral evidence).
    
    [Keep all existing parameters]
    
    log_density_threshold : float, optional
        Threshold below which both densities are set equal to avoid 
        spurious evidence from dividing tiny numbers. Default: -7.0
    
    Returns
    -------
    log_fp : np.ndarray or None
        Log pathogenic density (None if prior invalid)
    log_fb : np.ndarray or None
        Log benign density (None if prior invalid)
    """
    # Skip if prior estimation failed
    if np.isnan(prior) or prior <= 0 or prior >= 1:
        return None, None
    
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    
    # Get population density (always available)
    log_pop = density_utils.mixture_pdf(score_range, params, weights[gnomad_idx])
    pop_linear = np.exp(log_pop)
    
    have_pathogenic = pathogenic_idx is not None
    have_benign = (benign_idx is not None) or (synonymous_idx is not None)
    
    if not have_pathogenic and not have_benign:
        raise ValueError("Must have at least one of pathogenic or benign sample")
    
    if have_pathogenic:
        log_fp = density_utils.mixture_pdf(score_range, params, weights[pathogenic_idx])
    else:
        # Get effective benign weights
        if benign_method == 'synonymous' and synonymous_idx is not None:
            w_benign_eff = weights[synonymous_idx]
        elif benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
            w_benign_eff = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
        else:
            w_benign_eff = weights[synonymous_idx if benign_idx is None else benign_idx]
        
        log_fb_temp = density_utils.mixture_pdf(score_range, params, w_benign_eff)
        fb_linear = np.exp(log_fb_temp)
        
        # Unmix: f_p = [f_pop - (1-alpha)*f_b] / alpha
        fp_linear = (pop_linear - (1 - prior) * fb_linear) / prior
        
        # Clip negative values (numerical issues)
        fp_linear = np.maximum(fp_linear, pop_linear * 1e-10)  # At least 1e-10 of population
        log_fp = np.log(fp_linear)
    
    if have_benign:
        # Get effective benign weights
        if benign_method == "synonymous" and synonymous_idx is not None:
            w_benign_eff = weights[synonymous_idx]
        elif benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
            w_benign_eff = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
        else:
            w_benign_eff = weights[synonymous_idx if benign_idx is None else benign_idx]
        
        log_fb = density_utils.mixture_pdf(score_range, params, w_benign_eff)
    else:
        fp_linear = np.exp(log_fp)
        
        # Unmix: f_b = [f_pop - alpha*f_p] / (1-alpha)
        fb_linear = (pop_linear - prior * fp_linear) / (1 - prior)
        
        # Clip negative values
        fb_linear = np.maximum(fb_linear, pop_linear * 1e-10)  # At least 1e-10 of population
        log_fb = np.log(fb_linear)
    
    # # WHERE BOTH DENSITIES ARE VERY LOW, SET THEM EQUAL
    # # This ensures log_fp - log_fb ≈ 0
    # low_density_mask = np.logical_and(
    #     log_fp < log_density_threshold,
    #     log_fb < log_density_threshold
    # )
    
    # # Set both to the threshold value where both are low
    # log_fp = np.where(low_density_mask, log_density_threshold, log_fp)
    # log_fb = np.where(low_density_mask, log_density_threshold, log_fb)
    
    return log_fp, log_fb


def get_variant_oob_bootstrap_indices(scoreset, dataset_splits, valid_mask):
    """
    For each variant in scoreset, find which bootstrap iterations (filtered indices) 
    it appears in the validation set.
    
    Parameters:
    -----------
    valid_mask : np.ndarray
        Boolean mask of valid bootstrap iterations (from fit_priors filtering)
    
    Returns:
    --------
    variant_to_oob_boots : dict
        Maps scoreset index -> list of FILTERED bootstrap indices
    """
    
    # Create mapping: original boot_idx -> filtered boot_idx
    original_to_filtered = {}
    filtered_idx = 0
    for original_idx in range(len(valid_mask)):
        if valid_mask[original_idx]:
            original_to_filtered[original_idx] = filtered_idx
            filtered_idx += 1
    
    # Build index mapping for fast lookup
    score_class_to_indices = defaultdict(list)
    
    for idx in range(len(scoreset.scores)):
        score = scoreset.scores[idx]
        class_indices = np.where(scoreset.sample_assignments[idx])[0]
        
        for class_idx in class_indices:
            key = (score, class_idx)
            score_class_to_indices[key].append(idx)
    
    # Map each scoreset index to its OOB bootstrap iterations
    variant_to_oob_boots = defaultdict(list)
    
    print(f"Building OOB mapping for {len(scoreset.scores)} variants...")
    print(f"  Valid bootstraps: {sum(valid_mask)}/{len(valid_mask)}")
    
    for boot_idx in sorted(dataset_splits.keys()):
        # Skip if this bootstrap was filtered out
        if not valid_mask[boot_idx]:
            continue
        
        # Get filtered index
        filtered_boot_idx = original_to_filtered[boot_idx]
        
        val_obs = dataset_splits[boot_idx]["val_observations"]
        val_assign = dataset_splits[boot_idx]["val_sample_assignments"]
        
        for obs, assign in zip(val_obs, val_assign):
            class_idx = np.where(assign)[0][0]
            key = (obs, class_idx)
            matching_indices = score_class_to_indices.get(key, [])
            
            # Add FILTERED bootstrap index to all matching variants
            for variant_idx in matching_indices:
                variant_to_oob_boots[variant_idx].append(filtered_boot_idx)
    
    print(f"Found OOB samples for {len(variant_to_oob_boots)}/{len(scoreset.scores)} variants")
    
    return variant_to_oob_boots

import pandas as pd

def make_variant_id(v):
    return f"{v.ID}_{v.Gene}_{v.Chrom}_{v.hgvs_c}"

def flatten_point_ranges(point_ranges):
    """Flatten 2D arrays in point_ranges to 1D, asserting only one or zero arrays."""
    flattened = {}
    for key, ranges in point_ranges.items():
        if len(ranges) == 0:
            flattened[key] = []
        elif len(ranges) == 1:
            assert len(ranges[0]) == 2, f"Expected 2 values in range for key {key}, got {ranges[0]}"
            flattened[key] = ranges[0]
        else:
            raise AssertionError(f"Expected 0 or 1 range for key {key}, got {len(ranges)}")
    return flattened

def assign_points(assay_score, point_ranges):
    """Assign points based on which range the assay_score falls into."""
    if assay_score is None or pd.isna(assay_score):
        return None
    
    matched_points = []
    for point_str, range_vals in point_ranges.items():
        if len(range_vals) == 2:
            low, high = range_vals
            if low <= assay_score <= high:
                matched_points.append(int(point_str))
    
    assert len(matched_points) <= 1, f"Score {assay_score} matched multiple ranges: {matched_points}, point ranges: {point_ranges}"
    return matched_points[0] if matched_points else 0




def plot_oob_variant_calibrations(dataset, scoreset, variant_to_oob_points, 
                                   variant_to_oob_boots, fit_priors, log_lr_plus,
                                   score_range, point_values, scoreset_flipped,
                                   save_dir, n_variants_to_plot=20):
    """
    Plot OOB calibrations for individual variants to diagnose issues.
    
    For each variant, shows:
    - OOB LR+ curves (5th, median, 95th percentiles)
    - Computed point ranges
    - Thresholds
    - Where variant's score falls
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get variant IDs
    variants_by_id = scoreset.get_variants_by_id()
    
    # Select variants to plot (mix of P/LP and B/LB)
    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]
    
    plp_indices = np.where(plp_mask)[0][:n_variants_to_plot//2]
    blb_indices = np.where(blb_mask)[0][:n_variants_to_plot//2]
    
    variants_to_plot = list(plp_indices) + list(blb_indices)
    
    for variant_idx in variants_to_plot:
        if variant_idx not in variant_to_oob_boots:
            continue
        
        variant_list = list(variants_by_id.values())[variant_idx]
        variant_id = make_variant_id(variant_list[0])
        variant_score = scoreset.scores[variant_idx]
        
        # Get OOB data
        oob_boot_indices = variant_to_oob_boots[variant_idx]
        oob_result = variant_to_oob_points.get(variant_id, None)
        
        if oob_result is None:
            continue
        
        # Subset to OOB
        oob_priors = fit_priors[oob_boot_indices]
        oob_log_lr = log_lr_plus[oob_boot_indices, :]
        
        # Remove invalid
        valid_mask = ~np.isnan(oob_priors) & (oob_priors > 0) & (oob_priors < 1)
        oob_priors = oob_priors[valid_mask]
        oob_log_lr = oob_log_lr[valid_mask]
        
        if len(oob_priors) == 0:
            continue
        
        oob_prior = np.nanmedian(oob_priors)
        
        # Compute percentiles
        lr_5th = np.nanpercentile(oob_log_lr, 5, axis=0)
        lr_median = np.nanpercentile(oob_log_lr, 50, axis=0)
        lr_95th = np.nanpercentile(oob_log_lr, 95, axis=0)
        
        # Get thresholds
        tauP, tauB, _ = thresholds_from_prior(oob_prior, point_values)
        tauP_log = np.log(tauP)
        tauB_log = np.log(tauB)
        
        # Create figure
        fig, (ax_lr, ax_points) = plt.subplots(2, 1, figsize=(12, 10), 
                                                gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot LR+ curves
        ax_lr.plot(score_range, lr_5th, color='red', label='5th percentile', linewidth=2)
        ax_lr.plot(score_range, lr_median, color='black', label='Median', linewidth=2)
        ax_lr.plot(score_range, lr_95th, color='blue', label='95th percentile', linewidth=2)
        
        # Plot thresholds
        for i, (tau_p, tau_b) in enumerate(zip(tauP_log[:-1], tauB_log[:-1])):
            point_val = point_values[i]
            ax_lr.axhline(tau_p, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax_lr.axhline(tau_b, color='blue', linestyle='--', alpha=0.3, linewidth=1)
            ax_lr.text(score_range[-1], tau_p, f'+{point_val}', 
                      fontsize=8, ha='right', va='bottom', color='red')
            ax_lr.text(score_range[-1], tau_b, f'{-point_val}', 
                      fontsize=8, ha='right', va='top', color='blue')
        
        # Mark variant's score
        ax_lr.axvline(variant_score, color='green', linestyle='-', linewidth=2, 
                     label=f'Variant score ({variant_score:.3f})', alpha=0.7)
        
        # Determine ground truth
        is_plp = plp_mask[variant_idx]
        is_blb = blb_mask[variant_idx]
        ground_truth = 'P/LP' if is_plp else ('B/LB' if is_blb else 'Unknown')
        assigned_points = oob_result['points']
        
        ax_lr.set_xlabel('Score', fontsize=12)
        ax_lr.set_ylabel('Log LR+', fontsize=12)
        ax_lr.set_title(f'Variant {variant_idx}: {variant_id}\n'
                       f'Ground Truth: {ground_truth}, OOB Assigned: {assigned_points:+d} points\n'
                       f'OOB Prior: {oob_prior:.4f}, OOB Bootstraps: {len(oob_boot_indices)} ({len(oob_priors)} valid)',
                       fontsize=13, fontweight='bold')
        ax_lr.legend(fontsize=10)
        ax_lr.grid(True, alpha=0.3)
        ax_lr.set_ylim([tauB_log[-1]-1, tauP_log[-1]+1])
        
        # Plot point assignments (try to reconstruct from result)
        # We'd need to store point_ranges in the result to show this
        # For now, just show which point was assigned
        ax_points.axvline(variant_score, color='green', linestyle='-', linewidth=2, alpha=0.7)
        ax_points.text(variant_score, 0.5, f'Assigned: {assigned_points:+d}', 
                      ha='center', va='center', fontsize=12, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax_points.set_xlabel('Score', fontsize=12)
        ax_points.set_ylabel('Point Assignment', fontsize=12)
        ax_points.set_ylim([0, 1])
        ax_points.set_yticks([])
        ax_points.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset}_variant_{variant_idx}_oob.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(variants_to_plot)} OOB variant plots to {save_dir}")


def log_oob_variant_details(dataset, scoreset, variant_to_oob_points, 
                             variant_to_oob_boots, fit_priors, log_lr_plus,
                             score_range, point_values, log_filepath):
    """
    Log detailed OOB information for all variants to a file.
    """
    
    variants_by_id = scoreset.get_variants_by_id()
    
    # BUILD CORRECT MAPPING (same as OOB generation)
    kept_idx_to_variant_id = {}
    kept_idx = 0
    
    for all_idx, (variant_id, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            kept_idx_to_variant_id[kept_idx] = make_variant_id(variants[0])
            kept_idx += 1
    
    with open(log_filepath, 'w') as f:
        f.write(f"{'='*100}\n")
        f.write(f"OOB VARIANT CALIBRATION LOG: {dataset}\n")
        f.write(f"{'='*100}\n\n")
        
        plp_mask = scoreset.sample_assignments[:, 0]
        blb_mask = scoreset.sample_assignments[:, 1]
        
        for variant_idx in range(len(scoreset.scores)):
            if variant_idx not in variant_to_oob_boots:
                continue
            
            # USE CORRECT MAPPING
            variant_id = kept_idx_to_variant_id[variant_idx]
            variant_score = scoreset.scores[variant_idx]
            
            oob_boot_indices = variant_to_oob_boots[variant_idx]
            oob_result = variant_to_oob_points.get(variant_id, None)
            
            # Ground truth
            is_plp = plp_mask[variant_idx]
            is_blb = blb_mask[variant_idx]
            ground_truth = 'P/LP' if is_plp else ('B/LB' if is_blb else 'Unknown')

            if not is_plp and not is_blb:
                continue
            
            f.write(f"\n{'-'*100}\n")
            f.write(f"Variant {variant_idx}: {variant_id}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"  Score: {variant_score:.6f}\n")
            f.write(f"  Ground Truth: {ground_truth}\n")
            f.write(f"  OOB Bootstraps: {len(oob_boot_indices)}\n")
            
            if oob_result is None:
                f.write(f"  STATUS: FAILED (no OOB result)\n")
                continue
            
            # Summary
            correct = (is_plp and oob_result['points'] > 0) or (is_blb and oob_result['points'] < 0)
            wrong = (is_plp and oob_result['points'] < 0) or (is_blb and oob_result['points'] > 0)
            ir = oob_result['points'] == 0
            
            f.write(f"\n  RESULT: {'✓ CORRECT' if correct else ('✗ WRONG' if wrong else '- IR')}\n")


def create_oob_summary_plot(dataset, scoreset, variant_to_oob_boots, fit_priors, 
                             log_lr_plus, score_range, point_values, scoreset_flipped,
                             save_path):
    """
    Create one summary plot showing OOB LR+ curves for many variants overlaid.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]
    
    # Sample 50 P/LP and 50 B/LB variants
    plp_indices = np.where(plp_mask)[0][:50]
    blb_indices = np.where(blb_mask)[0][:50]
    
    # Compute global prior for thresholds
    global_prior = np.nanmedian(fit_priors)
    tauP, tauB, _ = thresholds_from_prior(global_prior, point_values)
    tauP_log = np.log(tauP)
    tauB_log = np.log(tauB)
    
    # Plot P/LP variants
    for variant_idx in plp_indices:
        if variant_idx not in variant_to_oob_boots:
            continue
        
        oob_boot_indices = variant_to_oob_boots[variant_idx]
        oob_log_lr = log_lr_plus[oob_boot_indices, :]
        
        # Remove invalid
        oob_priors = fit_priors[oob_boot_indices]
        valid_mask = ~np.isnan(oob_priors) & (oob_priors > 0) & (oob_priors < 1)
        oob_log_lr = oob_log_lr[valid_mask]
        
        if len(oob_log_lr) < 10:
            continue
        
        lr_5th = np.nanpercentile(oob_log_lr, 5, axis=0)
        
        ax1.plot(score_range, lr_5th, color='red', alpha=0.2, linewidth=0.5)
    
    # Plot B/LB variants
    for variant_idx in blb_indices:
        if variant_idx not in variant_to_oob_boots:
            continue
        
        oob_boot_indices = variant_to_oob_boots[variant_idx]
        oob_log_lr = log_lr_plus[oob_boot_indices, :]
        
        # Remove invalid
        oob_priors = fit_priors[oob_boot_indices]
        valid_mask = ~np.isnan(oob_priors) & (oob_priors > 0) & (oob_priors < 1)
        oob_log_lr = oob_log_lr[valid_mask]
        
        if len(oob_log_lr) < 10:
            continue
        
        lr_95th = np.nanpercentile(oob_log_lr, 95, axis=0)
        
        ax2.plot(score_range, lr_95th, color='blue', alpha=0.2, linewidth=0.5)
    
    # Add thresholds to both plots
    for ax, title, color in [(ax1, 'P/LP Variants (5th percentile)', 'red'),
                              (ax2, 'B/LB Variants (95th percentile)', 'blue')]:
        for i, pv in enumerate(point_values):
            ax.axhline(tauP_log[i], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(tauB_log[i], color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(score_range[-1], tauP_log[i], f'+{pv}', 
                   fontsize=9, ha='right', va='bottom', color='red', fontweight='bold')
            ax.text(score_range[-1], tauB_log[i], f'{-pv}', 
                   fontsize=9, ha='right', va='top', color='blue', fontweight='bold')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Log LR+', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([tauB_log[-1]-2, tauP_log[-1]+2])
    
    plt.suptitle(f'{dataset}: OOB LR+ Curves (n={len(plp_indices)} P/LP, {len(blb_indices)} B/LB)\n'
                f'Global Prior: {global_prior:.4f}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved OOB summary plot to {save_path}")





def _process_single_variant_oob_full(variant_idx, oob_boot_indices, variant_score, 
                                      fit_priors, log_fp_all, log_fb_all, score_range,
                                      point_values, scoreset_flipped, scoreset_name,
                                      min_oob_samples=10, log_density_threshold=-7.0):
    """
    Process a single variant using EXACT in-bag logic but with OOB bootstraps only.
    """
    
    # Skip if too few OOB samples
    if len(oob_boot_indices) < min_oob_samples:
        return variant_idx, None
    
    # Subset to OOB bootstraps
    oob_priors = fit_priors[oob_boot_indices]
    oob_log_fp = log_fp_all[oob_boot_indices, :]
    oob_log_fb = log_fb_all[oob_boot_indices, :]
    
    # Remove invalid priors
    nan_mask = np.isnan(oob_priors)
    invalid_range_mask = (oob_priors <= 0) | (oob_priors >= 1)
    valid_oob_mask = ~(nan_mask | invalid_range_mask)
    
    oob_priors = oob_priors[valid_oob_mask]
    oob_log_fp = oob_log_fp[valid_oob_mask]
    oob_log_fb = oob_log_fb[valid_oob_mask]
    
    if len(oob_priors) < min_oob_samples:
        return variant_idx, None
    
    # Compute OOB median prior
    oob_prior = np.nanmedian(oob_priors)
    
    if oob_prior <= 0 or oob_prior >= 1:
        return variant_idx, None
    
    # Compute OOB LR+
    oob_log_lr_plus = oob_log_fp - oob_log_fb
    
    # Filter score range
    nan_counts = np.isnan(oob_log_lr_plus).sum(0)
    range_subset = nan_counts < oob_log_lr_plus.shape[0]
    
    if not np.any(range_subset):
        return variant_idx, None
    
    # Apply subset
    valid_score_range = score_range[range_subset]
    valid_oob_lr_plus = oob_log_lr_plus[:, range_subset]
    
    # Calculate score ranges
    try:
        point_ranges_pathogenic, point_ranges_benign, C = calculate_score_ranges(
            np.nanpercentile(valid_oob_lr_plus, 5, axis=0),
            np.nanpercentile(valid_oob_lr_plus, 95, axis=0),
            oob_prior,
            valid_score_range,
            point_values
        )
        
        point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
        
        # Check if prior is valid (EXACT in-bag check in median_prior branch)
        if oob_prior <= 0 or oob_prior >= 1:
            for point in point_ranges:
                point_ranges[point] = []
        
        # DON'T WRAP! calculate_score_ranges returns [[low, high]] already
        # The in-bag median_prior branch does NOT wrap here
        
        # Apply hard-coded liberal setting
        liberal = False if scoreset_name in [
            "GCK_Gersing_2023_complementation", 
            "DDX3X_Radford_2023_cLFC_day15", 
            "DDX3X_Radford_2023_cLFC_day15_clinvar_2018", 
            "F9_Popp_2025_heavy_chain"
        ] else True
        
        # Enforce monotonicity (first pass)
        enforce_monotonicity_point_ranges(
            point_ranges, 
            point_values, 
            valid_score_range, 
            scoreset_flipped=scoreset_flipped,
            liberal=liberal,
            log_f=None
        )
        
        # Extend to xlims
        extend_points_to_xlims(point_ranges, point_values, valid_score_range, scoreset_flipped, log_f=None)
        
        # Enforce monotonicity again (second pass)
        enforce_monotonicity_point_ranges(
            point_ranges, 
            point_values, 
            valid_score_range, 
            scoreset_flipped=scoreset_flipped,
            liberal=liberal,
            log_f=None
        )
        
        # Flatten for assignment
        # flatten_point_ranges expects {1: [[low, high]], 2: []} format
        flattened_point_ranges = flatten_point_ranges(point_ranges)
        
        # Assign points to this variant's score
        assigned_points = assign_points(variant_score, flattened_point_ranges)
        
    except NotImplementedError as e:
        if variant_idx < 5:
            print(f"  Variant {variant_idx} FAILED: {type(e).__name__}: {str(e)[:200]}")
        return variant_idx, None
    
    result = {
        'points': assigned_points,
        'n_oob': len(oob_boot_indices),
        'n_oob_valid': len(oob_priors),
        'oob_prior': oob_prior,
        'score': variant_score,
    }
    
    return variant_idx, result

def _process_single_variant_oob_simple(variant_idx, oob_boot_indices, variant_score, 
                                        fit_priors, log_fp_all, log_fb_all, score_range,
                                        point_values, scoreset_flipped, scoreset_name,
                                        min_oob_samples=10, log_density_threshold=-7.0,
                                        debug=True):
    """
    OOB processing with MINIMAL post-processing - just low-density filtering.
    No enforce_monotonicity, no extend_to_xlims.
    """
    
    # Skip if too few OOB samples
    if len(oob_boot_indices) < min_oob_samples:
        if debug:
            print(f"  Variant {variant_idx} FAIL: Only {len(oob_boot_indices)} OOB samples")
        return variant_idx, None
    
    # Subset to OOB bootstraps
    oob_priors = fit_priors[oob_boot_indices]
    oob_log_fp = log_fp_all[oob_boot_indices, :]
    oob_log_fb = log_fb_all[oob_boot_indices, :]
    
    # Remove invalid priors
    nan_mask = np.isnan(oob_priors)
    invalid_range_mask = (oob_priors <= 0) | (oob_priors >= 1)
    valid_oob_mask = ~(nan_mask | invalid_range_mask)
    
    if debug and variant_idx < 100:
        print(f"  Variant {variant_idx}: {len(oob_boot_indices)} OOB -> {valid_oob_mask.sum()} valid priors")
    
    oob_priors = oob_priors[valid_oob_mask]
    oob_log_fp = oob_log_fp[valid_oob_mask]
    oob_log_fb = oob_log_fb[valid_oob_mask]
    
    if len(oob_priors) < min_oob_samples:
        if debug and variant_idx < 100:
            print(f"  Variant {variant_idx} FAIL: Only {len(oob_priors)} valid priors after filtering")
        return variant_idx, None
    
    # Compute OOB median prior
    oob_prior = np.nanmedian(oob_priors)
    
    if oob_prior <= 0 or oob_prior >= 1:
        if debug and variant_idx < 100:
            print(f"  Variant {variant_idx} FAIL: Invalid OOB prior {oob_prior}")
        return variant_idx, None
    
    # Compute OOB LR+
    oob_log_lr_plus = oob_log_fp - oob_log_fb
    
    # Find closest score in score_range
    score_idx = np.argmin(np.abs(score_range - variant_score))
    
    # Get OOB LR+ distribution at this variant's score
    oob_lr_at_score = oob_log_lr_plus[:, score_idx]
    
    # Check if all NaN
    if np.all(np.isnan(oob_lr_at_score)):
        if debug and variant_idx < 100:
            print(f"  Variant {variant_idx} FAIL: All NaN at score {variant_score}")
        return variant_idx, None
    
    # Conservative percentiles
    oob_lr_5th = np.nanpercentile(oob_lr_at_score, 5)
    oob_lr_95th = np.nanpercentile(oob_lr_at_score, 95)
    
    # Compute thresholds
    tau_p, tau_b, _ = thresholds_from_prior(oob_prior, point_values)
    tau_p_log = np.log(tau_p)
    tau_b_log = np.log(tau_b)
    
    # Assign pathogenic points (using 5th percentile)
    pathogenic_points = 0
    for point in reversed(point_values):
        if oob_lr_5th >= tau_p_log[point - 1]:
            pathogenic_points = point
            break
    
    # Assign benign points (using 95th percentile)
    benign_points = 0
    for point in reversed(point_values):
        if oob_lr_95th <= tau_b_log[point - 1]:
            benign_points = -point
            break
    
    # Final assignment
    if pathogenic_points != 0 and benign_points != 0:
        # Shouldn't happen but handle gracefully
        assigned_points = pathogenic_points if abs(pathogenic_points) > abs(benign_points) else benign_points
    elif pathogenic_points != 0:
        assigned_points = pathogenic_points
    elif benign_points != 0:
        assigned_points = benign_points
    else:
        assigned_points = 0
    
    result = {
        'points': assigned_points,
        'n_oob': len(oob_boot_indices),
        'n_oob_valid': len(oob_priors),
        'oob_prior': oob_prior,
        'score': variant_score,
    }
    
    return variant_idx, result


def compute_oob_variant_evidence_with_full_processing(
    dataset, fits, scoreset, dataset_to_splits, 
    fit_priors, valid_mask, log_fp_all, log_fb_all, score_range, 
    point_values, benign_method, n_c,
    scoreset_flipped=False, min_oob_samples=10,
    n_jobs=-1
):
    """
    Compute OOB evidence using EXACT in-bag processing for each variant.
    
    Each variant gets its own "mini in-bag calibration" using only its OOB bootstraps,
    with ALL post-processing steps identical to in-bag.
    """
    
    if dataset not in dataset_to_splits:
        raise ValueError(f"Dataset {dataset} not found in splits")
    
    dataset_splits = dataset_to_splits[dataset]
    
    # Get OOB mapping
    variant_to_oob_boots = get_variant_oob_bootstrap_indices(scoreset, dataset_splits, valid_mask)
    
    print(f"Processing {len(variant_to_oob_boots)} variants with full OOB processing...")
    
    # Get scoreset name for hard-coded logic
    scoreset_name = scoreset.scoreset_name

    lib_datasets = [
        "GCK_Gersing_2023_complementation", 
        "DDX3X_Radford_2023_cLFC_day15", 
        "DDX3X_Radford_2023_cLFC_day15_clinvar_2018", 
        "F9_Popp_2025_heavy_chain"
    ]
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        # delayed(_process_single_variant_oob_simple if dataset not in lib_datasets else _process_single_variant_oob_full)(
        delayed(_process_single_variant_oob_full)(
            variant_idx, oob_boot_indices, scoreset.scores[variant_idx],
            fit_priors, log_fp_all, log_fb_all, score_range,
            point_values, scoreset_flipped, scoreset_name, min_oob_samples
        )
        for variant_idx, oob_boot_indices in variant_to_oob_boots.items()
    )
    
    # BUILD CORRECT INDEX MAPPING
    variants_by_id = scoreset.get_variants_by_id()
    kept_idx_to_variant_id = {}
    kept_idx = 0
    
    for all_idx, (variant_id, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            kept_idx_to_variant_id[kept_idx] = make_variant_id(variants[0])
            kept_idx += 1
    
    # Convert to variant ID keys using CORRECT mapping
    variant_to_oob_points = {}
    
    for variant_idx, result in results:
        if result is not None:
            variant_id = kept_idx_to_variant_id[variant_idx]  # Use correct mapping!
            variant_to_oob_points[variant_id] = result
    
    print(f"\nOOB evidence computed for {len(variant_to_oob_points)} variants")
    if len(variant_to_oob_points) > 0:
        print(f"  Mean OOB samples per variant: {np.mean([v['n_oob'] for v in variant_to_oob_points.values()]):.1f}")
        
        assigned_points_dist = [v['points'] for v in variant_to_oob_points.values()]
        from collections import Counter
        point_counts = Counter(assigned_points_dist)
        print(f"  Point distribution: {dict(sorted(point_counts.items()))}")

    # # Create log file
    log_filepath = f'/data/ross/assay_calibration/oob_logs/{dataset}_oob_details.log'
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    log_lr_plus = log_fp_all - log_fb_all
    
    log_oob_variant_details(
        dataset, scoreset, variant_to_oob_points, 
        variant_to_oob_boots, fit_priors, log_lr_plus,
        score_range, point_values, log_filepath
    )
    
    # # Create plots
    # plot_dir = f'/data/ross/assay_calibration/oob_variant_plots/{dataset}'
    # plot_oob_variant_calibrations(
    #     dataset, scoreset, variant_to_oob_points,
    #     variant_to_oob_boots, fit_priors, log_lr_plus,
    #     score_range, point_values, scoreset_flipped,
    #     plot_dir, n_variants_to_plot=20
    # )
    
    # # Create summary plot
    # summary_path = f'/data/ross/assay_calibration/oob_summaries/{dataset}_oob_summary.png'
    # os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    # create_oob_summary_plot(
    #     dataset, scoreset, variant_to_oob_boots, fit_priors,
    #     log_lr_plus, score_range, point_values, scoreset_flipped,
    #     summary_path
    # )
    
    return variant_to_oob_points