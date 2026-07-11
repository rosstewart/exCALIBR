import sys
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging


os.environ["MPLLOGLEVEL"] = "WARNING"

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
logging.getLogger("joblib").setLevel(logging.ERROR)
logging.getLogger("loky").setLevel(logging.ERROR)
logging.getLogger("adjustText").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns
from ..fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from ..fit_utils.cfusn import density_utils  # noqa: E402
from ..data_utils.dataset import Scoreset  # noqa: E402
from ..fit_utils.utils import serialize_dict  # noqa: E402
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text  # pip install adjustText

def plot_scoreset(scoreset:Scoreset, summary: Dict, scoreset_fits: List[Dict], score_range, use_median_prior,use_2c_equation, n_c, benign_method, C):
    fig, ax = plt.subplots(2,scoreset.n_samples, figsize=(5*scoreset.n_samples,10),sharex=True,sharey=False)
    for sample_num in range(scoreset.n_samples):
        sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]],stat='density',ax=ax[1,sample_num],alpha=.5,color='pink',)
        density = sample_density(score_range, scoreset_fits, sample_num)
        for compNum in range(density.shape[1]):

            compDensity = density[:,compNum,:]
            d = np.nanpercentile(compDensity,[5,50,95],axis=0)
            ax[1,sample_num].plot(score_range,d[1],color=f"C{compNum}",linestyle='--',label=f"Comp {compNum+1}")
            ax[1,sample_num].legend()
        d = np.nansum(density,axis=1)
        d_perc = np.percentile(d,[5,50,95],axis=0)
        ax[1,sample_num].plot(score_range,d_perc[1],color='black',alpha=.5)
        ax[1,sample_num].fill_between(score_range,d_perc[0],d_perc[2],color='gray',alpha=0.3)
        ax[1,sample_num].set_xlabel("Score")
        ax[0,sample_num].set_title(f"{scoreset.sample_names[sample_num]} (n={scoreset.sample_assignments[:,sample_num].sum():,d})")
    point_ranges = sorted([(int(k), v) for k,v in summary['point_ranges'].items()])
    point_values = [pr[0] for pr in point_ranges]
    print(point_ranges)
    for axi in ax[0]:
        for pointIdx,(pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                axi.plot([sr[0], sr[1]], [pointIdx,pointIdx], color='red' if pointVal > 0 else 'blue', linestyle='-', alpha=0.7)
        axi.set_ylim(-1,len(point_values))
        axi.set_ylabel("Points")

        axi.set_yticks(range(len(point_values)),labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0",point_values)))
    ax[0,2].set_title(f"{scoreset.scoreset_name} ({n_c}, median:{use_median_prior},em:{not use_2c_equation}): (gnomAD pop, n={scoreset.sample_assignments[:,2].sum():,d})\nprior {summary['prior']:.3f}, C: {summary['C']}")
    return fig

def plot_scoreset_compare_point_assignments(dataset, scoresets, summary, scoreset_fits, score_ranges, n_samples, n_c_auto_selected):
    
    # Determine which scoreset types exist
    scoreset_keys = list(scoresets.keys())
    has_2c = any('2c' in k for k in scoreset_keys)
    has_3c = any('3c' in k for k in scoreset_keys)
    has_4c = any('4c' in k for k in scoreset_keys)
    
    # Build list of scoreset types in order
    scoreset_types = []
    if has_4c:
        scoreset_types.append('4c')
    if has_3c:
        scoreset_types.append('3c')
    if has_2c:
        scoreset_types.append('2c')
    
    n_rows = len(scoreset_types) * 2  # 2 rows per scoreset type (LR+ and fits/points)
    
    # Get scoresets and configs for each type
    scoreset_data = {}
    for st in scoreset_types:
        # Find the scoreset with this type
        st_key = [k for k in scoreset_keys if st in k][0]
        scoreset_data[st] = {
            'scoreset': scoresets[st_key],
            'score_range': score_ranges[st_key],
            'configs': sorted([k for k in summary.keys() if k[1] == st and 'avg' not in k]) + \
                      sorted([k for k in summary.keys() if k[1] == st and 'avg' in k]),
            'fits_key': st_key
        }
    
    # Determine layout dimensions
    max_samples = max(sd['scoreset'].n_samples for sd in scoreset_data.values())
    max_configs = max(len(sd['configs']) for sd in scoreset_data.values())
    n_cols_total = max_samples + max_configs
    
    fig, ax = plt.subplots(n_rows, n_cols_total, figsize=(10*n_cols_total, 10*n_rows), 
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    # Process each scoreset type
    for type_idx, st in enumerate(scoreset_types):
        row_lr = type_idx * 2      # LR+ row
        row_fits = type_idx * 2 + 1  # Fits/points row
        
        sd = scoreset_data[st]
        scoreset = sd['scoreset']
        score_range = sd['score_range']
        configs = sd['configs']
        n_samples_st = scoreset.n_samples
        
        # ===== LR+ row: Hide sample columns =====
        for col_idx in range(max_samples):
            ax[row_lr, col_idx].axis('off')
        
        # ===== Fits/points row: Plot fits =====
        num_skipped = 0
        for sample_num in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            ax_fit = ax[row_fits, sample_num-num_skipped]
            
            # print(scoreset.sample_names)
            # Plot histogram
            hist_data = sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num-num_skipped]], 
                                     stat='density', ax=ax_fit, alpha=.5, color='pink')
            
            # Get maximum density from histogram patches
            max_hist_density = max([patch.get_height() for patch in ax_fit.patches])
            
            # Plot fitted densities
            density = sample_density(score_range, scoreset_fits[sd['fits_key']], sample_num-num_skipped)
            for compNum in range(density.shape[1]):
                compDensity = density[:,compNum,:]
                d = np.nanpercentile(compDensity,[5,50,95],axis=0)
                ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
            ax_fit.legend(fontsize=8)
            
            d = np.nansum(density, axis=1)
            d_perc = np.percentile(d, [5,50,95], axis=0)
            ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
            ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
            ax_fit.set_title(f"{st}: {scoreset.sample_names[sample_num]}\n(n={scoreset.sample_assignments[:,sample_num-num_skipped].sum():,d})")
            ax_fit.set_xlabel("Score")
            ax_fit.set_ylabel("Density")
            ax_fit.set_ylim([0, max_hist_density * 1.1])
            ax_fit.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused sample columns
        for col_idx in range(n_samples_st, max_samples):
            ax[row_fits, col_idx].axis('off')
        
        # Get x-limits from fits
        xlim = ax[row_fits, 0].get_xlim()
        
        # ===== LR+ row: Plot LR+ summaries =====
        for config_idx, (config, n_c) in enumerate(configs):
            col_idx = max_samples + config_idx
            ax_lr = ax[row_lr, col_idx]
            
            log_lr_plus = summary[(config, n_c)]['log_lr_plus']
            llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
            labels = ['5th percentile','Median','95th percentile']
            
            for i, c in enumerate(['red','black','blue']):
                ax_lr.plot(score_range, llr_curves[i], color=c, label=labels[i])
            
            point_values = sorted(list(set([abs(int(k)) for k in summary[(config, n_c)]['point_ranges'].keys()])))
            tauP, tauB, _ = list(map(np.log, thresholds_from_prior(summary[(config, n_c)]['prior'], point_values + [10])))
            priors = np.percentile(np.array(summary[(config, n_c)]['priors']),[5,50,95])
            
            ax_lr.set_title(f"{st} LR+ {config}\nprior: {priors[1]:.3f} ({priors[0]:.3f}-{priors[2]:.3f}), C: {summary[(config, n_c)]['C']}", fontsize=10)
            add_thresholds(tauP[:-1], tauB[:-1], ax_lr)
            ax_lr.set_xlabel("Score")
            ax_lr.set_ylabel("Log LR+")
            ax_lr.legend(fontsize=6, loc='best')
            ax_lr.set_xlim(xlim)
            ax_lr.set_ylim([tauB[-1], tauP[-1]])  # Set y-limits based on ±10 thresholds
            ax_lr.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused config columns in LR+ row
        for col_idx in range(max_samples + len(configs), n_cols_total):
            ax[row_lr, col_idx].axis('off')
        
        # ===== Fits/points row: Plot point assignments =====
        for config_idx, (config, n_c) in enumerate(configs):
            col_idx = max_samples + config_idx
            ax_points = ax[row_fits, col_idx]
            
            point_ranges = sorted([(int(k), v) for k,v in summary[(config, n_c)]['point_ranges'].items()])
            point_values = [pr[0] for pr in point_ranges]
            
            # Plot all samples on same axis
            for sample_num in range(n_samples_st):
                for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
                    for sr in scoreRanges:
                        ax_points.plot([sr[0], sr[1]], [pointIdx, pointIdx], 
                                     color='red' if pointVal > 0 else 'blue', 
                                     linestyle='-', alpha=0.7, linewidth=2)
            
            ax_points.set_ylim(-1, len(point_values))
            ax_points.set_yticks(range(len(point_values)), 
                               labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0", point_values)))
            ax_points.set_xlabel("Score")
            ax_points.set_ylabel("Points")
            ax_points.set_title(f"{st} Points {config}", fontsize=10)
            ax_points.set_xlim(xlim)
            ax_points.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused config columns in fits/points row
        for col_idx in range(max_samples + len(configs), n_cols_total):
            ax[row_fits, col_idx].axis('off')
    
    fig.suptitle(f"{scoreset_data[scoreset_types[-1]]['scoreset'].scoreset_name} ({n_c_auto_selected} auto selected)", fontsize=16, y=0.995)
    
    return fig


def sample_density(x, fits, sampleNum):
    x = np.asarray(x)
    _density = np.stack([density_utils.joint_densities(x, _fit['fit']['component_params'],_fit['fit']['weights'][sampleNum])
                        for _fit in fits])
    density = np.full(_density.shape,np.nan)
    for fitIdx,fit in enumerate(fits):
        fit_xmin,fit_xmax = fit['fit']['xlims']
        mask = (x >= fit_xmin) & (x <= fit_xmax)
        density[fitIdx,:,mask] = _density[fitIdx,:,mask]
    return density


def add_thresholds(tauP, tauB, ax):
    for tp,tb in zip(tauP,tauB):
        ax.axhline(tp,color='red',linestyle='--',alpha=0.5)
        ax.axhline(tb,color='blue',linestyle='--',alpha=0.5)


def plot_summary(scoreset: Scoreset, fits: List[Dict], summary:Dict, score_range, log_fp, log_fb, use_median_prior,use_2c_equation, n_c, benign_method, C, dataset):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    log_lr_plus = log_fp - log_fb
    llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
    labels = ['5th percentile','Median','95th percentile']
    for i,c in enumerate(['red','black','blue']):
        ax.plot(score_range,llr_curves[i],color=c,label=labels[i])
    point_values = sorted(list(set([abs(int(k)) for k in summary['point_ranges'].keys()])))
    tauP,tauB,_ = list(map(np.log, thresholds_from_prior(summary['prior'],point_values)) )
    priors = np.percentile(np.array(summary['priors']),[5,50,95])
    ax.set_title(f"{dataset} ({n_c}, median:{use_median_prior},em:{not use_2c_equation}): prior: {priors[1]:.3f} ({priors[0]:.3f}-{priors[2]:.3f}), C: {C}")
    add_thresholds(tauP, tauB, ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Log LR+")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


# claude generated. do not want to deal with the matplotlib headache :)
def plot_scoreset_best_config(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot a single configuration with samples in one row, point assignments below, and LR+ below that.
    
    Parameters:
    -----------
    dataset : str
        Dataset name
    scoreset : Scoreset
        The scoreset object
    indv_summary : dict
        Summary dict for this specific (config, n_c) configuration
    fits : list
        Fitted models
    score_range : np.ndarray
        Score range for plotting
    config : tuple
        Configuration tuple (e.g., ('avg',) or ('benign',))
    n_c : str
        '2c' or '3c'
    n_samples : int
        Number of samples in the scoreset
    relax : bool
        Whether this is a relaxed fit
    flipped : bool
        Whether the scoreset is flipped (higher scores = more pathogenic)
        Pass True if P/LP is on the right side
    """
    
    # Create figure: 3 rows, n_samples columns (all square)
    fig, ax = plt.subplots(3, n_samples, figsize=(7*n_samples, 18), 
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    relax_code = "R" if relax else ""
    
    # ===== Row 0: Sample fits =====
    num_skipped = 0
    for sample_num in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue
        ax_fit = ax[0, sample_num-num_skipped]
        
        # Get sample mask
        sample_mask = scoreset.sample_assignments[:,sample_num-num_skipped]
        
        sample_scores = scoreset.scores[sample_mask]
        # Cap bins to 100: Freedman-Diaconis can produce millions of bins when
        # IQR is near zero (e.g. all P/LP scores clustered at one extreme).
        n = len(sample_scores)
        q1, q3 = np.percentile(sample_scores, [25, 75])
        iqr = q3 - q1
        fd_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
        score_range_width = sample_scores.max() - sample_scores.min()
        bins = min(100, int(score_range_width / fd_width)) if fd_width > 0 else 50
        bins = max(bins, 10)

        # Plot based on sample number (which category)
        if sample_num == 0:  # P/LP
            sns.histplot(sample_scores, bins=bins,
                         stat='density', ax=ax_fit, alpha=0.6, color='#CA7682')
        elif sample_num == 1:  # B/LB
            sns.histplot(sample_scores, bins=bins,
                         stat='density', ax=ax_fit, alpha=0.6, color='#1D7AAB')
        elif sample_num == 2:  # gnomAD
            sns.histplot(sample_scores, bins=bins,
                         stat='density', ax=ax_fit, alpha=0.3, color='#A0A0A0')
        elif sample_num == 3:  # Synonymous
            sns.histplot(sample_scores, bins=bins,
                         stat='density', ax=ax_fit, alpha=0.5, color='#6BAA75')
    
        max_hist_density = max([patch.get_height() for patch in ax_fit.patches]) if ax_fit.patches else 1.0
        
        density = sample_density(score_range, fits, sample_num-num_skipped)
        for compNum in range(density.shape[1]):
            compDensity = density[:,compNum,:]
            d = np.nanpercentile(compDensity,[5,50,95],axis=0)
            ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
        ax_fit.legend(fontsize=8)
        
        d = np.nansum(density, axis=1)
        d_perc = np.percentile(d, [5,50,95], axis=0)
        ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
        ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
        ax_fit.set_title(f"{n_c}{relax_code}: {scoreset.sample_names[sample_num].replace('population','gnomAD')}\n(n={scoreset.sample_assignments[:,sample_num-num_skipped].sum():,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        ax_fit.set_ylim([0, max_hist_density * 1.1])
        ax_fit.grid(linewidth=0.5, alpha=0.3)
    
    # Get x-limits from first fit
    xlim = ax[0, 0].get_xlim()
    
    # ===== Row 1: Point assignments (one per sample) =====
    point_ranges = sorted([(int(k), v) for k,v in indv_summary['point_ranges'].items()])
    point_values = [pr[0] for pr in point_ranges]
    
    for sample_num in range(n_samples):
        ax_points = ax[1, sample_num]
        
        # Plot only this sample's point assignments (clip inf to xlim for rendering)
        for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                x0 = xlim[0] if np.isneginf(sr[0]) else max(sr[0], xlim[0])
                x1 = xlim[1] if np.isposinf(sr[1]) else min(sr[1], xlim[1])
                ax_points.plot([x0, x1], [pointIdx, pointIdx],
                             color='red' if pointVal > 0 else 'blue',
                             linestyle='-', alpha=0.7, linewidth=2)
        
        ax_points.set_ylim(-1, len(point_values))
        ax_points.set_yticks(range(len(point_values)), 
                           labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0", point_values)))
        ax_points.set_xlabel("Score")
        ax_points.set_ylabel("Points")
        ax_points.set_title(f"Point Assignments", fontsize=11)
        ax_points.set_xlim(xlim)
        ax_points.grid(linewidth=0.5, alpha=0.3)
    
    # ===== Row 2: LR+ summaries (one per sample) =====
    point_values_all = sorted(list(set([abs(int(k)) for k in indv_summary['point_ranges'].keys()])))
    tauP, tauB, _ = list(map(np.log, thresholds_from_prior(indv_summary['prior'], point_values_all + [10])))
    priors = np.percentile(np.array(indv_summary['priors']),[5,50,95])
    
    # Identify which point values have insufficient evidence (empty ranges)
    point_ranges_dict = {int(k): v for k, v in indv_summary['point_ranges'].items()}
    pathogenic_points = [p for p in point_values_all if p > 0]
    benign_points = [-p for p in point_values_all if p > 0]
    
    # Find highest pathogenic point with evidence
    highest_pathogenic_with_evidence = None
    for p in sorted(pathogenic_points, reverse=True):
        if p in point_ranges_dict and len(point_ranges_dict[p]) > 0:
            highest_pathogenic_with_evidence = p
            break
    
    # Find lowest benign point with evidence (most negative)
    lowest_benign_with_evidence = None
    for p in sorted(benign_points):
        if p in point_ranges_dict and len(point_ranges_dict[p]) > 0:
            lowest_benign_with_evidence = p
            break
    
    for sample_num in range(n_samples):
        ax_lr = ax[2, sample_num]
        
        log_lr_plus = indv_summary['log_lr_plus']
        # Prefer a pre-computed [p5,p50,p95] percentile array when the caller
        # provides one (log_lr_pct -- set by run_igvf_batch.py's/
        # analysis/legacy_fits.py's disk-reload helpers) rather than always
        # re-deriving via np.nanpercentile(log_lr_plus, ...). Re-deriving from
        # an already-percentiled log_lr_plus (as happens when only percentiles
        # were ever saved to disk, or when a caller mistakenly passes
        # percentiles in that slot) is NOT equivalent to the true percentiles:
        # taking the 5th percentile of 3 sorted values [p5,p50,p95] gives
        # ~90% weight to p5 and ~10% to p50 (linear interpolation, n=3),
        # which measurably inflates the reconstructed "5th percentile" curve
        # wherever the true median is far above the true p5th percentile.
        log_lr_pct = indv_summary.get('log_lr_pct')
        if log_lr_pct is not None and np.asarray(log_lr_pct).shape[0] == 3:
            llr_curves = np.asarray(log_lr_pct)
        else:
            llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
        labels = ['5th percentile','Median','95th percentile']
        colors = ['red','black','blue']
        
        # Check 5th percentile: find max and its position
        lr_5th = llr_curves[0]
        max_5th_idx = np.nanargmax(lr_5th)
        max_5th = lr_5th[max_5th_idx]
        exceeds_pathogenic = any(max_5th > tau for tau in tauP[:-1])
        
        # Check if insufficient evidence causes cutoff
        insufficient_evidence_pathogenic_idx = None
        if highest_pathogenic_with_evidence is not None and highest_pathogenic_with_evidence < max(pathogenic_points):
            # Find the threshold for the highest point with evidence
            tau_idx = pathogenic_points.index(highest_pathogenic_with_evidence)
            if tau_idx < len(pathogenic_points) - 1:
                tau_cutoff = tauP[tau_idx+1]
                # Find where curve crosses this threshold
                if not flipped:
                    crossing_indices = np.where(lr_5th >= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_pathogenic_idx = crossing_indices[-1]  # Last crossing
                else:
                    crossing_indices = np.where(lr_5th >= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_pathogenic_idx = crossing_indices[0]  # First crossing
        
        # Check if it comes back down after exceeding and find the crossing point
        should_plot_pathogenic_dotted = False
        pathogenic_crossing_idx = None
        if exceeds_pathogenic:
            # Find the highest threshold exceeded
            highest_tau_exceeded = max([tau for tau in tauP[:-1] if max_5th > tau])
            
            if not flipped:
                # Normal: Use FIRST crossing (when going up towards max from the left)
                before_max = lr_5th[:max_5th_idx+1]
                crossing_indices = np.where(before_max >= highest_tau_exceeded)[0]
                if len(crossing_indices) > 0:
                    # Check if it comes back down after the max
                    if max_5th_idx < len(lr_5th) - 1:
                        comes_back_down = any(lr_5th[max_5th_idx+1:] < highest_tau_exceeded)
                        if comes_back_down:
                            pathogenic_crossing_idx = crossing_indices[0]
                            should_plot_pathogenic_dotted = True
            else:
                # Flipped: Use SECOND crossing (when coming back down from max to the right)
                if max_5th_idx < len(lr_5th) - 1:
                    after_max = lr_5th[max_5th_idx+1:]
                    crossing_indices = np.where(after_max < highest_tau_exceeded)[0]
                    if len(crossing_indices) > 0:
                        # Check if it came from below before the max
                        comes_from_below = any(lr_5th[:max_5th_idx] < highest_tau_exceeded)
                        if comes_from_below:
                            pathogenic_crossing_idx = max_5th_idx + 1 + crossing_indices[0]
                            should_plot_pathogenic_dotted = True
        
        # Check 95th percentile: find min and its position
        lr_95th = llr_curves[2]
        min_95th_idx = np.nanargmin(lr_95th)
        min_95th = lr_95th[min_95th_idx]
        exceeds_benign = any(min_95th < tau for tau in tauB[:-1])
        
        # Check if insufficient evidence causes cutoff
        insufficient_evidence_benign_idx = None
        if lowest_benign_with_evidence is not None and lowest_benign_with_evidence > min(benign_points):
            # Find the threshold for the lowest benign point with evidence
            tau_idx = benign_points.index(lowest_benign_with_evidence)
            if tau_idx < len(benign_points) - 1:
                tau_cutoff = tauB[tau_idx+1]
                # Find where curve crosses this threshold
                if not flipped:
                    crossing_indices = np.where(lr_95th <= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_benign_idx = crossing_indices[-1]  # Last crossing
                else:
                    crossing_indices = np.where(lr_95th <= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_benign_idx = crossing_indices[0]  # First crossing
        
        # Check if it comes back up after going below and find the crossing point
        should_plot_benign_dotted = False
        benign_crossing_idx = None
        if exceeds_benign:
            # Find the lowest threshold crossed
            lowest_tau_crossed = min([tau for tau in tauB[:-1] if min_95th < tau])
            
            if not flipped:
                # Normal: Use SECOND crossing (when coming back up from min to the right)
                if min_95th_idx < len(lr_95th) - 1:
                    after_min = lr_95th[min_95th_idx+1:]
                    crossing_indices = np.where(after_min > lowest_tau_crossed)[0]
                    if len(crossing_indices) > 0:
                        # Check if it came from above before the min
                        comes_from_above = any(lr_95th[:min_95th_idx] > lowest_tau_crossed)
                        if comes_from_above:
                            benign_crossing_idx = min_95th_idx + 1 + crossing_indices[0]
                            should_plot_benign_dotted = True
            else:
                # Flipped: Use FIRST crossing (when going down towards min from the left)
                before_min = lr_95th[:min_95th_idx+1]
                crossing_indices = np.where(before_min <= lowest_tau_crossed)[0]
                if len(crossing_indices) > 0:
                    # Check if it comes back up after the min
                    if min_95th_idx < len(lr_95th) - 1:
                        comes_back_up = any(lr_95th[min_95th_idx+1:] > lowest_tau_crossed)
                        if comes_back_up:
                            benign_crossing_idx = crossing_indices[0]
                            should_plot_benign_dotted = True
        
        # Debug output
        if debug:
            print(f"\nSample {sample_num}:")
            print(f"  Highest pathogenic with evidence: {highest_pathogenic_with_evidence}, cutoff_idx: {insufficient_evidence_pathogenic_idx}")
            print(f"  Lowest benign with evidence: {lowest_benign_with_evidence}, cutoff_idx: {insufficient_evidence_benign_idx}")
            print(f"  5th percentile: max={max_5th:.3f} at idx={max_5th_idx}, exceeds_pathogenic={exceeds_pathogenic}")
            if exceeds_pathogenic:
                print(f"    highest_tau_exceeded={highest_tau_exceeded:.3f}, crossing_idx={pathogenic_crossing_idx}, should_plot_dotted={should_plot_pathogenic_dotted}")
            print(f"  tauP thresholds: {[f'{t:.3f}' for t in tauP[:-1]]}")
            print(f"  95th percentile: min={min_95th:.3f} at idx={min_95th_idx}, exceeds_benign={exceeds_benign}")
            if exceeds_benign:
                print(f"    lowest_tau_crossed={lowest_tau_crossed:.3f}, crossing_idx={benign_crossing_idx}, should_plot_dotted={should_plot_benign_dotted}")
            print(f"  tauB thresholds: {[f'{t:.3f}' for t in tauB[:-1]]}")
            print(f"  flipped={flipped}")
        
        # Plot curves
        for i, c in enumerate(colors):
            if len(log_lr_plus) == 1 and i != 1:
                continue  # if no bootstraps only plot one curve
            
            curve = llr_curves[i]
            
            # Handle 5th percentile (i=0)
            if i == 0:
                # Handle insufficient evidence
                if insufficient_evidence_pathogenic_idx is not None:
                    if debug:
                        print(f"  Plotting 5th percentile with insufficient evidence cutoff at idx {insufficient_evidence_pathogenic_idx}")
                    if not flipped:
                        # Regular: dotted up to cutoff, solid after
                        ax_lr.plot(score_range[:insufficient_evidence_pathogenic_idx+1], curve[:insufficient_evidence_pathogenic_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_pathogenic_idx:], curve[insufficient_evidence_pathogenic_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                    else:
                        # Flipped: solid up to cutoff, dotted after
                        ax_lr.plot(score_range[:insufficient_evidence_pathogenic_idx+1], curve[:insufficient_evidence_pathogenic_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_pathogenic_idx:], curve[insufficient_evidence_pathogenic_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                # Handle non-monotonic
                elif should_plot_pathogenic_dotted and pathogenic_crossing_idx is not None:
                    if debug:
                        print(f"  Plotting 5th percentile with non-monotonic cutoff at idx {pathogenic_crossing_idx}")
                    if not flipped:
                        # Normal: dotted before crossing, solid after
                        ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                    else:
                        # Flipped: solid before crossing, dotted after
                        ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                else:
                    ax_lr.plot(score_range, curve, color=c, label=labels[i], linewidth=2)
            
            # Handle 95th percentile (i=2)
            elif i == 2:
                # Handle insufficient evidence
                if insufficient_evidence_benign_idx is not None:
                    if debug:
                        print(f"  Plotting 95th percentile with insufficient evidence cutoff at idx {insufficient_evidence_benign_idx}")
                    if not flipped:
                        # Regular: solid up to cutoff, dotted after
                        ax_lr.plot(score_range[:insufficient_evidence_benign_idx+1], curve[:insufficient_evidence_benign_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_benign_idx:], curve[insufficient_evidence_benign_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                    else:
                        # Flipped: dotted up to cutoff, solid after
                        ax_lr.plot(score_range[:insufficient_evidence_benign_idx+1], curve[:insufficient_evidence_benign_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_benign_idx:], curve[insufficient_evidence_benign_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                # Handle non-monotonic
                elif should_plot_benign_dotted and benign_crossing_idx is not None:
                    if debug:
                        print(f"  Plotting 95th percentile with non-monotonic cutoff at idx {benign_crossing_idx}")
                    if not flipped:
                        # Normal: solid before crossing, dotted after
                        ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                    else:
                        # Flipped: dotted before crossing, solid after
                        ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                # Plot normally for median (i=1)
                else:
                    ax_lr.plot(score_range, curve, color=c, 
                             label=labels[i] if len(log_lr_plus) != 1 else 'Single fit',
                             linewidth=2)
            
        
        ax_lr.set_title(f"Log LR+\nprior: {priors[1]:.3f}, C: {indv_summary['C']}", fontsize=11)
        add_thresholds(tauP[:-1], tauB[:-1], ax_lr)
        ax_lr.set_xlabel("Score")
        ax_lr.set_ylabel("Log LR+")
        ax_lr.legend(fontsize=8, loc='best')
        ax_lr.set_xlim(xlim)
        ax_lr.set_ylim([tauB[-1], tauP[-1]])
        ax_lr.grid(linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if len(scoreset.sample_counts) > 3 and scoreset.sample_counts[1] == 0 and scoreset.sample_counts[3] != 0:
        config = str(config).replace("(benign)","(synonymous)" if scoreset.sample_counts[0] != 0 else "(NU)")
    elif scoreset.sample_counts[1] == 0:
        config = str(config).replace("(benign)","(PU)").replace("(avg)","(PU)").replace("(synonymous)","(PU)")
    elif scoreset.sample_counts[0] == 0:
        config = str(config).replace("(benign)","(NU)").replace("(avg)","(NU)").replace("(synonymous)","(NU)")
        
    fig.suptitle(f"{dataset} - {n_c}{relax_code} {config}", fontsize=16, y=0.998)
    
    return fig

def plot_scores_only(dataset, scoreset):
    n_samples = len([s for s in scoreset.samples])
    score_range = [min(scoreset.scores), max(scoreset.scores)]

    fig, ax = plt.subplots(1, n_samples, figsize=(7*n_samples, 6),
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    COLORS = {0: '#CA7682', 1: '#1D7AAB', 2: '#A0A0A0', 3: '#6BAA75'}
    ALPHAS  = {0: 0.6,      1: 0.6,      2: 0.3,      3: 0.5}

    # Use a single bin width for every sample so histograms are comparable.
    # Each sample's adaptive bin count (sqrt(n), floor=5, cap=30) implies a
    # candidate width; take the smallest of these (i.e. the finest binning,
    # from the sample with the most observations) as the shared bin edges.
    n_bins_per_sample = [
        int(np.clip(np.sqrt(scoreset.sample_counts[sample_num]), 5, 30))
        for sample_num in range(len(scoreset.sample_counts))
        if scoreset.sample_counts[sample_num] > 0
    ]
    shared_n_bins = max(n_bins_per_sample) if n_bins_per_sample else 1
    bin_edges = np.linspace(score_range[0], score_range[1], shared_n_bins + 1)

    num_skipped = 0
    for sample_num in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue
        ax_fit = ax[0, sample_num - num_skipped]
        sample_mask = scoreset.sample_assignments[:, sample_num - num_skipped]
        sample_scores = scoreset.scores[sample_mask]
        n = sample_mask.sum()

        sns.histplot(
            sample_scores,
            bins=bin_edges,        # shared bin width across all samples
            stat='density',
            ax=ax_fit,
            alpha=ALPHAS.get(sample_num, 0.5),
            color=COLORS.get(sample_num, '#888888'),
        )

        max_hist_density = (
            max(p.get_height() for p in ax_fit.patches)
            if ax_fit.patches else 1.0
        )

        label = scoreset.sample_names[sample_num]#.replace('population', 'gnomAD')
        ax_fit.set_title(f"{label}\n(n={n:,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        ax_fit.set_ylim([0, max_hist_density * 1.1])
        ax_fit.grid(linewidth=0.5, alpha=0.3)

    fig.suptitle(dataset, fontsize=18, fontweight="heavy", y=1.02)
    return fig

def plot_scoreset_example_publication(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot each sample in separate vertical subplots with all thresholds overlayed.
    
    Parameters: (same as original)
    """
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.6, 0.6, 0.3, 0.5]
    
    # Threshold configuration for benign (negative) and pathogenic (positive)
    # point_values_to_plot = [1, 2, 3, 4, 8]
    # linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    # linewidths = [1.25, 1.25, 1.25, 1.25, 1.25]
    # labels_thresh = ['Supporting', 'Moderate', 'Moderate+', 'Strong', 'Very Strong']
    point_values_to_plot = [1, 2, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3))]
    linewidths = [1.25, 1.25, 1.25, 1.25]
    labels_thresh = ['Supporting', 'Moderate', 'Strong', 'Very Strong']
    
    relax_code = "R" if relax else ""
    
    # Create figure with n_samples rows
    scale = 3
    fig, axes = plt.subplots(n_samples, 1, figsize=(2*scale, scale*n_samples), squeeze=False)
    axes = axes.flatten()
    
    # Get point ranges for threshold plotting — normalize keys to int
    point_ranges = {int(k): v for k, v in indv_summary['point_ranges'].items()}

    # Plot each sample in its own subplot
    plot_idx = 0
    for sample_num in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sample_num] == 0:
            continue
        ax = axes[plot_idx]
        sample_mask = scoreset.sample_assignments[:, plot_idx]
        sample_name = scoreset.sample_names[sample_num]
        color_idx = min(sample_num, len(sample_colors) - 1)

        # Plot histogram for this sample
        sns.histplot(scoreset.scores[sample_mask],
                     stat='density', ax=ax,
                     alpha=sample_alphas[color_idx],
                     color=sample_colors[color_idx])
                     # label=sample_name)
        
        max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0

        # Plot fitted density curves with matching color
        density_sample = sample_density(score_range, fits, plot_idx)
        
        # Plot sum of components
        d = np.nansum(density_sample, axis=1)
        d_perc = np.percentile(d, [5, 50, 95], axis=0)
        ax.plot(score_range, d_perc[1], 
               color='black', 
               alpha=0.5,
               linewidth=2)
        ax.fill_between(score_range, d_perc[0], d_perc[2], 
                       color='gray', 
                       alpha=0.3)
        
        import matplotlib.lines as mlines

        # Add threshold lines for all point values (both positive and negative)
        for idx, point_val in enumerate(point_values_to_plot):
            # Find benign threshold (negative point value)
            for pv, score_ranges in point_ranges.items():
                if pv == -point_val:
                    for sr in score_ranges:
                        threshold_score = sr[0] if not flipped else sr[1]
                        ax.axvline(threshold_score, 
                                  color='b',
                                  linestyle=linestyles[idx],
                                  linewidth=linewidths[idx],
                                  alpha=0.7)
                                  # label=f"-{point_val}")
                        break
                    break
            
            # Find pathogenic threshold (positive point value)
            for pv, score_ranges in point_ranges.items():
                if pv == point_val:
                    for sr in score_ranges:
                        threshold_score = sr[1] if not flipped else sr[0]
                        ax.axvline(threshold_score, 
                                  color='r',
                                  linestyle=linestyles[idx],
                                  linewidth=linewidths[idx],
                                  alpha=0.7)
                                  # label=f"+{point_val}")
                        break
                    break

        handles = []
        for idx, point_val in enumerate(point_values_to_plot):
            if len(point_ranges[point_val]) != 0 or len(point_ranges[-point_val]) != 0:
                h = mlines.Line2D(
                    [], [],
                    color='gray',
                    linestyle=linestyles[idx],
                    linewidth=linewidths[idx],
                    label=f"±{point_val}"
                )
                handles.append(h)

        if sample_name == "population":
            sample_name = "gnomAD"

        if plot_idx == 0:
            ax.set_title(dataset, fontsize=12, fontweight='bold', pad=8)
        
        # if sample_name != "gnomAD":
        #     ax.set_title(f"{sample_name} (n={sample_mask.sum():,d})", fontsize=16, fontweight='bold')
        # else:
        #     ax.set_title(f"{sample_name} (n={sample_mask.sum():,d}, prior={indv_summary['prior']:.3f})", fontsize=16, fontweight='bold')

        last_sample = (plot_idx == n_samples - 1)

        if last_sample:
            ax.set_xlabel("Assay score", fontsize=14)
        else:
            ax.set_xticks([])
        
        ax.set_ylabel("Density", fontsize=14)
        ax.set_ylim([0, max_hist_density * 1.1])
        # ax.legend(fontsize=11, loc='best', ncol=1, handles=handles)

        n_count = sample_mask.sum()

        
        # Create histogram legend handle
        hist_patch = Patch(
            facecolor=sample_colors[color_idx],
            alpha=0.4,
            edgecolor='none'
        )
        
        if sample_name == "gnomAD":
            hist_label = f'{sample_name}\n(n={n_count:,d}, prior={indv_summary["prior"]:.3f})'
        else:
            hist_label = f'{sample_name}\n(n={n_count:,d})'
        
        # Create histogram legend on the left
        hist_legend = ax.legend(
            [hist_patch],
            [hist_label],
            loc='upper left',
            fontsize=9,
            framealpha=0.8
        )
        
        # Add point ranges legend on the right (if there are any)
        if handles:
            # Need to add the first legend back as an artist
            ax.add_artist(hist_legend)
            
            # Create points legend on the right
            ax.legend(
                handles,
                [h.get_label() for h in handles],
                loc='upper right',
                ncol=2,
                fontsize=8,
                framealpha=0.8,
                handlelength=2,
                columnspacing=1.0
            )
        
        ax.grid(linewidth=0.5, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()
    
    return fig

def plot_scoreset_calibration_comparison(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot histogram with P/LP, B/LB, all SNVs, threshold lines, and calibration comparisons below.
    """
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Strength colors for calibration bars
    strenth_color = {
        "BS3 Very Strong": "#4b91a6",
        "BS3 Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BS3 Moderate": "#d0e8f0",
        "BS3 Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PS3 Supporting": "#e6b1b8",
        "PS3 Moderate": "#d68f99",
        "+3": "#ca7682",
        "PS3 Strong": "#b85c6b",
        "PS3 Very Strong": "#943744"
    }
    
    relax_code = "R" if relax else ""
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 0.3], hspace=0.3)
    
    # Main histogram
    ax1 = plt.subplot(gs[0])
    # Jia 2021 (0-cutoff)
    ax2 = plt.subplot(gs[1])
    # DanZ-based calibration
    ax3 = plt.subplot(gs[2])
    # Legend axis
    leg_ax = plt.subplot(gs[3])
    leg_ax.axis('off')
    
    # Get all scores
    all_scores = scoreset.snv_scores
    x_min = min(all_scores.min(), scoreset.scores.min())
    x_max = max(all_scores.max(), scoreset.scores.max())
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Assume first two samples are P/LP and B/LB
    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]
    
    # Plot histograms
    sns.histplot(scoreset.scores[blb_mask], 
                 binwidth=bin_width, color='#1D7AAB', alpha=0.6, ax=ax1, label='ClinVar B/LB')
    sns.histplot(scoreset.scores[plp_mask], 
                 binwidth=bin_width, color='#CA7682', alpha=0.6, ax=ax1, label='ClinVar P/LP')
    
    # Overlay all SNVs on secondary axis
    ax1_twin = ax1.twinx()
    sns.histplot(all_scores, binwidth=bin_width, color='#A0A0A0', alpha=0.3, ax=ax1_twin, label='All SNVs')
    
    # Add threshold vertical lines
    threshold_scores_benign = []
    threshold_scores_path = []
    
    for idx, point_val in enumerate(point_values_to_plot):
        # Find benign threshold (negative point value)
        for pv, score_ranges in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    # ax1.axvline(threshold_score, 
                    #           color='b',
                    #           linestyle=linestyles[idx],
                    #           linewidth=linewidths[idx],
                    #           alpha=0.7)
                    break
                break
        
        # Find pathogenic threshold (positive point value)
        for pv, score_ranges in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    # ax1.axvline(threshold_score, 
                    #           color='r',
                    #           linestyle=linestyles[idx],
                    #           linewidth=linewidths[idx],
                    #           alpha=0.7)
                    break
                break
    
    ax1.set_xlim(x_min, x_max)
    ax1_twin.set_xlim(x_min, x_max)
    ax1.set_xlabel('')
    ax1.set_ylabel('P/B variant count', fontsize=14)
    ax1_twin.set_ylabel('SNV count', fontsize=14)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    
    # Add threshold legend
    import matplotlib.lines as mlines
    threshold_handles = []
    # for idx, point_val in enumerate(point_values_to_plot):
    #     if point_val in point_ranges or -point_val in point_ranges:
    #         h = mlines.Line2D([], [], color='gray', linestyle=linestyles[idx], 
    #                         linewidth=linewidths[idx], label=f"±{point_val}")
    #         threshold_handles.append(h)
    
    ax1.legend(lines1 + lines2 + threshold_handles, labels1 + labels2 + [h.get_label() for h in threshold_handles], 
              loc='best', fontsize=11, ncol=1)
    ax1_twin.get_legend().remove() if ax1_twin.get_legend() else None
    ax1.set_title(f'MSH2 Functional Score', fontsize=22, fontweight='bold')
    
    # Row 2: Scott
    ax2.axvspan(x_min, 0, color=strenth_color['BS3 Strong'], alpha=0.9)
    ax2.axvspan(0, 0.4, color=strenth_color['IR'], alpha=0.9)
    ax2.axvspan(0.4, x_max, color=strenth_color['PS3 Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_above_0 = (all_scores > 0.4).sum()
    ax2.text(x_min/2, 5, f'{count_below_0}', ha='center', va='center', color='black', fontsize=13)
    ax2.text(x_max/2, 5, f'{count_above_0}', ha='center', va='center', color='black', fontsize=13)
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 10)
    ax2.set_ylabel('SNV Count', fontsize=14)
    ax2.set_yticks([])
    ax2.set_xticks([])#[0], ['0'], fontsize=14)
    ax2.set_title('Scott et al. (2022)', loc='left', pad=5, fontsize=18, style='italic')
    ax2.grid(False)
    
    # Row 3: DanZ-based calibration with threshold intervals
    # Build intervals from thresholds
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)  # -8, -4, -3, -2, -1
    threshold_scores_path_sorted = sorted(threshold_scores_path)  # +1, +2, +3, +4, +8
    
    intervals = []
    
    # Benign intervals
    # if len(threshold_scores_benign_sorted) >= 3:
    intervals.append(("BS3 Moderate", x_min, threshold_scores_benign_sorted[0]))  # -3 and below
    intervals.append(("BS3 Supporting", threshold_scores_benign_sorted[0], threshold_scores_benign_sorted[1]))  # -2

    # print(threshold_scores_benign_sorted)
    # print(threshold_scores_path_sorted)
    
    # IR interval
    # if len(threshold_scores_benign_sorted) >= 5 and len(threshold_scores_path_sorted) >= 1:
    intervals.append(("IR", threshold_scores_benign_sorted[1], threshold_scores_path_sorted[0]))
    
    # Pathogenic intervals
    # if len(threshold_scores_path_sorted) >= 3:
    intervals.append(("PS3 Supporting", threshold_scores_path_sorted[0], threshold_scores_path_sorted[1]))  # +1
    intervals.append(("PS3 Moderate", threshold_scores_path_sorted[1], threshold_scores_path_sorted[2]))  # +2
    intervals.append(("+3", threshold_scores_path_sorted[2], x_max))  # +3 and above
    
    for name, start, end in intervals:
        ax3.axvspan(start, end, color=strenth_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.2:
            ax3.text((start + end) / 2, 5, str(count), ha='center', va='center', 
                    fontsize=13, color='black')
    
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(0, 10)
    ax3.set_ylabel('SNV Count', fontsize=14)
    ax3.set_yticks([])
    
    # Set x-axis ticks at thresholds
    all_thresholds = threshold_scores_benign_sorted + threshold_scores_path_sorted
    ax3.set_xticks([])#all_thresholds, [f'{x:.2f}' for x in all_thresholds], rotation=60, fontsize=12)
    ax3.set_title('Zeiberg et al. (2025)', loc='left', pad=5, fontsize=18, style='italic')
    ax3.grid(False)
    
    # Legend for strength colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=strenth_color[name], label=name) for name in [
        "BS3 Moderate", "BS3 Supporting", "IR", 
        "PS3 Supporting", "PS3 Moderate", "+3"
        # "BS3 Very Strong", "BS3 Strong", "-3", "BS3 Moderate", "BS3 Supporting", "IR", 
        # "PS3 Supporting", "PS3 Moderate", "+3", "PS3 Strong", "PS3 Very Strong"
    ]]
    leg_ax.legend(handles=legend_elements, loc='center', ncol=6, frameon=False, fontsize=12)
    
    # plt.suptitle(f'MSH2 Assay', fontsize=22, fontweight='bold')
    plt.tight_layout()
    
    return fig

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

def create_variant_id_spdi(entry):
    """Create variant ID in SPDI format: Chrom:Position:Reference:Alternate."""
    chrom = entry.get("Chrom")
    pos = entry.get("hg38_start")
    ref = entry.get("ref_allele")
    alt = entry.get("alt_allele")
    
    # Handle NaN or None values
    if pd.isna(chrom) or pd.isna(pos) or pd.isna(ref) or pd.isna(alt):
        return None
    
    # Convert position to int
    try:
        pos = int(pos)
    except (ValueError, TypeError):
        return None
    
    return f"{chrom}:{pos}:{ref}:{alt}"

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
    
    # Assert only one or no ranges matched
    assert len(matched_points) <= 1, f"Score {assay_score} matched multiple ranges: {matched_points}, point ranges: {point_ranges}"
    
    return matched_points[0] if matched_points else 0

def is_empty(value):
    """Check if value is None, NaN, or empty string"""
    return pd.isna(value) or value == "" or (isinstance(value, str) and value.strip() == "")


def calculate_confusion_mat(dataset, scoreset, calibration_f, verbose=True):#, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False, verbose=False):
    """
    Plot confusion matrix comparing DanZ calibration vs Author.
    Shows how P/LP and B/LB variants are classified into Benign/IR/Pathogenic ranges.
    """


    if pd.isna(scoreset.auth_labels).all():
        print(f"{dataset}: all auth labels are nan")
        return None, None
    
    # Get point ranges
    with open(calibration_f, 'r') as f:
        calibration_data = json.load(f)

    if calibration_data["point_ranges"] is None:
        raise ValueError(f"  ERROR: Dataset was uncalibratable {calibration_f}")
        return None, None
    
    point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
    
    # Assume first two samples are P/LP and B/LB
    plp_scores = scoreset.scores[scoreset.sample_assignments[:, 0]]
    plp_auth_labels = scoreset.auth_labels[scoreset.sample_assignments[:, 0]]
    blb_scores = scoreset.scores[scoreset.sample_assignments[:, 1]] 
    blb_auth_labels = scoreset.auth_labels[scoreset.sample_assignments[:, 1]] 

    # Determine benign, IR, and pathogenic score ranges for DanZ
    # benign_ranges = []
    # ir_ranges = []
    # pathogenic_ranges = []
    
    # for pv, score_ranges in point_ranges.items():
    #     if int(pv) < 0:  # Benign
    #         benign_ranges.extend(score_ranges)
    #     elif int(pv) > 0:  # Pathogenic
    #         pathogenic_ranges.extend(score_ranges)
    
    # Count B/LB variants in each category (DanZ)
    # blb_in_benign = sum(any(start <= score <= end for start, end in benign_ranges) for score in blb_scores)
    # blb_in_path   = sum(any(start <= score <= end for start, end in pathogenic_ranges) for score in blb_scores)
    blb_in_benign = sum(assign_points(score, point_ranges) < 0 for score in blb_scores)
    blb_in_path = sum(assign_points(score, point_ranges) > 0 for score in blb_scores)
    blb_in_ir = sum(assign_points(score, point_ranges) == 0 for score in blb_scores)
    
    plp_in_benign = sum(assign_points(score, point_ranges) < 0 for score in plp_scores)
    plp_in_path = sum(assign_points(score, point_ranges) > 0 for score in plp_scores)
    plp_in_ir = sum(assign_points(score, point_ranges) == 0 for score in plp_scores)
    
    # # IR = NOT benign AND NOT path
    # blb_in_ir = sum(
    #     not any(start <= score <= end for start, end in benign_ranges) and
    #     not any(start <= score <= end for start, end in pathogenic_ranges)
    #     for score in blb_scores
    # )
    
    # Count P/LP variants in each category (DanZ)
    # plp_in_benign = sum(any(start <= score <= end for start, end in benign_ranges) for score in plp_scores)
    # plp_in_path   = sum(any(start <= score <= end for start, end in pathogenic_ranges) for score in plp_scores)
    
    # plp_in_ir = sum(
    #     not any(start <= score <= end for start, end in benign_ranges) and
    #     not any(start <= score <= end for start, end in pathogenic_ranges)
    #     for score in plp_scores
    # )

    
    # Create DanZ DataFrame
    danz = pd.DataFrame({
        'BLB': [blb_in_benign, blb_in_ir, blb_in_path],
        'PLP': [plp_in_benign, plp_in_ir, plp_in_path]
    }).T

    indeterminate_codes = ["NOT SPECIFIED", "INDETERMINATE", "IGNORE"]

    # Count for Jia 2021 (0-cutoff)
    blb_abnorm = (np.char.upper(blb_auth_labels) == "ABNORMAL").sum()
    blb_ir = (np.isin(np.char.upper(blb_auth_labels), indeterminate_codes) | pd.isna(blb_auth_labels)).sum()
    blb_norm = (np.char.upper(blb_auth_labels) == "NORMAL").sum()
    plp_abnorm = (np.char.upper(plp_auth_labels) == "ABNORMAL").sum()
    plp_ir = (np.isin(np.char.upper(plp_auth_labels), indeterminate_codes) | pd.isna(plp_auth_labels)).sum()
    plp_norm = (np.char.upper(plp_auth_labels) == "NORMAL").sum()
    
    # Create Jia 2021 DataFrame
    auth = pd.DataFrame({
        'BLB': [blb_norm, blb_ir, blb_abnorm],
        'PLP': [plp_norm, plp_ir, plp_abnorm]
    }).T
        
    ind = ['Normal', 'IR', 'Abnormal']
    danz.columns = ind
    auth.columns = ind

    if verbose:
        print(dataset)
        print('danz',danz)
        print('auth',auth)

    return danz, auth

def plot_confusion_mat(danzs, auths):
    """
    danzs: list of DanZ count DataFrames
    auths: list of author count DataFrames
    """
    valid_indices = [
        i for i in range(len(danzs))
        if danzs[i] is not None and auths[i] is not None
    ]

    danzs_filtered = [danzs[i] for i in valid_indices]
    auths_filtered = [auths[i] for i in valid_indices]

    if len(danzs_filtered) == 0:
        raise ValueError("No valid datasets remaining after filtering!")
    
    print(f"Using {len(danzs_filtered)}/{len(danzs)} datasets")
    
    # ----------------------------------------------
    # 1. Raw accumulation
    # ----------------------------------------------
    danz_accum = danzs_filtered[0].copy() * 0
    auth_accum = auths_filtered[0].copy() * 0
    
    for D in danzs_filtered:
        danz_accum += D
    
    for A in auths_filtered:
        auth_accum += A
    
    # # ----------------------------------------------
    # # 2. Row-wise percentages
    # # ----------------------------------------------
    # danz_pct = danz_accum.div(danz_accum.sum(axis=1), axis=0) * 100
    # auth_pct = auth_accum.div(auth_accum.sum(axis=1), axis=0) * 100
    
    # # handle rows with total = 0
    # danz_pct = danz_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
    # auth_pct = auth_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # # ----------------------------------------------
    # # 3. Row-wise percent difference
    # #    (rows sum to zero)
    # # ----------------------------------------------
    # difference = danz_pct - auth_pct

    difference = (danz_accum - auth_accum) / auth_accum * 100
    difference = difference.replace([np.inf, -np.inf], np.nan)


    # ----------------------------------------------
    # 3. Plot
    # ----------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    colors = ["gold", "whitesmoke", "purple"]
    cmap_custom = LinearSegmentedColormap.from_list("PurpleYellow", colors)
    max_abs_value = 10
    step = 1
    posbounds = np.arange(0, max_abs_value + step, step)
    negbounds = -np.arange(step, max_abs_value + step, step)[::-1]
    bounds = np.concatenate([negbounds, posbounds])
    norm = BoundaryNorm(bounds, cmap_custom.N)
    
    def format_diff_value(x):
        if pd.isna(x):
            return ""
        if abs(x) < 0.1:
            return f"{x:+.2f}%"
        return f"{x:+.1f}%"
    
    annot_data = np.vectorize(format_diff_value)(difference)
    fig = plt.figure(figsize=(10, 3))
    sns.heatmap(
        difference, annot=annot_data, fmt='',
        cmap=cmap_custom, norm=norm,
        cbar_kws={'label': 'Percentage Point Difference', 'pad': 0.01},
        linewidths=0.5, linecolor='gray'
    )
    plt.title('Zeiberg et al. (2025) vs. Author Difference',
              fontsize=14, fontweight='bold')
    plt.ylabel('ClinVar Classification')
    plt.xlabel('Functional Category')
    plt.tight_layout()
    return fig

def plot_individual_confusion_matrices(dataset_names, danzs, auths, dataset_configs, 
                                       figsize_per_row=(12, 4)):
    """
    Plot confusion matrices for individual datasets showing counts in each bin.
    Each row shows one dataset with DanZ (left) and Author (right) annotations.
    
    Parameters:
    -----------
    dataset_names : list of str
        List of dataset names
    danzs : list of DataFrames
        List of DanZ count matrices
    auths : list of DataFrames
        List of author count matrices
    dataset_configs : dict
        Dataset configurations for title info
    figsize_per_row : tuple
        Figure size per dataset row (width, height)
    
    Returns:
    --------
    matplotlib figure
    """
    
    # Filter valid datasets
    valid_indices = []
    for i in range(len(danzs)):
        if danzs[i] is None or auths[i] is None:
            continue
        
        danz_row_sums = danzs[i].sum(axis=1)
        auth_row_sums = auths[i].sum(axis=1)
        
        if (danz_row_sums == 0).any() or (auth_row_sums == 0).any():
            continue
        
        valid_indices.append(i)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid datasets!")
    
    n_datasets = len(valid_indices)
    
    # Create figure with subplots: n_datasets rows × 2 columns
    fig, axes = plt.subplots(
        n_datasets, 2, 
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_datasets)
    )
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme similar to the reference image
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#F0F0F0", "#8FBC8F", "#4682B4", "#1E3A5F"]  # Light gray to dark blue
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for row_idx, data_idx in enumerate(valid_indices):
        dataset_name = dataset_names[data_idx]
        danz_df = danzs[data_idx]
        auth_df = auths[data_idx]
        
        # Get gene and author for title
        gene_name = dataset_name.split('_')[0]
        author_name = dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ""
        
        # Plot DanZ (left panel)
        ax_danz = axes[row_idx, 0]
        
        # Get max value for consistent color scale across both panels
        max_val = max(danz_df.values.max(), auth_df.values.max())
        
        sns.heatmap(
            danz_df,
            annot=True,
            fmt='d',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_danz,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Panel letter and title for DanZ
        # ax_danz.text(
        #     -0.15, 1.05,
        #     f"({panel_letters[row_idx*2]})",
        #     transform=ax_danz.transAxes,
        #     fontsize=12,
        #     fontweight='bold',
        #     verticalalignment='top'
        # )
        
        ax_danz.set_title(
            f"{gene_name} \u2013 {author_name}\nVariant-level Annotations",
            fontsize=11,
            fontweight='bold',
            pad=10
        )
        
        ax_danz.set_xlabel('Evidence Direction', fontsize=10)
        ax_danz.set_ylabel('ClinVar Classification', fontsize=10)
        ax_danz.tick_params(labelsize=9)
        
        # Rotate x-axis labels
        ax_danz.set_xticklabels(ax_danz.get_xticklabels(), rotation=45, ha='right')
        ax_danz.set_yticklabels(ax_danz.get_yticklabels(), rotation=0)
        
        # Plot Author (right panel)
        ax_auth = axes[row_idx, 1]
        
        sns.heatmap(
            auth_df,
            annot=True,
            fmt='d',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_auth,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Panel letter and title for Author
        # ax_auth.text(
        #     -0.15, 1.05,
        #     f"({panel_letters[row_idx*2 + 1]})",
        #     transform=ax_auth.transAxes,
        #     fontsize=12,
        #     fontweight='bold',
        #     verticalalignment='top'
        # )
        
        ax_auth.set_title(
            f"{gene_name} \u2013 {author_name}\nAuthor Annotations",
            fontsize=11,
            fontweight='bold',
            pad=10
        )
        
        ax_auth.set_xlabel('Functional Annotation', fontsize=10)
        ax_auth.set_ylabel('ClinVar Classification', fontsize=10)
        ax_auth.tick_params(labelsize=9)
        
        # Rotate x-axis labels
        ax_auth.set_xticklabels(ax_auth.get_xticklabels(), rotation=45, ha='right')
        ax_auth.set_yticklabels(ax_auth.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    return fig


def plot_scoreset_final_pillar_project(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Combined plot with:
    - All samples (P/LP, B/LB, gnomAD, Synonymous) overlayed in one histogram with fitted densities
    - Vertical threshold lines
    - Two calibration interval bars below (Scott et al. and Zeiberg et al.)
    
    Parameters: (same as original)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    from matplotlib.patches import Patch
    import seaborn as sns
    import numpy as np
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.5, 0.5, 0.15, 0.4]  # gnomAD more transparent as background
    
    # Darker versions of sample colors for fitted density lines
    fit_colors = ['#8B3A47', '#0D4A6B', '#505050', '#3A7A45']  # Darker versions
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Strength colors for calibration bars
    strength_color = {
        "BS3 Very Strong": "#4b91a6",
        "BS3 Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BS3 Moderate": "#d0e8f0",
        "BS3 Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PS3 Supporting": "#e6b1b8",
        "PS3 Moderate": "#d68f99",
        "+3": "#ca7682",
        "PS3 Strong": "#b85c6b",
        "PS3 Very Strong": "#943744"
    }
    
    relax_code = "R" if relax else ""
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.6, 0.6, 0.4], hspace=0.35)
    
    ax_hist = plt.subplot(gs[0])   # Main histogram with all samples
    ax_scott = plt.subplot(gs[1])  # Scott et al. calibration
    ax_zeiberg = plt.subplot(gs[2])  # Zeiberg et al. calibration
    leg_ax = plt.subplot(gs[3])    # Legend axis
    leg_ax.axis('off')
    
    # Get score ranges for x-axis limits
    all_scores = scoreset.snv_scores
    x_min = score_range[0]
    x_max = score_range[-1]
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Build legend handles for samples (histogram + fit)
    sample_handles = []
    
    # Plot each sample's histogram and fitted density
    for sample_num in range(n_samples):
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = scoreset.sample_names[sample_num]
        color = sample_colors[sample_num]
        fit_color = fit_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        # Plot histogram for this sample
        sns.histplot(
            scoreset.scores[sample_mask],
            binwidth=bin_width,
            stat='density',
            ax=ax_hist,
            alpha=alpha,
            color=color,
        )
        
        # Plot fitted density curve with darker color for visibility
        density_sample = sample_density(score_range, fits, sample_num)
        d = np.nansum(density_sample, axis=1)
        d_perc = np.percentile(d, [5, 50, 95], axis=0)
        
        fit_alpha = 0.5 if sample_num == 2 else 1.0  # gnomAD fit more subtle
        
        # Darker median line
        ax_hist.plot(
            score_range, d_perc[1],
            color=fit_color,
            alpha=fit_alpha,
            linewidth=2,
            zorder=10
        )
        # Confidence band
        ax_hist.fill_between(
            score_range, d_perc[0], d_perc[2],
            color=fit_color,
            alpha=0.15 if sample_num == 2 else 0.2,
            zorder=5
        )
        
        # Create combined legend handle: patch for histogram + line for fit
        hist_patch = Patch(facecolor=color, alpha=alpha, edgecolor='none')
        fit_line = mlines.Line2D([], [], color=fit_color, linewidth=2, alpha=fit_alpha)
        sample_handles.append((hist_patch, fit_line, f'{sample_name} (n={sample_mask.sum():,d})'))
    
    # Collect thresholds for calibration bars
    threshold_scores_benign = []
    threshold_scores_path = []
    
    # Add threshold vertical lines
    for idx, point_val in enumerate(point_values_to_plot):
        # Find benign threshold (negative point value)
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    ax_hist.axvline(
                        threshold_score,
                        color='#2166AC',  # Darker blue
                        linestyle=linestyles[idx],
                        linewidth=linewidths[idx],
                        alpha=0.8
                    )
                    break
                break
        
        # Find pathogenic threshold (positive point value)
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    ax_hist.axvline(
                        threshold_score,
                        color='#B2182B',  # Darker red
                        linestyle=linestyles[idx],
                        linewidth=linewidths[idx],
                        alpha=0.8
                    )
                    break
                break
    
    # Build threshold legend handles
    threshold_handles = []
    for idx, point_val in enumerate(point_values_to_plot):
        if len(point_ranges.get(point_val, [])) != 0 or len(point_ranges.get(-point_val, [])) != 0:
            h = mlines.Line2D(
                [], [],
                color='gray',
                linestyle=linestyles[idx],
                linewidth=linewidths[idx],
                label=f"±{point_val}"
            )
            threshold_handles.append(h)
    
    # Configure main histogram axis
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_xlabel('')
    ax_hist.set_ylabel('Density', fontsize=12)
    ax_hist.set_title(f'{dataset.split("_")[0]} Functional Score (2018 ClinVar)', fontsize=14, fontweight='bold')
    ax_hist.tick_params(axis='both', labelsize=10)
    
    # Create sample legend (upper left) with histogram + fit distinction
    from matplotlib.legend_handler import HandlerTuple
    sample_legend_handles = [(h[0], h[1]) for h in sample_handles]
    sample_legend_labels = [h[2] for h in sample_handles]
    
    legend1 = ax_hist.legend(
        sample_legend_handles, 
        sample_legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        loc='upper left',
        fontsize=10,
    )
    ax_hist.add_artist(legend1)
    
    # Create threshold legend (upper right)
    legend2 = ax_hist.legend(
        handles=threshold_handles,
        loc='upper right',
        fontsize=10,
    )
    
    # Row 2: Scott et al. (2022) calibration
    ax_scott.axvspan(x_min, 0, color=strength_color['BS3 Strong'], alpha=0.9)
    ax_scott.axvspan(0, 0.4, color=strength_color['IR'], alpha=0.9)
    ax_scott.axvspan(0.4, x_max, color=strength_color['PS3 Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_0_to_04 = ((all_scores >= 0) & (all_scores < 0.4)).sum()
    count_above_04 = (all_scores >= 0.4).sum()
    ax_scott.text((x_min + 0) / 2, 0.5, f'{count_below_0}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0 + 0.4) / 2, 0.5, f'{count_0_to_04}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0.4 + x_max) / 2, 0.5, f'{count_above_04}', ha='center', va='center', color='black', fontsize=11)
    
    ax_scott.set_xlim(x_min, x_max)
    ax_scott.set_ylim(0, 1)
    ax_scott.set_yticks([])
    ax_scott.set_xticks([])
    ax_scott.set_title('Scott et al. (2022)', loc='left', pad=3, fontsize=11, style='italic')
    
    # Row 3: Zeiberg et al. (2025) calibration with threshold intervals
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)
    threshold_scores_path_sorted = sorted(threshold_scores_path)
    

    legend_order = ["BS3 Very Strong", "BS3 Strong", "-3", "BS3 Moderate", "BS3 Supporting", "IR", "PS3 Supporting", "PS3 Moderate", "+3", "PS3 Strong", "PS3 Very Strong"]
    intervals = []
    
    # if len(threshold_scores_benign_sorted) >= 2 and len(threshold_scores_path_sorted) >= 3:
    for count,i in enumerate(range(len(threshold_scores_benign_sorted)-1, -1, -1)):
        evidence_code = legend_order[4-len(threshold_scores_benign_sorted)+1+count]
        intervals.append((evidence_code, threshold_scores_benign_sorted[i-1] if i > 0 else x_min,threshold_scores_benign_sorted[i]))
        # used_strengths.add(evidence_code)
    intervals = intervals[::-1]
    intervals.append(("IR", threshold_scores_benign_sorted[-1], threshold_scores_path_sorted[0]))
    # used_strengths.add("IR")
    for i in range(len(threshold_scores_path_sorted)):
        evidence_code = legend_order[6+i]
        intervals.append((evidence_code, threshold_scores_path_sorted[i],threshold_scores_path_sorted[i+1] if i < len(threshold_scores_path_sorted)-1 else x_max))
        # used_strengths.add(evidence_code)
    
    for name, start, end in intervals:
        ax_zeiberg.axvspan(start, end, color=strength_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.2:
            ax_zeiberg.text(
                (start + end) / 2, 0.5, str(count),
                ha='center', va='center',
                fontsize=11, color='black'
            )
    
    ax_zeiberg.set_xlim(x_min, x_max)
    ax_zeiberg.set_ylim(0, 1)
    ax_zeiberg.set_yticks([])
    ax_zeiberg.set_xlabel('Assay Score', fontsize=12)
    ax_zeiberg.tick_params(axis='x', labelsize=10)
    ax_zeiberg.set_title('Zeiberg et al. (2025)', loc='left', pad=3, fontsize=11, style='italic')
    
    # Collect all used strength names
    used_strengths = {"BS3 Strong", "IR", "PS3 Strong"}  # Scott always uses these
    for name, _, _ in intervals:
        used_strengths.add(name)
    
    # Build legend in order
    all_legend = [
        Patch(facecolor=strength_color[name], label=name, edgecolor='none')
        for name in legend_order
        if name in used_strengths
    ]
    
    leg_ax.legend(handles=all_legend, loc='upper center', ncol=len(all_legend), frameon=False, fontsize=9)
    
    plt.tight_layout()
    
    return fig


from matplotlib.colors import to_rgba

def plot_scoreset_final_pillar_project_v2(dataset, scoreset_2018, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Combined plot with:
    - Top row: Individual sample fits (one per column) showing mixture components
    - Second row: All samples overlayed in one histogram (no fits)
    - Third row: Scott et al. calibration
    - Fourth row: Zeiberg et al. calibration
    - Bottom: Legend
    
    Parameters: (same as original)
    """
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.5, 0.5, 0.15, 0.4]  # gnomAD more transparent as background

    # Component colors for fits (colorblind-friendly palette).
    # n_c is only ever used for this color count — callers historically pass
    # None here (see test/plot_MSH2_ex.py), so fall back to inferring the
    # component count directly from the fits themselves.
    if n_c is None:
        n_c = len(fits[0]['fit']['component_params'])
    component_colors = plt.cm.Set2(np.linspace(0, 1, n_c))
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Strength colors for calibration bars
    strength_color = {
        "BS3 Very Strong": "#4b91a6", 
        "BS3 Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BS3 Moderate": "#d0e8f0",
        "BS3 Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PS3 Supporting": "#e6b1b8",
        "PS3 Moderate": "#d68f99",
        "+3": "#ca7682",
        "PS3 Strong": "#b85c6b",
        "PS3 Very Strong": "#943744"
    }
    
    # Create figure with GridSpec
    # fig = plt.figure(figsize=(18, 13.17460317))
    # gs = gridspec.GridSpec(5, n_samples, height_ratios=[2, 2, 1, 1, 0.3], hspace=0.3, wspace=0.15)

    fig = plt.figure(figsize=(12, 13.2))

    
    gs = gridspec.GridSpec(
        5, 3,                                   # 5 rows, 3 columns
        height_ratios=[2, 2, 1, 1, 0.3],       # first 4 match, 5th extra
        hspace=0.3,
        wspace=0.15
    )

    
    # Top row: individual fit panels (one per sample)
    ax_fits = [plt.subplot(gs[0, i]) for i in range(n_samples)]
    
    # Second row: combined histogram spanning all columns
    ax_hist = plt.subplot(gs[1, :])
    
    # Third row: Scott et al. calibration
    ax_scott = plt.subplot(gs[2, :])
    
    # Fourth row: Zeiberg et al. calibration
    ax_zeiberg = plt.subplot(gs[3, :])
    
    # Bottom: Legend axis
    leg_ax = plt.subplot(gs[4, :])
    leg_ax.axis('off')
    
    # Add row title for top row
    # fig.text(0.5, 0.902, 'Model Fits (2018 ClinVar)', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Get score ranges for x-axis limits
    all_scores = scoreset.snv_scores
    x_min = score_range[0]
    x_max = score_range[-1]
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Pre-compute thresholds for use in fit panels
    threshold_scores_benign = []
    threshold_scores_path = []
    threshold_info = []  # Store (threshold_score, color, linestyle, linewidth) tuples
    
    for idx, point_val in enumerate(point_values_to_plot):
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    threshold_info.append((threshold_score, '#2166AC', linestyles[idx], linewidths[idx]))
                    break
                break
        
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    threshold_info.append((threshold_score, '#B2182B', linestyles[idx], linewidths[idx]))
                    break
                break

    sample_name_shortener = {
        "Pathogenic/Likely Pathogenic": "ClinVar P/LP",
        "Benign/Likely Benign": "ClinVar B/LB",
        "gnomAD": "gnomAD",
        "population": "gnomAD",  # "population" is the literal sample name written to disk for some datasets
    }
    
    # ===== TOP ROW: Individual fits with components =====
    for i,sample_num in enumerate([1,0,2]):
        ax = ax_fits[i]
        sample_mask = scoreset_2018.sample_assignments[:, sample_num]
        sample_name = sample_name_shortener[scoreset_2018.sample_names[sample_num]]
        color = sample_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        hist_data = scoreset_2018.scores[sample_mask]
        n_count = sample_mask.sum()
        
        # Plot histogram for this sample - full visibility for all
        alpha = 0.5
        sns.histplot(
            hist_data,
            binwidth=bin_width,
            stat='density',
            ax=ax,
            alpha=alpha,
            color=color,
        )
        
        density_sample = sample_density(score_range, fits, sample_num)  # shape: (n_bootstraps, n_c, n_scores)
        
        # Plot sum of components (total fit) in black - solid line
        d_total = np.nansum(density_sample, axis=1)  # shape: (n_bootstraps, n_scores)
        d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)  # shape: (3, n_scores)

        ax.fill_between(score_range, d_total_perc[0], d_total_perc[2], 
                       color='gray', 
                       alpha=0.3)
        ax.plot(score_range, d_total_perc[1], 
               color='black', 
               alpha=0.65,
               linewidth=2)

        fontsize_subtitle = 18
        fontsize_legend = 13
        fontsize_count = 14
        
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('')
        ax.set_ylabel('Density' if i == 0 else '', fontsize=fontsize_subtitle)
        ax.tick_params(axis='both', labelsize=9)
        
        # Add threshold lines to fit panels
        for thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
            ax.axvline(
                thresh_score,
                color=thresh_color,
                linestyle=thresh_ls,
                linewidth=thresh_lw,
                alpha=0.8
            )
        
        # Create legend inside each panel with sample name and count
        
        # create facecolor with alpha baked in
        face_rgba = to_rgba(color, alpha)
        hist_patch = Patch(facecolor=face_rgba, edgecolor='black')#, linewidth=1.2)
        if sample_num == 2:  # gnomAD - add prior
            legend_label = f'{sample_name}\nprior: {indv_summary["prior"]:.3f}\n(n={n_count:,d})'
        else:
            legend_label = f'{sample_name}\n(n={n_count:,d})'
        
        ax.legend([hist_patch], [legend_label], loc='upper right' if sample_num != 0 else 'upper left', fontsize=fontsize_legend, framealpha=0.9)
    
    # ===== SECOND ROW: Combined histogram =====
    sample_handles = []
    
    for sample_num in [1,0,2]:
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = sample_name_shortener[scoreset.sample_names[sample_num]]
        color = sample_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        # For gnomAD (sample 2), use all SNV scores and rename to "All SNVs"
        if sample_num == 2:
            hist_data = all_scores
            display_name = 'All SNVs'
            n_count = len(all_scores)
        else:
            hist_data = scoreset.scores[sample_mask]
            display_name = sample_name
            n_count = sample_mask.sum()
        
        sns.histplot(
            hist_data,
            binwidth=bin_width,
            stat='density',
            ax=ax_hist,
            alpha=alpha,
            color=color,
        )

        # create facecolor with alpha baked in
        face_rgba = to_rgba(color, alpha)
        hist_patch = Patch(facecolor=face_rgba, edgecolor='black')#, linewidth=1.2)
        sample_handles.append((hist_patch, f'{display_name} (n={n_count:,d})'))
    
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_xlabel('')
    ax_hist.set_ylabel('Density', fontsize=fontsize_subtitle)
    # ax_hist.set_title(f'{dataset.split("_")[0]} Functional Score (2025 ClinVar)', fontsize=14, fontweight='bold')
    ax_hist.tick_params(axis='both', labelsize=10)
    
    # Sample legend (upper left)
    sample_legend_handles = [h[0] for h in sample_handles]
    sample_legend_labels = [h[1] for h in sample_handles]
    
    ax_hist.legend(
        sample_legend_handles,
        sample_legend_labels,
        loc='upper left',
        fontsize=fontsize_legend,
    )
    
    # ===== THIRD ROW: Scott et al. calibration =====
    ax_scott.axvspan(x_min, 0, color=strength_color['BS3 Strong'], alpha=0.9)
    ax_scott.axvspan(0, 0.4, color=strength_color['IR'], alpha=0.9)
    ax_scott.axvspan(0.4, x_max, color=strength_color['PS3 Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_0_to_04 = ((all_scores >= 0) & (all_scores < 0.4)).sum()
    count_above_04 = (all_scores >= 0.4).sum()
    ax_scott.text((x_min + 0) / 2, 0.5, f'{count_below_0:,}', ha='center', va='center', color='black', fontsize=fontsize_count)
    ax_scott.text((0 + 0.4) / 2, 0.5, f'{count_0_to_04:,}', ha='center', va='center', color='black', fontsize=fontsize_count)
    ax_scott.text((0.4 + x_max) / 2, 0.5, f'{count_above_04:,}', ha='center', va='center', color='black', fontsize=fontsize_count)
    
    ax_scott.set_xlim(x_min, x_max)
    ax_scott.set_ylim(0, 1)
    ax_scott.set_yticks([])
    # ax_scott.set_ylabel('SNV Count', fontsize=fontsize_subtitle)
    ax_scott.set_xticks([])
    ax_scott.set_title('Scott et al. (2022)', loc='left', pad=3, fontsize=fontsize_subtitle, style='italic')
    
    # ===== FOURTH ROW: Zeiberg et al. calibration =====
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)
    threshold_scores_path_sorted = sorted(threshold_scores_path)
    
    legend_order = ["BS3 Very Strong", "BS3 Strong", "-3", "BS3 Moderate", "BS3 Supporting", "IR", "PS3 Supporting", "PS3 Moderate", "+3", "PS3 Strong", "PS3 Very Strong"]
    intervals = []
    used_strengths = {"BS3 Strong", "IR", "PS3 Strong"}

    evidence_to_standardized_text = {
        "BS3 Very Strong": "-8 (very strong)",
        "BS3 Strong": "-4 (strong)",
        "-3": "-3",
        "BS3 Moderate": "-2 (moderate)",
        "BS3 Supporting": "-1 (supporting)",
        "IR": "0 (indeterminate)",
        "PS3 Supporting": "+1 (supporting)",
        "PS3 Moderate": "+2 (moderate)",
        "+3": "+3",
        "PS3 Strong": "+4 (strong)",
        "PS3 Very Strong": "+8 (very strong)"
    }

    strength_color = {**strength_color, **{v: strength_color[k] for k,v in evidence_to_standardized_text.items()}}
    used_strengths = {evidence_to_standardized_text[strength] for strength in used_strengths}
    
    # if len(threshold_scores_benign_sorted) >= 2 and len(threshold_scores_path_sorted) >= 3:
    for count,i in enumerate(range(len(threshold_scores_benign_sorted)-1, -1, -1)):
        evidence_code = evidence_to_standardized_text[legend_order[4-len(threshold_scores_benign_sorted)+1+count]]
        intervals.append((evidence_code, threshold_scores_benign_sorted[i-1] if i > 0 else x_min,threshold_scores_benign_sorted[i]))
        # used_strengths.add(evidence_code)
    intervals = intervals[::-1]
    intervals.append(("0 (indeterminate)", threshold_scores_benign_sorted[-1], threshold_scores_path_sorted[0])) #IR
    # used_strengths.add("IR")
    for i in range(len(threshold_scores_path_sorted)):
        evidence_code = evidence_to_standardized_text[legend_order[6+i]]
        intervals.append((evidence_code, threshold_scores_path_sorted[i],threshold_scores_path_sorted[i+1] if i < len(threshold_scores_path_sorted)-1 else x_max))
        # used_strengths.add(evidence_code)
    
    for name, start, end in intervals:
        ax_zeiberg.axvspan(start, end, color=strength_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.3:
            ax_zeiberg.text(
                (start + end) / 2, 0.5, f'{count:,}',
                ha='center', va='center',
                fontsize=fontsize_count, color='white' if "very strong" in name else 'black'
            )
    
    ax_zeiberg.set_xlim(x_min, x_max)
    ax_zeiberg.set_ylim(0, 1)
    # ax_zeiberg.set_ylabel('SNV Count', fontsize=fontsize_subtitle)
    ax_zeiberg.set_yticks([])
    ax_zeiberg.set_xlabel('Assay Score', fontsize=fontsize_subtitle)
    ax_zeiberg.tick_params(axis='x', labelsize=10)
    ax_zeiberg.set_title('ExCALIBR', loc='left', pad=3, fontsize=fontsize_subtitle, style='italic')
    
    
    for name, _, _ in intervals:
        used_strengths.add(name)

    legend_order_standardized = [evidence_to_standardized_text[name] for name in legend_order]

    # Build legend using standardized names
    all_legend = [
        Patch(facecolor=strength_color[name], label=name, edgecolor='none')
        for name in legend_order_standardized
        if name in used_strengths
    ]
    
    leg_ax.legend(handles=all_legend, loc='upper center', ncol=len(all_legend), frameon=False, fontsize=fontsize_legend,

            # spacing controls
            columnspacing=0.8,     # horizontal space between columns
            # labelspacing=0.3,      # vertical space between rows
            handletextpad=0.6,     # space between marker and text
            handlelength=1.0,      # length of legend handles
            # borderaxespad=0.2      # space between legend and axes)

    )
    
    # all_legend = [
    #     Patch(facecolor=strength_color[name], label=name, edgecolor='none')
    #     for name in legend_order
    #     if name in used_strengths
    # ]
    
    # leg_ax.legend(handles=all_legend, loc='upper center', ncol=len(all_legend), frameon=False, fontsize=12)
    
    plt.tight_layout()
    
    return fig


from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def plot_four_datasets_publication(dataset_names, dataset_configs, dataset_relax_configs, keep_old_list, figsize=(16, 13.33333), HIDE_THRESHOLDS=False, HIDE_MIXTURE_FITS=False, HIDE_COMPONENT_FITS=True, HIDE_COMPONENT_VARIANCE=True, FIRST_ONLY=False, SHOW_PRIOR=True):
    """
    Create a 2x2 grid of dataset plots for publication.
    Each panel shows all samples for one dataset stacked vertically.
    Formatted to match Yang distance comparison plots.
    
    Parameters:
    -----------
    dataset_names : list of str
        List of 4 dataset names to compare
    dataset_configs : dict
        Dictionary mapping dataset names to (n_c, benign_method) tuples
    keep_old_list : list
        List of keep_old datasets
    figsize : tuple
        Figure size (default: (16, 13.33333))
    
    Returns:
    --------
    matplotlib figure
    """
    
    if len(dataset_names) != 4:
        raise ValueError("Must provide exactly 4 dataset names")
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    # sample_alphas = [0.6, 0.6, 0.3, 0.5]
    sample_alphas = [0.7, 0.7, 0.7, 0.7]
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3))]
    linewidths = [1.25, 1.25, 1.25, 1.25]
    
    panel_letters = ['A', 'B', 'C', 'D']
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create 2x2 grid for main panels
    outer_grid = GridSpec(2, 2, figure=fig, hspace=0.14, wspace=0.10)
    
    for panel_idx, (dataset, letter) in enumerate(zip(dataset_names, panel_letters)):
        if FIRST_ONLY and panel_idx > 0:
            continue
        
        # Load data for this dataset
        try:
            scoreset, indv_summary, fits, score_range, config, n_c, flipped, n_samples = load_dataset_for_plot(
                dataset, dataset_configs, dataset_relax_configs, keep_old_list
            )
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue
        
        # Create subplot grid for this panel's samples
        panel_row = panel_idx // 2
        panel_col = panel_idx % 2
        
        # Use GridSpecFromSubplotSpec for nested grids
        inner_grid = GridSpecFromSubplotSpec(
            n_samples, 1,
            subplot_spec=outer_grid[panel_row, panel_col],
            hspace=0.08
        )
        
        point_ranges = indv_summary['point_ranges']
        
        # Plot each sample
        num_skipped = 0
        for sample_num in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            
            sample_idx = sample_num - num_skipped
            ax = fig.add_subplot(inner_grid[sample_idx, 0])
            
            sample_mask = scoreset.sample_assignments[:, sample_idx]
            sample_name = scoreset.sample_names[sample_num]
            
            if sample_name == "population":
                sample_name = "gnomAD"
            
            # Plot histogram
            sns.histplot(
                scoreset.scores[sample_mask],
                stat='density', ax=ax,
                alpha=sample_alphas[sample_num],
                color=sample_colors[sample_num]
            )
            
            max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0

            if not HIDE_MIXTURE_FITS or not HIDE_COMPONENT_FITS or not HIDE_COMPONENT_VARIANCE:
                density_sample = sample_density(score_range, fits, sample_idx)
            
            if not HIDE_MIXTURE_FITS:
                # Plot fitted density
                d = np.nansum(density_sample, axis=1)
                d_perc = np.percentile(d, [5, 50, 95], axis=0)
                
                ax.plot(score_range, d_perc[1], color='black', alpha=0.5, linewidth=2)
                ax.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)

            if not HIDE_COMPONENT_FITS or not HIDE_COMPONENT_VARIANCE:
                comp_colors = ["#873A47", "#0E465F", "gray"]
                for compNum in range(density_sample.shape[1]):
                    compDensity = density_sample[:,compNum,:]
                    d_perc = np.nanpercentile(compDensity,[5,50,95],axis=0)
                    if not HIDE_COMPONENT_FITS:
                        ax.plot(score_range, d_perc[1], linestyle='--', color=comp_colors[compNum], alpha=sample_alphas[compNum], linewidth=2)
                    if not HIDE_COMPONENT_VARIANCE:
                        ax.fill_between(score_range, d_perc[0], d_perc[2], color=comp_colors[compNum], alpha=sample_alphas[compNum]/2)
                
            # Add threshold lines
            handles = []
            if not HIDE_THRESHOLDS:
                for idx, point_val in enumerate(point_values_to_plot):
                    # Benign threshold
                    for pv, score_ranges in point_ranges.items():
                        if pv == -point_val and score_ranges:
                            for sr in score_ranges:
                                threshold_score = sr[0] if not flipped else sr[1]
                                ax.axvline(
                                    threshold_score, color='b',
                                    linestyle=linestyles[idx],
                                    linewidth=linewidths[idx], alpha=0.7
                                )
                                break
                            break
                    
                    # Pathogenic threshold
                    for pv, score_ranges in point_ranges.items():
                        if pv == point_val and score_ranges:
                            for sr in score_ranges:
                                threshold_score = sr[1] if not flipped else sr[0]
                                ax.axvline(
                                    threshold_score, color='r',
                                    linestyle=linestyles[idx],
                                    linewidth=linewidths[idx], alpha=0.7
                                )
                                break
                            break
                    
                    # Create handle for legend
                    if len(point_ranges.get(point_val, [])) != 0 or len(point_ranges.get(-point_val, [])) != 0:
                        h = mlines.Line2D(
                            [], [], color='gray',
                            linestyle=linestyles[idx],
                            linewidth=linewidths[idx],
                            label=f"±{point_val}"
                        )
                        handles.append(h)
            
            # Title on first sample only - MATCHING YANG PLOT FORMAT
            if sample_idx == 0 and not FIRST_ONLY:
                gene_name = dataset.split('_')[0]
                author_name = dataset.split('_')[1]
                
                # Panel letter on far left (matching Yang plot position)
                ax.text(0.00, 1.23 if panel_idx < 2 else 1.18, f"({letter})",
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='left')
                
                # Gene and author name centered (matching Yang plot)
                ax.set_title(rf"$\mathbfit{{{gene_name}}}$ – {author_name}",
             fontsize=14, fontweight='bold', pad=8)
            
            # X-axis only on last sample
            is_last_sample = (sample_num == len(scoreset.sample_counts) - 1 or
                            (sample_num == len(scoreset.sample_counts) - 2 and
                             scoreset.sample_counts[-1] == 0))
            
            if is_last_sample and panel_idx >= 2:
                ax.set_xlabel("Assay score", fontsize=12)
            else:
                if not is_last_sample:
                    ax.set_xticks([])
                ax.set_xlabel("")

            if panel_idx % 2 == 0:
                ax.set_ylabel("Density", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlim([score_range[0], score_range[-1]])
            
            n_count = sample_mask.sum()
            
            # Create histogram legend handle with SHORT labels
            hist_patch = Patch(
                facecolor=sample_colors[sample_num],
                alpha=0.7, edgecolor='none'
            )
            
            # Short label mapping
            short_labels = {
                "Pathogenic/Likely Pathogenic": "P/LP",
                "Benign/Likely Benign": "B/LB",
                "gnomAD": "gnomAD",
                "Synonymous": "Synonymous"
            }
            
            short_name = short_labels.get(sample_name, sample_name)
            
            if sample_name == "gnomAD" and SHOW_PRIOR:
                hist_label = f'{short_name}\n(n={n_count:,d}, prior={indv_summary["prior"]:.3f})'
            else:
                hist_label = f'{short_name}\n(n={n_count:,d})'

            min_font_size = 10
            
            # Create histogram legend on the left
            hist_legend = ax.legend(
                [hist_patch],
                [hist_label],
                loc='upper left',
                fontsize=min_font_size,
                framealpha=0.8
            )

            ax.add_artist(hist_legend)

            if not HIDE_THRESHOLDS:
                # Add point ranges legend on the right (or "No evidence" text)
                if handles:
                    # Create points legend on the right
                    ax.legend(
                        handles,
                        [h.get_label() for h in handles],
                        loc='upper right',
                        ncol=2 if len(handles) > 3 else 1,
                        fontsize=min_font_size-1,
                        framealpha=0.5,
                        handlelength=2,
                        columnspacing=1.0
                    )
                else:
                    # No evidence thresholds - display "No evidence" text
                    ax.text(0.985, 0.945, 'No evidence',
                           transform=ax.transAxes,
                           fontsize=min_font_size,
                           ha='right', va='top',
                           bbox=dict(boxstyle='square,pad=0.4', 
                                    facecolor='white', 
                                    edgecolor='lightgray',
                                    alpha=0.5))
            
            ax.grid(True, alpha=0.3, axis='both', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
    
    plt.tight_layout()

    return fig


def plot_four_datasets_gmm_scores(dataset_names, dataset_configs, dataset_relax_configs, keep_old_list,
                                   figsize=(16, 13.33333), mode='gnomad_gmm'):
    """
    Like plot_four_datasets_publication but shows only the first dataset with
    histograms and a two-component mixture overlay.  Skew-normal fits and
    thresholds are hidden.

    mode='gnomad_gmm'  (default)
        Fit a 2-component GMM to the gnomAD sample.  For every other sample
        the component means/variances are held fixed and only the mixing
        proportions are re-estimated via EM.

    mode='plp_blb'
        Fit a 2-component GMM to the *pooled* P/LP + B/LB scores.  For every
        sample the mixing proportions are re-estimated via EM with those fixed
        components.

    mode='plp_blb_indep'
        Fit one Gaussian independently to the P/LP sample (component 0, dark
        red) and one to the B/LB sample (component 1, dark blue) using their
        sample means and standard deviations.  For every sample the mixing
        proportions are re-estimated via EM with those fixed components.
    """
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm as sp_norm

    if len(dataset_names) != 4:
        raise ValueError("Must provide exactly 4 dataset names")
    if mode not in ('gnomad_gmm', 'plp_blb', 'plp_blb_indep'):
        raise ValueError("mode must be 'gnomad_gmm', 'plp_blb', or 'plp_blb_indep'")

    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    sample_alphas = [0.7, 0.7, 0.7, 0.7]
    comp_colors = ["#873A47", "#0E465F"]  # component 0 = pathogenic-like, 1 = benign-like

    panel_letters = ['A', 'B', 'C', 'D']

    fig = plt.figure(figsize=figsize)
    outer_grid = GridSpec(2, 2, figure=fig, hspace=0.14, wspace=0.10)

    def _update_weights(scores, means, stds, n_iter=200):
        """EM with fixed component parameters; returns mixing proportions."""
        n_comp = len(means)
        w = np.ones(n_comp) / n_comp
        s = scores.flatten()
        for _ in range(n_iter):
            resp = np.column_stack([w[k] * sp_norm.pdf(s, means[k], stds[k]) for k in range(n_comp)])
            row_sums = resp.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-300, row_sums)
            resp /= row_sums
            w = resp.mean(axis=0)
            w = np.maximum(w, 1e-10)
            w /= w.sum()
        return w

    def _col_idx_for_sample_num(scoreset, target_num):
        """Return the sample_assignments column index for a given sample_num."""
        col = 0
        for i in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[i] == 0:
                continue
            if i == target_num:
                return col
            col += 1
        return None

    for panel_idx, (dataset, letter) in enumerate(zip(dataset_names, panel_letters)):
        if panel_idx > 0:
            continue

        try:
            scoreset, indv_summary, fits, score_range, config, n_c, flipped, n_samples = load_dataset_for_plot(
                dataset, dataset_configs, dataset_relax_configs, keep_old_list
            )
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

        # --- derive component parameters depending on mode ---
        gmm_means = gmm_stds = None

        if mode == 'gnomad_gmm':
            gnomad_col_idx = None
            col_idx = 0
            for i in range(len(scoreset.sample_counts)):
                if scoreset.sample_counts[i] == 0:
                    continue
                name = scoreset.sample_names[i].lower()
                if "population" in name or "gnomad" in name:
                    gnomad_col_idx = col_idx
                    break
                col_idx += 1

            if gnomad_col_idx is None:
                print(f"Could not find gnomAD sample for {dataset}; skipping GMM")
            else:
                gnomad_scores = scoreset.scores[scoreset.sample_assignments[:, gnomad_col_idx]].reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
                gmm.fit(gnomad_scores)
                gmm_means = gmm.means_.flatten()
                gmm_stds = np.sqrt(gmm.covariances_[:, 0, 0])

        elif mode == 'plp_blb':
            plp_col = _col_idx_for_sample_num(scoreset, 0)
            blb_col = _col_idx_for_sample_num(scoreset, 3)
            if plp_col is None or blb_col is None:
                print(f"Could not find P/LP or B/LB sample for {dataset}; skipping")
            else:
                plp_scores = scoreset.scores[scoreset.sample_assignments[:, plp_col]]
                blb_scores = scoreset.scores[scoreset.sample_assignments[:, blb_col]]
                combined = np.concatenate([plp_scores, blb_scores]).reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=10)
                gmm.fit(combined)
                gmm_means = gmm.means_.flatten()
                gmm_stds  = np.sqrt(gmm.covariances_[:, 0, 0])

        elif mode == 'plp_blb_indep':
            plp_col = _col_idx_for_sample_num(scoreset, 0)
            blb_col = _col_idx_for_sample_num(scoreset, 3)
            if plp_col is None or blb_col is None:
                print(f"Could not find P/LP or B/LB sample for {dataset}; skipping")
            else:
                plp_scores = scoreset.scores[scoreset.sample_assignments[:, plp_col]]
                blb_scores = scoreset.scores[scoreset.sample_assignments[:, blb_col]]
                gmm_means = np.array([plp_scores.mean(), blb_scores.mean()])
                gmm_stds  = np.array([plp_scores.std(),  blb_scores.std()])

        panel_row = panel_idx // 2
        panel_col = panel_idx % 2

        inner_grid = GridSpecFromSubplotSpec(
            n_samples, 1,
            subplot_spec=outer_grid[panel_row, panel_col],
            hspace=0.08
        )

        x_plot = np.array(score_range)

        num_skipped = 0
        for sample_num in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue

            sample_idx = sample_num - num_skipped
            ax = fig.add_subplot(inner_grid[sample_idx, 0])

            sample_mask = scoreset.sample_assignments[:, sample_idx]
            sample_name = scoreset.sample_names[sample_num]
            if sample_name == "population":
                sample_name = "gnomAD"

            sns.histplot(
                scoreset.scores[sample_mask],
                stat='density', ax=ax,
                alpha=sample_alphas[sample_num],
                color=sample_colors[sample_num]
            )

            # overlay mixture with sample-specific mixing proportions
            if gmm_means is not None:
                sample_scores = scoreset.scores[sample_mask]
                w = _update_weights(sample_scores, gmm_means, gmm_stds)

                mixture = np.zeros(len(x_plot))
                for k in range(2):
                    comp = w[k] * sp_norm.pdf(x_plot, gmm_means[k], gmm_stds[k])
                    mixture += comp
                    ax.plot(x_plot, comp, linestyle='--', color=comp_colors[k], alpha=sample_alphas[k], linewidth=2)
                # ax.plot(x_plot, mixture, color='black', alpha=0.5, linewidth=2)

            # title on first sample only
            if sample_idx == 0:
                gene_name = dataset.split('_')[0]
                author_name = dataset.split('_')[1]
                ax.set_title(rf"$\mathbfit{{{gene_name}}}$ – {author_name}",
                             fontsize=14, fontweight='bold', pad=8)

            is_last_sample = (sample_num == len(scoreset.sample_counts) - 1 or
                              (sample_num == len(scoreset.sample_counts) - 2 and
                               scoreset.sample_counts[-1] == 0))

            if is_last_sample:
                ax.set_xlabel("Assay score", fontsize=12)
            else:
                ax.set_xticks([])
                ax.set_xlabel("")

            ax.set_ylabel("Density", fontsize=12)
            ax.set_xlim([score_range[0], score_range[-1]])

            n_count = sample_mask.sum()
            hist_patch = Patch(facecolor=sample_colors[sample_num], alpha=0.7, edgecolor='none')
            short_labels = {
                "Pathogenic/Likely Pathogenic": "P/LP",
                "Benign/Likely Benign": "B/LB",
                "gnomAD": "gnomAD",
                "Synonymous": "Synonymous",
            }
            short_name = short_labels.get(sample_name, sample_name)
            hist_label = f'{short_name}\n(n={n_count:,d})'

            ax.legend([hist_patch], [hist_label], loc='upper left', fontsize=10, framealpha=0.8)

            ax.grid(True, alpha=0.3, axis='both', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig


def load_dataset_for_plot(dataset, dataset_configs, dataset_relax_configs, keep_old_list):
    """
    Load data for a single dataset (extracted from plot_for_pub logic).
    
    Returns:
    --------
    scoreset, indv_summary, fits, score_range, config, n_c, flipped
    """
    
    dataset = dataset.replace("_NEW_not_clinvar_2018", "").replace("CHK2_Gebbia_2024", "CHEK2_Gebbia_2024")
    
    keep_old = dataset in keep_old_list
    mode = "point_assignment_replicate_run"
    relax = True
    
    if keep_old:
        if dataset in dataset_relax_configs:
            relax = True
            mode = "point_assignment_semifinal_rerun"
        else:
            relax = False
            mode = "point_assignment_comparison"
            if dataset.endswith('_clinvar_2018'):
                mode = "point_assignment_clinvar_2018"
    
    use_median_prior, use_2c_equation = True, False
    config = dataset_configs[dataset]
    n_c, benign_method = config
    
    points_save_dir = f'/data/ross/assay_calibration/{mode}/{dataset}'
    experiment_code = f'{dataset}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
    pkl_filepath = f'{points_save_dir}/{experiment_code}.pkl'
    
    # Try fallback paths
    fallback_modes = ["point_assignment_semifinal_rerun", "point_assignment_comparison", "point_assignment_clinvar_2018"]
    if not os.path.exists(pkl_filepath):
        for fallback_mode in fallback_modes:
            test_path = pkl_filepath.replace("point_assignment_replicate_run", fallback_mode)
            if os.path.exists(test_path):
                pkl_filepath = test_path
                break
    
    # Try without _avg
    if not os.path.exists(pkl_filepath):
        pkl_filepath = pkl_filepath.replace("_avg", "")
        benign_method = "benign"
        for fallback_mode in fallback_modes:
            test_path = pkl_filepath.replace("point_assignment_replicate_run", fallback_mode)
            if os.path.exists(test_path):
                pkl_filepath = test_path
                break
    
    if not os.path.exists(pkl_filepath):
        raise FileNotFoundError(f"Could not find pkl file for {dataset}")
    
    # Check if flipped
    log_f = pkl_filepath.replace('.pkl', '.log')
    scoreset_flipped = False
    with open(log_f, 'r') as f:
        for line in f:
            if line.strip() == "scoreset_flipped: True":
                scoreset_flipped = True
    
    # Load data
    with open(pkl_filepath, 'rb') as f:
        scoreset, indv_summary, fits, score_range, _, n_c = pickle.load(f)

    n_samples = len([s for s in scoreset.samples])
    
    return scoreset, indv_summary, fits, score_range, config, n_c, scoreset_flipped, n_samples
    


def compute_classification_metrics(confusion_matrix):
    """
    Compute classification performance metrics from a confusion matrix.
    
    Parameters:
    -----------
    confusion_matrix : DataFrame
        Confusion matrix with rows = ClinVar (B/LB, P/LP)
                         and columns = Evidence categories (Benign, Indeterminate, Pathogenic)
                         OR Functional categories (Normal, Indeterminate, Abnormal)
    
    Returns:
    --------
    dict with metrics: accuracy, sensitivity, specificity, MCC, DOR (determinate only),
                       LR+ and DOR for dichotomized predictions (including uncertain)
    """
    
    # Get counts
    if confusion_matrix.shape[0] != 2 or confusion_matrix.shape[1] != 3:
        raise ValueError(f"Expected 2×3 confusion matrix, got {confusion_matrix.shape}")
    
    # Row 0 = B/LB (negatives), Row 1 = P/LP (positives)
    # Col 0 = Benign/Normal, Col 1 = Indeterminate, Col 2 = Pathogenic/Abnormal
    
    TN = confusion_matrix.iloc[0, 0]  # B/LB → Benign/Normal
    FP = confusion_matrix.iloc[0, 2]  # B/LB → Pathogenic/Abnormal
    FN = confusion_matrix.iloc[1, 0]  # P/LP → Benign/Normal
    TP = confusion_matrix.iloc[1, 2]  # P/LP → Pathogenic/Abnormal
    
    uncertain_benign = confusion_matrix.iloc[0, 1]  # B/LB → Indeterminate
    uncertain_path = confusion_matrix.iloc[1, 1]    # P/LP → Indeterminate
    
    total_variants = confusion_matrix.sum().sum()
    determinate_calls = TP + TN + FP + FN
    
    # Compute metrics
    coverage = determinate_calls / total_variants if total_variants > 0 else 0.0
    
    # ========================================================================
    # STANDARD METRICS (determinate calls only, excluding uncertain)
    # ========================================================================
    if determinate_calls > 0:
        accuracy = (TP + TN) / determinate_calls
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        
        # Matthews Correlation Coefficient
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = numerator / denominator if denominator > 0 else 0.0
        
        # Standard DOR (on determinate calls only)
        dor_standard = (TP * TN) / (FP * FN) if (FP > 0 and FN > 0) else float('inf')
        
        # Standard LR+ (on determinate calls only)
        lr_plus_standard = sensitivity / (1 - specificity) if specificity < 1.0 else float('inf')
        
    else:
        accuracy = sensitivity = specificity = mcc = 0.0
        dor_standard = lr_plus_standard = 0.0
    
    # ========================================================================
    # DICHOTOMIZED METRICS (including uncertain variants)
    # ========================================================================
    
    # For PATHOGENIC prediction: Pathogenic vs. non-Pathogenic (Benign + Uncertain)
    TP_path_dich = TP                                    # P/LP → Pathogenic
    FN_path_dich = FN + uncertain_path                   # P/LP → (Benign or Uncertain)
    FP_path_dich = FP                                    # B/LB → Pathogenic
    TN_path_dich = TN + uncertain_benign                 # B/LB → (Benign or Uncertain)
    
    sens_path_dich = TP_path_dich / (TP_path_dich + FN_path_dich) if (TP_path_dich + FN_path_dich) > 0 else 0.0
    spec_path_dich = TN_path_dich / (TN_path_dich + FP_path_dich) if (TN_path_dich + FP_path_dich) > 0 else 0.0
    
    lr_plus_pathogenic = sens_path_dich / (1 - spec_path_dich) if spec_path_dich < 1.0 else float('inf')
    dor_pathogenic = (TP_path_dich * TN_path_dich) / (FP_path_dich * FN_path_dich) if (FP_path_dich > 0 and FN_path_dich > 0) else float('inf')
    
    # For BENIGN prediction: Benign vs. non-Benign (Pathogenic + Uncertain)
    TP_ben_dich = TN                                     # B/LB → Benign
    FN_ben_dich = FP + uncertain_benign                  # B/LB → (Pathogenic or Uncertain)
    FP_ben_dich = FN                                     # P/LP → Benign
    TN_ben_dich = TP + uncertain_path                    # P/LP → (Pathogenic or Uncertain)
    
    sens_ben_dich = TP_ben_dich / (TP_ben_dich + FN_ben_dich) if (TP_ben_dich + FN_ben_dich) > 0 else 0.0
    spec_ben_dich = TN_ben_dich / (TN_ben_dich + FP_ben_dich) if (TN_ben_dich + FP_ben_dich) > 0 else 0.0
    
    lr_plus_benign = sens_ben_dich / (1 - spec_ben_dich) if spec_ben_dich < 1.0 else float('inf')
    dor_benign = (TP_ben_dich * TN_ben_dich) / (FP_ben_dich * FN_ben_dich) if (FP_ben_dich > 0 and FN_ben_dich > 0) else float('inf')
    
    return {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'uncertain': int(uncertain_benign + uncertain_path),
        'total': int(total_variants),
        'determinate': int(determinate_calls),
        'coverage': float(coverage),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'mcc': float(mcc),
        'dor_standard': float(dor_standard),
        'lr_plus_standard': float(lr_plus_standard),
        'lr_plus_pathogenic': float(lr_plus_pathogenic),
        'lr_plus_benign': float(lr_plus_benign),
        'dor_pathogenic': float(dor_pathogenic),
        'dor_benign': float(dor_benign),
    }

def compute_aggregate_metrics(danzs, auths, dataset_names):
    """
    Compute aggregate metrics across all datasets.
    
    Parameters:
    -----------
    danzs : list of DataFrames
        List of DanZ confusion matrices
    auths : list of DataFrames
        List of author confusion matrices
    dataset_names : list of str
        List of dataset names
    
    Returns:
    --------
    tuple: (danz_aggregate_metrics, auth_aggregate_metrics, individual_metrics_df)
    """
    
    # Aggregate confusion matrices
    danz_aggregate = None
    auth_aggregate = None
    
    danz_metrics_list = []
    auth_metrics_list = []
    
    for i, (danz_df, auth_df, dataset_name) in enumerate(zip(danzs, auths, dataset_names)):
        if danz_df is None or auth_df is None:
            continue

        if danz_aggregate is None:
            danz_aggregate = danz_df.copy()
            auth_aggregate = auth_df.copy()
        else:
            danz_aggregate += danz_df
            auth_aggregate += auth_df
        
        # Compute individual metrics
        danz_metrics = compute_classification_metrics(danz_df)
        auth_metrics = compute_classification_metrics(auth_df)
        
        danz_metrics['dataset'] = dataset_name
        auth_metrics['dataset'] = dataset_name
        
        danz_metrics_list.append(danz_metrics)
        auth_metrics_list.append(auth_metrics)
    
    # Compute aggregate metrics
    danz_aggregate_metrics = compute_classification_metrics(danz_aggregate)
    auth_aggregate_metrics = compute_classification_metrics(auth_aggregate)
    
    # Create DataFrame for individual metrics
    individual_df = pd.DataFrame({
        'dataset': [m['dataset'] for m in danz_metrics_list],
        'danz_accuracy': [m['accuracy'] for m in danz_metrics_list],
        'danz_sensitivity': [m['sensitivity'] for m in danz_metrics_list],
        'danz_specificity': [m['specificity'] for m in danz_metrics_list],
        'danz_mcc': [m['mcc'] for m in danz_metrics_list],
        'danz_coverage': [m['coverage'] for m in danz_metrics_list],
        'auth_accuracy': [m['accuracy'] for m in auth_metrics_list],
        'auth_sensitivity': [m['sensitivity'] for m in auth_metrics_list],
        'auth_specificity': [m['specificity'] for m in auth_metrics_list],
        'auth_mcc': [m['mcc'] for m in auth_metrics_list],
        'auth_coverage': [m['coverage'] for m in auth_metrics_list],
    })
    
    return danz_aggregate_metrics, auth_aggregate_metrics, individual_df


def plot_individual_confusion_matrices_with_metrics(dataset_names, danzs, auths, dataset_configs, 
                                                     figsize_per_row=(14, 4)):
    """
    Plot confusion matrices with performance metrics for individual datasets.
    Each row shows one dataset with DanZ (left) and Author (right) annotations.
    
    Parameters:
    -----------
    dataset_names : list of str
        List of dataset names
    danzs : list of DataFrames
        List of DanZ count matrices
    auths : list of DataFrames
        List of author count matrices
    dataset_configs : dict
        Dataset configurations for title info
    figsize_per_row : tuple
        Figure size per dataset row (width, height)
    
    Returns:
    --------
    matplotlib figure, metrics DataFrame
    """
    
    # Filter valid datasets
    valid_indices = []
    for i in range(len(danzs)):
        if danzs[i] is None or auths[i] is None:
            continue
        
        danz_row_sums = danzs[i].sum(axis=1)
        auth_row_sums = auths[i].sum(axis=1)
        
        if (danz_row_sums == 0).any() or (auth_row_sums == 0).any():
            continue
        
        valid_indices.append(i)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid datasets!")
    
    n_datasets = len(valid_indices)
    
    # Create figure with subplots: n_datasets rows × 2 columns
    fig, axes = plt.subplots(
        n_datasets, 2, 
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_datasets)
    )
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme similar to the reference image
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#F0F0F0", "#8FBC8F", "#4682B4", "#1E3A5F"]  # Light gray to dark blue
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    metrics_list = []
    
    for row_idx, data_idx in enumerate(valid_indices):
        dataset_name = dataset_names[data_idx]
        danz_df = danzs[data_idx]
        auth_df = auths[data_idx]
        
        # Compute metrics
        danz_metrics = compute_classification_metrics(danz_df)
        auth_metrics = compute_classification_metrics(auth_df)
        
        metrics_list.append({
            'dataset': dataset_name,
            **{f'danz_{k}': v for k, v in danz_metrics.items()},
            **{f'auth_{k}': v for k, v in auth_metrics.items()}
        })
        
        # Get gene and author for title
        gene_name = dataset_name.split('_')[0]
        author_name = dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ""
        
        # Plot DanZ (left panel)
        ax_danz = axes[row_idx, 0]
        
        # Get max value for consistent color scale across both panels
        max_val = max(danz_df.values.max(), auth_df.values.max())
        
        sns.heatmap(
            danz_df,
            annot=True,
            fmt='d',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_danz,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Title with metrics
        metrics_text = (f"Acc: {danz_metrics['accuracy']:.2f}, "
                       f"Sens: {danz_metrics['sensitivity']:.2f}, "
                       f"Spec: {danz_metrics['specificity']:.2f}, "
                       f"MCC: {danz_metrics['mcc']:.2f}\n"
                       f"Coverage: {danz_metrics['coverage']:.2f} "
                       f"({danz_metrics['determinate']}/{danz_metrics['total']})")
        
        ax_danz.set_title(
            f"({panel_letters[row_idx*2]}) {gene_name} \u2013 {author_name}\nVariant-level Annotations\n{metrics_text}",
            fontsize=10,
            fontweight='bold',
            pad=10
        )
        
        ax_danz.set_xlabel('Evidence Direction', fontsize=10)
        ax_danz.set_ylabel('ClinVar Classification', fontsize=10)
        ax_danz.tick_params(labelsize=9)
        
        # Rotate x-axis labels
        ax_danz.set_xticklabels(ax_danz.get_xticklabels(), rotation=45, ha='right')
        ax_danz.set_yticklabels(ax_danz.get_yticklabels(), rotation=0)
        
        # Plot Author (right panel)
        ax_auth = axes[row_idx, 1]
        
        sns.heatmap(
            auth_df,
            annot=True,
            fmt='d',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_auth,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Title with metrics
        metrics_text = (f"Acc: {auth_metrics['accuracy']:.2f}, "
                       f"Sens: {auth_metrics['sensitivity']:.2f}, "
                       f"Spec: {auth_metrics['specificity']:.2f}, "
                       f"MCC: {auth_metrics['mcc']:.2f}\n"
                       f"Coverage: {auth_metrics['coverage']:.2f} "
                       f"({auth_metrics['determinate']}/{auth_metrics['total']})")
        
        ax_auth.set_title(
            f"({panel_letters[row_idx*2 + 1]}) {gene_name} \u2013 {author_name}\nAuthor Annotations\n{metrics_text}",
            fontsize=10,
            fontweight='bold',
            pad=10
        )
        
        ax_auth.set_xlabel('Functional Annotation', fontsize=10)
        ax_auth.set_ylabel('ClinVar Classification', fontsize=10)
        ax_auth.tick_params(labelsize=9)
        
        # Rotate x-axis labels
        ax_auth.set_xticklabels(ax_auth.get_xticklabels(), rotation=45, ha='right')
        ax_auth.set_yticklabels(ax_auth.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    return fig, metrics_df


def print_aggregate_performance(danzs, auths, dataset_names):
    """
    Print aggregate performance statistics across all datasets.
    
    Parameters:
    -----------
    danzs : list of DataFrames
        List of DanZ confusion matrices
    auths : list of DataFrames
        List of author confusion matrices
    dataset_names : list of str
        List of dataset names
    """
    
    danz_agg, auth_agg, individual_df = compute_aggregate_metrics(danzs, auths, dataset_names)
    
    print("="*80)
    print("AGGREGATE PERFORMANCE METRICS")
    print("="*80)
    
    print("\n--- Variant-level Annotations (DanZ) ---")
    print(f"Total variants: {danz_agg['total']:,}")
    print(f"Determinate calls: {danz_agg['determinate']:,} ({danz_agg['coverage']:.1%})")
    print(f"  TP (P/LP → Pathogenic): {danz_agg['TP']:,}")
    print(f"  TN (B/LB → Benign): {danz_agg['TN']:,}")
    print(f"  FP (B/LB → Pathogenic): {danz_agg['FP']:,}")
    print(f"  FN (P/LP → Benign): {danz_agg['FN']:,}")
    print(f"  Uncertain: {danz_agg['uncertain']:,}")
    print(f"\nPerformance:")
    print(f"  Accuracy:    {danz_agg['accuracy']:.3f} ({danz_agg['accuracy']:.1%})")
    print(f"  Sensitivity: {danz_agg['sensitivity']:.3f} ({danz_agg['sensitivity']:.1%})")
    print(f"  Specificity: {danz_agg['specificity']:.3f} ({danz_agg['specificity']:.1%})")
    print(f"  MCC:         {danz_agg['mcc']:.3f}")
    print(f"  LR+ standard:         {danz_agg['lr_plus_standard']:.3f}")
    print(f"  LR+ pathogenic:         {danz_agg['lr_plus_pathogenic']:.3f}")
    print(f"  LR+ benign:         {danz_agg['lr_plus_benign']:.3f}")
    print(f"  DOR standard:         {danz_agg['dor_standard']:.3f}")
    print(f"  DOR pathogenic:         {danz_agg['dor_pathogenic']:.3f}")
    print(f"  DOR benign:         {danz_agg['dor_benign']:.3f}")
    
    print("\n--- Author Annotations ---")
    print(f"Total variants: {auth_agg['total']:,}")
    print(f"Determinate calls: {auth_agg['determinate']:,} ({auth_agg['coverage']:.1%})")
    print(f"  TP (P/LP → Abnormal): {auth_agg['TP']:,}")
    print(f"  TN (B/LB → Normal): {auth_agg['TN']:,}")
    print(f"  FP (B/LB → Abnormal): {auth_agg['FP']:,}")
    print(f"  FN (P/LP → Normal): {auth_agg['FN']:,}")
    print(f"  Uncertain: {auth_agg['uncertain']:,}")
    print(f"\nPerformance:")
    print(f"  Accuracy:    {auth_agg['accuracy']:.3f} ({auth_agg['accuracy']:.1%})")
    print(f"  Sensitivity: {auth_agg['sensitivity']:.3f} ({auth_agg['sensitivity']:.1%})")
    print(f"  Specificity: {auth_agg['specificity']:.3f} ({auth_agg['specificity']:.1%})")
    print(f"  MCC:         {auth_agg['mcc']:.3f}")
    print(f"  LR+ standard:         {auth_agg['lr_plus_standard']:.3f}")
    print(f"  LR+ pathogenic:         {auth_agg['lr_plus_pathogenic']:.3f}")
    print(f"  LR+ benign:         {auth_agg['lr_plus_benign']:.3f}")
    print(f"  DOR standard:         {auth_agg['dor_standard']:.3f}")
    print(f"  DOR pathogenic:         {auth_agg['dor_pathogenic']:.3f}")
    print(f"  DOR benign:         {auth_agg['dor_benign']:.3f}")
    
    print("\n" + "="*80)
    print("PER-DATASET METRICS SUMMARY")
    print("="*80)
    
    print("\n--- DanZ Performance (Mean ± SD across datasets) ---")
    print(f"  Accuracy:    {individual_df['danz_accuracy'].mean():.3f} ± {individual_df['danz_accuracy'].std():.3f}")
    print(f"  Sensitivity: {individual_df['danz_sensitivity'].mean():.3f} ± {individual_df['danz_sensitivity'].std():.3f}")
    print(f"  Specificity: {individual_df['danz_specificity'].mean():.3f} ± {individual_df['danz_specificity'].std():.3f}")
    print(f"  MCC:         {individual_df['danz_mcc'].mean():.3f} ± {individual_df['danz_mcc'].std():.3f}")
    print(f"  Coverage:    {individual_df['danz_coverage'].mean():.3f} ± {individual_df['danz_coverage'].std():.3f}")
    
    print("\n--- Author Performance (Mean ± SD across datasets) ---")
    print(f"  Accuracy:    {individual_df['auth_accuracy'].mean():.3f} ± {individual_df['auth_accuracy'].std():.3f}")
    print(f"  Sensitivity: {individual_df['auth_sensitivity'].mean():.3f} ± {individual_df['auth_sensitivity'].std():.3f}")
    print(f"  Specificity: {individual_df['auth_specificity'].mean():.3f} ± {individual_df['auth_specificity'].std():.3f}")
    print(f"  MCC:         {individual_df['auth_mcc'].mean():.3f} ± {individual_df['auth_mcc'].std():.3f}")
    print(f"  Coverage:    {individual_df['auth_coverage'].mean():.3f} ± {individual_df['auth_coverage'].std():.3f}")
    
    return danz_agg, auth_agg, individual_df


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap



def plot_aggregate_confusion_matrices(danzs, auths, dataset_names, figsize=(13, 4), letters=True):
    """
    Plot aggregate confusion matrices with conditional color scheme.
    If letters=True: Blue/Gray/Red diverging colors by column, row-normalized intensity
    If letters=False: Uniform purple gradient
    """
    
    # Aggregate confusion matrices
    danz_aggregate = None
    auth_aggregate = None
    n_datasets = 0
    
    for i, (danz_df, auth_df, dataset_name) in enumerate(zip(danzs, auths, dataset_names)):
        if danz_df is None or auth_df is None:
            continue
        
        if danz_aggregate is None:
            danz_aggregate = danz_df.copy()
            auth_aggregate = auth_df.copy()
        else:
            danz_aggregate += danz_df
            auth_aggregate += auth_df

        n_datasets += 1

    if danz_aggregate is None:
        raise ValueError("No valid datasets to aggregate!")

    print(f"Aggregated {n_datasets} datasets")

    # Compute metrics with DORs
    danz_metrics = compute_classification_metrics(danz_aggregate)
    auth_metrics = compute_classification_metrics(auth_aggregate)
    
    # Create figure with manual positioning
    fig = plt.figure(figsize=figsize)
    left_margin = 0.08
    right_margin = 0.02
    bottom_margin = 0.15
    top_margin = 0.12
    
    plot_width = 0.35
    cbar_width = 0.02
    space_left = 0.05
    space_right = 0.025
    
    ax_danz_pos = [left_margin, bottom_margin, plot_width, 1 - bottom_margin - top_margin]
    ax_auth_pos = [left_margin + plot_width + space_left, bottom_margin, plot_width, 1 - bottom_margin - top_margin]
    cbar_pos = [left_margin + 2*plot_width + space_left + space_right, bottom_margin, cbar_width, 1 - bottom_margin - top_margin]
    
    ax_danz = fig.add_axes(ax_danz_pos)
    ax_auth = fig.add_axes(ax_auth_pos)
    if not letters:
        cbar_ax = fig.add_axes(cbar_pos)
    
    max_val = max(danz_aggregate.values.max(), auth_aggregate.values.max())
    
    if letters:
        # Diverging color scheme - Blue/Gray/Red by column, row-normalized
        blue_colors = ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E']
        blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", blue_colors)
        
        red_colors = ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744']
        red_cmap = LinearSegmentedColormap.from_list("red_gradient", red_colors)
        
        gray_colors = ['#F5F5F5', '#CCCCCC', '#999999', '#666666']
        gray_cmap = LinearSegmentedColormap.from_list("gray_gradient", gray_colors)
        
        def plot_heatmap(df, ax):
            """Plot heatmap with column colors but row-based normalization"""
            colors = np.zeros((len(df), len(df.columns), 4))
            
            # First pass: compute colors with row-based normalization
            for i in range(len(df)):
                row_max = df.iloc[i].max()
                
                for j, col_name in enumerate(df.columns):
                    # Determine colormap based on column
                    if 'Benign' in str(col_name) or 'Normal' in str(col_name) or 'BLB' in str(col_name) or 'B/LB' in str(col_name):
                        cmap = blue_cmap
                    elif 'IR' in str(col_name) or 'Indeterminate' in str(col_name):
                        cmap = gray_cmap
                    elif 'Pathogenic' in str(col_name) or 'Abnormal' in str(col_name) or 'PLP' in str(col_name) or 'P/LP' in str(col_name):
                        cmap = red_cmap
                    else:
                        cmap = gray_cmap
                    
                    value = df.iloc[i, j]
                    normalized = value / row_max if row_max > 0 else 0
                    colors[i, j] = cmap(normalized)
            
            # Second pass: plot cells
            for i in range(len(df)):
                row_max = df.iloc[i].max()
                
                for j in range(len(df.columns)):
                    rect = mpatches.Rectangle((j, i), 1, 1, 
                                             facecolor=colors[i, j],
                                             edgecolor='white',
                                             linewidth=2.5)
                    ax.add_patch(rect)
                    
                    value = df.iloc[i, j]
                    # Text color based on row max
                    text_color = 'white' if value / row_max > 0.45 else 'black'
                    if 'IR' in str(df.columns[j]) or 'Indeterminate' in str(df.columns[j]):
                        text_color = 'black'
                    
                    ax.text(j + 0.5, i + 0.5, f'{value:,}',
                           ha='center', va='center',
                           fontsize=14, color=text_color)
            
            ax.set_xlim(0, len(df.columns))
            ax.set_ylim(0, len(df))
            ax.invert_yaxis()
            ax.set_aspect('equal')
        
        # Plot with diverging colors
        plot_heatmap(danz_aggregate, ax_danz)
        plot_heatmap(auth_aggregate, ax_auth)
        
    else:
        # Uniform purple gradient
        colors = ["whitesmoke", "purple"]
        cmap = LinearSegmentedColormap.from_list("nature_purple", colors)
        
        def get_text_color(value, max_value):
            normalized = value / max_value
            return 'white' if normalized > 0.45 else 'black'
        
        def create_annot(df):
            annot = df.copy().astype(str)
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    annot.iloc[row, col] = f"{df.iloc[row, col]:,}"
            return annot
        
        danz_annot = create_annot(danz_aggregate)
        auth_annot = create_annot(auth_aggregate)
        
        # Plot with seaborn
        sns.heatmap(
            danz_aggregate,
            annot=danz_annot,
            fmt='',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_danz,
            cbar=False,
            linewidths=2.5,
            linecolor='white',
            annot_kws={'fontsize': 14, 'ha': 'center', 'va': 'center'}
        )
        
        for text_obj in ax_danz.texts:
            x, y = text_obj.get_position()
            row, col = int(y), int(x)
            if row < len(danz_aggregate) and col < len(danz_aggregate.columns):
                value = danz_aggregate.iloc[row, col]
                text_obj.set_color(get_text_color(value, max_val))
        
        sns.heatmap(
            auth_aggregate,
            annot=auth_annot,
            fmt='',
            cmap=cmap,
            vmin=0,
            vmax=max_val,
            ax=ax_auth,
            cbar_ax=cbar_ax,
            cbar_kws={'label': 'Count'},
            linewidths=2.5,
            linecolor='white',
            annot_kws={'fontsize': 14, 'ha': 'center', 'va': 'center'}
        )
        
        for text_obj in ax_auth.texts:
            x, y = text_obj.get_position()
            row, col = int(y), int(x)
            if row < len(auth_aggregate) and col < len(auth_aggregate.columns):
                value = auth_aggregate.iloc[row, col]
                text_obj.set_color(get_text_color(value, max_val))
    
    # Common styling for both modes
    ax_danz.set_facecolor('#F9F9F9')
    ax_auth.set_facecolor('#F9F9F9')
    
    if letters:
        ax_danz.text(-0.10, 1.11, "(A)", transform=ax_danz.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left')
        ax_auth.text(-0.10, 1.11, "(B)", transform=ax_auth.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left')

    if letters:
        ax_danz.set_title("ExCALIBR Evidence",
                         fontsize=18, fontweight='bold', pad=10)
        ax_auth.set_title("Author Annotations", fontsize=18, fontweight='bold', pad=10)
    
    # Labels
    xlabels = list(danz_aggregate.columns) if letters else [t.get_text() for t in ax_danz.get_xticklabels()]
    ylabels = list(danz_aggregate.index) if letters else [t.get_text() for t in ax_danz.get_yticklabels()]
    label_map = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indeterminate",
                 "Normal": "Benign", "Abnormal": "Pathogenic"}
    xlabels = [label_map.get(str(x), str(x)) for x in xlabels]
    ylabels = [label_map.get(str(y), str(y)) for y in ylabels]
    
    ax_danz.set_xlabel('Evidence Direction', fontsize=14)
    ax_danz.set_ylabel('ClinVar Classification', fontsize=14)
    
    if letters:
        ax_danz.set_xticks(np.arange(len(xlabels)) + 0.5)
        ax_danz.set_yticks(np.arange(len(ylabels)) + 0.5)
    
    ax_danz.set_xticklabels(xlabels, rotation=0, fontsize=12)
    ax_danz.set_yticklabels(ylabels, rotation=0, fontsize=12)
    ax_danz.tick_params(length=0, labelsize=12)
    
    for spine in ax_danz.spines.values():
        spine.set_visible(False)

    if not letters:
        ax_danz.text(0.5, -0.18, 
                    f"DOR: {danz_metrics['dor_standard']:.1f}  |  Determinate: Controls {100*danz_metrics['coverage']:.1f}%, VUS 79.7%",
                    transform=ax_danz.transAxes,
                    fontsize=12, ha='center', va='top',
                    color='#555555')
    
    # Author panel
    xlabels = list(auth_aggregate.columns) if letters else [t.get_text() for t in ax_auth.get_xticklabels()]
    ylabels = list(auth_aggregate.index) if letters else [t.get_text() for t in ax_auth.get_yticklabels()]
    label_map_auth = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indeterminate"}
    xlabels = [label_map_auth.get(str(x), str(x)) for x in xlabels]
    ylabels = [label_map_auth.get(str(y), str(y)) for y in ylabels]
    
    ax_auth.set_xlabel('Functional Annotation', fontsize=14)
    ax_auth.set_ylabel('')
    
    if letters:
        ax_auth.set_xticks(np.arange(len(xlabels)) + 0.5)
        ax_auth.set_yticks(np.arange(len(ylabels)) + 0.5)
    
    ax_auth.set_xticklabels(xlabels, rotation=0, fontsize=12)
    ax_auth.set_yticklabels(ylabels, rotation=0, fontsize=12)
    ax_auth.tick_params(length=0, labelsize=12)
    
    for spine in ax_auth.spines.values():
        spine.set_visible(False)

    if not letters:
        ax_auth.text(0.5, -0.18, 
                    f"DOR: {auth_metrics['dor_standard']:.1f}  |  Determinate: Controls {100*auth_metrics['coverage']:.1f}%, VUS 93.2%",
                    transform=ax_auth.transAxes,
                    fontsize=12, ha='center', va='top',
                    color='#555555')
    
    return fig, danz_metrics, auth_metrics



def plot_aggregate_confusion_diff_matrix(danzs, auths, dataset_names, figsize=(12, 4)):
    """
    Plot aggregate confusion matrices summing all variants across datasets.
    Shows DanZ (left) and Author (right) with performance metrics.
    
    Parameters:
    -----------
    danzs : list of DataFrames
        List of DanZ confusion matrices
    auths : list of DataFrames
        List of author confusion matrices
    dataset_names : list of str
        List of dataset names
    figsize : tuple
        Figure size (default: (12, 4))
    
    Returns:
    --------
    matplotlib figure, danz_metrics dict, auth_metrics dict
    """
    
    # Aggregate confusion matrices
    danz_aggregate = None
    auth_aggregate = None
    n_datasets = 0
    
    for i, (danz_df, auth_df, dataset_name) in enumerate(zip(danzs, auths, dataset_names)):
        if danz_df is None or auth_df is None:
            continue
        
        # Aggregate
        if danz_aggregate is None:
            danz_aggregate = danz_df.copy()
            auth_aggregate = auth_df.copy()
        else:
            danz_aggregate += danz_df
            auth_aggregate += auth_df

        n_datasets += 1

    if danz_aggregate is None:
        raise ValueError("No valid datasets to aggregate!")

    print(f"Aggregated {n_datasets} datasets")

    # Compute aggregate metrics
    danz_metrics = compute_classification_metrics(danz_aggregate)
    auth_metrics = compute_classification_metrics(auth_aggregate)

    diff_aggregate = danz_aggregate - auth_aggregate
    
    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    # Color scheme
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#F0F0F0", "#8FBC8F", "#4682B4", "#1E3A5F"]
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    # Get max value for consistent color scale
    min_val = diff_aggregate.values.min()
    max_val = diff_aggregate.values.max()
    
    # Plot DanZ (left panel)
    ax_danz = axes
    
    sns.heatmap(
        diff_aggregate,
        annot=True,
        fmt='d',
        cmap=cmap,
        vmin=min_val,
        vmax=max_val,
        ax=ax_danz,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Title with aggregate metrics
    metrics_text = (
        f"Acc: {danz_metrics['accuracy']:.3f}, "
        f"Sens: {danz_metrics['sensitivity']:.3f}, "
        f"Spec: {danz_metrics['specificity']:.3f}, "
        f"MCC: {danz_metrics['mcc']:.3f}\n"
        f"Coverage: {danz_metrics['coverage']:.1%} "
        f"({danz_metrics['determinate']:,}/{danz_metrics['total']:,} variants, {n_datasets} datasets)"
    )
    
    # ax_danz.text(
    #     0.00, 1.10,
    #     "(A)",
    #     transform=ax_danz.transAxes,
    #     fontsize=12,
    #     fontweight='bold',
    #     verticalalignment='top',
    #     horizontalalignment='left'
    # )
    
    # ax_danz.set_title(
    #     f"Variant-level Annotations",
    #     fontsize=11,
    #     fontweight='bold',
    #     pad=10
    # )
    
    ax_danz.set_xlabel('Evidence Direction', fontsize=10)
    ax_danz.set_ylabel('ClinVar Classification', fontsize=10)
    ax_danz.tick_params(labelsize=9)
    ax_danz.set_xticklabels(ax_danz.get_xticklabels(), rotation=45, ha='right')
    ax_danz.set_yticklabels(["Indeterminate" if item == "IR" else item for item in ax_danz.get_yticklabels()], rotation=0)
    
    
    # Plot Author (right panel)
    # ax_auth = axes[1]
    
    # sns.heatmap(
    #     auth_aggregate,
    #     annot=True,
    #     fmt='d',
    #     cmap=cmap,
    #     vmin=0,
    #     vmax=max_val,
    #     ax=ax_auth,
    #     cbar_kws={'label': 'Count'},
    #     linewidths=0.5,
    #     linecolor='gray',
    #     annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    # )
    
    # # Title with aggregate metrics
    # metrics_text = (
    #     f"Acc: {auth_metrics['accuracy']:.3f}, "
    #     f"Sens: {auth_metrics['sensitivity']:.3f}, "
    #     f"Spec: {auth_metrics['specificity']:.3f}, "
    #     f"MCC: {auth_metrics['mcc']:.3f}\n"
    #     f"Coverage: {auth_metrics['coverage']:.1%} "
    #     f"({auth_metrics['determinate']:,}/{auth_metrics['total']:,} variants, {n_datasets} datasets)"
    # )
    
    # ax_auth.text(
    #     0.00, 1.10,
    #     "(B)",
    #     transform=ax_auth.transAxes,
    #     fontsize=12,
    #     fontweight='bold',
    #     verticalalignment='top',
    #     horizontalalignment='left'
    # )
    
    # ax_auth.set_title(
    #     f"Author Annotations",
    #     fontsize=11,
    #     fontweight='bold',
    #     pad=10
    # )
    
    # ax_auth.set_xlabel('Functional Annotation', fontsize=10)
    # ax_auth.set_ylabel('ClinVar Classification', fontsize=10)
    # ax_auth.tick_params(labelsize=9)
    # ax_auth.set_xticklabels(ax_auth.get_xticklabels(), rotation=45, ha='right')
    # ax_auth.set_yticklabels(ax_auth.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    return fig, danz_metrics, auth_metrics



def map_point_values_for_display(point_assignments):
    mapped = np.zeros_like(point_assignments)
    
    for i, point in enumerate(point_assignments):
        abs_point = abs(point)
        sign = np.sign(point)
        
        if abs_point == 0:
            mapped[i] = 0
        elif abs_point == 1:
            mapped[i] = 1 * sign
        elif abs_point in [2, 3]:
            mapped[i] = 2 * sign
        elif abs_point in [4, 5, 6, 7]:
            mapped[i] = 4 * sign
        elif abs_point == 8:
            mapped[i] = 8 * sign
        else:
            mapped[i] = 0  # Unknown, treat as no evidence
    
    return mapped.astype(int)

def compute_evidence_distribution(point_assignments, categories, category_labels=None):
    """
    Compute distribution of evidence strengths for each category.
    Groups point values: 1→1, 2,3→2, 4,5,6,7→4, 8→8
    Handles variants belonging to multiple categories.
    
    Parameters:
    -----------
    point_assignments : array-like
        Array of point values for each variant
    categories : array-like
        Category assignment for each variant
        Can be int (single category) or array-like (multi-label)
    category_labels : dict, optional
        Mapping of category values to display labels
    
    Returns:
    --------
    DataFrame with columns for each evidence strength and rows for each category
    """
    
    # Map point values to display groups
    mapped_points = map_point_values_for_display(point_assignments)
    
    # Check if multi-label or single-label
    first_cat = categories[0] if len(categories) > 0 else None
    is_multilabel = isinstance(first_cat, (list, np.ndarray))
    
    # Display point values (after grouping)
    point_values = [8, 4, 2, 1, 0, -1, -2, -4, -8]
    
    if is_multilabel:
        # Multi-label: each variant can belong to multiple categories
        n_categories = len(first_cat)
        
        distribution = {}
        for cat_idx in range(n_categories):
            # Get variants belonging to this category
            belongs_to_cat = np.array([cat[cat_idx] if len(cat) > cat_idx else False 
                                      for cat in categories])
            
            if not np.any(belongs_to_cat):
                continue  # Skip empty categories
            
            points_in_cat = mapped_points[belongs_to_cat]
            total = len(points_in_cat)
            
            # Count grouped point values
            counts = {p: (points_in_cat == p).sum() for p in point_values}
            
            # Convert to percentages
            percentages = {p: (counts[p] / total * 100) if total > 0 else 0 
                          for p in point_values}
            
            # Verify they sum to 100%
            total_pct = sum(percentages.values())
            if abs(total_pct - 100) > 0.1 and total > 0:
                print(f"Warning: category {cat_idx} percentages sum to {total_pct:.1f}%, not 100%")
            
            cat_label = category_labels.get(cat_idx, f"Category {cat_idx}") if category_labels else f"Category {cat_idx}"
            distribution[cat_label] = percentages
    
    else:
        # Single-label: each variant belongs to exactly one category
        df = pd.DataFrame({
            'point': mapped_points,
            'category': categories
        })
        
        distribution = {}
        for cat in df['category'].unique():
            if cat == -1:  # Skip invalid category
                continue
            
            cat_data = df[df['category'] == cat]
            total = len(cat_data)
            
            # Count grouped point values
            counts = {p: (cat_data['point'] == p).sum() for p in point_values}
            
            # Convert to percentages
            percentages = {p: (counts[p] / total * 100) if total > 0 else 0 
                          for p in point_values}
            
            # Verify they sum to 100%
            total_pct = sum(percentages.values())
            if abs(total_pct - 100) > 0.1 and total > 0:
                print(f"Warning: {cat} percentages sum to {total_pct:.1f}%, not 100%")
            
            cat_label = category_labels.get(cat, cat) if category_labels else cat
            distribution[cat_label] = percentages
    
    dist_df = pd.DataFrame(distribution).T
    return dist_df


def plot_evidence_distribution_by_category(point_assignments, categories, 
                                           category_labels, category_order,
                                           title="Evidence Strength Distribution",
                                           figsize=(12, 6)):
    """
    Plot horizontal stacked bar chart of evidence strengths by category.
    Handles both single-label and multi-label categories.
    """
    
    # Evidence strength colors
    strength_colors = {
        8: '#943744', 4: '#B85C6B', 2: '#D68F99', 1: '#E6B1B8',
        0: '#E0E0E0',
        -1: '#99C8DC', -2: '#7AB5D1', -4: '#4B91A6', -8: '#2E6B7E',
    }
    
    # Compute distribution
    dist_df = compute_evidence_distribution(point_assignments, categories, category_labels)
    
    # Reorder rows
    dist_df = dist_df.loc[[cat for cat in category_order if cat in dist_df.index]]
    
    # Get point values in order
    point_values = [8, 4, 2, 1, 0, -1, -2, -4, -8]
    
    # Check if multi-label
    first_cat = categories[0] if len(categories) > 0 else None
    is_multilabel = isinstance(first_cat, (list, np.ndarray))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bars
    left = np.zeros(len(dist_df))

    mapped_points = map_point_values_for_display(point_assignments)
    
    # Helper function to determine text color based on background
    def get_text_color(point_val):
        """Return white for dark backgrounds, black for light backgrounds"""
        # Dark backgrounds: strong positive and negative values
        if point_val in [8, 4, -4, -8]:
            return 'white'
        else:
            return 'black'
    
    for point_val in point_values:
        if point_val not in dist_df.columns:
            continue
        
        widths = dist_df[point_val].values
        
        ax.barh(
            range(len(dist_df)),
            widths,
            left=left,
            color=strength_colors[point_val],
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add text labels for bars wide enough
        for row_idx, width in enumerate(widths):
            if width > 4:
                x_center = left[row_idx] + width / 2
                y_center = row_idx
                
                cat = dist_df.index[row_idx]
                
                if is_multilabel:
                    cat_idx = [k for k, v in category_labels.items() if v == cat][0]
                    cat_mask = np.array([c[cat_idx] if len(c) > cat_idx else False 
                                        for c in categories])
                    count = (mapped_points[cat_mask] == point_val).sum()
                else:
                    cat_val = [k for k, v in category_labels.items() if v == cat][0]
                    cat_mask = categories == cat_val
                    count = (mapped_points[cat_mask] == point_val).sum()
                
                ax.text(
                    x_center, y_center,
                    f"{count:,}",
                    ha='center', va='center',
                    fontsize=12, 
                    color=get_text_color(point_val)
                )
        
        left += widths
    
    # Set y-axis
    y_labels_with_counts = []
    for cat in dist_df.index:
        if is_multilabel:
            cat_idx = [k for k, v in category_labels.items() if v == cat][0]
            n = sum([c[cat_idx] if len(c) > cat_idx else False for c in categories])
        else:
            cat_val = [k for k, v in category_labels.items() if v == cat][0]
            n = (categories == cat_val).sum()
        
        y_labels_with_counts.append(f"{cat}\n(n={n:,})")
    
    ax.set_yticks(range(len(dist_df)))
    ax.set_yticklabels(y_labels_with_counts, fontsize=12, rotation=0)
    
    # Set x-axis
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of Variants', fontsize=14)
    ax.set_ylabel('')
    
    # Title (commented out but with correct size)
    # ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    ax.set_axisbelow(True)

    map_point_to_text = {
        -8: "-8",# (very strong)",
        -4: "-4",# to -7",# (strong)",
        -2: "-2",# to -3",# (moderate)",
        -1: "-1",# (supporting)",
        0: "0",# (indeterminate)",
        1: "+1",# (supporting)",
        2: "+2",# to +2",# (moderate)",
        4: "+4",# to +4",# (strong)",
        8: "+8",# (very strong)",
    }
    
    # Legend
    legend_elements = [
        Patch(facecolor=strength_colors[p], label=map_point_to_text[p], edgecolor='none')
        for p in point_values if p in dist_df.columns
    ]
    
    ax.legend(
        handles=legend_elements,
        title='Evidence Strength',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(legend_elements),
        frameon=True,
        fontsize=12,
        title_fontsize=12,

        # spacing controls
        # columnspacing=0.8,     # horizontal space between columns
        # labelspacing=0.3,      # vertical space between rows
        # handletextpad=0.6,     # space between marker and text
        handlelength=1.0,      # length of legend handles
        # borderaxespad=0.2      # space between legend and axes)

    )
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_evidence_by_author_annotation_with_stats(danz_assignments, author_annotations, figsize=(12, 5)):
    """
    Plot evidence distribution by author functional annotation.
    Also computes and prints statistics for manuscript text.
    
    Parameters:
    -----------
    danz_assignments : array-like
        DanZ point assignments for each variant
    author_annotations : array-like
        Author functional annotations (0=Normal, 1=Indeterminate, 2=Abnormal)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib figure, dict of statistics
    """
    
    # Convert to arrays
    danz = np.array(danz_assignments)
    author = np.array(author_annotations)
    
    # Map to display groups (floor to power of 2)
    danz_mapped = map_point_values_for_display(danz)
    
    # Compute statistics
    abnormal_mask = author == 2
    normal_mask = author == 0
    
    n_abnormal = abnormal_mask.sum()
    n_normal = normal_mask.sum()
    
    # Abnormal variants with pathogenic evidence (any strength > 0)
    abnormal_with_path = (abnormal_mask & (danz_mapped > 0)).sum()
    pct_abnormal_path = 100 * abnormal_with_path / n_abnormal if n_abnormal > 0 else 0
    
    # Normal variants with benign evidence (any strength < 0)
    normal_with_benign = (normal_mask & (danz_mapped < 0)).sum()
    pct_normal_benign = 100 * normal_with_benign / n_normal if n_normal > 0 else 0
    
    # Abnormal variants reaching strong (+4 or higher)
    abnormal_strong_or_higher = (abnormal_mask & (danz_mapped >= 4)).sum()
    pct_abnormal_strong = 100 * abnormal_strong_or_higher / n_abnormal if n_abnormal > 0 else 0
    
    # Abnormal variants reaching very strong (+8)
    abnormal_very_strong = (abnormal_mask & (danz_mapped >= 8)).sum()
    pct_abnormal_vstrong = 100 * abnormal_very_strong / n_abnormal if n_abnormal > 0 else 0
    
    # Normal variants reaching very strong benign (-8 or lower)
    normal_very_strong = (normal_mask & (danz_mapped <= -8)).sum()
    pct_normal_vstrong = 100 * normal_very_strong / n_normal if n_normal > 0 else 0
    
    # Store stats
    stats = {
        'n_abnormal': n_abnormal,
        'n_normal': n_normal,
        'abnormal_with_path': abnormal_with_path,
        'pct_abnormal_path': pct_abnormal_path,
        'normal_with_benign': normal_with_benign,
        'pct_normal_benign': pct_normal_benign,
        'abnormal_strong_or_higher': abnormal_strong_or_higher,
        'pct_abnormal_strong': pct_abnormal_strong,
        'abnormal_very_strong': abnormal_very_strong,
        'pct_abnormal_vstrong': pct_abnormal_vstrong,
        'normal_very_strong': normal_very_strong,
        'pct_normal_vstrong': pct_normal_vstrong,
    }
    
    # Print formatted statistics for manuscript
    print("\n" + "="*80)
    print("MANUSCRIPT STATISTICS")
    print("="*80)
    print(f"\nOf the variants annotated by study authors as functionally abnormal,")
    print(f"{pct_abnormal_path:.1f}% were assigned pathogenic evidence of some strength,")
    print(f"and of variants annotated as functionally normal, {pct_normal_benign:.1f}%")
    print(f"were assigned benign evidence of some strength by our model.")
    
    print(f"\n{pct_abnormal_strong:.1f}% of variants annotated as functionally abnormal")
    print(f"reached the level of pathogenic strong (+4 or higher), and")
    print(f"{pct_abnormal_vstrong:.1f}% reached pathogenic very strong (+8).")
    
    print(f"\n{pct_normal_vstrong:.1f}% of variants annotated by authors as functionally")
    print(f"normal reached benign very strong (-8).")
    
    # Create plot
    category_labels_plot = {
        2: 'Abnormal',
        1: 'Indeterminate',
        0: 'Normal'
    }
    
    category_order_plot = ['Normal', 'Indeterminate', 'Abnormal']
    
    fig = plot_evidence_distribution_by_category(
        point_assignments=danz_assignments,
        categories=author_annotations,
        category_labels=category_labels_plot,
        category_order=category_order_plot,
        title='',
        # figsize=figsize
    )
    
    return fig#, stats


def plot_evidence_by_clinvar_class_with_stats(danz_assignments, clinvar_classes, figsize=(12, 6)):
    """
    Plot evidence distribution by ClinVar classification.
    Also computes and prints statistics for manuscript text.
    
    Parameters:
    -----------
    danz_assignments : array-like
        DanZ point assignments for each variant
    clinvar_classes : array-like or multi-label array
        ClinVar classes (0=B/LB, 1=VUS, 2=P/LP, 3=gnomAD, 4=Synonymous)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib figure, dict of statistics
    """
    
    # Convert to arrays
    danz = np.array(danz_assignments)
    
    # Map to display groups
    danz_mapped = map_point_values_for_display(danz)
    
    # Handle multi-label or single-label
    first_cat = clinvar_classes[0] if len(clinvar_classes) > 0 else None
    is_multilabel = isinstance(first_cat, (list, np.ndarray))
    
    if is_multilabel:
        # Extract masks for each class
        plp_mask = np.array([c[2] if len(c) > 2 else False for c in clinvar_classes])
        blb_mask = np.array([c[0] if len(c) > 0 else False for c in clinvar_classes])
        vus_mask = np.array([c[1] if len(c) > 1 else False for c in clinvar_classes])
    else:
        plp_mask = clinvar_classes == 2
        blb_mask = clinvar_classes == 0
        vus_mask = clinvar_classes == 1
    
    # Compute statistics
    n_plp = plp_mask.sum()
    n_blb = blb_mask.sum()
    n_vus = vus_mask.sum()
    
    # P/LP with pathogenic evidence (> 0)
    plp_with_path = (plp_mask & (danz_mapped > 0)).sum()
    pct_plp_path = 100 * plp_with_path / n_plp if n_plp > 0 else 0
    
    # B/LB with benign evidence (< 0)
    blb_with_benign = (blb_mask & (danz_mapped < 0)).sum()
    pct_blb_benign = 100 * blb_with_benign / n_blb if n_blb > 0 else 0
    
    # VUS with any evidence (!= 0)
    vus_with_evidence = (vus_mask & (danz_mapped != 0)).sum()
    pct_vus_evidence = 100 * vus_with_evidence / n_vus if n_vus > 0 else 0
    
    # VUS with benign evidence
    vus_with_benign = (vus_mask & (danz_mapped < 0)).sum()
    pct_vus_benign = 100 * vus_with_benign / n_vus if n_vus > 0 else 0
    
    # VUS with pathogenic evidence
    vus_with_path = (vus_mask & (danz_mapped > 0)).sum()
    pct_vus_path = 100 * vus_with_path / n_vus if n_vus > 0 else 0
    
    # Store stats
    stats = {
        'n_plp': n_plp,
        'n_blb': n_blb,
        'n_vus': n_vus,
        'plp_with_path': plp_with_path,
        'pct_plp_path': pct_plp_path,
        'blb_with_benign': blb_with_benign,
        'pct_blb_benign': pct_blb_benign,
        'vus_with_evidence': vus_with_evidence,
        'pct_vus_evidence': pct_vus_evidence,
        'vus_with_benign': vus_with_benign,
        'pct_vus_benign': pct_vus_benign,
        'vus_with_path': vus_with_path,
        'pct_vus_path': pct_vus_path,
    }
    
    # Print formatted statistics
    print("\n" + "="*80)
    print("CLINVAR CLASSIFICATION STATISTICS")
    print("="*80)
    
    print(f"\nP/LP variants (n={n_plp:,}):")
    print(f"  Assigned pathogenic evidence: {plp_with_path:,} ({pct_plp_path:.1f}%)")
    
    print(f"\nB/LB variants (n={n_blb:,}):")
    print(f"  Assigned benign evidence: {blb_with_benign:,} ({pct_blb_benign:.1f}%)")
    
    print(f"\nVUS variants (n={n_vus:,}):")
    print(f"  Assigned any evidence: {vus_with_evidence:,} ({pct_vus_evidence:.1f}%)")
    print(f"    Benign: {vus_with_benign:,} ({pct_vus_benign:.1f}%)")
    print(f"    Pathogenic: {vus_with_path:,} ({pct_vus_path:.1f}%)")
    
    # Create plot
    category_labels_plot = {
        2: 'P/LP',
        1: 'VUS',
        3: 'gnomAD',
        0: 'B/LB'
    }
    
    category_order_plot = ['P/LP', 'VUS', 'gnomAD', 'B/LB'][::-1]
    
    fig = plot_evidence_distribution_by_category(
        point_assignments=danz_assignments,
        categories=clinvar_classes,
        category_labels=category_labels_plot,
        category_order=category_order_plot,
        title='',
        # figsize=figsize
    )
    
    return fig#, stats


def plot_aggregate_evidence_distributions(all_danz, all_author,
                                          all_clinvar, figsize=(14, 10)):
    """
    Create aggregate evidence distribution plots across all datasets.
    """
    
    # Plot by author annotation
    fig_author = plot_evidence_by_author_annotation_with_stats(all_danz, all_author, figsize=figsize)
    
    # Plot by ClinVar class
    fig_clinvar = plot_evidence_by_clinvar_class_with_stats(all_danz, all_clinvar, figsize=figsize)
    
    return fig_author, fig_clinvar

def load_all_variant_assignments(dataset_names, dataset_configs, keep_old_list, internal_dataset_aliases,
                                 df_final, verbose=False):
    """
    Load DanZ point assignments, author annotations, and ClinVar classifications
    for all datasets.
    
    Parameters:
    -----------
    dataset_names : list of str
        List of dataset names to process
    dataset_configs : dict
        Dataset configurations
    keep_old_list : list
        List of keep_old datasets
    df_final : DataFrame
        Full dataset DataFrame
    verbose : bool
        Print progress
    
    Returns:
    --------
    tuple: (all_danz_assignments, all_author_annotations, all_clinvar_classes, all_scores, dataset_info)
    """
    
    all_danz_assignments = []
    all_author_annotations = []
    all_clinvar_classes = []
    all_scores = []
    dataset_info = []
    
    for dataset in tqdm(dataset_names, desc="Loading variant assignments"):
        
        # Handle dataset aliases
        dataset_std_name = dataset
        if dataset in internal_dataset_aliases:
            dataset_alias = internal_dataset_aliases[dataset]
            if dataset_alias is None:
                continue
            dataset = dataset_alias
        
        if dataset not in dataset_configs:
            if verbose:
                print(f"  {dataset}: not in dataset_configs, skipping")
            continue
        
        try:
            # Load calibration
            calibration_f = f"/data/ross/assay_calibration/calibrations_12_25_25/{dataset_std_name}.json"
            
            if not os.path.exists(calibration_f):
                if verbose:
                    print(f"  {dataset}: calibration not found, skipping")
                continue
            
            with open(calibration_f, 'r') as f:
                calibration_data = json.load(f)
            
            if calibration_data["point_ranges"] is None:
                if verbose:
                    print(f"  {dataset}: uncalibratable, skipping")
                continue
            
            point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
            
            # Load scoreset
            genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
            use_2018 = any(gene in dataset for gene in genes_2018)
            clinvar_rel = "2018" if use_2018 else "2025"
            
            scoreset = Scoreset(
                df_final[df_final["Dataset"] == dataset_std_name],
                clinvar_release=clinvar_rel
            )
            
            # Reconstruct FULL variant data (before keep_mask filtering)
            variants_by_id = scoreset.get_variants_by_id()
            
            n_plp = n_blb = n_gnomad = n_synonymous = n_vus = 0
            
            for idx, (variant_id, variants) in enumerate(variants_by_id.items()):
                variant = variants[0]
                score = variant.auth_reported_score
                auth_label = variant.auth_label
                
                # Assign DanZ points
                danz_point = assign_points(score, point_ranges)
                
                # Convert author label to numeric
                if pd.isna(auth_label):
                    author_num = 1  # Indeterminate
                else:
                    label_upper = str(auth_label).upper()
                    if label_upper == "NORMAL":
                        author_num = 0
                    elif label_upper == "ABNORMAL":
                        author_num = 2
                    else:
                        author_num = 1  # Indeterminate
                
                # Determine ClinVar classification
                if variant.is_pathogenic:
                    clinvar_class = 2  # P/LP
                    n_plp += 1
                elif variant.is_benign:
                    clinvar_class = 0  # B/LB
                    n_blb += 1
                elif variant.is_vus:
                    clinvar_class = 1  # VUS
                    n_vus += 1
                elif variant.is_synonymous:
                    clinvar_class = 4  # Synonymous
                    n_synonymous += 1
                elif scoreset.is_population_member(variant):
                    clinvar_class = 3  # gnomAD
                    n_gnomad += 1
                else:
                    continue  # Skip if no classification
                
                all_danz_assignments.append(danz_point)
                all_author_annotations.append(author_num)
                all_clinvar_classes.append(clinvar_class)
                all_scores.append(score)
            
            # Track dataset info
            dataset_info.append({
                'dataset': dataset_std_name,
                'n_variants': len(all_danz_assignments) - sum(d['n_variants'] for d in dataset_info),
                'n_plp': n_plp,
                'n_blb': n_blb,
                'n_gnomad': n_gnomad,
                'n_synonymous': n_synonymous,
                'n_vus': n_vus,
            })
            
            if verbose:
                total_this_dataset = n_plp + n_blb + n_gnomad + n_synonymous + n_vus
                print(f"  {dataset_std_name}: {total_this_dataset} variants")
        
        except Exception as e:
            if verbose:
                print(f"  {dataset}: Error - {e}")
            continue
    
    # Convert to arrays
    all_danz_assignments = np.array(all_danz_assignments)
    all_author_annotations = np.array(all_author_annotations)
    all_clinvar_classes = np.array(all_clinvar_classes)
    all_scores = np.array(all_scores)

    print(f"\n{'='*80}")
    print("LOADED VARIANT ASSIGNMENTS")
    print('='*80)
    print(f"Total variants: {len(all_danz_assignments):,}")
    print(f"Datasets: {len(dataset_info)}")
    print(f"\nClinVar distribution:")
    print(f"  B/LB: {(all_clinvar_classes == 0).sum():,}")
    print(f"  VUS: {(all_clinvar_classes == 1).sum():,}")
    print(f"  P/LP: {(all_clinvar_classes == 2).sum():,}")
    print(f"  gnomAD: {(all_clinvar_classes == 3).sum():,}")
    print(f"  Synonymous: {(all_clinvar_classes == 4).sum():,}")
    print(f"\nAuthor annotation distribution:")
    print(f"  Normal: {(all_author_annotations == 0).sum():,}")
    print(f"  Indeterminate: {(all_author_annotations == 1).sum():,}")
    print(f"  Abnormal: {(all_author_annotations == 2).sum():,}")
    print(f"\nDanZ point distribution:")
    for point in [8, 4, 2, 1, 0, -1, -2, -4, -8]:
        count = (all_danz_assignments == point).sum()
        if count > 0:
            print(f"  {point:+2d}: {count:,}")

    return all_danz_assignments, all_author_annotations, all_clinvar_classes, all_scores, pd.DataFrame(dataset_info)


from joblib import Parallel, delayed

def _process_single_dataset_assignments(dataset, dataset_configs, keep_old_list,
                                        internal_dataset_aliases, df_final,
                                        variant_to_oob_points_all, make_variant_id,
                                        sort_idx):
    """
    Process a single dataset to extract variant assignments.
    
    Returns:
    --------
    tuple: (sort_idx, results_dict or None)
    """
    
    # Handle dataset aliases
    dataset_std_name = dataset
    if dataset in internal_dataset_aliases:
        dataset_alias = internal_dataset_aliases[dataset]
        if dataset_alias is None:
            return sort_idx, None
        dataset = dataset_alias
    
    if dataset not in dataset_configs:
        return sort_idx, None
    
    try:
        # Load calibration
        calibration_f = f"/data/ross/assay_calibration/calibrations_12_25_25/{dataset_std_name}.json"
        
        if not os.path.exists(calibration_f):
            return sort_idx, None
        
        with open(calibration_f, 'r') as f:
            calibration_data = json.load(f)
        
        if calibration_data["point_ranges"] is None:
            return sort_idx, None
        
        point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
        
        # Get OOB dict
        genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
        use_2018 = any(gene in dataset for gene in genes_2018)
        
        oob_dataset_key = dataset_std_name
        if use_2018 and not dataset_std_name.endswith('_clinvar_2018'):
            oob_dataset_key = dataset_std_name + "_clinvar_2018"
        
        variant_to_oob_points = variant_to_oob_points_all.get(oob_dataset_key, {})
        if variant_to_oob_points is None:
            variant_to_oob_points = {}
        
        # Load scoreset
        # GIVE EVIDENCE FOR ALL VARIANTS REGARDLESS OF CLINVAR RELEASE USED FOR FIT
        # update: nevermind because we "exclude from analyses"
        clinvar_rel = "2018" if use_2018 else "2025"
        
        scoreset = Scoreset(
            df_final[df_final["Dataset"] == dataset_std_name],
            clinvar_release=clinvar_rel,
            synonymous_exclusive=False
        )
        
        if pd.isna(scoreset.auth_labels).all():
            return sort_idx, None
        
        # Build variant ID mapping
        variants_by_id = scoreset.get_variants_by_id()
        kept_idx_to_variant_id = {}
        kept_idx = 0
        
        for all_idx, (variant_id, variants) in enumerate(variants_by_id.items()):
            if scoreset._keep_mask[all_idx]:
                kept_idx_to_variant_id[kept_idx] = make_variant_id(variants[0])
                kept_idx += 1
        
        # Collect data for this dataset
        dataset_danz = []
        dataset_author = []
        dataset_clinvar = []
        dataset_scores = []
        dataset_is_snv = []
        
        n_plp = n_blb = n_gnomad = n_synonymous = n_vus = 0
        n_oob_used = 0
        n_inbag_fallback = 0
        
        # Process all variants
        for idx, (variant_id, variants) in enumerate(variants_by_id.items()):
            variant = variants[0]
            score = variant.auth_reported_score
            auth_label = variant.auth_label
            
            variant_id_str = make_variant_id(variant)
            
            # OOB first, in-bag fallback
            if variant_id_str in variant_to_oob_points:
                danz_point = variant_to_oob_points[variant_id_str]['points']
                n_oob_used += 1
            else:
                danz_point = assign_points(score, point_ranges)
                n_inbag_fallback += 1
            
            # Convert author label
            if pd.isna(auth_label):
                author_num = 1
            else:
                label_upper = str(auth_label).upper()
                if label_upper == "NORMAL":
                    author_num = 0
                elif label_upper == "ABNORMAL":
                    author_num = 2
                else:
                    author_num = 1
            
            # Determine ClinVar classification (multi-label)
            clinvar_class = np.zeros(5, dtype=bool)
            if any([variant.is_pathogenic for variant in variants]):
                clinvar_class[2] = True
                n_plp += 1
            if any([variant.is_benign for variant in variants]):
                clinvar_class[0] = True
                n_blb += 1
            if any([variant.is_vus for variant in variants]):
                clinvar_class[1] = True
                n_vus += 1
            if any([variant.is_synonymous for variant in variants]):
                clinvar_class[4] = True
                n_synonymous += 1
            if any([scoreset.is_population_member(variant) for variant in variants]):
                clinvar_class[3] = True
                n_gnomad += 1
            
            dataset_danz.append(danz_point)
            dataset_author.append(author_num)
            dataset_clinvar.append(clinvar_class)
            dataset_scores.append(score)
            dataset_is_snv.append(any([variant.is_snv for variant in variants]))
        print(dataset_std_name, 'n_oob_used:', n_oob_used)
        
        # Return results
        return sort_idx, {
            'dataset': dataset_std_name,
            'danz': dataset_danz,
            'author': dataset_author,
            'clinvar': dataset_clinvar,
            'scores': dataset_scores,
            'is_snv': dataset_is_snv,
            'n_plp': n_plp,
            'n_blb': n_blb,
            'n_gnomad': n_gnomad,
            'n_synonymous': n_synonymous,
            'n_vus': n_vus,
            'n_oob_used': n_oob_used,
            'n_inbag_fallback': n_inbag_fallback,
        }
    
    except Exception as e:
        if verbose:
            print(f"  {dataset}: Error - {e}")
        return sort_idx, None


def load_all_variant_assignments_with_oob(dataset_names, dataset_configs, keep_old_list, 
                                          internal_dataset_aliases, df_final, 
                                          variant_to_oob_points_all, make_variant_id, 
                                          n_jobs=-1, verbose=False):
    """
    Load variant assignments with OOB evidence (parallelized).
    """
    
    print(f"Processing {len(dataset_names)} datasets in parallel...")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_single_dataset_assignments)(
            dataset, dataset_configs, keep_old_list,
            internal_dataset_aliases, df_final,
            variant_to_oob_points_all, make_variant_id,
            idx
        )
        for idx, dataset in enumerate(dataset_names)
    )
    
    # Sort by original index and filter None
    results_sorted = sorted(results, key=lambda x: x[0])
    valid_results = [data for _, data in results_sorted if data is not None]
    
    print(f"Successfully processed {len(valid_results)}/{len(dataset_names)} datasets")
    
    # Aggregate all data
    all_danz_assignments = []
    all_author_annotations = []
    all_clinvar_classes = []
    all_datasets = []
    all_scores = []
    all_is_snv = []
    dataset_info = []
    
    for result in valid_results:
        all_danz_assignments.extend(result['danz'])
        all_author_annotations.extend(result['author'])
        all_clinvar_classes.extend(result['clinvar'])
        all_datasets.extend(result['dataset'])
        all_scores.extend(result['scores'])
        all_is_snv.extend(result['is_snv'])
        
        dataset_info.append({
            'dataset': result['dataset'],
            'n_variants': len(result['danz']),
            'n_plp': result['n_plp'],
            'n_blb': result['n_blb'],
            'n_gnomad': result['n_gnomad'],
            'n_synonymous': result['n_synonymous'],
            'n_vus': result['n_vus'],
            'n_oob_used': result['n_oob_used'],
            'n_inbag_fallback': result['n_inbag_fallback'],
            'pct_oob': 100 * result['n_oob_used'] / len(result['danz']) if len(result['danz']) > 0 else 0
        })
    
    # Convert to arrays
    all_danz_assignments = np.array(all_danz_assignments)
    all_author_annotations = np.array(all_author_annotations)
    all_clinvar_classes = np.array(all_clinvar_classes)
    all_scores = np.array(all_scores)
    all_is_snv = np.array(all_is_snv)
    
    dataset_info_df = pd.DataFrame(dataset_info)
    
    # Calculate coverage metrics
    # Coverage = % of non-indeterminate assignments
    # DanZ indeterminate = 0, Author indeterminate = 1
    
    def calculate_coverage(danz_arr, author_arr, mask):
        """Calculate coverage for DanZ and Author given a boolean mask"""
        if mask.sum() == 0:
            return None, None
        
        danz_coverage = 100 * (danz_arr[mask] != 0).sum() / mask.sum()
        author_coverage = 100 * (author_arr[mask] != 1).sum() / mask.sum()
        return danz_coverage, author_coverage
    
    # Create masks for different variant groups
    plp_blb_mask = np.array([c[0] or c[2] for c in all_clinvar_classes])  # B/LB or P/LP
    vus_mask = np.array([c[1] for c in all_clinvar_classes])  # VUS
    snv_mask = all_is_snv  # SNVs
    
    coverage_stats = {}
    
    # Coverage for P/LP + B/LB controls
    danz_cov, auth_cov = calculate_coverage(all_danz_assignments, all_author_annotations, plp_blb_mask)
    coverage_stats['controls'] = {
        'n': plp_blb_mask.sum(),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Coverage for VUS
    danz_cov, auth_cov = calculate_coverage(all_danz_assignments, all_author_annotations, vus_mask)
    coverage_stats['vus'] = {
        'n': vus_mask.sum(),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Coverage for SNVs
    danz_cov, auth_cov = calculate_coverage(all_danz_assignments, all_author_annotations, snv_mask)
    coverage_stats['snvs'] = {
        'n': snv_mask.sum(),
        'danz_coverage': danz_cov,
        'author_coverage': auth_cov
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("LOADED VARIANT ASSIGNMENTS (WITH OOB)")
    print('='*80)
    print(f"Total variants: {len(all_danz_assignments):,}")
    print(f"Datasets: {len(dataset_info)}")
    print(f"\nOOB vs In-bag:")
    print(f"  OOB evidence used: {dataset_info_df['n_oob_used'].sum():,} ({100*dataset_info_df['n_oob_used'].sum()/len(all_danz_assignments):.1f}%)")
    print(f"  In-bag fallback: {dataset_info_df['n_inbag_fallback'].sum():,} ({100*dataset_info_df['n_inbag_fallback'].sum()/len(all_danz_assignments):.1f}%)")
    
    # Handle multi-label clinvar for counting
    if len(all_clinvar_classes) > 0:
        print(f"\nClinVar distribution:")
        print(f"  B/LB: {sum([c[0] for c in all_clinvar_classes]):,}")
        print(f"  VUS: {sum([c[1] for c in all_clinvar_classes]):,}")
        print(f"  P/LP: {sum([c[2] for c in all_clinvar_classes]):,}")
        print(f"  gnomAD: {sum([c[3] for c in all_clinvar_classes]):,}")
        print(f"  Synonymous: {sum([c[4] for c in all_clinvar_classes]):,}")
        print(f"  SNVs: {snv_mask.sum():,}")
    
    print(f"\nAuthor annotation distribution:")
    print(f"  Normal: {(all_author_annotations == 0).sum():,}")
    print(f"  Indeterminate: {(all_author_annotations == 1).sum():,}")
    print(f"  Abnormal: {(all_author_annotations == 2).sum():,}")
    
    print(f"\nDanZ point distribution:")
    for point in [8, 4, 2, 1, 0, -1, -2, -4, -8]:
        count = (all_danz_assignments == point).sum()
        if count > 0:
            print(f"  {point:+2d}: {count:,}")
    
    # Print coverage statistics
    print(f"\n{'='*80}")
    print("COVERAGE STATISTICS (% Non-Indeterminate)")
    print('='*80)
    
    for group_name, stats in [('P/LP + B/LB Controls', coverage_stats['controls']),
                               ('VUS Variants', coverage_stats['vus']),
                               ('SNVs', coverage_stats['snvs'])]:
        if stats['n'] > 0:
            print(f"\n{group_name} (n={stats['n']:,}):")
            print(f"  DanZ Coverage:   {stats['danz_coverage']:.1f}%")
            print(f"  Author Coverage: {stats['author_coverage']:.1f}%")
    
    print('='*80)
    
    return all_danz_assignments, all_author_annotations, all_clinvar_classes, all_scores, dataset_info_df



def plot_scoreset_cartoon_overview(dataset, scoreset, indv_summary, fits, score_range, 
                                   n_samples, flipped=False, figsize=(14, 3.5)):
    """
    Cartoon overview: Generated histograms (left) | Fits with arrows (right).
    Left: Clean histograms generated from fitted distributions (counts)
    Right: Fitted densities with upward arrows
    """
    
    from scipy.stats import skewnorm
    
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    
    # Create side-by-side subplots - REMOVE sharey=True
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)  # No sharey!
    
    x_min = score_range[0]
    x_max = score_range[-1]
    x_range = x_max - x_min
    
    # Use first fit as reference
    ref_fit = fits[0]['fit']
    params = ref_fit['component_params']
    weights_plp = ref_fit['weights'][0]
    weights_blb = ref_fit['weights'][1]
    
    # ===== LEFT: Generate samples and plot as histograms =====
    
    np.random.seed(42)
    n_background = 6000
    n_plp_blb = 3000
    bins = 25
    
    # Generate balanced bimodal background
    weights_balanced = np.array([0.75,0.25])
    background_samples = []
    for (a, loc, scale), w in zip(params, weights_balanced):
        n_comp = int(n_background * w)
        samples = skewnorm.rvs(a, loc=loc, scale=scale*1.35, size=n_comp)
        background_samples.extend(samples)
    background_samples = np.array(background_samples)
    background_samples = background_samples[(background_samples >= x_min) & (background_samples <= x_max)]
    
    # Plot background
    ax_left.hist(background_samples, bins=bins, 
                alpha=0.4, color='#909090', edgecolor='none')
    
    # Generate P/LP samples
    plp_samples = []
    for (a, loc, scale), w in zip(params, weights_plp):
        n_comp = int(n_plp_blb * w)
        samples = skewnorm.rvs(a, loc=loc, scale=scale, size=n_comp)
        plp_samples.extend(samples)
    plp_samples = np.array(plp_samples)
    plp_samples = plp_samples[(plp_samples >= x_min) & (plp_samples <= x_max)]
    
    ax_left.hist(plp_samples, bins=bins,
                alpha=0.7, color=sample_colors[0], edgecolor='none')
    
    # Generate B/LB samples
    blb_samples = []
    for (a, loc, scale), w in zip(params, weights_blb):
        n_comp = int(n_plp_blb * w)
        samples = skewnorm.rvs(a, loc=loc, scale=scale, size=n_comp)
        blb_samples.extend(samples)
    blb_samples = np.array(blb_samples)
    blb_samples = blb_samples[(blb_samples >= x_min) & (blb_samples <= x_max)]
    
    ax_left.hist(blb_samples, bins=bins,
                alpha=0.7, color=sample_colors[1], edgecolor='none')
    
    # ===== RIGHT: Fitted densities with upward arrows =====
    
    density_scale = 0.6  # Shrink to 60% of original height
    
    # Plot P/LP density (scaled down)
    plp_density = np.zeros_like(score_range)
    for (a, loc, scale), w in zip(params, weights_plp):
        plp_density += w * skewnorm.pdf(score_range, a, loc=loc, scale=scale)
    plp_density_scaled = plp_density * density_scale
    ax_right.plot(score_range, plp_density_scaled, color=sample_colors[0], linewidth=3, zorder=10)
    
    # Plot B/LB density (scaled down)
    blb_density = np.zeros_like(score_range)
    for (a, loc, scale), w in zip(params, weights_blb):
        blb_density += w * skewnorm.pdf(score_range, a, loc=loc, scale=scale)
    blb_density_scaled = blb_density * density_scale
    ax_right.plot(score_range, blb_density_scaled, color=sample_colors[1], linewidth=3, zorder=10)
    
    # Set y-limit to make room for arrows (use max of scaled densities)
    max_density_scaled = max(plp_density_scaled.max(), blb_density_scaled.max())
    y_max = max_density_scaled * 1.8  # Extra headroom for arrows
    ax_right.set_ylim(0, y_max)
    
    # Add curved upward arrows (now y_max has more space)
    from matplotlib.patches import FancyArrowPatch
    
    # Left arrow
    left_x_start_arrow = x_min + x_range * 0.35
    left_x_end_arrow = x_min + x_range * 0.03
    left_arrow = FancyArrowPatch(
        (left_x_start_arrow, y_max * 0.7),  # Start lower relative to new y_max
        (left_x_end_arrow, y_max * 0.95),    # End higher
        arrowstyle='->', mutation_scale=30, lw=3.5, color='#2166AC',
        connectionstyle="arc3,rad=-0.3", zorder=15
    )
    ax_right.add_patch(left_arrow)
    ax_right.text((left_x_start_arrow+left_x_end_arrow)/2, y_max * 0.85, 
                 'Benign\nEvidence',
                 ha='center', va='center', fontsize=10, color='#2166AC', 
                 fontweight='bold')
    
    # Right arrow
    right_x_start_arrow = x_max - x_range * 0.35
    right_x_end_arrow = x_max - x_range * 0.03
    right_arrow = FancyArrowPatch(
        (right_x_start_arrow, y_max * 0.7),
        (right_x_end_arrow, y_max * 0.95),
        arrowstyle='->', mutation_scale=30, lw=3.5, color='#B2182B',
        connectionstyle="arc3,rad=0.3", zorder=15
    )
    ax_right.add_patch(right_arrow)
    ax_right.text((right_x_start_arrow+right_x_end_arrow)/2, y_max * 0.85, 
                 'Pathogenic\nEvidence',
                 ha='center', va='center', fontsize=10, color='#B2182B', 
                 fontweight='bold')
    
    # Clean appearance
    for ax in [ax_left, ax_right]:
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    ax_left.set_ylim(bottom=0)
    # ax_right y-lim already set above
    
    plt.tight_layout()
    return fig



def calculate_diagnostic_odds_ratio(evidence_points, clinvar_classes):
    """
    Calculate Diagnostic Odds Ratio for evidence predictions.
    
    DOR = (TP × TN) / (FP × FN)
    
    Parameters:
    -----------
    evidence_points : array
        Evidence points (negative=benign, positive=pathogenic, 0=indeterminate)
    clinvar_classes : array
        Multi-label array where [0]=B/LB, [2]=P/LP
    
    Returns:
    --------
    dict with DOR and contingency table
    """
    
    # Extract P/LP and B/LB masks
    plp_mask = np.array([c[2] for c in clinvar_classes])
    blb_mask = np.array([c[0] for c in clinvar_classes])
    
    # Get evidence predictions (excluding indeterminate)
    evidence_pathogenic = evidence_points > 0
    evidence_benign = evidence_points < 0
    evidence_indeterminate = evidence_points == 0
    
    # Calculate confusion matrix (excluding indeterminates)
    # TP: P/LP with pathogenic evidence
    TP = (plp_mask & evidence_pathogenic).sum()
    
    # TN: B/LB with benign evidence
    TN = (blb_mask & evidence_benign).sum()
    
    # FP: B/LB with pathogenic evidence
    FP = (blb_mask & evidence_pathogenic).sum()
    
    # FN: P/LP with benign evidence
    FN = (plp_mask & evidence_benign).sum()
    
    # Diagnostic Odds Ratio
    if FP == 0 or FN == 0:
        # Add 0.5 continuity correction if any cell is 0
        DOR = ((TP + 0.5) * (TN + 0.5)) / ((FP + 0.5) * (FN + 0.5))
        corrected = True
    else:
        DOR = (TP * TN) / (FP * FN)
        corrected = False
    
    # Additional metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    # Coverage (proportion with determinate evidence)
    n_plp_total = plp_mask.sum()
    n_blb_total = blb_mask.sum()
    n_plp_determinate = TP + FN
    n_blb_determinate = TN + FP
    
    plp_coverage = n_plp_determinate / n_plp_total if n_plp_total > 0 else 0
    blb_coverage = n_blb_determinate / n_blb_total if n_blb_total > 0 else 0
    
    results = {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'DOR': float(DOR),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'PPV': float(PPV),
        'NPV': float(NPV),
        'accuracy': float(accuracy),
        'plp_coverage': float(plp_coverage),
        'blb_coverage': float(blb_coverage),
        'corrected': corrected
    }
    
    # Print results
    print("="*80)
    print("DIAGNOSTIC ODDS RATIO CALCULATION")
    print("="*80)
    print(f"\nContingency Table (determinate evidence only):")
    print(f"                Predicted Pathogenic    Predicted Benign")
    print(f"P/LP (n={n_plp_total:,})     {TP:6,} (TP)              {FN:6,} (FN)")
    print(f"B/LB (n={n_blb_total:,})     {FP:6,} (FP)              {TN:6,} (TN)")
    
    print(f"\n{'With continuity correction' if corrected else 'No correction needed'}")
    print(f"\nDiagnostic Odds Ratio: {DOR:.2f}")
    print(f"  (Odds of pathogenic evidence in P/LP / Odds of pathogenic evidence in B/LB)")
    
    print(f"\nPerformance Metrics:")
    print(f"  Sensitivity: {sensitivity:.3f} ({100*sensitivity:.1f}%)")
    print(f"  Specificity: {specificity:.3f} ({100*specificity:.1f}%)")
    print(f"  PPV:         {PPV:.3f} ({100*PPV:.1f}%)")
    print(f"  NPV:         {NPV:.3f} ({100*NPV:.1f}%)")
    print(f"  Accuracy:    {accuracy:.3f} ({100*accuracy:.1f}%)")
    
    print(f"\nCoverage (determinate evidence):")
    print(f"  P/LP: {n_plp_determinate:,}/{n_plp_total:,} ({100*plp_coverage:.1f}%)")
    print(f"  B/LB: {n_blb_determinate:,}/{n_blb_total:,} ({100*blb_coverage:.1f}%)")
    
    return results

def import_dataset_configurations():
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
    
    with open('/home/rcstewart/exCALIBR/src/igvf_configs/dataset_configs_jan_2026.json','rt') as f:
        new_dataset_configs = json.load(f)

    # del new_dataset_configs['TP53_Fayer_2021_meta']
    # del new_dataset_configs['TP53_Fayer_2021_meta_clinvar_2018']
    # del new_dataset_configs['SFPQ_unpublished']
    
    keep_old_list = set()
    keep_old_path = '/data/ross/assay_calibration/old/keep_old_datasets.txt'
    try:
        with open(keep_old_path, 'r') as f:
            for line in f:
                keep_old_list.add(line.strip())
    except FileNotFoundError:
        pass

    return dataset_configs, dataset_relax_configs, new_dataset_configs, keep_old_list



func_class_map = {
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
                                            'nonfunctional': 'Abnormal'}
}

INTERVAL_CLASS_NORMALIZATION = {
    "normal": "Normal",
    "abnormal": "Abnormal",
    "not specified": "Indeterminate",
}


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
    """Extract interval-based classification from row."""
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


def classify_by_intervals(score, intervals):
    """
    Classify a score using interval-based rules.
    
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


def standardize_auth_label(dataset_name, func_class, score=None, intervals=None):
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


def standardize_class(c):
    """Standardize a class value."""
    if c is None:
        return None
    elif pd.isna(c) or (isinstance(c, str) and c.strip().lower() in {"nan", "not specified"}):
        return "Indeterminate"
    else:
        return c.capitalize()


def parse_author_label(row, dataset_name):
    """
    Parse and standardize author label following Variant.parse_auth_class logic.
    
    Priority:
      1) StandardizedClass column if present
      2) Dataset-specific func_class_map
      3) Interval-based classification
      4) Indeterminate
    """
    # Check for StandardizedClass column first
    if 'StandardizedClass' in row:
        standardized = standardize_class(row['StandardizedClass'])
        if standardized is not None:
            return standardized
    
    # Get functional class and score
    func_class = row.get('auth_reported_func_class')
    score = row.get('auth_reported_score')
    
    # Extract intervals from row
    intervals = extract_intervals_from_row(row)
    
    # Try dataset-specific mapping or interval-based classification
    if func_class is not None and dataset_name in func_class_map:
        return standardize_auth_label(
            dataset_name=dataset_name,
            func_class=func_class,
            score=score,
            intervals=intervals if intervals else None
        )
    else:
        if score is not None and intervals:
            return classify_by_intervals(score, intervals)
        else:
            return "Indeterminate"

import pandas as pd
import numpy as np
import json
import os
import math
import re
from pathlib import Path
def insert_evidence_into_dataframe(
    df,
    variant_to_oob_points_all,
    dataset_configs,
    internal_dataset_aliases=None,
    calibration_dir="/data/ross/assay_calibration/calibrations_12_25_25",
    make_variant_id=None,
    verbose=True
):
    """
    Insert OOB and in-bag evidence directly into the dataframe.
    
    For each row in the dataframe:
    - Applies filtering (invalid flags and splicing variants)
    - Looks up OOB evidence if available
    - Falls back to in-bag calibration
    - Adds new columns with evidence assignments
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe with variant data
    variant_to_oob_points_all : dict
        Dictionary mapping dataset names to variant_id -> OOB points
    dataset_configs : dict
        Dataset configuration dictionary
    assign_points : callable
        Function to assign points based on score and point_ranges
    flatten_point_ranges : callable
        Function to flatten point_ranges dictionary
    internal_dataset_aliases : dict, optional
        Mapping of dataset names to aliases
    calibration_dir : str or Path
        Directory containing calibration JSON files
    make_variant_id : callable, optional
        Function to create variant ID from row. If None, uses default.
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with new columns:
        - danz_points: The assigned DanZ point value
        - evidence_source: 'OOB' or 'in-bag'
        - author_label_standardized: Standardized author label
        - is_filtered: Boolean indicating if row was filtered out
    """
    if internal_dataset_aliases is None:
        internal_dataset_aliases = {}
    
    if make_variant_id is None:
        # Default variant ID function
        def make_variant_id(row):
            return f"{row.Gene}_{row.Chrom}_{row.hgvs_c}"
    
    calibration_dir = Path(calibration_dir)
    
    # Initialize new columns
    df = df.copy()
    df['danz_points'] = 0
    df['evidence_source'] = 'none'
    df['author_label_standardized'] = 'Indeterminate'
    df['is_filtered'] = False
    
    # Process by dataset
    for dataset_name in df['Dataset'].unique():
        if verbose:
            print(f"Processing dataset: {dataset_name}")
        
        # Handle dataset aliases
        dataset_std_name = dataset_name
        dataset_alias = dataset_name
        if dataset_name in internal_dataset_aliases:
            alias = internal_dataset_aliases[dataset_name]
            if alias is None:
                continue
            dataset_alias = alias
        
        if dataset_alias not in dataset_configs:
            if verbose:
                print(f"  Skipping {dataset_name} - not in dataset_configs")
            continue
        
        # Load calibration
        calibration_f = calibration_dir / f"{dataset_std_name}.json"
        
        if not calibration_f.exists():
            if verbose:
                print(f"  Skipping {dataset_name} - no calibration file")
            continue
        
        with open(calibration_f, 'r') as f:
            calibration_data = json.load(f)
        
        if calibration_data.get("point_ranges") is None:
            if verbose:
                print(f"  Skipping {dataset_name} - no point_ranges in calibration")
            continue
        
        point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
        
        # Get OOB dict for this dataset
        genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
        use_2018 = any(gene in dataset_name for gene in genes_2018)
        
        oob_dataset_key = dataset_std_name
        if use_2018 and not dataset_std_name.endswith('_clinvar_2018'):
            oob_dataset_key = dataset_std_name + "_clinvar_2018"
        
        variant_to_oob_points = variant_to_oob_points_all.get(oob_dataset_key, {})
        if variant_to_oob_points is None:
            variant_to_oob_points = {}
        
        # Check if dataset detects splice variants
        dataset_mask = df['Dataset'] == dataset_name
        dataset_df = df[dataset_mask]
        
        detects_splice = False
        if 'splice_measure' in dataset_df.columns:
            splice_vals = dataset_df['splice_measure'].unique()
            if len(splice_vals) > 0:
                detects_splice = splice_vals[0] == "Yes"
        
        # Process each row in this dataset
        n_oob = 0
        n_inbag = 0
        n_filtered = 0
        
        for idx in dataset_df.index:
            row = df.loc[idx]
            
            # FILTER 1: Invalid flag
            if 'Flag' in row and row['Flag'] == "*":
                df.at[idx, 'is_filtered'] = True
                n_filtered += 1
                continue
            
            # FILTER 2: Splicing filter (if dataset doesn't detect splice)
            if not detects_splice:
                should_filter = False
                
                # Check simplified_consequence
                if 'simplified_consequence' in row:
                    consequence = str(row['simplified_consequence']).lower()
                    if consequence == "splice region" or consequence == "splice_site_variant":
                        should_filter = True
                
                # Check spliceAI scores
                if not should_filter and all(col in row for col in ['spliceAI_DS_AG', 'spliceAI_DS_AL', 'spliceAI_DS_DG', 'spliceAI_DS_DL']):
                    try:
                        if not (row['spliceAI_DS_AG'] < 0.2 and 
                                row['spliceAI_DS_AL'] < 0.2 and 
                                row['spliceAI_DS_DG'] < 0.2 and 
                                row['spliceAI_DS_DL'] < 0.2):
                            should_filter = True
                    except (TypeError, ValueError):
                        # Handle NaN or invalid values
                        pass
                
                if should_filter:
                    df.at[idx, 'is_filtered'] = True
                    n_filtered += 1
                    continue
            
            # Create variant ID
            if callable(make_variant_id):
                try:
                    variant_id = make_variant_id(row)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not create variant_id for row {idx}: {e}")
                    continue
            else:
                variant_id = f"{row.Gene}_{row.Chrom}_{row.hgvs_c}"
            
            # Get score
            score = row.get('auth_reported_score')
            if pd.isna(score):
                continue
            score = float(score)
            
            # Check for OOB evidence first
            if variant_id in variant_to_oob_points:
                danz_point = variant_to_oob_points[variant_id]['points']
                evidence_source = 'OOB'
                n_oob += 1
            else:
                # Fall back to in-bag calibration
                danz_point = assign_points(score, point_ranges)
                evidence_source = 'in-bag'
                n_inbag += 1
            
            # Assign DanZ points and evidence source
            df.at[idx, 'danz_points'] = danz_point
            df.at[idx, 'evidence_source'] = evidence_source
            
            # Parse and standardize author label using the full logic
            standardized_label = parse_author_label(row, dataset_name)
            df.at[idx, 'author_label_standardized'] = standardized_label
        
        if verbose:
            total = n_oob + n_inbag
            pct_oob = 100 * n_oob / total if total > 0 else 0
            print(f"  {dataset_name}: {n_oob} OOB ({pct_oob:.1f}%), {n_inbag} in-bag, {n_filtered} filtered")
    
    # Print summary statistics
    if verbose:
        print("\n" + "="*80)
        print("EVIDENCE INSERTION SUMMARY")
        print("="*80)
        print(f"Total rows: {len(df):,}")
        print(f"Filtered rows: {df['is_filtered'].sum():,}")
        print(f"Processed rows: {(~df['is_filtered']).sum():,}")
        
        # Stats for non-filtered rows only
        df_processed = df[~df['is_filtered']]
        
        evidence_counts = df_processed['evidence_source'].value_counts()
        print(f"\nEvidence source distribution (non-filtered):")
        for source, count in evidence_counts.items():
            pct = 100 * count / len(df_processed) if len(df_processed) > 0 else 0
            print(f"  {source}: {count:,} ({pct:.1f}%)")
        
        print(f"\nDanZ point distribution (non-filtered):")
        point_counts = df_processed['danz_points'].value_counts().sort_index(ascending=False)
        for point, count in point_counts.items():
            if count > 0:
                pct = 100 * count / len(df_processed) if len(df_processed) > 0 else 0
                print(f"  {point:+2.0f}: {count:,} ({pct:.1f}%)")
        
        print(f"\nAuthor label distribution (non-filtered):")
        label_counts = df_processed['author_label_standardized'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(df_processed) if len(df_processed) > 0 else 0
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        print("="*80)
    
    return df


def plot_gene_level_performance_comparison(danzs, auths, dataset_names, gene_to_vus_count, 
                                          metric='accuracy', figsize=(9, 8)):
    
    # Aggregate by gene
    gene_danz = {}
    gene_auth = {}
    
    for danz_df, auth_df, dataset_name in zip(danzs, auths, dataset_names):
        if danz_df is None or auth_df is None:
            continue
        
        gene = dataset_name.split('_')[0]
        
        if gene not in gene_danz:
            gene_danz[gene] = danz_df.copy()
            gene_auth[gene] = auth_df.copy()
        else:
            gene_danz[gene] += danz_df
            gene_auth[gene] += auth_df
    
    # Compute metrics
    gene_results = []
    
    for gene in gene_danz.keys():
        danz_metrics = compute_classification_metrics(gene_danz[gene])
        auth_metrics = compute_classification_metrics(gene_auth[gene])
        
        metric_key = metric if metric == 'accuracy' else 'dor_standard'
        
        gene_results.append({
            'gene': gene,
            'danz_metric': danz_metrics[metric_key],
            'auth_metric': auth_metrics[metric_key],
            'vus_count': gene_to_vus_count.get(gene, 100)
        })
    
    # Categorize
    finite = [r for r in gene_results 
              if r['danz_metric'] not in [0, float('inf')] and 
                 r['auth_metric'] not in [0, float('inf')]]
    
    undefined = [r for r in gene_results 
                if r['danz_metric'] == 0 or r['auth_metric'] == 0]
    
    inf_both = [r for r in gene_results 
                if r['danz_metric'] == float('inf') and r['auth_metric'] == float('inf')]
    
    inf_danz = [r for r in gene_results 
                if r['danz_metric'] == float('inf') and r['auth_metric'] not in [0, float('inf')]]
    
    inf_auth = [r for r in gene_results 
                if r['auth_metric'] == float('inf') and r['danz_metric'] not in [0, float('inf')]]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Visually distinct colors (Set3 pastel palette)
    gene_list = sorted([r['gene'] for r in gene_results])
    colors = plt.cm.Set3(np.linspace(0, 1, len(gene_list)))
    gene_colors = {gene: colors[i] for i, gene in enumerate(gene_list)}
    
    # Limits
    if metric == 'accuracy' and finite:
        min_val = min(min(r['danz_metric'] for r in finite),
                     min(r['auth_metric'] for r in finite))
        max_val = max(max(r['danz_metric'] for r in finite),
                     max(r['auth_metric'] for r in finite))
        
        plot_min = max(0.5, min_val - 0.05)
        plot_max = min(1.0, max_val + 0.02)
        inf_pos = 0.995
        undefined_pos = plot_min * 0.97
        
    else:  # DOR
        if finite:
            max_val = max(max(r['danz_metric'] for r in finite),
                         max(r['auth_metric'] for r in finite))
            plot_max = max_val * 1.3
            plot_min = 1
            inf_pos = plot_max * 0.97
            undefined_pos = 1.5
        else:
            plot_min, plot_max, inf_pos = 1, 10000, 9700
            undefined_pos = 1.5
    
    vus_all = [r['vus_count'] for r in gene_results]
    max_vus = max(vus_all)
    
    def marker_size(vus):
        if max(vus_all) == min(vus_all):
            return 300
        norm = (vus - min(vus_all)) / (max(vus_all) - min(vus_all))
        return 150 + norm * 600
    
    # Reference line
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 
            'k--', alpha=0.35, linewidth=1.5, zorder=1)
    
    texts = []  # For adjustText
    
    # Plot finite - NO TEXT, just markers
    for r in finite:
        ax.scatter(r['auth_metric'], r['danz_metric'],
                  s=marker_size(r['vus_count']),
                  c=[gene_colors[r['gene']]],
                  alpha=0.75,
                  edgecolors='white',
                  linewidth=2,
                  zorder=3)
        
        # Add text for adjustText to position
        texts.append(ax.text(r['auth_metric'], r['danz_metric'], r['gene'],
                            fontsize=9, ha='center', va='center',
                            fontweight='bold', zorder=4))
    
    # Undefined
    if undefined:
        for i, r in enumerate(undefined):
            x_pos = undefined_pos * (1 + i * 0.15) if metric == 'dor' else plot_min + i * 0.02
            y_pos = undefined_pos if metric == 'dor' else undefined_pos
            
            ax.scatter(x_pos, y_pos,
                      s=marker_size(r['vus_count']),
                      facecolors='none',
                      edgecolors=gene_colors[r['gene']],
                      linewidth=3,
                      alpha=0.9,
                      zorder=3)
            texts.append(ax.text(x_pos, y_pos, r['gene'],
                                fontsize=8, ha='center', va='center', fontweight='bold'))
    
    # Infinite cases
    for i, r in enumerate(inf_both):
        jitter = 0.98 + 0.005 * (i - len(inf_both)/2)
        ax.scatter(inf_pos * jitter, inf_pos * jitter,
                  s=marker_size(r['vus_count']),
                  c=[gene_colors[r['gene']]],
                  marker='^', alpha=0.8,
                  edgecolors='white', linewidth=2, zorder=3)
        texts.append(ax.text(inf_pos * jitter, inf_pos * jitter, r['gene'],
                            fontsize=9, ha='center', va='center', fontweight='bold'))
    
    for i, r in enumerate(inf_danz):
        jitter = r['auth_metric'] * (1 + 0.02 * (i % 2 - 0.5))
        ax.scatter(jitter, inf_pos,
                  s=marker_size(r['vus_count']),
                  c=[gene_colors[r['gene']]],
                  marker='^', alpha=0.8,
                  edgecolors='white', linewidth=2, zorder=3)
        texts.append(ax.text(jitter, inf_pos, r['gene'],
                            fontsize=9, ha='center', va='center', fontweight='bold'))
    
    for i, r in enumerate(inf_auth):
        jitter = r['danz_metric'] * (1 + 0.02 * (i % 2 - 0.5))
        ax.scatter(inf_pos, jitter,
                  s=marker_size(r['vus_count']),
                  c=[gene_colors[r['gene']]],
                  marker='>', alpha=0.8,
                  edgecolors='white', linewidth=2, zorder=3)
        texts.append(ax.text(inf_pos, jitter, r['gene'],
                            fontsize=9, ha='center', va='center', fontweight='bold'))
    
    # Adjust text to avoid overlaps
    try:
        from adjustText import adjust_text
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.6),
                   expand_points=(1.5, 1.5))
    except ImportError:
        print("Install adjustText: pip install adjustText")
    
    # Labels
    metric_label = 'Diagnostic Odds Ratio' if metric == 'dor' else 'Accuracy'
    ax.set_xlabel(f'Author {metric_label}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'ExCALIBR {metric_label}', fontsize=14, fontweight='bold')
    
    if metric == 'dor':
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlim(plot_min * 0.95, plot_max)
    ax.set_ylim(plot_min * 0.95, plot_max)
    
    ax.grid(True, alpha=0.2, which='both')
    ax.set_facecolor('#FAFAFA')
    
    # Legend with round VUS numbers
    if max_vus < 500:
        vus_legend = [v for v in [50, 100, 250, 500] if min(vus_all) <= v <= max_vus]
    elif max_vus < 2000:
        vus_legend = [v for v in [100, 500, 1000, 2000] if min(vus_all) <= v <= max_vus]
    else:
        vus_legend = [v for v in [500, 1000, 2000, 5000] if min(vus_all) <= v <= max_vus]
    
    # Ensure we have at least 3 legend sizes
    if len(vus_legend) < 3:
        vus_legend = [min(vus_all), np.median(vus_all), max(vus_all)]
    
    legend_elements = [
        plt.scatter([], [], s=marker_size(vus), c='#888', alpha=0.6,
                   edgecolors='white', linewidth=2,
                   label=f'{int(vus):,}')
        for vus in vus_legend
    ]
    
    ax.legend(handles=legend_elements, loc='lower left',
             fontsize=11, frameon=True, edgecolor='#999', framealpha=0.95,
             title='VUS count', title_fontsize=11,
             labelspacing=2, handletextpad=2.5)
    
    plt.tight_layout()
    return fig, gene_results




def plot_combined_evidence_distributions(
    author_assignments, author_annotations,
    clinvar_assignments, clinvar_classes,
    save_path=None, figsize=(14, 10)):
    """
    Create combined figure with author and ClinVar evidence distributions.
    
    Parameters
    ----------
    author_assignments : array-like
        Evidence point assignments for author annotation subset
    author_annotations : array-like
        Author functional annotations (0=Normal, 1=Indeterminate, 2=Abnormal)
    clinvar_assignments : array-like
        Evidence point assignments for ClinVar subset
    clinvar_classes : array-like or multi-label array
        ClinVar classes (0=B/LB, 1=VUS, 2=P/LP, 3=gnomAD, 4=Synonymous)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    
    # Evidence strength colors
    strength_colors = {
        8: '#943744', 4: '#B85C6B', 2: '#D68F99', 1: '#E6B1B8',
        0: '#E0E0E0',
        -1: '#99C8DC', -2: '#7AB5D1', -4: '#4B91A6', -8: '#2E6B7E',
    }
    
    point_values = [8, 4, 2, 1, 0, -1, -2, -4, -8]
    
    map_point_to_text = {
        -8: "-8", -4: "-4", -2: "-2", -1: "-1",
        0: "0",
        1: "+1", 2: "+2", 4: "+4", 8: "+8",
    }
    
    def get_text_color(point_val):
        """Return white for dark backgrounds, black for light"""
        if point_val in [8, 4, -4, -8]:
            return 'white'
        else:
            return 'black'
    
    # Create figure with 2 subplots
    fig, (ax_author, ax_clinvar) = plt.subplots(2, 1, figsize=figsize, 
                                                gridspec_kw={'hspace': 0.15}, height_ratios=[3,4])
    
    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 12
    FONTSIZE_PANEL_LETTER = 16
    PANEL_LETTER_Y = 1.05
    
    # ========== PANEL A: AUTHOR ANNOTATIONS ==========
    
    category_labels_author = {2: 'Abnormal', 1: 'Indeterminate', 0: 'Normal'}
    category_order_author = ['Normal', 'Indeterminate', 'Abnormal']
    
    dist_author = compute_evidence_distribution(author_assignments, author_annotations, 
                                                category_labels_author)
    dist_author = dist_author.loc[[cat for cat in category_order_author if cat in dist_author.index]]
    
    # Plot stacked bars
    left = np.zeros(len(dist_author))
    mapped_points_author = map_point_values_for_display(author_assignments)
    
    for point_val in point_values:
        if point_val not in dist_author.columns:
            continue
        
        widths = dist_author[point_val].values
        
        ax_author.barh(range(len(dist_author)), widths, left=left,
                      color=strength_colors[point_val], edgecolor='white', linewidth=0.5)
        
        # Add count labels
        for row_idx, width in enumerate(widths):
            if width > 4:
                cat = dist_author.index[row_idx]
                cat_val = [k for k, v in category_labels_author.items() if v == cat][0]
                cat_mask = author_annotations == cat_val
                count = (mapped_points_author[cat_mask] == point_val).sum()
                
                ax_author.text(left[row_idx] + width / 2, row_idx, f"{count:,}",
                             ha='center', va='center', fontsize=12, 
                             color=get_text_color(point_val))
        
        left += widths
    
    # Y-axis labels with counts
    y_labels_author = []
    for cat in dist_author.index:
        cat_val = [k for k, v in category_labels_author.items() if v == cat][0]
        n = (author_annotations == cat_val).sum()
        y_labels_author.append(f"{cat}\n(n={n:,})")
    
    ax_author.set_yticks(range(len(dist_author)))
    ax_author.set_yticklabels(y_labels_author, fontsize=FONTSIZE_AXIS_TICK)
    ax_author.set_xlim(0, 100)
    # ax_author.set_xlabel('Percentage of Variants', fontsize=FONTSIZE_AXIS_LABEL)#, fontweight='bold')
    ax_author.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    ax_author.set_axisbelow(True)
    ax_author.spines['top'].set_visible(False)
    ax_author.spines['right'].set_visible(False)
    ax_author.text(-0.08, PANEL_LETTER_Y, '(A)', transform=ax_author.transAxes,
                  fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')
    ax_author.tick_params(labelsize=FONTSIZE_AXIS_TICK)
    
    # ========== PANEL B: CLINVAR CLASSIFICATIONS ==========
    
    category_labels_clinvar = {2: 'P/LP', 1: 'VUS', 3: 'gnomAD', 0: 'B/LB'}
    category_order_clinvar = ['P/LP', 'VUS', 'gnomAD', 'B/LB'][::-1]
    
    dist_clinvar = compute_evidence_distribution(clinvar_assignments, clinvar_classes,
                                                 category_labels_clinvar)
    dist_clinvar = dist_clinvar.loc[[cat for cat in category_order_clinvar if cat in dist_clinvar.index]]
    
    # Check if multi-label
    first_cat = clinvar_classes[0] if len(clinvar_classes) > 0 else None
    is_multilabel = isinstance(first_cat, (list, np.ndarray))
    
    # Plot stacked bars
    left = np.zeros(len(dist_clinvar))
    mapped_points_clinvar = map_point_values_for_display(clinvar_assignments)
    
    for point_val in point_values:
        if point_val not in dist_clinvar.columns:
            continue
        
        widths = dist_clinvar[point_val].values
        
        ax_clinvar.barh(range(len(dist_clinvar)), widths, left=left,
                       color=strength_colors[point_val], edgecolor='white', linewidth=0.5)
        
        # Add count labels
        for row_idx, width in enumerate(widths):
            if width > 4:
                cat = dist_clinvar.index[row_idx]
                cat_idx = [k for k, v in category_labels_clinvar.items() if v == cat][0]
                
                if is_multilabel:
                    cat_mask = np.array([c[cat_idx] if len(c) > cat_idx else False 
                                        for c in clinvar_classes])
                else:
                    cat_mask = clinvar_classes == cat_idx
                
                count = (mapped_points_clinvar[cat_mask] == point_val).sum()
                
                ax_clinvar.text(left[row_idx] + width / 2, row_idx, f"{count:,}",
                              ha='center', va='center', fontsize=12, 
                              color=get_text_color(point_val))
        
        left += widths
    
    # Y-axis labels with counts
    y_labels_clinvar = []
    for cat in dist_clinvar.index:
        cat_idx = [k for k, v in category_labels_clinvar.items() if v == cat][0]
        
        if is_multilabel:
            n = sum([c[cat_idx] if len(c) > cat_idx else False for c in clinvar_classes])
        else:
            n = (clinvar_classes == cat_idx).sum()
        
        y_labels_clinvar.append(f"{cat}\n(n={n:,})")
    
    ax_clinvar.set_yticks(range(len(dist_clinvar)))
    ax_clinvar.set_yticklabels(y_labels_clinvar, fontsize=FONTSIZE_AXIS_TICK)
    ax_clinvar.set_xlim(0, 100)
    ax_clinvar.set_xlabel('Percentage of Variants', fontsize=FONTSIZE_AXIS_LABEL)#, fontweight='bold')
    ax_clinvar.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    ax_clinvar.set_axisbelow(True)
    ax_clinvar.spines['top'].set_visible(False)
    ax_clinvar.spines['right'].set_visible(False)
    ax_clinvar.text(-0.08, PANEL_LETTER_Y, '(B)', transform=ax_clinvar.transAxes,
                   fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')
    ax_clinvar.tick_params(labelsize=FONTSIZE_AXIS_TICK)
    
    # ========== SHARED LEGEND ==========
    
    legend_elements = [
        Patch(facecolor=strength_colors[p], label=map_point_to_text[p], edgecolor='none')
        for p in point_values
    ]
    
    # Place legend below both plots
    fig.legend(
        handles=legend_elements,
        title='Evidence Strength',
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_elements),
        frameon=True,
        fontsize=12,
        title_fontsize=12,
        handlelength=1.0
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined evidence distribution plot saved to: {save_path}")
    
    return fig





import numpy as np
import pandas as pd


def compute_genewise_evidence_table(
    all_danz_oob, all_author, dataset_info_df,
    all_danz_oob_full, all_clinvar_full, dataset_info_df_full,
):
    """
    Compute gene-wise statistics of evidence direction vs. functional/clinical category.
    """

    AUTHOR_LABELS = {0: 'Normal', 1: 'Indeterminate', 2: 'Abnormal'}
    CLINVAR_LABELS = {0: 'B/LB', 1: 'VUS', 2: 'P/LP', 3: 'gnomAD'}
    clinvar_cat_order = ["P/LP", "VUS", "gnomAD", "B/LB"]

    DIRECTION_NAMES = ['Pathogenic (>0)', 'Indeterminate (0)', 'Benign (<0)']

    def _classify_direction(points):
        out = np.empty(len(points), dtype=object)
        out[points > 0] = 'Pathogenic (>0)'
        out[points == 0] = 'Indeterminate (0)'
        out[points < 0] = 'Benign (<0)'
        return out

    def _build_gene_array(dataset_info):
        genes = []
        for _, row in dataset_info.iterrows():
            gene = row['dataset'].split('_')[0]
            genes.extend([gene] * row['n_variants'])
        return np.array(genes)

    def _compute_stats(assignments, categories, cat_labels, gene_arr, multi_label):
        directions = _classify_direction(assignments)
        genes_unique = sorted(np.unique(gene_arr))

        records = []
        for gene in genes_unique:
            gene_mask = gene_arr == gene
            row = {'Gene': gene}
            for cat_idx, cat_name in cat_labels.items():
                cat_mask = gene_mask & (categories[:, cat_idx].astype(bool) if multi_label else (categories == cat_idx))
                n_cat = cat_mask.sum()
                for d in DIRECTION_NAMES:
                    n_d = ((directions == d) & cat_mask).sum()
                    pct = (n_d / n_cat * 100) if n_cat > 0 else 0.0
                    row[(cat_name, d, 'n')] = int(n_d)
                    row[(cat_name, d, '%')] = pct
                row[(cat_name, 'Total', 'n')] = int(n_cat)
            records.append(row)

        row = {'Gene': 'All'}
        for cat_idx, cat_name in cat_labels.items():
            cat_mask = categories[:, cat_idx].astype(bool) if multi_label else (categories == cat_idx)
            n_cat = cat_mask.sum()
            for d in DIRECTION_NAMES:
                n_d = ((directions == d) & cat_mask).sum()
                pct = (n_d / n_cat * 100) if n_cat > 0 else 0.0
                row[(cat_name, d, 'n')] = int(n_d)
                row[(cat_name, d, '%')] = pct
            row[(cat_name, 'Total', 'n')] = int(n_cat)
        records.append(row)

        df = pd.DataFrame(records).set_index('Gene')
        tuples = [c for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(tuples, names=['Category', 'Direction', 'Stat'])
        return df

    gene_arr_author = _build_gene_array(dataset_info_df)
    gene_arr_clinvar = _build_gene_array(dataset_info_df_full)

    assert len(gene_arr_author) == len(all_danz_oob) == len(all_author)
    assert len(gene_arr_clinvar) == len(all_danz_oob_full) == len(all_clinvar_full)

    author_table = _compute_stats(all_danz_oob, all_author, AUTHOR_LABELS, gene_arr_author, multi_label=False)
    clinvar_table = _compute_stats(all_danz_oob_full, all_clinvar_full, CLINVAR_LABELS, gene_arr_clinvar, multi_label=True)

    # ---- Build LaTeX ----

    def _fmt_pct(pct):
        if abs(pct - 100.0) < 0.05:
            return '100\\'
        elif abs(pct) < 0.05:
            return '0\\'
        else:
            return f'{pct:.1f}\\'

    def _df_to_latex_block(df, cat_labels, panel_label, label_tag):
        if "auth" in label_tag:
            cat_names = [cat_labels[k] for k in sorted(cat_labels.keys())][::-1]
        else:
            cat_names = clinvar_cat_order
        n_cat = len(cat_names)
        # Tight column spec: no dead column, @{} to remove outer padding
        col_spec = '@{}l' + ' rrrr' * n_cat + '@{}'

        lines = []
        lines.append(f'% ===== {panel_label} =====')
        lines.append(r'{\setlength{\tabcolsep}{3.5pt}')
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        # Header row 1: panel label + category names
        hdr1 = f'\\multicolumn{{1}}{{@{{}}l}}{{{panel_label}}}'
        for cn in cat_names:
            hdr1 += f' & \\multicolumn{{4}}{{c}}{{{cn}}}'
        hdr1 += r' \\'
        lines.append(hdr1)

        # cmidrules per category
        for i in range(n_cat):
            start = 2 + i * 4
            end = start + 3
            lines.append(r'\cmidrule(lr){' + str(start) + '-' + str(end) + '}')

        # Header row 2: sub-columns
        hdr2 = 'Gene'
        for cn in cat_names:
            hdr2 += ' & $n$ & Path. & Ind. & Ben.'
        hdr2 += r' \\'
        lines.append(hdr2)
        lines.append(r'\midrule')

        for gene in df.index:
            if gene == 'All':
                gene_str = '\\textbf{All}'
            else:
                gene_str = '\\textit{' + gene.replace('_', r'\_') + '}'

            parts = [gene_str]
            for cn in cat_names:
                total = int(df.loc[gene, (cn, 'Total', 'n')])
                parts.append(f'{total:,}')
                for d in DIRECTION_NAMES:
                    pct = df.loc[gene, (cn, d, '%')]
                    parts.append(_fmt_pct(pct))
            row_str = ' & '.join(parts) + r' \\'
            if gene == 'All':
                lines.append(r'\midrule')
            lines.append(row_str)

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}}')  # closes \setlength group
        return '\n'.join(lines)

    latex_author = _df_to_latex_block(author_table, AUTHOR_LABELS, 'Author Category', 'tab:gene_evidence_distr_auth')
    latex_clinvar = _df_to_latex_block(clinvar_table, CLINVAR_LABELS, 'Variant Group', 'tab:gene_evidence_distr_clinvar')

    author_caption = (
        r'Gene-wise distribution of out-of-bag evidence direction assigned by \excalibr '
        r'for variants with author-provided functional annotations. '
        r'For each gene and author category (Abnormal, Indeterminate, Normal), '
        r'the percentage of variants receiving pathogenic ($>0$), indeterminate ($=0$), '
        r'or benign ($<0$) evidence points is shown, alongside the total number of variants ($n$).'
    )

    clinvar_caption = (
        r'Gene-wise distribution of out-of-bag evidence direction assigned by \excalibr '
        r'for variants stratified by variant group. '
        r'For each gene and variant group (P/LP, VUS, gnomAD, B/LB), '
        r'the percentage of variants receiving pathogenic ($>0$), indeterminate ($=0$), '
        r'or benign ($<0$) evidence points is shown, alongside the total number of variants ($n$).'
    )

    latex_str = (
        '\\begin{table}[!htb]\n'
        '\\centering\n'
        '\\footnotesize\n'
        f'\\resizebox{{\\textwidth}}{{!}}{{\n'
        f'{latex_author}\n'
        f'}}\n'
        f'\\caption{{{author_caption}}}\n'
        '\\label{tab:gene_evidence_distr_auth}\n'
        '\\end{table}\n'
        '\n'
        '\\begin{table}[!htb]\n'
        '\\centering\n'
        '\\footnotesize\n'
        f'\\resizebox{{\\textwidth}}{{!}}{{\n'
        f'{latex_clinvar}\n'
        f'}}\n'
        f'\\caption{{{clinvar_caption}}}\n'
        '\\label{tab:gene_evidence_distr_clinvar}\n'
        '\\end{table}'
    )

    return author_table, clinvar_table, latex_str


