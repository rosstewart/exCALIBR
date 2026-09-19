import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [
    'Arial',
    'Helvetica',
    'Nimbus Sans',
    'DejaVu Sans'
]

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from src.assay_calibration.plot_utils.utils import sample_density, _bold_italic_gene_title

# Global styling constants
FONTSIZE_PANEL_LETTER = 24
FONTSIZE_TITLE = 18
FONTSIZE_SUBTITLE = 12
FONTSIZE_AXIS_LABEL = 11
FONTSIZE_TICK = 9
FONTSIZE_LEGEND = 9
FONTSIZE_ANNOTATION = 10

TITLE_SPACER = 0.10

SAMPLE_NAMES = ["ClinVar PLP", "ClinVar BLB", "gnomAD", "Synonymous"]
SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_ALPHAS = [0.7, 0.5, 0.15, 0.4]

STRENGTH_COLOR = {
    -8: '#4b91a6', -7: '#5DA3BD', -6: '#6FAACE', -5: '#74ABCE',
    -4: '#7ab5d1', -3: '#99c8dc', -2: '#d0e8f0', -1: '#e4f1f6',
    0: '#e0e0e0',
    1: '#e6b1b8', 2: '#d68f99', 3: '#ca7682', 4: '#b85c6b',
    5: '#B1535F', 6: '#AA4E58', 7: '#A2484F', 8: '#943744'
}

BENIGN_THRESHOLD_COLOR = '#2166AC'
PATHOGENIC_THRESHOLD_COLOR = '#B2182B'


def plot_figure_panel_a(scoreset_2018, indv_summary, fits, score_range, flipped, n_samples,
                        layout='horizontal', minimal=False, figsize=None):
    """
    Panel A: Multi-sample mixture model calibration
    
    Parameters
    ----------
    layout : str, optional
        'horizontal' for column layout (default), 'vertical' for row layout
    figsize : tuple, optional
        Figure size. Defaults: (12.25, 3.5) for horizontal, (6.6, 7.3) for vertical
    
    Returns
    -------
    fig : matplotlib figure
    """
    gene = scoreset_2018.scoreset_name.split("_")[0]
    
    # Set default figsize based on layout
    if figsize is None:
        figsize = (12.25, 3.5) if layout == 'horizontal' else (6.6, 7.3)
    
    fig = plt.figure(figsize=figsize)
    
    x_min, x_max = score_range[0], score_range[-1]
    bin_width = (x_max - x_min) / 50
    point_ranges = indv_summary['point_ranges']
    
    # Pre-compute thresholds
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    threshold_info = []
    
    for idx, point_val in enumerate([1, 2, 3, 4, 8]):
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val and score_ranges_pr:
                threshold_score = score_ranges_pr[0][0] if not flipped else score_ranges_pr[0][1]
                threshold_info.append((pv, threshold_score, BENIGN_THRESHOLD_COLOR, linestyles[idx], 1.0))
                break
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val and score_ranges_pr:
                threshold_score = score_ranges_pr[0][1] if not flipped else score_ranges_pr[0][0]
                threshold_info.append((pv, threshold_score, PATHOGENIC_THRESHOLD_COLOR, linestyles[idx], 1.0))
                break
    
    num_skipped = 0
    
    if layout == 'horizontal':
        # Original horizontal layout
        gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                    height_ratios=[0.08, 1], hspace=TITLE_SPACER)
        gs = gridspec.GridSpecFromSubplotSpec(1, n_samples, subplot_spec=gs_outer[1], wspace=0.10)
        
        first_max_density = None
        for ax_idx, sample_num in enumerate([1, 0, 2, 3] if n_samples == 4 else [1, 0, 2]):
            if scoreset_2018.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            
            sample_idx = sample_num - num_skipped
            ax = plt.subplot(gs[ax_idx])
            
            sample_mask = scoreset_2018.sample_assignments[:, sample_idx]
            color = SAMPLE_COLORS[sample_num]
            hist_data = scoreset_2018.scores[sample_mask]
            n_count = sample_mask.sum()
            
            sns.histplot(hist_data, binwidth=bin_width, stat='density', ax=ax,
                       alpha=0.5 if sample_num != 0 else 0.7, color=color)
            
            density_sample = sample_density(score_range, fits, sample_idx)
            d_total = np.nansum(density_sample, axis=1)
            d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)
            
            ax.fill_between(score_range, d_total_perc[0], d_total_perc[2], color='gray', alpha=0.3)
            ax.plot(score_range, d_total_perc[1], color='black', alpha=0.65, linewidth=2.5)
            
            ax.set_xlim(x_min, x_max)
            max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0
            if ax_idx == 0:
                first_max_density = max_hist_density
            ax.set_ylim(0, 1.2*first_max_density)
            ax.set_xlabel('Fitness score' if not minimal else 'Experimental score', fontsize=FONTSIZE_AXIS_LABEL)
            ax.set_ylabel('' if ax_idx != 0 else 'Density')
            ax.tick_params(axis='both', labelsize=FONTSIZE_TICK, left=ax_idx == 0, labelleft=ax_idx == 0)
            
            for pv, thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
                if abs(pv) in [1, 2, 4, 8]:
                    ax.axvline(thresh_score, color=thresh_color, linestyle=thresh_ls, 
                              linewidth=1.5, alpha=0.8)
            
            face_rgba = to_rgba(color, 0.5 if sample_num != 0 else 0.7)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')

            if not minimal:
                if sample_num == 2:
                    legend_label = f'{SAMPLE_NAMES[sample_idx]}\nprior: {indv_summary["prior"]:.3f}\n(n={n_count:,})'
                else:
                    legend_label = f'{SAMPLE_NAMES[sample_idx]}\n(n={n_count:,})'
            else:
                if sample_num == 2:
                    legend_label = f'{SAMPLE_NAMES[sample_idx]}\nprior: {indv_summary["prior"]:.3f}'
                else:
                    legend_label = f'{SAMPLE_NAMES[sample_idx]}'
            
            loc = 'upper left' if (sample_num == 0 and flipped) or (sample_num != 0 and not flipped) else 'upper right'
            ax.legend([hist_patch], [legend_label], loc=loc, fontsize=FONTSIZE_LEGEND, framealpha=0.9)
        
        # Add centered title in title row
        bbox = plt.subplot(gs_outer[0])
        bbox.axis('off')
    
    else:  # vertical layout
        # Vertical layout similar to four_datasets_publication
        gs = GridSpec(n_samples, 1, figure=fig, hspace=0.08)
        
        for sample_num in range(len(scoreset_2018.sample_counts)):
            if scoreset_2018.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            
            sample_idx = sample_num - num_skipped
            ax = fig.add_subplot(gs[sample_idx, 0])
            
            sample_mask = scoreset_2018.sample_assignments[:, sample_idx]
            color = SAMPLE_COLORS[sample_num]
            hist_data = scoreset_2018.scores[sample_mask]
            n_count = sample_mask.sum()
            
            # Plot histogram
            sns.histplot(hist_data, binwidth=bin_width, stat='density', ax=ax,
                       alpha=0.5 if sample_num != 0 else 0.7, color=color)

            ax.set_xlim(x_min, x_max)
            max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0
            
            # Plot fitted density
            density_sample = sample_density(score_range, fits, sample_idx)
            d_total = np.nansum(density_sample, axis=1)
            d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)
            
            ax.plot(score_range, d_total_perc[1], color='black', alpha=0.5, linewidth=2)
            ax.fill_between(score_range, d_total_perc[0], d_total_perc[2], color='gray', alpha=0.3)
            
            # Add threshold lines
            for pv, thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
                if abs(pv) in [1, 2, 4, 8]:
                    ax.axvline(thresh_score, color=thresh_color, linestyle=thresh_ls, 
                              linewidth=1.5, alpha=0.8)
            
            # Title on first sample only
            # if sample_idx == 0:
            #     ax.set_title(rf"$\mathbfit{{{gene}}}$ experimental scores",
            #                fontsize=FONTSIZE_SUBTITLE, fontweight='bold', pad=8)
            
            # X-axis only on last sample
            is_last_sample = (sample_num == len(scoreset_2018.sample_counts) - 1 or
                            (sample_num == len(scoreset_2018.sample_counts) - 2 and
                             scoreset_2018.sample_counts[-1] == 0))
            
            if is_last_sample:
                ax.set_xlabel("Fitness score" if not minimal else "Experimental score", fontsize=FONTSIZE_AXIS_LABEL)
            else:
                ax.set_xticks([])
                ax.set_xlabel("")
            
            ax.set_ylabel("Density", fontsize=FONTSIZE_AXIS_LABEL)
            
            # Create histogram legend handle
            face_rgba = to_rgba(color, 0.5 if sample_num != 0 else 0.7)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')

            if not minimal:
                if sample_num == 2:
                    hist_label = f'{SAMPLE_NAMES[sample_num]}\n(n={n_count:,d}, prior={indv_summary["prior"]:.3f})'
                else:
                    hist_label = f'{SAMPLE_NAMES[sample_num]}\n(n={n_count:,d})'
            else:
                if sample_num == 2:
                    hist_label = f'{SAMPLE_NAMES[sample_idx]}\nprior: {indv_summary["prior"]:.3f}'
                else:
                    hist_label = f'{SAMPLE_NAMES[sample_idx]}'
                
            # Create histogram legend on the left
            ax.legend([hist_patch], [hist_label],
                     loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.8)
            
            ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=FONTSIZE_TICK)
    
    plt.tight_layout()
    return fig


def plot_figure_panel_b(scoreset, indv_summary, score_range, flipped,
                        use_twin_axes=True, minimal=False, figsize=(6.6, 7.3)):
    """
    Panel B: Experimental score calibration comparison
    
    Parameters
    ----------
    use_twin_axes : bool, optional
        If True (default), plot control variants on left y-axis (count) and all SNVs on right y-axis (count).
        If False, plot all samples on single y-axis with density stat.
    figsize : tuple, optional
        Figure size (default: (6.6, 7.3))
    
    Returns
    -------
    fig : matplotlib figure
    legend_handles : list of patch handles for legend
    """

    all_scores = scoreset.snv_scores
    point_ranges = indv_summary['point_ranges']
    gene = scoreset.scoreset_name.split("_")[0]
    
    fig = plt.figure(figsize=figsize)
    
    # Add title row
    gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                height_ratios=[0.06, 1], hspace=TITLE_SPACER)
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[1],
                                         height_ratios=[1.5, 0.3, 0.3], hspace=0.3)
    
    x_min, x_max = score_range[0], score_range[-1]
    bin_width = (x_max - x_min) / 50
    
    # Histogram
    ax_hist = plt.subplot(gs[0])
    
    sample_handles = []
    num_skipped = 0
    
    if use_twin_axes:
        # Original behavior: twin axes with count stat
        ax_twin = ax_hist.twinx()
        
        for sample_num in [1, 0, 2]:
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            
            sample_idx = sample_num - num_skipped
            sample_mask = scoreset.sample_assignments[:, sample_idx]
            color = SAMPLE_COLORS[sample_num]
            alpha = SAMPLE_ALPHAS[sample_num]
            
            if sample_num == 2:
                hist_data = all_scores
                display_name = 'All SNVs'
                n_count = len(all_scores)
                sns.histplot(hist_data, binwidth=bin_width, stat='count', ax=ax_twin,
                           alpha=alpha, color=color)
            else:
                hist_data = scoreset.scores[sample_mask]
                display_name = SAMPLE_NAMES[sample_num]
                n_count = sample_mask.sum()
                sns.histplot(hist_data, binwidth=bin_width, stat='count', ax=ax_hist,
                           alpha=alpha, color=color)
            
            face_rgba = to_rgba(color, alpha)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')
            sample_handles.append((hist_patch, f'{display_name}\n(n={n_count:,})'))
        
        ax_hist.set_xlim(x_min, x_max)
        ax_twin.set_xlim(x_min, x_max)
        ax_hist.set_ylim(0, 1.18*max([patch.get_height() for patch in ax_hist.patches]) if ax_hist.patches else 1.0)
        ax_twin.set_ylim(0, 1.18*max([patch.get_height() for patch in ax_twin.patches]) if ax_twin.patches else 1.0)
        
        ax_hist.set_xlabel('')
        ax_hist.set_ylabel('Control variant count', fontsize=FONTSIZE_AXIS_LABEL)
        ax_twin.set_ylabel('SNV count', fontsize=FONTSIZE_AXIS_LABEL)
        
        ax_hist.tick_params(axis='both', labelsize=FONTSIZE_TICK)
        ax_twin.tick_params(axis='both', labelsize=FONTSIZE_TICK)
        
    else:
        # Single axis with density stat
        for sample_num in [1, 0, 2]:
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            
            sample_idx = sample_num - num_skipped
            sample_mask = scoreset.sample_assignments[:, sample_idx]
            color = SAMPLE_COLORS[sample_num]
            alpha = SAMPLE_ALPHAS[sample_num]
            
            if sample_num == 2:
                hist_data = all_scores
                display_name = 'All SNVs'
                n_count = len(all_scores)
            else:
                hist_data = scoreset.scores[sample_mask]
                display_name = SAMPLE_NAMES[sample_num]
                n_count = sample_mask.sum()
            
            sns.histplot(hist_data, binwidth=bin_width, stat='density', ax=ax_hist,
                       alpha=alpha, color=color)
            
            face_rgba = to_rgba(color, alpha)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')
            sample_handles.append((hist_patch, f'{display_name}\n(n={n_count:,})'))
        
        ax_hist.set_xlim(x_min, x_max)
        ax_hist.set_ylim(0, 1.18*max([patch.get_height() for patch in ax_hist.patches]) if ax_hist.patches else 1.0)
        
        ax_hist.set_xlabel('')
        ax_hist.set_ylabel('Density', fontsize=FONTSIZE_AXIS_LABEL)
        ax_hist.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    
    ax_hist.legend([h[0] for h in sample_handles], [h[1] for h in sample_handles],
                  loc='upper right' if flipped else 'upper left', fontsize=FONTSIZE_LEGEND)

    if not minimal:
        _bold_italic_gene_title(ax_hist, gene, ' fitness scores',
                                 x=0.5, y=0.98, fontsize=FONTSIZE_SUBTITLE,
                                 ha='center', va='top')
    
    # ExCALIBR bar (only this, no Scott bar)
    ax_excalibr = plt.subplot(gs[1])
    
    intervals = []
    for pv in sorted([p for p in point_ranges.keys() if p != 0]):
        if point_ranges[pv]:
            sr = point_ranges[pv][0]
            intervals.append((pv, sr[0], sr[1]))
    
    neg_int = [(pv, s, e) for pv, s, e in intervals if pv < 0]
    pos_int = [(pv, s, e) for pv, s, e in intervals if pv > 0]
    
    if neg_int and pos_int:
        neg_sorted = sorted(neg_int, key=lambda x: x[2])
        pos_sorted = sorted(pos_int, key=lambda x: x[1])
        ir_start = neg_sorted[-1][2] if flipped else pos_sorted[-1][2]
        ir_end = pos_sorted[0][1] if flipped else neg_sorted[0][1]
        intervals.append((0, ir_start, ir_end))
    
    intervals_sorted = sorted(intervals, key=lambda x: x[1])

    # Set limits before drawing/measuring the count labels below -- their
    # fit test needs ax_excalibr.transData to already reflect the final
    # data range, not the default [0, 1) autoscale a fresh subplot starts
    # with.
    ax_excalibr.set_xlim(x_min, x_max)
    ax_excalibr.set_ylim(0, 1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for point_val, start, end in intervals_sorted:
        # The outermost point value's interval is unbounded (start=-inf or
        # end=inf, from point_ranges' calibration.json) -- axvspan builds a
        # Rectangle from (start, end) directly, and start + width where
        # width = end - start works out to -inf + inf = NaN whenever start
        # itself is -inf, which silently renders nothing (the "left half of
        # the ExCALIBR bar is blank white" bug, see analysis/figure4/panels.py
        # plot_panel_b for the same fix). Clip to the axis's own finite data
        # range for drawing only; `count` below still uses the true (possibly
        # infinite) bounds so it keeps including every variant in that tier,
        # not just the ones inside x_min/x_max.
        draw_start, draw_end = max(start, x_min), min(end, x_max)
        ax_excalibr.axvspan(draw_start, draw_end, color=STRENGTH_COLOR[point_val], alpha=1.0)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        text_color = 'white' if abs(point_val) >= 7 else 'black'
        t = ax_excalibr.text((draw_start + draw_end) / 2, 0.5, f'{count:,}',
                           ha='center', va='center', fontsize=FONTSIZE_ANNOTATION, color=text_color)
        # A fixed segment-width threshold (e.g. "> 0.02 score units") can't
        # tell "59" from "2,044" apart -- the same narrow segment that fits
        # a 2-digit count would let a 5-digit one overflow into its
        # neighbors. Measure the actual rendered text width against the
        # segment's own rendered width instead, and drop the label (rather
        # than let it spill over) if it doesn't fit.
        text_width_px = t.get_window_extent(renderer).width
        x0_px = ax_excalibr.transData.transform((draw_start, 0.5))[0]
        x1_px = ax_excalibr.transData.transform((draw_end, 0.5))[0]
        if text_width_px > abs(x1_px - x0_px):
            t.remove()

    ax_excalibr.set_yticks([])
    ax_excalibr.set_xlabel(rf'$\mathit{{{gene}}}$ fitness score' if not minimal else "Experimental score", fontsize=FONTSIZE_AXIS_LABEL)
    ax_excalibr.tick_params(axis='x', labelsize=FONTSIZE_TICK)
    ax_excalibr.set_title('ExCALIBR calibration', loc='left', pad=3, fontsize=FONTSIZE_SUBTITLE, style='italic')
    
    # Create legend handles
    legend_order = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8] if flipped else [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
    
    point_labels = {
        -8: "-8 (very strong)", -4: "-4 (strong)", -3: "-3", -2: "-2 (moderate)", -1: "-1 (supporting)",
        0: "0 (indeterminate)",
        1: "+1 (supporting)", 2: "+2 (moderate)", 3: "+3", 4: "+4 (strong)", 8: "+8 (very strong)"
    }

    legend_handles = [Patch(facecolor=STRENGTH_COLOR[pv], label=point_labels.get(pv, f"{pv:+d}").split(" (")[0], edgecolor='none')
                     for pv in legend_order if any(p == pv for p, _, _ in intervals_sorted)]
        

    # Combine and deduplicate legend handles
    # Create a dict to maintain order and avoid duplicates by label
    combined_handles = {}
    for handle in legend_handles:
        label = handle.get_label()
        if label not in combined_handles:
            combined_handles[label] = handle
    
    
    # Convert back to list
    final_handles = sorted(list(combined_handles.values()), key=lambda x: int(x.get_label().split(' ')[0]), reverse=not flipped)
    
    ax_legend = plt.subplot(gs[2])
    ax_legend.axis('off')
    
    if len(final_handles) > 17: # impossible statement
        # Create ordered lists by point value
        handles_dict = {}
        for h in final_handles:
            point_val = int(h.get_label().split(' ')[0])
            handles_dict[point_val] = h
        
        # Row 1: -8, -6, -4, -2, 0, 2, 4, 6, 8 (evens + 0)
        # Row 2: -7, -5, -3, -1, SKIP, 1, 3, 5, 7 (odds)
        
        row1 = []
        row2 = []
        
        # Negative evens
        for val in [-8, -6, -4, -2]:
            if val in handles_dict:
                row1.append(handles_dict[val])
        
        # Negative odds
        for val in [-7, -5, -3, -1]:
            if val in handles_dict:
                row2.append(handles_dict[val])
        
        # Zero (top row only)
        if 0 in handles_dict:
            row1.append(handles_dict[0])
            # Add spacer to row 2
            row2.append(Patch(facecolor='none', edgecolor='none', label=''))
        
        # Positive evens
        for val in [2, 4, 6, 8]:
            if val in handles_dict:
                row1.append(handles_dict[val])
        
        # Positive odds
        for val in [1, 3, 5, 7]:
            if val in handles_dict:
                row2.append(handles_dict[val])
        
        # Combine: row1 fills first, then row2
        reordered_handles = row1 + row2
        ncol = len(row1)  # This makes it wrap after row1 is complete
        
        legend = ax_legend.legend(
            handles=reordered_handles,
            loc='center',
            ncol=ncol,
            frameon=True,
            fontsize=FONTSIZE_LEGEND,
            columnspacing=0.5,
            handletextpad=0.4,
            handlelength=0.8,
            borderpad=0.7
        )
    else:
        # Single row for 14 or fewer items
        ncol = len(final_handles)
        legend = ax_legend.legend(
            handles=final_handles, 
            loc='center', 
            ncol=ncol,
            frameon=True,
            fontsize=FONTSIZE_LEGEND,
            columnspacing=0.5, 
            handletextpad=0.4, 
            handlelength=0.8,
            borderpad=0.7
        )

    plt.tight_layout()
    return fig#, legend_handles

