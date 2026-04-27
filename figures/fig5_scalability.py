"""
Fig. 5: Plateau line plots at k=3 (A, B), k=4 (C, D), then heatmaps (E, F)

Layout: 3 rows × 2 cols
  Row 0: A = SNM plateau k=3,  B = ARM plateau k=3
  Row 1: C = SNM plateau k=4,  D = ARM plateau k=4
  Row 2: E = SNM heatmap,      F = ARM heatmap

Data:
  Heatmaps : result_for_scalability.pkl
  Plateaus  : result_for_plateau_k34.pkl  (N=20..100, k=3,4, 200 trials)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# ── Load data ─────────────────────────────────────────────────────────────────
df_sc = pd.read_pickle('result_for_scalability.pkl')
df_pl = pd.read_pickle('result_for_plateau_k34.pkl')

# ── Heatmap matrices ──────────────────────────────────────────────────────────
solve_rate = (df_sc.groupby(['label', 'node_num', 'color_num'])['solved']
                   .mean().reset_index()
                   .rename(columns={'solved': 'solution_rate'}))

n_vals = sorted(df_sc['node_num'].unique())
c_vals = sorted(df_sc['color_num'].unique())

def make_matrix(label):
    sub = solve_rate[solve_rate['label'] == label]
    mat = np.full((len(n_vals), len(c_vals)), np.nan)
    for i, n in enumerate(n_vals):
        for j, c in enumerate(c_vals):
            row = sub[(sub['node_num'] == n) & (sub['color_num'] == c)]
            if not row.empty:
                mat[i, j] = row['solution_rate'].values[0]
    return mat

mat_noise = make_matrix('Noise')
mat_async = make_matrix('Async')

# ── Plateau data prep ─────────────────────────────────────────────────────────
n_plateau = sorted(df_pl['node_num'].unique())

def get_plateau(label, k, param_col):
    sub = df_pl[(df_pl['label'] == label) & (df_pl['color_num'] == k)]
    return (sub.groupby(['node_num', param_col])['solved']
               .agg(['mean', 'std', 'count'])
               .reset_index()
               .rename(columns={'mean': 'rate', 'std': 'sd'}))

# ── Color scheme ──────────────────────────────────────────────────────────────
cmap_heat = LinearSegmentedColormap.from_list(
    'sol', ['#F7F7F7', '#92C5DE', '#0571B0'], N=256)

# Model colours — consistent with Fig.2, 3, 4
_cb     = sns.color_palette('colorblind')
COL_SNM = _cb[0]   # blue
COL_ARM = _cb[1]   # orange

# Perceptually uniform, colorblind-safe gradient for N values
n_colors = len(n_plateau)
palette  = [plt.cm.viridis(i / (n_colors - 1)) for i in range(n_colors)]
markers  = ['o', 's', '^', 'D', 'v', 'P', '<', '>', '*'][:n_colors]

# ── Figure layout: 3 rows × 2 cols ───────────────────────────────────────────
fig = plt.figure(figsize=(13, 14), constrained_layout=True)
gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

# New order: A/B = plateau k=3, C/D = plateau k=4, E/F = heatmaps
ax_A = fig.add_subplot(gs[0, 0])   # SNM plateau k=3
ax_B = fig.add_subplot(gs[0, 1])   # ARM plateau k=3
ax_C = fig.add_subplot(gs[1, 0])   # SNM plateau k=4
ax_D = fig.add_subplot(gs[1, 1])   # ARM plateau k=4
ax_E = fig.add_subplot(gs[2, 0])   # SNM heatmap
ax_F = fig.add_subplot(gs[2, 1])   # ARM heatmap

panel_labels = {'a': ax_A, 'b': ax_B, 'c': ax_C,
                'd': ax_D, 'e': ax_E, 'f': ax_F}

# ══════════════════════════════════════════════════════════════════════════════
# Panels E & F: heatmaps
# ══════════════════════════════════════════════════════════════════════════════
heatmap_specs = [
    (ax_E, mat_noise, r'SNM  ($\varepsilon = 0.05$)', 'E', COL_SNM),
    (ax_F, mat_async, r'ARM  ($p = 0.45$)',           'F', COL_ARM),
]

im = None
for ax, mat, title, plabel, tcol in heatmap_specs:
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap_heat,
                   aspect='auto', origin='upper', interpolation='nearest')
    for x in np.arange(-0.5, len(c_vals), 1):
        ax.axvline(x, color='white', lw=0.6)
    for y in np.arange(-0.5, len(n_vals), 1):
        ax.axhline(y, color='white', lw=0.6)

    # √2N overlay
    for i, n in enumerate(n_vals):
        k_theory = np.sqrt(2 * n)
        c_arr = np.array(c_vals, dtype=float)
        if k_theory <= c_arr[-1]:
            col_pos = np.interp(k_theory, c_arr, np.arange(len(c_arr)))
            ax.plot(col_pos, i, marker='D', color='#FF3300',
                    ms=6, mew=0, zorder=5, clip_on=False)

    xtick_step = 5
    xtick_idx  = [i for i, c in enumerate(c_vals) if c % xtick_step == 0]
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([c_vals[i] for i in xtick_idx], fontsize=8)
    ax.set_yticks(range(len(n_vals)))
    ax.set_yticklabels(n_vals)
    ax.set_xlabel(r'Number of colors $k$', fontsize=11)
    ax.set_ylabel(r'Network size $N$',     fontsize=11)
    ax.set_title(title, fontsize=12, pad=8, color=tcol, fontweight='bold')

# Shared colorbar
cbar = fig.colorbar(im, ax=[ax_E, ax_F], orientation='vertical',
                    fraction=0.025, pad=0.02, shrink=0.92)
cbar.set_label('Solution rate', fontsize=11)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

# Diamond legend on F
legend_diamond = Line2D([0], [0], marker='D', color='w',
                        markerfacecolor='#FF3300', markersize=7,
                        label=r'$\sqrt{2N}$  (BA max-degree, $m=2$)')
ax_F.legend(handles=[legend_diamond], loc='center right', fontsize=10, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# Panels C–F: plateau line plots
# ══════════════════════════════════════════════════════════════════════════════
plateau_specs = [
    (ax_A, 'Noise', 3, 'p_noise', r'SNM — solve rate vs $\varepsilon$  ($k=3$)',
     r'Noise parameter $\varepsilon$', 'A', COL_SNM),
    (ax_B, 'Async', 3, 'p_async', r'ARM — solve rate vs $p$  ($k=3$)',
     r'Freezing probability $p$', 'B', COL_ARM),
    (ax_C, 'Noise', 4, 'p_noise', r'SNM — solve rate vs $\varepsilon$  ($k=4$)',
     r'Noise parameter $\varepsilon$', 'C', COL_SNM),
    (ax_D, 'Async', 4, 'p_async', r'ARM — solve rate vs $p$  ($k=4$)',
     r'Freezing probability $p$', 'D', COL_ARM),
]

for ax, label, k, param_col, title, xlabel, plabel, tcol in plateau_specs:
    data = get_plateau(label, k, param_col)
    for color, marker, n in zip(palette, markers, n_plateau):
        sub = data[data['node_num'] == n].sort_values(param_col)
        if len(sub) == 0:
            continue
        x  = sub[param_col].values
        y  = sub['rate'].values
        se = (sub['sd'] / np.sqrt(sub['count'])).values

        ax.plot(x, y, color=color, lw=2.0,
                marker=marker, ms=5, markevery=2,
                label=f'$N={n}$')
        ax.fill_between(x, np.clip(y - se, 0, 1),
                           np.clip(y + se, 0, 1),
                        color=color, alpha=0.18)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(xlabel,          fontsize=11)
    ax.set_ylabel('Solution rate', fontsize=11)
    ax.set_title(title, fontsize=12, pad=8, color=tcol, fontweight='bold')
    ax.axhline(0.9, color='gray', lw=1.0, ls='--', alpha=0.6)
    ax.text(0.01, 0.91, '90%', fontsize=8, color='gray', va='bottom')
    sns.despine(ax=ax)

    # Legend on ARM panels (D and F)
    if label == 'Async':
        handles = [Line2D([0], [0], color=c, lw=2.0, marker=m, markersize=5,
                          label=f'$N={n}$')
                   for c, m, n in zip(palette, markers, n_plateau)]
        ax.legend(handles=handles, fontsize=8, loc='center right',
                  bbox_to_anchor=(0.98, 0.45),
                  framealpha=0.9, title=r'$N$', title_fontsize=9)

# ── Panel labels ──────────────────────────────────────────────────────────────
for lbl, ax in panel_labels.items():
    ax.text(-0.10, 1.05, lbl, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('../tex/scalability.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../tex/scalability_new.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: scalability.pdf / scalability_new.png")
