"""
Diagnostic: Distribution of activity signal (changed nodes per step)

Before choosing a B&P threshold θ, examine the empirical distribution of
changed_nodes per step for the ARM.

For each p value, plot:
  (A) PMF bar chart of changed_nodes in [0, N] — linear scale
  (B) CCDF of changed_nodes — log-log scale
  (C) Fraction of steps at val=0 and val=1 vs p  (threshold sensitivity summary)

Data: result_for_criticality_20_3.pkl (N=20, k=3, ARM)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_pickle('result_for_criticality_20_3.pkl')
arm = df[df['label'] == 'Async Model'].copy()

p_list = np.array([0.05, 0.25, 0.45, 0.65, 0.85])
N = 20
palette = sns.color_palette('coolwarm', n_colors=len(p_list))

# ── Collect all changed-node counts per p ─────────────────────────────────────
all_vals = {}
for p in p_list:
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]
    vals = np.concatenate([r['history_changed'].astype(int)
                           for _, r in rows.iterrows()])
    all_vals[p] = vals

# ── Figure layout: 3 rows ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs  = fig.add_gridspec(3, len(p_list), hspace=0.55, wspace=0.35)

# ── Row A: PMF bar chart (val = 0, 1, 2, ..., N) ─────────────────────────────
for ci, p in enumerate(p_list):
    ax = fig.add_subplot(gs[0, ci])
    vals = all_vals[p]
    counts = np.bincount(vals, minlength=N + 1)
    pmf    = counts / counts.sum()

    bars = ax.bar(np.arange(N + 1), pmf, color=palette[ci], alpha=0.75,
                  width=0.85, linewidth=0)
    # Highlight val=0 and val=1
    ax.bar([0], [pmf[0]], color='#d73027', alpha=0.9, width=0.85, linewidth=0,
           label='val=0 (silence)')
    ax.bar([1], [pmf[1]], color='#fc8d59', alpha=0.9, width=0.85, linewidth=0,
           label='val=1')

    ax.set_xlabel('changed nodes / step', fontsize=8)
    ax.set_ylabel('P(val)', fontsize=8)
    ax.set_title(f'ARM  p={p:.2f}', fontsize=9)
    ax.set_xlim(-0.5, N + 0.5)
    f0 = pmf[0] * 100
    f1 = pmf[1] * 100
    ax.text(0.98, 0.97, f'P(0)={f0:.1f}%\nP(1)={f1:.1f}%',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))
    if ci == 0:
        ax.legend(fontsize=7, loc='upper right')
    sns.despine(ax=ax)

# ── Row B: CCDF on log-log ────────────────────────────────────────────────────
ax_ccdf_all = fig.add_subplot(gs[1, :])   # span all columns
for ci, p in enumerate(p_list):
    vals = all_vals[p]
    # CCDF only over val >= 1 (exclude exact zeros to enable log)
    v_pos = vals[vals >= 1]
    unique_v = np.unique(v_pos)
    ccdf     = np.array([np.sum(v_pos >= s) for s in unique_v], dtype=float) / len(vals)

    ax_ccdf_all.plot(unique_v, ccdf, color=palette[ci], lw=1.8,
                     label=f'p={p:.2f}')

# Threshold reference lines
for thr_val, thr_col, thr_lab in [(1, '#636363', 'θ=1'), (2, '#bdbdbd', 'θ=2')]:
    ax_ccdf_all.axvline(thr_val, color=thr_col, lw=1.2, ls='--', alpha=0.8,
                        label=f'threshold {thr_lab}')

ax_ccdf_all.set_xscale('log')
ax_ccdf_all.set_yscale('log')
ax_ccdf_all.set_xlabel('changed nodes / step  (val)', fontsize=9)
ax_ccdf_all.set_ylabel('P(val ≥ v)  (fraction of all steps)', fontsize=9)
ax_ccdf_all.set_title('CCDF of activity signal — all p values'
                      '  (denominator = all steps including val=0)',
                      fontsize=9)
ax_ccdf_all.legend(fontsize=8, ncol=7, loc='lower left')
sns.despine(ax=ax_ccdf_all)

# ── Row C: P(val=0), P(val=1), P(val<=1) vs p — threshold sensitivity ─────────
ax_sum = fig.add_subplot(gs[2, :])
p_all  = np.sort(arm['p_async'].unique())
frac_0  = []
frac_1  = []
frac_le1 = []
for p in p_all:
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]
    vals = np.concatenate([r['history_changed'].astype(int)
                           for _, r in rows.iterrows()])
    total = len(vals)
    frac_0.append(np.sum(vals == 0) / total)
    frac_1.append(np.sum(vals == 1) / total)
    frac_le1.append(np.sum(vals <= 1) / total)

ax_sum.plot(p_all, np.array(frac_0)   * 100, 'o-', color='#d73027', lw=1.8,
            label='P(val=0)  — silence for θ=0 and θ=1')
ax_sum.plot(p_all, np.array(frac_1)   * 100, 's-', color='#fc8d59', lw=1.8,
            label='P(val=1)  — counts as silence only for θ=1')
ax_sum.plot(p_all, np.array(frac_le1) * 100, '^-', color='#4575b4', lw=1.8,
            label='P(val≤1) — silence fraction at θ=1')
ax_sum.axhline(50, color='gray', lw=0.8, ls=':', alpha=0.5)
ax_sum.set_xlabel('Freezing probability  p', fontsize=9)
ax_sum.set_ylabel('Fraction of steps (%)', fontsize=9)
ax_sum.set_title('Silence fraction vs p — impact of threshold choice',
                 fontsize=9)
ax_sum.legend(fontsize=8)
ax_sum.set_xlim(p_all.min() - 0.02, p_all.max() + 0.02)
ax_sum.set_ylim(0, 100)
sns.despine(ax=ax_sum)

plt.suptitle(r'Activity signal distribution — ARM ($N=20$, $k=3$)'
             '\nRows: (A) PMF per p,  (B) CCDF all p,  (C) silence fraction vs p',
             fontsize=10, y=1.01)

plt.savefig('../tex/diag_activity_dist.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_activity_dist.png')

# ── Text summary ──────────────────────────────────────────────────────────────
print(f'\n{"p":>5}  {"P(0)%":>7}  {"P(1)%":>7}  {"P(<=1)%":>9}  '
      f'{"P(>=2)%":>9}  {"mean_val":>9}  {"median":>7}')
print('-' * 65)
for p in p_all:
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]
    vals = np.concatenate([r['history_changed'].astype(int)
                           for _, r in rows.iterrows()])
    total = len(vals)
    p0   = np.sum(vals == 0) / total * 100
    p1   = np.sum(vals == 1) / total * 100
    ple1 = np.sum(vals <= 1) / total * 100
    pge2 = np.sum(vals >= 2) / total * 100
    print(f'{p:5.2f}  {p0:7.2f}  {p1:7.2f}  {ple1:9.2f}  '
          f'{pge2:9.2f}  {vals.mean():9.2f}  {np.median(vals):7.1f}')
