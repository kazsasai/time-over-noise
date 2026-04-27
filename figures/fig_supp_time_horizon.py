"""
Supp. Fig: Solution rate landscape across (noise parameter, time budget).

Two heatmaps side-by-side:
  (a) SNM:  X = epsilon  (state-selection noise),       Y = T (time budget, log)
  (b) ARM:  X = p        (freezing probability),         Y = T (time budget, log)
  Color   = solution rate within T steps  (viridis 0--1)

Demonstrates that the narrow SNM ridge and the broad ARM plateau persist
across all evaluation horizons T in [50, 1000] steps. Addresses the
``why 1,000?'' question raised by referees of Fig.~3.

Data source:  result_for_phase_space_20_3.pkl  (N=20, k=3, 1000 trials,
              step_max=1000, full p_async / p_noise sweep on [0, 0.99]).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_PATH   = 'result_for_phase_space_20_3.pkl'
OUT_PDF     = '../tex/supp_fig_time_horizon.pdf'
OUT_PNG     = '../tex/supp_fig_time_horizon.png'

# Time budget grid (log-spaced)
T_GRID      = np.array([50, 100, 200, 350, 500, 700, 1000])

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# ── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} ...")
df = pd.read_pickle(DATA_PATH)
print(f"Loaded. Rows={len(df)}")

# ── Compute solution-rate landscape for SNM and ARM ──────────────────────────
def landscape(df_sub, param_col):
    """For each (param value, T), compute solution rate = mean(solved & solve_time <= T)."""
    params = sorted(df_sub[param_col].unique())
    M = np.zeros((len(T_GRID), len(params)))
    for j, p in enumerate(params):
        rows = df_sub[df_sub[param_col] == p]
        st = rows['solve_time'].values
        solved = rows['solved'].values
        for i, T in enumerate(T_GRID):
            mask = solved & (st <= T)
            M[i, j] = mask.mean()
    return np.array(params), M

# SNM: label='Noise Model', x-axis = p_noise (epsilon)
snm = df[df['label'] == 'Noise Model']
arm = df[df['label'] == 'Async Model']

eps_vals, M_snm = landscape(snm, 'p_noise')
p_vals,   M_arm = landscape(arm, 'p_async')

print(f"SNM grid: {len(eps_vals)} eps values, {len(T_GRID)} T values")
print(f"ARM grid: {len(p_vals)} p values, {len(T_GRID)} T values")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                         constrained_layout=True,
                         sharey=True)

norm = Normalize(vmin=0, vmax=1)
cmap = plt.cm.viridis

def draw(ax, x_vals, M, xlabel, panel_title, panel_label):
    extent = [x_vals.min(), x_vals.max(),
              np.log10(T_GRID.min()), np.log10(T_GRID.max())]
    im = ax.imshow(M, origin='lower', aspect='auto',
                   extent=extent, cmap=cmap, norm=norm,
                   interpolation='bilinear')
    # Y-axis ticks at the actual T_GRID values
    ax.set_yticks(np.log10(T_GRID))
    ax.set_yticklabels([f'{int(T)}' for T in T_GRID])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Time budget $T$ (steps)', fontsize=12)
    ax.set_title(panel_title, fontsize=12, pad=8)
    ax.text(-0.14, 1.02, panel_label,
            transform=ax.transAxes,
            fontsize=15, fontweight='bold',
            va='bottom', ha='left')
    return im

im_a = draw(axes[0], eps_vals, M_snm,
            r'$\varepsilon$  (state-selection noise)',
            'Synchronous Noise Model (SNM)', 'a')
im_b = draw(axes[1], p_vals,   M_arm,
            r'$p$  (freezing probability)',
            'Asynchronous Random Model (ARM)', 'b')

# Shared colorbar
cbar = fig.colorbar(im_b, ax=axes, orientation='vertical',
                    shrink=0.85, pad=0.02)
cbar.set_label('Solution rate', fontsize=11)

plt.savefig(OUT_PDF, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
