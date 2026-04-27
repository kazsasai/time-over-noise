"""
Fig. 2: Time-series evolution of coordination dynamics (3×2 layout).

Layout: 3 rows × 2 cols
  Col 0 (SNM) | Col 1 (ARM)

  Row 0 (Base):      SNM ε=0 CN only        | ARM p=0 CN only
  Row 1 (Mechanism): SNM ε=0.05 CN only     | ARM p=0.45 CN only
  Row 2 (Activity):  SNM chg (base+opt)     | ARM chg (base+opt)

Messages:
  1. Base models (ε=0, p=0) cannot solve
  2. SNM solved state is unstable; ARM is robust
  3. Changed-node dynamics: base vs mechanism, SNM vs ARM
"""

import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import gcg_solver

# ── Settings ─────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

N        = 20
K        = 3
STEP_MAX = 2000
ROLL_WIN = 20

# per-panel seeds: snm_opt uses seed 7940 (shows fragile collapse/recovery)
#                  others use seed 2345 (confirmed deadlock for base, robust for arm_opt)
models = [
    # key,       p_noise, p_async, is_async, param_label,            seed
    ('snm_base', 0.00, 0.00, False, r'$\varepsilon=0$',              2345),
    ('arm_base', 0.00, 0.00, True,  r'$p=0$',                        2345),
    ('snm_opt',  0.05, 0.00, False, r'$\varepsilon=0.05$',           7940),
    ('arm_opt',  0.00, 0.45, True,  r'$p=0.45$',                     2345),
]

base_param = {
    'node_num':       N,
    'color_num':      K,
    'ba_m':           2,
    'bot_num':        N,
    'bot_type':       'c',
    'step_max':       STEP_MAX,
    'return_history': True,
}

# ── Run simulations ───────────────────────────────────────────────────────────
results = {}
t = np.arange(STEP_MAX)

for key, p_noise, p_async, is_async, param_label, seed in models:
    params = {**base_param,
              'seed':             seed,
              'label':            key,
              'p_noise':          p_noise,
              'p_async':          p_async,
              'is_async_model':   is_async}
    r = gcg_solver._worker_run_single(params)

    cf  = r['history_conflict_ratio']
    chg = r['history_changed'] / N

    roll = lambda x: pd.Series(x).rolling(ROLL_WIN, min_periods=1, center=True).mean().values

    solved_mask = cf == 0.0
    results[key] = dict(
        cf=cf, chg=chg,
        cf_r=roll(cf), chg_r=roll(chg),
        solved_mask=solved_mask,
        param_label=param_label,
    )
    solved_pct = solved_mask.mean() * 100
    print(f"{key:10s}: solved={solved_mask.any()!s:5s} ({solved_pct:.1f}%)  "
          f"mean_chg={chg.mean():.3f}")

# ── Colors ────────────────────────────────────────────────────────────────────
_pal       = sns.color_palette('colorblind')
COL_SNM    = _pal[0]     # blue   – SNM (consistent with Fig.3)
COL_ARM    = _pal[1]     # orange – ARM (consistent with Fig.3)
col_solved = '#52C77A'   # green  – solved background

ALPHA_RAW  = 0.20
ALPHA_ROLL = 0.90
ALPHA_BG   = 0.28

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(10, 10),
                         sharex=True,
                         gridspec_kw={'hspace': 0.38, 'wspace': 0.20})

col_titles  = ['Synchronous Noise Model (SNM)', 'Asynchronous Random Model (ARM)']
col_colors  = [COL_SNM, COL_ARM]
panel_letters = [['a', 'b'], ['c', 'd'], ['e', 'f']]


def shade_solved(ax, solved_mask):
    in_s, start = False, None
    for i, s in enumerate(solved_mask):
        if s and not in_s:
            start, in_s = i, True
        elif not s and in_s:
            ax.axvspan(start, i, color=col_solved, alpha=ALPHA_BG, zorder=0)
            in_s = False
    if in_s:
        ax.axvspan(start, STEP_MAX, color=col_solved, alpha=ALPHA_BG, zorder=0)


# ── Rows 0 & 1: CN only ───────────────────────────────────────────────────────
cn_panels = [
    # (row, col, key)
    (0, 0, 'snm_base'),
    (0, 1, 'arm_base'),
    (1, 0, 'snm_opt'),
    (1, 1, 'arm_opt'),
]

for row_idx, col_idx, key in cn_panels:
    ax  = axes[row_idx, col_idx]
    d   = results[key]
    col = COL_SNM if col_idx == 0 else COL_ARM

    shade_solved(ax, d['solved_mask'])
    ax.plot(t, d['cf'],   color=col, lw=0.5, alpha=ALPHA_RAW,  zorder=1)
    ax.plot(t, d['cf_r'], color=col, lw=2.0, alpha=ALPHA_ROLL, zorder=2)

    ax.text(0.97, 0.97, d['param_label'],
            transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2.5))
    ax.text(-0.10, 1.13, panel_letters[row_idx][col_idx],
            transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    sns.despine(ax=ax)

# ── Row 2: Changed nodes — base (dashed) + opt (solid) ───────────────────────
chg_panels = [
    # (col, base_key, opt_key, color)
    (0, 'snm_base', 'snm_opt', COL_SNM),
    (1, 'arm_base', 'arm_opt', COL_ARM),
]

for col_idx, base_key, opt_key, color in chg_panels:
    ax  = axes[2, col_idx]
    db  = results[base_key]
    do  = results[opt_key]

    # base (ε=0 or p=0): dashed, faded
    ax.plot(t, db['chg'],   color=color, lw=0.4, alpha=0.15,       ls='-',  zorder=1)
    ax.plot(t, db['chg_r'], color=color, lw=2.0, alpha=0.40,       ls='--', zorder=2)

    # mechanism (ε=0.05 or p=0.45): solid, full opacity
    ax.plot(t, do['chg'],   color=color, lw=0.4, alpha=ALPHA_RAW,  ls='-',  zorder=1)
    ax.plot(t, do['chg_r'], color=color, lw=2.2, alpha=ALPHA_ROLL, ls='-',  zorder=3)

    ax.text(-0.10, 1.13, panel_letters[2][col_idx],
            transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    ax.set_xlabel('Steps', fontsize=11)
    sns.despine(ax=ax)

# ── Axis limits ───────────────────────────────────────────────────────────────
for ax in axes.flat:
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, STEP_MAX)

# ── Column titles ─────────────────────────────────────────────────────────────
for col_idx, (title, tcol) in enumerate(zip(col_titles, col_colors)):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', pad=8, color=tcol)

# ── Row y-labels ──────────────────────────────────────────────────────────────
axes[0, 0].set_ylabel('CN ratio  (Base: $\\varepsilon$=0, $p$=0)', fontsize=11)
axes[1, 0].set_ylabel('CN ratio  (Mechanism)', fontsize=11)
axes[2, 0].set_ylabel('Changed nodes / $N$', fontsize=11)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color=COL_SNM,    lw=2.5, label='Conflict Nodes — SNM'),
    Line2D([0], [0], color=COL_ARM,    lw=2.5, label='Conflict Nodes — ARM'),
    mpatches.Patch(facecolor=col_solved, alpha=0.65, label='Solved state'),
    Line2D([0], [0], color='grey', lw=2.0, ls='--', alpha=0.5, label='Changed nodes (base, $\\varepsilon$=0 / $p$=0)'),
    Line2D([0], [0], color='grey', lw=2.2, ls='-',             label='Changed nodes (mechanism)'),
]
fig.legend(handles=legend_elements,
           loc='lower center', ncol=4,
           fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.03))

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('../tex/timeseries.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../tex/timeseries.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: timeseries.pdf / timeseries_new.png")
