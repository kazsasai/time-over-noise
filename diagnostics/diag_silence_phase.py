"""
Diagnostic: Where does val=0 (silence) actually occur?

Key question: does val=0 occur during the *unsolved* phase (CN>0),
or only after the system has solved (CN=0)?

If val=0 is rare during the unsolved phase, threshold=0 would produce
avalanches that never terminate until the system solves — making
avalanche analysis meaningless at that threshold.

For each p, decompose history_changed into:
  - unsolved steps  (CN > 0)
  - solved steps    (CN = 0)

And compute P(val=0 | unsolved) and P(val=0 | solved).

Also plot the PMF of changed_nodes separately for each phase.
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
palette_u = sns.color_palette('Blues_d',  n_colors=len(p_list))
palette_s = sns.color_palette('Oranges_d', n_colors=len(p_list))

# ── Figure: PMF split by phase ────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(p_list), figsize=(14, 8),
                         sharex='row', sharey='row')
plt.subplots_adjust(hspace=0.50, wspace=0.30)

print(f"\n{'p':>5}  {'P(0|unsolved)%':>15}  {'P(1|unsolved)%':>15}  "
      f"{'P(0|solved)%':>13}  {'unsolved_steps':>15}  {'solved_steps':>13}")
print('-' * 82)

summary = []
for ci, p in enumerate(p_list):
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]

    vals_u, vals_s = [], []   # unsolved / solved steps
    for _, r in rows.iterrows():
        ch  = r['history_changed'].astype(int)
        cn  = r['history_conflict_ratio']  # conflict-node ratio

        # CN=0 means solved; determine solved mask
        solved_mask = (np.array(cn) == 0)

        vals_u.extend(ch[~solved_mask])
        vals_s.extend(ch[ solved_mask])

    vals_u = np.array(vals_u, dtype=int)
    vals_s = np.array(vals_s, dtype=int)

    # PMF for unsolved
    ax_u = axes[0, ci]
    if len(vals_u) > 0:
        counts_u = np.bincount(vals_u, minlength=N + 1)
        pmf_u    = counts_u / counts_u.sum()
        ax_u.bar(np.arange(N + 1), pmf_u, color=palette_u[ci], alpha=0.8,
                 width=0.85, linewidth=0)
        ax_u.bar([0], [pmf_u[0]], color='#d73027', alpha=0.95,
                 width=0.85, linewidth=0)
        ax_u.bar([1], [pmf_u[1]], color='#fc8d59', alpha=0.95,
                 width=0.85, linewidth=0)
        p0u = pmf_u[0] * 100
        p1u = pmf_u[1] * 100
        ax_u.text(0.97, 0.97,
                  f'P(0)={p0u:.1f}%\nP(1)={p1u:.1f}%',
                  transform=ax_u.transAxes, fontsize=7.5,
                  va='top', ha='right',
                  bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))
    ax_u.set_title(f'p={p:.2f}  [unsolved]', fontsize=9)
    ax_u.set_xlabel('changed / step', fontsize=8)
    if ci == 0:
        ax_u.set_ylabel('P(val)  unsolved phase', fontsize=8)
    sns.despine(ax=ax_u)

    # PMF for solved
    ax_s = axes[1, ci]
    if len(vals_s) > 0:
        counts_s = np.bincount(vals_s, minlength=N + 1)
        pmf_s    = counts_s / counts_s.sum()
        ax_s.bar(np.arange(N + 1), pmf_s, color=palette_s[ci], alpha=0.8,
                 width=0.85, linewidth=0)
        ax_s.bar([0], [pmf_s[0]], color='#d73027', alpha=0.95,
                 width=0.85, linewidth=0)
        p0s = pmf_s[0] * 100
        ax_s.text(0.97, 0.97,
                  f'P(0)={p0s:.1f}%',
                  transform=ax_s.transAxes, fontsize=7.5,
                  va='top', ha='right',
                  bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))
    ax_s.set_title(f'p={p:.2f}  [solved]', fontsize=9)
    ax_s.set_xlabel('changed / step', fontsize=8)
    if ci == 0:
        ax_s.set_ylabel('P(val)  solved phase', fontsize=8)
    sns.despine(ax=ax_s)

    # Summary
    p0u_val = (np.sum(vals_u == 0) / len(vals_u) * 100) if len(vals_u) > 0 else np.nan
    p1u_val = (np.sum(vals_u == 1) / len(vals_u) * 100) if len(vals_u) > 0 else np.nan
    p0s_val = (np.sum(vals_s == 0) / len(vals_s) * 100) if len(vals_s) > 0 else np.nan
    print(f'{p:5.2f}  {p0u_val:15.2f}  {p1u_val:15.2f}  '
          f'{p0s_val:13.2f}  {len(vals_u):15d}  {len(vals_s):13d}')
    summary.append({'p': p, 'P0_unsolved': p0u_val, 'P1_unsolved': p1u_val,
                    'P0_solved': p0s_val,
                    'n_unsolved': len(vals_u), 'n_solved': len(vals_s)})

plt.suptitle(r'Activity distribution split by phase — ARM ($N=20$, $k=3$)'
             '\nRow 1: unsolved steps (CN>0)   Row 2: solved steps (CN=0)'
             '\nRed bar = val=0 (silence candidate),  Orange = val=1',
             fontsize=10, y=1.02)
plt.savefig('../tex/diag_silence_phase.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved: diag_silence_phase.png')

# ── Additional: cumulative silence rate over time (unsolved only) ──────────────
fig2, axes2 = plt.subplots(1, len(p_list), figsize=(14, 3.5), sharey=True)
plt.subplots_adjust(wspace=0.25)

for ci, p in enumerate(p_list):
    ax = axes2[ci]
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]

    # Collect unsolved-phase silence run lengths (consecutive val=0 during unsolved)
    silence_runs_u = []
    for _, r in rows.iterrows():
        ch         = r['history_changed'].astype(int)
        cn         = np.array(r['history_conflict_ratio'])
        unsolved   = cn > 0
        run = 0
        for i, (v, u) in enumerate(zip(ch, unsolved)):
            if not u:
                if run > 0:
                    silence_runs_u.append(run)
                run = 0
                continue
            if v == 0:
                run += 1
            else:
                if run > 0:
                    silence_runs_u.append(run)
                run = 0
        if run > 0:
            silence_runs_u.append(run)

    silence_runs_u = np.array(silence_runs_u, dtype=int)
    if len(silence_runs_u) > 10:
        unique_r = np.unique(silence_runs_u)
        ccdf_r   = np.array([np.sum(silence_runs_u >= s)
                              for s in unique_r], dtype=float) / len(silence_runs_u)
        ax.plot(unique_r, ccdf_r, color=palette_u[ci], lw=1.8)
        frac_nonzero = np.mean(silence_runs_u > 0) * 100
        ax.text(0.97, 0.97, f'n={len(silence_runs_u)}',
                transform=ax.transAxes, fontsize=7.5, va='top', ha='right')
    else:
        ax.text(0.5, 0.5, 'no silence\nin unsolved',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('silence run length', fontsize=8)
    ax.set_title(f'p={p:.2f}', fontsize=9)
    if ci == 0:
        ax.set_ylabel('P(run ≥ L)  [unsolved only]', fontsize=8)
    sns.despine(ax=ax)

plt.suptitle('Silence run length CCDF during unsolved phase  (val=0 consecutive runs)',
             fontsize=10)
plt.savefig('../tex/diag_silence_runs.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_silence_runs.png')
