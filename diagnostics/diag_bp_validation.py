"""
Validation: Are phase-separated avalanches consistent with the original mixed analysis?

For each p, overlay on the same axes:
  (A) original mixed CCDF  (threshold=1, as in fig4)
  (B) unsolved-phase CCDF  (threshold=0)
  (C) solved-phase   CCDF  (threshold=0)

Key questions:
  Q1. Does the large-scale portion of the mixed CCDF come from the unsolved phase?
  Q2. Does the small-scale portion come from the solved phase?
  Q3. Where does the 10^{-3} split line fall relative to the phase transition?

Also:
  Q4. What fraction of all avalanches (mixed, threshold=1) come from steps that
      are entirely in the unsolved phase vs entirely in the solved phase vs spanning both?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

# ── Load ──────────────────────────────────────────────────────────────────────
df  = pd.read_pickle('result_for_criticality_20_3.pkl')
arm = df[df['label'] == 'Async Model'].copy()

p_list = np.array([0.05, 0.25, 0.45, 0.65, 0.85])

# ── Helpers ───────────────────────────────────────────────────────────────────
def bp_avalanches_labeled(series, cr_series, threshold):
    """
    B&P avalanches with phase label.
    Returns list of (size, phase_label) where phase_label is:
      'unsolved' if all steps in burst have CR > 0
      'solved'   if all steps in burst have CR = 0
      'mixed'    if burst spans both phases
    """
    avs = []
    in_burst, sz = False, 0
    phase_set = set()
    for v, cr in zip(series, cr_series):
        if v > threshold:
            in_burst = True
            sz += v
            phase_set.add('unsolved' if cr > 0 else 'solved')
        else:
            if in_burst:
                if len(phase_set) == 1:
                    label = phase_set.pop()
                else:
                    label = 'mixed'
                avs.append((sz, label))
            in_burst, sz = False, 0
            phase_set = set()
    if in_burst:
        label = phase_set.pop() if len(phase_set) == 1 else 'mixed'
        avs.append((sz, label))
    return avs

def bp_avalanches(series, threshold):
    avs, in_burst, sz = [], False, 0
    for v in series:
        if v > threshold: in_burst = True; sz += v
        else:
            if in_burst: avs.append(sz)
            in_burst, sz = False, 0
    if in_burst: avs.append(sz)
    return np.array(avs, dtype=float)

def ccdf(sizes):
    if len(sizes) < 5: return None, None
    sizes = np.array(sizes, dtype=float)
    u = np.unique(sizes)
    p = np.array([np.sum(sizes >= s) for s in u], dtype=float) / len(sizes)
    return u, p

# ── Collect data ──────────────────────────────────────────────────────────────
data = {p: {'mixed_thr1': [], 'mixed_thr0': [],
            'unsolved_thr0': [], 'solved_thr0': [],
            'labeled_thr1': []}
        for p in p_list}

for p in p_list:
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]
    for _, r in rows.iterrows():
        ch = r['history_changed'].astype(float)
        cr = np.array(r['history_conflict_ratio'])
        sm = (cr == 0)

        data[p]['mixed_thr1'].extend(bp_avalanches(ch,        threshold=1))
        data[p]['mixed_thr0'].extend(bp_avalanches(ch,        threshold=0))
        data[p]['unsolved_thr0'].extend(bp_avalanches(ch[~sm], threshold=0))
        data[p]['solved_thr0'].extend(  bp_avalanches(ch[ sm], threshold=0))
        data[p]['labeled_thr1'].extend( bp_avalanches_labeled(ch, cr, threshold=1))

# ── Figure: CCDF overlay ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(p_list), figsize=(14, 8),
                         sharey='row')
plt.subplots_adjust(hspace=0.50, wspace=0.30)

for ci, p in enumerate(p_list):
    d = data[p]

    for row_idx, (use_log, ylabel) in enumerate(
            [(True, r'$P(S\geq s)$'), (False, r'$P(S\geq s)$  [linear x]')]):
        ax = axes[row_idx, ci]

        for key, col, lw, ls, label, zord in [
            ('mixed_thr1',    '#e41a1c', 2.2, '-',  'mixed  thr=1 (fig4)', 5),
            ('unsolved_thr0', '#377eb8', 1.6, '--', 'unsolved thr=0',      4),
            ('solved_thr0',   '#ff7f00', 1.6, ':',  'solved   thr=0',      3),
        ]:
            sizes = np.array(d[key])
            u, p_ccdf = ccdf(sizes)
            if u is None: continue
            x = np.log10(u) if use_log else u
            ax.plot(x, p_ccdf, color=col, lw=lw, ls=ls,
                    label=label if ci == 0 else '', alpha=0.85, zorder=zord)

        # Mark 10^{-3} split line
        ax.axhline(1e-3, color='gray', lw=0.8, ls=':', alpha=0.6)
        ax.text(ax.get_xlim()[0] if not use_log else 0.02,
                1.5e-3, r'$10^{-3}$', fontsize=7, color='gray',
                transform=ax.get_yaxis_transform() if not use_log else ax.transAxes)

        ax.set_yscale('log')
        if use_log: ax.set_xscale('log') if not use_log else None
        if use_log:
            ax.set_xlabel(r'avalanche size $s$  [log scale]', fontsize=8)
        else:
            ax.set_xlabel(r'avalanche size $s$  [linear]', fontsize=8)
        if ci == 0:
            ax.set_ylabel(ylabel, fontsize=8)
            if row_idx == 0:
                ax.legend(fontsize=7.5, loc='upper right')
        ax.set_title(f'p={p:.2f}', fontsize=9)
        sns.despine(ax=ax)

plt.suptitle(r'CCDF overlay: mixed (threshold=1) vs phase-separated (threshold=0)'
             r'  —  ARM $N=20$, $k=3$'
             '\nDoes the mixed large-scale tail coincide with the unsolved phase?',
             fontsize=10, y=1.02)
plt.savefig('../tex/diag_bp_validation_overlay.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_bp_validation_overlay.png')

# ── Q4: phase composition of mixed (threshold=1) avalanches ───────────────────
fig2, axes2 = plt.subplots(1, len(p_list), figsize=(14, 4), sharey=True)
plt.subplots_adjust(wspace=0.25)

print(f'\n{"p":>5}  {"n_total":>9}  {"unsolved%":>10}  {"solved%":>9}  '
      f'{"mixed%":>8}  {"median_s_unsolved":>18}  {"median_s_solved":>15}')
print('-' * 80)

for ci, p in enumerate(p_list):
    ax = axes2[ci]
    labeled = data[p]['labeled_thr1']
    if not labeled:
        ax.set_visible(False); continue

    sizes   = np.array([x[0] for x in labeled])
    labels  = np.array([x[1] for x in labeled])

    n_total = len(sizes)
    for lbl, col in [('unsolved', '#377eb8'), ('solved', '#ff7f00'), ('mixed', '#984ea3')]:
        mask = (labels == lbl)
        if mask.sum() == 0: continue
        u, p_ccdf = ccdf(sizes[mask])
        if u is None: continue
        ax.plot(u, p_ccdf, color=col, lw=1.8, label=f'{lbl} ({mask.sum()/n_total*100:.0f}%)')

    n_u = np.sum(labels == 'unsolved')
    n_s = np.sum(labels == 'solved')
    n_m = np.sum(labels == 'mixed')
    s_u = sizes[labels == 'unsolved']
    s_s = sizes[labels == 'solved']
    print(f'{p:5.2f}  {n_total:9d}  {n_u/n_total*100:10.1f}  {n_s/n_total*100:9.1f}  '
          f'{n_m/n_total*100:8.1f}  '
          f'{np.median(s_u) if len(s_u) else np.nan:18.1f}  '
          f'{np.median(s_s) if len(s_s) else np.nan:15.1f}')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'avalanche size $s$', fontsize=8)
    if ci == 0: ax.set_ylabel(r'$P(S\geq s)$', fontsize=8)
    ax.set_title(f'p={p:.2f}', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
    sns.despine(ax=ax)

plt.suptitle('Phase composition of mixed (threshold=1) avalanches\n'
             'CCDF split by whether burst is entirely unsolved / solved / spans both',
             fontsize=10)
plt.savefig('../tex/diag_bp_validation_composition.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_bp_validation_composition.png')
