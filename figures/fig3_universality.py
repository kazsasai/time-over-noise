"""
Fig. 3: Robustness of collective intelligence across parameter spaces.

Layout (mosaic):
    AAABBB
    CCDDDD

A: Solution rate vs ε  (SNM)
B: Solution rate vs p  (ARM)
C: Boxplot solution time  SNM vs ARM
D: 2×2 bar — 4 update rules compared at ε=0
   ① Sync+Stay, ② Async+Stay, ③ Sync+Random, ④ Async+Random (ARM)
   p_async = 0.45 for async conditions
"""

import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from statannotations.Annotator import Annotator
from multiprocessing import Pool, cpu_count
import gcg_solver

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='ticks', font_scale=1.0)
plt.rcParams.update({'font.family': 'Arial'})

COL_SNM  = sns.color_palette('colorblind')[0]   # blue
COL_ARM  = sns.color_palette('colorblind')[1]   # orange
COL_STAY = '#E8986A'
COL_RAND = 'steelblue'

# ── 2×2 simulation parameters ─────────────────────────────────────────────────
N        = 20
K        = 3
STEP_MAX = 1000
N_TRIALS = 500
N_JOBS   = max(1, cpu_count() - 1)
P_ASYNC  = 0.45

base_sim = dict(node_num=N, color_num=K, ba_m=2, bot_num=N, bot_type='c',
                step_max=STEP_MAX, return_history=False)

def run_rate(p_async, is_async_model, p_noise, seeds):
    tasks = [{**base_sim, 'seed': s,
              'p_async': p_async, 'p_noise': p_noise,
              'is_async_model': is_async_model, 'label': 'run'} for s in seeds]
    with Pool(N_JOBS) as pool:
        results = pool.map(gcg_solver._worker_run_single, tasks)
    solved = [r['solved'] for r in results]
    return np.mean(solved), np.std(solved) / np.sqrt(len(solved))

if __name__ == '__main__':

    # ── Run 2×2 simulations ───────────────────────────────────────────────────
    seeds = list(range(N_TRIALS))
    print("Running 2×2 conditions (ε=0 for all) ...")

    print("  ① Sync + Stay   (p=0,    ε=0) ...")
    r1, se1 = run_rate(0.0,     False, 0.0, seeds)
    print(f"     rate={r1:.3f} ± {se1:.3f}")

    print("  ② Async + Stay  (p=0.45, ε=0) ...")
    r2, se2 = run_rate(P_ASYNC, False, 0.0, seeds)
    print(f"     rate={r2:.3f} ± {se2:.3f}")

    print("  ③ Sync + Random (p=0,    ε=0) ...")
    r3, se3 = run_rate(0.0,     True,  0.0, seeds)
    print(f"     rate={r3:.3f} ± {se3:.3f}")

    print("  ④ Async + Random (ARM) (p=0.45, ε=0) ...")
    r4, se4 = run_rate(P_ASYNC, True,  0.0, seeds)
    print(f"     rate={r4:.3f} ± {se4:.3f}")

    # ── Load pickle data ──────────────────────────────────────────────────────
    df = pd.read_pickle('./result_for_phase_space_20_3.pkl')

    min_time_noise = (df.query('label == "Noise Model"')
                        .groupby('p_noise')['solve_time'].mean().idxmin())
    min_time_async = (df.query('label == "Async Model"')
                        .groupby('p_async')['solve_time'].mean().idxmin())
    print(f'\noptimal ε = {min_time_noise},  optimal p = {min_time_async}')

    df_min_time = df.query(
        'label == "Noise Model" and p_noise == @min_time_noise '
        'and solve_time < 1000 and solve_time > 0 '
        'or label == "Async Model" and p_async == @min_time_async '
        'and solve_time < 1000 and solve_time > 0'
    ).copy()
    df_min_time['Solution Time (logarithmic)'] = np.log10(df_min_time['solve_time'])

    # ── Figure layout ─────────────────────────────────────────────────────────
    layout = """
        AAABBB
        CCDDDD
    """
    fig, ax = plt.subplot_mosaic(layout, figsize=(11, 7.5), constrained_layout=True)

    # ── Panel A: SNM solution rate vs ε ──────────────────────────────────────
    sns.lineplot(df[df['label'] == 'Noise Model'],
                 ax=ax['A'], x='p_noise', y='solved',
                 linewidth=3, marker='o', ms=9, color=COL_SNM)
    ax['A'].set_xlabel(r'$\varepsilon$', fontsize=13)
    ax['A'].set_ylabel('Solution Rate in 1000 Steps', fontsize=11)

    # ── Panel B: ARM solution rate vs p ──────────────────────────────────────
    sns.lineplot(df[df['label'] == 'Async Model'],
                 ax=ax['B'], x='p_async', y='solved',
                 linewidth=3, marker='o', ms=9, color=COL_ARM)
    ax['B'].set_xlabel(r'$p$  (freezing probability)', fontsize=11)
    ax['B'].set_ylabel('Solution Rate in 1000 Steps', fontsize=11)

    # ── Panel C: boxplot solution time ────────────────────────────────────────
    sns.boxplot(df_min_time, ax=ax['C'],
                x='label', y='Solution Time (logarithmic)',
                hue='label', palette=[COL_SNM, COL_ARM],
                legend=False)

    annotator = Annotator(ax['C'], [('Noise Model', 'Async Model')],
                          data=df_min_time, x='label',
                          y='Solution Time (logarithmic)')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    ax['C'].set_xlabel('')
    ax['C'].set_xticklabels(
        [r'SNM' + '\n' + r'$(\varepsilon = ' + f'{min_time_noise:.2f}' + r')$',
         r'ARM' + '\n' + r'$(p = '           + f'{min_time_async:.2f}' + r')$'],
        fontsize=10)

    # ── Panel D: 2×2 bar chart ────────────────────────────────────────────────
    # Colours consistent with A/B/C: SNM=COL_SNM(blue), ARM=COL_ARM(orange)
    # Intermediate conditions: grey tones
    # Hatch: Synchronous=///, Asynchronous=solid
    COL_MID1 = '#888888'   # grey — Async+Stay
    COL_MID2 = '#AAAAAA'   # light grey — Sync+Random

    bar_labels = [
        '(1) Sync\n+ Stay\n(SNM)',
        '(2) Async\n+ Stay',
        '(3) Sync\n+ Random',
        '(4) Async\n+ Random\n(ARM)',
    ]
    rates   = [r1,      r2,       r3,       r4]
    errors  = [se1,     se2,      se3,      se4]
    colors  = [COL_SNM, COL_MID1, COL_MID2, COL_ARM]
    hatches = ['///',   '',       '///',    '']

    bars = ax['D'].bar(bar_labels, rates, yerr=errors, capsize=6,
                       color=colors, hatch=hatches,
                       edgecolor='white', linewidth=1.2,
                       error_kw=dict(ecolor='#333333', lw=1.5),
                       width=0.5, zorder=3)

    for bar, rate in zip(bars, rates):
        ax['D'].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04,
                     f'{rate:.0%}', ha='center', va='bottom', fontsize=11)

    ax['D'].set_ylabel('Solution Rate in 1000 Steps', fontsize=11)
    ax['D'].set_ylim(0, 1.18)
    ax['D'].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax['D'].set_title(
        f'2×2 factorial: Timing × Gridlock rule\n'
        f'($\\varepsilon=0$, async at $p={P_ASYNC}$, {N_TRIALS} trials)',
        fontsize=10, pad=6)
    ax['D'].axhline(0, color='black', lw=0.8)
    ax['D'].grid(axis='y', alpha=0.3, zorder=0)

    leg_snm  = mpatches.Patch(facecolor=COL_SNM,  hatch='///', edgecolor='white',
                               label='(1) SNM')
    leg_arm  = mpatches.Patch(facecolor=COL_ARM,  hatch='',    edgecolor='white',
                               label='(4) ARM')
    leg_mid1 = mpatches.Patch(facecolor=COL_MID1, hatch='',    edgecolor='white',
                               label='(2) Async+Stay')
    leg_mid2 = mpatches.Patch(facecolor=COL_MID2, hatch='///', edgecolor='white',
                               label='(3) Sync+Random')
    leg_sync = mpatches.Patch(facecolor='white',  hatch='///', edgecolor='grey',
                               label='Synchronous ($p=0$)')
    leg_asyn = mpatches.Patch(facecolor='white',  hatch='',    edgecolor='grey',
                               label=f'Asynchronous ($p={P_ASYNC}$)')
    ax['D'].legend(handles=[leg_snm, leg_mid1, leg_mid2, leg_arm, leg_sync, leg_asyn],
                   fontsize=9, framealpha=0.9, loc='upper left', ncol=2)

    sns.despine(ax=ax['D'])

    # ── Panel letters ─────────────────────────────────────────────────────────
    for lbl, a in ax.items():
        a.text(-0.10, 1.12, lbl.lower(),
               transform=a.transAxes,
               fontsize=15, fontweight='bold', va='top', ha='left')

    sns.despine()

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig('../tex/universality.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../tex/universality.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('\nSaved: universality.pdf / universality.png')
