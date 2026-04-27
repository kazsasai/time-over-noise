"""
Fig: 2x2 bar chart — 4 update rules compared at ε=0
  ① Sync + Stay:   p=0, ε=0  (pure deterministic)
  ② Async + Stay:  p=0.45, ε=0
  ③ Sync + Random: p=0, ε=0
  ④ Async + Random (ARM): p=0.45, ε=0

N=20, k=3, BA m=2, 500 trials, step_max=1000
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from multiprocessing import Pool, cpu_count
import gcg_solver

plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

N        = 20
K        = 3
STEP_MAX = 1000
N_TRIALS = 500
N_JOBS   = max(1, cpu_count() - 1)
P_ASYNC  = 0.45   # representative async value

base = dict(node_num=N, color_num=K, ba_m=2, bot_num=N, bot_type='c',
            step_max=STEP_MAX, return_history=False)

def run_rate(p_async, is_async_model, p_noise, seeds):
    tasks = [{**base, 'seed': s,
              'p_async': p_async, 'p_noise': p_noise,
              'is_async_model': is_async_model, 'label': 'run'} for s in seeds]
    with Pool(N_JOBS) as pool:
        results = pool.map(gcg_solver._worker_run_single, tasks)
    solved = [r['solved'] for r in results]
    return np.mean(solved), np.std(solved) / np.sqrt(len(solved))  # mean, SE

if __name__ == '__main__':
    seeds = list(range(N_TRIALS))

    print("Running 4 conditions (ε=0 for all) ...")

    print("① Sync + Stay   (p=0,    ε=0) ...")
    r1, se1 = run_rate(0.0, False, 0.0, seeds)
    print(f"  rate={r1:.3f} ± {se1:.3f}")

    print("② Async + Stay  (p=0.45, ε=0) ...")
    r2, se2 = run_rate(P_ASYNC, False, 0.0, seeds)
    print(f"  rate={r2:.3f} ± {se2:.3f}")

    print("③ Sync + Random (p=0,    ε=0) ...")
    r3, se3 = run_rate(0.0, True, 0.0, seeds)
    print(f"  rate={r3:.3f} ± {se3:.3f}")

    print("④ Async + Random / ARM (p=0.45, ε=0) ...")
    r4, se4 = run_rate(P_ASYNC, True, 0.0, seeds)
    print(f"  rate={r4:.3f} ± {se4:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    labels  = ['(1)\nSync\n+ Stay', '(2)\nAsync\n+ Stay',
               '(3)\nSync\n+ Random', '(4)\nAsync\n+ Random\n(ARM)']
    rates   = [r1, r2, r3, r4]
    errors  = [se1, se2, se3, se4]

    # colour: Stay=orange, Random=blue; pattern: Sync=hatch, Async=solid
    col_stay = '#E8986A'
    col_rand = 'steelblue'
    colors   = [col_stay, col_stay, col_rand, col_rand]
    hatches  = ['///', '', '///', '']

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, rates, yerr=errors, capsize=6,
                  color=colors, hatch=hatches,
                  edgecolor='white', linewidth=1.2,
                  error_kw=dict(ecolor='#333333', lw=1.5),
                  width=0.55, zorder=3)

    # add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Solution rate', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_title(f'2×2 factorial: Timing × Gridlock rule\n'
                 f'($N={N}$, $k={K}$, $\\varepsilon=0$, '
                 f'async at $p={P_ASYNC}$, {N_TRIALS} trials)',
                 fontsize=11, pad=10)
    ax.axhline(0, color='black', lw=0.8)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # legend
    p1 = mpatches.Patch(facecolor=col_stay, hatch='///', edgecolor='white', label='Stay on gridlock')
    p2 = mpatches.Patch(facecolor=col_rand, hatch='///', edgecolor='white', label='Random on gridlock')
    p3 = mpatches.Patch(facecolor='white',  hatch='///', edgecolor='grey',  label='Synchronous ($p=0$)')
    p4 = mpatches.Patch(facecolor='white',  hatch='',    edgecolor='grey',  label=f'Asynchronous ($p={P_ASYNC}$)')
    ax.legend(handles=[p1, p2, p3, p4], fontsize=9, framealpha=0.9,
              loc='upper left', ncol=2)

    sns.despine(ax=ax)
    plt.tight_layout()

    plt.savefig('../tex/fig_2x2_bar.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('../tex/fig_2x2_bar.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: fig_2x2_bar.pdf/png")
