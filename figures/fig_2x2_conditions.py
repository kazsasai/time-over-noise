"""
Fig: 2×2 factorial design — Timing × Gridlock rule
  Rows:    Synchronous (p_async=0)  vs  Asynchronous (p_async varies)
  Columns: Stay on gridlock         vs  Random on gridlock

Layout: 1 row, 2 panels
  Panel A: solution rate vs p_async for Async conditions (②, ④)
           + horizontal reference lines for Sync conditions (①, ③) at p_async=0
  Panel B: (optional extension) show ε sweep for SNM ① and Sync+Random ③

N=20, k=3, BA m=2, 500 trials, step_max=1000
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from multiprocessing import Pool, cpu_count
import gcg_solver

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

N        = 20
K        = 3
STEP_MAX = 1000
N_TRIALS = 500
N_JOBS   = max(1, cpu_count() - 1)

p_grid   = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
eps_grid = np.array([0.01, 0.03, 0.05, 0.08, 0.15, 0.25, 0.35, 0.50, 0.70, 0.95])

base = dict(node_num=N, color_num=K, ba_m=2, bot_num=N, bot_type='c',
            step_max=STEP_MAX, return_history=False)

# ── helper ────────────────────────────────────────────────────────────────────
def run_rate(p_async, is_async_model, p_noise, seeds):
    tasks = [{**base, 'seed': s,
              'p_async': p_async, 'p_noise': p_noise,
              'is_async_model': is_async_model, 'label': 'run'} for s in seeds]
    with Pool(N_JOBS) as pool:
        results = pool.map(gcg_solver._worker_run_single, tasks)
    return np.mean([r['solved'] for r in results])

if __name__ == '__main__':
    seeds = list(range(N_TRIALS))

    # ── Condition ① SNM (Sync + Stay) : vary ε ──────────────────────────────────
    print("① Sync+Stay (SNM): vary ε ...")
    rate_snm = []
    for eps in eps_grid:
        r = run_rate(0.0, False, eps, seeds)
        rate_snm.append(r)
        print(f"  ε={eps:.2f}  rate={r:.3f}")

    # ── Condition ③ Sync+Random : vary ε (on top of random-gridlock) ─────────────
    print("③ Sync+Random: vary ε ...")
    rate_sync_rnd = []
    for eps in eps_grid:
        r = run_rate(0.0, True, eps, seeds)
        rate_sync_rnd.append(r)
        print(f"  ε={eps:.2f}  rate={r:.3f}")

    # ── Condition ② Async+Stay : vary p ──────────────────────────────────────────
    print("② Async+Stay: vary p ...")
    rate_async_stay = []
    for p in p_grid:
        r = run_rate(p, False, 0.0, seeds)
        rate_async_stay.append(r)
        print(f"  p={p:.2f}  rate={r:.3f}")

    # ── Condition ④ ARM (Async+Random) : vary p ───────────────────────────────────
    print("④ Async+Random (ARM): vary p ...")
    rate_arm = []
    for p in p_grid:
        r = run_rate(p, True, 0.0, seeds)
        rate_arm.append(r)
        print(f"  p={p:.2f}  rate={r:.3f}")

    rate_snm        = np.array(rate_snm)
    rate_sync_rnd   = np.array(rate_sync_rnd)
    rate_async_stay = np.array(rate_async_stay)
    rate_arm        = np.array(rate_arm)

    # ── Summary ───────────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'param':>6}  {'(1)SNM':>8}  {'(3)Sync+Rnd':>12}  {'(2)Async+Stay':>14}  {'(4)ARM':>8}")
    for i in range(len(p_grid)):
        print(f"{p_grid[i]:6.2f}  {rate_snm[i]:8.3f}  {rate_sync_rnd[i]:10.3f}"
              f"  {rate_async_stay[i]:12.3f}  {rate_arm[i]:8.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True,
                             gridspec_kw={'wspace': 0.08})

    col_stay = '#E8986A'   # orange — Stay on gridlock
    col_rand = 'steelblue' # blue   — Random on gridlock
    ls_sync  = '--'
    ls_async = '-'

    # ── Panel A: Sync conditions (ε sweep) ────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps_grid, rate_snm,      color=col_stay, ls=ls_sync,  lw=2.2,
            marker='s', ms=7, label='(1) Sync + Stay (SNM)',    zorder=3)
    ax.plot(eps_grid, rate_sync_rnd, color=col_rand, ls=ls_sync,  lw=2.2,
            marker='^', ms=7, label='(3) Sync + Random',         zorder=3)

    ax.set_xlabel(r'State-noise probability $\varepsilon$', fontsize=11)
    ax.set_title('Synchronous updating\n($p=0$, vary $\\varepsilon$)', fontsize=11, pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('Solution rate', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85, loc='upper right')
    ax.text(-0.10, 1.08, 'a', transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    sns.despine(ax=ax)

    # ── Panel B: Async conditions (p sweep) ───────────────────────────────────────
    ax = axes[1]
    ax.plot(p_grid, rate_async_stay, color=col_stay, ls=ls_async, lw=2.2,
            marker='s', ms=7, label='(2) Async + Stay',           zorder=3)
    ax.plot(p_grid, rate_arm,        color=col_rand, ls=ls_async, lw=2.2,
            marker='o', ms=7, label='(4) Async + Random (ARM)',   zorder=3)

    ax.set_xlabel(r'Freezing probability $p$', fontsize=11)
    ax.set_title('Asynchronous updating\n($\\varepsilon=0$, vary $p$)', fontsize=11, pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=10, framealpha=0.85, loc='upper right')
    ax.text(-0.06, 1.08, 'b', transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    sns.despine(ax=ax)

    fig.suptitle(
        f'2×2 factorial: Timing × Gridlock rule  ($N={N}$, $k={K}$, {N_TRIALS} trials)',
        fontsize=12, y=1.02)

    plt.savefig('../tex/fig_2x2_conditions.pdf', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.savefig('../tex/fig_2x2_conditions.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: fig_2x2_conditions.pdf/png")
