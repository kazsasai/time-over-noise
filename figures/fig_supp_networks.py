"""
Supplementary Figure S3: Solution rate vs p/ε across network topologies.
  Networks: BA (Barabasi-Albert), ER (Erdos-Renyi), WS (Watts-Strogatz)
  N=20, k=3, 200 trials per parameter point, step_max=1000
  p/ε grid: [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from multiprocessing import Pool, cpu_count
import gcg_solver

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# ── Parameters ────────────────────────────────────────────────────────────────
N        = 20
K        = 3
BA_M     = 2
STEP_MAX = 1000
N_TRIALS = 200
N_JOBS   = max(1, cpu_count() - 1)

p_grid = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

# ER: q matched to BA mean degree ≈ 2*m*(N-m)/N(N-1) ≈ 2*ba_m/(N-1)
q_er   = 2 * BA_M / (N - 1)
# WS: k=4 (close to BA mean degree ~4), beta=0.3
k_ws   = 4
beta_ws = 0.3

graphs = [
    ('ba', 'Barabási-Albert\n($m=2$)',        {'ba_m': BA_M}),
    ('er', f'Erdős-Rényi\n($q={q_er:.3f}$)', {'ba_m': BA_M, 'graph_param': q_er}),
    ('ws', f'Watts-Strogatz\n($k={k_ws},\\,\\beta={beta_ws}$)',
                                              {'ba_m': BA_M, 'graph_param': beta_ws}),
]

base_param = dict(
    node_num=N, color_num=K, ba_m=BA_M,
    bot_num=N, bot_type='c',
    step_max=STEP_MAX, return_history=False,
)

# ── Run simulations ───────────────────────────────────────────────────────────
def run_grid(graph_type, graph_kwargs, model_label, is_async, p_or_eps_grid):
    """Return solution rates for one (graph, model) combination."""
    rates = []
    for pval in p_or_eps_grid:
        tasks = []
        for trial in range(N_TRIALS):
            p = {**base_param, **graph_kwargs,
                 'seed':            trial,
                 'label':           model_label,
                 'graph_type':      graph_type,
                 'is_async_model':  is_async,
                 'p_async':  pval if is_async  else 0.0,
                 'p_noise':  0.0  if is_async  else pval,
                 }
            tasks.append(p)
        with Pool(N_JOBS) as pool:
            results = pool.map(gcg_solver._worker_run_single, tasks)
        rates.append(np.mean([r['solved'] for r in results]))
        print(f"  {graph_type:2s} {model_label:3s} p={pval:.2f}  rate={rates[-1]:.3f}")
    return np.array(rates)

if __name__ == '__main__':
    print("=== Running experiments ===")
    data = {}  # (graph_type, model) -> solution rate array

    for gtype, gtitle, gkw in graphs:
        print(f"\n-- {gtype.upper()} --")
        data[(gtype, 'SNM')] = run_grid(gtype, gkw, 'SNM', False, p_grid)
        data[(gtype, 'ARM')] = run_grid(gtype, gkw, 'ARM', True,  p_grid)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n=== Solution Rate Summary ===")
    header = f"{'p/ε':>5}" + "".join(f"  {gt:>6}-SNM  {gt:>6}-ARM" for gt,_,_ in graphs)
    print(header)
    print("-" * len(header))
    for i, pval in enumerate(p_grid):
        row = f"{pval:5.2f}"
        for gtype, _, _ in graphs:
            row += f"  {data[(gtype,'SNM')][i]:9.3f}  {data[(gtype,'ARM')][i]:9.3f}"
        print(row)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                             gridspec_kw={'wspace': 0.12})

    col_snm = '#E8986A'
    col_arm = 'steelblue'

    for ax, (gtype, gtitle, _) in zip(axes, graphs):
        snm_r = data[(gtype, 'SNM')]
        arm_r = data[(gtype, 'ARM')]

        ax.plot(p_grid, snm_r, color=col_snm, lw=2.2, marker='s', ms=7,
                label='SNM', zorder=3)
        ax.plot(p_grid, arm_r, color=col_arm, lw=2.2, marker='o', ms=7,
                label='ARM', zorder=3)

        ax.set_xlabel(r'$p$ (ARM) / $\varepsilon$ (SNM)', fontsize=11)
        ax.set_title(gtitle, fontsize=11, pad=8)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        sns.despine(ax=ax)

    axes[0].set_ylabel('Solution rate', fontsize=11)
    axes[0].legend(fontsize=10, framealpha=0.85, loc='upper left')

    # panel labels
    for ax, lbl in zip(axes, ['a', 'b', 'c']):
        ax.text(-0.10, 1.08, lbl, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top', ha='left')

    fig.suptitle(
        f'Figure S3: Solution rate across network topologies ($N={N}$, $k={K}$, {N_TRIALS} trials)',
        fontsize=12, y=1.02)

    plt.savefig('../tex/supp_fig_s3_networks.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('../tex/supp_fig_s3_networks.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: supp_fig_s3_networks.pdf/png")
