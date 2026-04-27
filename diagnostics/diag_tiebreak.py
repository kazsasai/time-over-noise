"""
Diagnostic: Deterministic tie-break vs random tie-break

Test the hypothesis that Fluctuation B (random tie-breaking when multiple
optimal colors exist) inflates the small-avalanche region and distorts the
power-law CCDF.

Two ARM variants:
  - random_tb : standard ARM (random choice among conflict-free colors)
  - det_tb    : deterministic ARM (always choose lowest-index conflict-free color)

N=20, k=3, 300 trials, step_max=50000
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import pandas as pd
import networkx as nx
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.insert(0, '.')


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True)
def _step_random_tb(node_num, color_num, node_colors, adj_indices, adj_indptr,
                    p_async, rng):
    """Standard ARM: random tie-breaking."""
    next_cand = node_colors.copy()
    for i in range(node_num):
        ns = adj_indptr[i]; ne = adj_indptr[i + 1]
        used = np.zeros(color_num + 1, dtype=np.bool_)
        for j in range(ns, ne):
            used[node_colors[adj_indices[j]]] = True
        avail = []
        for c in range(1, color_num + 1):
            if not used[c]:
                avail.append(c)
        if len(avail) > 0:
            next_cand[i] = avail[rng.integers(0, len(avail))]
        else:
            next_cand[i] = rng.integers(1, color_num + 1)   # gridlock escape
    act = rng.random(node_num) < (1.0 - p_async)
    out = node_colors.copy()
    out[act] = next_cand[act]
    return out


@njit(cache=True)
def _step_det_tb(node_num, color_num, node_colors, adj_indices, adj_indptr,
                 p_async, rng):
    """Deterministic tie-break ARM: always choose minimum-index conflict-free color."""
    next_cand = node_colors.copy()
    for i in range(node_num):
        ns = adj_indptr[i]; ne = adj_indptr[i + 1]
        used = np.zeros(color_num + 1, dtype=np.bool_)
        for j in range(ns, ne):
            used[node_colors[adj_indices[j]]] = True
        first_avail = -1
        for c in range(1, color_num + 1):
            if not used[c]:
                first_avail = c
                break
        if first_avail > 0:
            next_cand[i] = first_avail                       # deterministic
        else:
            next_cand[i] = rng.integers(1, color_num + 1)   # gridlock escape (random)
    act = rng.random(node_num) < (1.0 - p_async)
    out = node_colors.copy()
    out[act] = next_cand[act]
    return out


# ── Single-trial runner ────────────────────────────────────────────────────────

def run_trial(seed, p, node_num, color_num, step_max, deterministic):
    net = nx.barabasi_albert_graph(node_num, 2, seed=seed)
    adj = nx.adjacency_matrix(net).tocsr()
    adj_indices = adj.indices
    adj_indptr  = adj.indptr
    edges = np.array(net.edges())

    rng = np.random.default_rng(seed)
    colors = rng.integers(1, color_num + 1, size=node_num)

    history = []
    step_fn = _step_det_tb if deterministic else _step_random_tb

    for _ in range(step_max):
        prev = colors.copy()
        colors = step_fn(node_num, color_num, colors, adj_indices, adj_indptr, p, rng)
        history.append(int(np.sum(prev != colors)))

    return np.array(history, dtype=np.int32)


# ── Avalanche extraction ───────────────────────────────────────────────────────

def beggs_plenz(series, thr=1):
    avs = []
    in_b = False; sz = 0
    for v in series:
        if v > thr:
            in_b = True; sz += v
        else:
            if in_b: avs.append(sz)
            in_b = False; sz = 0
    if in_b: avs.append(sz)
    return avs


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import powerlaw
    import warnings, io, contextlib

    NODE_NUM  = 20
    COLOR_NUM = 3
    STEP_MAX  = 50000
    N_TRIALS  = 200
    P_VALUES  = [0.05, 0.25, 0.45]
    MASTER_SEED = 9999

    rng0  = np.random.default_rng(MASTER_SEED)
    seeds = rng0.integers(0, 10**9, size=N_TRIALS)

    # Warm-up Numba
    print("Warming up Numba...")
    dummy_net = nx.barabasi_albert_graph(5, 2, seed=0)
    da = nx.adjacency_matrix(dummy_net).tocsr()
    drng = np.random.default_rng(0)
    dc = drng.integers(1, 4, size=5)
    _step_random_tb(5, 3, dc, da.indices, da.indptr, 0.1, drng)
    _step_det_tb(5, 3, dc, da.indices, da.indptr, 0.1, drng)
    print("Done.")

    results = {}

    for variant, det in [('random_tb', False), ('det_tb', True)]:
        results[variant] = {}
        for p in P_VALUES:
            print(f"{variant}  p={p:.2f}  running {N_TRIALS} trials …")
            all_sizes = []
            for s in seeds:
                hist = run_trial(int(s), p, NODE_NUM, COLOR_NUM, STEP_MAX, det)
                all_sizes.extend(beggs_plenz(hist, thr=1))
            all_sizes = np.array(all_sizes, dtype=float)
            unique_s = np.unique(all_sizes)
            ccdf = np.array([np.sum(all_sizes >= s) for s in unique_s], dtype=float) / len(all_sizes)
            log_s = np.log10(unique_s)
            log_c = np.log10(ccdf + 1e-12)
            finite = np.isfinite(log_c) & (log_c > -6)

            # MLE fit
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    buf = io.StringIO()
                    with contextlib.redirect_stderr(buf):
                        fit = powerlaw.Fit(all_sizes, discrete=True, verbose=False)
                tau = fit.alpha
                xmin = fit.xmin
                n_above = np.sum(all_sizes >= xmin)
                sr = np.log10(all_sizes.max()) - np.log10(max(xmin, 1.0))
                tag = f'τ={tau:.3f} xmin={xmin:.0f} n={n_above} range={sr:.1f}dec'
            except Exception as e:
                tau = np.nan; tag = f'FIT ERROR {e}'

            results[variant][p] = {
                'log_s': log_s[finite], 'log_c': log_c[finite],
                'tau': tau, 'tag': tag, 'n': len(all_sizes)
            }
            print(f"  n_avalanches={len(all_sizes):,}  {tag}")

    # ── Plot comparison ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(P_VALUES), figsize=(5 * len(P_VALUES), 4.5))
    colors_p = ['tab:orange', 'tab:blue', 'tab:green']

    for ax, p, col in zip(axes, P_VALUES, colors_p):
        for variant, ls, label in [('random_tb', '-', 'random TB'), ('det_tb', '--', 'det TB')]:
            d = results[variant][p]
            ax.plot(d['log_s'], d['log_c'], lw=1.5, ls=ls, color=col,
                    alpha=0.85, label=f'{label}\n{d["tag"]}')
        ax.set_title(f'p = {p:.2f}', fontsize=12)
        ax.set_xlabel('log₁₀ s')
        ax.set_ylabel('log₁₀ P(S≥s)')
        ax.legend(fontsize=7, framealpha=0.85)

    plt.suptitle('ARM: random tie-break vs deterministic tie-break\nN=20, k=3', fontsize=13)
    plt.tight_layout()
    plt.savefig('../tex/diag_tiebreak.png', dpi=180, bbox_inches='tight')
    print("\nSaved: ../tex/diag_tiebreak.png")
