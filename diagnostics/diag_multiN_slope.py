"""
Diagnostic: CCDF local slope across N values (k=3 fixed)

Loads result_criticality_multiN_k3.pkl and plots:
  - Left panels : CCDF  (log-log)
  - Right panels: local slope d(log CCDF)/d(log s)  with MF-DP reference τ=1.5

One row per N, columns = p values of interest.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import powerlaw
import warnings, io, contextlib

import os
pkl = ('result_criticality_multiN_k3_long.pkl'
       if os.path.exists('result_criticality_multiN_k3_long.pkl')
       else 'result_criticality_multiN_k3.pkl')
print(f"Loading: {pkl}")
df = pd.read_pickle(pkl)

N_VALUES = sorted(df['node_num'].unique())
P_SHOW   = [0.05, 0.25, 0.45, 0.65]
palette  = sns.color_palette('tab10', n_colors=len(P_SHOW))

def beggs_plenz(series, thr=1):
    avs = []; in_b = False; sz = 0
    for v in series:
        if v > thr: in_b = True; sz += v
        else:
            if in_b: avs.append(sz)
            in_b = False; sz = 0
    if in_b: avs.append(sz)
    return avs

def mle_tau(all_sizes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                fit = powerlaw.Fit(all_sizes, discrete=True, verbose=False)
        return fit.alpha, fit.xmin
    except Exception:
        return np.nan, np.nan

# ── Build summary table ────────────────────────────────────────────────────────
print(f"{'N':>5} {'p':>5} {'n_av':>10} {'τ_MLE':>7} {'xmin':>7} {'τ=1.5 range':>25}")
print('-' * 65)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(N_VALUES), 2, figsize=(13, 4.5 * len(N_VALUES)))

for row, N in enumerate(N_VALUES):
    ax_c  = axes[row, 0]
    ax_sl = axes[row, 1]

    for ci, p in enumerate(P_SHOW):
        rows = df.query('node_num == @N and label == "Async Model" and p_async == @p')
        if len(rows) == 0:
            continue
        all_sizes = []
        for hist in rows['history_changed']:
            all_sizes.extend(beggs_plenz(hist, thr=1))
        all_sizes = np.array(all_sizes, dtype=float)

        unique_s = np.unique(all_sizes)
        ccdf     = np.array([np.sum(all_sizes >= s) for s in unique_s], dtype=float) / len(all_sizes)
        log_s    = np.log10(unique_s)
        log_c    = np.log10(ccdf + 1e-12)
        finite   = np.isfinite(log_c) & (log_c > -6)
        ls = log_s[finite]; lc = log_c[finite]

        col = palette[ci]
        tau, xmin = mle_tau(all_sizes)

        ax_c.plot(ls, lc, color=col, lw=1.5, label=f'p={p:.2f}  τ={tau:.2f}')

        # Local slope
        if len(ls) > 10:
            slope   = np.gradient(lc, ls)
            win     = max(5, len(ls) // 25)
            s_smooth = uniform_filter1d(slope, size=win)
            ax_sl.plot(ls, s_smooth, color=col, lw=1.5, label=f'p={p:.2f}')

            # Find τ≈1.5 region (slope ≈ -0.5)
            close = np.abs(s_smooth + 0.5) < 0.15
            s_range_str = (f"[{ls[close].min():.2f},{ls[close].max():.2f}]"
                           if np.any(close) else "—")
        else:
            s_range_str = "—"

        print(f"{N:>5} {p:>5.2f} {len(all_sizes):>10,} {tau:>7.3f} {xmin:>7.0f} {s_range_str:>25}")

    # Formatting
    ax_c.set_title(f'N={N}, k=3  ARM CCDF', fontsize=12)
    ax_c.set_xlabel('log₁₀ s'); ax_c.set_ylabel('log₁₀ P(S≥s)')
    ax_c.legend(fontsize=8, framealpha=0.85)

    ax_sl.axhline(-0.5, color='gray', lw=1.5, ls='--', label='τ=1.5 (MF-DP)')
    ax_sl.axhline(-1.0, color='gray', lw=1.0, ls=':', alpha=0.5, label='τ=2.0')
    ax_sl.axhline(-2.0, color='gray', lw=1.0, ls=':', alpha=0.5, label='τ=3.0')
    ax_sl.set_title(f'N={N}, k=3  local CCDF slope', fontsize=12)
    ax_sl.set_xlabel('log₁₀ s'); ax_sl.set_ylabel('d(log CCDF)/d(log s)')
    ax_sl.set_ylim(-8, 0.5)
    ax_sl.legend(fontsize=8, framealpha=0.85)

plt.suptitle('ARM: CCDF & local slope vs N  (k=3, BA m=2)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('../tex/diag_multiN_slope.png', dpi=180, bbox_inches='tight')
print("\nSaved: diag_multiN_slope.png")
