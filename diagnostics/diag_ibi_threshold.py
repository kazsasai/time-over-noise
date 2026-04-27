"""
Diagnostic: IBI CCDF at N=50, k=3, ARM with different thresholds
Explores which threshold gives α≈2.0 (MF-DP) for inter-burst intervals.

Rows: threshold (5, 10, 15, N/5=10, percentile-based)
Cols: p values (0.25, 0.45, 0.65)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import powerlaw, warnings, io, contextlib, seaborn as sns

df = pd.read_pickle('result_criticality_multiN_k3_long.pkl')
df50 = df[(df['node_num'] == 50) & (df['label'] == 'Async Model')].copy()

def fmt_log10(val, pos):
    if val == int(val): return rf'$10^{{{int(val)}}}$'
    return rf'$10^{{{val:.1f}}}$'
log10_fmt = FuncFormatter(fmt_log10)

def beggs_plenz(series, thr):
    avs = []; in_b = False; sz = 0
    for v in series:
        if v > thr: in_b = True; sz += v
        else:
            if in_b: avs.append(sz)
            in_b = False; sz = 0
    if in_b: avs.append(sz)
    return avs

def get_ibi(series, thr):
    ivs = []; q = 0; in_b = False; first = False
    for v in series:
        if v > thr:
            if first and not in_b: ivs.append(q)
            in_b = True; first = True; q = 0
        else:
            if in_b: in_b = False
            if first: q += 1
    return ivs

def fit_pl(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 50:
        return np.nan, np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                fit = powerlaw.Fit(arr, discrete=True, verbose=False)
        n_above = np.sum(arr >= fit.xmin)
        rng = np.log10(arr.max()) - np.log10(max(fit.xmin, 1.0))
        if n_above < 50 or rng < 1.0:
            return np.nan, fit.xmin, rng
        return fit.alpha, fit.xmin, rng
    except Exception:
        return np.nan, np.nan, np.nan

P_SHOW = [0.25, 0.45, 0.65]
THRESHOLDS = [1, 5, 10, 15]
palette = sns.color_palette('tab10', n_colors=len(P_SHOW))
MF_DP_ALPHA = 2.0

fig, axes = plt.subplots(len(THRESHOLDS), len(P_SHOW),
                         figsize=(13, 4 * len(THRESHOLDS)), constrained_layout=True)

for row, thr in enumerate(THRESHOLDS):
    for col, p in enumerate(P_SHOW):
        ax = axes[row, col]
        rows = df50[df50['p_async'] == p]

        all_ivs = []
        for _, r in rows.iterrows():
            all_ivs.extend(get_ibi(r['history_changed'], thr))
        all_ivs = np.array(all_ivs, dtype=float)
        nonzero = all_ivs[all_ivs > 0]

        pct_active = np.mean([v > thr for _, r in rows.iterrows()
                              for v in r['history_changed']])

        if len(nonzero) > 50:
            unique_i = np.unique(nonzero)
            ccdf = np.array([np.sum(nonzero >= s) for s in unique_i], dtype=float) / len(nonzero)
            log_i = np.log10(unique_i)
            log_c = np.log10(ccdf + 1e-12)
            finite = np.isfinite(log_c) & (log_c > -6)
            ax.plot(log_i[finite], log_c[finite], color=palette[col], lw=1.5, alpha=0.8)

            alpha, xmin, rng = fit_pl(nonzero)
            if np.isfinite(alpha) and np.isfinite(xmin):
                # Fit line
                ls_min = np.log10(max(xmin, 1))
                ls_max = log_i[finite].max() - 0.1
                idx_x = np.argmin(np.abs(unique_i - xmin))
                ic = log_c[idx_x] + alpha * ls_min
                x_seg = np.array([ls_min, ls_max])
                ax.plot(x_seg, -alpha * x_seg + ic, color=palette[col],
                        lw=2.0, ls='--', alpha=0.9, zorder=5,
                        label=rf'$\alpha={alpha:.2f}$  xmin={xmin:.0f}')
                ax.axvline(ls_min, color=palette[col], lw=0.8, ls=':', alpha=0.4)

            # MF-DP reference slope
            if len(log_i[finite]) > 5:
                anc_x = log_i[finite][len(log_i[finite])//3]
                anc_idx = np.argmin(np.abs(log_i[finite] - anc_x))
                anc_y = log_c[finite][anc_idx]
                xr = np.array([anc_x, log_i[finite].max()])
                ax.plot(xr, anc_y - MF_DP_ALPHA * (xr - anc_x),
                        color='gray', lw=1.5, ls='-', alpha=0.45,
                        label=rf'$\alpha={MF_DP_ALPHA:.0f}$ (MF-DP)')

        ax.xaxis.set_major_formatter(log10_fmt)
        ax.yaxis.set_major_formatter(log10_fmt)
        ax.set_xlabel(r'IBI length $\ell$', fontsize=9)
        ax.set_ylabel(r'$P(L\geq\ell)$', fontsize=9)
        ax.set_title(f'thr={thr}  p={p:.2f}  ({pct_active:.0%} active)', fontsize=10)
        ax.legend(fontsize=7.5, framealpha=0.85, loc='upper right')
        sns.despine(ax=ax)

plt.suptitle('N=50, k=3, ARM — IBI CCDF for different thresholds', fontsize=13, y=1.01)
plt.savefig('../tex/diag_ibi_threshold.png', dpi=180, bbox_inches='tight')
print("Saved: diag_ibi_threshold.png")
