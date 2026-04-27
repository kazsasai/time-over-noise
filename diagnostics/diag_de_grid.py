"""
Reference image for panels D and E: scatter only + grid lines.
Use this to determine anchor coordinates for reference slope lines.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import seaborn as sns
import warnings
from scipy import stats, optimize

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

LN10 = np.log(10)

def fmt_log10(val, _pos):
    if val == int(val):
        return rf'$10^{{{int(val)}}}$'
    return rf'$10^{{{val:.1f}}}$'
log10_fmt = FuncFormatter(fmt_log10)

# ── reuse analysis results from fig4_criticality ──────────────────────────────
# (copy minimal code to compute CCDFs and knees)

df = pd.read_pickle('result_for_criticality_20_3.pkl')

p_list_display = np.array([0.05, 0.25, 0.45, 0.65, 0.85])
p_list_bp_all  = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
palette_disp   = sns.color_palette('tab10', n_colors=len(p_list_display))

def beggs_plenz_avalanches(series, threshold=1):
    avalanches, in_burst, burst_size = [], False, 0
    for val in series:
        if val > threshold:
            in_burst, burst_size = True, burst_size + val
        else:
            if in_burst:
                avalanches.append(burst_size)
            in_burst, burst_size = False, 0
    if in_burst:
        avalanches.append(burst_size)
    return avalanches

def compute_ccdf(all_sizes):
    all_sizes = np.array(all_sizes, dtype=float)
    if len(all_sizes) < 10:
        return None, None, None
    unique_s = np.unique(all_sizes)
    ccdf_u   = np.array([np.sum(all_sizes >= s) for s in unique_s], dtype=float) / len(all_sizes)
    log_s    = np.log10(unique_s)
    log_ccdf = np.log10(ccdf_u + 1e-12)
    finite   = np.isfinite(log_ccdf) & (log_ccdf > -6)
    return unique_s[finite], log_s[finite], log_ccdf[finite]

def knee_piecewise(log10_s, log10_P, min_pts=8, lP_cutoff=-4.0):
    n = len(log10_s)
    if n < 2 * min_pts:
        return None
    body = log10_P >= lP_cutoff
    best_rss_fb, best_xk_fb, best_pars_fb = np.inf, None, None
    best_ds, best_xk, best_pars = -np.inf, None, None
    for i in range(min_pts, n - min_pts):
        if not body[i]:
            continue
        xk = log10_s[i]
        ml = log10_s <= xk
        mr = (log10_s > xk) & body
        if ml.sum() < min_pts or mr.sum() < min_pts:
            continue
        sl, il, _, _, _ = stats.linregress(log10_s[ml], log10_P[ml])
        sr, ir, _, _, _ = stats.linregress(log10_s[mr], log10_P[mr])
        rss = (np.sum((log10_P[ml] - (sl * log10_s[ml] + il))**2) +
               np.sum((log10_P[mr] - (sr * log10_s[mr] + ir))**2))
        if rss < best_rss_fb:
            best_rss_fb, best_xk_fb, best_pars_fb = rss, xk, (sl, il, sr, ir)
        ds = sr - sl
        if ds > 0 and ds > best_ds:
            best_ds, best_xk, best_pars = ds, xk, (sl, il, sr, ir)
    if best_xk is None:
        best_xk, best_pars = best_xk_fb, best_pars_fb
    if best_xk is None:
        return None
    sl, il, sr, ir = best_pars
    return {'xk': best_xk}

# ── compute CCDFs ─────────────────────────────────────────────────────────────
snm_ccdfs = {}
for p in p_list_display:
    lp   = f'{p:.2f}'
    rows = df[(df['label'] == 'Noise Model') & (abs(df['p_noise'] - p) < 0.01)]
    avs  = [s for _, r in rows.iterrows()
            for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
    s_u, ls, lP = compute_ccdf(avs)
    if s_u is not None:
        snm_ccdfs[lp] = (ls, lP)

arm_ccdfs = {}
for p in p_list_display:
    lp   = f'{p:.2f}'
    rows = df[(df['label'] == 'Async Model') & (abs(df['p_async'] - p) < 0.01)]
    avs  = [s for _, r in rows.iterrows()
            for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
    s_u, ls, lP = compute_ccdf(avs)
    if s_u is not None:
        knee = knee_piecewise(ls, lP)
        xk   = knee['xk'] if knee else np.median(ls)
        arm_ccdfs[lp] = (ls, lP, xk)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plt.subplots_adjust(wspace=0.35)

# Panel D: SNM scatter + grid
ax = axes[0]
for idx, p in enumerate(p_list_display):
    lp = f'{p:.2f}'
    if lp not in snm_ccdfs: continue
    ls, lP = snm_ccdfs[lp]
    ax.scatter(ls, lP, facecolors='none', edgecolors=palette_disp[idx],
               s=16, alpha=0.6, label=f'ε={lp}', zorder=2, linewidths=0.8)

ax.set_title('D — SNM CCDF (scatter only)', fontsize=12, pad=6)
ax.set_xlabel(r'$\log_{10}\ s$', fontsize=11)
ax.set_ylabel(r'$\log_{10}\ P(S \geq s)$', fontsize=11)
ax.xaxis.set_major_formatter(log10_fmt)
ax.yaxis.set_major_formatter(log10_fmt)
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.grid(True, which='major', ls='--', lw=0.7, alpha=0.6)
ax.grid(True, which='minor', ls=':',  lw=0.4, alpha=0.4)
ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
sns.despine(ax=ax)

# Panel E: ARM scatter (filled=small, open=large) + knee lines + grid
ax = axes[1]
for idx, p in enumerate(p_list_display):
    lp = f'{p:.2f}'
    if lp not in arm_ccdfs: continue
    ls, lP, xk = arm_ccdfs[lp]
    color  = palette_disp[idx]
    mask_s = ls <= xk
    mask_l = ls >  xk
    ax.scatter(ls[mask_s], lP[mask_s], facecolors=color, edgecolors=color,
               s=12, alpha=0.7, label=f'p={lp}', zorder=2, linewidths=0.6, rasterized=True)
    ax.scatter(ls[mask_l], lP[mask_l], facecolors='none', edgecolors=color,
               s=12, alpha=0.5, zorder=2, linewidths=0.6, rasterized=True)
    ax.axvline(xk, color=color, lw=0.7, ls=':', alpha=0.5, zorder=1)

ax.set_title('E — ARM CCDF (filled=small, open=large)', fontsize=12, pad=6)
ax.set_xlabel(r'$\log_{10}\ s$', fontsize=11)
ax.set_ylabel(r'$\log_{10}\ P(S \geq s)$', fontsize=11)
ax.xaxis.set_major_formatter(log10_fmt)
ax.yaxis.set_major_formatter(log10_fmt)
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.grid(True, which='major', ls='--', lw=0.7, alpha=0.6)
ax.grid(True, which='minor', ls=':',  lw=0.4, alpha=0.4)
ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
sns.despine(ax=ax)

plt.savefig('../tex/diag_de_grid.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_de_grid.png')
