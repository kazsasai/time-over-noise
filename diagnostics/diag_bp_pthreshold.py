"""
Diagnostic: B&P CCDF split at P(S≥s) = 10^-3 (y-axis threshold)
ARM, N=20, k=3, all 10 p values

For each p, split CCDF at P(S≥s) = 10^-3:
  small scale: P(S≥s) >= 10^-3   (body)
  large scale: P(S≥s) <  10^-3   (tail)

For each region, fit:
  PL  : log10 P = -(τ-1)*log10(s) + c         [linear in log-log]
  TPL : log10 P = -(τ-1)*log10(s) - s/(s_c*ln10) + c  [power law × exp cutoff]
  EXP : log10 P = -λ/ln10 * s + c             [linear in log P vs s]

Model comparison via AIC (OLS in log10 P space).
Best model = lowest AIC.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
import seaborn as sns
import warnings

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_pickle('result_for_criticality_20_3.pkl')
p_list_all = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

CCDF_SPLIT = 1e-3        # y-axis threshold
LN10       = np.log(10)

# ── Avalanche helpers ─────────────────────────────────────────────────────────
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
    """Returns (s_unique, log10_s, log10_ccdf) for finite, non-trivial points."""
    all_sizes = np.array(all_sizes, dtype=float)
    if len(all_sizes) < 10:
        return None, None, None
    unique_s = np.unique(all_sizes)
    ccdf_u   = np.array([np.sum(all_sizes >= s) for s in unique_s], dtype=float) / len(all_sizes)
    log_s    = np.log10(unique_s)
    log_ccdf = np.log10(ccdf_u + 1e-12)
    finite   = np.isfinite(log_ccdf) & (log_ccdf > -6)
    return unique_s[finite], log_s[finite], log_ccdf[finite]

# ── AIC (OLS in log10 P space) ────────────────────────────────────────────────
def aic_ols(n, k, rss):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k

def r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ── Model fits ────────────────────────────────────────────────────────────────
def fit_pl(log10_s, log10_P):
    """PL: log10 P = -(τ-1)*log10_s + c"""
    n = len(log10_s)
    if n < 4:
        return None
    slope, intercept, _, _, _ = stats.linregress(log10_s, log10_P)
    pred = slope * log10_s + intercept
    rss  = np.sum((log10_P - pred)**2)
    return {'model': 'PL', 'tau': 1.0 - slope, 'slope': slope,
            'intercept': intercept, 'R2': r2(log10_P, pred),
            'AIC': aic_ols(n, 2, rss), 'n': n}

def fit_tpl(log10_s, log10_P, s_orig):
    """TPL: log10 P = -(τ-1)*log10_s - s/(s_c*ln10) + c"""
    n = len(log10_s)
    if n < 5:
        return None

    def tpl_model(X, tau, log10_sc, c):
        ls, s = X
        return -(tau - 1) * ls - s / (10**log10_sc * LN10) + c

    # initial guess from PL
    slope0, ic0, _, _, _ = stats.linregress(log10_s, log10_P)
    tau0 = 1.0 - slope0
    sc0  = np.log10(max(np.median(s_orig), 2.0))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = optimize.curve_fit(
                tpl_model, (log10_s, s_orig), log10_P,
                p0=[tau0, sc0, ic0],
                bounds=([1.0, -1.0, -20], [30.0, 10.0, 20]),
                maxfev=20000)
        tau, log10_sc, c = popt
        pred = tpl_model((log10_s, s_orig), *popt)
        rss  = np.sum((log10_P - pred)**2)
        return {'model': 'TPL', 'tau': tau, 's_c': 10**log10_sc,
                'log10_sc': log10_sc, 'c': c,
                'R2': r2(log10_P, pred), 'AIC': aic_ols(n, 3, rss), 'n': n}
    except Exception:
        return None

def fit_exp(log10_s, log10_P, s_orig):
    """EXP: log10 P = -(λ/ln10)*s + c  (λ in natural units)"""
    n = len(log10_s)
    if n < 4:
        return None
    slope, intercept, _, _, _ = stats.linregress(s_orig, log10_P)
    pred = slope * s_orig + intercept
    rss  = np.sum((log10_P - pred)**2)
    return {'model': 'EXP', 'lambda': -slope * LN10, 'slope': slope,
            'intercept': intercept, 'R2': r2(log10_P, pred),
            'AIC': aic_ols(n, 2, rss), 'n': n}

def select_best(fits):
    """Return name of model with lowest AIC."""
    valid = {k: v for k, v in fits.items() if v is not None}
    if not valid:
        return 'N/A'
    return min(valid, key=lambda k: valid[k]['AIC'])

# ── Plot helpers ──────────────────────────────────────────────────────────────
COLORS = {
    'small': {'PL': '#2166ac', 'TPL': '#4dac26', 'EXP': '#d01c8b'},
    'large': {'PL': '#b2182b', 'TPL': '#e08214', 'EXP': '#762a83'},
}
LS = {'PL': '-', 'TPL': '--', 'EXP': ':'}

def plot_fit_lines(ax, fits, best, log10_s, s_orig, region):
    ls_plot = np.linspace(log10_s.min(), log10_s.max(), 300)
    s_plot  = 10**ls_plot

    for mname, res in fits.items():
        if res is None:
            continue
        lw = 2.2 if mname == best else 0.9
        al = 0.95 if mname == best else 0.55
        col = COLORS[region][mname]

        if mname == 'PL':
            y = res['slope'] * ls_plot + res['intercept']
            ax.plot(ls_plot, y, color=col, lw=lw, ls=LS[mname], alpha=al, zorder=5)

        elif mname == 'TPL':
            tau = res['tau']
            sc  = res['s_c']
            c   = res['c']
            y = -(tau - 1) * ls_plot - s_plot / (sc * LN10) + c
            ax.plot(ls_plot, y, color=col, lw=lw, ls=LS[mname], alpha=al, zorder=5)

        elif mname == 'EXP':
            y = res['slope'] * s_plot + res['intercept']
            ax.plot(ls_plot, y, color=col, lw=lw, ls=LS[mname], alpha=al, zorder=5)

# ── Main loop ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
plt.subplots_adjust(wspace=0.38, hspace=0.50)
axes = axes.flatten()

summary_rows = []
split_log10  = np.log10(CCDF_SPLIT)

print(f"{'p':>5}  {'region':>6}  {'n':>5}  {'best':>4}  "
      f"{'PL_τ':>7}  {'PL_R2':>6}  {'PL_AIC':>8}  "
      f"{'TPL_τ':>7}  {'TPL_sc':>9}  {'TPL_R2':>6}  {'TPL_AIC':>8}  "
      f"{'EXP_λ':>10}  {'EXP_R2':>6}  {'EXP_AIC':>8}")
print('-' * 130)

for idx, p in enumerate(p_list_all):
    lp = f'{p:.2f}'
    ax = axes[idx]

    rows = df[(df['label'] == 'Async Model') & (abs(df['p_async'] - p) < 0.01)]
    all_avs = [s for _, r in rows.iterrows()
               for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]

    s_unique, log10_s, log10_P = compute_ccdf(all_avs)
    if s_unique is None:
        ax.set_visible(False)
        continue

    # CCDF scatter
    ax.scatter(log10_s, log10_P, s=5, alpha=0.4, color='#888888', zorder=2, rasterized=True)

    # threshold line
    ax.axhline(split_log10, color='black', lw=0.8, ls=':', alpha=0.5, zorder=1)
    ax.text(log10_s.min() + 0.05, split_log10 + 0.05,
            r'$10^{-3}$', fontsize=7, va='bottom', alpha=0.7)

    for region, mask in [('small', log10_P >= split_log10),
                          ('large', log10_P <  split_log10)]:
        if mask.sum() < 4:
            continue

        ls_r  = log10_s[mask]
        lp_r  = log10_P[mask]
        s_r   = s_unique[mask]

        pl_r  = fit_pl(ls_r, lp_r)
        tpl_r = fit_tpl(ls_r, lp_r, s_r)
        exp_r = fit_exp(ls_r, lp_r, s_r)
        fits  = {'PL': pl_r, 'TPL': tpl_r, 'EXP': exp_r}
        best  = select_best(fits)

        plot_fit_lines(ax, fits, best, ls_r, s_r, region)

        # highlight best-fit region dots
        dot_col = COLORS[region][best] if best != 'N/A' else '#888888'
        ax.scatter(ls_r, lp_r, s=8, alpha=0.7, color=dot_col, zorder=3, rasterized=True)

        # collect summary
        row = {'p': p, 'region': region, 'n': int(mask.sum()), 'best': best,
               'PL_tau':  pl_r['tau']   if pl_r  else np.nan,
               'PL_R2':   pl_r['R2']    if pl_r  else np.nan,
               'PL_AIC':  pl_r['AIC']   if pl_r  else np.nan,
               'TPL_tau': tpl_r['tau']  if tpl_r else np.nan,
               'TPL_sc':  tpl_r['s_c']  if tpl_r else np.nan,
               'TPL_R2':  tpl_r['R2']   if tpl_r else np.nan,
               'TPL_AIC': tpl_r['AIC']  if tpl_r else np.nan,
               'EXP_lam': exp_r['lambda'] if exp_r else np.nan,
               'EXP_R2':  exp_r['R2']     if exp_r else np.nan,
               'EXP_AIC': exp_r['AIC']    if exp_r else np.nan}
        summary_rows.append(row)

        print(f"{p:5.2f}  {region:>6}  {mask.sum():>5}  {best:>4}  "
              f"{row['PL_tau']:7.3f}  {row['PL_R2']:6.3f}  {row['PL_AIC']:8.1f}  "
              f"{row['TPL_tau']:7.3f}  {row['TPL_sc']:9.1f}  {row['TPL_R2']:6.3f}  {row['TPL_AIC']:8.1f}  "
              f"{row['EXP_lam']:10.5f}  {row['EXP_R2']:6.3f}  {row['EXP_AIC']:8.1f}")

    # legend patch (region label only)
    ax.text(0.98, 0.97, f'small→{next((r["best"] for r in summary_rows if r["p"]==p and r["region"]=="small"), "?")}\n'
                        f'large→{next((r["best"] for r in summary_rows if r["p"]==p and r["region"]=="large"), "?")}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))

    ax.set_title(rf'ARM  $p={p:.2f}$', fontsize=10)
    ax.set_xlabel(r'$\log_{10}\ s$', fontsize=8)
    ax.set_ylabel(r'$\log_{10}\ P(S \geq s)$', fontsize=8)
    sns.despine(ax=ax)

# ── Global legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color=COLORS['small']['PL'],  lw=2, ls='-',  label='small PL'),
    Line2D([0],[0], color=COLORS['small']['TPL'], lw=2, ls='--', label='small TPL'),
    Line2D([0],[0], color=COLORS['small']['EXP'], lw=2, ls=':',  label='small EXP'),
    Line2D([0],[0], color=COLORS['large']['PL'],  lw=2, ls='-',  label='large PL'),
    Line2D([0],[0], color=COLORS['large']['TPL'], lw=2, ls='--', label='large TPL'),
    Line2D([0],[0], color=COLORS['large']['EXP'], lw=2, ls=':',  label='large EXP'),
]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

plt.suptitle(r'B&P CCDF — ARM ($N=20$, $k=3$)  |  split: $P(S \geq s) = 10^{-3}$'
             '\nLine width: thick = best AIC;  color: blue/green/pink = small,  red/orange/purple = large',
             fontsize=10, y=1.01)

plt.savefig('../tex/diag_bp_pthreshold.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved: diag_bp_pthreshold.png')

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n=== Summary: best model per p and region ===')
df_sum = pd.DataFrame(summary_rows)
print(df_sum[['p', 'region', 'n', 'best',
              'PL_tau', 'PL_R2',
              'TPL_tau', 'TPL_sc', 'TPL_R2',
              'EXP_lam', 'EXP_R2']].to_string(index=False, float_format='{:.3f}'.format))
