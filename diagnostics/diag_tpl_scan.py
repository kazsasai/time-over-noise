"""
TPL fit scan for ARM p=0.05 B&P CCDF, xmin from 10^2.4 upward
"""
import sys
sys.path.insert(0, '/Users/kazuto/Library/CloudStorage/Dropbox/2025/202500_Papers/202503_Nature_GPG/claude/src')
import os
os.chdir('/Users/kazuto/Library/CloudStorage/Dropbox/2025/202500_Papers/202503_Nature_GPG/claude/src')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
import warnings

df = pd.read_pickle('result_for_criticality_20_3.pkl')

# ARM p=0.05 avalanche sizes
mask = (df['label'] == 'Async Model') & (abs(df['p_async'] - 0.05) < 0.01)
rows = df[mask]

all_sizes = []
for _, row in rows.iterrows():
    sig = np.array(row['history_changed'], dtype=float)
    # B&P: contiguous active bursts
    in_burst = False
    burst_size = 0
    for v in sig:
        if v > 0:
            in_burst = True
            burst_size += v
        else:
            if in_burst and burst_size > 0:
                all_sizes.append(burst_size)
            in_burst = False
            burst_size = 0
    if in_burst and burst_size > 0:
        all_sizes.append(burst_size)

sizes = np.array(all_sizes)
print(f"Total avalanches: {len(sizes)}, max={sizes.max():.0f}")

# Scan xmin values
xmin_vals = [10**2.4, 10**2.6, 10**2.8, 10**3.0, 10**3.2, 10**3.5]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

results = []

for i, xmin in enumerate(xmin_vals):
    ax = axes[i]
    tail = sizes[sizes >= xmin]
    n_tail = len(tail)

    if n_tail < 50:
        ax.set_title(f'xmin=10^{np.log10(xmin):.1f}: n={n_tail} (too few)')
        results.append({'xmin': xmin, 'n': n_tail, 'alpha': np.nan, 'cutoff': np.nan, 'ks_tpl': np.nan})
        continue

    # CCDF
    s_sorted = np.sort(tail)
    ccdf = 1 - np.arange(1, len(s_sorted)+1) / len(s_sorted)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(tail, xmin=xmin, discrete=False, verbose=False)

        alpha_pl  = fit.power_law.alpha
        ks_pl     = fit.power_law.KS(tail)

        alpha_tpl = fit.truncated_power_law.alpha
        cutoff    = fit.truncated_power_law.parameter2  # 1/lambda
        ks_tpl    = fit.truncated_power_law.KS(tail)

        # LR tests
        R_pl_tpl, p_pl_tpl = fit.distribution_compare('power_law', 'truncated_power_law')
        R_tpl_exp, p_tpl_exp = fit.distribution_compare('truncated_power_law', 'exponential')

    dec_range = np.log10(tail.max()) - np.log10(xmin)

    results.append({
        'xmin': xmin, 'n': n_tail,
        'alpha_tpl': alpha_tpl, 'cutoff': cutoff, 'ks_tpl': ks_tpl,
        'alpha_pl': alpha_pl, 'ks_pl': ks_pl,
        'R_pl_tpl': R_pl_tpl, 'p_pl_tpl': p_pl_tpl,
        'R_tpl_exp': R_tpl_exp, 'p_tpl_exp': p_tpl_exp,
        'dec_range': dec_range
    })

    print(f"xmin=10^{np.log10(xmin):.1f}  n={n_tail:5d}  range={dec_range:.2f}dec"
          f"  TPL: α={alpha_tpl:.3f}  cut={cutoff:.0f}  KS={ks_tpl:.4f}"
          f"  PL: α={alpha_pl:.3f}  KS={ks_pl:.4f}"
          f"  LR(PL>TPL): R={R_pl_tpl:.3f} p={p_pl_tpl:.4f}"
          f"  LR(TPL>Exp): R={R_tpl_exp:.3f} p={p_tpl_exp:.4f}")

    # Plot
    ax.scatter(np.log10(s_sorted), np.log10(ccdf+1e-10), s=3, alpha=0.4, color='steelblue', label='data')

    # TPL curve
    s_fit = np.linspace(xmin, s_sorted.max(), 500)
    # P(S>=s) ∝ ∫_s^∞ x^{-α} exp(-x/λ) dx  — approximate with empirical normalization
    tpl_pdf_unnorm = s_fit**(-alpha_tpl) * np.exp(-s_fit / cutoff)
    # normalize to CCDF at xmin
    ccdf_at_xmin = 1.0  # relative
    from scipy.integrate import cumtrapz
    tpl_cdf_unnorm = np.concatenate([[0], np.cumsum(tpl_pdf_unnorm[:-1] * np.diff(s_fit))])
    tpl_cdf_unnorm /= tpl_cdf_unnorm[-1]
    tpl_ccdf = (1 - tpl_cdf_unnorm) * ccdf[0]

    # PL curve
    pl_ccdf = (s_fit / xmin) ** (-(alpha_pl - 1)) * ccdf[0]

    ax.plot(np.log10(s_fit), np.log10(tpl_ccdf+1e-10), 'r-', lw=2,
            label=f'TPL α={alpha_tpl:.2f} cut={cutoff:.0f}')
    ax.plot(np.log10(s_fit), np.log10(pl_ccdf+1e-10), 'g--', lw=1.5,
            label=f'PL α={alpha_pl:.2f}')

    ax.axvline(np.log10(xmin), color='k', ls=':', lw=1)
    ax.set_xlabel('log10(s)')
    ax.set_ylabel('log10 CCDF')
    ax.set_title(f'xmin=10^{np.log10(xmin):.1f}  n={n_tail}  TPL KS={ks_tpl:.3f}')
    ax.legend(fontsize=7)

plt.suptitle('ARM p=0.05 B&P: TPL fit scan', fontsize=13)
plt.tight_layout()
plt.savefig('/Users/kazuto/Library/CloudStorage/Dropbox/2025/202500_Papers/202503_Nature_GPG/claude/tex/diag_tpl_scan.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: diag_tpl_scan.png")
