"""
Supplementary Figure S4: Hybrid parameter sweeps.
  Left:  SNM (ε=0.05) fixed, vary p_async
  Right: ARM (p=0.45)  fixed, vary p_noise

Data from result_static_random_20_3.pkl
N=20, k=3, BA m=2
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

COL_SNM = sns.color_palette('colorblind')[0]
COL_ARM = sns.color_palette('colorblind')[1]

# ── Load data ─────────────────────────────────────────────────────────────────
df_sr    = pd.read_pickle('./result_static_random_20_3.pkl')
df_noise = df_sr.query('label == "Static Model" and p_noise == 0.05')
df_async = df_sr.query('label == "Random Model" and p_async == 0.45')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True,
                         gridspec_kw={'wspace': 0.10})

# ── Panel A: SNM(ε=0.05), vary p ─────────────────────────────────────────────
sns.lineplot(df_noise, ax=axes[0], x='p_async', y='solved',
             linewidth=2.5, color=COL_SNM, marker='o', ms=8)
axes[0].set_title(r'SNM ($\varepsilon = 0.05$) + vary $p$', fontsize=11, pad=8)
axes[0].set_xlabel(r'Freezing probability $p$', fontsize=11)
axes[0].set_ylabel('Solution Rate in 1000 Steps', fontsize=11)
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.05, 1.05)
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axes[0].text(-0.10, 1.10, 'a', transform=axes[0].transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
sns.despine(ax=axes[0])

# ── Panel B: ARM(p=0.45), vary ε ─────────────────────────────────────────────
sns.lineplot(df_async, ax=axes[1], x='p_noise', y='solved',
             linewidth=2.5, color=COL_ARM, marker='o', ms=8)
axes[1].set_title(r'ARM ($p = 0.45$) + vary $\varepsilon$', fontsize=11, pad=8)
axes[1].set_xlabel(r'State-noise probability $\varepsilon$', fontsize=11)
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.05, 1.05)
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
axes[1].text(-0.06, 1.10, 'b', transform=axes[1].transAxes,
             fontsize=15, fontweight='bold', va='top', ha='left')
sns.despine(ax=axes[1])

fig.suptitle('Hybrid parameter sweeps ($N=20$, $k=3$)',
             fontsize=12, y=1.02)

plt.savefig('../tex/supp_fig_s4_hybrid.pdf', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.savefig('../tex/supp_fig_s4_hybrid.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: supp_fig_s4_hybrid.pdf/png")
