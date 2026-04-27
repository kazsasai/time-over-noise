"""
Supplementary figure: Finite-size scaling of the ARM solution plateau.

Extracts, from result_for_plateau_k34.pkl (k=4, N=20..100, p=0..1):
  (A) Heatmap of ARM solution rate over (p, N), with the plateau lower
      edge p_min(N) overlaid at two thresholds.
  (B) Plateau lower edge (1 - p_min(N)) vs N on log-log axes, with a
      least-squares fit to (1 - p_min) = c * N^(-alpha).
  (C) Scaling collapse: solution rate vs the effective active-agent
      count (1 - p) * sqrt(N). All N curves collapse onto a master curve.

The scaling constant c and exponent alpha are printed to stdout.
"""

import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
COL_ARM = sns.color_palette('colorblind')[1]

OUT_PDF = '../tex/supp_fig_plateau_scaling.pdf'
OUT_PNG = '../tex/supp_fig_plateau_scaling.png'
K_FOCUS = 4
THRESHOLDS = [0.99, 0.95, 0.90]

# ── Load + aggregate ─────────────────────────────────────────────────────────
df = pd.read_pickle('./result_for_plateau_k34.pkl')
arm = df[(df['label'] == 'Async') & (df['color_num'] == K_FOCUS)]
mat = (arm.groupby(['node_num', 'p_async'])['solved']
          .mean().unstack().sort_index())
Ns = mat.index.values
ps = mat.columns.values

# ── Extract plateau lower edge p_min(N) at each threshold ───────────────────
def p_min_edge(row, thr):
    ok = row[row >= thr]
    if len(ok) == 0:
        return np.nan
    return ok.index.min()

edges = {thr: np.array([p_min_edge(mat.loc[N], thr) for N in Ns])
         for thr in THRESHOLDS}

# ── Fit (1 - p_min) = c * N^(-alpha) using thr=0.99 on N >= 60 ──────────────
thr_fit = 0.99
N_fit_min = 60
mask = (Ns >= N_fit_min) & np.isfinite(edges[thr_fit])
logN = np.log(Ns[mask])
logq = np.log(1.0 - edges[thr_fit][mask])
alpha_neg, logc = np.polyfit(logN, logq, 1)
alpha = -alpha_neg
c_fit = np.exp(logc)
print(f"Fit on N >= {N_fit_min} @ thr={thr_fit}:")
print(f"  (1 - p_min) = {c_fit:.3f} * N^(-{alpha:.3f})")
print(f"  c_sqrt  = (1-p_min) * sqrt(N) ~ {c_fit:.2f}  (if alpha ~ 0.5)")

print(f"\nPer-N check at thr={thr_fit}:")
print(f"{'N':>4}  {'p_min':>6}  {'(1-p)*sqrt(N)':>14}")
for N, pm in zip(Ns, edges[thr_fit]):
    if np.isfinite(pm):
        print(f"{N:4d}  {pm:6.3f}  {(1-pm)*np.sqrt(N):14.2f}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
ax_A, ax_B, ax_C = axes

# Panel A: heatmap (p, N) ----------------------------------------------------
cmap_heat = LinearSegmentedColormap.from_list(
    'sol', ['#F7F7F7', '#92C5DE', '#0571B0'], N=256)
im = ax_A.imshow(mat.values, vmin=0, vmax=1, cmap=cmap_heat,
                 aspect='auto', origin='lower', interpolation='nearest',
                 extent=[ps.min() - 0.025, ps.max() + 0.025,
                         Ns.min() - 5, Ns.max() + 5])

# Overlay plateau lower-edge curves
edge_styles = [(0.99, 'o', 'black'),
               (0.95, 's', '#D62728'),
               (0.90, '^', '#555555')]
for thr, mk, col in edge_styles:
    yN = Ns[np.isfinite(edges[thr])]
    xp = edges[thr][np.isfinite(edges[thr])]
    ax_A.plot(xp, yN, marker=mk, ms=6, color=col, lw=1.6,
              label=fr'$p_{{\min}}$ (rate $\geq$ {thr})', zorder=5)

ax_A.set_xlabel(r'Freezing probability $p$', fontsize=11)
ax_A.set_ylabel(r'Network size $N$', fontsize=11)
ax_A.set_title(f'ARM solution rate  ($k={K_FOCUS}$)',
               fontsize=12, color=COL_ARM, fontweight='bold', pad=8)
ax_A.set_xlim(-0.02, 1.02)
# Upper-right, shifted 0.1 to the left in axis fraction (so the legend's
# right edge sits at x=0.9 instead of clipping the colorbar)
ax_A.legend(loc='upper right', bbox_to_anchor=(0.9, 1.0),
            fontsize=8, framealpha=0.95,
            handlelength=1.4, borderpad=0.4, labelspacing=0.3)
fig.colorbar(im, ax=ax_A, fraction=0.04, pad=0.02, label='Solution rate')

# Panel B: (1 - p_min) vs N, log-log with power-law fit ----------------------
for thr, mk, col in edge_styles:
    q = 1.0 - edges[thr]
    good = np.isfinite(q) & (q > 0)
    ax_B.loglog(Ns[good], q[good], marker=mk, color=col, lw=1.5, ms=7,
                label=fr'rate $\geq$ {thr}')

# Fit line
Nx = np.linspace(Ns[mask].min(), Ns.max(), 100)
ax_B.loglog(Nx, c_fit * Nx**(-alpha), ls='--', color='k', lw=1.4,
            label=fr'fit: $(1-p_{{\min}}) = {c_fit:.2f}\,N^{{-{alpha:.2f}}}$')
# Reference: N^(-1/2)
ref_c = (1 - edges[thr_fit][Ns == 100][0]) * np.sqrt(100)
ax_B.loglog(Nx, ref_c / np.sqrt(Nx), ls=':', color='gray', lw=1.2,
            label=r'slope $-1/2$ reference')

ax_B.set_xlabel(r'Network size $N$', fontsize=11)
ax_B.set_ylabel(r'$1 - p_{\min}(N)$  (active fraction)', fontsize=11)
ax_B.set_title('Plateau edge scaling', fontsize=12, pad=8, fontweight='bold')
ax_B.legend(loc='lower left', fontsize=8.5, framealpha=0.95,
            handlelength=1.6, borderpad=0.4, labelspacing=0.3)
ax_B.grid(which='both', alpha=0.25)

# Panel C: Scaling collapse --------------------------------------------------
# Use viridis as a continuous colormap for N; replace per-N legend with a
# colorbar to avoid 9-entry overlap.
n_colors = len(Ns)
norm_N = Normalize(vmin=Ns.min(), vmax=Ns.max())
cmap_N = plt.cm.viridis
markers = ['o', 's', '^', 'D', 'v', 'P', '<', '>', '*'][:n_colors]

for N, mk in zip(Ns, markers):
    row = mat.loc[N]
    x = (1 - row.index.values) * np.sqrt(N)
    y = row.values
    valid = (row.index.values > 0.0) & (row.index.values < 1.0)
    ax_C.plot(x[valid], y[valid], marker=mk, ms=4,
              color=cmap_N(norm_N(N)), lw=1.2, alpha=0.9)

# Reference vertical line for the fit constant
axv = ax_C.axvline(c_fit, color='k', ls='--', lw=1.4, zorder=3,
                   label=fr'$(1-p)\sqrt{{N}} = {c_fit:.2f}$')

ax_C.set_xlabel(r'$(1 - p)\sqrt{N}$  (effective active count / $\sqrt{N}$)',
                fontsize=11)
ax_C.set_ylabel('Solution rate', fontsize=11)
ax_C.set_title('Scaling collapse', fontsize=12, pad=8, fontweight='bold')
ax_C.set_xscale('log')
ax_C.set_xlim(left=0.4)
ax_C.set_ylim(-0.03, 1.05)

# Compact legend with only the reference line (lower-left, away from
# the master curve which is high-rate at large (1-p)sqrt(N))
ax_C.legend(loc='lower left', fontsize=9, framealpha=0.95,
            handlelength=1.8, borderpad=0.4)
ax_C.grid(which='both', alpha=0.25)

# Colorbar showing N
sm = ScalarMappable(norm=norm_N, cmap=cmap_N)
sm.set_array([])
cbar_C = fig.colorbar(sm, ax=ax_C, fraction=0.04, pad=0.02,
                      label=r'Network size $N$')
cbar_C.set_ticks(Ns)
cbar_C.ax.tick_params(labelsize=8)

# Panel labels
for lbl, ax in zip('abc', axes):
    ax.text(-0.12, 1.05, lbl, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

plt.savefig(OUT_PDF, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
