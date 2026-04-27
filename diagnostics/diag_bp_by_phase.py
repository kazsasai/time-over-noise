"""
Diagnostic: B&P avalanche analysis split by dynamical phase
           — with PL / TPL / EXP model comparison (same as fig4/diag_bp_pthreshold)

Separate each trial into:
  - unsolved phase : steps where conflict_ratio > 0
  - solved   phase : steps where conflict_ratio = 0

For each phase, fit three models (PL, TPL, EXP) and select by AIC.
This allows direct comparison with the original mixed analysis:
  τ_large (original) = TPL fit on large-scale CCDF tail → compare with unsolved τ
  τ_small (original) = PL  fit on small-scale CCDF body → compare with solved   τ
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
import seaborn as sns
import warnings

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})
LN10 = np.log(10)

# ── Load ──────────────────────────────────────────────────────────────────────
df  = pd.read_pickle('result_for_criticality_20_3.pkl')
arm = df[df['label'] == 'Async Model'].copy()

p_list   = np.array([0.05, 0.25, 0.45, 0.65, 0.85])
p_colors = sns.color_palette('coolwarm', n_colors=len(p_list))

# ── B&P helpers ───────────────────────────────────────────────────────────────
def bp_avalanches(series, threshold=0):
    avs, in_burst, sz = [], False, 0
    for v in series:
        if v > threshold:
            in_burst = True; sz += v
        else:
            if in_burst: avs.append(sz)
            in_burst, sz = False, 0
    if in_burst: avs.append(sz)
    return np.array(avs, dtype=float)

def compute_ccdf(sizes):
    sizes = np.array(sizes, dtype=float)
    if len(sizes) < 10:
        return None, None, None
    u    = np.unique(sizes)
    ccdf = np.array([np.sum(sizes >= s) for s in u], dtype=float) / len(sizes)
    ls   = np.log10(u)
    lp   = np.log10(ccdf + 1e-12)
    mask = np.isfinite(lp) & (lp > -6)
    return u[mask], ls[mask], lp[mask]

# ── AIC (OLS in log10 P space) ────────────────────────────────────────────────
def aic_ols(n, k, rss):
    if rss <= 0 or n <= k + 1: return np.inf
    return n * np.log(rss / n) + 2 * k

def r2(y, yp):
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ── Model fits (identical to diag_bp_pthreshold.py) ──────────────────────────
def fit_pl(ls, lp):
    n = len(ls)
    if n < 4: return None
    slope, ic, _, _, _ = stats.linregress(ls, lp)
    pred = slope * ls + ic
    rss  = np.sum((lp - pred)**2)
    return {'model': 'PL', 'tau': 1.0 - slope, 'slope': slope, 'ic': ic,
            'R2': r2(lp, pred), 'AIC': aic_ols(n, 2, rss)}

def fit_tpl(ls, lp, s_orig):
    n = len(ls)
    if n < 5: return None
    def model(X, tau, log10_sc, c):
        l, s = X
        return -(tau - 1) * l - s / (10**log10_sc * LN10) + c
    slope0, ic0, _, _, _ = stats.linregress(ls, lp)
    tau0 = 1.0 - slope0
    sc0  = np.log10(max(np.median(s_orig), 2.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = optimize.curve_fit(
                model, (ls, s_orig), lp, p0=[tau0, sc0, ic0],
                bounds=([1.0, -1.0, -20], [30.0, 10.0, 20]), maxfev=20000)
        tau, log10_sc, c = popt
        pred = model((ls, s_orig), *popt)
        rss  = np.sum((lp - pred)**2)
        return {'model': 'TPL', 'tau': tau, 's_c': 10**log10_sc,
                'R2': r2(lp, pred), 'AIC': aic_ols(n, 3, rss)}
    except Exception:
        return None

def fit_exp(ls, lp, s_orig):
    n = len(ls)
    if n < 4: return None
    slope, ic, _, _, _ = stats.linregress(s_orig, lp)
    pred = slope * s_orig + ic
    rss  = np.sum((lp - pred)**2)
    return {'model': 'EXP', 'lambda': -slope * LN10, 'slope': slope, 'ic': ic,
            'R2': r2(lp, pred), 'AIC': aic_ols(n, 2, rss)}

def best_fit(fits):
    valid = {k: v for k, v in fits.items() if v is not None}
    if not valid: return 'N/A', None
    key = min(valid, key=lambda k: valid[k]['AIC'])
    return key, valid[key]

# ── Collect avalanches per phase ──────────────────────────────────────────────
avs_u = {p: [] for p in p_list}
avs_s = {p: [] for p in p_list}

for p in p_list:
    rows = arm[np.abs(arm['p_async'] - p) < 0.01]
    for _, r in rows.iterrows():
        ch = r['history_changed'].astype(float)
        cr = np.array(r['history_conflict_ratio'])
        sm = (cr == 0)
        avs_u[p].extend(bp_avalanches(ch[~sm], threshold=0))
        avs_s[p].extend(bp_avalanches(ch[ sm], threshold=0))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs  = fig.add_gridspec(3, len(p_list), hspace=0.55, wspace=0.35)

COLORS = {'PL': '#2166ac', 'TPL': '#4dac26', 'EXP': '#d01c8b'}
LS     = {'PL': '-',       'TPL': '--',       'EXP': ':'}

summary = []

for ci, p in enumerate(p_list):
    col = p_colors[ci]

    for row_idx, (phase_label, avs_dict) in enumerate(
            [('unsolved', avs_u), ('solved', avs_s)]):
        ax = fig.add_subplot(gs[row_idx, ci])
        sizes = np.array(avs_dict[p])
        s_u, ls, lp = compute_ccdf(sizes)

        if s_u is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.set_title(f'p={p:.2f} [{phase_label}]', fontsize=9)
            sns.despine(ax=ax); continue

        ax.scatter(ls, lp, s=4, alpha=0.45, color=col, rasterized=True, zorder=2)

        fits   = {'PL': fit_pl(ls, lp),
                  'TPL': fit_tpl(ls, lp, s_u),
                  'EXP': fit_exp(ls, lp, s_u)}
        bname, bfit = best_fit(fits)

        # Draw all fits; thicken the best
        ls_plot = np.linspace(ls.min(), ls.max(), 300)
        s_plot  = 10**ls_plot
        for mn, res in fits.items():
            if res is None: continue
            lw = 2.2 if mn == bname else 0.8
            al = 0.95 if mn == bname else 0.45
            if mn == 'PL':
                y = res['slope'] * ls_plot + res['ic']
            elif mn == 'TPL':
                y = (-(res['tau'] - 1) * ls_plot
                     - s_plot / (res['s_c'] * LN10) + fits['TPL']['AIC'] * 0)
                # recalculate cleanly
                popt_tau  = res['tau']
                popt_sc   = res['s_c']
                # need intercept c: back-calculate from first point
                c_est = lp[0] + (popt_tau - 1) * ls[0] + s_u[0] / (popt_sc * LN10)
                y = (-(popt_tau - 1) * ls_plot
                     - s_plot / (popt_sc * LN10) + c_est)
            elif mn == 'EXP':
                y = res['slope'] * s_plot + res['ic']
            ax.plot(ls_plot, y, color=COLORS[mn], lw=lw, ls=LS[mn], alpha=al, zorder=5)

        # Annotation
        tau_best = bfit['tau'] if bfit and 'tau' in bfit else np.nan
        r2_best  = bfit['R2'] if bfit else np.nan
        ax.text(0.97, 0.97,
                f'best={bname}\nτ={tau_best:.2f}\nR²={r2_best:.2f}\nn={len(sizes)}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))

        ax.set_title(f'p={p:.2f}  [{phase_label}]', fontsize=9)
        ax.set_xlabel(r'$\log_{10}\ s$', fontsize=8)
        if ci == 0:
            ax.set_ylabel(rf'$\log_{{10}}\ P(S\geq s)$  [{phase_label}]', fontsize=8)
        sns.despine(ax=ax)

        summary.append({'p': p, 'phase': phase_label, 'best': bname,
                        'tau': tau_best, 'R2': r2_best, 'n': len(sizes)})

# ── Row 2: τ summary ──────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2, :])
for phase_label, marker, clr in [('unsolved', 'o', '#2166ac'),
                                  ('solved',   's', '#d6604d')]:
    sub = [x for x in summary if x['phase'] == phase_label and np.isfinite(x['tau'])]
    ps   = [x['p']   for x in sub]
    taus = [x['tau'] for x in sub]
    bests= [x['best'] for x in sub]
    ax2.plot(ps, taus, marker=marker, color=clr, lw=2, ms=7,
             label=f'{phase_label} (best-AIC τ)')
    for x_, y_, b_ in zip(ps, taus, bests):
        ax2.annotate(b_, (x_, y_), textcoords='offset points',
                     xytext=(4, 3), fontsize=7, color=clr)

ax2.axhline(1.38, color='gray',    lw=1.2, ls=':',  alpha=0.8,
            label=r'$\bar\tau_{\rm large}=1.38$ (mixed TPL)')
ax2.axhline(2.75, color='#999999', lw=1.2, ls='--', alpha=0.6,
            label=r'$\tau_{\rm small}\approx2.5$–$3.0$ (mixed PL)')
ax2.set_xlabel('Freezing probability  p', fontsize=9)
ax2.set_ylabel('Best-fit exponent  τ', fontsize=9)
ax2.set_title('Phase-separated τ vs mixed-analysis reference lines  (threshold=0)',
              fontsize=9)
ax2.legend(fontsize=8); ax2.set_xlim(0, 0.92)
sns.despine(ax=ax2)

plt.suptitle(r'B&P — phase-separated, best-AIC fit (PL/TPL/EXP)  '
             r'ARM $N=20$, $k=3$, threshold=0'
             '\nRow 1: unsolved  |  Row 2: solved  |  Row 3: τ summary',
             fontsize=10, y=1.01)
plt.savefig('../tex/diag_bp_by_phase.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: diag_bp_by_phase.png')

# ── Text summary ──────────────────────────────────────────────────────────────
print(f'\n{"p":>5}  {"phase":>10}  {"best":>5}  {"τ":>7}  {"R²":>7}  {"n":>12}')
print('-' * 55)
for row in summary:
    print(f"{row['p']:5.2f}  {row['phase']:>10}  {row['best']:>5}  "
          f"{row['tau']:7.3f}  {row['R2']:7.3f}  {row['n']:12d}")
