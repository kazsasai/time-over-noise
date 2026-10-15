"""
Fig. 4: SOC signatures — 2×3 layout
FFT panels (A/B/C) : N=20, k=3  (1,000 trials, 50,000 steps)
  Signal: history_conflict_ratio (macroscopic order parameter)
  ARM: β ≈ 0.59 robustly across all p (R²≥0.98)
  SNM: β ≈ 0.29 only at ε=0.05; white noise elsewhere
B&P panels (D/E/F) : N=20, k=3  (same dataset)
  Signal: history_changed (activity = changed nodes per step)
  ARM: knee auto-detected via direction-constrained piecewise OLS
       small scale (left of knee): PL/TPL fit
       large scale (right of knee): TPL fit → τ_large ≈ 1.4 universal

Row 1: FFT power spectra   — SNM (A) | ARM (B) | β vs p (C)
Row 2: Avalanche size CCDF — SNM (D) | ARM (E) | τ_BP vs p (F)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats, optimize
import seaborn as sns
import warnings

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

def fmt_log10(val, _pos):
    if val == int(val):
        return rf'$10^{{{int(val)}}}$'
    return rf'$10^{{{val:.1f}}}$'
log10_fmt = FuncFormatter(fmt_log10)

LN10 = np.log(10)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_pickle('result_for_criticality_20_3.pkl')

node_num = 20
step_max = int(df['history_conflict_ratio'].iloc[0].shape[0])

print(f"All data: N={node_num}, step_max={step_max}")
print("FFT signal : history_conflict_ratio (order parameter)")
print("B&P signal : history_changed (activity)")

p_list_display = np.array([0.05, 0.25, 0.45, 0.65, 0.85])
p_list         = p_list_display
p_list_bp_all  = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

# ── FFT matrices (conflict_ratio) ─────────────────────────────────────────────
freq_max  = step_max // 2
frequency = np.arange(1, freq_max) / step_max
log_freq  = np.log10(frequency)

df_fft_n = {}
df_fft_a = {}

for p in p_list_bp_all:
    lp = f'{p:.2f}'
    rows_n = df[(df['label'] == 'Noise Model') & (abs(df['p_noise'] - p) < 0.01)]
    rows_a = df[(df['label'] == 'Async Model') & (abs(df['p_async'] - p) < 0.01)]
    df_fft_n[lp] = np.mean(
        [np.log10(np.abs(np.fft.fft(r['history_conflict_ratio'].astype(float)))[1:freq_max] + 1e-12)
         for _, r in rows_n.iterrows()], axis=0)
    df_fft_a[lp] = np.mean(
        [np.log10(np.abs(np.fft.fft(r['history_conflict_ratio'].astype(float)))[1:freq_max] + 1e-12)
         for _, r in rows_a.iterrows()], axis=0)

# ── FFT fits (all 10 p values) ────────────────────────────────────────────────
fft_fit_lo, fft_fit_hi = -3.5, -1.0
fft_fit_mask = (log_freq >= fft_fit_lo) & (log_freq <= fft_fit_hi)

betas_arm, betas_snm = [], []

for p in p_list_bp_all:
    lp = f'{p:.2f}'
    sl, ic, r, _, _ = stats.linregress(log_freq[fft_fit_mask], df_fft_a[lp][fft_fit_mask])
    betas_arm.append(-sl if r**2 >= 0.90 else np.nan)

    sl, ic, r, _, _ = stats.linregress(log_freq[fft_fit_mask], df_fft_n[lp][fft_fit_mask])
    betas_snm.append(-sl if r**2 >= 0.50 else np.nan)

betas_arm = np.array(betas_arm)
betas_snm = np.array(betas_snm)

print("=== FFT β (N=20, k=3, conflict_ratio) ===")
print(f"ARM (R²≥0.90): {np.nanmean(betas_arm):.3f} ± {np.nanstd(betas_arm):.3f}")
for p_val, b in zip(p_list_bp_all, betas_snm):
    tag = f"β={b:.3f}" if np.isfinite(b) else "white noise (R²<0.50)"
    print(f"  SNM ε={p_val:.2f}: {tag}")

# ── B&P avalanche helpers ──────────────────────────────────────────────────────
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

# ── Knee detection (direction-constrained piecewise OLS) ─────────────────────
def knee_piecewise(log10_s, log10_P, min_pts=8, lP_cutoff=-4.0):
    """
    Find physical knee: transition from steep (small) to shallower (large) scale.
    Constraint: right segment shallower (sr > sl); search body only (log10_P >= lP_cutoff).
    Falls back to min-RSS within body if no direction-valid candidate exists.
    """
    n = len(log10_s)
    if n < 2 * min_pts:
        return None

    body = log10_P >= lP_cutoff

    best_rss_fb, best_xk_fb, best_pars_fb = np.inf, None, None
    best_ds,     best_xk,    best_pars    = -np.inf, None, None

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
    return {'xk': best_xk, 'sl': sl, 'il': il, 'sr': sr, 'ir': ir,
            'tau_l': 1 - sl, 'tau_r': 1 - sr, 'delta_slope': sr - sl}

# ── Model fits ────────────────────────────────────────────────────────────────
def aic_ols(n, k, rss):
    return n * np.log(rss / n) + 2 * k if (rss > 0 and n > k + 1) else np.inf

def _r2(y, yp):
    ss = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - yp)**2) / ss if ss > 0 else np.nan

def fit_pl(ls, lP):
    n = len(ls)
    if n < 4: return None
    sl, ic, _, _, _ = stats.linregress(ls, lP)
    pr = sl * ls + ic
    rss = np.sum((lP - pr)**2)
    return {'model': 'PL', 'tau': 1 - sl, 'sl': sl, 'ic': ic,
            'R2': _r2(lP, pr), 'AIC': aic_ols(n, 2, rss)}

def fit_tpl(ls, lP, s_orig):
    n = len(ls)
    if n < 5: return None
    def tpl(X, tau, lsc, c):
        l, s = X
        return -(tau - 1) * l - s / (10**lsc * LN10) + c
    sl0, ic0, *_ = stats.linregress(ls, lP)
    sc0 = np.log10(max(np.median(s_orig), 2.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = optimize.curve_fit(tpl, (ls, s_orig), lP,
                                         p0=[1 - sl0, sc0, ic0],
                                         bounds=([1.0, -1, -20], [30, 10, 20]),
                                         maxfev=20000)
        tau, lsc, c = popt
        pr = tpl((ls, s_orig), *popt)
        rss = np.sum((lP - pr)**2)
        return {'model': 'TPL', 'tau': tau, 's_c': 10**lsc, 'lsc': lsc, 'c': c,
                'R2': _r2(lP, pr), 'AIC': aic_ols(n, 3, rss), '_popt': popt}
    except Exception:
        return None

def fit_exp(ls, lP, s_orig):
    n = len(ls)
    if n < 4: return None
    sl, ic, _, _, _ = stats.linregress(s_orig, lP)
    pr = sl * s_orig + ic
    rss = np.sum((lP - pr)**2)
    return {'model': 'EXP', 'lambda': -sl * LN10, 'sl': sl, 'ic': ic,
            'R2': _r2(lP, pr), 'AIC': aic_ols(n, 2, rss)}

def best_of(fits):
    v = {k: f for k, f in fits.items() if f is not None}
    return min(v, key=lambda k: v[k]['AIC']) if v else 'N/A'

# ── SNM analysis: PL / TPL / EXP on full CCDF (no splitting) ─────────────────
def analyse_bp_snm_full():
    """
    Fit PL, TPL, EXP to the entire SNM CCDF for all 10 p values.
    Best model selected by AIC.  No knee splitting (SNM has no two-scale structure).
    """
    out = {}
    print(f"\n{'ε':>5}  {'n':>6}  {'best':>4}  {'exp':>8}  "
          f"{'PL_R2':>6}  {'TPL_R2':>6}  {'EXP_R2':>6}")
    print('-' * 52)
    for p in p_list_bp_all:
        lp   = f'{p:.2f}'
        rows = df[(df['label'] == 'Noise Model') & (abs(df['p_noise'] - p) < 0.01)]
        all_avs = [s for _, r in rows.iterrows()
                   for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
        s_u, ls, lP = compute_ccdf(all_avs)
        if s_u is None:
            out[lp] = None
            continue
        fits  = {'PL':  fit_pl(ls, lP),
                 'TPL': fit_tpl(ls, lP, s_u),
                 'EXP': fit_exp(ls, lP, s_u)}
        bname = best_of(fits)
        bf    = fits[bname]
        exp_val = bf['tau'] if bname != 'EXP' else bf['lambda']
        exp_lbl = 'τ' if bname != 'EXP' else 'λ'
        print(f"{p:5.2f}  {len(ls):6d}  {bname:>4}  "
              f"{exp_val:7.3f}{exp_lbl}  "
              f"{fits['PL']['R2'] if fits['PL'] else np.nan:6.3f}  "
              f"{fits['TPL']['R2'] if fits['TPL'] else np.nan:6.3f}  "
              f"{fits['EXP']['R2'] if fits['EXP'] else np.nan:6.3f}")
        out[lp] = {'fits': fits, 'best': bname, 'exponent': exp_val, 'exp_lbl': exp_lbl,
                   'ccdf': (ls, lP, s_u)}
    return out

# ── ARM analysis: knee-based split, model comparison ─────────────────────────
def analyse_bp_arm_knee():
    """
    For each p in p_list_bp_all:
      1. Compute CCDF
      2. Detect knee (piecewise OLS, direction-constrained)
      3. Fit PL/TPL/EXP to each region → select best by AIC
    Returns per-p dicts with knee info, fit results, CCDF arrays.
    """
    out = {}
    print(f"\n{'p':>5}  {'xk':>6}  {'Δsl':>6}  "
          f"{'τ_small':>8}  {'best_s':>6}  {'τ_large':>8}  {'best_l':>6}")
    print('-' * 58)

    for p in p_list_bp_all:
        lp   = f'{p:.2f}'
        rows = df[(df['label'] == 'Async Model') & (abs(df['p_async'] - p) < 0.01)]
        all_avs = [s for _, r in rows.iterrows()
                   for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
        s_u, ls, lP = compute_ccdf(all_avs)
        if s_u is None:
            out[lp] = None
            continue

        knee = knee_piecewise(ls, lP)
        xk   = knee['xk'] if knee else np.median(ls)

        mask_s = ls <= xk
        mask_l = ls >  xk

        region = {}
        for rname, mask in [('small', mask_s), ('large', mask_l)]:
            if mask.sum() < 4:
                region[rname] = None
                continue
            ls_r, lP_r, s_r = ls[mask], lP[mask], s_u[mask]
            fits  = {'PL': fit_pl(ls_r, lP_r),
                     'TPL': fit_tpl(ls_r, lP_r, s_r),
                     'EXP': fit_exp(ls_r, lP_r, s_r)}
            bname = best_of(fits)
            region[rname] = {'fits': fits, 'best': bname,
                             'ls': ls_r, 's': s_r, 'lP': lP_r}

        def exponent(rr):
            if not rr: return np.nan, '?'
            f = rr['fits'].get(rr['best'])
            if f is None: return np.nan, rr['best']
            return (f['tau'] if rr['best'] != 'EXP' else f['lambda']), rr['best']

        tau_s, b_s = exponent(region.get('small'))
        tau_l, b_l = exponent(region.get('large'))
        ds = knee['delta_slope'] if knee else np.nan
        print(f"{p:5.2f}  {xk:6.3f}  {ds:6.3f}  "
              f"{tau_s:8.3f}  {b_s:>6}  {tau_l:8.3f}  {b_l:>6}")

        out[lp] = {'knee': knee, 'xk': xk,
                   'ccdf': (ls, lP, s_u),
                   'region': region,
                   'tau_small': tau_s, 'best_small': b_s,
                   'tau_large': tau_l, 'best_large': b_l}
    return out

print("\n=== B&P analysis (N=20, k=3) ===")
print("--- SNM (PL/TPL/EXP, full CCDF) ---")
snm_results = analyse_bp_snm_full()
print("--- ARM (knee-based) ---")
arm_results = analyse_bp_arm_knee()

# collect panel-F arrays
tau_small_arr  = np.array([arm_results[f'{p:.2f}']['tau_small']
                            if arm_results.get(f'{p:.2f}') else np.nan
                            for p in p_list_bp_all])
tau_large_arr  = np.array([arm_results[f'{p:.2f}']['tau_large']
                            if arm_results.get(f'{p:.2f}') else np.nan
                            for p in p_list_bp_all])
best_large_arr = [arm_results[f'{p:.2f}']['best_large']
                  if arm_results.get(f'{p:.2f}') else 'N/A'
                  for p in p_list_bp_all]

delta_slope_arr = np.array([arm_results[f'{p:.2f}']['knee']['delta_slope']
                            if arm_results.get(f'{p:.2f}') and arm_results[f'{p:.2f}']['knee'] else np.nan
                            for p in p_list_bp_all])

# τ_large where TPL wins on a direction-valid knee (Δslope > 0; Methods).
# Knees from the min-RSS fallback (Δslope ≤ 0) do not define a two-scale split and are excluded.
tpl_large_mask = np.array([b == 'TPL' for b in best_large_arr]) & (delta_slope_arr > 0)
exp_large_mask = np.array([b == 'EXP' for b in best_large_arr])

tau_large_tpl = np.where(tpl_large_mask, tau_large_arr, np.nan)
lam_large_exp = np.where(exp_large_mask, tau_large_arr, np.nan)  # λ stored in tau_large for EXP

valid_small = np.isfinite(tau_small_arr)
if tpl_large_mask.any():
    print(f"\nARM τ_large (TPL, direction-valid knee; p = {p_list_bp_all[tpl_large_mask].tolist()}): "
          f"{np.nanmean(tau_large_tpl):.3f} ± {np.nanstd(tau_large_tpl):.3f}")

# ── Plot settings ─────────────────────────────────────────────────────────────
palette_disp = sns.color_palette('colorblind', n_colors=len(p_list_display))
palette_all  = sns.color_palette('colorblind', n_colors=len(p_list_bp_all))

# Model colours — consistent with Fig.2 and Fig.3
_cb    = sns.color_palette('colorblind')
COL_SNM = _cb[0]   # blue
COL_ARM = _cb[1]   # orange

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(wspace=0.38, hspace=0.45)

# ── Panel A: SNM FFT — scatter only (5 display p values) ─────────────────────
for idx, p in enumerate(p_list_display):
    lp = f'{p:.2f}'
    al = 1.0 if lp == '0.05' else 0.30
    ms = 4   if lp == '0.05' else 3
    axes[0,0].scatter(log_freq, df_fft_n[lp],
                      color=palette_disp[idx], s=ms, alpha=al, label=lp, rasterized=True)
axes[0,0].set_title(r'SNM: amplitude spectrum ($N=20$)', fontsize=12, pad=6)
axes[0,0].legend(title=r'$\varepsilon$', loc='upper right',
                 fontsize=8, title_fontsize=8, framealpha=0.85, markerscale=3)

# ── Panel B: ARM FFT — scatter only (5 display p values) ─────────────────────
beta_mean_arm = np.nanmean(betas_arm)
beta_std_arm  = np.nanstd(betas_arm)
for idx, p in enumerate(p_list_display):
    lp = f'{p:.2f}'
    axes[0,1].scatter(log_freq, df_fft_a[lp],
                      color=palette_disp[idx], s=3, alpha=0.35, label=lp, rasterized=True)
axes[0,1].set_title(r'ARM: amplitude spectrum ($N=20$)', fontsize=12, pad=6)
axes[0,1].legend(title=r'$p$', loc='upper right',
                 fontsize=8, title_fontsize=8, framealpha=0.85, markerscale=3)

for ax in axes[0, :2]:
    ax.set_xlabel(r'Frequency $f$', fontsize=11)
    ax.set_ylabel(r'Amplitude $|\hat{C}(f)|$', fontsize=11)
    ax.xaxis.set_major_formatter(log10_fmt)
    ax.yaxis.set_major_formatter(log10_fmt)
    sns.despine(ax=ax)

# ── Panel C: β vs p (all 10 p values) ────────────────────────────────────────
axes[0,2].plot(p_list_bp_all, betas_snm, color=COL_SNM, lw=2, marker='s', ms=7,
               label='SNM', zorder=2)
axes[0,2].plot(p_list_bp_all, betas_arm, color=COL_ARM, lw=2, marker='o', ms=7,
               label='ARM', zorder=2)
axes[0,2].axhline(beta_mean_arm, color=COL_ARM, lw=1.2, ls='--', alpha=0.5)
axes[0,2].fill_between(p_list_bp_all,
                        beta_mean_arm - np.nanstd(betas_arm),
                        beta_mean_arm + np.nanstd(betas_arm),
                        color=COL_ARM, alpha=0.12)
axes[0,2].text(0.87, 0.40,
               rf'$\bar{{\beta}}_\mathrm{{ARM}}={beta_mean_arm:.2f}\pm{beta_std_arm:.2f}$',
               transform=axes[0,2].transAxes, fontsize=10, va='bottom', ha='right',
               color=COL_ARM,
               bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=3))
axes[0,2].set_xlabel(r'$p$ / $\varepsilon$', fontsize=11)
axes[0,2].set_ylabel(r'$\beta$ (spectral exponent)', fontsize=11)
axes[0,2].set_title(r'Spectral exponent $\beta$', fontsize=12, pad=6)
axes[0,2].legend(fontsize=9, framealpha=0.85)
sns.despine(ax=axes[0,2])

# ── Panel D: SNM CCDF — scatter only ─────────────────────────────────────────
for idx, p in enumerate(p_list_display):
    lp  = f'{p:.2f}'
    res = snm_results.get(lp)
    if not res: continue
    ls, lP, _ = res['ccdf']
    axes[1,0].scatter(ls, lP, facecolors='none', edgecolors=palette_disp[idx],
                      s=16, alpha=0.6, label=lp, zorder=2, linewidths=0.8)

axes[1,0].set_title(r'SNM: avalanche size CCDF ($N=20$)', fontsize=12, pad=6)
axes[1,0].set_xlabel(r'Avalanche size $s$', fontsize=11)
axes[1,0].set_ylabel(r'$P(S \geq s)$', fontsize=11)
axes[1,0].xaxis.set_major_formatter(log10_fmt)
axes[1,0].yaxis.set_major_formatter(log10_fmt)
axes[1,0].legend(title=r'$\varepsilon$', loc='lower left',
                  fontsize=8, title_fontsize=8, framealpha=0.85)
sns.despine(ax=axes[1,0])

# ── Panel E: ARM CCDF — scatter + knee lines ─────────────────────────────────
for idx, p in enumerate(p_list_display):
    lp  = f'{p:.2f}'
    res = arm_results.get(lp)
    if not res: continue
    ls, lP, _ = res['ccdf']
    xk    = res['xk']
    color = palette_disp[idx]
    mask_s = ls <= xk
    mask_l = ls >  xk
    axes[1,1].scatter(ls[mask_s], lP[mask_s], facecolors=color, edgecolors=color,
                      s=12, alpha=0.7, label=lp, zorder=2, linewidths=0.6, rasterized=True)
    axes[1,1].scatter(ls[mask_l], lP[mask_l], facecolors='none', edgecolors=color,
                      s=12, alpha=0.5, zorder=2, linewidths=0.6, rasterized=True)
    axes[1,1].axvline(xk, color=color, lw=0.7, ls=':', alpha=0.5, zorder=1)

axes[1,1].set_title(r'ARM: avalanche size CCDF ($N=20$)', fontsize=12, pad=6)
axes[1,1].set_xlabel(r'Avalanche size $s$', fontsize=11)
axes[1,1].set_ylabel(r'$P(S \geq s)$', fontsize=11)
axes[1,1].xaxis.set_major_formatter(log10_fmt)
axes[1,1].yaxis.set_major_formatter(log10_fmt)
axes[1,1].legend(title=r'$p$', loc='upper right',
                  fontsize=8, title_fontsize=8, framealpha=0.85)
sns.despine(ax=axes[1,1])

# ── Panel F: τ_BP vs p ────────────────────────────────────────────────────────
ax_f = axes[1, 2]

# small-scale exponent (all p)
if valid_small.any():
    ax_f.plot(p_list_bp_all[valid_small], tau_small_arr[valid_small],
              color=COL_ARM, lw=2, marker='o', ms=7, ls='-',
              label=r'ARM small ($\tau_\mathrm{small}$)', zorder=3)

# large-scale TPL exponent (p ≤ 0.65, τ_large ≈ 1.4)
if tpl_large_mask.any():
    ax_f.plot(p_list_bp_all[tpl_large_mask], tau_large_tpl[tpl_large_mask],
              color=COL_ARM, lw=2, marker='s', ms=7, ls='--',
              mfc='white', label=r'ARM large TPL ($\tau_\mathrm{large}$)', zorder=3)
    tau_univ = np.nanmean(tau_large_tpl)
    tau_univ_std = np.nanstd(tau_large_tpl)
    ax_f.axhline(tau_univ, color=COL_ARM, lw=1.0, ls='--', alpha=0.35, zorder=1)
    ax_f.fill_between(p_list_bp_all,
                       tau_univ - tau_univ_std,
                       tau_univ + tau_univ_std,
                       color=COL_ARM, alpha=0.08, zorder=0)

# large-scale EXP (p ≥ 0.75, λ)
if exp_large_mask.any():
    ax_f.plot(p_list_bp_all[exp_large_mask], lam_large_exp[exp_large_mask],
              color=COL_ARM, lw=2, marker='^', ms=8, ls='none', mfc='white',
              label=r'ARM large EXP ($\lambda$)', zorder=3)

# SNM: plot exponents for all 10 p values, marker by best model
SNM_MARKERS = {'PL': 's', 'TPL': 'D', 'EXP': '^'}
snm_p, snm_exp = [], []
for p in p_list_bp_all:
    lp  = f'{p:.2f}'
    res = snm_results.get(lp)
    if not res: continue
    snm_p.append(p)
    snm_exp.append(res['exponent'])
    ax_f.plot(p, res['exponent'], color=COL_SNM, lw=0,
              marker=SNM_MARKERS.get(res['best'], 'o'), ms=7, zorder=2)
if snm_p:
    ax_f.plot(snm_p, snm_exp, color=COL_SNM, lw=2, ls='-', alpha=0.6,
              zorder=1, label='SNM (s=PL, D=TPL, ^=EXP)')

# annotation: universal τ_large
if tpl_large_mask.any():
    ax_f.text(0.57, 0.55,
              rf'$\bar{{\tau}}_{{\rm large}}={tau_univ:.2f}\pm{tau_univ_std:.2f}$',
              transform=ax_f.transAxes, fontsize=10, va='bottom', ha='right',
              color=COL_ARM,
              bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=3))

ax_f.set_xlabel(r'$p$ / $\varepsilon$', fontsize=11)
ax_f.set_ylabel(r'Exponent', fontsize=11)
ax_f.set_title(r'B&P exponents vs $p$', fontsize=12, pad=6)
ax_f.legend(fontsize=9, framealpha=0.85, loc='upper left')
sns.despine(ax=ax_f)

# ── Panel labels ──────────────────────────────────────────────────────────────
for ax, lbl in zip(axes.flat, list('abcdef')):
    ax.text(-0.13, 1.06, lbl, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('../tex/criticality.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('../tex/criticality_new.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: criticality.pdf / criticality_new.png")
print(f"\nSummary:")
print(f"  FFT ARM β  = {beta_mean_arm:.3f} ± {beta_std_arm:.3f}")
if tpl_large_mask.any():
    print(f"  B&P ARM τ_large (universal, TPL) = {tau_univ:.3f} ± {tau_univ_std:.3f}")
