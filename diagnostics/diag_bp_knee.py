"""
Detect knee/inflection point of B&P CCDF (log-log space) automatically.
Two methods:
  1. Piecewise linear regression (breakpoint search) — minimise total RSS
  2. Second derivative of smoothed log-log CCDF — find maximum curvature

ARM, N=20, k=3, all 10 p values.
After knee detection, fit PL / TPL / EXP to each region (same as diag_bp_pthreshold.py)
and compare via AIC.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize, signal
import seaborn as sns
import warnings

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_pickle('result_for_criticality_20_3.pkl')
p_list_all = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
LN10 = np.log(10)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Knee detection ────────────────────────────────────────────────────────────
def knee_piecewise(log10_s, log10_P, min_pts=8, lP_cutoff=-4.0):
    """
    Breakpoint search in log-log space for the PHYSICAL inflection:
    the transition from steep (small scale) to shallower (large scale).

    Constraints:
      1. Candidate breakpoints restricted to points where log10_P >= lP_cutoff
         (avoid finite-size tail).
      2. Among valid candidates, maximise |Δslope| = sr − sl where sr > sl
         (right segment SHALLOWER = less negative slope = smaller τ).
         If no such candidate exists, fall back to minimum-RSS without direction
         constraint within the restricted range.
    """
    n = len(log10_s)
    if n < 2 * min_pts:
        return None

    # restrict candidate pool to body of distribution (not deep tail)
    body = log10_P >= lP_cutoff

    best_rss_body  = np.inf   # fallback: min RSS in body range
    best_xk_body   = None
    best_pars_body = None

    best_ds   = -np.inf       # primary: max Δslope where sr > sl
    best_xk   = None
    best_pars = None

    for i in range(min_pts, n - min_pts):
        if not body[i]:
            continue          # skip deep-tail candidates
        xk = log10_s[i]
        ml = log10_s <= xk
        mr = (log10_s > xk) & body   # right segment also within body

        if ml.sum() < min_pts or mr.sum() < min_pts:
            continue

        sl, il, _, _, _ = stats.linregress(log10_s[ml], log10_P[ml])
        sr, ir, _, _, _ = stats.linregress(log10_s[mr], log10_P[mr])

        rss = (np.sum((log10_P[ml] - (sl * log10_s[ml] + il))**2) +
               np.sum((log10_P[mr] - (sr * log10_s[mr] + ir))**2))

        # fallback: min RSS within body
        if rss < best_rss_body:
            best_rss_body  = rss
            best_xk_body   = xk
            best_pars_body = (sl, il, sr, ir)

        # primary: right shallower (sr > sl) → Δslope = sr − sl > 0
        ds = sr - sl
        if ds > 0 and ds > best_ds:
            best_ds   = ds
            best_xk   = xk
            best_pars = (sl, il, sr, ir)

    # prefer direction-constrained result; fall back to min-RSS in body
    if best_xk is None:
        best_xk, best_pars = best_xk_body, best_pars_body
    if best_xk is None:
        return None

    sl, il, sr, ir = best_pars
    return {'xk': best_xk, 'sl': sl, 'il': il, 'sr': sr, 'ir': ir,
            'tau_l': 1 - sl, 'tau_r': 1 - sr,
            'delta_slope': sr - sl,   # >0: right shallower (physical knee)
            'method': 'piecewise'}

def knee_deriv2(log10_s, log10_P, sg_window=None, sg_poly=3):
    """
    Find knee via maximum absolute 2nd derivative (curvature) of smoothed CCDF.
    Uses Savitzky-Golay filter.
    Returns {'xk': log10_s at max curvature, 'd2': full d2 array}.
    """
    n = len(log10_s)
    if n < 10:
        return None

    # auto window: odd, ~10% of n but ≥ sg_poly+2
    if sg_window is None:
        sg_window = max(sg_poly + 2, int(n * 0.12) | 1)   # force odd
        if sg_window % 2 == 0:
            sg_window += 1
        sg_window = min(sg_window, n - (1 if n % 2 == 0 else 0))
        if sg_window % 2 == 0:
            sg_window -= 1
        sg_window = max(sg_window, sg_poly + 2)

    try:
        smooth = signal.savgol_filter(log10_P, window_length=sg_window, polyorder=sg_poly)
    except ValueError:
        smooth = log10_P

    # numerical derivatives on uniform-ish grid (use np.gradient with spacing)
    dx  = np.gradient(log10_s)
    d1  = np.gradient(smooth, log10_s)
    d2  = np.gradient(d1,     log10_s)

    # maximum |d2| in interior (avoid edges)
    edge = max(2, n // 10)
    interior = np.arange(edge, n - edge)
    idx = interior[np.argmin(d2[interior])]   # most negative = downward bend
    return {'xk': log10_s[idx], 'd2': d2, 'smooth': smooth, 'method': 'deriv2'}

# ── Model fits (same as diag_bp_pthreshold.py) ────────────────────────────────
def aic_ols(n, k, rss):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k

def r2(y, yp):
    ss = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - yp)**2) / ss if ss > 0 else np.nan

def fit_pl(ls, lP):
    n = len(ls)
    if n < 4: return None
    sl, ic, _, _, _ = stats.linregress(ls, lP)
    pr = sl * ls + ic
    rss = np.sum((lP - pr)**2)
    return {'model':'PL', 'tau':1-sl, 'sl':sl, 'ic':ic,
            'R2':r2(lP,pr), 'AIC':aic_ols(n,2,rss)}

def fit_tpl(ls, lP, s_orig):
    n = len(ls)
    if n < 5: return None
    def tpl(X, tau, lsc, c):
        l, s = X
        return -(tau-1)*l - s/(10**lsc * LN10) + c
    sl0, ic0, *_ = stats.linregress(ls, lP)
    sc0 = np.log10(max(np.median(s_orig), 2.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = optimize.curve_fit(tpl, (ls, s_orig), lP,
                                         p0=[1-sl0, sc0, ic0],
                                         bounds=([1.0,-1,-20],[30,10,20]),
                                         maxfev=20000)
        tau, lsc, c = popt
        pr = tpl((ls, s_orig), *popt)
        rss = np.sum((lP - pr)**2)
        return {'model':'TPL', 'tau':tau, 's_c':10**lsc, 'lsc':lsc, 'c':c,
                'R2':r2(lP,pr), 'AIC':aic_ols(n,3,rss), '_popt':popt}
    except Exception:
        return None

def fit_exp(ls, lP, s_orig):
    n = len(ls)
    if n < 4: return None
    sl, ic, _, _, _ = stats.linregress(s_orig, lP)
    pr = sl * s_orig + ic
    rss = np.sum((lP - pr)**2)
    return {'model':'EXP', 'lambda':-sl*LN10, 'sl':sl, 'ic':ic,
            'R2':r2(lP,pr), 'AIC':aic_ols(n,2,rss)}

def best_of(fits):
    v = {k:f for k,f in fits.items() if f is not None}
    return min(v, key=lambda k: v[k]['AIC']) if v else 'N/A'

def plot_fit_curve(ax, fit, ls_range, s_range, color, lw, ls_style):
    if fit is None: return
    m = fit['model']
    if m == 'PL':
        ax.plot(ls_range, fit['sl']*ls_range + fit['ic'],
                color=color, lw=lw, ls=ls_style, alpha=0.92, zorder=5)
    elif m == 'TPL':
        tau, sc, c = fit['tau'], fit['s_c'], fit['c']
        ax.plot(ls_range, -(tau-1)*ls_range - s_range/(sc*LN10) + c,
                color=color, lw=lw, ls=ls_style, alpha=0.92, zorder=5)
    elif m == 'EXP':
        ax.plot(ls_range, fit['sl']*s_range + fit['ic'],
                color=color, lw=lw, ls=ls_style, alpha=0.92, zorder=5)

# ── Main loop ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(19, 8))
plt.subplots_adjust(wspace=0.40, hspace=0.55)
axes_flat = axes.flatten()

# secondary axes for d2 (inset)
COLORS = {
    'small': {'PL':'#2166ac','TPL':'#4dac26','EXP':'#d01c8b'},
    'large': {'PL':'#b2182b','TPL':'#e08214','EXP':'#762a83'},
}
LS_STYLE = {'PL':'-','TPL':'--','EXP':':'}

summary = []

print(f"{'p':>5}  {'xk_pw':>7}  {'xk_d2':>7}  {'Δslope':>7}  "
      f"{'τ_small':>8}  {'τ_large':>8}  "
      f"{'best_s':>5}  {'best_l':>5}")
print('-'*70)

for idx, p in enumerate(p_list_all):
    ax = axes_flat[idx]

    rows = df[(df['label']=='Async Model') & (abs(df['p_async']-p)<0.01)]
    all_avs = [s for _,r in rows.iterrows()
               for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]

    s_u, ls, lP = compute_ccdf(all_avs)
    if s_u is None:
        ax.set_visible(False)
        continue

    # ── knee detection ────────────────────────────────────────────────────────
    kpw = knee_piecewise(ls, lP)
    kd2 = knee_deriv2(ls, lP)

    # choose: prefer piecewise; use d2 as cross-check
    xk = kpw['xk'] if kpw else (kd2['xk'] if kd2 else np.median(ls))

    # ── split at knee ─────────────────────────────────────────────────────────
    mask_s = ls <= xk
    mask_l = ls >  xk

    # ── CCDF scatter ─────────────────────────────────────────────────────────
    ax.scatter(ls[mask_s], lP[mask_s], s=6, alpha=0.5, color='#6baed6', zorder=2, rasterized=True)
    ax.scatter(ls[mask_l], lP[mask_l], s=6, alpha=0.5, color='#fc8d59', zorder=2, rasterized=True)
    ax.axvline(xk, color='black', lw=0.9, ls='--', alpha=0.6, zorder=1)

    # d2 zero-crossing line (optional cross-check)
    if kd2:
        ax.axvline(kd2['xk'], color='gray', lw=0.7, ls=':', alpha=0.5, zorder=1)

    # ── fit each region ───────────────────────────────────────────────────────
    region_results = {}
    for rname, mask in [('small', mask_s), ('large', mask_l)]:
        if mask.sum() < 4:
            region_results[rname] = None
            continue
        ls_r, lP_r, s_r = ls[mask], lP[mask], s_u[mask]
        fits = {'PL': fit_pl(ls_r, lP_r),
                'TPL': fit_tpl(ls_r, lP_r, s_r),
                'EXP': fit_exp(ls_r, lP_r, s_r)}
        bname = best_of(fits)
        region_results[rname] = {'fits': fits, 'best': bname,
                                  'ls': ls_r, 's': s_r, 'lP': lP_r}

        ls_pl = np.linspace(ls_r.min(), ls_r.max(), 200)
        s_pl  = 10**ls_pl
        for mname, fit in fits.items():
            lw = 2.2 if mname == bname else 0.9
            plot_fit_curve(ax, fit, ls_pl, s_pl,
                           COLORS[rname][mname], lw, LS_STYLE[mname])

    # ── annotations ──────────────────────────────────────────────────────────
    def get_exponent(rr):
        """Return best-fit exponent (τ for PL/TPL, λ for EXP) and label."""
        if not rr:
            return np.nan, '?', '?'
        bname = rr['best']
        fit   = rr['fits'].get(bname)
        if fit is None:
            return np.nan, bname, '?'
        if bname == 'EXP':
            return fit['lambda'], bname, 'λ'
        return fit.get('tau', np.nan), bname, 'τ'

    exp_s, b_s, lbl_s = get_exponent(region_results.get('small'))
    exp_l, b_l, lbl_l = get_exponent(region_results.get('large'))
    tau_s = exp_s   # used in summary (may be λ for EXP)
    tau_l = exp_l

    ann = (f"xk={xk:.2f}\n"
           f"small→{b_s} {lbl_s}={exp_s:.2f}\n"
           f"large→{b_l} {lbl_l}={exp_l:.2f}")
    ax.text(0.98, 0.97, ann, transform=ax.transAxes,
            fontsize=7, va='top', ha='right',
            bbox=dict(fc='white', ec='#AAAAAA', alpha=0.85, pad=2))

    ax.set_title(rf'ARM  $p={p:.2f}$', fontsize=10)
    ax.set_xlabel(r'$\log_{10}\ s$', fontsize=8)
    ax.set_ylabel(r'$\log_{10}\ P(S \geq s)$', fontsize=8)
    sns.despine(ax=ax)

    # ── console summary ───────────────────────────────────────────────────────
    xk_pw = kpw['xk'] if kpw else np.nan
    xk_d2 = kd2['xk'] if kd2 else np.nan
    ds    = kpw['delta_slope'] if kpw else np.nan
    print(f"{p:5.2f}  {xk_pw:7.3f}  {xk_d2:7.3f}  {ds:7.3f}  "
          f"{tau_s:8.3f}  {tau_l:8.3f}  {b_s:>5}  {b_l:>5}")

    summary.append({'p':p, 'xk_pw':xk_pw, 'xk_d2':xk_d2,
                    'tau_small':tau_s, 'tau_large':tau_l,
                    'best_small':b_s, 'best_large':b_l})

# ── Global legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Patch(color='#6baed6', alpha=0.7, label='small scale (left of knee)'),
    Patch(color='#fc8d59', alpha=0.7, label='large scale (right of knee)'),
    Line2D([0],[0], color='k',    lw=1.5, ls='--', label='knee (piecewise)'),
    Line2D([0],[0], color='gray', lw=1.0, ls=':',  label='knee (2nd deriv)'),
    Line2D([0],[0], color='gray', lw=2.0, ls='-',  label='PL'),
    Line2D([0],[0], color='gray', lw=2.0, ls='--', label='TPL'),
    Line2D([0],[0], color='gray', lw=2.0, ls=':',  label='EXP'),
]
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=8,
           bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

plt.suptitle(r'B&P CCDF — ARM ($N=20$, $k=3$)  |  knee by piecewise OLS + 2nd derivative'
             '\nThick line = best AIC per region',
             fontsize=10, y=1.01)

plt.savefig('../tex/diag_bp_knee.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved: diag_bp_knee.png')
