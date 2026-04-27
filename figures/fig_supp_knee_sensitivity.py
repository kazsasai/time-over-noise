"""
Supp. sensitivity analysis for the ARM knee detection.

Sweeps two design choices:
  (1) body-range threshold  lP_cutoff ∈ {-3, -4, -5}
  (2) fallback rule         {on, off}

For each (lP_cutoff, fallback) × p, reports:
  - detected knee  xk      (log10 s)
  - left  slope    sl      (=> tau_small = 1 - sl)
  - right slope    sr      (=> tau_large = 1 - sr)
  - Δslope         sr - sl
  - "direction-valid" flag (whether the direction-constrained branch fired)

Outputs:
  - tex/sensitivity_knee.csv   (full grid)
  - tex/sensitivity_knee.tex   (LaTeX-ready table, p × cutoff for tau_large)
  - stdout summary

Data source: same pkl as fig4_criticality.py
  result_for_criticality_20_3.pkl  (N=20, k=3, 1000 trials, 50k steps)

Intended to be executed once, then the resulting .tex is \input{} into
supplementary.tex (or the numbers are transcribed manually).
"""

import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats, optimize

LN10 = np.log(10)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = 'result_for_criticality_20_3.pkl'
OUT_CSV      = '../tex/sensitivity_knee.csv'
OUT_TEX      = '../tex/sensitivity_knee.tex'
P_GRID       = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
CUTOFFS      = [-3.0, -4.0, -5.0]        # body-range thresholds to sweep
FALLBACKS    = [True, False]             # whether to allow min-RSS fallback
MIN_PTS      = 8

# ── Helpers (verbatim from fig4_criticality.py) ──────────────────────────────
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

# ── Knee detection with configurable cutoff and fallback switch ──────────────
def knee_piecewise_configurable(log10_s, log10_P,
                                min_pts=MIN_PTS,
                                lP_cutoff=-4.0,
                                use_fallback=True):
    """
    Same logic as fig4_criticality.knee_piecewise, except:
      - lP_cutoff is a parameter (body range).
      - use_fallback toggles the min-RSS fallback off when no
        direction-valid split is found.
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
        rss = (np.sum((log10_P[ml] - (sl * log10_s[ml] + il)) ** 2) +
               np.sum((log10_P[mr] - (sr * log10_s[mr] + ir)) ** 2))

        if rss < best_rss_fb:
            best_rss_fb, best_xk_fb, best_pars_fb = rss, xk, (sl, il, sr, ir)

        ds = sr - sl
        if ds > 0 and ds > best_ds:
            best_ds, best_xk, best_pars = ds, xk, (sl, il, sr, ir)

    direction_valid = best_xk is not None
    if not direction_valid:
        if use_fallback:
            best_xk, best_pars = best_xk_fb, best_pars_fb
        else:
            return None
    if best_xk is None:
        return None

    sl, il, sr, ir = best_pars
    return {'xk': best_xk, 'sl': sl, 'il': il, 'sr': sr, 'ir': ir,
            'tau_small': 1 - sl, 'tau_large': 1 - sr,
            'delta_slope': sr - sl,
            'direction_valid': direction_valid}

# ── PL / TPL / EXP fitting (verbatim from fig4_criticality.py) ───────────────
def aic_ols(n, k, rss):
    return n * np.log(rss / n) + 2 * k if (rss > 0 and n > k + 1) else np.inf

def _r2(y, yp):
    ss = np.sum((y - np.mean(y)) ** 2)
    return 1 - np.sum((y - yp) ** 2) / ss if ss > 0 else np.nan

def fit_pl(ls, lP):
    n = len(ls)
    if n < 4:
        return None
    sl, ic, *_ = stats.linregress(ls, lP)
    pr = sl * ls + ic
    rss = np.sum((lP - pr) ** 2)
    return {'model': 'PL', 'tau': 1 - sl, 'R2': _r2(lP, pr),
            'AIC': aic_ols(n, 2, rss)}

def fit_tpl(ls, lP, s_orig):
    n = len(ls)
    if n < 5:
        return None
    def tpl(X, tau, lsc, c):
        l, s = X
        return -(tau - 1) * l - s / (10 ** lsc * LN10) + c
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
        rss = np.sum((lP - pr) ** 2)
        return {'model': 'TPL', 'tau': tau, 's_c': 10 ** lsc,
                'R2': _r2(lP, pr), 'AIC': aic_ols(n, 3, rss)}
    except Exception:
        return None

def fit_exp(ls, lP, s_orig):
    n = len(ls)
    if n < 4:
        return None
    sl, ic, *_ = stats.linregress(s_orig, lP)
    pr = sl * s_orig + ic
    rss = np.sum((lP - pr) ** 2)
    return {'model': 'EXP', 'lambda': -sl * LN10, 'R2': _r2(lP, pr),
            'AIC': aic_ols(n, 2, rss)}

def best_of(fits):
    v = {k: f for k, f in fits.items() if f is not None}
    return min(v, key=lambda k: v[k]['AIC']) if v else 'N/A'

def tau_large_best_aic(ls, lP, s_orig):
    """
    Run PL/TPL/EXP on (ls, lP, s_orig) and return:
      tau_large : the τ from the AIC-best model (NaN if EXP wins; EXP has λ, no τ)
      best_name : 'PL' / 'TPL' / 'EXP' / 'N/A'
    Mirrors fig4_criticality.analyse_bp_arm_knee → exponent() right-region path.
    """
    if len(ls) < 4:
        return np.nan, 'N/A'
    fits  = {'PL':  fit_pl(ls, lP),
             'TPL': fit_tpl(ls, lP, s_orig),
             'EXP': fit_exp(ls, lP, s_orig)}
    bname = best_of(fits)
    if bname == 'N/A':
        return np.nan, bname
    bf = fits[bname]
    if bname == 'EXP':
        return np.nan, bname
    return bf['tau'], bname

# ── Load data ────────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} ...")
df = pd.read_pickle(DATA_PATH)
print(f"Loaded. Rows={len(df)}")

# ── Run sweep ────────────────────────────────────────────────────────────────
records = []
print("\n=== Knee sensitivity sweep (ARM) ===")
print("τ_large is the AIC-best (PL/TPL/EXP) exponent re-fit on the "
      "right region of the detected knee, mirroring fig4_criticality.py.")
print(f"\n{'p':>5}  {'cutoff':>7}  {'fb':>3}  "
      f"{'xk':>6}  {'best':>5}  "
      f"{'τ_large':>8}  {'valid':>5}")
print('-' * 56)

for p in P_GRID:
    rows = df[(df['label'] == 'Async Model') & (abs(df['p_async'] - p) < 0.01)]
    all_avs = [s for _, r in rows.iterrows()
               for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
    s_u, ls, lP = compute_ccdf(all_avs)
    if s_u is None:
        continue

    for cutoff in CUTOFFS:
        for fb in FALLBACKS:
            knee = knee_piecewise_configurable(ls, lP,
                                               lP_cutoff=cutoff,
                                               use_fallback=fb)
            if knee is None:
                rec = dict(p=p, cutoff=cutoff, fallback=fb,
                           xk=np.nan,
                           tau_large=np.nan,
                           best_large='N/A',
                           direction_valid=False)
            else:
                xk = knee['xk']
                mask_l = ls > xk
                if mask_l.sum() >= 4:
                    tl, bl = tau_large_best_aic(ls[mask_l], lP[mask_l],
                                                s_u[mask_l])
                else:
                    tl, bl = np.nan, 'N/A'
                rec = dict(p=p, cutoff=cutoff, fallback=fb,
                           xk=xk,
                           tau_large=tl,
                           best_large=bl,
                           direction_valid=knee['direction_valid'])
            records.append(rec)
            print(f"{p:5.2f}  {cutoff:7.1f}  {str(fb)[0]:>3}  "
                  f"{rec['xk']:6.3f}  {rec['best_large']:>5}  "
                  f"{rec['tau_large']:8.3f}  "
                  f"{str(rec['direction_valid'])[0]:>5}")

res = pd.DataFrame.from_records(records)
res.to_csv(OUT_CSV, index=False)
print(f"\nSaved CSV → {OUT_CSV}")

# ── Build LaTeX-ready pivot table (tau_large) ────────────────────────────────
# Rows: p ; Columns: (cutoff, fallback)
pivot = res.pivot_table(index='p',
                        columns=['cutoff', 'fallback'],
                        values='tau_large',
                        dropna=False)
# Mean and spread summary per p over the whole grid
mean_p  = res.groupby('p')['tau_large'].mean()
sd_p    = res.groupby('p')['tau_large'].std()

# Compose LaTeX manually to keep Nature-friendly formatting
lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\caption{\textbf{Sensitivity of the large-scale exponent $\tau_{\rm large}$ to'
             r' knee-detection design choices.} For each $p$ and each (cutoff,~fallback)'
             r' configuration, the direction-constrained piecewise OLS is re-run to detect a'
             r' knee, after which the right (large-scale) region is fit independently with PL,'
             r' TPL, and EXP and the AIC-best model is selected (mirroring the main pipeline'
             r' in \texttt{fig4\_criticality.py}). Each cell reports the resulting'
             r' $\tau_{\rm large}$. ``F=T'' enables the min-RSS fallback when no direction-valid'
             r' split exists; ``F=F'' disables it. Right-most columns report the mean and'
             r' standard deviation of $\tau_{\rm large}$ across all six configurations at each'
             r' $p$. ``---'' denotes either (i) no direction-valid split with fallback disabled,'
             r' or (ii) AIC-best on the right region was EXP, in which case $\tau_{\rm large}$'
             r' is undefined.}')
lines.append(r'\label{tab:knee_sensitivity}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{c' + 'r' * (len(CUTOFFS) * len(FALLBACKS)) + 'rr}')
lines.append(r'\toprule')

# header row
cols_header = ' & '.join(
    [rf'\multicolumn{{2}}{{c}}{{$\log_{{10}}P\ge{c:g}$}}' for c in CUTOFFS]
)
lines.append(r'$p$ & ' + cols_header + r' & mean & s.d.\\')

# sub-header (fallback)
subheader = ' & '.join(['F=T & F=F'] * len(CUTOFFS))
lines.append(r' & ' + subheader + r' & & \\')
lines.append(r'\midrule')

for p in P_GRID:
    cells = []
    for c in CUTOFFS:
        for fb in FALLBACKS:
            if p in pivot.index and (c, fb) in pivot.columns:
                v = pivot.loc[p, (c, fb)]
            else:
                v = np.nan
            cells.append(f'{v:.3f}' if np.isfinite(v) else '---')
    m  = mean_p.loc[p]  if p in mean_p.index else np.nan
    sd = sd_p.loc[p]    if p in sd_p.index   else np.nan
    m_str  = f'{m:.3f}'  if np.isfinite(m)  else '---'
    sd_str = f'{sd:.3f}' if np.isfinite(sd) else '---'
    lines.append(f'{p:.2f} & ' + ' & '.join(cells) + f' & {m_str} & {sd_str}\\\\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

with open(OUT_TEX, 'w') as f:
    f.write('\n'.join(lines))
print(f"Saved LaTeX table → {OUT_TEX}")

# ── Short robustness summary printed to stdout ───────────────────────────────
plateau   = res[(res['p'] >= 0.05) & (res['p'] <= 0.65)]
finite_pl = plateau.dropna(subset=['tau_large'])
print("\n── Robustness summary (p ∈ [0.05, 0.65] plateau, AIC-best τ) ──")
print(f"  Configurations total   = {len(plateau)}")
print(f"  Finite-τ configurations = {len(finite_pl)} "
      f"({len(finite_pl)/len(plateau):.1%})")
if len(finite_pl) > 0:
    print(f"  τ_large  mean  = {finite_pl['tau_large'].mean():.3f}")
    print(f"           s.d.  = {finite_pl['tau_large'].std():.3f}")
    print(f"           range = [{finite_pl['tau_large'].min():.3f}, "
          f"{finite_pl['tau_large'].max():.3f}]")
print(f"  direction_valid fraction = {plateau['direction_valid'].mean():.2%}")
print(f"  best_large breakdown     = "
      + ", ".join(f"{k}:{v}" for k, v in
                  plateau['best_large'].value_counts().items()))

# Per-p s.d. (used for the supplementary narrative)
print("\n── Per-p spread (configurations actually returning τ) ──")
print(f"{'p':>5}  {'n_cfg':>5}  {'mean':>6}  {'s.d.':>6}  {'min':>6}  {'max':>6}")
for p, grp in res.groupby('p'):
    fg = grp.dropna(subset=['tau_large'])
    if len(fg) == 0:
        print(f"{p:5.2f}  {0:5d}  ---   ---     ---     ---")
        continue
    print(f"{p:5.2f}  {len(fg):5d}  "
          f"{fg['tau_large'].mean():6.3f}  "
          f"{fg['tau_large'].std() if len(fg)>1 else float('nan'):6.3f}  "
          f"{fg['tau_large'].min():6.3f}  "
          f"{fg['tau_large'].max():6.3f}")
