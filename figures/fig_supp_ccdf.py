"""
Supplementary Figures S1 & S2:
  S1: SNM CCDF with PL/TPL/EXP fits (all 10 ε values)
  S2: ARM CCDF with knee + small/large fits (all 10 p values)
Also prints tables for S1/S2/S3.
"""
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats, optimize
import seaborn as sns
import warnings

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

LN10 = np.log(10)

def fmt_log10(val, _pos):
    if val == int(val): return rf'$10^{{{int(val)}}}$'
    return rf'$10^{{{val:.1f}}}$'
log10_fmt = FuncFormatter(fmt_log10)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_pickle('result_for_criticality_20_3.pkl')
N  = 20

p_list = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])

def beggs_plenz_avalanches(series, threshold=1):
    avs, in_b, bsz = [], False, 0
    for v in series:
        if v > threshold: in_b, bsz = True, bsz + v
        else:
            if in_b: avs.append(bsz)
            in_b, bsz = False, 0
    if in_b: avs.append(bsz)
    return avs

def compute_ccdf(all_sizes):
    a = np.array(all_sizes, dtype=float)
    if len(a) < 10: return None, None, None
    u   = np.unique(a)
    cdf = np.array([np.sum(a >= s) for s in u], dtype=float) / len(a)
    ls  = np.log10(u); lP = np.log10(cdf + 1e-12)
    ok  = np.isfinite(lP) & (lP > -6)
    return u[ok], ls[ok], lP[ok]

def aic_ols(n, k, rss):
    return n * np.log(rss/n) + 2*k if (rss > 0 and n > k+1) else np.inf

def _r2(y, yp):
    ss = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y-yp)**2)/ss if ss > 0 else np.nan

def fit_pl(ls, lP):
    if len(ls)<4: return None
    sl,ic,_,_,_ = stats.linregress(ls, lP)
    pr  = sl*ls+ic; rss = np.sum((lP-pr)**2)
    return {'model':'PL','tau':1-sl,'sl':sl,'ic':ic,'R2':_r2(lP,pr),'AIC':aic_ols(len(ls),2,rss)}

def fit_tpl(ls, lP, s_orig):
    if len(ls)<5: return None
    def tpl(X,tau,lsc,c): l,s=X; return -(tau-1)*l - s/(10**lsc*LN10)+c
    sl0,ic0,*_ = stats.linregress(ls,lP)
    sc0 = np.log10(max(np.median(s_orig),2.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt,_ = optimize.curve_fit(tpl,(ls,s_orig),lP,p0=[1-sl0,sc0,ic0],
                                        bounds=([1.0,-1,-20],[30,10,20]),maxfev=20000)
        tau,lsc,c = popt; pr = tpl((ls,s_orig),*popt); rss=np.sum((lP-pr)**2)
        return {'model':'TPL','tau':tau,'s_c':10**lsc,'lsc':lsc,'c':c,
                'R2':_r2(lP,pr),'AIC':aic_ols(len(ls),3,rss),'_popt':popt}
    except: return None

def fit_exp(ls, lP, s_orig):
    if len(ls)<4: return None
    sl,ic,_,_,_ = stats.linregress(s_orig,lP)
    pr=sl*s_orig+ic; rss=np.sum((lP-pr)**2)
    return {'model':'EXP','lambda':-sl*LN10,'sl':sl,'ic':ic,
            'R2':_r2(lP,pr),'AIC':aic_ols(len(ls),2,rss)}

def best_of(fits):
    v={k:f for k,f in fits.items() if f is not None}
    return min(v,key=lambda k: v[k]['AIC']) if v else 'N/A'

def knee_piecewise(ls, lP, min_pts=8, lP_cutoff=-4.0):
    n=len(ls)
    if n<2*min_pts: return None
    body = lP >= lP_cutoff
    best_ds, best_xk, best_pars = -np.inf, None, None
    best_rss_fb, best_xk_fb, best_pars_fb = np.inf, None, None
    for i in range(min_pts, n-min_pts):
        if not body[i]: continue
        xk=ls[i]; ml=ls<=xk; mr=(ls>xk)&body
        if ml.sum()<min_pts or mr.sum()<min_pts: continue
        sl,il,_,_,_ = stats.linregress(ls[ml],lP[ml])
        sr,ir,_,_,_ = stats.linregress(ls[mr],lP[mr])
        rss=(np.sum((lP[ml]-(sl*ls[ml]+il))**2)+np.sum((lP[mr]-(sr*ls[mr]+ir))**2))
        if rss<best_rss_fb: best_rss_fb,best_xk_fb,best_pars_fb=rss,xk,(sl,il,sr,ir)
        ds=sr-sl
        if ds>0 and ds>best_ds: best_ds,best_xk,best_pars=ds,xk,(sl,il,sr,ir)
    if best_xk is None: best_xk,best_pars=best_xk_fb,best_pars_fb
    if best_xk is None: return None
    sl,il,sr,ir=best_pars
    return {'xk':best_xk,'sl':sl,'il':il,'sr':sr,'ir':ir,'tau_l':1-sl,'tau_r':1-sr}

# ── Collect results ───────────────────────────────────────────────────────────
snm_res = {}
arm_res = {}

for p in p_list:
    lp = f'{p:.2f}'
    # SNM
    rows_n = df[(df['label']=='Noise Model') & (abs(df['p_noise']-p)<0.01)]
    avs_n  = [s for _,r in rows_n.iterrows()
               for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
    s_u,ls,lP = compute_ccdf(avs_n)
    if s_u is not None:
        fits = {'PL':fit_pl(ls,lP),'TPL':fit_tpl(ls,lP,s_u),'EXP':fit_exp(ls,lP,s_u)}
        bname = best_of(fits)
        snm_res[lp] = {'fits':fits,'best':bname,'ccdf':(ls,lP,s_u)}
    else:
        snm_res[lp] = None

    # ARM
    rows_a = df[(df['label']=='Async Model') & (abs(df['p_async']-p)<0.01)]
    avs_a  = [s for _,r in rows_a.iterrows()
               for s in beggs_plenz_avalanches(r['history_changed'].astype(float))]
    s_u,ls,lP = compute_ccdf(avs_a)
    if s_u is not None:
        knee   = knee_piecewise(ls,lP)
        xk     = knee['xk'] if knee else np.median(ls)
        ms,ml  = ls<=xk, ls>xk
        reg    = {}
        for rn,msk in [('small',ms),('large',ml)]:
            if msk.sum()<4: reg[rn]=None; continue
            lr,lr2,sr2=ls[msk],lP[msk],s_u[msk]
            fits2={'PL':fit_pl(lr,lr2),'TPL':fit_tpl(lr,lr2,sr2),'EXP':fit_exp(lr,lr2,sr2)}
            bn2=best_of(fits2); reg[rn]={'fits':fits2,'best':bn2,'ls':lr,'lP':lr2,'s':sr2}
        arm_res[lp]={'knee':knee,'xk':xk,'ccdf':(ls,lP,s_u),'region':reg}
    else:
        arm_res[lp]=None

# ── FFT data for Table S3 ─────────────────────────────────────────────────────
step_max = int(df['history_conflict_ratio'].iloc[0].shape[0])
freq_max = step_max // 2
frequency = np.arange(1, freq_max) / step_max
log_freq  = np.log10(frequency)
fft_lo, fft_hi = -3.5, -1.0
fmask = (log_freq >= fft_lo) & (log_freq <= fft_hi)

fft_table = []
for p in p_list:
    lp = f'{p:.2f}'
    rows_n = df[(df['label']=='Noise Model') & (abs(df['p_noise']-p)<0.01)]
    rows_a = df[(df['label']=='Async Model') & (abs(df['p_async']-p)<0.01)]
    pn = np.mean([np.log10(np.abs(np.fft.fft(r['history_conflict_ratio'].astype(float)))[1:freq_max]+1e-12)
                  for _,r in rows_n.iterrows()],axis=0)
    pa = np.mean([np.log10(np.abs(np.fft.fft(r['history_conflict_ratio'].astype(float)))[1:freq_max]+1e-12)
                  for _,r in rows_a.iterrows()],axis=0)
    sln,icn,rn,_,_ = stats.linregress(log_freq[fmask],pn[fmask])
    sla,ica,ra,_,_ = stats.linregress(log_freq[fmask],pa[fmask])
    fft_table.append({'p':p,
                      'SNM_beta':-sln if rn**2>=0.50 else np.nan,
                      'SNM_R2':rn**2,
                      'ARM_beta':-sla if ra**2>=0.90 else np.nan,
                      'ARM_R2':ra**2})

# ── Print Tables ──────────────────────────────────────────────────────────────
print("="*70)
print("TABLE S1: SNM B&P fitting results")
print(f"{'ε':>5}  {'best':>4}  {'τ/λ':>8}  {'PL_AIC':>8}  {'TPL_AIC':>8}  {'EXP_AIC':>8}  {'PL_R2':>6}  {'TPL_R2':>6}  {'EXP_R2':>6}")
print("-"*70)
for p in p_list:
    lp=f'{p:.2f}'; res=snm_res.get(lp)
    if not res: print(f"{p:5.2f}  {'N/A':>4}"); continue
    fits=res['fits']; bname=res['best']
    bf=fits[bname]
    val=bf['tau'] if bname!='EXP' else bf['lambda']
    pl  =fits['PL'];  tpl =fits['TPL'];  exp =fits['EXP']
    print(f"{p:5.2f}  {bname:>4}  {val:8.3f}  "
          f"{pl['AIC'] if pl else np.nan:8.1f}  {tpl['AIC'] if tpl else np.nan:8.1f}  {exp['AIC'] if exp else np.nan:8.1f}  "
          f"{pl['R2'] if pl else np.nan:6.3f}  {tpl['R2'] if tpl else np.nan:6.3f}  {exp['R2'] if exp else np.nan:6.3f}")

print()
print("="*70)
print("TABLE S2: ARM B&P fitting results")
print(f"{'p':>5}  {'xk':>6}  {'best_s':>6}  {'τ_small':>8}  {'sc_small':>8}  {'best_l':>6}  {'τ_large':>8}  {'sc_large':>8}")
print("-"*70)
for p in p_list:
    lp=f'{p:.2f}'; res=arm_res.get(lp)
    if not res: print(f"{p:5.2f}  N/A"); continue
    xk=res['xk']
    def get_exp(r):
        if not r: return np.nan,'N/A',np.nan
        f=r['fits'].get(r['best'])
        if not f: return np.nan,r['best'],np.nan
        tau=f['tau'] if r['best']!='EXP' else f['lambda']
        sc=f.get('s_c',np.nan)
        return tau,r['best'],sc
    ts,bs,scs=get_exp(res['region'].get('small'))
    tl,bl,scl=get_exp(res['region'].get('large'))
    print(f"{p:5.2f}  {xk:6.3f}  {bs:>6}  {ts:8.3f}  {scs:8.1f}  {bl:>6}  {tl:8.3f}  {scl:8.1f}")

print()
print("="*70)
print("TABLE S3: FFT spectral exponent β")
print(f"{'p/ε':>5}  {'SNM_β':>8}  {'SNM_R2':>7}  {'ARM_β':>8}  {'ARM_R2':>7}")
print("-"*42)
for row in fft_table:
    snm_b=f"{row['SNM_beta']:.3f}" if np.isfinite(row['SNM_beta']) else "—"
    arm_b=f"{row['ARM_beta']:.3f}" if np.isfinite(row['ARM_beta']) else "—"
    print(f"{row['p']:5.2f}  {snm_b:>8}  {row['SNM_R2']:7.3f}  {arm_b:>8}  {row['ARM_R2']:7.3f}")

# ── Fig S1: SNM CCDF with fits ────────────────────────────────────────────────
palette = sns.color_palette('colorblind', n_colors=10)
MODEL_COLORS = {'PL':'#4477AA','TPL':'#228833','EXP':'#EE6677'}

fig1, axes1 = plt.subplots(2, 5, figsize=(16, 7), sharey=False)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

for idx, p in enumerate(p_list):
    ax = axes1[idx//5, idx%5]
    lp = f'{p:.2f}'; res = snm_res.get(lp)
    if not res: ax.set_visible(False); continue
    ls,lP,s_u = res['ccdf']
    ax.scatter(ls, lP, s=8, color=palette[idx], alpha=0.5, zorder=1)

    fits=res['ccdf']; bname=res['best']
    for mname, fit in snm_res[lp]['fits'].items():
        if fit is None: continue
        lw   = 2.2 if mname==bname else 1.2
        alpha= 0.9 if mname==bname else 0.5
        ls2  = '-' if mname==bname else '--'
        col  = MODEL_COLORS[mname]
        xs   = np.linspace(ls.min(), ls.max(), 300)
        ss   = 10**xs
        if mname=='PL':
            ys = fit['sl']*xs + fit['ic']
        elif mname=='TPL':
            tau,lsc,c = fit['_popt']
            ys = -(tau-1)*xs - ss/(10**lsc*LN10) + c
        else:  # EXP
            ys = fit['sl']*ss + fit['ic']
        ax.plot(xs, ys, color=col, lw=lw, alpha=alpha, ls=ls2, zorder=2,
                label=f"{mname} (R²={fit['R2']:.2f})")

    ax.set_title(rf'SNM $\varepsilon={p:.2f}$  [{bname}]', fontsize=9)
    ax.xaxis.set_major_formatter(log10_fmt)
    ax.yaxis.set_major_formatter(log10_fmt)
    ax.set_xlabel(r'$s$', fontsize=9); ax.set_ylabel(r'$P(S\geq s)$', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.8)
    sns.despine(ax=ax)

fig1.suptitle('Figure S1: SNM Avalanche CCDF — PL / TPL / EXP fits (best model bold)',
              fontsize=12, y=1.01)
fig1.savefig('../tex/supp_fig_s1_snm_ccdf.pdf', dpi=200, bbox_inches='tight', facecolor='white')
fig1.savefig('../tex/supp_fig_s1_snm_ccdf.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("\nSaved: supp_fig_s1_snm_ccdf.pdf/png")

# ── Fig S2: ARM CCDF with knee + fits ────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 5, figsize=(16, 7), sharey=False)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

for idx, p in enumerate(p_list):
    ax = axes2[idx//5, idx%5]
    lp = f'{p:.2f}'; res = arm_res.get(lp)
    if not res: ax.set_visible(False); continue
    ls,lP,s_u = res['ccdf']
    xk = res['xk']

    ms = ls <= xk; ml = ls > xk
    ax.scatter(ls[ms], lP[ms], s=8, color=palette[idx], alpha=0.6, zorder=1,
               label='small scale')
    ax.scatter(ls[ml], lP[ml], s=8, facecolors='none', edgecolors=palette[idx],
               alpha=0.5, zorder=1, linewidths=0.6, label='large scale')
    ax.axvline(xk, color='gray', lw=1.0, ls=':', alpha=0.7)

    for rname, msk in [('small',ms),('large',ml)]:
        reg = res['region'].get(rname)
        if not reg: continue
        bname2 = reg['best']
        for mname, fit in reg['fits'].items():
            if fit is None: continue
            lw   = 2.0 if mname==bname2 else 1.0
            alpha= 0.9 if mname==bname2 else 0.4
            ls2  = '-' if mname==bname2 else '--'
            col  = MODEL_COLORS[mname]
            xs   = np.linspace(reg['ls'].min(), reg['ls'].max(), 200)
            ss2  = 10**xs
            if mname=='PL':   ys = fit['sl']*xs + fit['ic']
            elif mname=='TPL':
                tau,lsc,c = fit['_popt']
                ys = -(tau-1)*xs - ss2/(10**lsc*LN10) + c
            else: ys = fit['sl']*ss2 + fit['ic']
            ax.plot(xs, ys, color=col, lw=lw, alpha=alpha, ls=ls2, zorder=2)

    ax.set_title(rf'ARM $p={p:.2f}$  [knee={xk:.2f}]', fontsize=9)
    ax.xaxis.set_major_formatter(log10_fmt)
    ax.yaxis.set_major_formatter(log10_fmt)
    ax.set_xlabel(r'$s$', fontsize=9); ax.set_ylabel(r'$P(S\geq s)$', fontsize=9)
    sns.despine(ax=ax)

# shared legend for model colors
from matplotlib.lines import Line2D
leg = [Line2D([0],[0],color=c,lw=2,label=n) for n,c in MODEL_COLORS.items()]
fig2.legend(handles=leg, loc='lower center', ncol=3, fontsize=9,
            frameon=False, bbox_to_anchor=(0.5,-0.02))
fig2.suptitle('Figure S2: ARM Avalanche CCDF — knee detection + region fits (PL/TPL/EXP)',
              fontsize=12, y=1.01)
fig2.savefig('../tex/supp_fig_s2_arm_ccdf.pdf', dpi=200, bbox_inches='tight', facecolor='white')
fig2.savefig('../tex/supp_fig_s2_arm_ccdf.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Saved: supp_fig_s2_arm_ccdf.pdf/png")
