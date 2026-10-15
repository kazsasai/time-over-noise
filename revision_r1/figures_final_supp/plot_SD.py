"""S-D: conflict spectra, trial heterogeneity, renewal surrogates and fit windows."""
from figure_common import *

def main():
    fig,axs=plt.subplots(3,3,figsize=(10.1,8.6),layout='constrained')
    z=np.load(DATA/'spec/spectrum_curves.npz')
    spec=read('spec/spectra.json'); supp=read('spec/spectra_supplement.json')
    pp=np.array([float(k) for k in supp]); ss=list(supp.values())
    condition='ARM_0.45__'; f=z[condition+'full_f_cycles_per_step']
    ax=axs[0,0]; panel(ax,'a')
    for key,label,col,sty in [('all','All trials',ARM,'-'),('never_leave','Never leave after solution',GREY,'--'),('other','Trials with excursions',PURPLE,'-')]:
        spectrum(ax,f,z[condition+'full_geomean_amp_'+key],color=col,ls=sty,label=label)
    ax.axvspan(10**-3.5,.1,color=GREY,alpha=.08)
    ax.set(xlabel='Frequency (cycles per round)',ylabel='Geometric mean amplitude',xlim=(1e-4,.5))
    ax.legend(fontsize=6.5)
    ax=axs[0,1]; panel(ax,'b')
    for key,label,col,sty in [('fig4_full','All trials',ARM,'o-'),('fig4_full_never_leave','Never leave',GREY,'s--'),('fig4_full_other','With excursions',PURPLE,'^-')]:
        ax.plot(P,[spec[f'ARM_{p:.2f}'][key][0] for p in P],sty,color=col,label=label)
    ax.set_ylabel(r'Amplitude exponent $\beta$'); parameter(ax);ax.legend(fontsize=6.5)
    f=z[condition+'win_f_cycles_per_sweep']
    ax=axs[0,2];panel(ax,'c')
    x,lo=smooth_spectrum(f,z[condition+'win_normalised_power_trial_p05'])
    _,hi=smooth_spectrum(f,z[condition+'win_normalised_power_trial_p95'])
    ax.fill_between(x,lo,hi,color=GREY,alpha=.17,label='5th-95th percentiles')
    for key,label,col in [('median','Median trial',INK),('mean','Mean across trials',ARM)]:
        spectrum(ax,f,z[condition+'win_normalised_power_trial_'+key],color=col,label=label)
    ax.axhline(1,color=GREY,ls=':',lw=.8)
    ax.set(xlabel='Frequency (cycles per sweep)',ylabel='Power / trial mean power',xlim=(1e-4,.8),ylim=(1e-3,1e3))
    ax.legend(fontsize=6.4)
    ax=axs[1,0];panel(ax,'d')
    for key,label,col,sty in [('win_mean_power_real','Observed',ARM,'-'),
                              ('win_mean_power_renewal_surrogate','Renewal surrogate',INK,'--'),
                              ('win_mean_power_without_top10pct_trials','Below 90th variance percentile',PURPLE,':')]:
        spectrum(ax,f,z[condition+key],color=col,ls=sty,label=label)
    ax.set(xlabel='Frequency (cycles per sweep)',ylabel='Mean power',xlim=(1e-4,.8))
    ax.legend(fontsize=6.4)
    ax=axs[1,1];panel(ax,'e')
    ax.plot(pp,[r['beta_meanpower'] for r in ss],'o-',color=ARM,label='Exponent of mean power')
    med=np.array([r['per_trial_beta_median_iqr'] for r in ss])
    ax.errorbar(pp,med[:,0],yerr=np.array([med[:,0]-med[:,1],med[:,2]-med[:,0]]),fmt='s-',color=INK,capsize=3,label='Trial median and IQR')
    ax.axhline(0,color=GREY,ls=':',lw=.8); parameter(ax);ax.set_ylabel(r'Amplitude exponent $\beta$');ax.legend(fontsize=6.5)
    ax=axs[1,2];panel(ax,'f')
    ax.plot(pp,[r['power_share_top5pct_trials'] for r in ss],'o-',color=ARM)
    ax.axhline(.5,color=GREY,ls='--',lw=.8)
    ax.set_ylim(.4,.62);ax.set_yticks([.4,.45,.5,.55,.6]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_ylabel('Power share of top 5% of nonconstant trials');parameter(ax)
    ax=axs[2,0];panel(ax,'g')
    for key,label,col,sty in [('beta_meanpower','All nonconstant trials',ARM,'o-'),
                             ('beta_meanpower_below_q95','Below 95th percentile','#7A4E00','s--'),
                             ('beta_meanpower_below_q90','Below 90th percentile',PURPLE,'^-'),
                             ('beta_meanpower_below_q50','Below median',GREY,'d:')]:
        ax.plot(pp,[r[key] for r in ss],sty,color=col,label=label)
    ax.set_ylabel(r'Amplitude exponent $\beta$');parameter(ax);ax.set_ylim(-.03,.95);ax.legend(fontsize=6.3,ncol=2,loc='upper center')
    ax=axs[2,1];panel(ax,'h')
    windows=list(spec['ARM_0.45']['win_power_sweep'])
    for w,col,sty in zip(windows,[ARM,GREY,PURPLE,CLASS['resolve']],['o-','s--','^-','d:']):
        ax.plot(P,[spec[f'ARM_{p:.2f}']['win_power_sweep'][w]['beta'] for p in P],sty,color=col,label=w)
    ax.set_ylabel(r'Amplitude exponent $\beta$');parameter(ax);ax.set_ylim(.1,.85);ax.legend(title=r'$\log_{10} f$ bounds',title_fontsize=7,fontsize=6.4,ncol=2,loc='lower left')
    ax=axs[2,2];panel(ax,'i')
    w='[-3.0,-1.0]'
    for key,label,col,sty in [('X_real_sweep','Observed',ARM,'o-'),('X_shuffled_sweep','Renewal surrogate',INK,'s--'),('X_telegraph_sweep','Exponential spells',GREY,'^:')]:
        ax.plot(P,[spec[f'ARM_{p:.2f}'][key][w]['beta'] for p in P],sty,color=col,label=label)
    ax.set_ylabel(r'Indicator amplitude exponent $\beta$');parameter(ax);ax.legend(fontsize=6.3)
    z.close();finish(fig,'D')

if __name__=='__main__':
    main()
