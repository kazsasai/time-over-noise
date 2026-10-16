"""Figure 4: random reselection source and retention after first solution."""
from figure_common import *

def main():
    robust=read('robust/robustness.json')['rows']
    long=read('fpt/first_passage.json')['rows']['long']
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.6))
    fig.subplots_adjust(left=.085,right=.975,top=.94,bottom=.10,hspace=.67,wspace=.55)
    ax=axes[0,0]; panel(ax,'a')
    for model,c in [('SNM',SNM),('ARM',ARM)]:
        rr=rows(robust,model); x=np.array([r['rand_per_sweep_search'] for r in rr]); triples=np.array([r['succ'] for r in rr])
        ax.errorbar(x,triples[:,0],yerr=[triples[:,0]-triples[:,1],triples[:,2]-triples[:,0]],
                    fmt='o',ms=2.3,alpha=.65,color=c,elinewidth=.4,label=model)
        if model=='ARM':
            high=np.array([r['p']>=.93 for r in rr])
            ax.scatter(x[high],triples[high,0],facecolors='white',edgecolors=ARM,s=17,lw=.8,zorder=4)
    ax.set_xscale('symlog',linthresh=1,linscale=.7); ax.set_xlim(-.05,21)
    log_ticks(ax,[0,1,2,4,10,20]); fraction(ax)
    ax.set_xlabel('Random reselections / sweep\n(before first solution)'); ax.set_ylabel('Solved within 1,000 rounds')
    ax.legend(loc='lower left',handlelength=1)
    d=at(long,'ARM',.95)
    note(ax,f"p ≥ 0.93:\nopen circles\n\np = 0.95:\n{d['succ_s200'][0]:.1%} within\n200 sweeps",xy=(.02,.36))

    for j,model in enumerate(['SNM','ARM']):
        ax=axes[0,j+1]; c=SNM if model=='SNM' else ARM
        panel(ax,'bc'[j])
        rr=rows(robust,model); x=[r['p'] for r in rr]
        ax.plot(x,[r['rand_per_sweep_search'] for r in rr],color=c,alpha=.5,lw=2.4,label='Before first solution')
        ax.plot(x,[r['rand_per_sweep_stationary'] for r in rr],color=c,ls='--',label='After first solution')
        parameter(ax,r'Behavioral noise $\varepsilon$' if model=='SNM' else r'Freezing probability $p$')
        ax.set_ylabel('Random reselections / sweep'); ax.set_ylim(-.15,21 if model=='SNM' else 4.6)
        ax.legend(loc='upper left' if model=='SNM' else 'center right',fontsize=6.3,handlelength=1.5)
        if model=='SNM': note(ax,'Before = after\n'+r'$20\varepsilon$ throughout',xy=(.06,.49))
        else: note(ax,'Gridlock-driven;\nnearly stops after solving',xy=(.06,.08))

    ax=axes[1,0]; panel(ax,'d')
    for model,c in [('SNM',SNM),('ARM',ARM)]:
        rr=rows(long,model)
        # Pooled estimate has no matching CI. Do not mix trial-mean CIs into it.
        ax.plot([r['p'] for r in rr],[r['time_frac_solved_pooled'] for r in rr],
                marker='o',color=c,label=model)
    parameter(ax,r'$\varepsilon$ (SNM) / $p$ (ARM)'); fraction(ax)
    ax.set_ylabel('Consistent time / time after solving')
    ax.legend(loc='center right'); note(ax,'Pooled over trials;\n50,000-round records',xy=(.04,.37))

    ax=axes[1,1]; panel(ax,'e')
    thresholds=[1,10,100,1000]
    for model,p,c in [('SNM',.05,SNM),('ARM',.05,ARM_SHADES[0]),('ARM',.45,ARM_SHADES[1]),('ARM',.95,ARM_SHADES[3])]:
        r=at(long,model,p)
        ax.plot(thresholds,[r[f'P_first_solved_run_ge_{t}sw'] for t in thresholds],
                color=c,marker='o',label=f'{model} '+(r'$\varepsilon$' if model=='SNM' else 'p')+f' = {p:.2f}')
    ax.set_xscale('log'); ax.set_xlim(.8,1400); log_ticks(ax,thresholds); fraction(ax)
    ax.set_xlabel('Spell lasts at least (sweeps)'); ax.set_ylabel('Fraction of first spells')
    ax.legend(loc='lower right',bbox_to_anchor=(1,.06),fontsize=6.2,handlelength=1.4)

    ax=axes[1,2]; panel(ax,'f')
    for model,c in [('SNM',SNM),('ARM',ARM)]:
        rr=rows(long,model); n=np.array([r['n_trials_solved'] for r in rr]); k=np.rint(n*np.array([r['share_never_leave'] for r in rr]))
        errors(ax,[r['p'] for r in rr],wilson(k,n),c,marker='o',label=model)
    parameter(ax,r'$\varepsilon$ (SNM) / $p$ (ARM)'); fraction(ax)
    ax.set_ylabel('Fraction never observed leaving'); ax.legend(loc='upper right')
    note(ax,'ARM: 45.4-48.6%\nheld until 50,000 rounds',xy=(.04,.15))
    finish(fig,4,'a-c: 1,000-round records; d-f: 50,000-round records. a, f: 95% Wilson intervals; b-e: point estimates.\n'
           'a uses a symmetric-log axis (linear from 0 to 1). e joins four supplied duration thresholds; terminal spells are censored.')

if __name__=='__main__':
    main()
