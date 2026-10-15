"""Figure 3: budgets, first-passage survival, and the timing/rule factorial."""
from figure_common import *

def main():
    fpt = read('fpt/first_passage.json')['rows']
    survival = read('fpt/survival_at_optima.json')
    tests = read('fpt/mannwhitney.json')
    factorial = read('fpt/factorial_2x2.json')
    fig = plt.figure(figsize=(7.2, 5.35))
    gs = fig.add_gridspec(2, 6, left=.09, right=.98, top=.94, bottom=.09,
                          hspace=.64, wspace=1.12)
    for j, model in enumerate(['SNM','ARM']):
        ax = fig.add_subplot(gs[0,j*3:(j+1)*3])
        panel(ax, 'ab'[j])
        rr = rows(fpt['fine'],model)
        ribbon(ax,[r['p'] for r in rr],[r['succ_r1000'] for r in rr],
               SNM if model=='SNM' else ARM, label='1,000 rounds')
        for h,c,m,ls in [(5000,PURPLE,'o','--'),(50000,INK,'s',':')]:
            rr = rows(fpt['long'],model)
            errors(ax,[r['p'] for r in rr],[r[f'succ_r{h}'] for r in rr],
                   c,label=f'{h:,} rounds',marker=m,ls=ls,markersize=3)
        fraction(ax); parameter(ax, r'Behavioural noise $\varepsilon$' if model=='SNM' else r'Freezing probability $p$')
        ax.set_ylabel('Fraction reaching a solution')
        ax.legend(loc='center right' if model=='SNM' else 'lower center')

    for j, unit in enumerate(['sweeps','rounds']):
        ax = fig.add_subplot(gs[1,j*2:(j+1)*2])
        panel(ax,'cd'[j])
        for model,p,c in [('SNM',.05,SNM),('ARM',.45,ARM)]:
            d = survival[f'fine_{model}_{p:.2f}']
            scale = d['rate'] if unit=='sweeps' else 1
            times, counts = np.unique(np.array(d['T'])*scale,return_counts=True)
            horizon = 1000*scale
            x = np.r_[.5*scale,times,horizon]
            remain = np.r_[d['n'],d['n']-np.cumsum(counts),d['n']-len(d['T'])]
            ci = wilson(remain,d['n'])
            ax.step(x,ci[:,0],where='post',color=c,label=model)
            ax.fill_between(x,ci[:,1],ci[:,2],step='post',color=c,alpha=.13,lw=0)
            if remain[-1]:
                ax.plot(horizon,ci[-1,0],marker='+',color=c,ms=7,mew=1.3)
        ax.set_xscale('log'); ax.set_xlim(.8,1150); log_ticks(ax,[1,10,100,1000])
        fraction(ax); ax.set_ylabel('Fraction not yet solved'); ax.set_xlabel(f'Time to first solution ({unit})')
        ax.axhline(.5, color=GREY,linewidth=.6,ls=':')
        # medians and test over all trials of the 50,000-round records (every trial solved); the medians equal those of
        # the solved trials of the 1,000-round scans drawn here
        t=tests[f'long_SNM_0.05|long_ARM_0.45|{unit}']; tf=tests[f'fine_SNM_0.05|fine_ARM_0.45|{unit}']
        assert (t['med_x'], t['med_y']) == (tf['med_x'], tf['med_y']) and t['nx'] == t['ny'] == 1000
        if unit=='sweeps':
            txt=f"Medians {t['med_x']:.1f} vs {t['med_y']:.1f}\nMann-Whitney p = {t['p']:.3f}"
        else:
            txt=f"Medians {t['med_x']:.0f} vs {t['med_y']:.0f} rounds"
        note(ax,txt,xy=(.04,.04)); ax.legend(loc='upper right',handlelength=1.3)   # test over all trials of the 50,000-round records (legend)

    ax = fig.add_subplot(gs[1,4:6]); panel(ax,'e')
    order=['Sync+Stay (SNM, eps=0)','Async+Stay','Sync+Random','Async+Random (ARM)']
    labels=['Sync\nstay','Async\nstay','Sync\nrandom','Async\nrandom']
    cols=[SNM,'#8C8C8C','#BDBDBD',ARM]
    for i,k in enumerate(order):
        d=factorial[k]
        ax.bar(i,d['rate'],width=.62,color=cols[i],edgecolor='none')
        ax.errorbar(i,d['rate'],yerr=[[d['rate']-d['ci'][0]],[d['ci'][1]-d['rate']]],color=INK,capsize=2.5,elinewidth=.8,lw=0)
        ax.text(i,d['ci'][1]+.03,f"{d['rate']:.1%}",ha='center',va='bottom',fontsize=6.8,color=INK)
    ax.set_xticks(range(4),labels); ax.tick_params(axis='x',length=0,labelsize=6.8)
    ax.set_xlim(-.6,3.6); fraction(ax); ax.set_ylim(0,1.12)
    ax.set_ylabel('Solved within 1,000 rounds')
    finish(fig,3,'N = 20, k = 3, m = 2; 1,000 trials per condition. Shading / bars: 95% Wilson intervals.\n'
           'c-d: SNM '+r'$\varepsilon=0.05$'+', ARM p = 0.45; 3 ARM trials censored at 1,000 rounds (+). Sweeps = rounds '+r'$\times(1-p)$'+' for ARM.')

if __name__=='__main__':
    main()
