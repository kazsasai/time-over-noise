"""Figure 5: natural excursions and responses to an injected inconsistency."""
from figure_common import *

def main():
    summary=rows(read('aval/planC_sweep/planC.json')['summary'],'ARM')
    episodes=read('aval/planC_sweep/episodes.json')
    growth=read('kick/kick_growth.json')
    ps=np.array([float(r['p']) for r in summary])
    fig,axes=plt.subplots(3,2,figsize=(7.2,8.0))
    fig.subplots_adjust(left=.105,right=.975,top=.955,bottom=.06,hspace=.65,wspace=.40)
    # a, b: the two populations analysed separately, each normalised within itself (R1-3)
    pops=[('a','Solved at every bin end',['solved']),('b','Unsolved at some bin end',['resolve','open'])]
    for j,(letter,title,keys) in enumerate(pops):
        ax=axes[0,j]; panel(ax,letter,title)
        for p,col in zip([.05,.45,.85],ARM_SHADES[:3]):
            z=np.load(DATA/f'aval/hist_ARM_{p:.2f}.npz')
            arr=np.concatenate([z['sweep|'+k] for k in keys],axis=1)
            u,inv=np.unique(arr[1],return_inverse=True); c=np.bincount(inv,weights=arr[2])
            y=(c.sum()-np.r_[0,np.cumsum(c)[:-1]])/c.sum()          # fraction of this population with size >= s
            ax.step(u,y,where='post',color=col,lw=1.3,label=f'p = {p:.2f}')
            ax.axvline(at(summary,'ARM',p)['s_star'],color=col,ls='--',lw=.8)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1.5,30000); ax.set_ylim(1e-6,1.5)
        log_ticks(ax,[10,100,1000,10000]); ax.set_xlabel('Avalanche size $s$ (colour changes)')
        ax.set_ylabel(r'Fraction of population $\geq s$')
    axes[0,0].legend(loc='lower left',fontsize=6.5)

    ax=axes[1,0]; panel(ax,'c')
    errors(ax,ps,[episodes[f'{p:.2f}']['nontrivial_via_large'] for p in ps],ARM,
           label='Non-trivial episodes resolved\nthrough a large avalanche',marker='o',ls='-')
    errors(ax,ps,[[r['resolve_share_large'],r['resolve_share_large_lo'],r['resolve_share_large_hi']] for r in summary],
           CLASS['resolve'],label='Large excursions that\nreach a solution',marker='s',ls='--')
    parameter(ax,r'Freezing probability $p$'); fraction(ax); ax.set_ylabel('Fraction')
    ax.legend(loc='lower left',fontsize=6.6,handlelength=1.5)
    note(ax,'No quiet resolution observed',xy=(.30,.94),va='top')

    ax=axes[1,1]; panel(ax,'d')
    triples=[[r[k] for k in ['mean_excess','mean_excess_lo','mean_excess_hi']] for r in summary]
    # mean_excess = mean(s - s*) over resolve + open avalanches with s >= s* (checked against the raw counts)
    for r,tri in zip(summary,triples):
        arr=np.concatenate([np.load(DATA/f"aval/hist_ARM_{r['p']}.npz")['sweep|'+k] for k in ['resolve','open']],axis=1)
        ix=arr[1]>=r['s_star']
        assert np.isclose(np.average(arr[1,ix]-r['s_star'],weights=arr[2,ix]),tri[0])
    errors(ax,ps,triples,ARM,marker='o',ls='-')
    ax.set_yscale('log'); ax.set_ylim(80,3000); ax.set_yticks([100,300,1000,3000]); ax.yaxis.set_major_formatter(ScalarFormatter())
    parameter(ax,r'Freezing probability $p$'); ax.set_ylabel('Mean excess (colour changes)')
    note(ax,'Resolve + open classes;\nno functional fit',xy=(.04,.06))

    ax=axes[2,0]; panel(ax,'e')
    datasets=[]
    for model,p,c in [('SNM',.05,SNM),('ARM',.05,ARM_SHADES[0]),('ARM',.45,ARM_SHADES[1]),('ARM',.85,ARM_SHADES[2])]:
        arr=np.load(DATA/f'kick/kick_raw_{model}_{p:.2f}.npz')['kicks']
        vals=arr[arr[:,0]==1,2]
        datasets.append((model,p,c,vals))
    edges=log_edges(max(v.max() for _,_,_,v in datasets),bins_per_decade=7)
    for model,p,c,v in datasets:
        y=density(v,np.ones(len(v)),edges,len(v))
        ax.stairs(y,edges,color=c,lw=1.6 if model=='SNM' else 1.2,
                  label=f'{model} '+(r'$\varepsilon$' if model=='SNM' else 'p')+f' = {p:.2f}')
    ax.set_xscale('log'); ax.set_xlim(.7,150000); ax.set_ylim(0,1.35)
    log_ticks(ax,[1,10,100,1000,10000,100000]); ax.tick_params(axis='x',labelsize=6.2)
    ax.set_xlabel('Response size (colour changes)'); ax.set_ylabel('Probability density per decade')
    ax.legend(loc='upper right',fontsize=6.5,handlelength=1.4)
    note(ax,'Completed responses only',xy=(.04,.93),va='top')

    ax=axes[2,1]; panel(ax,'f')
    ax.axhline(0,color=INK,lw=.7,ls='--')
    for tau,c,marker in [(.5,ARM_SHADES[0],'o'),(2,ARM_SHADES[1],'s'),(5,ARM_SHADES[2],'^'),(10,GREY,'D')]:
        triples=[]
        for p in ps:
            d=growth[f'ARM_{p:.2f}']; i=d['taus'].index(tau)
            triples.append([d[k][i] for k in ['Lambda','Lambda_lo','Lambda_hi']])
        errors(ax,ps,triples,c,marker=marker,ls='-',label=rf'$\tau={tau:g}$ sweeps',ms=3)
    parameter(ax,r'Freezing probability $p$'); ax.set_ylabel(r'Growth rate $\Lambda(\tau)$ (sweep$^{-1}$)')
    ax.set_ylim(-.53,.83); ax.legend(loc='upper right',fontsize=6.6,handlelength=1.3)
    note(ax,'Short-time crossover\nnear p = 0.45-0.53',xy=(.04,.07))

    finish(fig,5,'a-d: sweep-clock avalanches; a, b: each population normalised within itself; dashed, s*. c, d, f: 95% trial-bootstrap intervals.\n'
           'e: completed responses. f: ARM; healed responses count as zero. The cascade probability is Supplementary Fig. S7j.')

if __name__=='__main__':
    main()
