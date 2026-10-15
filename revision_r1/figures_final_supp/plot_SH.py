"""S-H: healing, matched large responses, and stationary activity regressions."""
from figure_common import *

def main():
    fig,axs=plt.subplots(4,3,figsize=(10.1,11.4),layout='constrained')
    growth=read('kick/kick_growth.json');large=read('kick/kick_vs_natural_large.json')
    branch=read('branching/branching.json');classes=read('branching/branching_by_class.json')
    conditions=[('ARM',.05,SHADES[0]),('ARM',.45,ARM),('ARM',.85,SHADES[2]),('SNM',.05,SNM)]
    ax=axs[0,0];panel(ax,'a')
    for model,p,col in conditions:
        with np.load(DATA/f'kick/kick_raw_{model}_{p:.2f}.npz') as z:
            a=z['kicks'];t=a[a[:,0]>0,1];s,n=np.unique(t,return_counts=True)
        ccdf(ax,s,n,color=col,label=f'ARM $p$ = {p:.2f}' if model=='ARM' else r'SNM $\epsilon$ = 0.05')
    ax.set(xscale='log',yscale='log',xlabel='Healing time (sweeps)',ylabel=r'Completed responses $\geq t$',ylim=(1e-4,1.1))
    ax.legend(fontsize=6.4)
    ax=axs[0,1];panel(ax,'b')
    for tau,col,sty in [(1,SHADES[0],'s-'),(5,ARM,'o-'),(20,SHADES[2],'^-'),(50,GREY,'d--')]:
        y=[growth[f'ARM_{p:.2f}']['survival'][growth[f'ARM_{p:.2f}']['taus'].index(tau)] for p in P]
        ax.plot(P,y,sty,color=col,label=rf'$\tau$ = {tau} sweep'+('s' if tau>1 else ''))
    fraction(ax);parameter(ax);ax.set_ylabel(r'Fraction inconsistent at $\tau$');ax.legend(fontsize=6.5)
    ax=axs[0,2];panel(ax,'c')
    qq=[];completion=[]
    for p in P:
        with np.load(DATA/f'kick/kick_raw_ARM_{p:.2f}.npz') as z:a=z['kicks']
        qq.append(np.quantile(a[a[:,0]>0,1],[.1,.5,.9]));completion.append(np.mean(a[:,0]>0))
    qq=np.array(qq);ax.fill_between(P,qq[:,0],qq[:,2],color=ARM,alpha=.15,label='10th-90th percentiles')
    ax.plot(P,qq[:,1],'o-',color=ARM,label='Median completed healing time')
    ax.set_yscale('log');parameter(ax);ax.set_ylabel('Healing time (sweeps)');ax.legend(fontsize=6.3)
    for j,kind in enumerate(['size','dur']):
        ax=axs[1,j];panel(ax,chr(100+j))
        for origin,col,sty in [('kick',ARM,'o-'),('natural',GREY,'s--')]:
            ax.plot(P,[large[f'{p:.2f}'][f'{origin}_large_median_{kind}'] for p in P],sty,color=col,label='Injected' if origin=='kick' else 'Natural')
        parameter(ax);ax.set_yscale('log');ax.set_ylabel('Median size (colour changes)' if kind=='size' else 'Median duration (sweeps)');ax.legend()
    ax=axs[1,2];panel(ax,'f')
    for model,p,col in conditions[:3]:
        with np.load(DATA/f'kick/kick_raw_ARM_{p:.2f}.npz') as z:
            a=z['kicks'];k=a[(a[:,0]>0)&(a[:,2]>=100),2];n=z['natural_size'];n=n[n>=100]
        for vals,sty,label in [(k,'-',f'$p$ = {p:.2f}, injected'),(n,'--',f'$p$ = {p:.2f}, natural')]:
            s,c=np.unique(vals,return_counts=True);ccdf(ax,s,c,color=col,ls=sty,label=label)
    ax.set(xscale='log',yscale='log',xlabel='Size $s$ (colour changes)',ylabel=r'Conditional fraction $\geq s$',ylim=(1e-3,1.1))
    ax.legend(fontsize=6,ncol=1)
    ax=axs[2,0];panel(ax,'g')
    for model,col in [('ARM',ARM),('SNM',SNM)]:
        rr=[branch[f'{model}_{p:.2f}']['stationary'] for p in P]
        ci_line(ax,P,[r['m'] for r in rr],[r['m_ci'] for r in rr],col,label=model,fmt='o-')
    ax.axhline(1,color=GREY,lw=.8,ls='--');ax.set_ylim(0,1.08);ax.set_ylabel('Estimated activity ratio $m$')
    mixed_parameter(ax);ax.legend()
    pp=np.array([float(k) for k in classes]);cr=list(classes.values())
    ax=axs[2,1];panel(ax,'h')
    for key,label,col,sty in [('never_leave','Never leave',GREY,'s--'),('leaving','With excursions',ARM,'o-')]:
        ax.plot(pp,[r[key][0] for r in cr],sty,color=col,label=label)
    ax.axhline(1,color=GREY,lw=.8,ls=':');ax.set_ylabel('Estimated activity ratio $m$');parameter(ax);ax.set_ylim(0,1.08);ax.legend()
    ax=axs[2,2];panel(ax,'i')
    for key,label,col,sty in [('never_leave','Never leave',GREY,'s--'),('leaving','With excursions',ARM,'o-')]:
        ax.plot(pp,[r[key][2] for r in cr],sty,color=col,label=label)
    ax.set_yscale('log');ax.set_ylabel('Regression timescale (sweeps)');parameter(ax);ax.legend()
    ax=axs[3,0];panel(ax,'j')                      # moved from main Fig. 5f
    inj=read('graph_types/graph_types.json')['injection']   # pooled over injections, 95% interval from bootstrap over graphs
    rows=[inj[f'{p:.2f}']['all_cascade'] for p in P]
    ci_line(ax,P,[r[0] for r in rows],[[r[1],r[2]] for r in rows],ARM,label='ARM',fmt='o-')
    parameter(ax);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_ylabel('Fraction of injected responses')
    axs[3,1].axis('off');axs[3,2].axis('off')
    finish(fig,'H')

if __name__=='__main__':
    main()
