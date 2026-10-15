"""S-G: first-solution times and avalanches starting before the first solution."""
from figure_common import *

def fit_matrix(ax,data,key):
    pp=list(data); a=np.array([[data[p][key]['dAIC'][f] for f in FAMILIES] for p in pp])
    im=ax.pcolormesh(np.arange(6)-.5,np.arange(len(pp)+1)-.5,np.maximum(1,a),
                     norm=LogNorm(1,1e4),cmap='YlOrBr',shading='flat',rasterized=False)
    ax.set_xlim(-.5,4.5);ax.set_ylim(len(pp)-.5,-.5)
    ax.grid(False)
    for i,p in enumerate(pp):
        for j,f in enumerate(FAMILIES):
            d=a[i,j];label=f'{d:.1f}' if d<10 else f'{d:.0f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if d>500 else INK,fontweight='bold' if d==0 else 'normal')
    ax.set_xticks(range(5),FAMILIES);ax.set_yticks(range(len(pp)),pp);ax.set_ylabel('$p$')
    return im

def main():
    fig,axs=plt.subplots(4,2,figsize=(7.8,10.1),layout='constrained')
    data=read('pre/pre_solution.json'); pp=[float(k) for k in data]
    colors=['#7A4E00','#AA6D00',ARM,'#ECAF40','#F0C987']
    z=np.load(DATA/'fpt/solve_times_long.npz')
    for j,clock in enumerate(['rounds','sweeps']):
        ax=axs[0,j];panel(ax,chr(97+j))
        for model,p,col in [('ARM',p,c) for p,c in zip(pp,colors)]+[('SNM',.05,SNM)]:
            key=f'{model}_{p:.2f}';solved=z[key+'__solved'].astype(bool)
            t=z[key+'__solve_time_step'][solved]+1
            # All these selected conditions solved; retain denominator for censoring correctness.
            t=t.astype(float)*(1-p if clock=='sweeps' and model=='ARM' else 1)
            s,n=np.unique(t,return_counts=True)
            survival=1-np.cumsum(n)/len(solved)
            keep=survival>0
            ax.step(s[keep],survival[keep],where='post',color=col,label=f'ARM {p:.2f}' if model=='ARM' else r'SNM $\epsilon$ = 0.05')
        ax.set(xscale='log',yscale='log',xlabel='Time to first solution ('+clock+')',ylabel='Fraction still unsolved',ylim=(8e-4,1.1))
        if j==0:ax.legend(title='Model / $p$',title_fontsize=7,ncol=2,fontsize=6.5)
    ax=axs[1,0];panel(ax,'c')
    for p,col in zip(pp,colors):
        s,n=subtract_hist(pooled('ARM',p),pooled('ARM',p,'sweep_stat|all'))
        ccdf(ax,s,n,color=col,label=f'$p$ = {p:.2f}')
    ax.set(xscale='log',yscale='log',xlabel='Size $s$ (colour changes)',ylabel=r'Conditional fraction $\geq s$',ylim=(1e-3,1.1))
    ax.legend(fontsize=6.5)
    ax=axs[1,1];panel(ax,'d')
    ax.plot(pp,[data[f'{p:.2f}']['pre_aval_per_trial'] for p in pp],'o-',color=ARM)
    parameter(ax);ax.set_ylabel('Mean number per trial');ax.set_ylim(0,3)
    for ax,letter,key,title in [(axs[2,0],'e','T_fit_xmin1','First-solution time, full range'),
                               (axs[2,1],'f','T_fit_KS','First-solution time, selected tail'),
                               (axs[3,0],'g','pre_fit_xmin2','Pre-solution size, full range'),
                               (axs[3,1],'h','pre_fit_KS','Pre-solution size, selected tail')]:
        panel(ax,letter,title); im=fit_matrix(ax,data,key)
    cb=fig.colorbar(im,ax=axs[2:,:].ravel().tolist(),fraction=.024,pad=.025)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.set_label(r'$\Delta$AIC (colour floor = 1)');cb.set_ticks([1,100,10000])
    z.close();finish(fig,'G')

if __name__=='__main__':
    main()
