"""S-E: site-wise colour spectra under the matched short-record protocol."""
from figure_common import *

def main():
    fig,axs=plt.subplots(2,3,figsize=(10,5.8),layout='constrained')
    data=read('agentspec/agent_spectra.json');z=np.load(DATA/'agentspec/agent_spectra_curves.npz')
    for j,(window,title) in enumerate([('a_step','Random start'),('b_step','After first solution')]):
        ax=axs[0,j];panel(ax,chr(97+j),title)
        for m,col,sty in [('ARM',ARM,'o-'),('SNM',SNM,'s--')]:
            keys=sorted([k for k in data if k.startswith(m+'_')],key=lambda k:float(k.split('_')[1]))
            ax.plot([float(k.split('_')[1]) for k in keys],[data[k][window]['alpha_of_mean_spectrum'] for k in keys],sty,color=col,label=m)
        ax.set(ylabel=r'Power-spectrum slope $\alpha$',ylim=(0,1.9));mixed_parameter(ax)
        ax.legend()
    ax=axs[0,2];panel(ax,'c')
    for w,label,col,sty in [('a_step','Random start, steps',ARM,'o-'),('a_sweep','Random start, sweeps',ARM,'o--'),
                            ('b_step','Stationary, steps',GREY,'s-'),('b_sweep','Stationary, sweeps',GREY,'s--')]:
        ax.plot(P,[data[f'ARM_{p:.2f}'][w]['alpha_of_mean_spectrum'] for p in P],sty,color=col,label=label)
    parameter(ax);ax.set_ylabel(r'Power-spectrum slope $\alpha$');ax.legend(fontsize=6.5)
    f=z['f_index']
    for j,(cond,title) in enumerate([('ARM_0.45','ARM $p$ = 0.45'),('SNM_0.05',r'SNM $\epsilon$ = 0.05'),('SNM_0.15',r'SNM $\epsilon$ = 0.15')]):
        ax=axs[1,j];panel(ax,chr(100+j),title)
        col=ARM if cond.startswith('ARM') else SNM
        for w,label,sty,alpha in [('a_step','Random start','-',1),('b_step','Stationary','--',.65)]:
            y=z[cond+'__'+w+'__mean_S']
            ax.loglog(f,y,color=col,ls=sty,alpha=alpha,label=label)
        # Stored slope; only its vertical placement is computed for display.
        y=z[cond+'__a_step__mean_S'];a=data[cond]['a_step']['alpha_of_mean_spectrum']
        mask=f<=100; intercept=np.mean(np.log(y[mask])+a*np.log(f[mask]))
        ax.loglog(f[mask],np.exp(intercept)*f[mask]**(-a),':',color=INK,lw=1,label=rf'Protocol fit: $\alpha$ = {a:.2f}')
        ax.axvspan(1,100,color=GREY,alpha=.07)
        ax.set(xlabel='DFT frequency index $f$',ylabel='Mean site-summed power $S_f$',xlim=(1,512))
        ax.legend(fontsize=6.5)
    z.close();finish(fig,'E')

if __name__=='__main__':
    main()
