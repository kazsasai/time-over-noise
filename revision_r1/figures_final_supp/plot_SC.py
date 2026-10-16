"""S-C: definitions, record selection, clock, and the two-sweep limiting case."""
from figure_common import *

def load(v):
    return rows(read(f'aval/planC_{v}/planC.json'))

def compare(ax, variants, key, log=False):
    for v, label, col, sty in variants:
        rr = load(v)
        values=np.array(triples(rr,key),float)
        if log:
            values[values[:,0]<=0,:]=np.nan
            values[values<=0]=np.nan
        band(ax, [float(r['p']) for r in rr], values, col, label,
             linestyle=sty, marker='o', markersize=2.5)
    parameter(ax)
    if log:
        ax.set_yscale('log'); ax.set_ylabel('Mean excess size (color changes)')
    else:
        fraction(ax); ax.set_ylabel('Resolved share of large avalanches')

def main():
    fig, axs = plt.subplots(5, 2, figsize=(7.8, 12.8), layout='constrained')
    thresholds = [('sweep','Threshold 1 (primary)',ARM,'-'),
                  ('A3_sweep_th0_k1','Threshold 0','#7A4E00','--'),
                  ('A3_sweep_th2_k1','Threshold 2',PURPLE,'-.'),
                  ('A3_sweep_th3_k1','Threshold 3',CLASS['resolve'],':')]
    records = [('sweep','Primary',ARM,'-'), ('A3_sweep_nobound','No boundaries',GREY,'--'),
               ('A3_sweep_stat','After first solution',PURPLE,':'),
               ('A3_sweep_T10000','10,000 rounds','#7A4E00','-.'),
               ('A3_sweep_T25000','25,000 rounds',CLASS['resolve'],'--')]
    clocks = [('sweep','Sweep clock',ARM,'-'), ('primary','Step clock',GREY,'--'),
              ('A3_sweep_th1_k2','Two-sweep bins',PURPLE,':')]
    for i, (variants, title) in enumerate([(thresholds,'Activity threshold'),(records,'Record selection'),(clocks,'Clock and bin size')]):
        for j, key in enumerate(['resolve_share_large', 'mean_excess']):
            ax=axs[i,j]; panel(ax, chr(97+2*i+j))
            compare(ax, variants, key, log=j==1)
            if i==0 and j==1:
                ax.set_ylim(5, 5e3)
                ax.legend(handles=axs[0,0].get_lines(), fontsize=6.7, loc='lower left', ncol=2)
            elif i>0 and j==0: ax.legend(fontsize=6.7, loc='lower left', ncol=2 if i==1 else 1)
    for j, key in enumerate(['sweep','sweep_th1_k2']):
        ax=axs[3,j]; panel(ax, chr(103+j), 'One-sweep bins' if j==0 else 'Two-sweep bins')
        s,n=pooled('ARM',.45,key+'|all'); ccdf(ax,s,n,color=INK,label='All',lw=.9)
        for c in CLASS:
            sc,nc=pooled('ARM',.45,key+'|'+c)
            ccdf(ax,sc,nc,denominator=n.sum(),color=CLASS[c],label=c.capitalize())
        r=load('sweep' if j==0 else 'A3_sweep_th1_k2')[4]
        ax.axvline(r['s_star'],color=GREY,ls='--',lw=.8)
        ax.set(xscale='log',yscale='log',ylim=(1e-7,1.2),xlim=(2,50000),
               xlabel='Size $s$ (color changes)',ylabel=r'Fraction of all avalanches $\geq s$')
        if j==0: ax.legend(loc='upper right',fontsize=6.7)
    for j,p in enumerate([.05,.15]):
        ax=axs[4,j];panel(ax,chr(105+j),rf'SNM $\epsilon$ = {p:.2f}')
        total=pooled('SNM',p)[1].sum()
        for key,sty in [('sweep','-'),('th1_dt1','--')]:
            for c,col in [('all',SNM),('solved',CLASS['solved'])]:
                s,n=pooled('SNM',p,key+'|'+c)
                ccdf(ax,s,n,denominator=total,color=col,ls=sty,
                     label=('Sweep' if key=='sweep' else 'Step')+', '+c)
        ax.set(xscale='log',yscale='log',ylim=(1e-7,1.2),
               xlabel='Size $s$ (color changes)',ylabel=r'Fraction of all avalanches $\geq s$')
        if j==0:ax.legend(fontsize=6.5)
    finish(fig,'C')

if __name__=='__main__':
    main()
