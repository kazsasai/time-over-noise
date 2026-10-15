"""S-F: round/sweep success budgets and supplied operating-window scans."""
from figure_common import *

def main():
    fig,axs=plt.subplots(3,2,figsize=(7.8,8.4),layout='constrained')
    data=read('fpt/first_passage.json')
    for j,(model,base,cols) in enumerate([('SNM',SNM,['#88BEDA','#3A95C3',SNM,'#004A73','#002C44']),
                                        ('ARM',ARM,['#F7D49E','#F0B65E',ARM,'#9D6500','#573800'])]):
        ax=axs[0,j];panel(ax,chr(97+j))
        for grid,H,col,sty in [('fine',200,cols[0],'-'),('fine',500,cols[1],'-'),('fine',1000,cols[2],'-'),('long',5000,cols[3],'--'),('long',50000,cols[4],':')]:
            rr=rows(data['rows'][grid],model)
            band(ax,[r['p'] for r in rr],[r[f'succ_r{H}'] for r in rr],col,f'{H:,}',linestyle=sty)
        ax.axhline(.9,color=GREY,lw=.8,ls='--');fraction(ax);parameter(ax,model);ax.set_ylabel('Success probability')
        if model=='SNM':ax.set_xlim(0,.45);ax.set_xticks([0,.1,.2,.3,.4])
        ax.legend(title='Rounds',title_fontsize=7,ncol=2,fontsize=6.5,
                  **({'loc':'lower left','bbox_to_anchor':(.03,.02)} if model=='SNM' else {'loc':'lower center'}))
        ax=axs[1,j];panel(ax,chr(99+j))
        selections=[('fine',200),('fine',500),('fine',1000),('long',5000),('long',50000)]
        for i,(grid,H) in enumerate(selections):
            r=data['scans'][f'{model}_{grid}_r{H}']
            if np.isfinite(r['win_lo']):
                ax.plot([r['win_lo'],r['win_hi']],[i,i],color=base,lw=5,alpha=.35,solid_capstyle='butt')
                ax.plot([r['win_lo'],r['win_hi']],[i,i],'|',color=base,ms=9)
                ax.text(r['win_hi']+.015,i,f"{r['width']:.2f}",fontsize=7,va='center')
            else:
                ax.text(.62,i,'No range reaches 90%',fontsize=7,va='center',color=GREY)
            ax.errorbar(r['argmax'],i+.13,xerr=[[r['argmax']-r['argmax_ci'][0]],[r['argmax_ci'][1]-r['argmax']]],fmt='o',color=base,capsize=2,ms=3)
        ax.set_yticks(range(5),[f'{H:,}' for _,H in selections]);ax.set_ylabel('Round budget');parameter(ax,model)
        ax.set_xlim(0,.5 if model=='SNM' else 1.06);ax.invert_yaxis();ax.grid(False)
        ax=axs[2,j];panel(ax,chr(101+j))
        for H,col,sty in [(50,cols[0],'-'),(100,cols[1],'-'),(200,cols[2],'-'),(500,cols[3],'--')]:
            rr=rows(data['rows']['long'],model)
            rr=[r for r in rr if f'succ_s{H}' in r and np.isfinite(r[f'succ_s{H}'][0])]
            band(ax,[r['p'] for r in rr],[r[f'succ_s{H}'] for r in rr],col,f'{H:,}',linestyle=sty,marker='o')
        ax.axhline(.9,color=GREY,lw=.8,ls='--');fraction(ax);parameter(ax,model);ax.set_ylabel('Success probability')
        if model=='SNM':ax.set_xlim(0,.45);ax.set_xticks([0,.1,.2,.3,.4])
        ax.legend(title='Sweeps',title_fontsize=7,ncol=2,fontsize=6.5,**({'loc':'center right','bbox_to_anchor':(1,.55)} if model=='SNM' else {'loc':'lower right'}))
    finish(fig,'F')

if __name__=='__main__':
    main()
