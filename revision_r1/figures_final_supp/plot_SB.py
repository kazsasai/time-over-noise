"""S-B: stored likelihood comparisons and local shape diagnostics."""
from figure_common import *

def main():
    fig, axs = plt.subplots(4, 3, figsize=(10.3, 11.8), layout='constrained')
    mle = read('aval/mle_sweep/point.json')
    same = read('aval/hump_same_region.json')
    sens = read('aval/hump_lowercut_sensitivity.json')
    pp = np.array([float(k) for k in same if k != 'joint'])
    sr = [same[f'{p:.2f}'] for p in pp]
    ax = axs[0, 0]; panel(ax, 'a')
    for p, col in zip([.05, .45, .85], SHADES):
        s, n = pooled('ARM', p)
        x, y, counts = pmf(s, n)
        keep = (counts >= 100) & (y > 0)
        x, y = x[keep], y[keep]
        exponent = -np.gradient(np.log(y), np.log(x))
        ax.semilogx(x, exponent, 'o-', color=col, label=f'$p$ = {p:.2f}')
    ax.set(xlabel='Size $s$ (color changes)', ylabel=r'$-\mathrm{d}\log P(s)/\mathrm{d}\log s$', ylim=(-.5, 7))
    ax.legend(loc='upper left')
    ax = axs[0, 1]; panel(ax, 'b')
    colors = plt.get_cmap('copper')(np.linspace(.15, .85, 10))
    for p, col in zip(P, colors):
        s, n = pooled('ARM', p, 'sweep|solved')
        x, y, c = pmf(s, n, np.unique(np.r_[1.5,2.5,3.5,4.5,np.geomspace(4.5, 500, 30)]))
        keep = c >= 30
        ax.loglog(x[keep], y[keep], color=col, alpha=.85, lw=1, label=f'{p:.2f}')
    ax.set(xlabel='Size $s$ (color changes)', ylabel='Conditional PMF', xlim=(2, 500), ylim=(1e-6, 1))
    ax.legend(title='$p$', ncol=2, fontsize=6, title_fontsize=7)
    ax = axs[0, 2]; panel(ax, 'c')
    ax.plot(pp, [r['tau_OLS_paper'] for r in sr], 's--', color=GREY, label=r'CCDF least squares, $\tau$')
    ci_line(ax, pp, [r['TPL_alpha'] for r in sr], [r['TPL_alpha_ci'] for r in sr], ARM,
            label=r'TPL maximum likelihood, $\alpha$', fmt='o-')
    ax.set_ylabel('Reported exponent'); parameter(ax); ax.set_xlim(0, .7); ax.legend(fontsize=6.5)
    for ax, letter, model, region in [(axs[1,0],'d','ARM','full'), (axs[1,1],'e','ARM','clauset'),
                                     (axs[2,0],'g','SNM','full'), (axs[2,1],'h','SNM','clauset')]:
        panel(ax, letter, f'{model}, '+('full range' if region=='full' else 'KS-selected tail'))
        im = aic_matrix(ax, rows(mle, model), region)
    # One compact colour key applies to all four likelihood matrices.
    cb = fig.colorbar(im, ax=[axs[1,0],axs[1,1],axs[2,0],axs[2,1]], fraction=.025, pad=.02)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.set_label(r'$\Delta$AIC (color floor = 1)'); cb.set_ticks([1, 1e2, 1e4, 1e6])
    ax = axs[1, 2]; panel(ax, 'f')
    for fam, col, sty in [('PL', GREY, 's--'), ('TPL', ARM, 'o-'), ('LN', PURPLE, '^:')]:
        ax.plot(pp, [r['KS'][fam] for r in sr], sty, color=col, label=fam)
    ax.set_ylabel('KS distance'); ax.set_ylim(0, .16); parameter(ax); ax.set_xlim(0,.7); ax.legend(ncol=3)
    ax = axs[2, 2]; panel(ax, 'i')
    ci_line(ax, pp, [r['TPL_cutoff'] for r in sr], [r['TPL_cutoff_ci'] for r in sr], ARM, fmt='o-')
    ax.set_yscale('log'); ax.set_ylabel('TPL cut-off (color changes)'); parameter(ax); ax.set_xlim(0,.7)
    ax = axs[3, 0]; panel(ax, 'j')
    for p, col in zip([.05,.25,.45], SHADES):
        r = sens[f'{p:.2f}']['lower_cut']
        ax.plot([v['factor'] for v in r], [v['alpha'] for v in r], 'o-', color=col, label=f'$p$ = {p:.2f}')
    ax.set(xlabel='Lower cut-off / CCDF knee', ylabel=r'TPL exponent $\alpha$', xticks=[1,1.5,2,3])
    ax.legend()
    ax = axs[3, 1]; panel(ax, 'k')
    r = sens['0.45']['local_exponent']; ax.semilogx(r['size'], r['exponent'], 'o-', color=ARM)
    ax.set(xlabel='Size $s$ (color changes; step clock)', ylabel='Local exponent', ylim=(0,5.5))
    ax = axs[3, 2]; panel(ax, 'l')
    for m, col in [('ARM', ARM), ('SNM', SNM)]:
        rr = rows(mle, m)
        ax.plot(P, [r['fits']['full'][r['fits']['full']['best_aic']]['ks'] for r in rr], 'o-', color=col, label=m)
    ax.set_ylabel('KS distance of AIC-selected fit'); parameter(ax); ax.legend()
    finish(fig, 'B')

if __name__ == '__main__':
    main()
