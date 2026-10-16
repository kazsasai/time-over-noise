"""S-A: class contributions at all ARM parameters and SNM record dependence."""
from figure_common import *
from matplotlib.lines import Line2D

def main():
    fig = plt.figure(figsize=(10, 8.1), layout='constrained')
    fig.get_layout_engine().set(rect=(0, 0, 1, .95))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15])
    rr = rows(read('aval/planC_sweep/planC.json'))
    for i, (p, r) in enumerate(zip(P, rr)):
        ax = fig.add_subplot(gs[i//5, i%5])
        panel(ax, chr(97+i), f'$p$ = {p:.2f}')
        s, n = pooled('ARM', p)
        ccdf(ax, s, n, color=INK, linewidth=.9)
        for c in CLASS:
            sc, nc = pooled('ARM', p, 'sweep|'+c)
            ccdf(ax, sc, nc, denominator=n.sum(), color=CLASS[c])
        ax.axvline(r['s_star'], color=GREY, linestyle='--', linewidth=.8)
        ax.set(xscale='log', yscale='log', xlim=(2, 50000), ylim=(8e-8, 1.2))
        ax.set_xticks([10, 1000])
        ax.set_yticks([1, 1e-3, 1e-6])
        if i%5 == 0:
            ax.set_ylabel(r'Fraction of all avalanches $\geq s$')
        if i >= 5:
            ax.set_xlabel('Size $s$ (color changes)')
    bottom = gs[2, :].subgridspec(1, 2)
    ax = fig.add_subplot(bottom[0])
    panel(ax, 'k')
    for m, col, marker in [('ARM', ARM, 'o'), ('SNM', SNM, 's')]:
        rs = rows(read('aval/planC_sweep/planC.json'), m)
        a = np.array(triples(rs, 'frac_solved'), float)
        nz = a[:, 0] > 0
        lo = np.where(a[:, 1] > 0, a[:, 1], 1e-7)
        b = np.c_[a[:, 0], lo, a[:, 2]]
        band(ax, P[nz], b[nz], col, m, marker=marker)
        if (~nz).any():
            # Exact zeros are not drawn on the line; they are marked on the floor.
            ax.plot(P[~nz], np.full((~nz).sum(), 1.6e-7), 'v', color=col, mfc='white',
                    mew=.9, ms=4, clip_on=False, label=f'{m}: none observed')
    ax.set_yscale('log'); ax.set_ylim(1e-7, 1.4)
    ax.set_ylabel('Solved-class fraction'); mixed_parameter(ax)
    ax.legend(loc='center right')
    ax = fig.add_subplot(bottom[1])
    panel(ax, 'l')
    for key, T, col in [('sweep_T10000|all', 10000, '#88BEDA'),
                        ('sweep_T25000|all', 25000, '#3A95C3'),
                        ('sweep|all', 50000, SNM)]:
        y = [quantile_hist(*pooled('SNM', p, key)) for p in P]
        ax.plot(P, y, 'o-', color=col, label=f'{T:,} rounds')
    ax.set_yscale('log'); ax.set_ylabel('Median size (color changes)')
    parameter(ax, 'SNM'); ax.legend(loc='lower right')
    handles = [Line2D([], [], color=INK, label='All')]
    handles += [Line2D([], [], color=CLASS[c], label=c.capitalize()) for c in CLASS]
    handles += [Line2D([], [], color=GREY, ls='--', label='$s^*$')]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(.5, 1))
    finish(fig, 'A')

if __name__ == '__main__':
    main()
