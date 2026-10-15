"""Supplementary Figs. S3 (hybrid), S10 (plateau edge), S11 (first solution against N) and S12 (network ensembles),
regenerated for rev1.  Data: ../analyses/runs/supp/{hybrid,plateau,networks,colourability}.json,
../analyses/runs/fig6/fig6_grids.json and ../analyses/runs/fpt_size/fpt_size.json (written by
analyses/run_supp_extra.py, run_colourability.py, fig6_numbers.py and run_fpt_size.py)."""
from figure_common import *


def save(fig, name):
    fig.savefig(ROOT / f'fig_{name}.pdf', bbox_inches='tight', pad_inches=.05,
                metadata={'Title': name, 'Creator': 'plot_extra.py (Matplotlib)', 'CreationDate': None})
    fig.savefig(ROOT / f'fig_{name}.png', dpi=300, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def hybrid():
    h = read('supp/hybrid.json')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), layout='constrained')
    for ax, key, col, letter, title, xl in [
            (axs[0], 'SNM_eps0.05_vary_p', SNM, 'a', r'SNM rule, $\varepsilon = 0.05$, with freezing $p$', 'Freezing probability $p$'),
            (axs[1], 'ARM_p0.45_vary_eps', ARM, 'b', r'ARM rule, $p = 0.45$, with noise $\varepsilon$', r'Behavioural noise $\varepsilon$')]:
        panel(ax, letter)
        rr = h[key]
        band(ax, [r['value'] for r in rr], [r['solved'] for r in rr], col, marker='o')
        fraction(ax); ax.set_xlim(-.02, 1.0); ax.set_xticks([0, .25, .5, .75])
        ax.set_xlabel(xl); ax.set_ylabel('Solved within 1,000 rounds')
    save(fig, 'hybrid')


def plateau():
    g = read('fig6/fig6_grids.json')['plateau']['ARM_k4']
    pl = read('supp/plateau.json')
    Ns = np.array(g['N']); ps = np.array(g['param'])
    rate = np.array(g['solved']) / np.array(g['trials'])
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), layout='constrained')
    ax = axs[0]; panel(ax, 'a')
    from matplotlib.colors import LinearSegmentedColormap
    heat = LinearSegmentedColormap.from_list('solved', ['#FFFFFF', '#C9D3DA', '#7E909C', '#2F3E46'])
    im = ax.imshow(rate, origin='lower', aspect='auto', cmap=heat, vmin=0, vmax=1,
                   extent=[ps[0] - .025, ps[-1] + .025, -.5, len(Ns) - .5])
    edge = [pl['edge'][str(n)] for n in Ns]
    ax.plot(edge, range(len(Ns)), 'o-', color=ARM, ms=3, lw=1, label='90% edge')
    ax.set_yticks(range(len(Ns)), Ns); ax.set_ylabel('Network size $N$'); ax.set_xlabel('Freezing probability $p$')
    ax.set_xlim(0, .6); ax.grid(False); ax.legend(loc='lower right', fontsize=6.5, frameon=True)
    ax = axs[1]; panel(ax, 'b')
    styles = {0.9: ('o', SHADES[0]), 0.95: ('s', SHADES[1]), 0.99: ('^', SHADES[2])}
    for s in pl['sensitivity']:
        if s['N_min'] != 60:
            continue
        mk, col = styles[s['threshold']]
        x = np.array([int(n) for n in s['edge']]); y = 1 - np.array([s['edge'][str(n)] for n in x])
        ax.loglog(x, y, mk, color=col, ms=3.5, label=f"{int(100 * s['threshold'])}%: exponent {s['exponent']:.2f}")
        m = x >= 60
        b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
        xx = np.array([55, 110]); ax.loglog(xx, np.exp(a) * xx ** b, '-', color=col, lw=.8)
    y100 = 1 - pl['edge']['100']
    xx = np.array([50, 110]); ax.loglog(xx, y100 * (xx / 100) ** -0.5, ':', color=GREY, lw=1.2, label=r'$\propto N^{-1/2}$')
    ax.set_xlabel('Network size $N$'); ax.set_ylabel(r'$1 - p_{\min}$'); ax.legend(fontsize=5.8, loc='lower left')
    ax.set_xticks([20, 50, 100], ['20', '50', '100']); ax.minorticks_off()
    ax.set_yticks([0.7, 0.8, 0.9, 1.0], ['0.7', '0.8', '0.9', '1.0']); ax.set_ylim(0.66, 1.1)
    ax = axs[2]; panel(ax, 'c')
    cols = plt.cm.viridis(np.linspace(0, .9, len(Ns)))
    for n, c, row in zip(Ns, cols, rate):
        ax.plot((1 - ps) * np.sqrt(n), row, color=c, lw=1, label=f'{n}')
    ax.set_xlabel(r'$(1-p)\sqrt{N}$'); ax.set_ylabel('Fraction solved'); fraction(ax); ax.set_xlim(0, 10.5)
    ax.legend(title='$N$', fontsize=5.5, title_fontsize=6, ncol=2, loc='center left', bbox_to_anchor=(0.02, 0.42))
    save(fig, 'plateau')


def fpt_size():
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    d = read('fpt_size/fpt_size.json'); budget = d['budget']; rep = d['representative']
    cells = d['cells']; Ns = sorted({c['N'] for c in cells})
    cols = plt.cm.viridis(np.linspace(0, .9, len(Ns)))                  # as in main Fig. 6a-d
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.9), layout='constrained')
    for ax, model, letter, title in [(axs[0, 0], 'SNM', 'a', r'SNM, $\varepsilon = 0.05$'), (axs[0, 1], 'ARM', 'b', r'ARM, $p = 0.45$')]:
        panel(ax, letter, title)
        for n, c in zip(Ns, cols):
            t, s = d['survival'][f'{model}_k4_N{n}']
            ax.step(np.maximum(t, 1), s, where='post', color=c, lw=1, label=f'{n}')
        ax.set_xscale('log'); ax.set_xlim(1, budget); fraction(ax)
        ax.set_xlabel('Rounds'); ax.set_ylabel('Fraction not yet solved')
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cax = axs[0, 1].inset_axes([0.05, 0.08, 0.035, 0.5])               # empty lower-left corner of panel b
    bounds = np.arange(len(Ns) + 1) - .5
    cbn = fig.colorbar(plt.cm.ScalarMappable(BoundaryNorm(bounds, len(Ns)), ListedColormap(cols)), cax=cax, ticks=range(len(Ns)))
    cbn.ax.set_yticklabels(Ns, fontsize=5.5); cbn.ax.set_title('$N$', fontsize=6, pad=2); cbn.outline.set_visible(False)
    for ax, letter, key, ylab in [(axs[0, 2], 'c', 'rounds', 'Median first solution (rounds)'),
                                  (axs[1, 0], 'd', 'sweeps', 'Median first solution (sweeps)')]:
        panel(ax, letter)
        for model, col in [('SNM', SNM), ('ARM', ARM)]:
            for k, ls, mk in [(4, '-', 'o'), (3, '--', 's')]:
                dx = {('SNM', 4): -2.4, ('SNM', 3): -0.8, ('ARM', 4): 0.8, ('ARM', 3): 2.4}[(model, k)]   # keep open markers apart
                rr = [c for c in cells if c['model'] == model and c['k'] == k and c['par'] == rep[model]]
                x = np.array([c['N'] for c in rr]); m = np.array([c[f'median_{key}'] if c[f'median_{key}'] is not None else np.nan for c in rr])
                lo = np.array([c[f'ci_{key}'][0] if c[f'ci_{key}'][0] is not None else np.nan for c in rr])
                hi = np.array([c[f'ci_{key}'][1] if c[f'ci_{key}'][1] is not None else np.nan for c in rr])
                ok = np.isfinite(m)
                ax.plot(x[ok], m[ok], ls=ls, marker=mk, ms=3, color=col, lw=1, label=f'{model}, $k = {k}$')
                ax.fill_between(x[ok], lo[ok], np.where(np.isfinite(hi[ok]), hi[ok], budget), color=col, alpha=.15, lw=0)
                if key == 'rounds' and (~ok).any():
                    ax.plot(x[~ok] + dx, np.full((~ok).sum(), budget), marker=mk, ls='none', ms=3.5, mfc='white', mec=col)
        ax.set_yscale('log'); ax.set_xlabel('Network size $N$'); ax.set_ylabel(ylab); ax.set_xticks(Ns[::2])
        if key == 'rounds':
            ax.axhline(budget, color=GREY, ls=':', lw=.8)
    axs[0, 2].legend(fontsize=5.8, loc='lower right')
    heat = LinearSegmentedColormap.from_list('t', ['#F3F6F8', '#9FB3C1', '#4D6577', '#1C2A33'])
    heat.set_bad('#E9B8B1')
    for ax, model, letter, xl in [(axs[1, 1], 'SNM', 'e', r'Behavioural noise $\varepsilon$'), (axs[1, 2], 'ARM', 'f', 'Freezing probability $p$')]:
        panel(ax, letter)
        rr = [c for c in cells if c['model'] == model and c['k'] == 4]
        ps = sorted({c['par'] for c in rr})
        grid = np.full((len(Ns), len(ps)), np.nan)
        for c in rr:
            if c['median_rounds'] is not None:
                grid[Ns.index(c['N']), ps.index(c['par'])] = c['median_rounds']
        im = ax.imshow(np.ma.masked_invalid(grid), origin='lower', aspect='auto', cmap=heat, norm=LogNorm(10, budget),
                       extent=[ps[0] - .025, ps[-1] + .025, -.5, len(Ns) - .5], interpolation='nearest')
        ax.set_yticks(range(len(Ns)), Ns); ax.set_ylabel('Network size $N$'); ax.set_xlabel(xl)
        ax.set_xticks([0, .25, .5, .75, 1]); ax.grid(False)
    cb = fig.colorbar(im, ax=[axs[1, 1], axs[1, 2]], fraction=.04, pad=.02)
    cb.set_label('Median first solution (rounds)', fontsize=6.5)
    save(fig, 'fpt_size')


def networks():
    d = read('supp/networks.json'); cz = read('supp/colourability.json')
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), layout='constrained', sharey=True)
    for ax, g, letter, title in [(axs[0], 'ba', 'a', 'Barabási–Albert'),
                                 (axs[1], 'er', 'b', 'Erdős–Rényi'),
                                 (axs[2], 'ws', 'c', 'Watts–Strogatz')]:
        panel(ax, letter, title)
        for m, col, mk in [('SNM', SNM, 's'), ('ARM', ARM, 'o')]:
            rr = d[f'{g}_{m}']
            band(ax, [r['value'] for r in rr], [r['solved'] for r in rr], col, m, marker=mk)
        ax.axhline(cz[g]['share_3colourable'], color=GREY, ls='--', lw=.8, label='3-colourable graphs')
        fraction(ax); mixed_parameter(ax); ax.set_xlim(0, 1)
        if g == 'ba':
            ax.set_ylabel('Solved within 1,000 rounds')
    axs[1].legend(fontsize=6.5, loc='upper right')
    save(fig, 'networks')


if __name__ == '__main__':
    hybrid(); plateau(); fpt_size(); networks()
