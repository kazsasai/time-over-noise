"""Fig. 6: coordination across network size and number of colours.

a-d: fraction solved vs eps (SNM) or p (ARM) at k = 3 and 4, N = 20..100 (200 trials, 10,000 rounds), Wilson 95%.
e-f: fraction solved over N = 20..500 and k = 2..50 at the representative values (200 trials, 5,000 rounds).
     Cells with at least 90% are outlined; a marker shows the smallest k reaching 90% at each N.
Input: ../analyses/runs/fig6/fig6_grids.json (written by analyses/fig6_numbers.py from the figure data).
"""
from figure_common import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HEAT = LinearSegmentedColormap.from_list('solved', ['#FFFFFF', '#C9D3DA', '#7E909C', '#2F3E46'])


def plateau(ax, cell, model, k):
    ns = cell['N']
    x = np.array(cell['param'])
    colours = plt.cm.viridis(np.linspace(0, .9, len(ns)))
    for n, col, s, t in zip(ns, colours, cell['solved'], cell['trials']):
        w = wilson(s, t)
        ax.fill_between(x, w[:, 1], w[:, 2], color=col, alpha=.13, linewidth=0)
        ax.plot(x, w[:, 0], color=col, lw=1.1, label=f'{n}')
    ax.axhline(.9, color=GREY, lw=.7, ls='--')
    fraction(ax)
    parameter(ax, r'Behavioral noise $\varepsilon$' if model == 'SNM' else r'Freezing probability $p$')
    ax.set_ylabel('Solved within 10,000 rounds')
    return colours


def outline(ax, ok):
    """Draw the boundary of the cells with ok == True."""
    rows, cols = ok.shape
    for i in range(rows):
        for j in range(cols):
            if not ok[i, j]:
                continue
            for di, dj, seg in [(-1, 0, ([j-.5, j+.5], [i-.5, i-.5])), (1, 0, ([j-.5, j+.5], [i+.5, i+.5])),
                                (0, -1, ([j-.5, j-.5], [i-.5, i+.5])), (0, 1, ([j+.5, j+.5], [i-.5, i+.5]))]:
                ii, jj = i + di, j + dj
                if not (0 <= ii < rows and 0 <= jj < cols) or not ok[ii, jj]:
                    ax.plot(*seg, color='white', lw=1.6, solid_capstyle='butt', zorder=3)
                    ax.plot(*seg, color=INK, lw=.6, solid_capstyle='butt', zorder=4)


def heatmap(ax, cell, model, colour):
    ns, ks = cell['N'], cell['k']
    rate = np.array(cell['solved']) / np.array(cell['trials'])
    im = ax.imshow(rate, vmin=0, vmax=1, cmap=HEAT, aspect='auto', origin='upper', interpolation='nearest')
    ok = rate >= .9
    outline(ax, ok)
    for i in range(len(ns)):
        j = np.flatnonzero(ok[i])
        if j.size:
            ax.plot(j[0], i, marker='o', ms=4.2, mfc=colour, mec='white', mew=.8, zorder=5)
    ticks = [i for i, k in enumerate(ks) if k in (2, 10, 20, 30, 40, 50)]
    ax.set_xticks(ticks, [ks[i] for i in ticks])
    ax.set_yticks(range(len(ns)), ns)
    ax.set_xlabel('Number of colors $k$')
    ax.set_ylabel('Network size $N$')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def check_caption(g):
    """Numbers quoted in the Fig. 6 caption, recomputed from the plotted counts."""
    def rate(cell):
        return np.array(cell['solved']) / np.array(cell['trials'])
    def smallest_k(model):
        c = g['heatmap'][model]; r = rate(c)
        return {n: (c['k'][int(np.flatnonzero(row >= .9)[0])] if (row >= .9).any() else None) for n, row in zip(c['N'], r)}
    arm, snm = smallest_k('ARM'), smallest_k('SNM')
    def span(d, lo, hi):
        v = [d[n] for n in d if lo <= n <= hi]
        return f'{min(v)}–{max(v)}' if min(v) != max(v) else f'{v[0]}'
    hs = rate(g['heatmap']['SNM']); ks = g['heatmap']['SNM']['k']; ns = g['heatmap']['SNM']['N']
    best200 = hs[ns.index(200)].max()
    p3s, p3a = rate(g['plateau']['SNM_k3']), rate(g['plateau']['ARM_k3'])
    n3 = g['plateau']['SNM_k3']['N']
    q = [f"smallest such k is {span(arm, 0, 125)} for N ≤ 125, {arm[150]} at N = 150 and {span(arm, 200, 500)} for N = 200–500",
         f"{span(snm, 60, 150)} for N = 60–150",
         f"best {100 * best200:.1f}% at N = 200",
         f"{100 * hs[ns.index(50)][ks.index(3)]:.0f}% at k = 3, {100 * hs[ns.index(50)][ks.index(6)]:.0f}% at k = 6",
         f"N = 50: {p3s[n3.index(50)].max():.3f} vs {p3a[n3.index(50)].max():.2f}; N = 60: {p3s[n3.index(60)].max():.3f} vs {p3a[n3.index(60)].max():.3f}"]
    # The quoted numbers now appear in the main text and are matched by tex_rev1/check_manuscript.py.
    assert all(arm[n] is not None for n in arm), 'ARM misses 90% at some N'
    assert all(snm[n] is None for n in snm if n >= 200) and hs[ns.index(400):].max() == 0
    print('Fig. 6 structural checks passed')


def main():
    g = read('fig6/fig6_grids.json')
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.35], hspace=.5, wspace=.32,
                          left=.08, right=.9, top=.96, bottom=.06)
    specs = [('a', 'SNM', 3), ('b', 'ARM', 3), ('c', 'SNM', 4), ('d', 'ARM', 4)]
    for idx, (letter, model, k) in enumerate(specs):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        panel(ax, letter, f'$k = {k}$')
        colours = plateau(ax, g['plateau'][f'{model}_k{k}'], model, k)
        if letter == 'b':
            handles = [Line2D([], [], color=c, lw=1.4, label=f'{n}') for c, n in zip(colours, g['plateau'][f'{model}_k{k}']['N'])]
            ax.legend(handles=handles, title='$N$', title_fontsize=7, ncol=1, loc='center left',
                      bbox_to_anchor=(1.01, .5), handlelength=1.4)
    for j, (letter, model, colour, label) in enumerate([('e', 'SNM', SNM, r'SNM, $\varepsilon = 0.05$'),
                                                       ('f', 'ARM', ARM, r'ARM, $p = 0.45$')]):
        ax = fig.add_subplot(gs[2, j])
        panel(ax, letter, label)
        im = heatmap(ax, g['heatmap'][model], model, colour)
    cax = fig.add_axes([.915, .06, .012, .22])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Fraction solved', fontsize=7)
    cb.set_ticks([0, .25, .5, .75, 1])
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False)
    finish(fig, 6)
    check_caption(g)


if __name__ == '__main__':
    main()
