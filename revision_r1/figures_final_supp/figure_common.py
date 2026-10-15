"""Local styling and small deterministic reductions of read-only inputs.

All paths are anchored to this directory. No simulation or fitting is performed.
"""
from pathlib import Path
from functools import lru_cache
import json
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'analyses' / 'runs'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

SNM, ARM = '#0173B2', '#DE8F05'
INK, GREY, PURPLE = '#263238', '#78838D', '#8B63A0'
CLASS = {'solved': '#949494', 'resolve': '#029E73', 'open': '#CC78BC'}
P = np.arange(.05, 1, .1)
SHADES = ['#7A4E00', ARM, '#F0B65E']
FAMILIES = ['PL', 'TPL', 'LN', 'SE', 'EXP']
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#A7AFB4', 'axes.linewidth': .6,
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.4, 'lines.markersize': 3.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'figure.constrained_layout.h_pad': .12,
    'figure.constrained_layout.w_pad': .12,
    'figure.constrained_layout.hspace': .10,
    'figure.constrained_layout.wspace': .08,
})

def read(path):
    return json.loads((DATA / path).read_text())

def rows(data, model='ARM'):
    if isinstance(data, dict):
        data = data['summary']
    return sorted([r for r in data if r['model'] == model], key=lambda r: float(r['p']))

def panel(ax, letter, title=None):
    """Panel letter; a title only where it carries the condition that tells otherwise identical panels apart."""
    if title:
        ax.set_title(title, loc='left', pad=10)
    ax.text(-.18, 1.065, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E9EB', linewidth=.5)

def parameter(ax, model='ARM'):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Freezing probability $p$' if model == 'ARM' else r'Noise probability $\epsilon$')

def mixed_parameter(ax):
    # Axis shared by p (ARM) and epsilon (SNM); same ticks as parameter().
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel(r'Model parameter ($p$ or $\epsilon$)')

def fraction(ax):
    ax.set_ylim(-.025, 1.045)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

def band(ax, x, vals, color, label=None, **kwargs):
    a = np.asarray(vals, float)
    ax.fill_between(x, a[:, 1], a[:, 2], color=color, alpha=.14, linewidth=0)
    return ax.plot(x, a[:, 0], color=color, label=label, **kwargs)[0]

def triples(rr, key):
    return [[r.get(key, np.nan), r.get(key+'_lo', np.nan), r.get(key+'_hi', np.nan)] for r in rr]

def ci_line(ax, x, y, ci, color, label=None, **kwargs):
    y, ci = np.asarray(y), np.asarray(ci)
    return ax.errorbar(x, y, yerr=np.maximum(0, np.array([y-ci[:, 0], ci[:, 1]-y])),
                       color=color, label=label, capsize=2, elinewidth=.8, **kwargs)

@lru_cache(None)
def pooled(model, p, key='sweep|all'):
    with np.load(DATA / f'aval/hist_{model}_{float(p):.2f}.npz') as z:
        a = z[key]
    if a.shape[1] == 0:
        return np.array([], int), np.array([], int)
    s, ind = np.unique(a[1], return_inverse=True)
    n = np.bincount(ind, weights=a[2]).astype(np.int64)
    return s, n

def subtract_hist(first, second):
    s = np.union1d(first[0], second[0])
    n = np.zeros(len(s), dtype=np.int64)
    n[np.searchsorted(s, first[0])] += first[1]
    n[np.searchsorted(s, second[0])] -= second[1]
    assert np.all(n >= 0), 'Subtraction yielded a negative count'
    return s[n > 0], n[n > 0]

def ccdf(ax, s, n, denominator=None, **kwargs):
    if not len(s):
        return
    y = np.cumsum(n[::-1])[::-1] / (n.sum() if denominator is None else denominator)
    ax.step(s, y, where='pre', **kwargs)

def quantile_hist(s, n, q=.5):
    return s[np.searchsorted(np.cumsum(n), q*n.sum())]

def pmf(s, n, edges=None):
    if edges is None:
        edges = np.unique(np.r_[1.5, 2.5, 3.5, 4.5, np.geomspace(4.5, s.max()+.5, 40)])
    c = np.histogram(s, bins=edges, weights=n)[0]
    # Integer support count makes a pmf approximation, rather than density per decade.
    width = np.maximum(1, np.ceil(edges[1:]) - np.ceil(edges[:-1]))
    x = np.sqrt(edges[:-1]*edges[1:])
    return x, c / n.sum() / width, c

def per_decade(ax, s, n, edges, **kwargs):
    c = np.histogram(s, bins=edges, weights=n)[0]
    ax.stairs(c/n.sum()/np.diff(np.log10(edges)), edges, **kwargs)

def smooth_spectrum(f, y, nbin=60):
    f, y = np.asarray(f), np.asarray(y)
    edges = np.geomspace(f[f > 0].min(), f.max()*(1+1e-8), nbin+1)
    x, out = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (f >= a) & (f < b) & np.isfinite(y)
        if mask.any():
            x.append(np.exp(np.mean(np.log(f[mask]))))
            out.append(np.mean(y[mask]))
    return np.asarray(x), np.asarray(out)

def spectrum(ax, f, y, **kwargs):
    x, v = smooth_spectrum(f, y)
    keep = v > 0
    return ax.loglog(x[keep], v[keep], **kwargs)[0]

def aic_matrix(ax, rr, region):
    a = np.array([[r['fits'][region][k]['aic'] for k in FAMILIES] for r in rr])
    a -= np.min(a, axis=1, keepdims=True)
    ax.grid(False)
    im = ax.pcolormesh(np.arange(6)-.5, np.arange(len(rr)+1)-.5,
                      np.maximum(1, a), norm=LogNorm(1, 1e6), cmap='YlOrBr',
                      shading='flat', rasterized=False)
    ax.set_xlim(-.5,4.5);ax.set_ylim(len(rr)-.5,-.5)
    for i, r in enumerate(rr):
        for j, k in enumerate(FAMILIES):
            d = a[i, j]
            label = '0' if d < .05 else (f'{d:.0f}' if d < 1000 else f'{d:.0e}')
            if r['fits'][region][k].get('at_bound'):
                label += '*'
            ax.text(j, i, label, ha='center', va='center', fontsize=6.3,
                    color='white' if d > 1e4 else INK,
                    fontweight='bold' if d < .05 else 'normal')
    ax.set_xticks(range(5), FAMILIES)
    ax.set_yticks(range(len(rr)), [f"{float(r['p']):.2f}" for r in rr])
    ax.set_ylabel('$p$' if rr[0]['model']=='ARM' else r'$\epsilon$')
    return im

def finish(fig, letter):
    fig.savefig(ROOT / f'fig_S{letter}.pdf', bbox_inches='tight', pad_inches=.05,
                metadata={'Title': f'Supplementary Figure S-{letter}',
                          'Creator': f'plot_S{letter}.py (Matplotlib)', 'CreationDate': None})
    fig.savefig(ROOT / f'fig_S{letter}.png', dpi=300, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)
