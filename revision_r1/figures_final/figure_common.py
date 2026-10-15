"""Shared styling and deterministic data reductions; never modifies inputs.
Colours as in the other figures of the paper
(seaborn colorblind: SNM = [0] blue, ARM = [1] orange) and Arial, as in Figs 1, 2 and 6."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'analyses' / 'runs'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, ScalarFormatter

SNM = '#0173B2'                      # seaborn colorblind[0], as in the other figures
ARM = '#DE8F05'                      # seaborn colorblind[1]
INK = '#263238'
GREY = '#78838D'
PURPLE = '#8B63A0'
ARM_SHADES = ['#7A4E00', '#DE8F05', '#F0B65E', '#F7D49E']   # dark -> light, for several p or tau of the ARM
CLASS = {'solved': '#949494', 'resolve': '#029E73', 'open': '#CC78BC'}   # avalanche classes (not model colours)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
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
})

def read(path):
    return json.loads((DATA / path).read_text())

def rows(data, model):
    return sorted([r for r in data if r['model'] == model], key=lambda r: float(r['p']))

def at(data, model, p):
    return next(r for r in data if r['model'] == model and abs(float(r['p'])-p) < 1e-8)

def panel(ax, letter, title=None):
    """Panel letter; a title only where it carries the condition that tells otherwise identical panels apart."""
    if title:
        ax.set_title(title, loc='left', pad=10, fontweight='medium')
    ax.text(-.17, 1.065, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E9EB', linewidth=.5)

def fraction(ax):
    ax.set_ylim(-.025, 1.045)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

def parameter(ax, label):
    ax.set_xlim(-.015, 1.005)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel(label)

def ribbon(ax, x, triples, color, label=None, **kw):
    a = np.asarray(triples, float)
    ax.fill_between(x, a[:,1], a[:,2], color=color, alpha=.16, linewidth=0)
    return ax.plot(x, a[:,0], color=color, label=label, **kw)[0]

def errors(ax, x, triples, color, label=None, **kw):
    a = np.asarray(triples, float)
    return ax.errorbar(x, a[:,0], yerr=np.maximum(0, np.array([a[:,0]-a[:,1], a[:,2]-a[:,0]])),
                       color=color, label=label, capsize=2, elinewidth=.8, **kw)

def wilson(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.array([p, np.maximum(0, center-half), np.minimum(1, center+half)]).T

def note(ax, text, xy=(.04,.06), **kw):
    ax.text(*xy, text, transform=ax.transAxes, fontsize=7, va=kw.pop('va', 'bottom'),
            bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2), **kw)

def log_ticks(ax, values):
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ScalarFormatter())

def log_edges(maximum, bins_per_decade=9):
    # Discrete small sizes retain separate bins; broader bins collect the tail.
    return np.unique(np.r_[.5, 1.5, 2.5, 3.5, 4.5,
                           np.geomspace(4.5, max(5.5, maximum+1),
                                        int(np.ceil(np.log10(maximum/4.5)*bins_per_decade))+1)])

def density(sizes, weights, edges, denominator):
    counts = np.histogram(sizes, bins=edges, weights=weights)[0]
    return counts / denominator / np.diff(np.log10(edges))

def finish(fig, number, footer=None):
    # Footer notes are not drawn on the figure: they belong to the figure legend.
    # The argument is kept so that each script still records what its caption must say.
    kw = dict(bbox_inches='tight', pad_inches=0.04)
    fig.savefig(ROOT / f'fig{number}.pdf', metadata={'Title': f'Figure {number}',
                'Creator': f'plot_fig{number}.py (Matplotlib)', 'CreationDate': None}, **kw)
    fig.savefig(ROOT / f'fig{number}.png', dpi=300, **kw)
    plt.close(fig)
