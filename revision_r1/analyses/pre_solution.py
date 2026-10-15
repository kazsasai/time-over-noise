"""
The period BEFORE the first solution (initial relaxation from the random colouring): (1) its duration
(first-passage time, rounds and sweeps) and (2) the sizes of avalanches that start in it (primary sweep-clock
definition: key 'sweep' minus 'sweep_stat'), with discrete MLE (PL/TPL/LN/SE/EXP) and a KS-optimal TPL tail.
Output: runs/pre/pre_solution.json, runs/pre/figs/pre_solution.png.  Run from revision_r1/analyses.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aval_mle as am

warnings.filterwarnings('ignore')
OUT = 'runs/pre'
PS = ['0.05', '0.25', '0.45', '0.65', '0.85']


def pooled(z, key):
    h = am.Hist(z, key)
    s, c = h.pooled()
    return dict(zip(s.tolist(), c.tolist()))


def fits(s, c, xmin):
    st, ct = am.tail_of(s, c, xmin)
    if ct.sum() < 50:
        return None
    f, out = am.fit_all(st, ct, xmin)
    aic = {m: am.ic(r, f.n)['aic'] for m, r in out.items()}
    best = min(aic, key=aic.get)
    return dict(xmin=int(xmin), n=float(ct.sum()), best=best, dAIC={m: round(aic[m] - aic[best], 1) for m in aic},
                PL_alpha=round(out['PL']['theta'][0], 2),
                TPL=[round(out['TPL']['theta'][0], 2), round(float(np.exp(-out['TPL']['theta'][1])), 1)],
                KS={m: round(am.ks_distance(f, out[m]), 4) for m in ('PL', 'TPL', 'LN')})


def ks_tpl_xmin(s, c):
    best = None
    for x in am.xmin_grid(s, c, min_tail=200, upto=min(5000, s.max())):
        st, ct = am.tail_of(s, c, x)
        f = am.Fitter(st, ct, x); r = f.fit('TPL'); D = am.ks_distance(f, r)
        if np.isfinite(D) and (best is None or D < best[1]):
            best = (int(x), D)
    return best[0] if best else None


def main():
    os.makedirs(f'{OUT}/figs', exist_ok=True)
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))
    cm = plt.get_cmap('viridis')
    for i, p in enumerate(PS):
        col = cm(i / (len(PS) - 1))
        pf = float(p)
        g = df[(df['label'] == 'Async Model') & ((df['p_async'] - pf).abs() < 1e-6)]
        T = (g['solve_time'].values + 1)[g['solved'].values == 1].astype(float)
        sv, cv = np.unique(T, return_counts=True)
        r = dict(n_solved=int(T.size), T_median_rounds=float(np.median(T)), T_median_sweeps=float(np.median(T) * (1 - pf)),
                 T_p99_rounds=float(np.percentile(T, 99)))
        r['T_fit_xmin1'] = fits(sv, cv.astype(float), 1)
        x0 = ks_tpl_xmin(sv, cv.astype(float))
        r['T_fit_KS'] = fits(sv, cv.astype(float), x0) if x0 else None
        ax[0].loglog(np.sort(T), 1 - np.arange(T.size) / T.size, color=col, label=f'ARM p={p}')
        ax[1].loglog(np.sort(T) * (1 - pf), 1 - np.arange(T.size) / T.size, color=col, label=f'ARM p={p}')
        # avalanches that start before (or at) the first solution, sweep clock
        with np.load(f'runs/aval/hist_ARM_{p}.npz') as z:
            a_all, a_stat = pooled(z, 'sweep|all'), pooled(z, 'sweep_stat|all')
            n_trials = int(z['n_trials'])
        pre = {s: a_all[s] - a_stat.get(s, 0) for s in a_all if a_all[s] - a_stat.get(s, 0) > 0}
        s = np.array(sorted(pre), dtype=float); c = np.array([pre[k] for k in sorted(pre)], dtype=float)
        r['pre_aval_n'] = float(c.sum()); r['pre_aval_per_trial'] = float(c.sum() / n_trials)
        r['pre_aval_share_of_all'] = float(c.sum() / sum(a_all.values()))
        r['pre_aval_size_median'] = float(np.repeat(s, c.astype(int))[int(c.sum()) // 2])
        r['pre_aval_size_max'] = float(s.max())
        r['pre_fit_xmin2'] = fits(s, c, 2)
        x1 = ks_tpl_xmin(s, c)
        r['pre_fit_KS'] = fits(s, c, x1) if x1 else None
        cc = (c.sum() - np.concatenate([[0], np.cumsum(c)[:-1]])) / c.sum()
        ax[2].loglog(s, cc, '.', ms=2, color=col, label=f'ARM p={p} (n={int(c.sum()):,})')
        res[p] = r
        print(p, json.dumps(r, ensure_ascii=False), flush=True)
    ax[0].set(xlabel='rounds to the first solution', ylabel='CCDF', title='duration of the pre-solution period (rounds)')
    ax[1].set(xlabel='sweeps to the first solution', ylabel='CCDF', title='duration of the pre-solution period (sweeps)')
    ax[2].set(xlabel='avalanche size (sweep clock)', ylabel='CCDF', title='avalanches starting before the first solution')
    for a in ax:
        a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f'{OUT}/figs/pre_solution.png', dpi=85)
    json.dump(res, open(f'{OUT}/pre_solution.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
