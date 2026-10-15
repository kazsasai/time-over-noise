"""Fig 6: every number quoted in the text, recomputed from the figure data.

Inputs (read-only, in the repository root; repo_path.py):
  result_for_scalability.pkl  heatmaps e, f: N in {20, ..., 500}, k = 2..50, 200 trials, 5,000 rounds,
                              SNM eps = 0.05, ARM p = 0.45
  result_for_plateau_k34.pkl  panels a-d: N = 20..100, k = 3, 4, eps / p on a 0.05 grid, 200 trials, 10,000 rounds
Output: runs/fig6/fig6_numbers.json (quoted numbers), runs/fig6/fig6_grids.json (cell counts for plot_fig6.py)

Equal-activation comparisons use the stored solve_time (rounds): ARM sweeps = rounds x (1 - p), SNM sweeps = rounds.

Usage (from revision_r1/analyses):  python3 fig6_numbers.py
"""
import json, math, os
import numpy as np
import pandas as pd
import networkx as nx

from repo_path import REPO
OUT = 'runs/fig6'


def rate_within(sub, sweeps, p_async):
    """Share of trials solved within `sweeps` sweeps (solve_time is in rounds; unsolved trials count as failures)."""
    t = sub['solve_time'].to_numpy(float) * (1 - p_async)
    return float(np.mean((sub['solved'].to_numpy() == 1) & (t <= sweeps)))


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = pd.read_pickle(os.path.join(REPO, 'result_for_scalability.pkl'))
    pl = pd.read_pickle(os.path.join(REPO, 'result_for_plateau_k34.pkl'))
    res = {'heatmap': {}, 'plateau': {}, 'design': {}}
    res['design'] = dict(
        heatmap_N=sorted(int(n) for n in sc.node_num.unique()), heatmap_k=[int(sc.color_num.min()), int(sc.color_num.max())],
        heatmap_rounds=sorted(int(x) for x in sc.step_max.unique()),
        heatmap_trials=int(sc.groupby(['label', 'node_num', 'color_num']).size().unique()[0]),
        plateau_N=sorted(int(n) for n in pl.node_num.unique()), plateau_rounds=sorted(int(x) for x in pl.step_max.unique()),
        plateau_trials=int(pl.groupby(['label', 'node_num', 'color_num', 'p_async', 'p_noise']).size().unique()[0]))

    # ---- heatmaps (e, f) ----
    r = sc.groupby(['label', 'node_num', 'color_num'])['solved'].mean()
    for N in res['design']['heatmap_N']:
        kc = math.ceil(math.sqrt(2 * N))
        row = {'k_ref': kc}
        for lab, model, p in [('Noise', 'SNM', 0.0), ('Async', 'ARM', 0.45)]:
            s = r[lab][N]
            ok = [int(k) for k, v in s.items() if v >= 0.9]
            sub = sc[(sc.label == lab) & (sc.node_num == N) & (sc.color_num == kc)]
            row[model] = dict(at_k_ref=float(s[kc]), best=float(s.max()), best_k=int(s.idxmax()),
                              k_ge90=[min(ok), max(ok)] if ok else None,
                              # same number of agent activations as the ARM record (5,000 rounds x 0.55 = 2,750 sweeps)
                              at_k_ref_2750sw=rate_within(sub, 2750, p))
        seeds = sc[(sc.label == 'Async') & (sc.node_num == N) & (sc.color_num == 3)]['seed']
        md = [max(d for _, d in nx.barabasi_albert_graph(N, 2, seed=int(x)).degree()) for x in seeds]
        row['max_degree'] = dict(median=float(np.median(md)), p10=float(np.percentile(md, 10)), p90=float(np.percentile(md, 90)))
        res['heatmap'][str(N)] = row

    # ---- plateaus (a-d) ----
    for lab, model, par in [('Noise', 'SNM', 'p_noise'), ('Async', 'ARM', 'p_async')]:
        for k in (3, 4):
            sub = pl[(pl.label == lab) & (pl.color_num == k)]
            g = sub.groupby(['node_num', par])['solved'].mean().unstack(par)
            for N, row in g.iterrows():
                ok = row[row >= 0.9]
                best_par = float(row.idxmax())
                bsub = sub[(sub.node_num == N) & (sub[par] == best_par)]
                pa = best_par if model == 'ARM' else 0.0
                res['plateau'][f'{model}_k{k}_N{int(N)}'] = dict(
                    best=float(row.max()), best_param=best_par,
                    range_ge90=[float(ok.index.min()), float(ok.index.max())] if len(ok) else None,
                    best_sweeps_budget=10000 * (1 - pa),
                    best_within_2500sw=rate_within(bsub, 2500, pa))
    json.dump(res, open(f'{OUT}/fig6_numbers.json', 'w'), indent=1)

    # ---- compact grids for plotting (solved counts and trials per cell) ----
    grids = {'heatmap': {}, 'plateau': {}}
    for lab, model in [('Noise', 'SNM'), ('Async', 'ARM')]:
        g = sc[sc.label == lab].groupby(['node_num', 'color_num'])['solved'].agg(['sum', 'count'])
        Ns, ks = sorted(sc.node_num.unique()), sorted(sc.color_num.unique())
        grids['heatmap'][model] = dict(N=[int(n) for n in Ns], k=[int(k) for k in ks],
                                       solved=[[int(g.loc[(n, k), 'sum']) for k in ks] for n in Ns],
                                       trials=[[int(g.loc[(n, k), 'count']) for k in ks] for n in Ns])
    for lab, model, par in [('Noise', 'SNM', 'p_noise'), ('Async', 'ARM', 'p_async')]:
        for k in (3, 4):
            sub = pl[(pl.label == lab) & (pl.color_num == k)]
            g = sub.groupby(['node_num', par])['solved'].agg(['sum', 'count'])
            Ns, ps = sorted(sub.node_num.unique()), sorted(sub[par].unique())
            grids['plateau'][f'{model}_k{k}'] = dict(N=[int(n) for n in Ns], param=[round(float(p), 4) for p in ps],
                                                    solved=[[int(g.loc[(n, p), 'sum']) for p in ps] for n in Ns],
                                                    trials=[[int(g.loc[(n, p), 'count']) for p in ps] for n in Ns])
    json.dump(grids, open(f'{OUT}/fig6_grids.json', 'w'))
    print('written', f'{OUT}/fig6_numbers.json', f'{OUT}/fig6_grids.json')


if __name__ == '__main__':
    main()
