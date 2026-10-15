"""Data for three supplementary figures that are regenerated for the revision (rev1).

  hybrid    : mixing the two models at N = 20, k = 3 (SNM rule with eps = 0.05 and freezing p; ARM rule with p = 0.45
              and behavioural noise eps), p, eps = 0.00..0.95 in steps of 0.05, 1,000 trials with the core seeds,
              1,000 rounds, with the seeded solver.
  networks  : Barabasi-Albert (m = 2), Erdos-Renyi (q = 2m/(N-1)) and Watts-Strogatz (k = 4, beta = 0.3) graphs,
              N = 20, k = 3, eps / p = 0.05..0.95 in steps of 0.1, 1,000 trials with the core seeds, 1,000 rounds.
  plateau   : lower edge p_min(N) of the ARM 90% range at k = 4 (data of Fig. 6a-d), linearly interpolated between
              grid points, and the exponent of (1 - p_min) ~ N^(-a) fitted on N = 60..100 with a 95% interval from
              bootstrap over trials (resampling the 200 trials of every (N, p) cell).
Output: runs/supp/{hybrid,networks,plateau}.json.  Run from revision_r1/analyses.
"""
import json, os, sys
import multiprocessing as mp
import numpy as np
import pandas as pd

from repo_path import REPO
OUT = 'runs/supp'
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [float(p), float(max(0, c - h)), float(min(1, c + h))]


def _hybrid_one(args):
    from gcg_kick import simulate
    rule, p, eps, seed = args
    r = simulate(dict(seed=int(seed), node_num=20, color_num=3, ba_m=2, bot_num=20, bot_type='c', step_max=1000,
                      gridlock_rule=rule, p_async=p, p_noise=eps))
    return int((r['history_conflict_nodes'] == 0).any())


def _network_one(task):
    sys.path.insert(0, REPO)
    import gcg_solver
    return int(gcg_solver._worker_run_single(task)['solved'])


def hybrid(pool):
    grid = np.round(np.arange(20) * 0.05, 2)
    out = {}
    for name, rule, fixed in [('SNM_eps0.05_vary_p', 'stay', 0.05), ('ARM_p0.45_vary_eps', 'random', 0.45)]:
        rows = []
        for v in grid:
            p, eps = (v, fixed) if rule == 'stay' else (fixed, v)
            s = pool.map(_hybrid_one, [(rule, p, eps, sd) for sd in SEEDS], chunksize=25)
            rows.append(dict(value=float(v), solved=wilson(sum(s), len(s))))
        out[name] = rows
        print(name, [round(r['solved'][0], 3) for r in rows], flush=True)
    return out


def networks(pool):
    grid = np.round(np.arange(0.05, 1.0, 0.1), 2)
    q_er = 2 * 2 / (20 - 1)
    graphs = [('ba', {}), ('er', {'graph_param': q_er}), ('ws', {'graph_param': 0.3})]
    out = {'q_er': q_er, 'ws_k': 4, 'ws_beta': 0.3}
    for g, extra in graphs:
        for model, is_async in [('SNM', False), ('ARM', True)]:
            rows = []
            for v in grid:
                tasks = [dict(node_num=20, color_num=3, ba_m=2, bot_num=20, bot_type='c', step_max=1000,
                              return_history=False, seed=int(sd), label=model, graph_type=g, is_async_model=is_async,
                              p_async=float(v) if is_async else 0.0, p_noise=0.0 if is_async else float(v), **extra)
                         for sd in SEEDS]
                s = pool.map(_network_one, tasks, chunksize=25)
                rows.append(dict(value=float(v), solved=wilson(sum(s), len(s))))
            out[f'{g}_{model}'] = rows
            print(g, model, [round(r['solved'][0], 3) for r in rows], flush=True)
    return out


def _edge(ps, rates, thr=0.9):
    """Smallest p at which the rate reaches thr, interpolated linearly between grid points."""
    ok = np.flatnonzero(rates >= thr)
    if ok.size == 0:
        return np.nan
    i = ok[0]
    if i == 0:
        return ps[0]
    x0, x1, y0, y1 = ps[i - 1], ps[i], rates[i - 1], rates[i]
    return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)


def plateau(B=1000):
    df = pd.read_pickle(os.path.join(REPO, 'result_for_plateau_k34.pkl'))
    sub = df[(df.label == 'Async') & (df.color_num == 4)]
    Ns = sorted(sub.node_num.unique())
    ps = np.array(sorted(sub.p_async.unique()))
    ps_fit = ps[ps <= 0.5]                                      # the lower edge lies well below p = 0.5
    cells = {(n, p): sub[(sub.node_num == n) & (sub.p_async == p)]['solved'].to_numpy() for n in Ns for p in ps_fit}
    fitN = [n for n in Ns if n >= 60]

    def edges(sample):
        return {n: _edge(ps_fit, np.array([sample[(n, p)].mean() for p in ps_fit])) for n in Ns}

    def slope(ed):
        x = np.log([n for n in fitN]); y = np.log([1 - ed[n] for n in fitN])
        b, a = np.polyfit(x, y, 1)
        return b, a

    e0 = edges(cells); b0, a0 = slope(e0)
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(B):
        smp = {key: v[rng.integers(0, v.size, v.size)] for key, v in cells.items()}
        boots.append(slope(edges(smp))[0])
    boots = np.array(boots)
    out = dict(threshold=0.9, fit_N=[int(n) for n in fitN], edge={str(int(n)): float(e0[n]) for n in Ns},
               exponent=float(-b0), exponent_ci=[float(-np.percentile(boots, 97.5)), float(-np.percentile(boots, 2.5))],
               prefactor=float(np.exp(a0)), B=B)
    # sensitivity of the exponent to the threshold and to the fitted range of N
    sens = []
    for thr in (0.90, 0.95, 0.99):
        for nmin in (60, 70):
            fN = [n for n in Ns if n >= nmin]
            def fit(sample):
                ed = {n: _edge(ps_fit, np.array([sample[(n, p)].mean() for p in ps_fit]), thr) for n in Ns}
                return -np.polyfit(np.log(fN), np.log([1 - ed[n] for n in fN]), 1)[0], ed
            e, ed = fit(cells)
            bs = np.array([fit({k: v[rng.integers(0, v.size, v.size)] for k, v in cells.items()})[0] for _ in range(B // 2)])
            sens.append(dict(threshold=thr, N_min=nmin, exponent=float(e), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             edge={str(int(n)): float(ed[n]) for n in Ns}))
    out['sensitivity'] = sens
    print('plateau', out, flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['plateau', 'hybrid', 'networks']
    if 'plateau' in which:
        json.dump(plateau(), open(f'{OUT}/plateau.json', 'w'), indent=1)
    with mp.get_context('spawn').Pool(12) as pool:
        if 'hybrid' in which:
            json.dump(hybrid(pool), open(f'{OUT}/hybrid.json', 'w'), indent=1)
        if 'networks' in which:
            json.dump(networks(pool), open(f'{OUT}/networks.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
