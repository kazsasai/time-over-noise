"""ARM results split by graph type (Supplementary Table S11).

Graph types (Methods 6; run_departure_mechanism.py).  A graph is 'breakable' if some proper 3-colouring has an edge
whose two ends can change, in the same round, to the same colour; in a 'locked' graph (324 of the 1,000 core graphs)
no proper colouring has such an edge, so a consistent colouring, once reached, is never lost.  The type is a property
of the graph (seed) and is the same at every p.

A. Holding and returns after the first solution: retention() of run_first_passage.py on each subset of the core data.
B. Avalanche populations: the per-trial histograms of avalanche_histograms.py (trial index = row order of each
   (model, p) group of the core data), split by the graph of each trial; s* is the pooled value of planC.json.
C. Injected conflicts: the kicked runs of run_kick.py and run_kick_growth.py (identical task), re-simulated with
   the seed kept so that each response can be assigned to its graph.  The cascade probability is pooled over injections
   (cascades / injections); its 95% interval comes from bootstrap over graphs, each graph keeping all its injections,
   because the up to 40 injections of a trial share one graph (also for all graphs together: Supplementary Fig. S7j).
Every split is checked against the pooled value it must reproduce.

Output: runs/graph_types/graph_types.json.  Run from revision_r1/analyses.
"""
import json, math, os
import multiprocessing as mp
import numpy as np
import pandas as pd
from gcg_kick import simulate
from run_first_passage import groups, retention
import run_kick as rk
import run_kick_growth as rg

OUT = 'runs/graph_types'
PS = [round(p, 2) for p in np.arange(10) / 10 + 0.05]
TYPE = {x['seed']: ('breakable' if x['structural'] else 'locked') for x in json.load(open('runs/departure/departure_0.45.json'))}
def cascade_ci(per, rng, B=2000):
    """Pooled share of cascades among the injections, with a 95% interval from bootstrap over graphs.
    per: one array of response rows per graph (columns as run_kick.responses)."""
    k = np.array([((r[:, 3] >= 15) | (r[:, 2] >= 100)).sum() if len(r) else 0 for r in per], float)
    n = np.array([len(r) for r in per], float)
    pick = rng.integers(0, k.size, (B, k.size))
    bs = k[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(k.sum() / n.sum()), float(lo), float(hi)]


def kicked(args):
    """One kicked run (the task of run_kick.worker and run_kick_growth.worker), with both outputs."""
    par, seed = args
    rate = 1 - par
    r = simulate(dict(seed=int(seed), node_num=20, color_num=3, step_max=rk.T, gridlock_rule='random',
                      p_async=par, p_noise=0.0, n_kicks=40, kick_wait=int(math.ceil(50 / rate)),
                      kick_gap=int(math.ceil(1000 / rate))))
    resp = rk.responses(r, rate)
    C = r['history_conflict_nodes'].astype(float)
    idx = np.maximum(1, np.round(rg.TAUS / rate).astype(int))
    grow = [np.concatenate([[c0], C[t0 + idx]]) for t0, c0 in zip(r['kick_times'], r['kick_c0']) if t0 + idx[-1] < rg.T]
    return int(seed), resp, (np.array(grow) if grow else np.zeros((0, rg.TAUS.size + 1)))


def part_c(pool):
    out = {}
    pooled_ref = json.load(open('runs/kick/kick_growth.json'))
    for p in PS:
        res = pool.map(kicked, [(p, s) for s in rk.SEEDS], chunksize=8)
        rows = np.array([x for _, resp, _ in res for x in resp], float)
        ref = np.load(f'runs/kick/kick_raw_ARM_{p:.2f}.npz')['kicks']
        assert np.array_equal(rows, ref), f'kick responses at p={p} do not reproduce'
        M = np.concatenate([g for _, _, g in res if g.size])
        assert np.allclose(rg.stats(M)[2], pooled_ref[f'ARM_{p:.2f}']['Lambda'], equal_nan=True)
        d = {}
        for t in ('breakable', 'locked'):
            sel = [(resp, g) for s, resp, g in res if TYPE[s] == t]
            rr = np.array([x for resp, _ in sel for x in resp], float)
            casc = (rr[:, 3] >= 15) | (rr[:, 2] >= 100)
            per = [g for _, g in sel]
            Mt = np.concatenate([g for g in per if g.size])
            lam = rg.stats(Mt)[2]
            rng = np.random.default_rng([2024, 77, int(round(p * 100)), t == 'locked'])
            bs = []
            for _ in range(200):
                pick = rng.integers(0, len(per), len(per))
                bs.append(rg.stats(np.concatenate([per[i] for i in pick if per[i].size]))[2])
            lo, hi = np.nanpercentile(np.array(bs), [2.5, 97.5], axis=0)
            comp = rr[rr[:, 0] == 1]
            per_resp = [np.array(resp, float).reshape(-1, 4) for resp, _ in sel]
            cascade = cascade_ci(per_resp, np.random.default_rng([2024, 78, int(round(p * 100)), t == 'locked']))
            assert np.isclose(cascade[0], casc.mean())
            d[t] = dict(n_graphs=len(sel), n_kicks=int(len(rr)), completed=float(rr[:, 0].mean()), cascade=cascade,
                        median_size=float(np.median(comp[:, 2])), taus=rg.TAUS.tolist(), Lambda=lam.tolist(),
                        Lambda_lo=lo.tolist(), Lambda_hi=hi.tolist())
        d['all_cascade'] = cascade_ci([np.array(resp, float).reshape(-1, 4) for _, resp, _ in res], np.random.default_rng([2024, 78, int(round(p * 100)), 2]))
        assert np.isclose(d['all_cascade'][0], ((rows[:, 3] >= 15) | (rows[:, 2] >= 100)).mean())
        out[f'{p:.2f}'] = d
        print('C', p, {t: (round(d[t]['cascade'][0], 3), round(d[t]['Lambda'][1], 3)) for t in ('breakable', 'locked')}, flush=True)
    return out


def part_ab():
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    plan = {r['p']: r for r in json.load(open('runs/aval/planC_sweep/planC.json'))['summary'] if r['model'] == 'ARM'}
    fp = {round(r['p'], 2): r for r in json.load(open('runs/fpt/first_passage.json'))['rows']['long'] if r['model'] == 'ARM'}
    A, B = {}, {}
    for model, p, rate, g in groups(df):
        if model != 'ARM':
            continue
        p = round(p, 2)
        types = g['seed'].map(TYPE).to_numpy()
        assert not pd.isna(types).any()
        whole = retention(g, rate)[0]
        assert np.isclose(whole['time_frac_solved_pooled'], fp[p]['time_frac_solved_pooled'])
        A[f'{p:.2f}'] = {t: {k: retention(g[types == t], rate)[0][k] for k in
                             ('n_trials_solved', 'time_frac_solved_pooled', 'crossings_per_1000_sweeps', 'share_never_leave',
                              'leaving_time_frac_solved')} for t in ('breakable', 'locked')}
        # avalanche populations by graph type
        sstar = plan[f'{p:.2f}']['s_star']
        with np.load(f'runs/aval/hist_ARM_{p:.2f}.npz') as z:
            H = {c: z[f'sweep|{c}'] for c in ('solved', 'resolve', 'open')}
            Hs = {c: z[f'sweep_stat|{c}'] for c in ('solved', 'resolve', 'open')}
        b = {}
        for t in ('breakable', 'locked', 'all'):
            keep = np.ones(len(types), bool) if t == 'all' else types == t
            sel = {c: a[:, keep[a[0].astype(int)]] for c, a in H.items()}
            sels = {c: a[:, keep[a[0].astype(int)]] for c, a in Hs.items()}
            n = {c: float(a[2].sum()) for c, a in sel.items()}
            uns = np.concatenate([sel['resolve'], sel['open']], axis=1)
            big = uns[1] >= sstar
            res_big = float(sel['resolve'][2][sel['resolve'][1] >= sstar].sum())
            sol = sel['solved']
            b[t] = dict(n_trials=int(keep.sum()), solved_share=n['solved'] / sum(n.values()),
                        q999_solved=float(np.percentile(np.repeat(sol[1], sol[2].astype(int)), 99.9)),
                        large_per_trial=float(uns[2][big].sum() / keep.sum()),
                        resolve_share_large=res_big / float(uns[2][big].sum()) if big.any() else None,
                        mean_excess=float(np.average(uns[1][big] - sstar, weights=uns[2][big])) if big.any() else None,
                        unsolved_after_first_solution=float(sels['resolve'][2].sum() + sels['open'][2].sum()))
        assert np.isclose(b['all']['mean_excess'], plan[f'{p:.2f}']['mean_excess'])
        B[f'{p:.2f}'] = b
        print('A/B', p, {t: (round(A[f'{p:.2f}'][t]['time_frac_solved_pooled'], 4), round(A[f'{p:.2f}'][t]['crossings_per_1000_sweeps'], 1))
                         for t in ('breakable', 'locked')}, {t: (round(b[t]['solved_share'], 3), b[t]['mean_excess'] and round(b[t]['mean_excess']),
                         b[t]['unsolved_after_first_solution']) for t in ('breakable', 'locked')}, flush=True)
    return A, B


def main():
    os.makedirs(OUT, exist_ok=True)
    with mp.get_context('fork').Pool(12) as pool:          # before the core data are loaded
        C = part_c(pool)
    A, B = part_ab()
    json.dump(dict(n_graphs={t: sum(v == t for v in TYPE.values()) for t in ('breakable', 'locked')},
                   holding=A, avalanches=B, injection=C), open(f'{OUT}/graph_types.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
