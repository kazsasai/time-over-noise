"""
Finite-time growth rate of an injected inconsistency (a transverse-Lyapunov-like exponent, cf. Abe 2020;
Platt, Spiegel & Tresser 1993), from the injection experiment of run_kick.py.
After each kick: c0 = conflicting nodes right after the kick; c(tau) = conflicting nodes tau sweeps later
(0 once healed; healed responses are kept, as in damage-spreading averages).
  Lambda(tau) = (1/tau) ln( <c(tau)> / <c0> )      > 0 grows on average, < 0 decays, ~ 0 marginal
also the survival P(c(tau) > 0) and the mean among survivors.  Trial (graph) bootstrap, B = 500.
Conditions: ARM p = 0.05..0.95 and SNM eps = 0.05 (unstable-phase reference), 1000 core graphs.
Output: runs/kick/kick_growth.json.  Run from revision_r1/analyses.
"""
import json, math
import multiprocessing as mp
import numpy as np
from gcg_kick import simulate

T = 50000
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
TAUS = np.array([0.5, 1, 2, 3, 5, 10, 20, 50])
B = 500


def worker(args):
    model, par, seed = args
    rate = (1 - par) if model == 'ARM' else 1.0
    r = simulate(dict(seed=int(seed), node_num=20, color_num=3, step_max=T,
                      gridlock_rule='random' if model == 'ARM' else 'stay',
                      p_async=par if model == 'ARM' else 0.0, p_noise=par if model == 'SNM' else 0.0,
                      n_kicks=40, kick_wait=int(math.ceil(50 / rate)) if model == 'ARM' else 1,
                      kick_gap=int(math.ceil(1000 / rate))))
    C = r['history_conflict_nodes'].astype(float)
    idx = np.maximum(1, np.round(TAUS / rate).astype(int))          # steps after the kick
    rows = []
    for t0, c0 in zip(r['kick_times'], r['kick_c0']):
        if t0 + idx[-1] < T:
            rows.append(np.concatenate([[c0], C[t0 + idx]]))
    return np.array(rows) if rows else np.zeros((0, TAUS.size + 1))


def stats(M):
    c0 = M[:, 0].mean()
    c = M[:, 1:].mean(axis=0)
    with np.errstate(divide='ignore'):
        lam = np.log(c / c0) / TAUS
    surv = (M[:, 1:] > 0).mean(axis=0)
    cond = np.array([M[M[:, 1 + j] > 0, 1 + j].mean() if (M[:, 1 + j] > 0).any() else np.nan for j in range(TAUS.size)])
    return c0, c, lam, surv, cond


if __name__ == '__main__':
    res = {}
    conds = [('ARM', round(p, 2)) for p in np.arange(10) / 10 + 0.05] + [('SNM', 0.05)]
    with mp.get_context('fork').Pool(12) as pool:
        for model, par in conds:
            per = pool.map(worker, [(model, par, s) for s in SEEDS], chunksize=8)
            M = np.concatenate([m for m in per if m.size])
            c0, c, lam, surv, cond = stats(M)
            rng = np.random.default_rng([2024, 55, int(round(par * 100)), model == 'ARM'])
            bs = []
            for _ in range(B):
                pick = rng.integers(0, len(per), len(per))
                Mb = np.concatenate([per[i] for i in pick if per[i].size])
                bs.append(stats(Mb)[2])
            lo, hi = np.nanpercentile(np.array(bs), [2.5, 97.5], axis=0)
            res[f'{model}_{par:.2f}'] = dict(n_kicks=int(M.shape[0]), c0_mean=float(c0), taus=TAUS.tolist(),
                                             mean_c=c.tolist(), Lambda=lam.tolist(), Lambda_lo=lo.tolist(), Lambda_hi=hi.tolist(),
                                             survival=surv.tolist(), mean_c_survivors=cond.tolist())
            print(model, par, f'n={M.shape[0]} c0={c0:.2f} |', ' '.join(f'tau{t:g}: L={l:+.3f}[{a:+.3f},{b:+.3f}] S={s:.2f} c|s={q:.1f}'
                  for t, l, a, b, s, q in zip(TAUS, lam, lo, hi, surv, cond)), flush=True)
    json.dump(res, open('runs/kick/kick_growth.json', 'w'), indent=1)
