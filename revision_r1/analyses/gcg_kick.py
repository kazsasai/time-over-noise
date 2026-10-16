"""
Bernoulli-schedule solver (SNM / ARM) with two additions that use NO random numbers, so that a run without kicks
reproduces gcg_variants / gcg_solver bit for bit:
  - colour history: int8 array (steps x N), for per-agent spectra (Uragami & Gunji 2018 protocol);
  - kicks: after the update of step t, if the colouring has been solved for at least `kick_wait` consecutive steps
    and t >= next_kick, one conflict is injected: node u of the next pre-drawn edge (u, v) takes v's colour.
    The next kick is allowed `kick_gap` steps later.  Kick edges are drawn beforehand from a separate generator.
Validated by check_kick_bitidentity() against the stored core data.
"""
import numpy as np
import networkx as nx
from numba import njit
from gcg_variants import _decide, _n_conflict_nodes


@njit(cache=True)
def _run_k(n, k, colors, adj_indices, adj_indptr, eu, ev, bot_pos, step_max, rule, p_async, p_noise, rng,
           hist_changed, hist_conf, hist_colors, kick_u, kick_v, kick_wait, kick_gap, kick_times,
           kick_ufree, kick_vfree, kick_c0):
    rec_col = hist_colors.shape[0] > 0
    solved_run = 0
    next_kick = 0
    kidx = 0
    n_kicks = 0
    for t in range(step_max):
        cand = colors.copy()
        for i in range(n):
            c, g = _decide(i, k, colors, adj_indices, adj_indptr, rule, 1.0, rng)
            cand[i] = c
        for b in range(bot_pos.shape[0]):
            if rng.random() < p_noise:
                cand[bot_pos[b]] = rng.integers(1, k + 1)
        act = rng.random(n) < 1 - p_async
        changed = 0
        for i in range(n):
            if act[i]:
                if cand[i] != colors[i]:
                    changed += 1
                colors[i] = cand[i]
        n_conf = _n_conflict_nodes(colors, eu, ev, n)
        hist_changed[t] = changed
        hist_conf[t] = n_conf
        if rec_col:
            for i in range(n):
                hist_colors[t, i] = colors[i]
        solved_run = solved_run + 1 if n_conf == 0 else 0
        if kick_u.shape[0] > 0 and kidx < kick_u.shape[0] and solved_run >= kick_wait and t >= next_kick:
            u = kick_u[kidx]; v = kick_v[kidx]
            colors[u] = colors[v]
            kick_times[n_kicks] = t
            kick_c0[n_kicks] = _n_conflict_nodes(colors, eu, ev, n)     # conflicting nodes right after the kick
            # number of conflict-free colours left for u and for v right after the kick (no RNG used)
            for side in range(2):
                x = u if side == 0 else v
                used = np.zeros(k + 1, dtype=np.int64)
                for j in range(adj_indptr[x], adj_indptr[x + 1]):
                    used[colors[adj_indices[j]]] += 1
                nf = 0
                for c in range(1, k + 1):
                    if used[c] == 0:
                        nf += 1
                if side == 0:
                    kick_ufree[n_kicks] = nf
                else:
                    kick_vfree[n_kicks] = nf
            n_kicks += 1
            kidx += 1
            next_kick = t + kick_gap
            solved_run = 0
    return n_kicks


def simulate(task):
    seed, n, k = task['seed'], task['node_num'], task['color_num']
    net = nx.barabasi_albert_graph(n, task.get('ba_m', 2), seed=seed)
    adj = nx.adjacency_matrix(net).tocsr()
    edges = np.array(net.edges())
    eu, ev = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    degrees = np.array([net.degree()[i] for i in range(n)])
    bot_pos = np.argsort(degrees)[-task.get('bot_num', n):]
    rng = np.random.default_rng(seed)
    colors = rng.integers(1, k + 1, size=n)
    T = task['step_max']
    hc = np.zeros(T, dtype=np.int16)
    hk = np.zeros(T, dtype=np.int16)
    hcol = np.zeros((T, n) if task.get('record_colors') else (0, n), dtype=np.int8)
    nk = int(task.get('n_kicks', 0))
    if nk:
        kr = np.random.default_rng([seed, 777])
        e = kr.integers(0, eu.size, nk)
        side = kr.random(nk) < 0.5
        ku = np.where(side, eu[e], ev[e]).astype(np.int64)
        kv = np.where(side, ev[e], eu[e]).astype(np.int64)
    else:
        ku = kv = np.zeros(0, dtype=np.int64)
    ktimes = np.full(max(nk, 1), -1, dtype=np.int64)
    kuf = np.full(max(nk, 1), -1, dtype=np.int64)
    kvf = np.full(max(nk, 1), -1, dtype=np.int64)
    kc0 = np.full(max(nk, 1), -1, dtype=np.int64)
    rule = {'stay': 0, 'random': 1}[task['gridlock_rule']]
    nkicks = _run_k(n, k, colors, adj.indices, adj.indptr, eu, ev, bot_pos, T, rule,
                    float(task.get('p_async', 0.0)), float(task.get('p_noise', 0.0)), rng,
                    hc, hk, hcol, ku, kv, int(task.get('kick_wait', 1)), int(task.get('kick_gap', 1)), ktimes, kuf, kvf, kc0)
    out = dict(task)
    out.update(history_changed=hc, history_conflict_nodes=hk, kick_times=ktimes[:nkicks],
               kick_ufree=kuf[:nkicks], kick_vfree=kvf[:nkicks], kick_c0=kc0[:nkicks])
    if task.get('record_colors'):
        out['history_colors'] = hcol
    return out


def check_kick_bitidentity(n_check=3):
    """Without kicks, A(t) and C(t) must equal the stored core data for the same seeds."""
    import pandas as pd
    seeds = np.random.default_rng(2024).integers(0, 10**9, size=1000)
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    ok = True
    for lab, model, par, rule in (('Async Model', 'ARM', 0.45, 'random'), ('Noise Model', 'SNM', 0.05, 'stay')):
        for s in seeds[:n_check]:
            task = dict(seed=int(s), node_num=20, color_num=3, step_max=50000, gridlock_rule=rule,
                        p_async=par if model == 'ARM' else 0.0, p_noise=par if model == 'SNM' else 0.0,
                        record_colors=True)
            r = simulate(task)
            col = 'p_async' if model == 'ARM' else 'p_noise'
            row = df[(df['label'] == lab) & (df['seed'] == int(s)) & ((df[col] - par).abs() < 1e-9)].iloc[0]
            same = (np.array_equal(r['history_changed'], np.asarray(row['history_changed'])) and
                    np.array_equal(r['history_conflict_nodes'] == 0, np.asarray(row['history_conflict_ratio']) < 1e-12))
            rec_A = np.concatenate([[np.nan], (np.diff(r['history_colors'].astype(np.int16), axis=0) != 0).sum(axis=1)])
            same_col = np.array_equal(rec_A[1:], r['history_changed'][1:])
            ok &= same and same_col
            print(model, s, 'A(t),C(t) identical to core:', same, '| A(t) recovered from colours:', same_col)
    return ok


if __name__ == '__main__':
    print('ALL OK' if check_kick_bitidentity() else 'MISMATCH')
