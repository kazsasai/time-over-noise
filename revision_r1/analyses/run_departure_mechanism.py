"""How does the ARM leave a proper colouring?  (k = 3; Methods section 6)

Claim (follows from the update rule): starting from a proper colouring, a conflict on edge (u, v) after one round
requires that u and v both change colour in that round, to the same colour, and that each had at least two
conflict-free colours.  With k = 3 this means N(u)\\{v} all hold v's colour and N(v)\\{u} all hold u's colour
("two-coloured edge"), so (u, v) has no common neighbour.  A single change never creates a conflict.

This script checks the claim on the core graphs and tests a structural prediction: a graph in which no proper
3-colouring has a two-coloured edge can never leave the solved state.

Re-simulates the core long records (same 1,000 seeds as run_kick.py, bit-identical without kicks) with colour
history.  Output: runs/departure/departure_<p>.json and departure_summary.json.
Usage (from revision_r1/analyses):  python3 run_departure_mechanism.py [--ps 0.05,0.45,0.85] [--T 50000]
"""
import argparse, json, os, itertools
import multiprocessing as mp
import numpy as np
import networkx as nx
from gcg_kick import simulate

OUT = 'runs/departure'
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
K = 3


def two_coloured_edge_possible(g):
    """True if some proper 3-colouring of g has an edge (u, v) with N(u)\\{v} all coloured like v and N(v)\\{u}
    all coloured like u.  Exact: for each candidate edge, fix those colours and search for a proper extension."""
    nodes = list(g.nodes())
    adj = {u: set(g[u]) for u in nodes}
    order = sorted(nodes, key=lambda x: -len(adj[x]))
    for u, v in g.edges():
        for a, b in ((u, v), (v, u)):
            if adj[a] & adj[b]:
                continue                      # a common neighbour cannot hold both colours
            fixed = {a: 0, b: 1}
            ok = True
            for w in adj[a] - {b}:
                if fixed.get(w, 1) != 1: ok = False
                fixed[w] = 1
            for w in adj[b] - {a}:
                if fixed.get(w, 0) != 0: ok = False
                fixed[w] = 0
            if not ok or any(fixed[x] == fixed[y] for x in fixed for y in adj[x] if y in fixed):
                continue
            if _extend(order, adj, dict(fixed)):
                return True
    return False


def _extend(order, adj, col):
    free = [x for x in order if x not in col]
    def bt(i):
        if i == len(free):
            return True
        x = free[i]
        used = {col[y] for y in adj[x] if y in col}
        for c in range(K):
            if c not in used:
                col[x] = c
                if bt(i + 1):
                    return True
                del col[x]
        return False
    return bt(0)


def one(args):
    seed, p, T = args
    r = simulate(dict(seed=int(seed), node_num=20, color_num=K, ba_m=2, bot_num=20, bot_type='c', step_max=T,
                      gridlock_rule='random', p_async=p, p_noise=0.0, record_colors=True))
    g = nx.barabasi_albert_graph(20, 2, seed=int(seed))
    adj = [list(g[i]) for i in range(20)]
    deg = np.array([len(a) for a in adj])
    C, H = r['history_conflict_nodes'], r['history_colors'].astype(np.int64)
    solved = C == 0
    first = int(np.argmax(solved)) if solved.any() else -1
    out = dict(seed=int(seed), first=first, departures=0, structural=bool(two_coloured_edge_possible(g)),
               checked=0, violations=0, pair_degrees=[], n_conflict_edges=[])
    if first < 0:
        return out
    dep = np.flatnonzero(solved[first:-1] & ~solved[first + 1:]) + first + 1      # round t: proper at t-1, not at t
    out['departures'] = int(dep.size)
    for t in dep[:20]:                                                               # inspect up to 20 per trial
        prev, now = H[t - 1], H[t]
        bad = [(u, v) for u in range(20) for v in adj[u] if u < v and now[u] == now[v]]
        out['n_conflict_edges'].append(len(bad))
        for u, v in bad:
            out['checked'] += 1
            free_u = {c for c in range(1, K + 1) if c not in {prev[w] for w in adj[u]}}
            free_v = {c for c in range(1, K + 1) if c not in {prev[w] for w in adj[v]}}
            ok = (prev[u] != now[u]) and (prev[v] != now[v]) and len(free_u) >= 2 and len(free_v) >= 2 \
                and not (set(adj[u]) & set(adj[v]))
            out['violations'] += int(not ok)
            out['pair_degrees'].append(sorted([int(deg[u]), int(deg[v])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ps', default='0.05,0.45,0.85')
    ap.add_argument('--T', type=int, default=50000)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for p in [float(x) for x in a.ps.split(',')]:
        with mp.get_context('spawn').Pool(a.workers) as pool:
            res = pool.map(one, [(s, p, a.T) for s in SEEDS], chunksize=8)
        json.dump(res, open(f'{OUT}/departure_{p:.2f}.json', 'w'))
        solvedr = [x for x in res if x['first'] >= 0]
        never = np.array([x['departures'] == 0 for x in solvedr])
        struct = np.array([x['structural'] for x in solvedr])
        pd_ = [tuple(d) for x in solvedr for d in x['pair_degrees']]
        summary[f'{p:.2f}'] = dict(
            n_solved=len(solvedr), share_never_leave=float(never.mean()),
            share_structural=float(struct.mean()),
            never_leave_if_not_structural=float(never[~struct].mean()) if (~struct).any() else None,
            never_leave_if_structural=float(never[struct].mean()) if struct.any() else None,
            n_not_structural=int((~struct).sum()),
            conflict_edges_checked=int(sum(x['checked'] for x in solvedr)),
            violations=int(sum(x['violations'] for x in solvedr)),
            share_departures_one_edge=float(np.mean([n == 1 for x in solvedr for n in x['n_conflict_edges']])),
            pair_min_degree_counts={str(d): int(sum(1 for q in pd_ if q[0] == d)) for d in sorted({q[0] for q in pd_})},
            pair_max_degree_median=float(np.median([q[1] for q in pd_])) if pd_ else None)
        print(f'p={p:.2f}', summary[f'{p:.2f}'], flush=True)
    json.dump(summary, open(f'{OUT}/departure_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
