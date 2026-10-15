"""Exact 3-colourability of the graphs used in the network-ensemble comparison (Supplementary Fig. S12).
Graphs are built exactly as gcg_solver does (BA m=2; ER q=2m/(N-1), regenerated until connected; WS k=4, beta=0.3)
with the 1,000 core seeds.  Output: runs/supp/colourability.json.  Run from revision_r1/analyses."""
import json
import numpy as np
import networkx as nx

SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
N, M = 20, 2


def build(kind, seed):
    if kind == 'ba':
        return nx.barabasi_albert_graph(N, M, seed=seed)
    if kind == 'er':
        q = 2 * M / (N - 1)
        g = nx.erdos_renyi_graph(N, q, seed=seed)
        while not nx.is_connected(g):
            seed += 1
            g = nx.erdos_renyi_graph(N, q, seed=seed)
        return g
    return nx.watts_strogatz_graph(N, max(4, 2 * M), 0.3, seed=seed)


def colourable(g, k=3):
    adj = {u: set(g[u]) for u in g}
    col = {}
    def pick():                                        # DSATUR: most constrained uncoloured node first
        best, key = None, None
        for u in adj:
            if u in col: continue
            kk = (len({col[v] for v in adj[u] if v in col}), len(adj[u]))
            if key is None or kk > key: best, key = u, kk
        return best
    def bt():
        u = pick()
        if u is None: return True
        used = {col[v] for v in adj[u] if v in col}
        for c in range(k):
            if c not in used:
                col[u] = c
                if bt(): return True
                del col[u]
        return False
    return bt()


out = {}
for kind in ('ba', 'er', 'ws'):
    ok = [colourable(build(kind, int(s))) for s in SEEDS]
    out[kind] = dict(share_3colourable=float(np.mean(ok)), n=len(ok))
    print(kind, out[kind], flush=True)
json.dump(out, open('runs/supp/colourability.json', 'w'), indent=1)
