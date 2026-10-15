"""Times to the first solution against network size (R1-2), from the parameter scans of Fig. 6a-d.

Data: result_for_plateau_k34.pkl (BA m = 2, N = 20, 30, ..., 100, k = 3, 4, eps / p = 0, 0.05, ..., 1, 200 trials,
10,000 rounds).  No new simulation.  Unsolved trials are right-censored at the budget.  Because every censoring time
equals the budget, the Kaplan-Meier estimate below the budget is the empirical distribution of the solved times over
all trials, and a quantile q exists only where at least a fraction q of the trials are solved.  Medians carry 95%
intervals from 1,000 bootstrap replicates over trials; replicates in which the median is not reached are kept as
"above the budget", so an interval can be open at the top.  Sweeps: rounds for the SNM, rounds x (1 - p) for the ARM.
Times are counted as in run_first_passage.py (Fig. 3): T = solve_time + 1 rounds, solve_time being the 0-based index of
the first round with a proper colouring.

Output: runs/fpt_size/fpt_size.json.  Run from revision_r1/analyses.
"""
import json, os
import numpy as np
import pandas as pd

from repo_path import REPO
OUT = 'runs/fpt_size'
B = 1000
REP = {'SNM': 0.05, 'ARM': 0.45}


def quantile(t, solved, q, budget):
    """Smallest time at which the fraction solved reaches q (inf if it never does within the budget)."""
    ts = np.sort(t[solved == 1])
    need = int(np.ceil(q * len(t)))
    return float(ts[need - 1]) if len(ts) >= need else np.inf


def survival(t, solved, budget):
    """Step function of the fraction not yet solved: times and values just after each time."""
    ts = np.sort(t[solved == 1])
    u, c = np.unique(ts, return_counts=True)
    s = 1 - np.cumsum(c) / len(t)
    return [0] + u.tolist() + [budget], [1.0] + s.tolist() + [float(s[-1]) if len(s) else 1.0]


def main():
    df = pd.read_pickle(os.path.join(REPO, 'result_for_plateau_k34.pkl'))
    budget = int(df.step_max.max())
    rng = np.random.default_rng(7)
    cells = []
    for (label, k, N, pe, pa), g in df.groupby(['label', 'color_num', 'node_num', 'p_noise', 'p_async']):
        model = 'ARM' if label == 'Async' else 'SNM'
        par = float(pa if model == 'ARM' else pe)
        t, s = g.solve_time.to_numpy() + 1, g.solved.to_numpy()      # T = solve_time + 1, as in Fig. 3
        n = len(t)
        med = quantile(t, s, 0.5, budget)
        boot = np.array([quantile(t[i], s[i], 0.5, budget) for i in (rng.integers(0, n, n) for _ in range(B))])
        lo, hi = np.percentile(boot, [2.5, 97.5])       # inf where the median is not reached
        f = 1 - par if model == 'ARM' else 1.0          # rounds -> sweeps
        cells.append(dict(model=model, k=int(k), N=int(N), par=round(par, 2), n=int(n), solved=float(s.mean()),
                          median_rounds=med, ci_rounds=[float(lo), float(hi)],
                          q25_rounds=quantile(t, s, 0.25, budget), q75_rounds=quantile(t, s, 0.75, budget),
                          median_sweeps=med * f, ci_sweeps=[float(lo) * f, float(hi) * f]))
    curves = {}
    for model, par in REP.items():
        for k in (3, 4):
            for N in sorted(df.node_num.unique()):
                g = df[(df.label == ('Async' if model == 'ARM' else 'Noise')) & (df.color_num == k) & (df.node_num == N)
                       & np.isclose(df.p_async if model == 'ARM' else df.p_noise, par)]
                curves[f'{model}_k{k}_N{N}'] = survival(g.solve_time.to_numpy() + 1, g.solved.to_numpy(), budget)
    os.makedirs(OUT, exist_ok=True)
    clean = lambda x: None if isinstance(x, float) and not np.isfinite(x) else x
    cells = [{key: ([clean(v) for v in val] if isinstance(val, list) else clean(val)) for key, val in c.items()} for c in cells]
    json.dump(dict(budget=budget, B=B, representative=REP, cells=cells, survival=curves), open(f'{OUT}/fpt_size.json', 'w'))
    for model, par in REP.items():
        for k in (3, 4):
            row = [c for c in cells if c['model'] == model and c['k'] == k and c['par'] == par]
            print(model, k, [(c['N'], c['median_rounds'], c['ci_rounds'], round(c['solved'], 3)) for c in row])


if __name__ == '__main__':
    main()
