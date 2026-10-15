"""
Step-budget extension quoted in the Results (SNM at N = 125, k = 16; ARM at N = 400, k = 28, 29), with the scalability design
(master_seed 2024, 200 trials, BA m=2, SNM eps=0.05, ARM p=0.45) and step_max = 50,000 (10x the
5,000-step heatmap budget). Solution rates at 5,000 / 10,000 / 50,000 steps are read off the
first-passage time. Run from revision_r1/analyses/runs/extension with the repository root on PYTHONPATH
(PYTHONPATH=../../../..)
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import numpy as np, pandas as pd
import gcg_solver

SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=200)
MODELS = [('Noise', False, 0.05, 0.0), ('Async', True, 0.0, 0.45)]
COND = [(125, 16), (400, 28), (400, 29)]
if __name__ == '__main__':
    tasks = [dict(seed=int(s), node_num=n, color_num=k, ba_m=2, bot_num=n, bot_type='c', step_max=50000,
                  return_history=False, label=lab, is_async_model=ia, p_noise=pn, p_async=pa)
             for n, k in COND for lab, ia, pn, pa in MODELS for s in SEEDS]
    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle('budget_extension.pkl')
    rows = []
    for (lab, n, k), g in df.groupby(['label', 'node_num', 'color_num']):
        t = g['solve_time'].values; s = g['solved'].values == 1
        rows.append(dict(model=lab, N=n, k=k, rate_5000=np.mean(s & (t < 5000)), rate_10000=np.mean(s & (t < 10000)),
                         rate_50000=np.mean(s & (t < 50000)), trials=len(g)))
    out = pd.DataFrame(rows)
    out.to_csv('budget_extension_summary.csv', index=False)
    print(out.to_string(index=False))
