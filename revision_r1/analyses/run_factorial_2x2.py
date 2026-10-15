"""
2x2 factorial (Fig 3) with Wilson CIs on the core seeds (master seed 2024, 1000 graphs), 1,000 rounds, N=20, k=3,
eps = 0: timing (synchronous p=0 / asynchronous p=0.45) x gridlock rule (stay / random).  Uses gcg_variants, which
reproduces gcg_solver bit for bit.  Output: runs/fpt/factorial_2x2.json.
"""
import json
import numpy as np
from gcg_variants import run_tasks
from run_first_passage import wilson

SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
CONDS = [('Sync+Stay (SNM, eps=0)', 0.0, 'stay'), ('Async+Stay', 0.45, 'stay'),
         ('Sync+Random', 0.0, 'random'), ('Async+Random (ARM)', 0.45, 'random')]
if __name__ == '__main__':
    out = {}
    for name, p, rule in CONDS:
        tasks = [dict(seed=int(s), node_num=20, color_num=3, step_max=1000, schedule='bernoulli', gridlock_rule=rule,
                      p_async=p, p_noise=0.0) for s in SEEDS]
        df = run_tasks(tasks, n_jobs=12)
        k = int(df['solved'].sum()); n = len(df)
        out[name] = dict(p_async=p, rule=rule, n=n, solved=k, rate=k / n, ci=list(wilson(k, n)))
        print(name, out[name], flush=True)
    json.dump(out, open('runs/fpt/factorial_2x2.json', 'w'), indent=1)
