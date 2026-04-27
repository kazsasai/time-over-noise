"""
Full 2D (p_async, p_noise) grid at N=20, k=3.

Design:
  N          : 20
  k          : 3
  p_async    : 20 values (0.00, 0.05, ..., 0.95)
  p_noise    : 20 values (0.00, 0.05, ..., 0.95)
  trials     : 1,000 per (model, p_async, p_noise)
  step_max   : 1,000
  Used in the supplementary hybrid / static-random comparison.

Output: result_static_random_20_3.pkl
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver


MASTER_SEED = 2024
NUM_TRIALS  = 1000
STEP_MAX    = 1000
NODE_NUM    = 20
COLOR_NUM   = 3
P_LIST      = np.arange(20) / 20

MODELS = [
    {'label': 'Static Model', 'p_async': 0.0, 'p_noise': 0.0, 'is_async_model': False},
    {'label': 'Random Model', 'p_async': 0.0, 'p_noise': 0.0, 'is_async_model': True},
]


if __name__ == '__main__':
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(0, 10**9, size=NUM_TRIALS)

    tasks = []
    for p in P_LIST:
        for q in P_LIST:
            for model_conf in MODELS:
                model_conf['p_async'] = float(p)
                model_conf['p_noise'] = float(q)
                for s in seeds:
                    param = {
                        'seed':           int(s),
                        'node_num':       NODE_NUM,
                        'color_num':      COLOR_NUM,
                        'ba_m':           2,
                        'bot_num':        NODE_NUM,
                        'bot_type':       'c',
                        'step_max':       STEP_MAX,
                        'return_history': False,
                    }
                    param.update(model_conf)
                    tasks.append(param)

    print(f"Total tasks : {len(tasks):,}")
    df = gcg_solver.run_parallel_simulation(tasks)
    df.to_pickle('./result_static_random_20_3.pkl')
    print(f"Saved: result_static_random_20_3.pkl  (rows={len(df):,})")
