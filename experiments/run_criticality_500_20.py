"""
Large-scale criticality sweep at N=500, k=20.

Design:
  N        : 500
  k        : 20
  p values : 10 values (0.05, 0.15, ..., 0.95)
  trials   : 1,000 per (model, p)
  step_max : 50,000

Output: result_for_criticality_500_20.pkl
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
STEP_MAX    = 50000
NODE_NUM    = 500
COLOR_NUM   = 20
P_LIST      = (np.arange(10) / 10) + 0.05

MODELS = [
    {'label': 'Noise Model', 'p_async': 0.0, 'p_noise': 0.06, 'is_async_model': False},
    {'label': 'Async Model', 'p_async': 0.45, 'p_noise': 0.0, 'is_async_model': True},
]


if __name__ == '__main__':
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(0, 10**9, size=NUM_TRIALS)

    tasks = []
    for p in P_LIST:
        for model_conf in MODELS:
            if model_conf['label'] == 'Async Model':
                model_conf['p_async'] = float(p)
            else:
                model_conf['p_noise'] = float(p)
            for s in seeds:
                param = {
                    'seed':           int(s),
                    'node_num':       NODE_NUM,
                    'color_num':      COLOR_NUM,
                    'ba_m':           2,
                    'bot_num':        NODE_NUM,
                    'bot_type':       'c',
                    'step_max':       STEP_MAX,
                    'return_history': True,
                }
                param.update(model_conf)
                tasks.append(param)

    print(f"Total tasks : {len(tasks):,}")
    df = gcg_solver.run_parallel_simulation(tasks)
    df.to_pickle('./result_for_criticality_500_20.pkl')
    print(f"Saved: result_for_criticality_500_20.pkl  (rows={len(df):,})")
