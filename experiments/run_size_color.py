"""
Coarse size × colour sweep.

Design:
  N values : [20, 50, 100, 200, 500]
  k values : [3, 5, 10, 20, 50]
  trials   : 1,000 per (model, N, k)
  step_max : 10,000
  Fixed parameters: SNM ε=0.05, ARM p_async=0.45.

Output: result_for_size_color.pkl
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
STEP_MAX    = 10000
N_LIST      = [20, 50, 100, 200, 500]
C_LIST      = [3, 5, 10, 20, 50]

MODELS = [
    {'label': 'Noise', 'p_async': 0.0,  'p_noise': 0.05, 'is_async_model': False},
    {'label': 'Async', 'p_async': 0.45, 'p_noise': 0.0,  'is_async_model': True},
]


if __name__ == '__main__':
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(0, 10**9, size=NUM_TRIALS)

    tasks = []
    for n in N_LIST:
        for c in C_LIST:
            for model_conf in MODELS:
                for s in seeds:
                    param = {
                        'seed':           int(s),
                        'node_num':       n,
                        'color_num':      c,
                        'ba_m':           2,
                        'bot_num':        n,
                        'bot_type':       'c',
                        'step_max':       STEP_MAX,
                        'return_history': False,
                    }
                    param.update(model_conf)
                    tasks.append(param)

    print(f"Total tasks : {len(tasks):,}")
    df = gcg_solver.run_parallel_simulation(tasks)
    df.to_pickle('./result_for_size_color.pkl')
    print(f"Saved: result_for_size_color.pkl  (rows={len(df):,})")
