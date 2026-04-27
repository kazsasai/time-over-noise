"""
Phase-space sweep at N=20, k=3 used in Fig. 3 (universality).

Design:
  N        : 20
  k        : 3   (chromatic boundary for BA m=2)
  p values : 100 values (0.00, 0.01, ..., 0.99)
  trials   : 1,000 per (model, p)
  step_max : 1,000
  Both SNM (Noise Model) and ARM (Async Model) are swept over p.

Output: result_for_phase_space_20_3.pkl
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
P_LIST      = np.arange(100) / 100

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
    df.to_pickle('./result_for_phase_space_20_3.pkl')
    print(f"Saved: result_for_phase_space_20_3.pkl  (rows={len(df):,})")
