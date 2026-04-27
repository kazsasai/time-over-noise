"""Run scalability simulation → result_for_scalability.pkl"""
import multiprocessing as mp
import platform

# Windows does not support 'fork'; use platform-safe start method
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
import gcg_solver

master_seed = 2024
num_trials  = 200
step_max    = 5000

rng   = np.random.default_rng(master_seed)
seeds = rng.integers(0, 10**9, size=num_trials)

n_list = [20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500]
c_list = np.arange(2, 51)

models = [
    {'label': 'Noise', 'p_async': 0.0, 'p_noise': 0.05, 'is_async_model': False},
    {'label': 'Async', 'p_async': 0.45, 'p_noise': 0.0,  'is_async_model': True},
]

if __name__ == '__main__':
    tasks = []
    for n in n_list:
        for c in c_list:
            for model_conf in models:
                for s in seeds:
                    p = {
                        'seed': int(s), 'node_num': int(n), 'color_num': int(c),
                        'ba_m': 2, 'bot_num': int(n), 'bot_type': 'c',
                        'step_max': step_max, 'return_history': False,
                    }
                    p.update(model_conf)
                    tasks.append(p)

    print(f"Total tasks : {len(tasks):,}")
    print(f"N values    : {n_list}")
    print(f"k range     : {c_list[0]}..{c_list[-1]}  ({len(c_list)} values)")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle("./result_for_scalability.pkl")
    print(f"\nSaved: result_for_scalability.pkl  (rows={len(df):,})")
