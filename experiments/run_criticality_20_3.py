"""
Run criticality simulation at N=20, k=3 with longer time series → result_for_criticality_20_3.pkl

Goal:
  Improve FFT quality by using longer time series (100,000 steps).
  The post-solve dynamics at N=20, k=3 show re-conflict events that need
  longer observation windows for clear 1/f characterization.

Design:
  N        : 20
  k        : 3  (≈ χ(G) for BA m=2 → chromatic boundary)
  p values : 11 values (0.00, 0.05, 0.15, ..., 0.95)
  trials   : 500 per (model, p)
  step_max : 100000
  history  : True
  Total tasks: 11 × 2 models × 500 = 11,000
"""

import multiprocessing as mp
import platform

if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver

master_seed = 2030
num_trials  = 500
step_max    = 100000
node_num    = 20
color_num   = 3

p_values = np.round([0.00, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], 4)

rng   = np.random.default_rng(master_seed)
seeds = rng.integers(0, 10**9, size=num_trials)

models = [
    {'label': 'Noise Model', 'is_async_model': False},
    {'label': 'Async Model', 'is_async_model': True},
]

if __name__ == '__main__':
    tasks = []

    for model_conf in models:
        is_async = model_conf['is_async_model']
        for p in p_values:
            for s in seeds:
                task = {
                    'seed': int(s),
                    'node_num': node_num,
                    'color_num': color_num,
                    'ba_m': 2,
                    'bot_num': node_num,
                    'bot_type': 'c',
                    'step_max': step_max,
                    'return_history': True,
                    'label': model_conf['label'],
                    'is_async_model': is_async,
                    'p_noise': float(p) if not is_async else 0.0,
                    'p_async': float(p) if is_async     else 0.0,
                }
                tasks.append(task)

    print(f"Total tasks  : {len(tasks):,}")
    print(f"N={node_num}, k={color_num}  (chromatic boundary for BA m=2)")
    print(f"p values     : {p_values.tolist()}")
    print(f"Trials       : {num_trials}  |  step_max: {step_max:,}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle("./result_for_criticality_20_3.pkl")
    print(f"\nSaved: result_for_criticality_20_3.pkl  (rows={len(df):,})")
