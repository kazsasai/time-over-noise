"""
Run criticality simulation for a large, unsolvable setting → result_for_criticality_500_3.pkl

Goal:
  Test whether ARM exhibits SOC signatures (1/f spectrum, power-law avalanches)
  even in a frustrated configuration where no global coloring solution exists.

  N=500, k=3:  sqrt(2*500) ≈ 31.6  >>  k=3  →  provably unsolvable by any model.
  This isolates the intrinsic dynamical criticality of ARM from solution-finding success.

Design (matching result_for_criticality_20_3.pkl format):
  N        : 500
  k        : 3
  p values : 11 values (0.00, 0.05, 0.15, ..., 0.95)
  trials   : 30 per (model, p)   [fewer than N=20 case; N=500 gives rich per-trial stats]
  step_max : 10000               [good FFT resolution; freq range 10^-4 to 0.5]
  history  : True                [required for FFT and avalanche analysis]
  Total tasks: 11 × 2 models × 30 = 660
"""

import multiprocessing as mp
import platform

if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
import gcg_solver

master_seed = 2026
num_trials  = 30
step_max    = 10000
node_num    = 500
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
    print(f"N={node_num}, k={color_num}  (sqrt(2N) ≈ {np.sqrt(2*node_num):.1f}  >> k → unsolvable)")
    print(f"p values     : {p_values.tolist()}")
    print(f"Trials       : {num_trials}  |  step_max: {step_max}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle("./result_for_criticality_500_3.pkl")
    print(f"\nSaved: result_for_criticality_500_3.pkl  (rows={len(df):,})")
