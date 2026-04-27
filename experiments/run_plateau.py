"""
Run plateau robustness simulation → result_for_plateau.pkl

Goal:
  For each network size N, sweep the model parameter (p_noise or p_async)
  across 21 values in [0, 1] and record the solve rate.
  The number of colors k is set to ceil(sqrt(2*N)) for each N,
  which equals the expected maximum degree of a BA network with m=2.

Design:
  N      : 6 values (log-spaced, matching fig5 range)
  k      : ceil(sqrt(2*N))  [per N]
  p/ε    : 21 values  (0.00, 0.05, 0.10, ..., 1.00)
  trials : 200
  step_max: 5000
  Total tasks: 6 × 21 × 2 models × 200 = 50,400
"""

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

master_seed = 2025
num_trials  = 200
step_max    = 5000

rng   = np.random.default_rng(master_seed)
seeds = rng.integers(0, 10**9, size=num_trials)

# N values: 6 points (round numbers, matching fig5 range)
n_list = [20, 50, 100, 200, 300, 500]

# Parameter sweep: 21 points in [0, 1]
param_vals = np.round(np.linspace(0, 1, 21), 4).tolist()

if __name__ == '__main__':
    tasks = []

    for n in n_list:
        k = int(np.ceil(np.sqrt(2 * n)))  # theoretical minimum colors

        for param in param_vals:
            # ── SNM: sweep p_noise, p_async fixed at 0 ───────────────────
            for s in seeds:
                tasks.append({
                    'seed': int(s), 'node_num': int(n), 'color_num': k,
                    'ba_m': 2, 'bot_num': int(n), 'bot_type': 'c',
                    'step_max': step_max, 'return_history': False,
                    'label': 'Noise',
                    'p_noise': float(param), 'p_async': 0.0,
                    'is_async_model': False,
                })

            # ── ARM: sweep p_async, p_noise fixed at 0 ───────────────────
            for s in seeds:
                tasks.append({
                    'seed': int(s), 'node_num': int(n), 'color_num': k,
                    'ba_m': 2, 'bot_num': int(n), 'bot_type': 'c',
                    'step_max': step_max, 'return_history': False,
                    'label': 'Async',
                    'p_noise': 0.0, 'p_async': float(param),
                    'is_async_model': True,
                })

    print(f"Total tasks  : {len(tasks):,}")
    print(f"N values     : {n_list}")
    print(f"k per N      : {[int(np.ceil(np.sqrt(2*n))) for n in n_list]}")
    print(f"param values : {param_vals}")
    print(f"Trials       : {num_trials}  |  step_max: {step_max}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle("./result_for_plateau.pkl")
    print(f"\nSaved: result_for_plateau.pkl  (rows={len(df):,})")
