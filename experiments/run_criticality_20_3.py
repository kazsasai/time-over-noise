"""
Long records at N=20, k=3 (core data; see revision_r1/data/README.md) → result_for_criticality_20_3.pkl

Design (matches the Methods: 1,000 trials, 50,000 steps each):
  N        : 20
  k        : 3   (chromatic boundary for BA m=2)
  p values : 10 values (0.05, 0.15, ..., 0.95), for both SNM (p = ε) and ARM (p = freezing prob.)
  trials   : 1,000 per (model, p)
  step_max : 50,000
  history  : True (per-step changed-node count, conflict ratio, gridlock count)
  Total tasks: 10 × 2 models × 1,000 = 20,000

Note: P_LIST is generated as `np.arange(10) / 10 + 0.05` (not rounded) so that the
stored float values match those used by the figure scripts.
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
NODE_NUM    = 20
COLOR_NUM   = 3
P_LIST      = np.arange(10) / 10 + 0.05

MODELS = [
    {'label': 'Noise Model', 'is_async_model': False},
    {'label': 'Async Model', 'is_async_model': True},
]


def build_tasks():
    rng   = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(0, 10**9, size=NUM_TRIALS)
    tasks = []
    for p in P_LIST:
        for model in MODELS:
            is_async = model['is_async_model']
            for s in seeds:
                tasks.append({
                    'seed':           int(s),
                    'node_num':       NODE_NUM,
                    'color_num':      COLOR_NUM,
                    'ba_m':           2,
                    'bot_num':        NODE_NUM,
                    'bot_type':       'c',
                    'step_max':       STEP_MAX,
                    'return_history': True,
                    'label':          model['label'],
                    'is_async_model': is_async,
                    'p_noise':        float(p) if not is_async else 0.0,
                    'p_async':        float(p) if is_async     else 0.0,
                })
    return tasks


if __name__ == '__main__':
    tasks = build_tasks()
    print(f"Total tasks  : {len(tasks):,}")
    print(f"N={NODE_NUM}, k={COLOR_NUM}  (chromatic boundary for BA m=2)")
    print(f"p values     : {P_LIST.tolist()}")
    print(f"Trials       : {NUM_TRIALS}  |  step_max: {STEP_MAX:,}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle("./result_for_criticality_20_3.pkl")
    print(f"\nSaved: result_for_criticality_20_3.pkl  (rows={len(df):,})")
