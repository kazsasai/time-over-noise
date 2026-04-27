"""
N=40, k=3, ARM only — IBI + B&P analysis
step_max = 100,000  (intermediate: N=20→50k, N=50→200k, N=40→100k)
200 trials per p value
p values: 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75

Output: result_criticality_N40_k3.pkl
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver

MASTER_SEED = 4444
N_TRIALS    = 200
NODE_NUM    = 40
COLOR_NUM   = 3
STEP_MAX    = 100_000
P_VALUES    = np.round([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75], 4)

rng   = np.random.default_rng(MASTER_SEED)
seeds = rng.integers(0, 10**9, size=N_TRIALS)

if __name__ == '__main__':
    tasks = []
    for p in P_VALUES:
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
                'label':          'Async Model',
                'is_async_model': True,
                'p_noise':        0.0,
                'p_async':        float(p),
            })

    total = len(tasks)
    print(f"Total tasks : {total:,}")
    print(f"N={NODE_NUM}  k={COLOR_NUM}  step_max={STEP_MAX:,}  trials={N_TRIALS}")
    print(f"p values    : {P_VALUES.tolist()}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle('./result_criticality_N40_k3.pkl')
    print(f"\nSaved: result_criticality_N40_k3.pkl  (rows={len(df):,})")
