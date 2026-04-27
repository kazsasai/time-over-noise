"""
Multi-N comparison at k=3 (ARM only)
k=3 is fixed by the theorem k >= m+1 (BA m=2)

N values : 20, 50, 100
k         : 3  (chromatic boundary)
p values  : 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75
trials    : 300 per (N, p)
step_max  : 50000
ARM only  (is_async_model=True)

Output: result_criticality_multiN_k3.pkl
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver

MASTER_SEED = 4242
N_TRIALS    = 300
STEP_MAX    = 50000
COLOR_NUM   = 3   # k = m+1 = 3
N_VALUES    = [20, 50, 100]
P_VALUES    = np.round([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75], 4)

rng   = np.random.default_rng(MASTER_SEED)
seeds = rng.integers(0, 10**9, size=N_TRIALS)

if __name__ == '__main__':
    tasks = []
    for node_num in N_VALUES:
        for p in P_VALUES:
            for s in seeds:
                tasks.append({
                    'seed':           int(s),
                    'node_num':       node_num,
                    'color_num':      COLOR_NUM,
                    'ba_m':           2,
                    'bot_num':        node_num,
                    'bot_type':       'c',
                    'step_max':       STEP_MAX,
                    'return_history': True,
                    'label':          'Async Model',
                    'is_async_model': True,
                    'p_noise':        0.0,
                    'p_async':        float(p),
                })

    print(f"Total tasks : {len(tasks):,}")
    print(f"N values    : {N_VALUES}")
    print(f"k           : {COLOR_NUM}  (= m+1, chromatic boundary for BA m=2)")
    print(f"p values    : {P_VALUES.tolist()}")
    print(f"Trials      : {N_TRIALS}  |  step_max: {STEP_MAX:,}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle('./result_criticality_multiN_k3.pkl')
    print(f"\nSaved: result_criticality_multiN_k3.pkl  (rows={len(df):,})")
