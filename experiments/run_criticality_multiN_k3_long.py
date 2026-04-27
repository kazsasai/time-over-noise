"""
Multi-N comparison at k=3 (ARM only) — longer step_max for larger N

N=20  : step_max =  50,000  (same as before)
N=50  : step_max = 200,000
N=100 : step_max = 500,000

k=3 (= m+1, chromatic boundary for BA m=2)
trials : 200 per (N, p)
p values: 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75

Output: result_criticality_multiN_k3_long.pkl
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver

MASTER_SEED = 5555
N_TRIALS    = 200
COLOR_NUM   = 3
P_VALUES    = np.round([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75], 4)

N_STEPMAX = {
    20:  50_000,
    50:  200_000,
    100: 500_000,
}

rng   = np.random.default_rng(MASTER_SEED)
seeds = rng.integers(0, 10**9, size=N_TRIALS)

if __name__ == '__main__':
    tasks = []
    for node_num, step_max in N_STEPMAX.items():
        for p in P_VALUES:
            for s in seeds:
                tasks.append({
                    'seed':           int(s),
                    'node_num':       node_num,
                    'color_num':      COLOR_NUM,
                    'ba_m':           2,
                    'bot_num':        node_num,
                    'bot_type':       'c',
                    'step_max':       step_max,
                    'return_history': True,
                    'label':          'Async Model',
                    'is_async_model': True,
                    'p_noise':        0.0,
                    'p_async':        float(p),
                    'node_num_key':   node_num,   # keep for filtering
                })

    print(f"Total tasks : {len(tasks):,}")
    for n, sm in N_STEPMAX.items():
        print(f"  N={n:3d}  step_max={sm:>7,}  tasks={len(P_VALUES)*N_TRIALS:,}")
    print(f"k={COLOR_NUM}  |  trials={N_TRIALS}  |  p_values={P_VALUES.tolist()}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle('./result_criticality_multiN_k3_long.pkl')
    print(f"\nSaved: result_criticality_multiN_k3_long.pkl  (rows={len(df):,})")
