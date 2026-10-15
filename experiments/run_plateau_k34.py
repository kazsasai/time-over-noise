"""
Plateau analysis at fixed k=3 and k=4, varying N=20..100
Both SNM and ARM, varying p/ε from 0.00 to 1.00 (step 0.05)
200 trials per condition (master seed 7777)
step_max=10000

Output: result_for_plateau_k34.pkl
"""

import multiprocessing as mp
import platform
if platform.system() != 'Windows':
    mp.set_start_method('fork', force=True)

import numpy as np
import sys
sys.path.insert(0, '.')
import gcg_solver

MASTER_SEED = 7777
N_TRIALS    = 200
STEP_MAX    = 10000
N_VALUES    = [20, 30, 40, 50, 60, 70, 80, 90, 100]
K_VALUES    = [3, 4]
P_VALUES    = np.round(np.arange(0.00, 1.01, 0.05), 4)

rng   = np.random.default_rng(MASTER_SEED)
seeds = rng.integers(0, 10**9, size=N_TRIALS)

MODELS = [
    {'label': 'Noise', 'is_async_model': False},
    {'label': 'Async', 'is_async_model': True},
]

if __name__ == '__main__':
    tasks = []
    for k in K_VALUES:
        for n in N_VALUES:
            for model in MODELS:
                is_async = model['is_async_model']
                for p in P_VALUES:
                    for s in seeds:
                        tasks.append({
                            'seed':           int(s),
                            'node_num':       n,
                            'color_num':      k,
                            'ba_m':           2,
                            'bot_num':        n,
                            'bot_type':       'c',
                            'step_max':       STEP_MAX,
                            'return_history': False,
                            'label':          model['label'],
                            'is_async_model': is_async,
                            'p_noise':        float(p) if not is_async else 0.0,
                            'p_async':        float(p) if is_async     else 0.0,
                        })

    total = len(tasks)
    print(f"Total tasks : {total:,}")
    print(f"k values    : {K_VALUES}")
    print(f"N values    : {N_VALUES}")
    print(f"p values    : {P_VALUES.tolist()}")
    print(f"Trials      : {N_TRIALS}  |  step_max: {STEP_MAX:,}")

    df = gcg_solver.run_parallel_simulation_batched(tasks)
    df.to_pickle('./result_for_plateau_k34.pkl')
    print(f"\nSaved: result_for_plateau_k34.pkl  (rows={len(df):,})")
