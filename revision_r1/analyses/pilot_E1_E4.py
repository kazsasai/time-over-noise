"""
Control experiments of Supplementary Table S2 (factorial, E1, E2).

N=20, k=3, BA m=2, 1,000 trials per condition with the Fig. 3 seeds (master_seed 2024).
Run from revision_r1/analyses:  python3 pilot_E1_E4.py
Outputs: runs/pilots/pilot_<block>.pkl
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import os, time
import numpy as np
import gcg_variants as gv

OUT = 'runs/pilots'
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)
BASE = dict(node_num=20, color_num=3, ba_m=2, bot_num=20, bot_type='c')
T_ROUNDS = 1000            # rounds budget for synchronous and partially asynchronous schedules
T_SEQ = 1000 * 20          # sequential: 1,000 sweeps = 20,000 single-node steps


def tasks_for(**cond):
    return [{**BASE, **cond, 'seed': int(s)} for s in SEEDS]


def block_factorial():
    """E1 + E3: timing (sync / partial p=0.45 / sequential) × gridlock rule (stay / random / minconf), ε = 0."""
    timings = [('sync', 'bernoulli', 0.0, T_ROUNDS),
               ('partial', 'bernoulli', 0.45, T_ROUNDS),
               ('sequential', 'sequential', 0.0, T_SEQ)]
    tasks = []
    for tname, sched, pa, tmax in timings:
        for rule in ('stay', 'random', 'minconf'):
            tasks += tasks_for(timing=tname, schedule=sched, p_async=pa, step_max=tmax,
                               gridlock_rule=rule, q=1.0, p_noise=0.0)
    return tasks


def block_e1_snm_minconf():
    """E1: synchronous min-conflicts SNM, ε swept."""
    tasks = []
    for eps in np.round(np.arange(0.0, 1.0, 0.02), 2):
        tasks += tasks_for(schedule='bernoulli', p_async=0.0, step_max=T_ROUNDS,
                           gridlock_rule='minconf', q=1.0, p_noise=float(eps))
    return tasks


def block_e2_sync_gated_q():
    """E2: synchronous updating, gridlocked nodes reselect at random with probability q (ε = 0)."""
    tasks = []
    for q in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        tasks += tasks_for(schedule='bernoulli', p_async=0.0, step_max=T_ROUNDS,
                           gridlock_rule='random', q=float(q), p_noise=0.0)
    return tasks


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [('factorial', block_factorial), ('e1_snm_minconf', block_e1_snm_minconf),
                     ('e2_sync_gated_q', block_e2_sync_gated_q)]:
        t0 = time.time()
        tasks = fn()
        df = gv.run_tasks(tasks)
        df.to_pickle(f'{OUT}/pilot_{name}.pkl')
        print(f'{name}: {len(df):,} trials in {time.time() - t0:.0f}s -> {OUT}/pilot_{name}.pkl', flush=True)
