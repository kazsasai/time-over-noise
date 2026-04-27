"""
gcg_solver_gpu.py
=================
GPU-accelerated graph coloring solver using PyTorch (MPS / CUDA / CPU).

Strategy
--------
For N ≤ gpu_n_max (default 250):
  * Build adjacency matrices for all T=1000 seeds at once → (T, N, N) tensor
  * Run all T trials simultaneously per (k, model) config via torch.bmm
  * Reuse the adj-batch for every k value → O(n_colors) bmm calls per N group

For N > gpu_n_max:
  * Falls back to CPU run_parallel_simulation_batched (same as before)

Expected speedup on Apple M4 Pro:
  N=50   → ~20× faster
  N=100  → ~12×
  N=200  → ~8×
  N=250  → ~6×

Main entry point
----------------
  from gcg_solver_gpu import run_hybrid_simulation_batched
  df = run_hybrid_simulation_batched(task_list, gpu_n_max=250)
"""

import numpy as np
import networkx as nx
import pandas as pd
import time
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F


# ── Device selection ──────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Core GPU simulation ───────────────────────────────────────────────────────
def _run_gpu_config(adj_batch, bot_mask, node_num, color_num,
                    p_noise, p_async, is_async_model,
                    step_max, device):
    """
    Simulate T trials in parallel on GPU for one (k, model) configuration.

    Parameters
    ----------
    adj_batch : FloatTensor (T, N, N)   adjacency matrices for T graphs
    bot_mask  : BoolTensor  (T, N)      True where each trial's node is a bot
    node_num  : int
    color_num : int
    p_noise   : float
    p_async   : float
    is_async_model : bool
    step_max  : int
    device    : torch.device

    Returns
    -------
    solved_arr    : np.ndarray int32 (T,)   1 if solved
    solve_time_arr: np.ndarray int32 (T,)   step at which solved (step_max if not)
    """
    T = adj_batch.shape[0]
    N = node_num
    k = color_num

    # Random initial colors: (T, N) in [0, k-1]
    colors = torch.randint(0, k, (T, N), device=device)

    solved    = torch.zeros(T, dtype=torch.bool, device=device)
    sol_time  = torch.full((T,), step_max, dtype=torch.long, device=device)

    # Pre-compute neighbor usage for the initial state
    onehot         = F.one_hot(colors, k).float()           # (T, N, k)
    neighbor_cnt   = torch.bmm(adj_batch, onehot)           # (T, N, k)
    neighbor_used  = neighbor_cnt > 0                        # (T, N, k) bool

    for t in range(step_max):
        # ── Conflict check on current state (uses pre-computed neighbor_used) ──
        has_conflict = (neighbor_used & onehot.bool()).any(dim=(1, 2))   # (T,)
        newly_solved = ~has_conflict & ~solved
        sol_time[newly_solved] = t
        solved = solved | newly_solved

        if solved.all():
            break

        # ── Sample new colors from available ──────────────────────────────────
        available   = ~neighbor_used                         # (T, N, k)
        rand_scores = torch.rand(T, N, k, device=device)
        rand_scores[~available] = -1.0
        new_colors  = rand_scores.argmax(dim=-1)             # (T, N)

        # Gridlock: no available color → random fallback (async) or keep (sync)
        has_available = available.any(dim=-1)                # (T, N)
        if is_async_model:
            fallback   = torch.randint(0, k, (T, N), device=device)
            new_colors = torch.where(has_available, new_colors, fallback)

        # ── Noise bots ────────────────────────────────────────────────────────
        if p_noise > 0 and bot_mask is not None:
            noise_fire = (torch.rand(T, N, device=device) < p_noise) & bot_mask
            random_col = torch.randint(0, k, (T, N), device=device)
            new_colors = torch.where(noise_fire, random_col, new_colors)

        # ── Async update ──────────────────────────────────────────────────────
        if p_async > 0:
            update_mask = torch.rand(T, N, device=device) >= p_async
            new_colors  = torch.where(update_mask, new_colors, colors)

        colors = new_colors

        # ── Pre-compute neighbor_used for next iteration ──────────────────────
        onehot        = F.one_hot(colors, k).float()
        neighbor_cnt  = torch.bmm(adj_batch, onehot)
        neighbor_used = neighbor_cnt > 0

    return solved.cpu().numpy().astype(np.int32), sol_time.cpu().numpy().astype(np.int32)


# ── GPU path: group by N, build adj-batch once, sweep k values ───────────────
def _run_gpu_group(tasks_for_n, node_num, ba_m, bot_type, bot_num,
                   device, T_batch_max=None):
    """
    Process all tasks sharing the same (node_num, ba_m, bot_type, bot_num).
    Builds adj matrices ONCE per unique seed, reuses for all k / model configs.
    """
    N = node_num

    # Memory guard: limit trials per GPU batch to keep adj_batch < 1 GB
    if T_batch_max is None:
        bytes_per_trial = N * N * 4 + N * 4    # adj + onehot slice
        T_batch_max = max(1, int(1e9 / bytes_per_trial))

    # ── Collect unique seeds and configs ─────────────────────────────────────
    seeds_sorted = sorted({p['seed'] for p in tasks_for_n})
    seed_to_idx  = {s: i for i, s in enumerate(seeds_sorted)}
    T_total      = len(seeds_sorted)

    unique_configs = {}          # ckey → list of tasks
    for p in tasks_for_n:
        ckey = (p['color_num'], round(p['p_noise'], 6), round(p['p_async'], 6),
                bool(p.get('is_async_model', False)), int(p.get('step_max', 2000)),
                str(p.get('label', '')))
        unique_configs.setdefault(ckey, []).append(p)

    # ── Build adjacency matrices for all seeds ────────────────────────────────
    adj_np   = np.zeros((T_total, N, N), dtype=np.float32)
    bot_np   = np.zeros((T_total, N),    dtype=bool)

    for idx, seed in enumerate(seeds_sorted):
        net     = nx.barabasi_albert_graph(N, ba_m, seed=seed)
        adj_arr = nx.to_numpy_array(net, dtype=np.float32)
        adj_np[idx] = adj_arr

        degs = np.array([net.degree(i) for i in range(N)])
        rng  = np.random.default_rng(seed)
        if bot_type == 'c':
            bp = np.argsort(degs)[-bot_num:]
        elif bot_type == 'p':
            bp = np.argsort(degs)[:bot_num]
        else:
            bp = rng.choice(N, bot_num, replace=False)
        bot_np[idx, bp] = True

    # ── Simulate each config, sweeping over T_batch windows ──────────────────
    config_results = {}  # ckey → (solved_arr, sol_time_arr) of length T_total

    for ckey, _ in unique_configs.items():
        color_num, p_noise, p_async, is_async, step_max, _label = ckey
        solved_all   = np.zeros(T_total, dtype=np.int32)
        sol_time_all = np.full(T_total, step_max, dtype=np.int32)

        for b_start in range(0, T_total, T_batch_max):
            b_end = min(b_start + T_batch_max, T_total)

            adj_batch = torch.from_numpy(adj_np[b_start:b_end]).to(device)
            bot_mask  = torch.from_numpy(bot_np[b_start:b_end]).to(device)

            solved, sol_time = _run_gpu_config(
                adj_batch, bot_mask, N, color_num,
                p_noise, p_async, is_async, step_max, device
            )
            solved_all[b_start:b_end]   = solved
            sol_time_all[b_start:b_end] = sol_time

        config_results[ckey] = (solved_all, sol_time_all)

    # ── Reconstruct output records matching input task format ─────────────────
    results = []
    for ckey, task_list_ckey in unique_configs.items():
        solved_arr, sol_time_arr = config_results[ckey]
        for p in task_list_ckey:
            idx  = seed_to_idx[p['seed']]
            out  = p.copy()
            out['solved']               = int(solved_arr[idx])
            out['solve_time']           = int(sol_time_arr[idx])
            out['final_conflict_ratio'] = 0.0  # not tracked in GPU path
            results.append(out)

    return results


# ── Public API ────────────────────────────────────────────────────────────────
def run_hybrid_simulation_batched(task_list, gpu_n_max=250,
                                   device=None, n_jobs_cpu=None):
    """
    Hybrid GPU + CPU batched simulation.

    Parameters
    ----------
    task_list   : list of dicts (same format as gcg_solver)
    gpu_n_max   : int   N ≤ this value uses GPU (default 250)
    device      : torch.device or None (auto-detect)
    n_jobs_cpu  : int or None  CPU worker count for large-N tasks

    Returns
    -------
    pd.DataFrame  (same columns as run_parallel_simulation_batched)
    """
    if device is None:
        device = get_device()

    gpu_tasks = [t for t in task_list if t['node_num'] <= gpu_n_max]
    cpu_tasks = [t for t in task_list if t['node_num'] >  gpu_n_max]

    print(f"Device     : {device}")
    print(f"GPU tasks  : {len(gpu_tasks):,}   (N ≤ {gpu_n_max})")
    print(f"CPU tasks  : {len(cpu_tasks):,}   (N > {gpu_n_max})")

    all_results = []
    t0 = time.time()

    # ── GPU path ──────────────────────────────────────────────────────────────
    if gpu_tasks and str(device) != 'cpu':
        # Group by (node_num, ba_m, bot_type, bot_num)
        n_groups = defaultdict(list)
        for p in gpu_tasks:
            key = (p['node_num'], p['ba_m'],
                   p.get('bot_type', 'r'), p.get('bot_num', 0))
            n_groups[key].append(p)

        print(f"\nGPU: {len(n_groups)} N-groups")
        for (node_num, ba_m, bot_type, bot_num), tasks in tqdm(
                n_groups.items(), desc="GPU groups"):
            results = _run_gpu_group(
                tasks, node_num, ba_m, bot_type, bot_num, device
            )
            all_results.extend(results)

        print(f"GPU done in {time.time()-t0:.1f}s")

    elif gpu_tasks:
        # No real GPU: push gpu_tasks to CPU
        cpu_tasks = gpu_tasks + cpu_tasks

    # ── CPU path ──────────────────────────────────────────────────────────────
    if cpu_tasks:
        from gcg_solver import run_parallel_simulation_batched
        t1 = time.time()
        df_cpu = run_parallel_simulation_batched(cpu_tasks, n_jobs=n_jobs_cpu)
        all_results.extend(df_cpu.to_dict('records'))
        print(f"CPU done in {time.time()-t1:.1f}s")

    elapsed = time.time() - t0
    print(f"Total: {elapsed:.1f}s")

    return pd.DataFrame(all_results)
