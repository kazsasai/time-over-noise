"""
Variants of the GCG solver for the control experiments (Fig. 3 factorial, Supplementary Table S2).

For the existing cells (SNM, ARM and the 2×2 factorial)
this module reproduces gcg_solver.py bit-for-bit: same graph, same initial
colours, same RNG draw order (checked by validate()).

schedule
  'bernoulli'  every node is active with probability 1 - p_async (p_async = 0: synchronous)   [SNM / ARM]
  'sequential' one uniformly chosen node per step                                              [E3]
  'exclusive'  Bernoulli activation, then active neighbours never update in the same step:
               a node updates only if its random priority beats every active neighbour.
               Needs knowledge of the neighbours' timing → oracle condition                     [E4]
gridlock_rule
  'stay'     keep the current colour                                                           [SNM]
  'random'   uniform over all k colours                                                        [ARM]
  'minconf'  colour minimising U_i(c), ties broken at random                                  [E1]
q        probability that a gridlocked active node acts (rules 'random' / 'minconf')           [E2]
p_noise  SNM ε-noise on bot nodes (uniform over k)
return_history  also return per-step A(t) (changed nodes) and conflicting-node counts (int16)

Gridlock = no conflict-free colour exists for the node (min_c U_i(c) ≥ 1).
Non-gridlocked active nodes pick uniformly among conflict-free colours (all models).
"""
import multiprocessing as mp
import numpy as np
import networkx as nx
import pandas as pd
from numba import njit

SCHEDULES = {'bernoulli': 0, 'sequential': 1, 'exclusive': 2}
RULES     = {'stay': 0, 'random': 1, 'minconf': 2}


@njit(cache=True)
def _decide(i, k, colors, adj_indices, adj_indptr, rule, q, rng):
    """Candidate colour for node i. RNG use mirrors gcg_solver.compute_next_step for
    rule stay/random with q == 1. Returns (colour, is_gridlocked)."""
    used = np.zeros(k + 1, dtype=np.int64)
    for j in range(adj_indptr[i], adj_indptr[i + 1]):
        used[colors[adj_indices[j]]] += 1
    n_free = 0
    for c in range(1, k + 1):
        if used[c] == 0:
            n_free += 1
    if n_free > 0:
        idx = rng.integers(0, n_free)
        cnt = 0
        for c in range(1, k + 1):
            if used[c] == 0:
                if cnt == idx:
                    return c, 0
                cnt += 1
    # gridlock
    if rule == 0:
        return colors[i], 1
    if q < 1.0:
        if rng.random() >= q:
            return colors[i], 1
    if rule == 1:
        return rng.integers(1, k + 1), 1
    m = used[1]
    for c in range(2, k + 1):
        if used[c] < m:
            m = used[c]
    n_min = 0
    for c in range(1, k + 1):
        if used[c] == m:
            n_min += 1
    idx = rng.integers(0, n_min)
    cnt = 0
    for c in range(1, k + 1):
        if used[c] == m:
            if cnt == idx:
                return c, 1
            cnt += 1
    return colors[i], 1


@njit(cache=True)
def _n_conflict_nodes(colors, eu, ev, n):
    mark = np.zeros(n, dtype=np.bool_)
    for e in range(eu.shape[0]):
        if colors[eu[e]] == colors[ev[e]]:
            mark[eu[e]] = True
            mark[ev[e]] = True
    return mark.sum()


@njit(cache=True)
def _run(n, k, colors, adj_indices, adj_indptr, eu, ev, bot_pos, is_bot,
         step_max, schedule, rule, q, p_async, p_noise, rng, hist_changed, hist_conf):
    # hist_changed / hist_conf: length step_max to record A(t) (nodes whose colour changed)
    # and the number of conflicting nodes per step; length 0 to skip. Recording uses no RNG.
    solved_step = -1
    act_to_solve = -1
    total_act = 0
    total_attempt = 0
    steps_solved_after = 0
    breaks = 0
    prev_solved = False
    for t in range(step_max):
        if schedule == 1:                                   # sequential
            i = rng.integers(0, n)
            c, g = _decide(i, k, colors, adj_indices, adj_indptr, rule, q, rng)
            if p_noise > 0.0 and is_bot[i]:
                if rng.random() < p_noise:
                    c = rng.integers(1, k + 1)
            changed = 1 if c != colors[i] else 0
            colors[i] = c
            n_act = 1
            n_try = 1
        else:                                               # bernoulli / exclusive
            cand = colors.copy()
            for i in range(n):
                c, g = _decide(i, k, colors, adj_indices, adj_indptr, rule, q, rng)
                cand[i] = c
            for b in range(bot_pos.shape[0]):
                if rng.random() < p_noise:
                    cand[bot_pos[b]] = rng.integers(1, k + 1)
            act = rng.random(n) < 1 - p_async
            n_try = act.sum()
            if schedule == 2:                               # neighbour-exclusive (oracle)
                prio = rng.random(n)
                allow = act.copy()
                for i in range(n):
                    if act[i]:
                        for j in range(adj_indptr[i], adj_indptr[i + 1]):
                            nb = adj_indices[j]
                            if act[nb] and prio[nb] > prio[i]:
                                allow[i] = False
                                break
                act = allow
            changed = 0
            for i in range(n):
                if act[i]:
                    if cand[i] != colors[i]:
                        changed += 1
                    colors[i] = cand[i]
            n_act = act.sum()
        total_act += n_act
        total_attempt += n_try
        n_conf = _n_conflict_nodes(colors, eu, ev, n)
        if hist_changed.shape[0] > 0:
            hist_changed[t] = changed
            hist_conf[t] = n_conf
        is_sol = n_conf == 0
        if is_sol and solved_step == -1:
            solved_step = t
            act_to_solve = total_act
        if solved_step != -1:
            if is_sol:
                steps_solved_after += 1
            if prev_solved and not is_sol:
                breaks += 1
        prev_solved = is_sol
    final_conf = _n_conflict_nodes(colors, eu, ev, n)
    return solved_step, act_to_solve, total_act, total_attempt, steps_solved_after, breaks, final_conf


def simulate(task):
    """One trial. Graph, initial colours and bot placement are built exactly as in
    gcg_solver._worker_run_batch."""
    seed, n, k = task['seed'], task['node_num'], task['color_num']
    net = nx.barabasi_albert_graph(n, task.get('ba_m', 2), seed=seed)
    adj = nx.adjacency_matrix(net).tocsr()
    edges = np.array(net.edges())
    eu, ev = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    degrees = np.array([net.degree()[i] for i in range(n)])
    bot_pos = np.argsort(degrees)[-task.get('bot_num', n):]
    is_bot = np.zeros(n, dtype=np.bool_)
    is_bot[bot_pos] = True
    rng = np.random.default_rng(seed)
    colors = rng.integers(1, k + 1, size=n)
    step_max = task['step_max']
    want_hist = bool(task.get('return_history', False))
    hc = np.zeros(step_max if want_hist else 0, dtype=np.int16)
    hk = np.zeros(step_max if want_hist else 0, dtype=np.int16)
    out = _run(n, k, colors, adj.indices, adj.indptr, eu, ev, bot_pos, is_bot, step_max,
               SCHEDULES[task['schedule']], RULES[task['gridlock_rule']], float(task.get('q', 1.0)),
               float(task.get('p_async', 0.0)), float(task.get('p_noise', 0.0)), rng, hc, hk)
    s_step, act_solve, total_act, total_try, solved_after, breaks, final_conf = out
    res = dict(task)
    res.update({
        'solved': int(s_step != -1),
        'solve_time': int(s_step) if s_step != -1 else step_max,      # rounds (steps) to first solution
        'activations_to_solve': int(act_solve) if s_step != -1 else int(total_act),
        'total_activations': int(total_act),
        'total_attempts': int(total_try),
        'frac_solved_after_first': (solved_after / (step_max - s_step)) if s_step != -1 else np.nan,
        'n_breaks_after_first': int(breaks),
        'final_conflict_ratio': final_conf / n,
    })
    if want_hist:
        res['history_changed'] = hc            # A(t): nodes whose colour changed at step t
        res['history_conflict_nodes'] = hk     # conflicting nodes after step t (0 = solved)
    return res


def run_tasks(tasks, n_jobs=None):
    ctx = mp.get_context('fork')
    with ctx.Pool(n_jobs or mp.cpu_count()) as pool:
        rows = pool.map(simulate, tasks, chunksize=max(1, len(tasks) // (64 * (n_jobs or mp.cpu_count()))))
    return pd.DataFrame(rows)
