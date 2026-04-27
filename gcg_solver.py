# gcg_solver.py
import numpy as np
import networkx as nx
import pandas as pd
import multiprocessing as mp
from numba import njit, prange
import time
from tqdm.auto import tqdm


# --- 1. Numbaによる高速計算ロジック ---
@njit(cache=True)
def compute_next_step(node_num, color_num, node_colors, adj_indices, adj_indptr,
                      p_noise, bot_pos, p_async, is_async_model, rng):
    """
    1ステップ分の色更新計算 (Numbaで高速化)
    """
    next_colors_cand = node_colors.copy()

    grid_lock = 0

    # 各ノードの次状態計算
    for i in range(node_num):
        neighbor_start = adj_indptr[i]
        neighbor_end = adj_indptr[i + 1]

        used_colors = set()
        for j in range(neighbor_start, neighbor_end):
            neighbor_idx = adj_indices[j]
            used_colors.add(node_colors[neighbor_idx])

        possible = []
        for c in range(1, color_num + 1):
            if c not in used_colors:
                possible.append(c)

        #print(used_colors)
        #print(possible)


        if len(possible) > 0:
            rand_idx = rng.integers(0, len(possible))
            next_colors_cand[i] = possible[rand_idx]
        else:
            grid_lock += 1
            if is_async_model:
                next_colors_cand[i] = rng.integers(1, color_num + 1)

        #print(next_colors_cand[i])

    # ボット介入
    for b_idx in bot_pos:
        if rng.random() < p_noise:
            next_colors_cand[b_idx] = rng.integers(1, color_num + 1)

    # 非同期更新
    act = rng.random(node_num) < 1 - p_async
    updated_colors = node_colors.copy()
    updated_colors[act] = next_colors_cand[act]

    return updated_colors, grid_lock


@njit(parallel=True, cache=True)
def compute_next_step_par(node_num, color_num, node_colors, adj_indices, adj_indptr,
                          p_noise, bot_pos, p_async, is_async_model,
                          rand_node, rand_noise, rand_async, rand_fallback):
    """
    並列版ステップ計算 (prange でノードを並列処理)。
    乱数は呼び出し前に生成して渡す (スレッドセーフ)。
    rand_node    : (node_num,)  float  [0,1)  色選択用
    rand_noise   : (bot_num,)   float  [0,1)  ノイズ判定用
    rand_async   : (node_num,)  float  [0,1)  非同期更新用
    rand_fallback: (node_num,)  int64  [1,k]  グリッドロック/ノイズ時の代替色
    """
    next_colors_cand = np.empty(node_num, dtype=node_colors.dtype)

    # ── ノードごとの色選択 (並列) ────────────────────────────────────────────
    for i in prange(node_num):
        neighbor_start = adj_indptr[i]
        neighbor_end   = adj_indptr[i + 1]

        # 隣接ノードの使用色を boolean 配列でマーク (スタック確保 → スレッドセーフ)
        used = np.zeros(color_num + 1, dtype=np.bool_)
        for j in range(neighbor_start, neighbor_end):
            used[node_colors[adj_indices[j]]] = True

        # 利用可能色を数える
        n_avail = 0
        for c in range(1, color_num + 1):
            if not used[c]:
                n_avail += 1

        if n_avail > 0:
            # rand_node[i] を使って一様選択
            target = int(rand_node[i] * n_avail)
            if target >= n_avail:
                target = n_avail - 1
            count = 0
            chosen = 1
            for c in range(1, color_num + 1):
                if not used[c]:
                    if count == target:
                        chosen = c
                        break
                    count += 1
            next_colors_cand[i] = chosen
        else:
            # グリッドロック
            if is_async_model:
                next_colors_cand[i] = rand_fallback[i]
            else:
                next_colors_cand[i] = node_colors[i]

    # ── ボット介入 (逐次: bot_pos は通常小さい) ─────────────────────────────
    for b in range(len(bot_pos)):
        if rand_noise[b] < p_noise:
            next_colors_cand[bot_pos[b]] = rand_fallback[bot_pos[b]]

    # ── 非同期更新 ────────────────────────────────────────────────────────────
    updated_colors = node_colors.copy()
    for i in range(node_num):
        if rand_async[i] < 1.0 - p_async:
            updated_colors[i] = next_colors_cand[i]

    return updated_colors, 0  # gridlock count 省略


# --- 2. シミュレーションクラス ---
class FastGCG:
    def __init__(self, seed, p_async, p_noise, color_num, node_num, ba_m, bot_type, bot_num,
                 is_async_model=False, graph_type='ba', graph_param=None):
        self.seed = seed
        self.p_async = p_async
        self.p_noise = p_noise
        self.color_num = color_num
        self.node_num = node_num
        self.is_async_model = is_async_model

        # グラフ生成とCSR変換
        rng_graph = np.random.default_rng(seed)
        if graph_type == 'ba':
            self.net = nx.barabasi_albert_graph(node_num, ba_m, seed=seed)
        elif graph_type == 'er':
            # Erdos-Renyi: graph_param = edge probability q
            q = graph_param if graph_param is not None else 2 * ba_m / (node_num - 1)
            self.net = nx.erdos_renyi_graph(node_num, q, seed=seed)
            # 孤立ノードが出た場合は接続補完
            while not nx.is_connected(self.net):
                seed += 1
                self.net = nx.erdos_renyi_graph(node_num, q, seed=seed)
        elif graph_type == 'ws':
            # Watts-Strogatz: graph_param = rewiring probability beta
            k_ws = max(4, 2 * ba_m)   # 平均次数を BA と同程度に
            beta = graph_param if graph_param is not None else 0.3
            self.net = nx.watts_strogatz_graph(node_num, k_ws, beta, seed=seed)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        adj_matrix = nx.adjacency_matrix(self.net).tocsr()
        self.adj_indices = adj_matrix.indices
        self.adj_indptr = adj_matrix.indptr
        self.edges = np.array(self.net.edges())

        self.rng = np.random.default_rng(seed)
        self.node_colors = self.rng.integers(1, color_num + 1, size=node_num)

        degrees = np.array([self.net.degree()[i] for i in range(node_num)])
        if bot_type == "c":
            self.bot_pos = np.argsort(degrees)[-bot_num:]
        elif bot_type == "p":
            self.bot_pos = np.argsort(degrees)[:bot_num]
        else:
            self.bot_pos = self.rng.choice(node_num, bot_num, replace=False)

    def sol_check(self):
        if len(self.edges) == 0: return 0
        u, v = self.edges[:, 0], self.edges[:, 1]
        conflicts = (self.node_colors[u] == self.node_colors[v])
        if not np.any(conflicts): return 0
        conflict_nodes = np.unique(np.concatenate((u[conflicts], v[conflicts])))
        return len(conflict_nodes)

    def sim_run(self, step_max, return_history=False):
        history_changed = []
        history_conflict_ratio = []
        history_gridlock = []
        solved_step = -1

        for t in range(step_max):
            last_colors = self.node_colors.copy()
            self.node_colors, gridlock = compute_next_step(
                self.node_num, self.color_num, self.node_colors,
                self.adj_indices, self.adj_indptr,
                self.p_noise, self.bot_pos, self.p_async, self.is_async_model,
                self.rng
            )

            n_conflicts = self.sol_check()

            if return_history:
                changed_num = np.sum(last_colors != self.node_colors)
                history_changed.append(changed_num)
                history_conflict_ratio.append(n_conflicts / self.node_num)
                history_gridlock.append(gridlock)

            if solved_step == -1:
                if n_conflicts == 0:
                    solved_step = t
                    if not return_history: break

        result = {
            'solved': 1 if solved_step != -1 else 0,
            'solve_time': solved_step if solved_step != -1 else step_max,
            'final_conflict_ratio': self.sol_check() / self.node_num
        }
        if return_history:
            result['history_changed'] = np.array(history_changed, dtype=np.int32)
            result['history_conflict_ratio'] = np.array(history_conflict_ratio)
            result['history_gridlock'] = np.array(history_gridlock)
        return result


# --- 3. ワーカー関数 (トップレベルに定義必須) ---
def _worker_run_single(params):
    """Poolから呼び出される単一試行関数"""
    try:
        model = FastGCG(
            seed=params['seed'],
            p_async=params['p_async'],
            p_noise=params['p_noise'],
            color_num=params['color_num'],
            node_num=params['node_num'],
            ba_m=params['ba_m'],
            bot_type=params.get('bot_type', 'r'),
            bot_num=params.get('bot_num', 0),
            is_async_model=params.get('is_async_model', False),
            graph_type=params.get('graph_type', 'ba'),
            graph_param=params.get('graph_param', None),
        )
        step_max = params.get('step_max', 2000)
        return_history = params.get('return_history', False)

        res = model.sim_run(step_max=step_max, return_history=return_history)

        # 入力パラメータと結果をマージして返す
        output = params.copy()
        output.update(res)
        return output
    except Exception as e:
        # エラー時の情報を含めて返す
        print(f"Error in seed {params.get('seed')}: {e}")
        return params | {'error': str(e)}


# 並列版で使用するノードしきい値
# N ≤ 501 のスキャンでは prange の恩恵が小さいため、実質無効化 (多プロセス逐次を優先)
_PAR_NODE_THRESHOLD = 10000  # N > この値で compute_next_step_par を使用


# --- 4a. バッチワーカー (グラフ再利用版) ---
def _worker_run_batch(batch):
    """同じ (seed, node_num, ba_m, bot_type, bot_num) のタスクをまとめて処理。
    グラフは1回だけ構築し、color_num / model パラメータを変えながら再利用する。
    N > _PAR_NODE_THRESHOLD では compute_next_step_par (prange) を使用。
    """
    seed, node_num, ba_m, bot_type, bot_num = batch['key']

    # グラフを1回だけ構築
    net = nx.barabasi_albert_graph(node_num, ba_m, seed=seed)
    adj_matrix = nx.adjacency_matrix(net).tocsr()
    adj_indices = adj_matrix.indices
    adj_indptr  = adj_matrix.indptr
    edges = np.array(net.edges()) if net.number_of_edges() > 0 else np.empty((0, 2), dtype=int)

    degrees = np.array([net.degree()[i] for i in range(node_num)])
    rng_init = np.random.default_rng(seed)
    if bot_type == 'c':
        bot_pos = np.argsort(degrees)[-bot_num:]
    elif bot_type == 'p':
        bot_pos = np.argsort(degrees)[:bot_num]
    else:
        bot_pos = rng_init.choice(node_num, bot_num, replace=False)

    use_par = node_num > _PAR_NODE_THRESHOLD

    results = []
    for params in batch['tasks']:
        color_num      = params['color_num']
        step_max       = params.get('step_max', 2000)
        return_history = params.get('return_history', False)
        p_noise        = params['p_noise']
        p_async        = params['p_async']
        is_async_model = params.get('is_async_model', False)

        rng         = np.random.default_rng(seed)
        node_colors = rng.integers(1, color_num + 1, size=node_num)

        history_changed        = []
        history_conflict_ratio = []
        history_gridlock       = []
        solved_step = -1

        for _ in range(step_max):
            last_colors = node_colors.copy()
            if use_par:
                rand_node     = rng.random(node_num)
                rand_noise    = rng.random(len(bot_pos))
                rand_async    = rng.random(node_num)
                rand_fallback = rng.integers(1, color_num + 1, size=node_num)
                node_colors, gridlock = compute_next_step_par(
                    node_num, color_num, node_colors,
                    adj_indices, adj_indptr,
                    p_noise, bot_pos, p_async, is_async_model,
                    rand_node, rand_noise, rand_async, rand_fallback
                )
            else:
                node_colors, gridlock = compute_next_step(
                    node_num, color_num, node_colors,
                    adj_indices, adj_indptr,
                    p_noise, bot_pos, p_async, is_async_model, rng
                )

            if edges.shape[0] > 0:
                u, v = edges[:, 0], edges[:, 1]
                mask = node_colors[u] == node_colors[v]
                n_conflicts = (len(np.unique(np.concatenate((u[mask], v[mask]))))
                               if np.any(mask) else 0)
            else:
                n_conflicts = 0

            if return_history:
                history_changed.append(int(np.sum(last_colors != node_colors)))
                history_conflict_ratio.append(n_conflicts / node_num)
                history_gridlock.append(gridlock)

            if solved_step == -1 and n_conflicts == 0:
                solved_step = _
                if not return_history:
                    break

        # 最終コンフリクト計算
        if edges.shape[0] > 0:
            u, v = edges[:, 0], edges[:, 1]
            mask = node_colors[u] == node_colors[v]
            final_conflict_nodes = (len(np.unique(np.concatenate((u[mask], v[mask]))))
                                    if np.any(mask) else 0)
        else:
            final_conflict_nodes = 0

        res = {
            'solved':                1 if solved_step != -1 else 0,
            'solve_time':            solved_step if solved_step != -1 else step_max,
            'final_conflict_ratio':  final_conflict_nodes / node_num,
        }
        if return_history:
            res['history_changed']        = np.array(history_changed,        dtype=np.int32)
            res['history_conflict_ratio'] = np.array(history_conflict_ratio, dtype=np.float32)
            res['history_gridlock']       = np.array(history_gridlock,       dtype=np.int32)

        output = params.copy()
        output.update(res)
        results.append(output)

    return results


# --- 4b. バッチ版 並列実行管理関数 ---
def run_parallel_simulation_batched(task_list, n_jobs=None):
    """タスクを (seed, node_num, ba_m, bot_type, bot_num) でグループ化し、
    グラフ構築を1回に抑えて高速化するバッチ版。
    """
    from collections import defaultdict

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    import numba as _numba
    numba_threads = _numba.get_num_threads()

    # バッチ化
    batches_dict = defaultdict(list)
    for params in task_list:
        key = (params['seed'], params['node_num'], params['ba_m'],
               params.get('bot_type', 'r'), params.get('bot_num', 0))
        batches_dict[key].append(params)

    batches   = [{'key': k, 'tasks': v} for k, v in batches_dict.items()]
    reuse_x   = len(task_list) // max(len(batches), 1)

    # N ≤ threshold → 多プロセス/逐次Numba
    # N > threshold → 少プロセス/prange Numba  (プロセス×スレッド ≒ CPU コア数)
    small_batches = [b for b in batches if b['key'][1] <= _PAR_NODE_THRESHOLD]
    large_batches = [b for b in batches if b['key'][1] >  _PAR_NODE_THRESHOLD]
    n_jobs_large  = max(1, mp.cpu_count() // numba_threads)  # コア数 / スレッド数

    # N順に並んでいると終盤ほど遅くなるため、シャッフルして進捗を平坦化
    import random as _random
    _random.shuffle(small_batches)
    _random.shuffle(large_batches)

    print(f"Starting batched parallel simulation (prange threshold N>{_PAR_NODE_THRESHOLD})")
    print(f"Total tasks: {len(task_list):,}  ->  {len(batches):,} batches (graph reuse ~{reuse_x}x)")
    print(f"  Small-N batches: {len(small_batches):,}  workers={n_jobs}")
    print(f"  Large-N batches: {len(large_batches):,}  workers={n_jobs_large} "
          f"x {numba_threads} prange-threads")

    start_time = time.time()
    all_results: list = []

    # ── 小N バッチ (多プロセス, 逐次Numba) ────────────────────────────────────
    if small_batches:
        with mp.Pool(n_jobs) as pool:
            for batch_result in tqdm(
                    pool.imap_unordered(_worker_run_batch, small_batches),
                    total=len(small_batches), desc="Small-N batches"):
                all_results.extend(batch_result)

    # ── 大N バッチ (少プロセス, prange Numba) ─────────────────────────────────
    if large_batches:
        with mp.Pool(n_jobs_large) as pool:
            for batch_result in tqdm(
                    pool.imap_unordered(_worker_run_batch, large_batches),
                    total=len(large_batches), desc="Large-N batches (prange)"):
                all_results.extend(batch_result)

    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds.")

    return pd.DataFrame(all_results)


# --- 4. 並列実行管理関数 (ここを修正) ---
def run_parallel_simulation(task_list, n_jobs=None):
    """
    タスクリストを受け取り、並列実行してDataFrameで返す関数
    tqdmによる進捗バー付き
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    print(f"Starting parallel simulation with {n_jobs} processes...")
    print(f"Total tasks: {len(task_list)}")

    start_time = time.time()

    # Poolの作成と実行
    with mp.Pool(n_jobs) as pool:
        # pool.map の代わりに pool.imap_unordered を使うことで、
        # 1つタスクが終わるごとに tqdm のカウンターを進めることができます。
        # (結果の順序が入れ替わっても、DataFrameにするので問題ありません)

        results_iter = pool.imap_unordered(_worker_run_single, task_list)

        # tqdmでイテレータをラップしてリスト化
        results = list(tqdm(results_iter, total=len(task_list), desc="Simulation Progress"))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds.")

    return pd.DataFrame(results)