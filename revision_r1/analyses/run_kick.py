"""
Injection experiment (Fig. 5e, f; Supplementary Fig. S7): inject one conflict into a colouring that has stayed solved for at
least 50 sweeps (ARM) and measure the response until the colouring is consistent again.
  heal time  : sweeps from the kick to the first step with no conflict (censored at the next kick / record end)
  size       : colour changes from the step after the kick up to and including the healing step
  max_conf   : largest number of conflicting nodes during the response
For comparison, natural excursions (unsolved periods after the first solution) are measured on the core data at the
same p with the same definitions (duration in sweeps, size = colour changes incl. the solving step).
Conditions: ARM p = 0.05..0.95, 1000 core graphs (seed 2024), 50,000 steps, up to 40 kicks per trial spaced
1000 sweeps apart; SNM eps=0.05 with a 1-step wait as an unstable-phase reference.
Output: runs/kick/kick.json and runs/kick/kick_raw_<cond>.npz.  Run from revision_r1/analyses.
"""
import json, os, math
import multiprocessing as mp
import numpy as np
import pandas as pd
from gcg_kick import simulate

OUT = 'runs/kick'
T = 50000
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)


def responses(r, rate):
    A = r['history_changed'].astype(np.int64)
    C = r['history_conflict_nodes']
    kt = r['kick_times']
    out = []
    for j, t0 in enumerate(kt):
        end = kt[j + 1] if j + 1 < len(kt) else T - 1
        seg = C[t0 + 1:end + 1]
        z = np.flatnonzero(seg == 0)
        if z.size:
            h = int(z[0])                                     # steps after t0+1 until healed
            out.append((1, (h + 1) * rate, int(A[t0 + 1:t0 + 2 + h].sum()), int(seg[:h + 1].max())))
        else:
            out.append((0, seg.size * rate, int(A[t0 + 1:end + 1].sum()), int(seg.max()) if seg.size else 0))
    return out


def worker(args):
    model, par, seed = args
    rate = (1 - par) if model == 'ARM' else 1.0
    task = dict(seed=int(seed), node_num=20, color_num=3, step_max=T,
                gridlock_rule='random' if model == 'ARM' else 'stay',
                p_async=par if model == 'ARM' else 0.0, p_noise=par if model == 'SNM' else 0.0,
                n_kicks=40,
                kick_wait=int(math.ceil(50 / rate)) if model == 'ARM' else 1,
                kick_gap=int(math.ceil(1000 / rate)))
    r = simulate(task)
    return responses(r, rate)


def natural(df, model, par, rate):
    """Natural excursions in the core data: unsolved periods after the first solution."""
    lab, col = ('Async Model', 'p_async') if model == 'ARM' else ('Noise Model', 'p_noise')
    g = df[(df['label'] == lab) & ((df[col] - par).abs() < 1e-6)]
    dur, size = [], []
    for A, cr in zip(g['history_changed'], g['history_conflict_ratio']):
        A = np.asarray(A, dtype=np.int64); s = np.asarray(cr) < 1e-12
        if not s.any():
            continue
        f = int(np.argmax(s)); post = s[f:]; Ap = A[f:]
        e = np.flatnonzero(np.diff(post.astype(np.int8))) + 1
        st = np.concatenate([[0], e]); ln = np.diff(np.concatenate([st, [post.size]]))
        for a, l in zip(st, ln):
            if not post[a] and a + l < post.size:            # completed unsolved period
                dur.append((l + 1) * rate)                   # incl. the solving step, as for kicks
                size.append(int(Ap[a:a + l + 1].sum()))
    return np.array(dur), np.array(size)


def summarize(rows):
    rows = np.array(rows, dtype=float)
    healed, ht, sz, mc = rows.T
    q = lambda x, pr: float(np.percentile(x, pr)) if x.size else np.nan
    h = healed == 1
    return dict(n_kicks=int(rows.shape[0]), share_healed=float(h.mean()),
                share_healed_within_1sweep=float(np.mean(h & (ht <= 1.0))),
                share_healed_within_10sweeps=float(np.mean(h & (ht <= 10.0))),
                heal_time_median=q(ht[h], 50), heal_time_p90=q(ht[h], 90), heal_time_p99=q(ht[h], 99),
                size_median=q(sz[h], 50), size_p90=q(sz[h], 90), size_p99=q(sz[h], 99),
                maxconf_median=q(mc, 50), maxconf_p90=q(mc, 90))


def main():
    os.makedirs(OUT, exist_ok=True)
    conds = [('ARM', round(p, 2)) for p in np.arange(10) / 10 + 0.05] + [('SNM', 0.05)]
    res = {}
    ctx = mp.get_context('fork')
    with ctx.Pool(12) as pool:
        df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')     # loaded after the workers were forked
        for model, par in conds:
            rows = [x for lst in pool.map(worker, [(model, par, s) for s in SEEDS], chunksize=8) for x in lst]
            rate = (1 - par) if model == 'ARM' else 1.0
            dur, size = natural(df, model, par, rate)
            s = summarize(rows)
            s['natural'] = dict(n=int(dur.size), share_within_1sweep=float(np.mean(dur <= 1.0)) if dur.size else np.nan,
                                share_within_10sweeps=float(np.mean(dur <= 10.0)) if dur.size else np.nan,
                                dur_median=float(np.median(dur)) if dur.size else np.nan,
                                dur_p90=float(np.percentile(dur, 90)) if dur.size else np.nan,
                                dur_p99=float(np.percentile(dur, 99)) if dur.size else np.nan,
                                size_median=float(np.median(size)) if size.size else np.nan,
                                size_p90=float(np.percentile(size, 90)) if size.size else np.nan,
                                size_p99=float(np.percentile(size, 99)) if size.size else np.nan)
            res[f'{model}_{par:.2f}'] = s
            np.savez_compressed(f'{OUT}/kick_raw_{model}_{par:.2f}.npz', kicks=np.array(rows, dtype=float),
                                natural_dur=dur, natural_size=size)
            print(model, par, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != 'natural'}),
                  '| natural', json.dumps({k: round(v, 3) for k, v in s['natural'].items()}), flush=True)
    json.dump(res, open(f'{OUT}/kick.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
