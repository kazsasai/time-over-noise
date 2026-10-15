"""
Observable-matched 1/f comparison with Uragami & Gunji (2018, Complex Systems 27(4)), who follow Ninagawa:
spacetime pattern c_x(t), x = 1..N, t = 0..T-1 from a random initial condition; per-site DFT; power
S_f = sum_x |c_x(f)|^2; least-squares slope of log S_f vs log f over f = 1..100 with T = 1024 (power exponent alpha,
alpha ~ 1 for 1/f).  Our sites are agents with k = 3 colours, so each agent contributes three indicator series
1[colour = c] (a symmetric treatment of the categorical state).
Variants: (a) their protocol, T = 1024 steps from t = 0 (includes the initial transient); (b) a stationary window
of 1024 steps starting 1,000 steps after the first solution; (c) the same two in the sweep clock (state at the end
of each sweep bin, T = 1024 sweeps).  100 core graphs per condition (seed 2024), colours re-simulated with
gcg_kick (bit-identical to the core data).
Output: runs/agentspec/agent_spectra.json.  Run from revision_r1/analyses.
"""
import json, os
import multiprocessing as mp
import numpy as np
from gcg_kick import simulate

OUT = 'runs/agentspec'
TW = 1024
FR = 100
SEEDS = np.random.default_rng(2024).integers(0, 10**9, size=1000)[:100]


def spec(col):
    """col: T x N colours (1..3) -> S_f, f = 1..T/2."""
    S = np.zeros(col.shape[0] // 2)
    for c in (1, 2, 3):
        X = (col == c).astype(np.float64)
        F = np.fft.rfft(X, axis=0)[1:col.shape[0] // 2 + 1]
        S += (np.abs(F) ** 2).sum(axis=1)
    return S


def sweep_states(col, rate):
    """colour at the end of each sweep bin floor(t * rate)."""
    t = np.arange(col.shape[0])
    j = np.floor(t * rate + 1e-9).astype(int)
    last = np.flatnonzero(np.diff(j)) if j.size > 1 else np.array([], int)
    return col[last]


def worker(args):
    model, par, seed = args
    rate = (1 - par) if model == 'ARM' else 1.0
    r = simulate(dict(seed=int(seed), node_num=20, color_num=3, step_max=50000,
                      gridlock_rule='random' if model == 'ARM' else 'stay',
                      p_async=par if model == 'ARM' else 0.0, p_noise=par if model == 'SNM' else 0.0,
                      record_colors=True))
    col = r['history_colors']
    C = r['history_conflict_nodes']
    out = {'a_step': spec(col[:TW])}
    solved = np.flatnonzero(C == 0)
    t0 = (solved[0] + 1000) if solved.size else 1000
    if t0 + TW <= col.shape[0]:
        out['b_step'] = spec(col[t0:t0 + TW])
    sw = sweep_states(col, rate)
    if sw.shape[0] >= TW:
        out['a_sweep'] = spec(sw[:TW])
        t0s = int(np.floor(t0 * rate))
        if t0s + TW <= sw.shape[0]:
            out['b_sweep'] = spec(sw[t0s:t0s + TW])
    return out


def slope(S):
    f = np.arange(1, FR + 1)
    y = np.log10(S[:FR] + 1e-30)
    a, b = np.polyfit(np.log10(f), y, 1)
    return float(-a), float(np.mean((y - (a * np.log10(f) + b)) ** 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    conds = [('ARM', round(p, 2)) for p in np.arange(10) / 10 + 0.05] + [('SNM', e) for e in (0.05, 0.15, 0.25, 0.45)]
    res = {}
    ctx = mp.get_context('fork')
    with ctx.Pool(12) as pool:
        for model, par in conds:
            outs = pool.map(worker, [(model, par, s) for s in SEEDS], chunksize=4)
            r = {}
            for key in ('a_step', 'b_step', 'a_sweep', 'b_sweep'):
                Ss = [o[key] for o in outs if key in o]
                if not Ss:
                    continue
                per = np.array([slope(S) for S in Ss])
                a_mean, rss_mean = slope(np.mean(Ss, axis=0))
                r[key] = dict(n=len(Ss), alpha_of_mean_spectrum=a_mean, rss_of_mean=rss_mean,
                              alpha_trials_mean=float(per[:, 0].mean()), alpha_trials_sd=float(per[:, 0].std()),
                              alpha_trials_median=float(np.median(per[:, 0])))
            res[f'{model}_{par:.2f}'] = r
            print(model, par, {k: (round(v['alpha_of_mean_spectrum'], 3), round(v['alpha_trials_mean'], 3), round(v['alpha_trials_sd'], 3), v['n']) for k, v in r.items()}, flush=True)
    json.dump(res, open(f'{OUT}/agent_spectra.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
