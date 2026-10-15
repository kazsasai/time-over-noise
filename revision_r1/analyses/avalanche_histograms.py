"""
Pass 1 of the r1 avalanche re-analysis: per-trial avalanche-size histograms from the rev1 core data.

Beggs–Plenz avalanches: A(t) = number of agents whose colour changed at step t (`history_changed`).
With bin width dt, A_b(j) = sum of A over bin j; an avalanche is a maximal run of bins with A_b > theta*dt,
and its size is the sum of A_b over the run.

Definitions computed (key = "th{theta}_dt{dt}"):
  th1_dt1 (primary), th0_dt1, th2_dt1, th3_dt1, th1_dt2, th1_dt4.
For every definition (and for the truncated records) we keep, per trial, histograms of the classes
  all / solved / resolve / open  (definitions below), and the same restricted to avalanches that touch neither the
  first nor the last bin of the record (key suffix "_nobound"; R3-3), and to avalanches that start after the first
  solution (key suffix "_stat"; first-passage episode excluded).
For the primary definition we additionally keep
  stat      : avalanches that start after the first solution (R3-1, R1-3)
  burn1k    : avalanches that start after a fixed burn-in of 1,000 steps (transient excluded)
  cross     : avalanches containing an unsolved -> solved crossing (C>0 at the bin before, C=0 inside)
  unsolved  : avalanches lying entirely in the unsolved phase (C>0 throughout)
  solved    : avalanches lying entirely in the solved phase (C=0 throughout)
  resolve   : not 'solved', and a solution is reached inside the avalanche or in the quiet gap before the next avalanche
              (the final solving move is often a single, sub-threshold change)
  open      : neither 'solved' nor 'resolve' (the system is still unsolved when the next avalanche starts)
and, for record-length tests (R3-3), the class histograms on records truncated to 10,000 and 25,000 steps.
Key "sweep" (core mode): time measured in sweeps (R1-7): ARM step t goes to bin floor(t (1-p)), so each bin holds
  on average N activations; threshold = one colour change per bin. For the SNM a step is already one sweep
  (rate 1), so "sweep" equals the primary definition there.  "sweep" is the PRIMARY definition for r1 (author
  decision 2026-10-08).  Sensitivity variants: sweep_th{0,2,3}_k1 (threshold per sweep), sweep_th1_k2 (bins of
  two sweeps), sweep_T{10000,25000} (records truncated, in steps), and the _nobound / _stat suffixes.
--mode multiN builds the primary-definition histograms of the multiN datasets (one file per N and p).

Output: <out>/hist_<model>_<p>.npz (or hist_ARM_N<N>_<p>.npz) with sparse (trial, size, count) triplets per key.
Run from revision_r1/analyses:
  python3 avalanche_histograms.py
  python3 avalanche_histograms.py --mode multiN --data ../../src/result_criticality_multiN_k3_long.pkl --out runs/aval_multiN_long
"""
import os, time
import numpy as np
import pandas as pd
from numba import njit

DATA = '../data/result_for_criticality_20_3.pkl'
OUT = 'runs/aval'
N = 20
DEFS = [(1, 1), (0, 1), (2, 1), (3, 1), (1, 2), (1, 4)]          # (theta, dt)
TRUNC = (10000, 25000)
BURN = 1000                                                      # fixed burn-in (steps)
SWEEP_DEFS = [(1, 1), (0, 1), (2, 1), (3, 1), (1, 2)]           # (theta per sweep, sweeps per bin)


@njit(cache=True)
def bin_fixed(A, Csolved, dt):
    """Sum activity over consecutive bins of dt steps; solved flag = state at the end of the bin."""
    nb = A.shape[0] // dt
    Ab = np.zeros(nb, dtype=np.int64)
    sb = np.zeros(nb, dtype=np.bool_)
    for j in range(nb):
        s = 0
        for t in range(j * dt, (j + 1) * dt):
            s += A[t]
        Ab[j] = s
        sb[j] = Csolved[(j + 1) * dt - 1]
    return Ab, sb


@njit(cache=True)
def bin_rate(A, Csolved, rate):
    """Sweep clock: step t belongs to bin floor(t * rate); with rate = 1 - p every bin holds on average
    (1 - p) N * 1/(1 - p) = N node activations.  Incomplete last bin dropped."""
    T = A.shape[0]
    nb = int(np.floor(T * rate + 1e-9))
    Ab = np.zeros(nb, dtype=np.int64)
    sb = np.zeros(nb, dtype=np.bool_)
    for t in range(T):
        j = int(np.floor(t * rate + 1e-9))
        if j >= nb:
            break
        Ab[j] += A[t]
        sb[j] = Csolved[t]            # overwritten until the last step of the bin
    return Ab, sb


@njit(cache=True)
def avalanches(A, Csolved, theta, dt):
    """Avalanches of the activity binned in dt steps, threshold theta per step (theta*dt per bin)."""
    Ab, sb = bin_fixed(A, Csolved, dt)
    return segment(Ab, sb, theta * dt)


@njit(cache=True)
def segment(Ab, sb, thr):
    """Avalanches = maximal runs of bins with Ab > thr. Returns (start_bin, end_bin_excl, size, has_cross,
    all_unsolved, all_solved, after_first_solution, n_bins, cross_incl_gap)."""
    nb = Ab.shape[0]
    first = -1
    for j in range(nb):
        if sb[j]:
            first = j
            break
    st_a = np.empty(nb, np.int64); en_a = np.empty(nb, np.int64); sz_a = np.empty(nb, np.int64)
    hc_a = np.empty(nb, np.bool_); au_a = np.empty(nb, np.bool_); as_a = np.empty(nb, np.bool_)
    af_a = np.empty(nb, np.bool_)
    n = 0
    j = 0
    while j < nb:
        if Ab[j] > thr:
            st = j
            size = 0
            hc = False
            au = True
            asv = True
            while j < nb and Ab[j] > thr:
                size += Ab[j]
                prev_solved = sb[j - 1] if j > 0 else False
                if sb[j] and not prev_solved:
                    hc = True
                if sb[j]:
                    au = False
                else:
                    asv = False
                j += 1
            st_a[n] = st; en_a[n] = j; sz_a[n] = size
            hc_a[n] = hc; au_a[n] = au; as_a[n] = asv; af_a[n] = first >= 0 and st > first
            n += 1
        else:
            j += 1
    # crossing (unsolved -> solved) inside the avalanche or in the quiet gap before the next avalanche
    cg_a = np.zeros(n, np.bool_)
    for i in range(n):
        hi = st_a[i + 1] if i + 1 < n else nb
        for j in range(max(st_a[i], 1), hi):
            if sb[j] and not sb[j - 1]:
                cg_a[i] = True
                break
    return st_a[:n], en_a[:n], sz_a[:n], hc_a[:n], au_a[:n], as_a[:n], af_a[:n], nb, cg_a


def add(store, key, trial, sizes):
    if sizes.size == 0:
        return
    u, c = np.unique(sizes, return_counts=True)
    store.setdefault(key, []).append(np.stack([np.full(u.size, trial), u, c]).astype(np.int64))


def classify(store, key, trial, A, Cs, theta, dt, extra=False, res=None):
    st, en, sz, hc, au, asv, aft, nb, cg = res if res is not None else avalanches(A, Cs, theta, dt)
    nob = (st > 0) & (en < nb)
    for suffix, m in (('', np.ones(sz.size, np.bool_)), ('_nobound', nob), ('_stat', aft)):
        add(store, f'{key}{suffix}|all', trial, sz[m])
        add(store, f'{key}{suffix}|solved', trial, sz[m & asv])
        add(store, f'{key}{suffix}|resolve', trial, sz[m & cg & ~asv])
        add(store, f'{key}{suffix}|open', trial, sz[m & ~cg & ~asv])
    if extra:
        add(store, f'{key}|stat', trial, sz[aft])
        add(store, f'{key}|burn1k', trial, sz[st >= BURN])
        add(store, f'{key}|cross', trial, sz[hc])
        add(store, f'{key}|unsolved', trial, sz[au])


def process(g, defs, trunc, rate=None):
    store = {}
    for trial, (A, cr) in enumerate(zip(g['history_changed'], g['history_conflict_ratio'])):
        A = np.asarray(A, dtype=np.int64)
        Cs = np.asarray(cr, dtype=np.float64) < 1e-12
        for theta, dt in defs:
            classify(store, f'th{theta}_dt{dt}', trial, A, Cs, theta, dt, extra=(theta, dt) == (1, 1))
        if rate is not None:                      # sweep clock (primary for r1), threshold per sweep
            for theta, k in SWEEP_DEFS:           # k = sweeps per bin; threshold theta*k per bin
                key = 'sweep' if (theta, k) == (1, 1) else f'sweep_th{theta}_k{k}'
                classify(store, key, trial, A, Cs, theta, k, res=segment(*bin_rate(A, Cs, rate / k), theta * k))
            for T in trunc:
                classify(store, f'sweep_T{T}', trial, A[:T], Cs[:T], 1, 1, res=segment(*bin_rate(A[:T], Cs[:T], rate), 1))
        for T in trunc:
            classify(store, f'th1_dt1_T{T}', trial, A[:T], Cs[:T], 1, 1)
    return {k: np.concatenate(v, axis=1) for k, v in store.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='core', choices=['core', 'multiN'])
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    df = pd.read_pickle(a.data)
    print(f'loaded {a.data}: {len(df):,} rows in {time.time() - t0:.0f}s', flush=True)
    if a.mode == 'core':
        groups = []
        for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
            sub = df[df['label'] == lab]
            groups += [(f'{model}_{p:.2f}', g, (1.0 - p) if model == 'ARM' else 1.0)
                       for p, g in sub.groupby(sub[col].round(2))]
        defs, trunc = DEFS, TRUNC
    else:                                   # ARM only, one file per (N, p); primary definition only
        groups = [(f'ARM_N{N}_{p:.2f}', g, None) for (N, p), g in df.groupby([df['node_num'], df['p_async'].round(2)])]
        defs, trunc = [(1, 1)], ()
    for name, g, rate in groups:
        t1 = time.time()
        arrays = process(g, defs, trunc, rate)
        np.savez_compressed(f'{a.out}/hist_{name}.npz', n_trials=len(g),
                            n_steps=int(len(g['history_changed'].iloc[0])), **arrays)
        print(f'{name}: {len(g)} trials, primary avalanches {arrays["th1_dt1|all"][2].sum():,} '
              f'({time.time() - t1:.0f}s)', flush=True)
    del df


if __name__ == '__main__':
    main()
