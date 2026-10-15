"""
A7: spectral checks (R1-4, R3-2) on the rev1 core data (N=20, k=3, 50,000 steps, 1000 trials per condition).
Numbers only; the 1/f interpretation is left to the authors.

Signal: C(t) = history_conflict_ratio.  beta is the exponent of the AMPLITUDE spectrum |C^(f)| ~ f^-beta
(power exponent = 2 beta).
 (a) fig4 method, reproduced: mean over trials of log10|FFT(C)| over the full record, OLS slope in
     log10 f in [-3.5, -1.0] (f in cycles per step).
 (b) the same, split by trial class (raw step series): never_leave = solved and C == 0 at every step after the first
     solution; other = everything else.
 (c) stationary window = last 2^15 steps; trials whose window is constant are excluded (share reported).
     Two estimators: mean of log amplitudes (fig4 style) and the standard one (arithmetic mean of power over
     trials, slope of its log; beta = slope_power / 2).  Fit windows in steps and, by rescaling the frequency axis
     f_sweep = f_step / (1-p), in sweeps.  Trial bootstrap (B=200) for the standard estimator.
 (d) binarised X(t) = [C(t) > 0] on the same window, and two non-critical comparisons per trial:
     telegraph  : alternating exponential durations with the trial's mean solved / unsolved run lengths
     shuffled   : the trial's own solved and unsolved run lengths, each shuffled (order destroyed)
Output: runs/spec/spectra.json.  Run from revision_r1/analyses:  python3 run_spectra.py
"""
import json, os
import numpy as np
import pandas as pd

DATA = '../data/result_for_criticality_20_3.pkl'
OUT = 'runs/spec'
L = 2 ** 15
B = 200
WIN_STEP = {'orig[-3.5,-1.0]': (-3.5, -1.0), '[-3.0,-1.0]': (-3.0, -1.0), '[-3.5,-2.0]': (-3.5, -2.0),
            '[-2.5,-1.0]': (-2.5, -1.0), '[-3.0,-1.5]': (-3.0, -1.5)}
WIN_SWEEP = {'[-3.0,-1.0]': (-3.0, -1.0), '[-3.0,-2.0]': (-3.0, -2.0), '[-2.5,-1.0]': (-2.5, -1.0),
             '[-2.0,-0.5]': (-2.0, -0.5)}


def fit(logf, logy, win):
    m = (logf >= win[0]) & (logf <= win[1])
    if m.sum() < 5:
        return np.nan, np.nan
    sl, ic = np.polyfit(logf[m], logy[m], 1)
    pred = sl * logf[m] + ic
    r2 = 1 - np.sum((logy[m] - pred) ** 2) / np.sum((logy[m] - logy[m].mean()) ** 2)
    return float(-sl), float(r2)


def runs_of(x):
    edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    return x[starts], np.diff(np.concatenate([starts, [x.size]]))


def rebuild(first_val, sol_lens, uns_lens, n):
    """Alternating series starting with first_val, solved runs (X=0) and unsolved runs (X=1), cut to n."""
    out = np.empty(0, np.int8)
    si = ui = 0
    cur = first_val
    parts = []
    tot = 0
    while tot < n and (si < len(sol_lens) or ui < len(uns_lens)):
        if cur:            # unsolved
            if ui >= len(uns_lens):
                cur = 0
                continue
            ln = uns_lens[ui]; ui += 1
        else:
            if si >= len(sol_lens):
                cur = 1
                continue
            ln = sol_lens[si]; si += 1
        parts.append(np.full(int(ln), cur, np.int8))
        tot += int(ln)
        cur = 1 - cur
    out = np.concatenate(parts) if parts else np.zeros(n, np.int8)
    if out.size < n:
        out = np.concatenate([out, np.full(n - out.size, out[-1] if out.size else 0, np.int8)])
    return out[:n]


def surrogates(x, rng):
    v, ln = runs_of(x)
    sol, uns = ln[v == 0], ln[v == 1]
    shuf = rebuild(int(v[0]), rng.permutation(sol), rng.permutation(uns), x.size)
    ms = sol.mean() if sol.size else x.size
    mu = uns.mean() if uns.size else 1.0
    # telegraph: draw enough exponential durations (at least 1 step)
    k = int(x.size / max(min(ms, mu), 1.0)) + 10
    tel = rebuild(int(v[0]), np.maximum(1, np.round(rng.exponential(ms, k))).astype(int),
                  np.maximum(1, np.round(rng.exponential(mu, k))).astype(int), x.size)
    return shuf, tel


def power_beta(P, logf, wins, rng=None, boot=False):
    """P: trials x freq power; standard estimator on the mean power."""
    mean = P.mean(axis=0)
    res = {k: fit(logf, np.log10(mean), w) for k, w in wins.items()}
    out = {k: dict(beta=b / 2, r2=r) for k, (b, r) in res.items()}
    if boot:
        n = P.shape[0]
        k0 = next(iter(wins))
        bs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, n, n), minlength=n) / n
            bs.append(fit(logf, np.log10(w @ P), wins[k0])[0] / 2)
        out[k0]['ci'] = np.percentile(bs, [2.5, 97.5]).tolist()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_pickle(DATA)
    T = len(df['history_conflict_ratio'].iloc[0])
    f_full = np.arange(1, T // 2) / T
    lf_full = np.log10(f_full)
    f_win = np.fft.rfftfreq(L)[1:]
    lf_win = np.log10(f_win)
    res = {}
    for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
        sub = df[df['label'] == lab]
        for p, g in sub.groupby(sub[col].round(2)):
            rate = (1 - p) if model == 'ARM' else 1.0
            rng = np.random.default_rng([2024, 5, int(round(p * 100)), model == 'ARM'])
            C = np.stack([np.asarray(c, dtype=np.float64) for c in g['history_conflict_ratio']])
            # (a) fig4 method, full record
            logamp = np.log10(np.abs(np.fft.fft(C, axis=1))[:, 1:T // 2] + 1e-12)
            r = dict(n=int(C.shape[0]))
            r['fig4_full'] = fit(lf_full, logamp.mean(axis=0), WIN_STEP['orig[-3.5,-1.0]'])
            # (b) class split
            solved0 = C < 1e-12
            nl = np.array([s.any() and s[int(np.argmax(s)):].all() for s in solved0])
            r['share_never_leave'] = float(nl.mean())
            r['fig4_full_never_leave'] = fit(lf_full, logamp[nl].mean(axis=0), WIN_STEP['orig[-3.5,-1.0]']) if nl.any() else None
            r['fig4_full_other'] = fit(lf_full, logamp[~nl].mean(axis=0), WIN_STEP['orig[-3.5,-1.0]']) if (~nl).any() else None
            del logamp
            # (c) stationary window
            W = C[:, -L:]
            keep = W.std(axis=1) > 0
            r['window_share_constant'] = float(1 - keep.mean())
            if keep.sum() >= 20:
                Fw = np.fft.rfft(W[keep], axis=1)[:, 1:]
                amp = np.abs(Fw)
                P = amp ** 2
                r['win_meanlog_step'] = {k: fit(lf_win, np.log10(amp + 1e-12).mean(axis=0), w) for k, w in WIN_STEP.items()}
                r['win_power_step'] = power_beta(P, lf_win, WIN_STEP, rng, boot=True)
                lf_sw = lf_win - np.log10(rate)
                r['win_power_sweep'] = power_beta(P, lf_sw, WIN_SWEEP, rng, boot=True)
                del Fw, amp, P
                # (d) binarised and surrogates
                X = (W[keep] > 1e-12).astype(np.int8)
                Ps = {'X_real': [], 'X_shuffled': [], 'X_telegraph': []}
                for x in X:
                    sh, te = surrogates(x, rng)
                    for name, s in (('X_real', x), ('X_shuffled', sh), ('X_telegraph', te)):
                        Ps[name].append(np.abs(np.fft.rfft(s.astype(np.float64))[1:]) ** 2)
                for name, lst in Ps.items():
                    Pm = np.array(lst)
                    r[f'{name}_step'] = power_beta(Pm, lf_win, WIN_STEP)
                    r[f'{name}_sweep'] = power_beta(Pm, lf_win - np.log10(rate), WIN_SWEEP)
            res[f'{model}_{p:.2f}'] = r
            b = r['win_power_sweep']['[-3.0,-1.0]']['beta'] if 'win_power_sweep' in r else np.nan
            print(f"{model} {p:.2f}: fig4 beta {r['fig4_full'][0]:.3f} (R2 {r['fig4_full'][1]:.2f}) | never-leave {r['share_never_leave']:.2f} "
                  f"beta NL {r['fig4_full_never_leave'][0] if r['fig4_full_never_leave'] else np.nan:.3f} other {r['fig4_full_other'][0] if r['fig4_full_other'] else np.nan:.3f} "
                  f"| window const {r['window_share_constant']:.2f} | power-est sweep[-3,-1] {b:.3f}", flush=True)
            del C, W
    json.dump(res, open(f'{OUT}/spectra.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
