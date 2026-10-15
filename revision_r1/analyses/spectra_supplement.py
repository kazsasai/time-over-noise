"""
A7 supplement: (i) renewal surrogate of C itself (unsolved excursions kept whole with their C values, solved runs
and excursions each shuffled in order); (ii) how much the standard (mean-power) estimator depends on a few trials:
beta after removing the top 5 % / 10 % of trials by window variance; (iii) per-trial slope distribution.
Stationary window = last 2^15 steps, trials with a constant window excluded (as in run_spectra.py).
Output: runs/spec/spectra_supplement.json.  Run from revision_r1/analyses.
"""
import json
import numpy as np
import pandas as pd
from run_spectra import fit, runs_of, L

WIN = (-3.0, -1.0)           # sweep units


def c_shuffle(c, rng):
    s = c < 1e-12
    v, ln = runs_of(s)
    starts = np.concatenate([[0], np.cumsum(ln)[:-1]])
    segs_u = [c[a:a + l] for a, l, vv in zip(starts, ln, v) if not vv]
    segs_s = [l for l, vv in zip(ln, v) if vv]
    iu = rng.permutation(len(segs_u)); isv = rng.permutation(len(segs_s))
    out, ku, ks, cur = [], 0, 0, bool(v[0])
    while ku < len(segs_u) or ks < len(segs_s):
        if cur and ks < len(segs_s):
            out.append(np.zeros(segs_s[isv[ks]])); ks += 1
        elif not cur and ku < len(segs_u):
            out.append(segs_u[iu[ku]]); ku += 1
        cur = not cur
    return np.concatenate(out)[:c.size]


def main():
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    lf = np.log10(np.fft.rfftfreq(L)[1:])
    res = {}
    sub = df[df['label'] == 'Async Model']
    for p, g in sub.groupby(sub['p_async'].round(2)):
        if round(p, 2) not in (0.05, 0.25, 0.45, 0.65, 0.85):
            continue
        rate = 1 - p
        rng = np.random.default_rng([2024, 6, int(round(p * 100))])
        W = np.stack([np.asarray(c, dtype=np.float64)[-L:] for c in g['history_conflict_ratio']])
        W = W[W.std(axis=1) > 0]
        P = np.abs(np.fft.rfft(W, axis=1)[:, 1:]) ** 2
        lfs = lf - np.log10(rate)
        r = dict(n=int(W.shape[0]), beta_meanpower=fit(lfs, np.log10(P.mean(0)), WIN)[0] / 2)
        var = W.var(axis=1)
        for q in (0.95, 0.90, 0.50):
            k = var <= np.quantile(var, q)
            r[f'beta_meanpower_below_q{int(q*100)}'] = fit(lfs, np.log10(P[k].mean(0)), WIN)[0] / 2
        tot = P.sum(axis=1); order = np.argsort(tot)[::-1]
        r['power_share_top5pct_trials'] = float(tot[order[:max(1, len(order) // 20)]].sum() / tot.sum())
        per = np.array([fit(lfs, np.log10(pp + 1e-30), WIN)[0] / 2 for pp in P])
        r['per_trial_beta_median_iqr'] = [float(np.median(per)), *np.percentile(per, [25, 75]).tolist()]
        Ps = np.array([np.abs(np.fft.rfft(c_shuffle(c, rng))[1:]) ** 2 for c in W])
        r['beta_C_shuffled_meanpower'] = fit(lfs, np.log10(Ps.mean(0)), WIN)[0] / 2
        res[f'{p:.2f}'] = r
        print(f'{p:.2f}', {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()}, flush=True)
    json.dump(res, open('runs/spec/spectra_supplement.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
