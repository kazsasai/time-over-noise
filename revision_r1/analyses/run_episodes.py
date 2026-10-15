"""
Episode view of resolution under the sweep clock (Supplementary Table S3).

Sweep clock and avalanches exactly as the primary r1 definition (avalanche_histograms.py, key "sweep").
An unsolved episode = maximal run of unsolved bins that ends with a solving bin (unsolved -> solved crossing);
its size = colour changes in the unsolved bins plus the solving bin.  Episodes starting after the first solution
are "stationary"; the first-passage episode is reported separately.
Where is the crossing?
  in_large / in_small : inside an avalanche of size >= s* / < s*
  after_large / after_small : in the quiet gap right after an avalanche that overlaps the episode
  quiet               : no avalanche overlaps the episode before the crossing (sub-threshold changes only)
s*(p) is taken from runs/aval/planC_sweep/planC.json (flicker < 1 % rule).  Episodes are "non-trivial" when their
size >= s*.  CIs: trials resampled with replacement (B = 1000).

Run from revision_r1/analyses:  python3 run_episodes.py
"""
import json
import numpy as np
import pandas as pd
from avalanche_histograms import bin_rate, segment

DATA = '../data/result_for_criticality_20_3.pkl'
CATS = ('in_large', 'after_large', 'in_small', 'after_small', 'quiet')
B = 1000


def episodes(A, Cs, rate, sstar):
    Ab, sb = bin_rate(A, Cs, rate)
    st, en, sz, *_ = segment(Ab, sb, 1)
    nb = Ab.size
    cross = np.where(sb[1:] & ~sb[:-1])[0] + 1
    first = int(np.argmax(sb)) if sb.any() else -1
    csum = np.concatenate([[0], np.cumsum(Ab)])
    rows = []
    for c in cross:
        k = c - 1
        while k >= 0 and not sb[k]:
            k -= 1
        start = k + 1                                     # first unsolved bin of the episode
        size = csum[c + 1] - csum[start]
        i = np.searchsorted(st, c, side='right') - 1     # last avalanche starting at or before c
        if i >= 0 and c < en[i]:
            cat = 'in_large' if sz[i] >= sstar else 'in_small'
        elif i >= 0 and en[i] > start:
            cat = 'after_large' if sz[i] >= sstar else 'after_small'
        else:
            cat = 'quiet'
        rows.append((start > first, size >= sstar, CATS.index(cat)))
    return rows


def main():
    S = {(r['model'], r['p']): r for r in json.load(open('runs/aval/planC_sweep/planC.json'))['summary']}
    df = pd.read_pickle(DATA)
    out = {}
    g_all = df[df['label'] == 'Async Model']
    for p, g in g_all.groupby(g_all['p_async'].round(2)):
        key = f'{p:.2f}'
        sstar = S[('ARM', key)]['s_star']
        rate = 1.0 - p
        # per-trial counts: [stationary non-trivial cats(5), stationary all cats(5), first-passage cats(5)]
        M = np.zeros((len(g), 15))
        for t, (A, cr) in enumerate(zip(g['history_changed'], g['history_conflict_ratio'])):
            A = np.asarray(A, dtype=np.int64)
            Cs = np.asarray(cr, dtype=np.float64) < 1e-12
            for stat, big, ci in episodes(A, Cs, rate, sstar):
                if stat:
                    M[t, 5 + ci] += 1
                    if big:
                        M[t, ci] += 1
                else:
                    M[t, 10 + ci] += 1
        rng = np.random.default_rng([2024, 99, int(round(p * 100))])
        res = dict(s_star=sstar, n_trials=len(g))
        for name, sl in (('nontrivial', slice(0, 5)), ('all', slice(5, 10)), ('first_passage', slice(10, 15))):
            tot = M[:, sl].sum(axis=0)
            res[f'{name}_n'] = float(tot.sum())
            if tot.sum() == 0:
                continue
            frac = tot / tot.sum()
            bs = []
            for _ in range(B):
                w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
                tb = w @ M[:, sl]
                bs.append(tb / tb.sum() if tb.sum() else np.full(5, np.nan))
            lo, hi = np.nanpercentile(np.array(bs), [2.5, 97.5], axis=0)
            for j, c in enumerate(CATS):
                res[f'{name}_{c}'] = [float(frac[j]), float(lo[j]), float(hi[j])]
            lg = np.array([b[0] + b[1] for b in bs])
            res[f'{name}_via_large'] = [float(frac[0] + frac[1]), *np.nanpercentile(lg, [2.5, 97.5]).tolist()]
        out[key] = res
        print(key, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in res.items()
                               if k.startswith(('nontrivial', 's_star', 'first_passage_via', 'all_via'))}), flush=True)
    json.dump(out, open('runs/aval/planC_sweep/episodes.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
