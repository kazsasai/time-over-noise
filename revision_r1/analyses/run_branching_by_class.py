"""Branching ratio after the first solution, by trial class (Supplementary Fig. S7): ARM trials that never leave the
consistent state after their first solution, trials that leave it, and the longest consistent stretches (over 60
sweeps) of the latter.  Same estimator (multistep regression on the sweep-clock activity, k = 1..40) and the same
per-segment demeaning as run_branching.py; no bootstrap.  Each entry is (m, number of segments, timescale -1/ln m).

Input: ../data/result_for_criticality_20_3.pkl.
Output: runs/branching/branching_by_class.json.  Run from revision_r1/analyses.
"""
import argparse, json
import numpy as np
import pandas as pd
from avalanche_histograms import bin_rate
from run_branching import sums, mr_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/branching/branching_by_class.json')
    a = ap.parse_args()
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    out = {}
    sub = df[df.label == 'Async Model']
    for p in (0.05, 0.25, 0.45, 0.65, 0.85):
        g = sub[(sub.p_async - p).abs() < 1e-6]
        S = {'never_leave': [], 'leaving': [], 'leaving_solved_bins_only': []}
        for A, cr in zip(g['history_changed'], g['history_conflict_ratio']):
            A = np.asarray(A, dtype=np.int64); s = np.asarray(cr) < 1e-12
            Ab, sb = bin_rate(A, s, 1 - p)
            if not sb.any():
                continue
            f = int(np.argmax(sb)); post = Ab[f + 1:]; spost = sb[f + 1:]
            if s[int(np.argmax(s)):].all():
                S['never_leave'].append(sums(post, 40))
            else:
                S['leaving'].append(sums(post, 40))
                # consistent stretches longer than 60 sweeps (changes inside the solved state, no excursion)
                e = np.flatnonzero(np.diff(spost.astype(np.int8))) + 1
                st = np.concatenate([[0], e]); ln = np.diff(np.concatenate([st, [spost.size]]))
                for a0, l in zip(st, ln):
                    if spost[a0] and l > 60:
                        S['leaving_solved_bins_only'].append(sums(post[a0:a0 + l], 40))
        r = {}
        for k, L in S.items():
            if L:
                m, b, rk = mr_fit(np.sum(L, axis=0))
                r[k] = (round(m, 3), len(L), round(float(-1 / np.log(m)), 1) if 0 < m < 1 else None)
        out[p] = r
        print(p, r)
    json.dump({str(k): v for k, v in out.items()}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
