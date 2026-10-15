"""Mann-Whitney comparison of the times to the first solution at the representative values (Results; Supplementary Table S1): SNM eps = 0.05 vs ARM p = 0.45, and the optima of the fine grid (eps = 0.06 vs p = 0.47), in
rounds and in sweeps.  Solved trials only.  P_snm_slower = U / (n_x n_y), the probability that an SNM time exceeds an
ARM time (ties counted half).
Paired comparison (Results; Methods 5): in the long records every trial is solved, and trials with the same seed share
the graph and the initial colouring, so the SNM and ARM times are also compared trial by trial in sweeps: median of the
per-trial differences (SNM - ARM) with a bootstrap interval over trials (B replicates), and the two-sided Wilcoxon
signed-rank test.

Input: runs/fpt/survival_at_optima.json (run_first_passage.py).
Output: runs/fpt/mannwhitney.json.  Run from revision_r1/analyses.
"""
import argparse, json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

PAIRS = [('fine_SNM_0.05', 'fine_ARM_0.45'), ('fine_SNM_0.06', 'fine_ARM_0.47'), ('long_SNM_0.05', 'long_ARM_0.45')]
PAIRED = [('long_SNM_0.05', 'long_ARM_0.45')]
B = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/fpt/mannwhitney.json')
    a = ap.parse_args()
    S = json.load(open('runs/fpt/survival_at_optima.json'))
    out = {}
    for x_key, y_key in PAIRS:
        x, y = np.array(S[x_key]['T']), np.array(S[y_key]['T'])
        for unit, fx, fy in (('rounds', 1, 1), ('sweeps', S[x_key]['rate'], S[y_key]['rate'])):
            X, Y = x * fx, y * fy
            U, pv = mannwhitneyu(X, Y, alternative='two-sided')
            cl = U / (X.size * Y.size)
            print(f'{x_key} vs {y_key} [{unit}]: n={X.size}/{Y.size} median {np.median(X):.1f} vs {np.median(Y):.1f} | '
                  f'U={U:.0f} p={pv:.3g} | P(SNM slower)={cl:.3f}')
            out[f'{x_key}|{y_key}|{unit}'] = dict(nx=int(X.size), ny=int(Y.size), med_x=float(np.median(X)), med_y=float(np.median(Y)),
                                                 U=float(U), p=float(pv), P_snm_slower=float(cl))
    for x_key, y_key in PAIRED:
        sx, sy = S[x_key], S[y_key]
        assert sx['seed'] == sy['seed']
        X = np.array(sx['T_by_seed'], float) * sx['rate']
        Y = np.array(sy['T_by_seed'], float) * sy['rate']
        assert np.isfinite(X).all() and np.isfinite(Y).all()
        D = X - Y
        W, pv = wilcoxon(X, Y, alternative='two-sided')
        rng = np.random.default_rng(2024)
        lo, hi = np.percentile(np.median(D[rng.integers(0, D.size, (B, D.size))], axis=1), [2.5, 97.5])
        print(f'{x_key} vs {y_key} [sweeps, paired]: n={D.size} median difference {np.median(D):.2f} [{lo:.2f}, {hi:.2f}] | '
              f'Wilcoxon W={W:.0f} p={pv:.3g}')
        out[f'{x_key}|{y_key}|sweeps|paired'] = dict(n=int(D.size), median_diff=float(np.median(D)), median_diff_lo=float(lo),
                                                      median_diff_hi=float(hi), W=float(W), p=float(pv),
                                                      P_snm_slower=float((D > 0).mean() + 0.5 * (D == 0).mean()))
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
