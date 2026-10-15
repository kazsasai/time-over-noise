"""Lower-cut sensitivity of the maximum-likelihood TPL exponent on the range above the CCDF knee (step clock;
Supplementary Fig. S5): the lower end is moved to 1, 1.5, 2 and 3 times the knee, and the local exponent of the
log-binned size distribution is computed along the range.

Input: runs/aval/hist_ARM_<p>.npz (avalanche_histograms.py), runs/aval/hump_same_region.json (hump_same_region.py).
Output: runs/aval/hump_lowercut_sensitivity.json.  Run from revision_r1/analyses with OMP_NUM_THREADS=1.
"""
import argparse, json, warnings
import numpy as np
import aval_mle as am

warnings.filterwarnings('ignore')
PS = ['0.05', '0.15', '0.25', '0.35', '0.45', '0.55', '0.65']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/aval/hump_lowercut_sensitivity.json')
    a = ap.parse_args()
    R = json.load(open('runs/aval/hump_same_region.json'))
    out = {}
    for p in PS:
        with np.load(f'runs/aval/hist_ARM_{p}.npz') as z:
            h = am.Hist(z, 'th1_dt1|all')
        s, c = h.pooled(); c = c.astype(float)
        x0 = R[p]['xmin']; rows = []
        for fac in (1, 1.5, 2, 3):
            xm = int(x0 * fac); st, ct = am.tail_of(s, c, xm)
            f = am.Fitter(st, ct, xm); r = f.fit('TPL')
            rows.append(dict(factor=fac, xmin=xm, n_tail=float(ct.sum()), alpha=float(r['theta'][0]),
                             cutoff=float(np.exp(-r['theta'][1])), ks=am.ks_distance(f, r)))
        # local exponent of the pdf along the range (logarithmic bins)
        edges = np.unique(np.round(np.logspace(np.log10(x0), np.log10(s.max() + 1), 14)).astype(int))
        cnt = np.array([c[(s >= lo) & (s < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
        mid = np.sqrt(edges[:-1] * (edges[1:] - 1)); pmf = cnt / np.diff(edges) / c.sum(); ok = cnt >= 30
        loc = -np.gradient(np.log(pmf[ok]), np.log(mid[ok]))
        out[p] = dict(knee_xmin=x0, tau_OLS_paper=R[p]['tau_OLS_paper'], lower_cut=rows,
                      local_exponent=dict(size=mid[ok].tolist(), exponent=loc.tolist()))
        print(p, [(r['factor'], round(r['alpha'], 2)) for r in rows])
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
