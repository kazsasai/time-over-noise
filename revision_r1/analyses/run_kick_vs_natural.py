"""Large responses to an injected conflict compared with large natural unsolved episodes (Supplementary
Fig. S7): share of responses and episodes of at least 100 colour changes, and the median size and duration of
those, from the healed kicks of run_kick.py.  KS_logsize is the two-sample KS distance of log10 size.

Input: runs/kick/kick_raw_ARM_<p>.npz (run_kick.py).
Output: runs/kick/kick_vs_natural_large.json.  Run from revision_r1/analyses.
"""
import argparse, json
import numpy as np
from scipy.stats import ks_2samp

PS = ['0.05', '0.15', '0.25', '0.35', '0.45', '0.55', '0.65', '0.75', '0.85', '0.95']
THR = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/kick/kick_vs_natural_large.json')
    a = ap.parse_args()
    res = {}
    for p in PS:
        z = np.load(f'runs/kick/kick_raw_ARM_{p}.npz'); K = z['kicks']; ns = z['natural_size']; nd = z['natural_dur']
        h = K[:, 0] == 1; ks = K[h, 2]; kd = K[h, 1]
        kl, nl = ks[ks >= THR], ns[ns >= THR]
        kdl, ndl = kd[ks >= THR], nd[ns >= THR]
        D, _ = ks_2samp(np.log10(kl), np.log10(nl)) if nl.size > 20 else (np.nan, np.nan)
        r = dict(kick_large_share=float(np.mean(ks >= THR)), natural_large_share=float(np.mean(ns >= THR)),
                 n_natural_large=int(nl.size),
                 kick_large_median_size=float(np.median(kl)), natural_large_median_size=float(np.median(nl)) if nl.size else np.nan,
                 kick_large_median_dur=float(np.median(kdl)), natural_large_median_dur=float(np.median(ndl)) if ndl.size else np.nan,
                 KS_logsize=float(D))
        res[p] = r
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
