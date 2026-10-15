"""
Branching ratio m by the multistep-regression (MR) estimator (Wilting & Priesemann 2018) on the activity A
in the sweep clock (primary r1 clock), separately for the search phase (bins up to the first solution) and the
stationary phase (bins after the first solution).  rk = Cov(A_t, A_{t+k}) / Var(A_t) pooled over all pairs that
lie inside the same phase segment of the same trial; fit rk = b m^k over k = 1..kmax.  m < 1 subcritical
(perturbations decay), m ~ 1 marginal, m > 1 supercritical.  tau = -1 / ln m (sweeps).
Each segment is demeaned first (DEMEAN; within-trial covariance).  Trial bootstrap (B=200).
Output: runs/branching/branching.json (by trial class: run_branching_by_class.py).  Run from revision_r1/analyses.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from avalanche_histograms import bin_rate

warnings.filterwarnings('ignore')
OUT = 'runs/branching'
KMAX = {'search': 10, 'stationary': 40}
B = 200


DEMEAN = True     # subtract each segment's own mean (between-trial differences in mean activity would otherwise
                  # add a lag-independent covariance and push m towards 1)


def sums(x, kmax):
    """per-k sufficient statistics for pairs (x_t, x_{t+k}) inside one segment: n, Sx, Sy, Sxx, Sxy."""
    S = np.zeros((kmax, 5))
    x = x.astype(float)
    if DEMEAN and x.size:
        x = x - x.mean()
    for k in range(1, kmax + 1):
        if x.size <= k:
            break
        a, b = x[:-k], x[k:]
        S[k - 1] = (a.size, a.sum(), b.sum(), (a * a).sum(), (a * b).sum())
    return S


def mr_fit(S):
    n, sx, sy, sxx, sxy = S.T
    ok = n > 10
    if ok.sum() < 3:
        return np.nan, np.nan, None
    mx, my = sx[ok] / n[ok], sy[ok] / n[ok]
    var = sxx[ok] / n[ok] - mx ** 2
    rk = (sxy[ok] / n[ok] - mx * my) / var
    k = np.arange(1, S.shape[0] + 1)[ok]
    try:
        (b, m), _ = curve_fit(lambda kk, b, m: b * m ** kk, k, rk, p0=[rk[0], 0.9], bounds=([-10, 0], [10, 2]))
    except Exception:
        return np.nan, np.nan, rk
    return float(m), float(b), rk


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    res = {}
    for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
        sub = df[df['label'] == lab]
        for p, g in sub.groupby(sub[col].round(2)):
            rate = (1 - p) if model == 'ARM' else 1.0
            per = {ph: [] for ph in KMAX}
            for A, cr in zip(g['history_changed'], g['history_conflict_ratio']):
                A = np.asarray(A, dtype=np.int64)
                s = np.asarray(cr, dtype=np.float64) < 1e-12
                Ab, sb = bin_rate(A, s, rate)
                f = int(np.argmax(sb)) if sb.any() else Ab.size
                per['search'].append(sums(Ab[:f + 1], KMAX['search']))
                per['stationary'].append(sums(Ab[f + 1:], KMAX['stationary']) if f + 1 < Ab.size else np.zeros((KMAX['stationary'], 5)))
            r = {}
            rng = np.random.default_rng([2024, 31, int(round(p * 100)), model == 'ARM'])
            for ph, L in per.items():
                Sall = np.array(L)                       # trials x k x 5
                m, b, rk = mr_fit(Sall.sum(axis=0))
                n_tr = Sall.shape[0]
                bs = []
                for _ in range(B):
                    w = np.bincount(rng.integers(0, n_tr, n_tr), minlength=n_tr)
                    bs.append(mr_fit(np.tensordot(w, Sall, axes=1))[0])
                bs = np.array(bs, dtype=float)
                r[ph] = dict(m=m, b=b, m_ci=np.nanpercentile(bs, [2.5, 97.5]).tolist(),
                             tau_sweeps=float(-1 / np.log(m)) if 0 < m < 1 else np.nan,
                             r1=float(rk[0]) if rk is not None else np.nan,
                             n_pairs_k1=float(Sall.sum(axis=0)[0, 0]))
            res[f'{model}_{p:.2f}'] = r
            print(f"{model} {p:.2f}: search m={r['search']['m']:.3f} {np.round(r['search']['m_ci'], 3)} | "
                  f"stationary m={r['stationary']['m']:.3f} {np.round(r['stationary']['m_ci'], 3)} tau={r['stationary']['tau_sweeps']:.1f} sw", flush=True)
    json.dump(res, open(f'{OUT}/branching.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
