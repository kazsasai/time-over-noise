"""
MLE on exactly the paper's large-scale region: same data and avalanche definition as fig4 (step clock, theta=1),
all avalanches with size above the knee x_k of the CCDF fit (Methods; Supplementary Table S7), p = 0.05-0.65.
(1) per-p MLE of PL / TPL / LN / SE / EXP; TPL alpha (pdf exponent) and cutoff 1/lambda with trial bootstrap;
(2) is the shape common?  joint TPL fit with one alpha for all p and a separate lambda per p, vs separate alphas.
Output: runs/aval/hump_same_region.json.  Run from revision_r1/analyses.
"""
import json
import numpy as np
from scipy import optimize
import aval_mle as am

XK = {'0.05': 2.389, '0.15': 2.117, '0.25': 1.875, '0.35': 1.653, '0.45': 1.447, '0.55': 1.255, '0.65': 1.079}
TAU_OLS = {'0.05': 1.411, '0.15': 1.383, '0.25': 1.311, '0.35': 1.312, '0.45': 1.420, '0.55': 1.445, '0.65': 1.223}
B = 200
res, data = {}, {}
for p, xk in XK.items():
    with np.load(f'runs/aval/hist_ARM_{p}.npz') as z:
        h = am.Hist(z, 'th1_dt1|all')
    s, c = h.pooled()
    xmin = int(np.floor(10 ** xk)) + 1                       # fig4 uses s > 10^xk
    st, ct = am.tail_of(s, c.astype(float), xmin)
    f, out = am.fit_all(st, ct, xmin)
    aic = {m: am.ic(r, f.n)['aic'] for m, r in out.items()}
    best = min(aic, key=aic.get)
    a, u = out['TPL']['theta']
    rng = np.random.default_rng([2024, 77, int(float(p) * 100)])
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, h.n_trials, h.n_trials), minlength=h.n_trials).astype(float)
        sb, cb = h.pooled(w)
        sbt, cbt = am.tail_of(sb, cb, xmin)
        rb = am.Fitter(sbt, cbt, xmin).fit('TPL', x0s=[[a, u]])
        bs.append([rb['theta'][0], np.exp(-rb['theta'][1])])
    bs = np.array(bs)
    res[p] = dict(xmin=xmin, n_tail=float(ct.sum()), share_of_all=float(ct.sum() / c.sum()), s_max=int(st.max()),
                  decades=float(np.log10(st.max() / xmin)), tau_OLS_paper=TAU_OLS[p], best_MLE=best,
                  dAIC={m: round(aic[m] - aic[best], 1) for m in aic},
                  TPL_alpha=float(a), TPL_alpha_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                  TPL_cutoff=float(np.exp(-u)), TPL_cutoff_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                  PL_alpha=float(out['PL']['theta'][0]),
                  KS={m: round(am.ks_distance(f, out[m]), 4) for m in ('PL', 'TPL', 'LN', 'SE')})
    data[p] = (f, out['TPL']['ll'])
    r = res[p]
    print(f"p={p}: xmin={xmin} n={r['n_tail']:.0f} ({r['share_of_all']:.1e}) range {r['decades']:.2f} decades | paper tau(OLS) {r['tau_OLS_paper']} "
          f"| MLE best {best} dAIC {r['dAIC']} | TPL alpha {a:.2f} [{r['TPL_alpha_ci'][0]:.2f},{r['TPL_alpha_ci'][1]:.2f}] "
          f"cutoff {r['TPL_cutoff']:.0f} [{r['TPL_cutoff_ci'][0]:.0f},{r['TPL_cutoff_ci'][1]:.0f}] | PL alpha {r['PL_alpha']:.2f} | KS {r['KS']}", flush=True)

# joint fit: common alpha, separate lambda per p
def profile(alpha):
    tot, lams = 0.0, {}
    for p, (f, _) in data.items():
        r = optimize.minimize_scalar(lambda u: f.nll('TPL', np.array([alpha, u])), bounds=(-30, 3), method='bounded')
        tot += -r.fun; lams[p] = float(np.exp(-r.x))
    return tot, lams
grid = np.linspace(-0.5, 2.0, 51)
prof = [profile(a)[0] for a in grid]
i = int(np.argmax(prof))
r = optimize.minimize_scalar(lambda a: -profile(a)[0], bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method='bounded')
a_common = float(r.x); ll_common, lams = profile(a_common)
ll_sep = sum(ll for _, ll in data.values())
n_tot = sum(f.n for f, _ in data.values())
k_c, k_s = 1 + len(data), 2 * len(data)
res['joint'] = dict(alpha_common=a_common, cutoffs=lams, ll_common=ll_common, ll_separate=ll_sep,
                    dLL=ll_sep - ll_common, dLL_per_avalanche=(ll_sep - ll_common) / n_tot,
                    dAIC_common_minus_separate=2 * (k_c - k_s) - 2 * (ll_common - ll_sep),
                    dBIC_common_minus_separate=(k_c - k_s) * np.log(n_tot) - 2 * (ll_common - ll_sep))
print('joint:', json.dumps(res['joint'], indent=None, default=float))
json.dump(res, open('runs/aval/hump_same_region.json', 'w'), indent=1, default=float)
