"""Maximum-likelihood fits of each avalanche population separately (R1-3: analyse the solved and the unsolved phase
separately rather than as one distribution).  Sweep clock, threshold 1.

Populations: 'solved' (avalanches within the solved state) and 'unsolved' (avalanches that begin in an unsolved
state: the resolve and open classes together).  Same five candidates, same x_min rule (KS-optimal power law with at
least 500 avalanches in the tail, extended grid) and the same trial bootstrap (B = 200, x_min held fixed) as the
pooled fits in run_aval_mle.py; no broken power law, because a single population needs no knee.

Output: runs/aval/mle_pop/point.json and boot.json.  Run from revision_r1/analyses with
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
"""
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import aval_mle as am
import run_aval_mle as r

warnings.filterwarnings('ignore')
OUT = f'{r.HIST}/mle_pop'
KEYS = {'solved': ['sweep|solved'], 'unsolved': ['sweep|resolve', 'sweep|open']}


class _Combined(dict):
    """npz-like view in which several class keys are concatenated under one population key."""
    @property
    def files(self):
        return list(self)


def load(model, p, pop):
    with np.load(f'{r.HIST}/hist_{model}_{p}.npz') as z:
        parts = [z[k] for k in KEYS[pop] if k in z.files]
        d = _Combined(n_trials=z['n_trials'])
        if parts:
            d[pop] = np.concatenate(parts, axis=1)
        return am.Hist(d, pop)


def point(args):
    model, p, pop = args
    h = load(model, p, pop)
    s, c = h.pooled(); c = c.astype(float)
    res = dict(model=model, p=p, pop=pop, n_trials=h.n_trials, n_total=float(c.sum()), s_max=int(s.max()) if s.size else 0)
    if c.sum() < 50:
        return res
    xmins = {'full': int(s.min())}
    cl = am.clauset_xmin(s, c, r.MIN_TAIL, extend=True)
    if cl is not None:
        xmins['clauset'] = cl[0]
        res['clauset_D_PL'] = cl[1]
    res['xmin'] = xmins
    res['fits'] = {}
    for name, x in xmins.items():
        f, out = r.fits_at(s, c, x)
        if out is None:
            continue
        d = dict(xmin=x, n_tail=float(f.n))
        for m, o in out.items():
            d[m] = dict(theta=o['theta'].tolist(), ll=o['ll'], **am.ic(o, f.n), ks=am.ks_distance(f, o),
                        at_bound=bool(np.any([np.isclose(v, lo) or np.isclose(v, hi) for v, (lo, hi)
                                              in zip(o['theta'], am.MODELS[m]['bounds'])])))
        d['best_aic'] = min(out, key=lambda m: d[m]['aic'])
        res['fits'][name] = d
    return res


def boot(args):
    model, p, pop, xmins, B, chunk = args
    h = load(model, p, pop)
    rng = np.random.default_rng([2024, ['SNM', 'ARM'].index(model), r.PS.index(p), ['solved', 'unsolved'].index(pop), chunk])
    rows = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, h.n_trials, h.n_trials), minlength=h.n_trials).astype(float)
        s, c = h.pooled(w)
        row = {}
        for name, x in xmins.items():
            f, out = r.fits_at(s, c, x)
            if out is None:
                continue
            row[name] = {m: (out[m]['theta'].tolist(), out[m]['ll']) for m in r.MODELS}
            row[name]['n_tail'] = float(f.n)
        rows.append(row)
    return model, p, pop, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=20)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    conds = [(m, p, pop) for m in ('ARM', 'SNM') for p in r.PS for pop in ('solved', 'unsolved')]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        points = list(ex.map(point, conds))
    json.dump(points, open(f'{OUT}/point.json', 'w'), indent=1, default=float)
    print(f'point estimates done ({time.time() - t0:.0f}s)', flush=True)
    for q in points:
        if 'fits' in q:
            print(q['model'], q['p'], q['pop'], int(q['n_total']),
                  {k: (v['xmin'], v['best_aic'], round(v['PL']['theta'][0], 2)) for k, v in q['fits'].items()}, flush=True)
    tasks = [(q['model'], q['p'], q['pop'], q['xmin'], min(a.chunk, a.B - ch), ch)
             for q in points if 'xmin' in q for ch in range(0, a.B, a.chunk)]
    boots = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for model, p, pop, rows in ex.map(boot, tasks):
            boots.setdefault(f'{model}_{p}_{pop}', []).extend(rows)
    json.dump(boots, open(f'{OUT}/boot.json', 'w'), default=float)
    print(f'bootstrap done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
