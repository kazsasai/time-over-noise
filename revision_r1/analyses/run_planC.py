"""
Two-population estimands with trial-level bootstrap CIs (rev1 core data, N=20, k=3; primary avalanche definition).

(A) Composition: fractions of flicker / resolve / open avalanches; among large avalanches (s >= s*), the share that
    is followed by a solution (resolve).  s*(p) = smallest size at which flicker avalanches are < 1 % of all
    avalanches of that size or larger; fixed from the full data and held fixed in the bootstrap.
(B) Scale of the large-avalanche population (non-flicker = resolve + open):
    - TPL cutoff 1/lam, MLE above a fixed xmin (KS-optimal TPL xmin from the full data);
    - model-free: mean excess E[s - s* | s >= s*] of non-flicker avalanches.
(F) Flicker scale: s_c of a TPL fitted above xmin = 10 (for the collapse statement; optional in the paper).

Trials are resampled with replacement (B replicates); s*, xmin are held fixed.
Usage (from revision_r1/analyses):  python3 run_planC.py [--key th1_dt1] [--B 500] [--tag primary]
"""
import argparse, json, os, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import sparse
import aval_mle as am

warnings.filterwarnings('ignore')
HIST = 'runs/aval'
PS = [f'{p:.2f}' for p in np.arange(10) / 10 + 0.05]
CLASSES = ('solved', 'resolve', 'open')
FLICKER_XMIN = 10


class Multi(am.Hist):
    """Hist built from the union of several keys (counts of identical (trial, size) pairs are summed)."""

    def __init__(self, npz, keys):
        self.n_trials = int(npz['n_trials'])
        trip = [npz[k] for k in keys if k in npz.files]
        if not trip:
            self.sizes = np.zeros(0, dtype=np.int64)
            self.M = sparse.csr_matrix((self.n_trials, 0))
            return
        t, s, c = np.concatenate(trip, axis=1)
        self.sizes, idx = np.unique(s, return_inverse=True)
        self.M = sparse.csr_matrix((c, (t, idx)), shape=(self.n_trials, self.sizes.size))


def load(model, p, key, hist_dir=HIST):
    with np.load(f'{hist_dir}/hist_{model}_{p}.npz') as z:
        H = {k: am.Hist(z, f'{key}|{k}') for k in CLASSES}
        H['large'] = Multi(z, [f'{key}|resolve', f'{key}|open'])
    return H


def tail_n(s, c, x):
    return c[s >= x].sum()


def s_star(H):
    s_all = np.union1d(H['solved'].sizes, H['large'].sizes)
    sf, cf = H['solved'].pooled()
    sl, cl = H['large'].pooled()
    for x in s_all:
        nf, nl = tail_n(sf, cf, x), tail_n(sl, cl, x)
        if nl > 0 and nf < 0.01 * (nf + nl):
            return int(x)
    return None


def ks_tpl_xmin(s, c, min_tail=200, upto=5000):
    best = None
    for x in am.xmin_grid(s, c, min_tail=min_tail, upto=min(upto, s.max())):
        st, ct = am.tail_of(s, c, x)
        f = am.Fitter(st, ct, x)
        r = f.fit('TPL')
        D = am.ks_distance(f, r)
        if not np.isfinite(D):
            continue
        if best is None or D < best[1]:
            best = (int(x), D)
    return best


def estimands(H, w, sstar, xmin_large):
    out = {}
    tot = {k: H[k].pooled(w)[1].sum() for k in CLASSES}
    n = sum(tot.values())
    for k in CLASSES:
        out[f'frac_{k}'] = tot[k] / n if n else np.nan
    if sstar is not None:
        sr, cr = H['resolve'].pooled(w)
        so, co = H['open'].pooled(w)
        nr, no = tail_n(sr, cr, sstar), tail_n(so, co, sstar)
        out['n_large'] = nr + no
        out['resolve_share_large'] = nr / (nr + no) if nr + no else np.nan
        sl, cl = H['large'].pooled(w)
        m = sl >= sstar
        out['mean_excess'] = (cl[m] @ (sl[m] - sstar)) / cl[m].sum() if cl[m].sum() else np.nan
    if xmin_large is not None:
        sl, cl = H['large'].pooled(w)
        st, ct = am.tail_of(sl, cl.astype(float), xmin_large)
        if ct.sum() >= 50:
            r = am.Fitter(st, ct, xmin_large).fit('TPL')
            out['large_tpl_alpha'], out['large_tpl_scale'] = r['theta'][0], float(np.exp(-r['theta'][1]))
            out['large_n_tail'] = float(ct.sum())
    sf, cf = H['solved'].pooled(w)
    st, ct = am.tail_of(sf, cf.astype(float), FLICKER_XMIN)
    if ct.sum() >= 500:
        r = am.Fitter(st, ct, FLICKER_XMIN).fit('TPL')
        out['flicker_tpl_alpha'], out['flicker_tpl_scale'] = r['theta'][0], float(np.exp(-r['theta'][1]))
    return out


def setup(args):
    model, p, key, hist_dir = args
    H = load(model, p, key, hist_dir)
    sstar = s_star(H) if model.startswith('ARM') else None
    sl, cl = H['large'].pooled()
    xl = ks_tpl_xmin(sl, cl.astype(float)) if cl.sum() >= 200 else None
    point = estimands(H, None, sstar, xl[0] if xl else None)
    return dict(model=model, p=p, key=key, n_trials=H['solved'].n_trials, s_star=sstar,
                xmin_large=xl[0] if xl else None, ks_large=xl[1] if xl else None, point=point)


def boot(args):
    model, p, key, sstar, xmin_large, B, chunk, hist_dir = args
    H = load(model, p, key, hist_dir)
    nt = H['solved'].n_trials
    mid = ['SNM', 'ARM'].index(model) if model in ('SNM', 'ARM') else sum(map(ord, model))
    rng = np.random.default_rng([2024, mid, PS.index(p), chunk, 7])
    rows = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, nt, nt), minlength=nt).astype(float)
        rows.append(estimands(H, w, sstar, xmin_large))
    return model, p, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='th1_dt1')
    ap.add_argument('--B', type=int, default=500)
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--tag', default='primary')
    ap.add_argument('--models', default='ARM,SNM')
    ap.add_argument('--ps', default=','.join(PS))
    ap.add_argument('--hist-dir', default=HIST)
    a = ap.parse_args()
    out_dir = f'{a.hist_dir}/planC_{a.tag}'
    os.makedirs(out_dir, exist_ok=True)
    conds = [(m, p, a.key, a.hist_dir) for m in a.models.split(',') for p in a.ps.split(',')
             if os.path.exists(f'{a.hist_dir}/hist_{m}_{p}.npz')]
    with ProcessPoolExecutor(a.workers) as ex:
        setups = list(ex.map(setup, conds))
    tasks = [(r['model'], r['p'], a.key, r['s_star'], r['xmin_large'], min(a.chunk, a.B - ch), ch, a.hist_dir)
             for r in setups for ch in range(0, a.B, a.chunk)]
    boots = {}
    if a.B > 0:
        with ProcessPoolExecutor(a.workers) as ex:
            for model, p, rows in ex.map(boot, tasks):
                boots.setdefault(f'{model}_{p}', []).extend(rows)
    summary = []
    for r in setups:
        bs = boots.get(f"{r['model']}_{r['p']}", [])
        row = dict(model=r['model'], p=r['p'], s_star=r['s_star'], xmin_large=r['xmin_large'], ks_large=r['ks_large'])
        for k, v in r['point'].items():
            vals = np.array([b.get(k, np.nan) for b in bs], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[k] = v
            if vals.size:
                row[f'{k}_lo'], row[f'{k}_hi'] = np.percentile(vals, [2.5, 97.5])
        summary.append(row)
    json.dump(dict(setups=setups, summary=summary, B=a.B), open(f'{out_dir}/planC.json', 'w'), indent=1, default=float)
    json.dump(boots, open(f'{out_dir}/planC_boot.json', 'w'), default=float)
    for row in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == '__main__':
    main()
