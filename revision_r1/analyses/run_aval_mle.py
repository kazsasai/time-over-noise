"""
A1/A2: discrete MLE of avalanche-size distributions for SNM and ARM (rev1 core data), identical procedure for both.

Point estimates (pooled over the 1000 trials of each condition):
  xmin = 2 (full range), xmin = Clauset KS-optimal (PL), xmin = knee sb of the broken power law
  models PL, TPL, LN, SE, EXP; Vuong ratios (i.i.d. assumption, reported but not used for decisions); KS distances;
  broken power law (BPL, global exponential cutoff) vs single TPL on the full range (A2 / R3-4).
Trial-level bootstrap (B replicates; trials resampled with replacement, xmin and the sb grid held fixed):
  all fits repeated; we keep parameters and log-likelihoods, from which CIs and the fraction of replicates in
  which one model beats another are computed.

Usage (from revision_r1/analyses):
  python3 run_aval_mle.py [--key th1_dt1|all] [--B 200] [--workers 12] [--tag primary]
"""
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import aval_mle as am

warnings.filterwarnings('ignore')
HIST = 'runs/aval'
PS = [f'{p:.2f}' for p in np.arange(10) / 10 + 0.05]
MODELS = ('PL', 'TPL', 'LN', 'SE', 'EXP')
MIN_TAIL = 500


def load(model, p, key):
    with np.load(f'{HIST}/hist_{model}_{p}.npz') as z:
        return am.Hist(z, key)


def fits_at(s, c, xmin):
    st, ct = am.tail_of(s, c, xmin)
    if ct.sum() < 50:
        return None, None
    return am.fit_all(st, ct, xmin, MODELS)


def point(args):
    model, p, key = args
    h = load(model, p, key)
    s, c = h.pooled()
    c = c.astype(float)
    res = dict(model=model, p=p, key=key, n_trials=h.n_trials, n_total=float(c.sum()),
               s_max=int(s.max()) if s.size else 0)
    if c.sum() < 50:
        return res
    xmins = {'full': int(s.min())}
    cl = am.clauset_xmin(s, c, MIN_TAIL, extend=os.environ.get('AVAL_XMIN_EXTEND') == '1')
    if cl is not None:
        xmins['clauset'] = cl[0]
        res['clauset_D_PL'] = cl[1]
    # broken power law on the full range
    f_full, out_full = fits_at(s, c, xmins['full'])
    grid = am.sb_grid_for(s, c, min_above=MIN_TAIL)
    if grid.size >= 3:
        b = am.BPL(s[s >= xmins['full']], c[s >= xmins['full']], xmins['full'])
        r = b.fit(out_full['TPL']['theta'], grid)
        res['bpl'] = dict(sb=r['sb'], a1=r['a1'], a2=r['a2'], loglam=r['loglam'], ll=r['ll'],
                          dll_vs_tpl=r['ll'] - out_full['TPL']['ll'],
                          dbic_tpl_minus_bpl=2 * (r['ll'] - out_full['TPL']['ll']) - 2 * np.log(c.sum()),
                          profile=r['profile'].tolist(), sb_grid=grid.tolist())
        xmins['knee'] = r['sb']
    res['xmin'] = xmins
    res['fits'] = {}
    for name, x in xmins.items():
        f, out = (f_full, out_full) if name == 'full' else fits_at(s, c, x)
        if out is None:
            continue
        n = f.n
        d = dict(xmin=x, n_tail=float(n))
        for m, r in out.items():
            d[m] = dict(theta=r['theta'].tolist(), ll=r['ll'], **am.ic(r, n), ks=am.ks_distance(f, r),
                        at_bound=bool(np.any([np.isclose(v, lo) or np.isclose(v, hi) for v, (lo, hi)
                                              in zip(r['theta'], am.MODELS[m]['bounds'])])))
        d['best_aic'] = min(out, key=lambda m: d[m]['aic'])
        d['best_bic'] = min(out, key=lambda m: d[m]['bic'])
        d['vuong'] = {f'TPL_vs_{o}': am.vuong(f, out['TPL'], out[o])[1:] for o in ('LN', 'SE', 'PL', 'EXP')}
        res['fits'][name] = d
    return res


def boot(args):
    model, p, key, xmins, grid, B, chunk = args
    h = load(model, p, key)
    rng = np.random.default_rng([2024, ['SNM', 'ARM'].index(model), PS.index(p), chunk])
    rows = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, h.n_trials, h.n_trials), minlength=h.n_trials).astype(float)
        s, c = h.pooled(w)
        row = {}
        for name, x in xmins.items():
            f, out = fits_at(s, c, x)
            if out is None:
                continue
            row[name] = {m: (out[m]['theta'].tolist(), out[m]['ll']) for m in MODELS}
            row[name]['n_tail'] = float(f.n)
            if name == 'full' and grid:
                b = am.BPL(*am.tail_of(s, c, x), x)
                r = b.fit(out['TPL']['theta'], np.array(grid))
                row['bpl'] = dict(sb=r['sb'], a1=r['a1'], a2=r['a2'], loglam=r['loglam'],
                                  dll_vs_tpl=r['ll'] - out['TPL']['ll'], n=float(f.n))
        rows.append(row)
    return model, p, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='th1_dt1|all')
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=20)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--tag', default='primary')
    ap.add_argument('--models', default='SNM,ARM')
    a = ap.parse_args()
    out_dir = f'{HIST}/mle_{a.tag}'
    os.makedirs(out_dir, exist_ok=True)
    conds = [(m, p, a.key) for m in a.models.split(',') for p in PS]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        points = list(ex.map(point, conds))
    json.dump(points, open(f'{out_dir}/point.json', 'w'), indent=1, default=float)
    print(f'point estimates done ({time.time() - t0:.0f}s)', flush=True)
    if a.B <= 0:
        return
    tasks = []
    for r in points:
        if 'xmin' not in r:
            continue
        grid = r.get('bpl', {}).get('sb_grid', [])
        for ch in range(0, a.B, a.chunk):
            tasks.append((r['model'], r['p'], a.key, r['xmin'], grid, min(a.chunk, a.B - ch), ch))
    boots = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for model, p, rows in ex.map(boot, tasks):
            boots.setdefault(f'{model}_{p}', []).extend(rows)
    json.dump(boots, open(f'{out_dir}/boot.json', 'w'), default=float)
    print(f'bootstrap done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
