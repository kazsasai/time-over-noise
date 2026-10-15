"""Trial-level bootstrap for an existing MLE point run (point.json is read, never rewritten).

Same resampling as run_aval_mle.boot(): trials drawn with replacement, xmin and the BPL sb grid held fixed.

Usage (from revision_r1/analyses):
  python3 run_aval_mle_boot.py --tag sweep --key 'sweep|all' [--B 200] [--workers 10]
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import run_aval_mle as r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='sweep')
    ap.add_argument('--key', default='sweep|all')
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--chunk', type=int, default=20)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--only', default='')          # e.g. 'ARM_0.05' for a timing test
    a = ap.parse_args()
    out_dir = f'{r.HIST}/mle_{a.tag}'
    points = json.load(open(f'{out_dir}/point.json'))
    tasks = []
    for p in points:
        if 'xmin' not in p or p['key'] != a.key:
            continue
        if a.only and f"{p['model']}_{p['p']}" != a.only:
            continue
        grid = p.get('bpl', {}).get('sb_grid', [])
        for ch in range(0, a.B, a.chunk):
            tasks.append((p['model'], p['p'], a.key, p['xmin'], grid, min(a.chunk, a.B - ch), ch))
    t0 = time.time()
    boots = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for model, p, rows in ex.map(r.boot, tasks):
            boots.setdefault(f'{model}_{p}', []).extend(rows)
    name = 'boot.json' if not a.only else f'boot_test_{a.only}.json'
    json.dump(boots, open(f'{out_dir}/{name}', 'w'), default=float)
    print(f'bootstrap done: {len(tasks)} tasks, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
