"""Sensitivity of the two-population statistics to the rule that defines s* (sweep clock; R2-4).

s* is the smallest size above which avalanches within the solved state make up less than a fraction f of all
avalanches of that size or larger (primary f = 1%).  For f = 0.5%, 1% and 2% we recompute s*, the share of large
avalanches followed by a solution and the mean excess, on the sweep-clock histograms.
Output: runs/aval/planC_sweep/sstar_rule_sensitivity.json  {p: [[s*, resolve share, mean excess] for f in 0.5, 1, 2 %]}
Run from revision_r1/analyses.
"""
import json
import numpy as np
import run_planC as pc


def s_star(H, f):
    s_all = np.union1d(H['solved'].sizes, H['large'].sizes)
    sf, cf = H['solved'].pooled()
    sl, cl = H['large'].pooled()
    for x in s_all:
        nf, nl = pc.tail_n(sf, cf, x), pc.tail_n(sl, cl, x)
        if nl > 0 and nf < f * (nf + nl):
            return int(x)
    return None


def main():
    out = {}
    for p in pc.PS if hasattr(pc, 'PS') else [f'{q:.2f}' for q in np.arange(10) / 10 + 0.05]:
        H = pc.load('ARM', p, 'sweep')
        rows = []
        for f in (0.005, 0.01, 0.02):
            ss = s_star(H, f)
            e = pc.estimands(H, None, ss, None)
            rows.append([ss, round(float(e.get('resolve_share_large', np.nan)), 3), round(float(e.get('mean_excess', np.nan)), 1)])
        out[p] = rows
        print(p, rows, flush=True)
    json.dump(out, open('runs/aval/planC_sweep/sstar_rule_sensitivity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
