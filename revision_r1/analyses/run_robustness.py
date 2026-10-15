"""
A10: parametrisation-invariant robustness (R1-6, R3-7) on the rev1 core data (N=20, k=3).

1. Common observable: unconstrained random reselections (uniform draws over all k colours; a draw may return the
   current colour with probability 1/k, the same for both models) per sweep (= per N activations) and per round.
     SNM: eps * N per round (= per sweep)                     -- expectation
     ARM: mean(grid_lock) * (1-p) per round, mean(grid_lock) per sweep, where grid_lock (history_gridlock) counts
          all gridlocked nodes before the activation mask; only active ones redraw (probability 1-p, independent)
   averaged over the search phase (steps before the first solution; the whole record if unsolved) and over the
   stationary phase (from the first solution on).  Success within 1,000 rounds is plotted against it; the set of
   observable values with success >= 0.9 is reported in decades (no interval is forced).
2. Relative perturbations around the representative values eps=0.05 and p=0.45: theta (1 +- delta),
   delta in {0.1, 0.25, 0.5}, nearest 0.01 grid point; for the ARM also (1-p)(1 +- delta).
3. The paper's ratios recomputed on rev1 data with the uniform-prior assumption explicit: width of the >= 0.9 window
   (parameter units) and the mean success over a uniform grid, at 1,000 rounds (fine grid) and, as a sensitivity, at
   1,000 rounds and 200 sweeps on the long 0.1 grid.
Output: runs/robust/robustness.json.  Run from revision_r1/analyses:  python3 run_robustness.py
"""
import json, os
import numpy as np
import pandas as pd
from run_first_passage import wilson

OUT = 'runs/robust'
N = 20
DELTAS = (0.1, 0.25, 0.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_pickle('../data/result_for_phase_space_20_3.pkl')
    rows = []
    for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
        sub = df[df['label'] == lab]
        for p, g in sub.groupby(sub[col].round(2)):
            rate = (1 - p) if model == 'ARM' else 1.0
            succ = g['solved'].values.astype(bool)
            gl_search, gl_stat, n_search, n_stat = 0.0, 0.0, 0, 0
            for gl, cr in zip(g['history_gridlock'], g['history_conflict_ratio']):
                gl = np.asarray(gl, dtype=np.float64)
                s = np.asarray(cr) < 1e-12
                f = int(np.argmax(s)) if s.any() else gl.size
                gl_search += gl[:f].sum(); n_search += f
                gl_stat += gl[f:].sum(); n_stat += gl.size - f
            mgl_search = gl_search / n_search if n_search else np.nan
            mgl_stat = gl_stat / n_stat if n_stat else np.nan
            if model == 'SNM':
                rs_sweep = rs_round = p * N
                rs_sweep_stat = p * N
            else:
                rs_sweep, rs_round = mgl_search, mgl_search * rate
                rs_sweep_stat = mgl_stat
            k = int(succ.sum())
            rows.append(dict(model=model, p=float(p), succ=[k / succ.size, *wilson(k, succ.size)],
                             gridlock_mean_search=mgl_search, gridlock_mean_stationary=mgl_stat,
                             rand_per_sweep_search=rs_sweep, rand_per_round_search=rs_round,
                             rand_per_sweep_stationary=rs_sweep_stat))
    del df
    out = dict(rows=rows)
    # 1. >= 0.9 set on the common observable
    for model in ('SNM', 'ARM'):
        R = [r for r in rows if r['model'] == model and r['succ'][0] >= 0.9 and r['rand_per_sweep_search'] > 0]
        x = np.array([r['rand_per_sweep_search'] for r in R])
        out[f'{model}_ge90_obs'] = dict(n_params=len(R), min=float(x.min()), max=float(x.max()),
                                        decades=float(np.log10(x.max() / x.min())),
                                        params=[r['p'] for r in R])
    # 2. relative perturbations
    by = {(r['model'], round(r['p'], 2)): r for r in rows}
    pert = {}
    for model, th0 in (('SNM', 0.05), ('ARM', 0.45)):
        for d in DELTAS:
            for sgn in (-1, 1):
                th = round(th0 * (1 + sgn * d), 2)
                pert[f'{model}_theta{"+" if sgn > 0 else "-"}{d}'] = dict(param=th, succ=by[(model, th)]['succ'])
                if model == 'ARM':
                    th2 = round(1 - (1 - th0) * (1 + sgn * d), 2)
                    pert[f'ARM_(1-p){"+" if sgn > 0 else "-"}{d}'] = dict(param=th2, succ=by[(model, th2)]['succ'])
    out['perturbation'] = pert
    # 3. ratios with the uniform prior written out
    def window(model, key='succ'):
        R = sorted([r for r in rows if r['model'] == model], key=lambda r: r['p'])
        ps = np.array([r['p'] for r in R]); s = np.array([r[key][0] for r in R])
        i = int(np.argmax(s)); lo = hi = i
        while lo > 0 and s[lo - 1] >= 0.9: lo -= 1
        while hi < s.size - 1 and s[hi + 1] >= 0.9: hi += 1
        return float(ps[lo]), float(ps[hi]), float(ps[hi] - ps[lo]), float(s.mean())
    w_s, w_a = window('SNM'), window('ARM')
    out['ratios_fine_1000rounds'] = dict(SNM_window=w_s[:2], ARM_window=w_a[:2], width_ratio=w_a[2] / w_s[2],
                                         SNM_mean_rate_uniform=w_s[3], ARM_mean_rate_uniform=w_a[3],
                                         mean_rate_ratio=w_a[3] / w_s[3],
                                         note='uniform prior over the 0.00-0.99 grid of each parameter')
    fp = json.load(open('runs/fpt/first_passage.json'))
    long = fp['rows']['long']
    for key, label in (('succ_r1000', 'long_1000rounds'), ('succ_s200', 'long_200sweeps')):
        m = {model: np.array([r[key][0] for r in long if r['model'] == model and key in r]) for model in ('SNM', 'ARM')}
        out[f'ratios_{label}'] = dict(SNM_mean_rate_uniform=float(m['SNM'].mean()), ARM_mean_rate_uniform=float(m['ARM'].mean()),
                                      mean_rate_ratio=float(m['ARM'].mean() / m['SNM'].mean()),
                                      note='uniform prior over the 0.05-0.95 grid (step 0.1)')
    json.dump(out, open(f'{OUT}/robustness.json', 'w'), indent=1, default=float)
    for k, v in out.items():
        if k != 'rows':
            print(k, v)


if __name__ == '__main__':
    main()
