"""
A5 (first passage; R1-2, R2-9, R1-7, R3-7, R3-8) and A6 (retention; R2-8, R3-7, R3-1) on the rev1 core data, N=20, k=3.

Units: rounds = global steps; sweeps = rounds x (1-p) for the ARM (expected N activations per sweep) and = rounds for
the SNM, the same expectation-based clock as the primary avalanche definition.
First-passage time T = solve_time + 1 rounds (solve_time = 0-based index of the first step with C=0); trials not
solved within the record are right-censored at the horizon (all censoring is at the horizon, so Kaplan-Meier
reduces to the empirical survival function).

Data
  fine : ../data/result_for_phase_space_20_3.pkl (eps, p = 0.00..0.99, 1000 trials, 1,000 steps, histories)
  long : ../data/result_for_criticality_20_3.pkl (eps, p = 0.05..0.95, 1000 trials, 50,000 steps, histories)
Output: runs/fpt/*.json and runs/fpt/figs/*.png.  Run from revision_r1/analyses:  python3 run_first_passage.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from avalanche_histograms import bin_rate

OUT = 'runs/fpt'
B = 1000
H_ROUNDS = [50, 100, 200, 500, 1000]
H_SWEEPS = [25, 50, 100, 200]
H_LONG = [1000, 5000, 50000]
HS_LONG = [50, 100, 200, 500, 2500]
TAUS = [1, 10, 100, 1000]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def runs_of(x):
    """(value, length) of maximal runs in a boolean array."""
    if x.size == 0:
        return np.zeros(0, bool), np.zeros(0, int)
    edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    lens = np.diff(np.concatenate([starts, [x.size]]))
    return x[starts], lens


def groups(df):
    for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
        sub = df[df['label'] == lab]
        for p, g in sub.groupby(sub[col].round(2)):
            yield model, float(p), (1.0 - p) if model == 'ARM' else 1.0, g


def first_passage(g, rate, step_max, rng):
    T = np.where(g['solved'].values == 1, g['solve_time'].values + 1, np.inf).astype(float)
    n = T.size
    Ts = T * rate
    out = dict(n=n, solved=float(np.isfinite(T).mean()))
    for H in (H_ROUNDS if step_max == 1000 else H_LONG):
        k = int((T <= H).sum())
        out[f'succ_r{H}'] = [k / n, *wilson(k, n)]
    for Hs in (H_SWEEPS if step_max == 1000 else HS_LONG):
        if Hs / rate <= step_max:
            k = int((Ts <= Hs).sum())
            out[f'succ_s{Hs}'] = [k / n, *wilson(k, n)]
    fin = np.isfinite(T)
    out['cond_mean_r'] = float(T[fin].mean()) if fin.any() else np.nan
    out['cond_median_r'] = float(np.median(T[fin])) if fin.any() else np.nan
    out['cond_mean_s'] = out['cond_mean_r'] * rate
    out['cond_median_s'] = out['cond_median_r'] * rate
    # unconditional median (KM) and restricted mean (RMST) up to the horizon, with trial bootstrap
    def stats_of(Tb):
        med = float(np.median(Tb)) if np.isfinite(Tb).mean() >= 0.5 else np.nan
        return med, float(np.minimum(Tb, step_max).mean())
    med, rmst = stats_of(T)
    bs = np.array([stats_of(T[rng.integers(0, n, n)]) for _ in range(B)])
    out['median_r'] = [med, *np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist()] if np.isfinite(med) else [np.nan] * 3
    out['rmst_r'] = [rmst, *np.percentile(bs[:, 1], [2.5, 97.5]).tolist()]
    out['median_s'] = [v * rate for v in out['median_r']]
    out['rmst_s_at_horizon'] = [v * rate for v in out['rmst_r']]      # restricted at step_max rounds
    if fin.any():
        bm = np.array([T[fin][rng.integers(0, fin.sum(), fin.sum())].mean() for _ in range(B)])
        out['cond_mean_r_ci'] = np.percentile(bm, [2.5, 97.5]).tolist()
    return out, T


def retention(g, rate):
    """After the first solution: time fraction solved (raw steps); residence runs on the sweep clock."""
    frac, post_steps, solved_steps = [], 0, 0
    never_left, lv_post, lv_solved, lv_runs = 0, 0, 0, []     # 'leaving' trials: left the solved state at least once
    sol_runs, uns_runs, first_runs, cross, sweeps_total = [], [], [], 0, 0
    sol_runs_r, cross_r = [], 0
    for cr in g['history_conflict_ratio']:
        s = np.asarray(cr, dtype=np.float64) < 1e-12
        if not s.any():
            continue
        f = int(np.argmax(s))
        post = s[f:]
        frac.append(post.mean())
        post_steps += post.size
        solved_steps += int(post.sum())
        v, L = runs_of(post)
        sol_runs_r += L[v].tolist()
        if post.all():
            never_left += 1
        else:
            lv_post += post.size
            lv_solved += int(post.sum())
            lv_runs += (L[v][:-1] if v[-1] else L[v]).tolist()     # completed solved runs (rounds)
        cross_r += int((~v[:-1] & v[1:]).sum()) if v.size > 1 else 0
        # sweep clock: bins of 1/(1-p) steps, state at the end of each bin
        _, sb = bin_rate(np.zeros(s.size, np.int64), s, rate)
        if not sb.any():
            continue
        fb = int(np.argmax(sb))
        pb = sb[fb:]
        sweeps_total += pb.size
        v, L = runs_of(pb)
        sol_runs += L[v][:-1].tolist() if v[-1] else L[v].tolist()       # drop the run censored by the record end
        uns_runs += L[~v][:-1].tolist() if not v[-1] else L[~v].tolist()
        first_runs.append((L[0], pb.size))                                 # first solved run, sweeps observed
        cross += int((~v[:-1] & v[1:]).sum()) if v.size > 1 else 0
    frac = np.array(frac)
    sol_runs, uns_runs = np.array(sol_runs), np.array(uns_runs)
    fr = np.array([r[0] for r in first_runs]); avail = np.array([r[1] for r in first_runs])
    out = dict(n_trials_solved=int(frac.size),
               share_never_leave=never_left / frac.size if frac.size else np.nan,
               leaving_time_frac_solved=lv_solved / lv_post if lv_post else np.nan,
               leaving_mean_completed_solved_run_rounds=float(np.mean(lv_runs)) if lv_runs else np.nan,
               leaving_mean_completed_solved_run_sweeps=float(np.mean(lv_runs)) * rate if lv_runs else np.nan,
               leaving_median_completed_solved_run_sweeps=float(np.median(lv_runs)) * rate if lv_runs else np.nan,
               time_frac_solved_pooled=solved_steps / post_steps if post_steps else np.nan,
               time_frac_solved_trial_mean=float(frac.mean()) if frac.size else np.nan,
               mean_solved_run_rounds=float(np.mean(sol_runs_r)) if sol_runs_r else np.nan,
               crossings_per_1000_rounds=1000 * cross_r / post_steps if post_steps else np.nan,
               mean_solved_run_sweeps=float(sol_runs.mean()) if sol_runs.size else np.nan,   # completed runs only
               median_solved_run_sweeps=float(np.median(sol_runs)) if sol_runs.size else np.nan,
               mean_unsolved_run_sweeps=float(uns_runs.mean()) if uns_runs.size else np.nan,
               median_unsolved_run_sweeps=float(np.median(uns_runs)) if uns_runs.size else np.nan,
               frac_unsolved_runs_one_sweep=float((uns_runs == 1).mean()) if uns_runs.size else np.nan,
               crossings_per_1000_sweeps=1000 * cross / sweeps_total if sweeps_total else np.nan)
    for tau in TAUS:
        m = avail >= tau                       # only trials whose record after the first solution is long enough
        out[f'P_first_solved_run_ge_{tau}sw'] = float((fr[m] >= tau).mean()) if m.any() else np.nan
        out[f'n_first_solved_run_{tau}sw'] = int(m.sum())
    if frac.size:
        rng = np.random.default_rng(7)
        bs = [frac[rng.integers(0, frac.size, frac.size)].mean() for _ in range(B)]
        out['time_frac_solved_trial_mean_ci'] = np.percentile(bs, [2.5, 97.5]).tolist()
    return out, sol_runs, uns_runs


def horizon_scan(rows, key):
    """argmax and the contiguous >= 0.9 window around it, with binomial (trial) bootstrap of every rate."""
    ps = np.array([r['p'] for r in rows])
    ok = np.array([key in r for r in rows])
    ps = ps[ok]
    rate = np.array([r[key][0] for r, o in zip(rows, ok) if o])
    n = np.array([r['n'] for r, o in zip(rows, ok) if o])
    def scan(rt):
        i = int(np.argmax(rt))
        lo = hi = i
        if rt[i] >= 0.9:
            while lo > 0 and rt[lo - 1] >= 0.9:
                lo -= 1
            while hi < rt.size - 1 and rt[hi + 1] >= 0.9:
                hi += 1
            return ps[i], rt[i], ps[lo], ps[hi], ps[hi] - ps[lo]
        return ps[i], rt[i], np.nan, np.nan, 0.0
    point = scan(rate)
    rng = np.random.default_rng(11)
    bs = np.array([scan(rng.binomial(n, rate) / n) for _ in range(B)])
    return dict(p_range=[float(ps.min()), float(ps.max())], argmax=float(point[0]), max=float(point[1]),
                win_lo=float(point[2]), win_hi=float(point[3]), width=float(point[4]),
                argmax_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                width_ci=np.percentile(bs[:, 4], [2.5, 97.5]).tolist())


def main():
    os.makedirs(f'{OUT}/figs', exist_ok=True)
    res = {}
    surv = {}
    for name, path, step_max in (('fine', '../data/result_for_phase_space_20_3.pkl', 1000),
                                 ('long', '../data/result_for_criticality_20_3.pkl', 50000)):
        df = pd.read_pickle(path)
        rng = np.random.default_rng(2024)
        rows = []
        for model, p, rate, g in groups(df):
            fp, T = first_passage(g, rate, step_max, rng)
            ret, sol_runs, uns_runs = retention(g, rate)
            rows.append(dict(model=model, p=p, rate=rate, **fp, **ret))
            if (model, round(p, 2)) in (('SNM', 0.05), ('SNM', 0.06), ('ARM', 0.45), ('ARM', 0.47)):
                surv[f'{name}_{model}_{p:.2f}'] = dict(T=np.sort(T[np.isfinite(T)]).tolist(), n=int(T.size), rate=rate,
                                                       sol_runs=np.sort(sol_runs).tolist(),
                                                       uns_runs=np.sort(uns_runs).tolist())
            print(name, model, f'{p:.2f}', f"solved {fp['solved']:.3f} median_r {fp['median_r'][0]} "
                  f"time_frac_solved {ret['time_frac_solved_pooled']:.3f} "
                  f"solved_run_sw {ret['mean_solved_run_sweeps']:.1f}", flush=True)
        res[name] = rows
        del df
    # horizon scans (R1-2): rounds and sweeps
    scans = {}
    for model in ('SNM', 'ARM'):
        fine = [r for r in res['fine'] if r['model'] == model]
        long = [r for r in res['long'] if r['model'] == model]
        for H in H_ROUNDS:
            scans[f'{model}_fine_r{H}'] = horizon_scan(fine, f'succ_r{H}')
        for Hs in H_SWEEPS:
            scans[f'{model}_fine_s{Hs}'] = horizon_scan(fine, f'succ_s{Hs}')
        for H in H_LONG:
            scans[f'{model}_long_r{H}'] = horizon_scan(long, f'succ_r{H}')
        for Hs in HS_LONG:
            scans[f'{model}_long_s{Hs}'] = horizon_scan(long, f'succ_s{Hs}')
    json.dump(dict(rows=res, scans=scans), open(f'{OUT}/first_passage.json', 'w'), indent=1, default=float)
    json.dump(surv, open(f'{OUT}/survival_at_optima.json', 'w'))
    for k, v in scans.items():
        print(k, {a: (np.round(b, 3).tolist() if isinstance(b, list) else round(b, 3)) for a, b in v.items()})


if __name__ == '__main__':
    main()
