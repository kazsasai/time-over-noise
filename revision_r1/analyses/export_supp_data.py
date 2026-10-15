"""
Compact data exports for the supplementary figures (read-only use of the rev1 core data).
  runs/spec/spectrum_curves.npz        conflict-ratio spectra (ARM p = 0.05, 0.45, 0.85; SNM eps = 0.05)
  runs/agentspec/agent_spectra_curves.npz   per-agent spectra, Uragami & Gunji protocol (mean S_f)
  runs/fpt/solve_times_long.npz        first-solution step and solved flag per trial (50,000-round data)
Run from revision_r1/analyses.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
from run_spectra import L
from spectra_supplement import c_shuffle


def spectra(df):
    out = {}
    for lab, col, model, pars in (('Async Model', 'p_async', 'ARM', (0.05, 0.45, 0.85)), ('Noise Model', 'p_noise', 'SNM', (0.05,))):
        for p in pars:
            g = df[(df['label'] == lab) & ((df[col] - p).abs() < 1e-6)]
            C = np.stack([np.asarray(c, dtype=np.float64) for c in g['history_conflict_ratio']])
            T = C.shape[1]; key = f'{model}_{p:.2f}'
            out[f'{key}__full_f_cycles_per_step'] = np.arange(1, T // 2) / T
            la = np.log10(np.abs(np.fft.fft(C, axis=1))[:, 1:T // 2] + 1e-12)
            s0 = C < 1e-12
            nl = np.array([s.any() and s[int(np.argmax(s)):].all() for s in s0])
            out[f'{key}__full_geomean_amp_all'] = 10 ** la.mean(0)
            if nl.any():
                out[f'{key}__full_geomean_amp_never_leave'] = 10 ** la[nl].mean(0)
            out[f'{key}__full_geomean_amp_other'] = 10 ** la[~nl].mean(0)
            out[f'{key}__share_never_leave'] = np.array(nl.mean())
            del la
            W = C[:, -L:]; W = W[W.std(axis=1) > 0]
            rate = (1 - p) if model == 'ARM' else 1.0
            out[f'{key}__win_f_cycles_per_sweep'] = np.fft.rfftfreq(L)[1:] / rate
            P = np.abs(np.fft.rfft(W, axis=1)[:, 1:]) ** 2
            out[f'{key}__win_mean_power_real'] = P.mean(0)
            rng = np.random.default_rng(1)
            out[f'{key}__win_mean_power_renewal_surrogate'] = np.mean([np.abs(np.fft.rfft(c_shuffle(c, rng))[1:]) ** 2 for c in W], axis=0)
            var = W.var(axis=1); k = var <= np.quantile(var, 0.9)
            out[f'{key}__win_mean_power_without_top10pct_trials'] = P[k].mean(0)
            Pn = P / P.mean(axis=1, keepdims=True)
            q = np.percentile(Pn, [5, 50, 95], axis=0)
            out[f'{key}__win_normalised_power_trial_p05'] = q[0]
            out[f'{key}__win_normalised_power_trial_median'] = q[1]
            out[f'{key}__win_normalised_power_trial_p95'] = q[2]
            out[f'{key}__win_normalised_power_trial_mean'] = Pn.mean(0)
            out[f'{key}__win_n_nonconstant_trials'] = np.array(W.shape[0])
            print('spectra', key, flush=True)
    np.savez_compressed('runs/spec/spectrum_curves.npz', **out)


def agent_curves():
    from run_agent_spectra import worker, SEEDS
    conds = [('ARM', 0.05), ('ARM', 0.45), ('ARM', 0.85), ('SNM', 0.05), ('SNM', 0.15)]
    out = {}
    with mp.get_context('fork').Pool(12) as pool:
        for model, par in conds:
            outs = pool.map(worker, [(model, par, s) for s in SEEDS], chunksize=4)
            for key in ('a_step', 'b_step', 'a_sweep', 'b_sweep'):
                Ss = [o[key] for o in outs if key in o]
                if Ss:
                    out[f'{model}_{par:.2f}__{key}__mean_S'] = np.mean(Ss, axis=0)
            print('agent', model, par, flush=True)
    out['f_index'] = np.arange(1, 513)
    np.savez_compressed('runs/agentspec/agent_spectra_curves.npz', **out)


if __name__ == '__main__':
    agent_curves()
    df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
    st = {}
    for lab, col, model in (('Noise Model', 'p_noise', 'SNM'), ('Async Model', 'p_async', 'ARM')):
        sub = df[df['label'] == lab]
        for p, g in sub.groupby(sub[col].round(2)):
            st[f'{model}_{p:.2f}__solve_time_step'] = g['solve_time'].values.astype(np.int64)
            st[f'{model}_{p:.2f}__solved'] = g['solved'].values.astype(np.int8)
    np.savez_compressed('runs/fpt/solve_times_long.npz', **st)
    print('solve times saved', flush=True)
    spectra(df)
