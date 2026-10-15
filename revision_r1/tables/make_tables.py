"""Supplementary Tables S1-S11, generated from the analysis outputs (never typed by hand).

Reads only ../analyses/runs (read-only) and writes table_S*.tex, preview.tex and quoted_numbers.json in this directory.
Numbers quoted in the supplementary text are recomputed here and, when the manuscript source
../tex_rev1/supplementary.tex is present, asserted to appear there verbatim, so tables and text cannot drift.

Usage (from revision_r1/tables):  python3 make_tables.py
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / 'analyses' / 'runs'
PS = [f'{p:.2f}' for p in np.arange(10) / 10 + 0.05]
MODELS = ('PL', 'TPL', 'LN', 'SE', 'EXP')
NPAR = {'PL': 1, 'TPL': 2, 'LN': 2, 'SE': 2, 'EXP': 1}
DASH = '---'


def read(path):
    return json.loads((RUNS / path).read_text())


# ---------- formatting ----------
def num(x, d=2):
    if x is None or not np.isfinite(x):
        return DASH
    return f'{x:.{d}f}'


def big(x):
    """Integer with LaTeX thin thousands separators."""
    if x is None or not np.isfinite(x):
        return DASH
    return f'{int(round(x)):,}'.replace(',', '{,}')


def sig(x, n=3):
    """Size-like quantity: n significant figures, thousands separators."""
    if x is None or not np.isfinite(x):
        return DASH
    if x >= 10 ** n:
        return big(x)
    return f'{float(f"{x:.{n}g}"):g}'


def tfmt(x):
    """Times: one decimal below 100, integers above."""
    return DASH if not np.isfinite(x) else (f'{x:.1f}' if x < 100 else big(x))


def ci(x, lo, hi, f=num, **kw):
    if x is None or not np.isfinite(x):
        return DASH
    return f'{f(x, **kw)} [{f(lo, **kw)}, {f(hi, **kw)}]'


def pct(x, d=1):
    return DASH if x is None or not np.isfinite(x) else f'{100 * x:.{d}f}'


def rng(vals, d=2, sep='–'):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return f'{min(v):.{d}f}{sep}{max(v):.{d}f}'


def table(label, caption, colspec, header, rows, size='\\footnotesize', sep='4pt', notes=None):
    out = ['\\begin{table}[H]', '\\centering',
           f'\\caption{{{caption}}}', f'\\label{{{label}}}', size,
           f'\\setlength{{\\tabcolsep}}{{{sep}}}', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule']
    out += [h + ' \\\\' for h in header]
    out.append('\\midrule')
    for r in rows:
        out.append(r if isinstance(r, str) else ' & '.join(r) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    if notes:
        out.append('\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize ' + notes + '}')
    out.append('\\end{table}')
    return '\n'.join(out) + '\n'


# ---------- S1, S2: maximum likelihood, sweep clock ----------
def aic(m, ll):
    return 2 * NPAR[m] - 2 * ll


def boot_summary(boot_rows, region):
    """Percentile CIs of the PL exponent and the share of replicates in which PL has the lowest AIC."""
    alphas, pl_best = [], []
    for row in boot_rows:
        if region not in row:
            continue
        fits = row[region]
        a = {m: aic(m, fits[m][1]) for m in MODELS}
        alphas.append(fits['PL'][0][0])
        pl_best.append(min(a, key=a.get) == 'PL')
    alphas = np.asarray(alphas)
    return dict(n=len(alphas), alpha_lo=np.percentile(alphas, 2.5), alpha_hi=np.percentile(alphas, 97.5),
                p_pl_best=float(np.mean(pl_best)))


def tail_share_above_sstar(p, xmin, s_star):
    with np.load(RUNS / f'aval/hist_ARM_{p}.npz') as z:
        a = z['sweep|all']; b = z['sweep|solved']
    tot = a[2][a[1] >= xmin].sum()
    return a[2][a[1] >= s_star].sum() / tot, b[2][b[1] >= xmin].sum() / tot


def mle_tables(Q):
    point = read('aval/mle_sweep/point.json')
    boot = read('aval/mle_sweep/boot.json')
    plan = {r['p']: r for r in read('aval/planC_sweep/planC.json')['summary'] if r['model'] == 'ARM'}
    out = {}
    for model in ('ARM', 'SNM'):
        rows, alphas, ks, xmins, above, flick = [], [], [], [], [], []
        for r in sorted([r for r in point if r['model'] == model], key=lambda r: float(r['p'])):
            full, tail = r['fits']['full'], r['fits']['clauset']
            bf = full['best_aic']; bt = tail['best_aic']
            b = boot_summary(boot[f"{model}_{r['p']}"], 'clauset')
            assert b['n'] == 200, (model, r['p'], b['n'])
            a_pl = tail['PL']['theta'][0]
            d_pl_full = aic('PL', full['PL']['ll']) - aic(bf, full[bf]['ll'])
            d_tpl_tail = aic('TPL', tail['TPL']['ll']) - aic(bt, tail[bt]['ll'])
            # Tails with fewer than two avalanches per trial are flagged: their fits are not interpreted.
            edge = tail['n_tail'] < 2 * r['n_trials']
            row = [r['p'], big(r['n_total']), bf, big(d_pl_full), num(full[bf]['ks'], 3),
                   big(tail['xmin']) + ('$^{b}$' if edge else ''), big(tail['n_tail'])]
            if model == 'ARM':
                sh, fl = tail_share_above_sstar(r['p'], tail['xmin'], plan[r['p']]['s_star'])
                above.append(sh); flick.append(fl)
                row += [pct(sh), pct(fl, 0)]
            row += [bt + ('' if bt == 'PL' else '$^{a}$'), (num(d_tpl_tail, 1) if d_tpl_tail < 1000 else big(d_tpl_tail)), pct(b['p_pl_best'], 0),
                    ci(a_pl, b['alpha_lo'], b['alpha_hi']) + ('$^{c}$' if tail['PL']['at_bound'] else ''), num(tail['PL']['ks'], 3)]
            rows.append(row)
            if model == 'ARM':
                alphas.append(a_pl); ks.append(tail['PL']['ks']); xmins.append(tail['xmin'])
                Q.setdefault('S1_plbest', {})[r['p']] = b['p_pl_best']
        par = '$p$' if model == 'ARM' else r'$\varepsilon$'
        if model == 'ARM':
            colspec = 'c r c r c r r r r c r r c c'
            header = [' & & \\multicolumn{3}{c}{Full range} & \\multicolumn{9}{c}{KS-selected tail ($s \\ge x_{\\min}$)}',
                      '\\cmidrule(lr){3-5}\\cmidrule(lr){6-14}',
                      f'{par} & $n$ & best & $\\Delta$AIC$_{{\\rm PL}}$ & KS & $x_{{\\min}}$ & $n_{{\\rm tail}}$ & \\% $\\ge s^*$ & \\% solved & best & $\\Delta$AIC$_{{\\rm TPL}}$ & \\% PL best & $\\alpha_{{\\rm PL}}$ [95\\% CI] & KS$_{{\\rm PL}}$']
            Q['S1_alpha'] = rng(alphas); Q['S1_ks'] = rng(ks, 3)
            Q['S1_xmin'] = f'{min(xmins)}–{max(xmins)}'
            Q['S1_above'] = f'{100 * (1 - max(above)):.0f}–{100 * (1 - min(above)):.0f}%'
            Q['S1_flick_lo'] = f'{100 * flick[0]:.0f}%'; Q['S1_flick_hi'] = f'{100 * flick[-1]:.0f}%'
            notes = ('$n$, number of avalanches pooled over 1{,}000 trials of 50{,}000 rounds. The full range starts at the smallest observed size ($s = 2$ for every ARM condition). Best, the candidate with the lowest AIC '
                     '(PL, power law; TPL, truncated power law; LN, log-normal; SE, stretched exponential; EXP, exponential). '
                     '$\\Delta$AIC$_{\\rm PL}$ and $\\Delta$AIC$_{\\rm TPL}$, AIC of PL and TPL minus that of the best candidate. '
                     '$x_{\\min}$ minimises the KS distance of PL over 40 log-spaced candidates from the smallest size to 5{,}000, continued above 5{,}000 at about 11.5 candidates per decade up to the largest size that leaves 500 avalanches in the tail (the same rule for both models). \\% $\\ge s^*$, share of tail avalanches at or above $s^*$ (the size above which avalanches within the solved state make up less than 1\\%); '
                     '\\% solved, share of tail avalanches that stay within the solved state. '
                     '\\% PL best, share of 200 trial-bootstrap replicates (trials resampled with replacement, $x_{\\min}$ fixed) in which PL has the lowest AIC on the tail. '
                     'Intervals are bootstrap percentiles. $^{a}$At $p=0.45$, LN is best on the tail.')
        else:
            colspec = 'c r c r c r r c r r c c'
            header = [' & & \\multicolumn{3}{c}{Full range} & \\multicolumn{7}{c}{KS-selected tail ($s \\ge x_{\\min}$)}',
                      '\\cmidrule(lr){3-5}\\cmidrule(lr){6-12}',
                      f'{par} & $n$ & best & $\\Delta$AIC$_{{\\rm PL}}$ & KS & $x_{{\\min}}$ & $n_{{\\rm tail}}$ & best & $\\Delta$AIC$_{{\\rm TPL}}$ & \\% PL best & $\\alpha_{{\\rm PL}}$ [95\\% CI] & KS$_{{\\rm PL}}$']
            rows = [r[:7] + [r[7].replace('$^{a}$', '')] + r[8:] for r in rows]
            notes = ('Columns as in Table~\\ref{tab:supp_mle_arm}. $^{b}$Fewer than two avalanches per trial in the tail: the KS criterion selects the far end of the distribution, '
                     'and these tail fits are not interpreted. $^{c}$At the bound of the optimisation. For $\\varepsilon \\ge 0.45$ there are only 1.1--1.7 avalanches per trial, and their size grows '
                     'almost in proportion to the record length (Supplementary Fig.~S4\\,l); these rows are listed for completeness and are not compared with the ARM.')
        cap = (f'\\textbf{{Maximum-likelihood comparison of avalanche-size distributions, {model} (sweep clock).}} '
               'Identical procedure for both models: discrete maximum likelihood on the full size range and on the tail '
               'selected by the KS criterion, with five candidate distributions.')
        out[model] = table(f'tab:supp_mle_{model.lower()}', cap, colspec, header, rows,
                           size='\\scriptsize', sep='2pt', notes=notes)
    return out['ARM'], out['SNM']


# ---------- S10: maximum likelihood for each population separately (R1-3) ----------
def population_fits():
    """Per-population fits (analyses/run_aval_mle_pop.py) with bootstrap summaries; shared with check_manuscript.py."""
    point = read('aval/mle_pop/point.json'); boot = read('aval/mle_pop/boot.json')
    plan = {r['p']: r for r in read('aval/planC_sweep/planC.json')['summary'] if r['model'] == 'ARM'}
    out = []
    for r in point:
        if 'fits' not in r:
            continue
        bs = boot[f"{r['model']}_{r['p']}_{r['pop']}"]
        assert len(bs) == 200
        full, tail = r['fits']['full'], r['fits'].get('clauset')
        bf = full['best_aic']
        best_of = lambda fits: min(MODELS, key=lambda m: aic(m, fits[m][1]))
        d = dict(model=r['model'], p=r['p'], pop=r['pop'], n=r['n_total'], n_trials=r['n_trials'], best=bf,
                 same=float(np.mean([best_of(b['full']) == bf for b in bs])),
                 d_pl=aic('PL', full['PL']['ll']) - aic(bf, full[bf]['ll']), ks=full[bf]['ks'],
                 cut=float(np.exp(-full['TPL']['theta'][1])),
                 cut_ci=[float(x) for x in np.percentile([np.exp(-b['full']['TPL'][0][1]) for b in bs], [2.5, 97.5])])
        if tail is not None:
            al = [b['clauset']['PL'][0][0] for b in bs if 'clauset' in b]
            d.update(xmin=tail['xmin'], n_tail=tail['n_tail'], tbest=tail['best_aic'], alpha=tail['PL']['theta'][0],
                     alpha_ci=[float(x) for x in np.percentile(al, [2.5, 97.5])], ks_pl=tail['PL']['ks'],
                     pl_best=float(np.mean([best_of(b['clauset']) == 'PL' for b in bs if 'clauset' in b])),
                     at_bound=tail['PL']['at_bound'])
        with np.load(RUNS / f"aval/hist_{r['model']}_{r['p']}.npz") as z:
            keys = ['sweep|solved'] if r['pop'] == 'solved' else ['sweep|resolve', 'sweep|open']
            arr = np.concatenate([z[k] for k in keys if k in z.files], axis=1)
        if r['pop'] == 'solved':
            d['q999'] = float(np.percentile(np.repeat(arr[1], arr[2].astype(int)), 99.9))
        if r['model'] == 'ARM' and tail is not None:
            d['below_sstar'] = float(arr[2][(arr[1] >= tail['xmin']) & (arr[1] < plan[r['p']]['s_star'])].sum() / arr[2][arr[1] >= tail['xmin']].sum())
        out.append(d)
    return out


def mle_population_table(Q):
    rows = []
    fits = population_fits()
    for model in ('ARM', 'SNM'):
        sel = [d for d in fits if d['model'] == model and (model == 'ARM' or float(d['p']) <= 0.35)]
        rows.append(f'\\multicolumn{{14}}{{l}}{{\\textit{{{model}}}}} \\\\')
        for d in sorted(sel, key=lambda d: (float(d['p']), d['pop'] != 'solved')):
            edge = d.get('n_tail', 1e9) < 2 * d['n_trials']
            rows.append([d['p'], 'solved' if d['pop'] == 'solved' else 'unsolved', (num(d['n'] / 1000, 1) if d['n'] < 10000 else big(d['n'] / 1000)), d['best'], pct(d['same'], 0),
                         (num(d['d_pl'] / 1000, 1) if d['d_pl'] < 10000 else big(d['d_pl'] / 1000)), num(d['ks'], 3), ci(d['cut'], *d['cut_ci'], f=num, d=0),
                         big(d.get('xmin')) + ('$^{b}$' if edge else ''), d.get('tbest', DASH), pct(d.get('pl_best'), 0),
                         ci(d.get('alpha'), *(d.get('alpha_ci') or [None, None])) + ('$^{c}$' if d.get('at_bound') else ''),
                         num(d.get('ks_pl'), 3), pct(d.get('below_sstar'), 0) if 'below_sstar' in d else DASH])
    arm = [d for d in fits if d['model'] == 'ARM']
    sol = [d for d in arm if d['pop'] == 'solved']; uns = [d for d in arm if d['pop'] == 'unsolved']
    Q['P_solved_best'] = {b: sum(d['best'] == b for d in sol) for b in ('TPL', 'SE', 'LN')}
    Q['P_cut'] = f"{min(d['cut'] for d in sol):.0f}–{max(d['cut'] for d in sol):.0f}"
    Q['P_q999'] = f"{min(d['q999'] for d in sol):.0f}–{max(d['q999'] for d in sol):.0f}"
    assert all(d['best'] == 'LN' and d['same'] == 1.0 for d in uns)
    low = [d for d in uns if float(d['p']) <= 0.45]
    assert all(d['tbest'] == 'PL' for d in low) and all(d['tbest'] != 'PL' for d in uns if float(d['p']) > 0.45)
    Q['P_alpha_low'] = rng([d['alpha'] for d in low]); Q['P_xmin'] = f"{min(d['xmin'] for d in uns)}–{max(d['xmin'] for d in uns)}"
    Q['P_below'] = f"{100 * min(d['below_sstar'] for d in uns):.0f}–{100 * max(d['below_sstar'] for d in uns):.0f}%"
    notes = ('Each population fitted separately (sweep clock): solved, avalanches within the solved state; unsolved, avalanches that begin in an unsolved state (classes resolve and open). '
             'Columns as in Table~\\ref{tab:supp_mle_arm}; \\% same, share of 200 trial-bootstrap replicates in which the full-range best candidate stays best; '
             '$n$ (number of avalanches) and $\\Delta$AIC$_{\\rm PL}$ in thousands; cut-off, $1/\\lambda$ of the truncated power law fitted on the full range (colour changes), with its 95\\% bootstrap interval; \\% PL, share of replicates in which PL is best on the tail; $\\alpha_{\\rm PL}$ with its 95\\% bootstrap interval; \\% $< s^*$, share of tail avalanches below $s^*$. '
             '$^{b}$Fewer than two avalanches per trial in the tail: not interpreted. $^{c}$At the bound of the optimisation. '
             'SNM rows are listed for $\\varepsilon \\le 0.35$; within the solved state the SNM has 651 avalanches at $\\varepsilon = 0.15$ and fewer than 50 from $\\varepsilon = 0.25$; for $\\varepsilon \\ge 0.45$ the avalanche sizes are limited by the record length (Supplementary Fig.~S4\\,l).')
    header = [' & & & \\multicolumn{5}{c}{Full range} & \\multicolumn{6}{c}{KS-selected tail ($s \\ge x_{\\min}$)}',
              '\\cmidrule(lr){4-8}\\cmidrule(lr){9-14}',
              '$p$/$\\varepsilon$ & pop. & $n$ & best & \\% same & $\\Delta$AIC$_{\\rm PL}$ & KS & cut-off & $x_{\\min}$ & best & \\% PL & $\\alpha_{\\rm PL}$ & KS$_{\\rm PL}$ & \\% $< s^*$']
    cap = ('\\textbf{Maximum-likelihood comparison for each avalanche population separately (sweep clock).} '
           'The same five candidates, $x_{\\min}$ rule and trial bootstrap as in Tables~\\ref{tab:supp_mle_arm} and~\\ref{tab:supp_mle_snm}, applied to the avalanches within the solved state and to those that begin in an unsolved state.')
    return table('tab:supp_mle_pop', cap, 'c c r c r r c c r c r c c r', header, rows, size='\\scriptsize', sep='1.3pt', notes=notes)


# ---------- S11: ARM results by graph type (breakable / locked) ----------
def graph_type_table(Q):
    d = read('graph_types/graph_types.json')
    rows = []
    for p in sorted(d['holding'], key=float):
        h, a, c = d['holding'][p], d['avalanches'][p], d['injection'][p]
        for t in ('breakable', 'locked'):
            lv = h[t]['leaving_time_frac_solved']
            rows.append([p if t == 'breakable' else '', t, pct(h[t]['time_frac_solved_pooled']), pct(h[t]['share_never_leave']),
                         num(h[t]['crossings_per_1000_sweeps'], 1), pct(a[t]['solved_share']), num(a[t]['large_per_trial']),
                         num(a[t]['resolve_share_large']), big(a[t]['mean_excess']),
                         ci(*c[t]['cascade']), ci(c[t]['Lambda'][1], c[t]['Lambda_lo'][1], c[t]['Lambda_hi'][1])])
    br = [d['holding'][p]['breakable']['time_frac_solved_pooled'] for p in d['holding']]
    assert all(d['holding'][p]['locked']['time_frac_solved_pooled'] == 1.0 and d['holding'][p]['locked']['crossings_per_1000_sweeps'] == 0
               and d['avalanches'][p]['locked']['unsolved_after_first_solution'] == 0 for p in d['holding'])
    Q['G_n'] = d['n_graphs']; Q['G_held'] = f'{100 * min(br):.1f}–{100 * max(br):.1f}%'
    notes = ('Breakable, graphs in which some proper 3-colouring has an edge whose two ends can change, in the same round, to the same colour (Methods~6 of the main text); '
             'locked, graphs without such an edge, in which a consistent colouring, once reached, is never lost. The type is a property of the graph and is the same at every $p$. '
             'Held, never and returns as in Table~\\ref{tab:supp_reaching}. Solved, share of avalanches within the solved state; large, avalanches at or above the pooled $s^*$ that begin in an unsolved state, '
             'with their number per trial, the share followed by a solution (resolve) and the mean excess $\\langle s - s^*\\rangle$; in locked graphs they come only from the period before the first solution. '
             'Cascade, probability that an injection becomes a system-wide cascade (Wilson 95\\% interval); $\\Lambda$, growth rate of an injected inconsistency at $\\tau = 1$ sweep (bootstrap over graphs).')
    header = [' & & \\multicolumn{3}{c}{Holding} & \\multicolumn{4}{c}{Avalanches} & \\multicolumn{2}{c}{Injection}',
              '\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}',
              '$p$ & graphs & held (\\%) & never (\\%) & returns & solved (\\%) & large/trial & resolve & excess & cascade [95\\% CI] & $\\Lambda$ [95\\% CI]']
    cap = (f"\\textbf{{ARM results by graph type.}} {d['n_graphs']['breakable']:,} breakable and {d['n_graphs']['locked']} locked graphs among the 1{{,}}000 core graphs; "
           'holding and avalanches from the core data (50{,}000 rounds), injections from the kicked runs of main Fig.~5e, f.').replace(',000 core', '{,}000 core')
    return table('tab:supp_graph_types', cap, 'c l c c c c c c r c c', header, rows, size='\\scriptsize', sep='1.6pt', notes=notes)


# ---------- S3: the range above the CCDF knee, step clock ----------
def knee_table(Q):
    h = read('aval/hump_same_region.json')
    rows, alphas, ksv = [], [], []
    for p in [k for k in h if k != 'joint']:
        r = h[p]
        rows.append([p, str(r['xmin']), big(r['n_tail']), num(r['decades'], 1), num(r['tau_OLS_paper'], 2), r['best_MLE'],
                     big(r['dAIC']['PL']), ci(r['TPL_alpha'], *r['TPL_alpha_ci']),
                     ci(r['TPL_cutoff'], *r['TPL_cutoff_ci'], f=sig), num(r['KS']['TPL'], 3), num(r['KS']['PL'], 3)])
        if float(p) <= 0.45:
            alphas.append(r['TPL_alpha']); ksv.append(r['KS']['TPL'])
    j = h['joint']
    Q['S3_alpha'] = rng(alphas); Q['S3_ks'] = rng(ksv, 3)
    Q['S3_cut_lo'] = big(h['0.05']['TPL_cutoff']); Q['S3_cut_hi'] = big(h['0.65']['TPL_cutoff'])
    cap = ('\\textbf{Maximum likelihood on the range above the CCDF knee (step clock).} '
           'The range is all avalanches above the knee located by a least-squares fit of the complementary cumulative distribution (CCDF; main Methods, section~7); '
           'the same data and the same range are used for both estimates.')
    notes = (f'Lower end, the knee; decades, $\\log_{{10}}(s_{{\\max}}/s_{{\\rm knee}})$; $\\tau$, exponent of the least-squares CCDF fit; '
             '$\\alpha$ and cut-off, maximum-likelihood TPL parameters with 95\\% trial-bootstrap intervals; '
             '$\\Delta$AIC$_{\\rm PL}$, AIC of PL minus that of the best candidate. '
             f'A TPL with one $\\alpha$ shared by all $p$ ($\\alpha = {j["alpha_common"]:.2f}$, cut-offs free) is worse than separate exponents by '
             f'$\\Delta$AIC $= {big(j["dAIC_common_minus_separate"])}$ ({j["dLL_per_avalanche"]:.3f} nats per avalanche). '
             'KS distances of 0.06--0.10 compare with 0.005--0.02 for well-fitting ranges; the fitted $\\alpha$ also depends strongly on the lower end '
             '(Supplementary Fig.~S5\\,j), so the range is described by its characteristic size rather than by an exponent.')
    header = ['$p$ & lower end & $n$ & decades & $\\tau$ (CCDF) & best & $\\Delta$AIC$_{\\rm PL}$ & TPL $\\alpha$ [95\\% CI] & TPL cut-off [95\\% CI] & KS$_{\\rm TPL}$ & KS$_{\\rm PL}$']
    return table('tab:supp_knee', cap, 'c r r c c c r c c c c', header, rows, size='\\scriptsize', sep='2.5pt', notes=notes)


# ---------- S4: two populations, primary values and sensitivity ----------
def population_table(Q):
    plan = sorted([r for r in read('aval/planC_sweep/planC.json')['summary'] if r['model'] == 'ARM'], key=lambda r: float(r['p']))
    ep = read('aval/planC_sweep/episodes.json')
    rows_a = []
    for r in plan:
        e = ep[r['p']]
        rows_a.append([r['p'], str(r['s_star']), num(r['frac_solved'], 3), num(r['frac_resolve'], 3), num(r['frac_open'], 3),
                       big(e['nontrivial_n']), ci(*e['nontrivial_via_large']),
                       ci(r['resolve_share_large'], r['resolve_share_large_lo'], r['resolve_share_large_hi']),
                       ci(r['mean_excess'], r['mean_excess_lo'], r['mean_excess_hi'], f=sig)])
    Q['S4_solved_range'] = f"{min(r['frac_solved'] for r in plan):.2f}–{max(r['frac_solved'] for r in plan):.2f}"
    Q['S4_quiet_max'] = max(max(ep[p]['nontrivial_quiet']) for p in ep)
    cap_a = ('\\textbf{Two populations of avalanches (ARM, sweep clock, primary definition).} '
             'Avalanches are runs of sweeps with more than one colour change per sweep.')
    notes_a = ('$s^*$, the size above which avalanches within the solved state make up less than 1\\%. '
               'Solved / resolve / open, shares of all avalanches that stay within the solved state, are followed by a solution, or leave the system unsolved. '
               '$n_{\\rm ep}$, number of non-trivial unsolved episodes (size $\\ge s^*$); via large, share of them resolved through a large avalanche. '
               'Resolve share, share of large avalanches ($s \\ge s^*$, solved class excluded) that are followed by a solution. '
               'Mean excess, mean of $s - s^*$ over the same avalanches (colour changes). No non-trivial episode resolved without an avalanche. '
               'Intervals are 95\\% trial-bootstrap intervals.')
    header_a = ['$p$ & $s^*$ & solved & resolve & open & $n_{\\rm ep}$ & via large [95\\% CI] & resolve share [95\\% CI] & mean excess [95\\% CI]']
    ta = table('tab:supp_populations', cap_a, 'c r c c c r c c c', header_a, rows_a, size='\\scriptsize', sep='3.5pt', notes=notes_a)

    variants = [('Threshold (changes per sweep)', [('sweep', '1 (primary)'), ('A3_sweep_th0_k1', '0'), ('A3_sweep_th2_k1', '2'), ('A3_sweep_th3_k1', '3')]),
                ('Record', [('A3_sweep_nobound', 'record ends excluded'), ('A3_sweep_stat', 'after first solution'),
                            ('A3_sweep_T10000', 'first 10{,}000 rounds'), ('A3_sweep_T25000', 'first 25{,}000 rounds')]),
                ('Clock', [('primary', 'step clock'), ('A3_sweep_th1_k2', 'bins of two sweeps')]),
                ('Rule defining $s^*$', [('SSTAR_0', 'solved class $<0.5\\%$'), ('SSTAR_2', 'solved class $<2\\%$')])]
    sel = ['0.05', '0.25', '0.45', '0.65', '0.85']
    rows_b = []
    for group, vv in variants:
        rows_b.append(f'\\multicolumn{{11}}{{l}}{{\\textit{{{group}}}}} \\\\')
        for key, label in vv:
            if key.startswith('SSTAR_'):
                sr = read('aval/planC_sweep/sstar_rule_sensitivity.json'); j = int(key[-1])
                rr = {p: {'resolve_share_large': v[j][1], 'mean_excess': v[j][2]} for p, v in sr.items()}
            else:
                rr = {r['p']: r for r in read(f'aval/planC_{key}/planC.json')['summary'] if r['model'] == 'ARM'}
            def v(p, k, f):
                x = rr[p].get(k)
                return DASH if x is None or not np.isfinite(x) or (k == 'mean_excess' and x <= 0) else f(x)
            rows_b.append([label] + [v(p, 'resolve_share_large', lambda x: num(x, 2)) for p in sel]
                          + [v(p, 'mean_excess', lambda x: sig(x)) for p in sel])
    cap_b = ('\\textbf{Sensitivity of the two-population statistics to the avalanche definition (ARM).} '
             'Each definition uses its own $s^*$.')
    notes_b = ('Resolve share and mean excess as in Table~\\ref{tab:supp_populations}; intervals for the threshold, record and clock variants are shown in Supplementary Fig.~S6. '
               'Rule defining $s^*$: the solved-class share above $s^*$ is required to be below 0.5\\% or 2\\% instead of 1\\%. '
               '--- marks an undefined value (no large avalanche or $s^*$ undefined) or a zero mean excess. '
               'In the step clock a step contains on average only $20(1-p)$ activations, so a fixed per-step threshold becomes stricter as $p$ grows; '
               'bins of two sweeps merge the two populations. Both are reported as limiting cases, not as alternative definitions.')
    header_b = [' & \\multicolumn{5}{c}{Resolve share of large avalanches} & \\multicolumn{5}{c}{Mean excess size}',
                '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}',
                'Definition \\hfill $p$ & ' + ' & '.join(sel) + ' & ' + ' & '.join(sel)]
    tb = table('tab:supp_sensitivity', cap_b, 'l c c c c c r r r r r', header_b, rows_b, size='\\scriptsize', sep='3pt', notes=notes_b)
    return ta + '\n' + tb


# ---------- S5: spectra ----------
def spectra_table(Q):
    sp = read('spec/spectra.json'); su = read('spec/spectra_supplement.json'); ag = read('agentspec/agent_spectra.json')
    w = '[-3.0,-1.0]'
    rows, full_beta, nl = [], [], []
    for p in PS:
        s = sp[f'ARM_{p}']; a = ag[f'ARM_{p}']; x = su.get(p)
        st = s['win_power_sweep'][w]
        rows.append([p, num(s['fig4_full'][0]), num(s['fig4_full_never_leave'][0]), num(s['fig4_full_other'][0]), pct(s['share_never_leave'], 1),
                     ci(st['beta'], *st['ci']) if 'ci' in st else num(st['beta']),
                     num(x['per_trial_beta_median_iqr'][0], 3) + f" [{num(x['per_trial_beta_median_iqr'][1], 3)}, {num(x['per_trial_beta_median_iqr'][2], 3)}]" if x else DASH,
                     pct(x['power_share_top5pct_trials'], 0) if x else DASH,
                     num(s['X_real_sweep'][w]['beta']), num(s['X_shuffled_sweep'][w]['beta']),
                     num(a['a_step']['alpha_of_mean_spectrum']), num(a['b_step']['alpha_of_mean_spectrum'])])
        full_beta.append(s['fig4_full'][0]); nl.append(s['share_never_leave'])
    rows.append('\\midrule')
    for e in ['0.05', '0.15', '0.25', '0.45']:
        s = sp[f'SNM_{e}']; a = ag[f'SNM_{e}']; st = s['win_power_sweep'][w] if s.get('win_power_sweep') else None
        rows.append([f'SNM {e}', num(s['fig4_full'][0]), DASH, DASH, pct(s['share_never_leave'], 1) if s.get('share_never_leave') is not None else DASH,
                     ci(st['beta'], *st['ci']) if st and 'ci' in st else (num(st['beta']) if st else DASH), DASH, DASH, DASH, DASH,
                     num(a['a_step']['alpha_of_mean_spectrum']), num(a['b_step']['alpha_of_mean_spectrum'])])
    Q['S5_beta_mean'] = f'{np.mean(full_beta):.3f}'
    Q['S5_never'] = f'{100 * min(nl):.0f}–{100 * max(nl):.0f}%'
    Q['S5_top5'] = f"{100 * min(x['power_share_top5pct_trials'] for x in su.values()):.0f}–{100 * max(x['power_share_top5pct_trials'] for x in su.values()):.0f}%"
    low = [p for p in PS if float(p) <= 0.65]
    Q['S5_agent_a'] = rng([ag[f'ARM_{p}']['a_step']['alpha_of_mean_spectrum'] for p in low])
    Q['S5_agent_b'] = rng([ag[f'ARM_{p}']['b_step']['alpha_of_mean_spectrum'] for p in low])
    Q['S5_snm005'] = num(ag['SNM_0.05']['a_step']['alpha_of_mean_spectrum'])
    cap = '\\textbf{Spectral exponents of the conflict ratio and per-agent colour spectra.}'
    notes = ('Full record: amplitude exponent $\\beta$ of the trial-averaged $\\log_{10}|\\hat C(f)|$, fitted on $10^{-3.5}$--$10^{-1}$ cycles per step, '
             'for all trials, for trials that never leave the consistent state after their first solution (\\emph{never}), and for the others (\\emph{exc.}); '
             '\\% never, share of the never-leaving trials. Stationary: last $2^{15}$ rounds, fitted on $10^{-3}$--$10^{-1}$ cycles per sweep; '
             '\\emph{ensemble} is the exponent of the mean power, \\emph{trial} the median per-trial exponent [interquartile range], '
             '\\emph{top 5\\%} the share of power held by the 5\\% of nonconstant trials with the highest variance (computed at five $p$). '
             'Indicator: solved/unsolved indicator, observed and renewal surrogate (excursions and consistent spells reshuffled). '
             'Per agent: slope $\\alpha$ of the site-summed colour spectrum in 1{,}024 rounds, from a random start (protocol of Uragami and Gunji) and after the first solution; 100 trials per condition. '
             '$\\beta$ is an amplitude exponent; the power exponent is $2\\beta$.')
    header = [' & \\multicolumn{4}{c}{Full record} & \\multicolumn{3}{c}{Stationary} & \\multicolumn{2}{c}{Indicator} & \\multicolumn{2}{c}{Per agent $\\alpha$}',
              '\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}',
              '$p$ / $\\varepsilon$ & all & never & exc. & \\% never & ensemble [95\\% CI] & trial [IQR] & top 5\\% & obs. & renewal & start & after']
    return table('tab:supp_spectra', cap, 'l c c c c c c c c c c c', header, rows, size='\\scriptsize', sep='3pt', notes=notes)


# ---------- S6: reaching and holding ----------
def passage_table(Q):
    fp = read('fpt/first_passage.json'); rb = read('robust/robustness.json'); mw = read('fpt/mannwhitney.json')
    fa = read('fpt/factorial_2x2.json')
    rob = {(r['model'], round(r['p'], 2)): r for r in rb['rows']}
    rows, held, never, rand_s, held_lv = [], [], [], [], []
    for model in ('SNM', 'ARM'):
        for r in sorted([r for r in fp['rows']['long'] if r['model'] == model], key=lambda r: r['p']):
            q = rob[(model, round(r['p'], 2))]
            ms = [np.nan if v is None else v for v in r['median_s']]; mr = [np.nan if v is None else v for v in r['median_r']]
            rows.append([f"{model} {r['p']:.2f}", pct(r['succ_r1000'][0]), pct(r['succ_r50000'][0]),
                         ci(*ms, f=tfmt) if np.isfinite(ms[0]) else DASH, tfmt(mr[0]) if np.isfinite(mr[0]) else DASH,
                         pct(r['time_frac_solved_pooled']),
                         pct(r['leaving_time_frac_solved']) if np.isfinite(r['leaving_time_frac_solved']) else DASH, pct(r['share_never_leave'], 1),
                         num(r['crossings_per_1000_sweeps'], 1) if np.isfinite(r['crossings_per_1000_sweeps']) else DASH,
                         num(q['rand_per_sweep_search']), num(q['rand_per_sweep_stationary'])])
            if model == 'ARM':
                held.append(r['time_frac_solved_pooled']); never.append(r['share_never_leave']); held_lv.append(r['leaving_time_frac_solved'])
                if r['p'] >= 0.1:
                    rand_s.append(q['rand_per_sweep_search'])
        if model == 'SNM':
            rows.append('\\midrule')
    m = mw['fine_SNM_0.05|fine_ARM_0.45|sweeps']
    Q['S6_med'] = f"{m['med_x']:.1f} and {m['med_y']:.1f} sweeps"; Q['S6_mw'] = f"p = {m['p']:.3f}"
    Q['S6_held'] = f'{100 * min(held):.0f}–{100 * max(held):.1f}%'
    Q['S6_never'] = f'{100 * min(never):.0f}–{100 * max(never):.0f}%'
    Q['S6_held_leaving'] = f'{100 * min(held_lv):.1f}–{100 * max(held_lv):.1f}%'
    Q['S6_fact'] = [f"{100 * v['rate']:.1f}%" for v in fa.values()]
    fine_rand = [r['rand_per_sweep_search'] for r in rb['rows'] if r['model'] == 'ARM' and r['p'] >= 0.0999]
    Q['S6_rand'] = rng(fine_rand, 1)
    cap = '\\textbf{Reaching and holding a consistent colouring.} 1{,}000 trials of 50{,}000 rounds per condition.'
    notes = ('Reached, share of trials with a proper colouring within 1{,}000 or 50{,}000 rounds. Median time to the first solution in sweeps '
             '(95\\% bootstrap interval) and in rounds; a sweep is $N$ activations, one round of the SNM and on average $1/(1-p)$ rounds of the ARM. '
             'Held, share of time in a consistent colouring after the first solution (pooled); if left, the same restricted to trials that leave the consistent state at least once. Never, share of trials whose first consistent spell lasts to the end of the record. '
             'Returns, returns to a consistent colouring per 1{,}000 sweeps after the first solution, pooled over trials (sweep clock). '
             'Random reselections per sweep before and after the first solution (1{,}000-round records): $\\varepsilon N$ for the SNM, the number of active gridlocked agents for the ARM. '
             f"At the representative values the median times are {m['med_x']:.1f} (SNM, $\\varepsilon=0.05$) and {m['med_y']:.1f} sweeps (ARM, $p=0.45$); "
             f"Mann--Whitney $p = {m['p']:.3f}$.")
    header = [' & \\multicolumn{2}{c}{Reached (\\%)} & \\multicolumn{2}{c}{Median first solution} & \\multicolumn{4}{c}{Holding} & \\multicolumn{2}{c}{Reselections / sweep}',
              '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}',
              'Model & 1{,}000 r & 50{,}000 r & sweeps [95\\% CI] & rounds & held (\\%) & if left (\\%) & never (\\%) & returns & before & after']
    return table('tab:supp_reaching', cap, 'l c c c r c c c c c c', header, rows, size='\\scriptsize', sep='1.8pt', notes=notes)


# ---------- S2: controls (E1-E3) ----------
def wilson(k, n, z=1.959963984540054):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0, c - h), min(1, c + h)


def controls_table(Q):
    import pandas as pd
    d = pd.read_pickle(RUNS / 'pilots/pilot_factorial.pkl')
    names = {'sync': 'common clock ($p = 0$)', 'partial': 'individual time ($p = 0.45$)', 'sequential': 'random sequential'}
    rules = {'stay': 'stay', 'random': 'random reselection', 'minconf': 'min-conflicts'}
    rows = []
    for tm in ('sync', 'partial', 'sequential'):
        for ru in ('stay', 'random', 'minconf'):
            x = d[(d.timing == tm) & (d.gridlock_rule == ru)]
            w = wilson(int(x.solved.sum()), len(x))
            s = x[x.solved == 1]
            med = np.median(s.activations_to_solve / 20) if len(s) else np.nan
            held = s.frac_solved_after_first.mean() if len(s) else np.nan
            brk = s.n_breaks_after_first.mean() if len(s) else np.nan
            rows.append([names[tm] if ru == 'stay' else '', rules[ru], ci(*w, f=lambda v: pct(v)), tfmt(med),
                         pct(held) if np.isfinite(held) else DASH, num(brk, 1) if np.isfinite(brk) else DASH])
        rows.append('\\midrule' if tm != 'sequential' else '')
    rows = [r for r in rows if r != '']
    e2 = pd.read_pickle(RUNS / 'pilots/pilot_e2_sync_gated_q.pkl').groupby('q')['solved'].agg(['sum', 'count'])
    e1 = pd.read_pickle(RUNS / 'pilots/pilot_e1_snm_minconf.pkl').groupby('p_noise')['solved'].agg(['sum', 'count'])
    qs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    es = [0.0, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4]
    line_q = ' & '.join(pct(e2.loc[q, 'sum'] / e2.loc[q, 'count']) for q in qs)
    line_e = ' & '.join(pct(e1.loc[e, 'sum'] / e1.loc[e, 'count']) for e in es)
    Q['T2_minconf'] = (pct(e1['sum'].max() / 1000), f"{e1.index[np.argmax(e1['sum'].to_numpy())]:.2f}")
    cap = ('\\textbf{Control models for the timing and the gridlock rule} ($N=20$, $k=3$, $\\varepsilon=0$, 1{,}000 graphs with the seeds of the main data).')
    notes = ('Reached: share of trials with a proper colouring within 1{,}000 rounds (common clock, individual time) or 1{,}000 sweeps (random sequential), with Wilson 95\\% intervals. '
             'Median sweeps to the first solution, held share of time after it, and mean number of departures from the consistent state after it, over solved trials. '
             'Min-conflicts: a gridlocked agent moves to a colour with the fewest conflicts (random tie-break). Random sequential updating lets exactly one agent act per step and therefore presupposes a global ordering.')
    header = ['Timing & Gridlock rule & Reached (\\%) [95\\% CI] & Median sweeps & Held (\\%) & Departures']
    t1 = table('tab:supp_controls', cap, 'l l c r c r', header, rows, size='\\footnotesize', sep='3pt', notes=notes)
    def block(title, first, labels, values, note):
        return ('\\par\\medskip{\\footnotesize\\textbf{' + title + '}}\\par\\smallskip\n\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
                '\\begin{tabular}{l' + ' c' * len(labels) + '}\n\\toprule\n' + first + ' & ' + ' & '.join(labels) + ' \\\\\n\\midrule\n'
                'Reached (\\%) & ' + values + ' \\\\\n\\bottomrule\n\\end{tabular}\n'
                '\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize ' + note + '}\n')
    b2 = block('Gated reselection on a common clock (1{,}000 rounds).', '$q$', [f'{q:g}' for q in qs], line_q,
               'All agents act at every round; each gridlocked agent redraws uniformly with probability $q$ and otherwise stays. '
               '$q = 0$ is the synchronous stay rule and $q = 1$ synchronous random reselection.')
    b3 = block('SNM with the min-conflicts gridlock rule (1{,}000 rounds).', '$\\varepsilon$', [f'{e:g}' for e in es], line_e,
               'A gridlocked agent moves to a colour with the fewest conflicts instead of staying; behavioural noise $\\varepsilon$ as in the SNM. '
               'The full scan covers $\\varepsilon = 0$--$0.98$ in steps of 0.02.')
    return t1.replace('\\end{table}', b2 + b3 + '\\end{table}')



# ---------- S9: departures from a proper colouring ----------
def departure_table(Q):
    dp = read('departure/departure_summary.json')
    rows = []
    for p in sorted(dp):
        r = dp[p]
        deg2 = r['pair_min_degree_counts'].get('2', 0) / r['conflict_edges_checked']
        rows.append([p, pct(r['share_never_leave']), pct(r['n_not_structural'] / r['n_solved']),
                     pct(r['never_leave_if_not_structural']), pct(r['never_leave_if_structural']),
                     big(r['conflict_edges_checked']), str(r['violations']), pct(r['share_departures_one_edge'], 0), pct(deg2, 0)])
    Q['T9_struct'] = pct(dp['0.45']['n_not_structural'] / dp['0.45']['n_solved'], 0)
    cap = ('\\textbf{Departures from a proper colouring (ARM, $N=20$, $k=3$, 1{,}000 graphs, 50{,}000 rounds).}')
    notes = ('All shares in \\%. Never leave: share of trials that do not leave the consistent state after the first solution. No pair: share of graphs in which no proper 3-colouring '
             'contains an edge whose endpoints each have a second conflict-free colour (decided by exhaustive search); such graphs cannot leave the consistent state. '
             'Edges examined: conflicting edges at the first round of up to 20 departures per trial; violations: edges that do not consist of two agents that both changed to the same colour '
             'with two conflict-free colours each. One edge: share of departures with a single conflicting edge. Degree 2: share of departure edges whose lower-degree agent has degree 2.')
    header = [' & & & \\multicolumn{2}{c}{Never leave, graphs} & \\multicolumn{4}{c}{Departures}',
              '\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}',
              '$p$ & Never leave & No pair & without pair & with pair & Edges & Violations & One edge & Degree 2']
    return table('tab:supp_departures', cap, 'c c c c c r c c c', header, rows, size='\\footnotesize', sep='5pt', notes=notes)


# ---------- consistency with the supplementary text ----------
SUPP_TEX = ROOT.parent / 'tex_rev1' / 'supplementary.tex'


def norm_tex(s):
    s = s.replace('{,}', ',').replace('--', '–').replace('\\%', '%').replace('~', ' ').replace('$', '')
    for a, b in [('\\varepsilon', 'ε'), ('\\alpha', 'α'), ('\\beta', 'β'), ('\\tau', 'τ'), ('s^*', 's*'),
                 ('\\le ', '≤ '), ('\\ge ', '≥ '), ("\\'", ''), ('\\H', '')]:
        s = s.replace(a, b)
    return s


def check_captions(Q):
    pb = {p: v for p, v in Q['S1_plbest'].items() if p != '0.45'}
    hi = [v for p, v in pb.items() if p != '0.85']
    assert len(hi) == 8 and min(hi) > 0.85 > 0.7 > pb['0.85']
    Q['S1_boot_hi'] = f'{pct(min(hi), 0)}–{pct(max(hi), 0)}%'; Q['S1_boot_085'] = f"{pct(pb['0.85'], 0)}%"
    supp = norm_tex(SUPP_TEX.read_text()) if SUPP_TEX.exists() else ''
    fp = read('fpt/first_passage.json'); sc = fp['scans']
    hy = read('supp/hybrid.json'); nw = read('supp/networks.json'); cz = read('supp/colourability.json'); pl = read('supp/plateau.json')
    hv = {r['value']: r['solved'][0] for r in hy['ARM_p0.45_vary_eps']}; sv = {r['value']: r['solved'][0] for r in hy['SNM_eps0.05_vary_p']}
    sens = [s['exponent'] for s in pl['sensitivity']]
    er_snm = nw['er_SNM'][0]['solved'][0] / cz['er']['share_3colourable']; ws_snm = nw['ws_SNM'][0]['solved'][0] / cz['ws']['share_3colourable']
    arm_rel = [r['solved'][0] / cz[g]['share_3colourable'] for g in ('er', 'ws') for r in nw[f'{g}_ARM'] if 0.24 < r['value'] < 0.76]
    sst = read('aval/planC_sweep/sstar_rule_sensitivity.json')
    assert max(max(abs(v[0][1] - v[1][1]), abs(v[2][1] - v[1][1])) for v in sst.values()) <= 0.04
    fs = read('fpt_size/fpt_size.json')
    fcell = lambda m, k, n, par: next(c for c in fs['cells'] if c['model'] == m and c['k'] == k and c['N'] == n and c['par'] == par)
    arm4 = [fcell('ARM', 4, n, 0.45) for n in range(20, 101, 10)]; assert all(c['solved'] == 1.0 for c in arm4)
    half = min(c['N'] for c in fs['cells'] if c['model'] == 'SNM' and c['k'] == 4 and c['par'] == 0.05 and c['solved'] < 0.5)
    reach = lambda m, n: sorted(c['par'] for c in fs['cells'] if c['model'] == m and c['k'] == 4 and c['N'] == n and c['median_rounds'] is not None)
    snm_last = max(n for n in range(20, 101, 10) if reach('SNM', n))
    k3 = [(fcell('ARM', 3, n, 0.45)['median_rounds'], fcell('SNM', 3, n, 0.05)['median_rounds']) for n in range(20, 51, 10)]
    assert all(a > s for a, s in k3)
    fpt_expect = [
        f"grows from {arm4[0]['median_rounds']:.0f} rounds at N = 20 to {arm4[-1]['median_rounds']:.0f} at N = 100",
        f"grows from {fcell('SNM', 4, 20, 0.05)['median_rounds']:.0f} rounds to {fcell('SNM', 4, 60, 0.05)['median_rounds']:,.0f} at N = 60, and from N = {half} fewer than half",
        f"only up to N = {snm_last} (at N = {snm_last} only for ε = {reach('SNM', snm_last)[0]:.2f})" if len(reach('SNM', snm_last)) == 1 else 'SNM range changed',
        f"for p = {reach('ARM', 100)[0]:.2f}–{reach('ARM', 100)[-1]:.2f} at N = 100",
    ]
    casc = {}
    for q in ('0.05', '0.95'):
        with np.load(RUNS / f'kick/kick_raw_ARM_{q}.npz') as z:
            k = z['kicks']; casc[q] = float(((k[:, 3] >= 15) | (k[:, 2] >= 100)).mean())
    fpt_expect.append(f"falls from {casc['0.05']:.2f} at p = 0.05 to {casc['0.95']:.2f} at p = 0.95 (panel j)")
    gt = read('graph_types/graph_types.json'); plan_all = {r['p']: r for r in read('aval/planC_sweep/planC.json')['summary'] if r['model'] == 'ARM'}
    gb = [gt['holding'][q]['breakable']['time_frac_solved_pooled'] for q in gt['holding']]
    dex = max(abs(gt['avalanches'][q]['breakable']['mean_excess'] / plan_all[q]['mean_excess'] - 1) for q in gt['avalanches'])
    dres = max(abs(gt['avalanches'][q]['breakable']['resolve_share_large'] - plan_all[q]['resolve_share_large']) for q in gt['avalanches'])
    inj = gt['injection']; ps_ = sorted(inj, key=float)
    lam = {t_: [inj[q][t_]['Lambda'][1] for q in ps_] for t_ in ('breakable', 'locked')}
    assert all(lam[t_][4] > 0 > lam[t_][5] for t_ in lam) and all(all(x > y for x, y in zip(lam[t_], lam[t_][1:])) for t_ in lam)
    tt = read('graph_types/trial_types.json'); ksmax, share, med_l = 0.0, [], None
    for q in sorted(gt['avalanches'], key=float):
        types = np.array(tt[f'ARM_{q}'])
        with np.load(RUNS / f'aval/hist_ARM_{q}.npz') as z:
            sol = z['sweep|solved']; uns = np.concatenate([z['sweep|resolve'], z['sweep|open']], axis=1)
        def part(a, ty):
            return a[:, types[a[0].astype(int)] == ty]
        def cdf(a, grid):
            u, inv = np.unique(a[1], return_inverse=True); c = np.bincount(inv, weights=a[2]); F = np.cumsum(c) / c.sum()
            i = np.searchsorted(u, grid, side='right') - 1
            return np.where(i >= 0, F[np.clip(i, 0, None)], 0.0)
        b_, l_ = part(sol, 'breakable'), part(sol, 'locked'); grid = np.union1d(b_[1], l_[1])
        ksmax = max(ksmax, float(np.max(np.abs(cdf(b_, grid) - cdf(l_, grid)))))
        ul = part(uns, 'locked'); share.append(ul[2].sum() / uns[2].sum())
        if q == '0.05':
            o = np.argsort(ul[1]); cs = np.cumsum(ul[2][o]); med_l = int(ul[1][o][np.searchsorted(cs, 0.5 * cs[-1])])
    fpt_expect += [
        f"between the types at most {ksmax:.2f}",
        f"locked graphs contribute {100 * min(share):.2f}–{100 * max(share):.0f}% of them",
        f"(median {med_l:,} colour changes at p = 0.05)",
        f"for the {gt['n_graphs']['breakable']} breakable and the {gt['n_graphs']['locked']} locked graphs",
        f"spends {100 * min(gb):.1f}–{100 * max(gb):.1f}% of the time after the first solution",
        f"by at most {100 * dex:.0f}% and {dres:.2f}",
        f"({inj[ps_[0]]['breakable']['cascade'][0]:.2f} to {inj[ps_[-1]]['breakable']['cascade'][0]:.2f} in breakable and {inj[ps_[0]]['locked']['cascade'][0]:.2f} to {inj[ps_[-1]]['locked']['cascade'][0]:.2f} in locked graphs)",
    ]
    expect = fpt_expect + [
        f"exponent {Q['S1_alpha']}", f"KS distances {Q['S1_ks']}", f"s ≥ {Q['S1_xmin']}",
        f"{Q['S1_above']} of the avalanches in these tails lie below s*",
        f"from {Q['S1_flick_lo']} at p = 0.05 to {Q['S1_flick_hi']} at p = 0.95",
        f"in {Q['S1_boot_hi']} of replicates at eight of these nine p and in {Q['S1_boot_085']} at p = 0.85",
        f"α = {Q['S3_alpha']} for p ≤ 0.45, but with KS distances {Q['S3_ks']}",
        f"make up {Q['S4_solved_range']} of all ARM avalanches",
        f"β = {Q['S5_beta_mean']}", f"make up {Q['S5_never']} of all trials", f"hold {Q['S5_top5']} of the power",
        f"α = {Q['S5_agent_a']} for p ≤ 0.65", f"α = {Q['S5_agent_b']} in windows", f"(α = {Q['S5_snm005']})",
        f"widens from {sc['SNM_fine_r200']['width']:.2f} at 200 rounds to {sc['SNM_fine_r500']['width']:.2f} at 500 and {sc['SNM_fine_r1000']['width']:.2f} at 1,000 rounds",
        f"is {sc['ARM_fine_r500']['width']:.2f} wide at 500 rounds and {sc['ARM_fine_r1000']['width']:.2f} at 1,000 rounds",
        f"from {100 * hv[0.0]:.1f}% at ε = 0 to {100 * hv[0.15]:.1f}% at ε = 0.15 and {100 * hv[0.25]:.1f}% at ε = 0.25",
        f"at or above {100 * min(v for k, v in sv.items() if k <= 0.7):.1f}% up to p = 0.70", f"{100 * sv[0.95]:.1f}% at p = 0.95",
        f"a = {pl['exponent']:.2f} (95% bootstrap interval {pl['exponent_ci'][0]:.2f}–{pl['exponent_ci'][1]:.2f})",
        f"it lies between {min(sens):.2f} and {max(sens):.2f}",
        f"rises to {pl['edge']['100']:.2f} at N = 100",
        f"only {100 * cz['er']['share_3colourable']:.0f}% and {100 * cz['ws']['share_3colourable']:.0f}% of the graphs are 3-colourable",
        f"({100 * er_snm:.1f}% and {100 * ws_snm:.1f}% of it at ε = 0.05)",
        f"reaches {100 * min(arm_rel):.0f}–{100 * max(arm_rel):.0f}% of it over p = 0.25–0.75",
    ]
    missing = [s for s in expect if s not in supp]
    assert Q['S4_quiet_max'] == 0, 'a non-trivial episode resolved without an avalanche'
    return expect, missing


def main():
    Q = {}
    s1, s2 = mle_tables(Q)
    pop = population_table(Q)
    pop_a, pop_b = pop.split('\n\n', 1) if '\n\n' in pop else (pop, '')
    files = {'table_S1_reaching_holding.tex': passage_table(Q), 'table_S2_controls.tex': controls_table(Q),
             'table_S3_populations.tex': pop_a, 'table_S4_definitions.tex': pop_b,
             'table_S5_mle_arm.tex': s1, 'table_S6_mle_snm.tex': s2, 'table_S7_knee_region.tex': knee_table(Q),
             'table_S8_spectra.tex': spectra_table(Q), 'table_S9_departures.tex': departure_table(Q),
             'table_S10_mle_populations.tex': mle_population_table(Q), 'table_S11_graph_types.tex': graph_type_table(Q)}
    for old in ROOT.glob('table_S*.tex'):
        if old.name not in files:
            old.unlink()
    for name, text in files.items():
        (ROOT / name).write_text('% Generated by make_tables.py from ../analyses/runs; do not edit by hand.\n' + text)
    preview = ['\\documentclass[11pt,a4paper]{article}', '\\usepackage[margin=2.5cm]{geometry}', '\\usepackage{amsmath,amssymb,booktabs,caption,float,array}',
               '\\usepackage[T1]{fontenc}', '\\usepackage[utf8]{inputenc}', '\\renewcommand{\\thetable}{S\\arabic{table}}', '\\begin{document}']
    preview += [f'\\input{{{n}}}\n\\clearpage' for n in files] + ['\\end{document}']
    (ROOT / 'preview.tex').write_text('\n'.join(preview) + '\n')
    expect, missing = check_captions(Q)
    (ROOT / 'quoted_numbers.json').write_text(json.dumps({k: v for k, v in Q.items()}, indent=1, ensure_ascii=False, default=str) + '\n')
    if not SUPP_TEX.exists():
        print(f'{len(files)} table files written; {SUPP_TEX} not found, quoted numbers not checked against the text')
        return
    print(f'{len(files)} table files written; {len(expect) - len(missing)}/{len(expect)} numbers in supplementary.tex matched')
    for s in missing:
        print('  NOT FOUND in supplementary.tex:', s)
    assert not missing, 'caption numbers and tables disagree'


if __name__ == '__main__':
    main()
