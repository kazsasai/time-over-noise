"""
Discrete maximum-likelihood fitting of avalanche-size distributions, working on per-trial histograms
(output of avalanche_histograms.py).  All models are densities evaluated at integer sizes s >= xmin and
normalised over s = xmin..inf: explicit sum for s < M0, analytic tail integral from M0 - 1/2 (midpoint rule).

Models (pmf ∝ f(s)):
  PL   s^-a                                (normaliser: Hurwitz zeta)
  TPL  s^-a exp(-lam s)
  LN   s^-1 exp(-(ln s - mu)^2 / 2 sig^2)
  SE   s^(b-1) exp(-(lam s)^b)             (stretched exponential / Weibull)
  EXP  exp(-lam s)
  BPL  s^-a1 exp(-lam s) for s < sb,  sb^(a2-a1) s^-a2 exp(-lam s) for s >= sb   (broken power law with a
       global exponential cutoff; a1 = a2 recovers TPL exactly)
"""
import numpy as np
from scipy import optimize, special, stats, sparse

M0 = 10_000
A_TAIL = M0 - 0.5
LOG2PI = np.log(2 * np.pi)


# ───────────────────────── data ─────────────────────────
class Hist:
    """Per-trial size histograms of one condition as a sparse (trial x size) count matrix."""

    def __init__(self, npz, key):
        self.n_trials = int(npz['n_trials'])
        if key not in npz.files:
            self.sizes = np.zeros(0, dtype=np.int64)
            self.M = sparse.csr_matrix((self.n_trials, 0))
            return
        t, s, c = npz[key]
        self.sizes, idx = np.unique(s, return_inverse=True)
        self.M = sparse.csr_matrix((c, (t, idx)), shape=(self.n_trials, self.sizes.size))

    def pooled(self, weights=None):
        """Counts per unique size, summed over trials (weights = bootstrap multiplicities)."""
        if weights is None:
            return self.sizes, np.asarray(self.M.sum(axis=0)).ravel()
        return self.sizes, np.asarray(self.M.T @ weights).ravel()


def tail_of(s, c, xmin):
    m = (s >= xmin) & (c > 0)
    return s[m].astype(float), c[m].astype(float)


# ───────────────────────── tails ─────────────────────────
def log_tail_tpl(a, lam, A=A_TAIL):
    """log ∫_A^∞ s^-a exp(-lam s) ds  (= log lam^(a-1) Γ(1-a, lam A)), lam > 0."""
    x = lam * A
    if x > 600:
        return -np.inf
    s = 1.0 - a
    if abs(s - round(s)) < 1e-9:
        s += 1e-7
    m = 0 if s > 0 else int(np.floor(-s)) + 1
    s0 = s + m
    T = lam ** (-s0) * special.gamma(s0) * special.gammaincc(s0, x)
    for k in range(m - 1, -1, -1):
        sk = s + k
        T = (lam * T - A ** sk * np.exp(-x)) / sk
    return np.log(T) if T > 0 else -np.inf


# ───────────────────────── models ─────────────────────────
# Each model: (param names, bounds, logf(theta, s, logs), log_tail(theta, xmin) or None for exact PL)
def _pl_logf(th, s, ls):
    return -th[0] * ls


def _tpl_logf(th, s, ls):
    return -th[0] * ls - np.exp(th[1]) * s


def _tpl_tail(th):
    return log_tail_tpl(th[0], np.exp(th[1]))


def _ln_logf(th, s, ls):
    return -ls - (ls - th[0]) ** 2 / (2 * th[1] ** 2)


def _ln_tail(th):
    return np.log(th[1]) + 0.5 * LOG2PI + stats.norm.logsf((np.log(A_TAIL) - th[0]) / th[1])


def _se_logf(th, s, ls):
    b, u = th
    return (b - 1) * ls - np.exp(b * (u + ls))


def _se_tail(th):
    b, u = th
    return -np.exp(b * (u + np.log(A_TAIL))) - np.log(b) - b * u


def _exp_logf(th, s, ls):
    return -np.exp(th[0]) * s


def _exp_tail(th):
    lam = np.exp(th[0])
    return -lam * A_TAIL - np.log(lam)


MODELS = {
    'PL':  dict(names=('alpha',), bounds=[(1.0001, 6.0)], logf=_pl_logf, tail=None),
    'TPL': dict(names=('alpha', 'loglam'), bounds=[(-1.0, 6.0), (-30.0, 3.0)], logf=_tpl_logf, tail=_tpl_tail),
    'LN':  dict(names=('mu', 'sigma'), bounds=[(-60.0, 40.0), (0.05, 30.0)], logf=_ln_logf, tail=_ln_tail),
    'SE':  dict(names=('beta', 'loglam'), bounds=[(0.02, 3.0), (-30.0, 12.0)], logf=_se_logf, tail=_se_tail),
    'EXP': dict(names=('loglam',), bounds=[(-30.0, 3.0)], logf=_exp_logf, tail=_exp_tail),
}
NPARAM = {k: len(v['names']) for k, v in MODELS.items()}


class Fitter:
    """Negative log-likelihoods for the tail s >= xmin of one pooled histogram."""

    def __init__(self, s, c, xmin):
        self.s, self.c, self.xmin = s, c, int(xmin)
        self.ls = np.log(s)
        self.n = c.sum()
        self.grid = np.arange(self.xmin, M0, dtype=float)
        self.lgrid = np.log(self.grid)

    def log_norm(self, model, th):
        mdl = MODELS[model]
        if mdl['tail'] is None:                                   # PL: exact
            return np.log(special.zeta(th[0], self.xmin))
        head = special.logsumexp(mdl['logf'](th, self.grid, self.lgrid)) if self.grid.size else -np.inf
        return np.logaddexp(head, mdl['tail'](th))

    def nll(self, model, th):
        lf = MODELS[model]['logf'](th, self.s, self.ls)
        v = -(self.c @ lf - self.n * self.log_norm(model, th))
        return v if np.isfinite(v) else 1e300

    def logpmf(self, model, th, s):
        s = np.asarray(s, dtype=float)
        return MODELS[model]['logf'](th, s, np.log(s)) - self.log_norm(model, th)

    def starts(self, model):
        w = self.c / self.n
        a0 = 1 + 1 / max(w @ np.log(self.s / (self.xmin - 0.5)), 1e-3)
        mean = w @ self.s
        ml = w @ self.ls
        sl = np.sqrt(max(w @ (self.ls - ml) ** 2, 1e-4))
        return {
            'PL':  [[min(a0, 5.0)]],
            'TPL': [[a0, -np.log(mean)], [max(a0 - 0.4, 0.2), -np.log(10 * mean)],
                    [a0 + 0.3, -np.log(self.s.max())], [1.2, -12.0]],
            'LN':  [[ml, sl], [ml - 5, 2 * sl + 1], [0.0, 3.0]],
            'SE':  [[0.5, -np.log(mean)], [0.2, -np.log(mean)], [0.9, -np.log(mean)]],
            'EXP': [[-np.log(max(mean - self.xmin + 1, 1.0))]],
        }[model]

    def fit(self, model, x0s=None):
        best = None
        for x0 in (x0s or self.starts(model)):
            x0 = np.clip(x0, [b[0] for b in MODELS[model]['bounds']], [b[1] for b in MODELS[model]['bounds']])
            r = optimize.minimize(lambda th: self.nll(model, th), x0, method='L-BFGS-B',
                                  bounds=MODELS[model]['bounds'])
            if best is None or r.fun < best.fun:
                best = r
        return dict(model=model, theta=best.x, ll=-best.fun, k=NPARAM[model])


def fit_all(s, c, xmin, models=('PL', 'TPL', 'LN', 'SE', 'EXP')):
    f = Fitter(s, c, xmin)
    out = {m: f.fit(m) for m in models}
    return f, out


def ic(res, n):
    k, ll = res['k'], res['ll']
    return dict(aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll)


def vuong(f, r1, r2):
    """Normalised log-likelihood ratio (Vuong) assuming i.i.d. avalanches; positive favours r1."""
    l = f.logpmf(r1['model'], r1['theta'], f.s) - f.logpmf(r2['model'], r2['theta'], f.s)
    R = f.c @ l
    var = (f.c @ (l - R / f.n) ** 2) / f.n
    Rn = R / np.sqrt(f.n * var) if var > 0 else 0.0
    return R, Rn, 2 * stats.norm.sf(abs(Rn))


def ks_distance(f, res, smax_cap=2_000_000):
    """KS distance between the empirical and fitted CDF on s >= xmin (evaluated at observed sizes)."""
    smax = int(min(f.s.max(), smax_cap))          # sizes above the cap are left out of the comparison
    if smax < f.xmin:
        return np.nan
    grid = np.arange(f.xmin, smax + 1, dtype=float)
    cdf_model = np.cumsum(np.exp(f.logpmf(res['model'], res['theta'], grid)))
    emp = np.cumsum(f.c) / f.n
    keep = f.s <= smax
    if not keep.any():
        return np.nan
    idx = (f.s[keep].astype(np.int64) - f.xmin)
    return float(np.max(np.abs(emp[keep] - cdf_model[idx])))


# ───────────────────────── xmin selection (Clauset et al. 2009) ─────────────────────────
D_STD = np.log10(5000 / 2) / 39      # grid step (decades) of the standard grid from s = 2 to 5,000


def xmin_grid(s, c, min_tail=1000, upto=5000, extend=False):
    """Candidate xmin values: 40 log-spaced points from the smallest size to `upto`.

    extend=True (rev1, 2026-10-09) continues the grid above `upto` with the step D_STD of the standard grid (from s = 2,
    i.e. about 11.5 points per decade), up to the largest size that leaves at least `min_tail` avalanches; below `upto`
    the candidates are unchanged. If the smallest size already exceeds `upto`, the grid starts there.
    """
    lo = max(s.min(), 1)
    if not extend:                     # unchanged rule used by all earlier runs
        cand = np.unique(np.round(np.logspace(np.log10(lo), np.log10(upto), 40)).astype(int))
        ntail = np.array([c[s >= x].sum() for x in cand])
        return cand[ntail >= min_tail]
    if lo < upto:
        cand = np.logspace(np.log10(lo), np.log10(upto), 40)
    else:
        cand = np.array([lo], float)
    top = cand[-1]
    more = top * 10 ** (D_STD * np.arange(1, int(np.ceil(np.log10(s.max() / top) / D_STD)) + 2))
    cand = np.r_[cand, more[more <= s.max()]]
    cand = np.unique(np.round(cand).astype(int))
    ntail = np.array([c[s >= x].sum() for x in cand])
    return cand[ntail >= min_tail]


def clauset_xmin(s, c, min_tail=1000, extend=False):
    """xmin minimising the KS distance of the discrete PL fit above xmin."""
    best = None
    for x in xmin_grid(s, c, min_tail, extend=extend):
        st, ct = tail_of(s, c, x)
        f = Fitter(st, ct, x)
        r = f.fit('PL')
        a = r['theta'][0]
        cdf = 1 - special.zeta(a, st + 1) / special.zeta(a, x)        # P(S <= s)
        D = np.max(np.abs(np.cumsum(ct) / ct.sum() - cdf))
        if best is None or D < best[1]:
            best = (int(x), D, a)
    return best


# ───────────────────────── broken power law (knee) ─────────────────────────
class BPL:
    """s^-a1 e^{-lam s} (s < sb) and sb^(a2-a1) s^-a2 e^{-lam s} (s >= sb), on s >= xmin."""

    def __init__(self, s, c, xmin=2):
        self.s, self.c, self.xmin = s, c, xmin
        self.ls = np.log(s)
        self.n = c.sum()
        self.grid = np.arange(xmin, M0, dtype=float)
        self.lgrid = np.log(self.grid)

    @staticmethod
    def _logf(th, sb, s, ls):
        a1, a2, u = th
        return np.where(s < sb, -a1 * ls, -a2 * ls + (a2 - a1) * np.log(sb)) - np.exp(u) * s

    def nll(self, th, sb):
        a1, a2, u = th
        head = special.logsumexp(self._logf(th, sb, self.grid, self.lgrid))
        tail = (a2 - a1) * np.log(sb) + log_tail_tpl(a2, np.exp(u))
        v = -(self.c @ self._logf(th, sb, self.s, self.ls) - self.n * np.logaddexp(head, tail))
        return v if np.isfinite(v) else 1e300

    def fit_sb(self, sb, x0s):
        bounds = [(-1.0, 6.0), (-1.0, 6.0), (-30.0, 3.0)]
        best = None
        for x0 in x0s:
            r = optimize.minimize(lambda th: self.nll(th, sb), x0,
                                  method='L-BFGS-B', bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best

    def fit(self, tpl_theta, sb_grid):
        a, u = tpl_theta
        x0s = [[a, a, u], [a + 0.4, a - 0.3, u], [a - 0.3, a + 0.4, u]]
        prof = []
        for sb in sb_grid:
            r = self.fit_sb(sb, x0s)
            prof.append((sb, -r.fun, r.x))
        i = int(np.argmax([p[1] for p in prof]))
        sb, ll, th = prof[i]
        return dict(sb=int(sb), ll=ll, a1=th[0], a2=th[1], loglam=th[2], k=4,
                    profile=np.array([[p[0], p[1]] for p in prof]))


def sb_grid_for(s, c, min_above=1000, upto=3000, n=30):
    cand = np.unique(np.round(np.logspace(np.log10(3), np.log10(upto), n)).astype(int))
    nab = np.array([c[s >= x].sum() for x in cand])
    return cand[nab >= min_above]
