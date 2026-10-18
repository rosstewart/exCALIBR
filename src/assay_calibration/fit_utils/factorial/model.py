"""Factorial (axis-tied) mixture -- EXPERIMENTAL, opt-in.

Dimensions are grouped into AXES. Each variant has one latent STATE per
axis; a CELL is one state per axis, and each sample s has a weight table
W[s] over cells:

    f(x | s) = sum_cells W[s, c] * prod_axes f_{a, c_a}(x_a)

Axis-state densities are shared by every cell and sample, so a rare
combination (normal on one axis, defective on another) needs only a cell
weight, not its own component. One axis holding every dim is exactly the
standard (canonical) mixture.

Axis states reuse the production skew-normal M-step leaves (cfusn/
update_steps: get_location_update_mv / get_Delta_update_mv /
get_Gamma_update_mv for q=1 MSN or Gaussian, get_location_update_cfusn /
get_Delta_update_cfusn (Delta_prev cap) / get_Gamma_update_cfusn for q=2)
on the axis's columns, with responsibilities = cell posteriors summed over
the other axes. Q separates over axes, so this is the exact ECM step for each
axis. Gamma gets a factorial-specific inverse-Wishart MAP shrinkage,
IW(Psi = n0 * s * diag(var_axis), nu = n0), much stronger than production's
1e-3 ridge (the product of axes has many cells, and the weak ridge lets
states collapse and overfit -- measured on TP53: train/held-out LL gap 1.3-2.0
per row); the same prior enters the tracked objective, and production's
_psd_floor_or_reject guard reverts a state on rejection. Categorical dims inside an axis use the closed-form MAP
level-probability update (location model within the axis).

Penalised objective (monotone; checked every iteration, decrease handling as
in cfusn/fit.py):

    sum_n log f(x_n | s_n)
    + sum_{axis, state} gamma_log_prior_total(state, X_axis)
    + sum_{cat dims, states, levels} alpha log pi
    + alpha_W * sum_{s, c} log W[s, c]          (Dirichlet floor: no exact zeros)
    - tau * sum_s TC(W[s])                      (shrink toward cross-axis independence)

TC(W) = KL(W || product of W's own per-axis marginals).
"""
import itertools

import numpy as np
from scipy.special import logsumexp

from ..cfusn import density_utils as _du
from ..cfusn import update_steps as _us

STATE_KINDS = ("gaussian", "msn", "cfusn")
PLATEAU_REL_TOL = 1e-8
CATASTROPHIC_REL_TOL = 0.1
CAT_TAU, CAT_LAMBDA = 3.0, 0.1


# ── weight step: Dirichlet floor + total-correlation penalty ─────────────────
def _marginals(Wt, shape):
    T = Wt.reshape(shape)
    return [T.sum(axis=tuple(j for j in range(len(shape)) if j != a)) for a in range(len(shape))]


def total_correlation(Wt, shape):
    T = Wt.reshape(shape)
    prod = np.ones(shape)
    for a, m in enumerate(_marginals(Wt, shape)):
        sh = [1] * len(shape)
        sh[a] = shape[a]
        prod = prod * m.reshape(sh)
    pos = T > 0
    return float(np.sum(T[pos] * (np.log(T[pos]) - np.log(prod[pos]))))


def _tc_grad(Wt, shape):
    T = Wt.reshape(shape)
    g = np.log(np.where(T > 0, T, 1.0))
    for a, m in enumerate(_marginals(Wt, shape)):
        sh = [1] * len(shape)
        sh[a] = shape[a]
        g = g - np.log(np.where(m > 0, m, 1.0)).reshape(sh)
    return g.ravel()


def weight_objective(Wt, counts, tau, shape):
    """sum (nbar + alpha_W) log W - tau TC(W)."""
    pos = counts > 0
    if np.any(Wt[pos] <= 0):
        return -np.inf
    return float(np.sum(counts[pos] * np.log(Wt[pos]))) - tau * total_correlation(Wt, shape)


def weight_step(counts, W_old, tau, shape, iters=100):
    """GEM W step: exponentiated-gradient ascent on the simplex with
    backtracking, from the closed-form optimum and from W_old; never returns
    a W with a lower objective than W_old (the minoriser's touching point)."""
    tot = counts.sum()

    def ascend(W):
        W = np.maximum(W, 1e-300)
        W = W / W.sum()
        f = weight_objective(W, counts, tau, shape)
        for _ in range(iters):
            G = counts / W - tau * _tc_grad(W, shape)
            step, fn, Wn = 1.0, -np.inf, W
            while step > 1e-10:
                g = G * step / (tot + tau)
                g = g - g.max()
                Wn = W * np.exp(g)
                Wn = Wn / Wn.sum()
                fn = weight_objective(Wn, counts, tau, shape)
                if fn > f:
                    break
                step *= 0.5
            if not fn > f:
                break
            rel = (fn - f) / max(abs(f), 1.0)
            W, f = Wn, fn
            if rel < 1e-13:
                break
        return W, f

    W1, f1 = ascend(counts / tot if tot > 0 else W_old)
    f_old = weight_objective(W_old, counts, tau, shape)
    if f1 >= f_old:
        return W1
    W2, f2 = ascend(W_old.copy())
    return W2 if f2 >= f_old else W_old.copy()


# ── one axis ─────────────────────────────────────────────────────────────────
class Axis:
    """States of one axis: continuous columns (skew-normal / Gaussian, via the
    production M-step) and/or categorical columns (level probabilities)."""

    def __init__(self, cont, cats, levels, n_states, kind="msn", prior_n0_frac=0.05, prior_scale=0.1):
        if kind not in STATE_KINDS:
            raise ValueError(f"kind must be one of {STATE_KINDS}")
        self.cont, self.cats, self.levels = list(cont), list(cats), list(levels)
        self.K, self.kind = int(n_states), kind
        self.prior_n0_frac, self.prior_scale = float(prior_n0_frac), float(prior_scale)
        self.params, self.pi, self.alpha = [], [], []
        self.Psi, self.nu = None, None

    @property
    def q(self):
        return 2 if self.kind == "cfusn" else 1

    def _cont_rows(self, X):
        return ~np.all(np.isnan(X[:, self.cont]), axis=1) if self.cont else np.zeros(len(X), bool)

    def logpdfs(self, X):
        out = np.zeros((self.K, len(X)))
        if self.cont:
            rows = self._cont_rows(X)
            if rows.any():
                Xc = X[rows][:, self.cont]
                for k in range(self.K):
                    out[k, rows] = np.atleast_1d(_du._single_component_logpdf(Xc, self.params[k], True)).ravel()
        for j, c in enumerate(self.cats):
            lab = np.where(np.isnan(X[:, c]), -1, X[:, c]).astype(int)
            ok = lab >= 0
            for k in range(self.K):
                out[k, ok] += np.log(self.pi[j][k][lab[ok]])
        return out

    def init(self, X, rng):
        from sklearn.cluster import KMeans
        has = self._cont_rows(X).copy()
        for c in self.cats:
            has |= ~np.isnan(X[:, c])
        feats = []
        if self.cont:
            Xc = X[:, self.cont]
            feats.append(np.where(np.isnan(Xc), np.nanmean(Xc, axis=0), Xc))
        for j, c in enumerate(self.cats):
            oh = np.zeros((len(X), self.levels[j]))
            ok = ~np.isnan(X[:, c])
            oh[ok, X[ok, c].astype(int)] = 1.0
            oh[~ok] = oh[ok].mean(0) if ok.any() else 1.0 / self.levels[j]
            feats.append(oh)
        F = np.hstack(feats)[has]
        lab_k = KMeans(self.K, n_init=5, random_state=int(rng.integers(1 << 30))).fit(F).labels_
        idx = np.where(has)[0]
        self.params = []
        if self.cont:
            Xc = X[:, self.cont]
            p = len(self.cont)
            var = np.nanvar(Xc, axis=0) + 1e-6
            n_axis = int(self._cont_rows(X).sum())
            self.nu = max(self.prior_n0_frac * n_axis, 1.0)
            self.Psi = self.nu * self.prior_scale * np.diag(var)
            Xf = np.where(np.isnan(Xc), np.nanmean(Xc, axis=0), Xc)
            for k in range(self.K):
                pts = Xf[idx[lab_k == k]]
                mu = pts.mean(0) if len(pts) else np.nanmean(Xc, axis=0)
                cov = np.atleast_2d(np.cov(pts.T)) if len(pts) > p + 1 else np.diag(var)
                cov = cov + 1e-2 * np.diag(var)
                if self.kind == "gaussian":
                    D = np.zeros(p)
                elif self.kind == "msn":
                    D = rng.uniform(-0.1, 0.1, p) * np.sqrt(np.diag(cov))
                else:
                    D = rng.uniform(-0.1, 0.1, (p, 2)) * np.sqrt(np.diag(cov))[:, None]
                G = cov - (np.outer(D, D) if D.ndim == 1 else D @ D.T)
                ev = np.linalg.eigvalsh(G)
                if ev.min() < 1e-6:
                    G = G + (1e-6 - ev.min()) * np.eye(p)
                self.params.append((mu, D, G))
        self.pi, self.alpha = [], []
        for j, c in enumerate(self.cats):
            lab = X[:, c][~np.isnan(X[:, c])].astype(int)
            freq = np.bincount(lab, minlength=self.levels[j]).astype(float)
            freq = freq / freq.sum() if freq.sum() > 0 else np.full(self.levels[j], 1.0 / self.levels[j])
            a = CAT_TAU * ((1 - CAT_LAMBDA) * freq + CAT_LAMBDA / self.levels[j])
            self.alpha.append(a)
            states = []
            for k in range(self.K):
                lk = X[idx[lab_k == k], c]
                lk = lk[~np.isnan(lk)].astype(int)
                cnt = np.bincount(lk, minlength=self.levels[j]) + a
                states.append(cnt / cnt.sum())
            self.pi.append(states)

    def mstep(self, X, Rk, rng, n_mc=500, guard_stats=None):
        """Rk: (K, N) responsibilities of this axis's states."""
        if self.cont:
            rows = self._cont_rows(X)
            if rows.any():
                Xc = X[rows][:, self.cont]
                p = len(self.cont)
                new = []
                for k in range(self.K):
                    z = Rk[k, rows]
                    mu0, D0, G0 = self.params[k]
                    if self.kind == "cfusn":
                        D0 = _du._ensure_matrix_delta(D0)
                        mu1, eta, Psi_t, completed = _us.get_location_update_cfusn(
                            Xc, z, mu0, D0, G0, n_mc=n_mc, rng=rng, return_completed=True)
                        D1 = _us.get_Delta_update_cfusn(mu1, Xc, z, eta, Psi_t, completed=completed, Delta_prev=D0)
                        G1 = _us.get_Gamma_update_cfusn(mu1, D1, Xc, z, eta, Psi_t, completed=completed)
                    else:
                        mu1 = _us.get_location_update_mv(Xc, z, mu0, D0, G0)
                        D1 = (np.zeros_like(D0) if self.kind == "gaussian"
                              else _us.get_Delta_update_mv(mu1, Xc, z, mu0, D0, G0))
                        G1 = _us.get_Gamma_update_mv(mu1, D1, Xc, z, mu0, D0, G0)
                    n_eff = float(z.sum())
                    # exact IW MAP given (mu, Delta): the leaf returns S / n
                    G1 = 0.5 * (np.asarray(G1) + np.asarray(G1).T)
                    G1 = (G1 * n_eff + self.Psi) / (n_eff + self.nu + p + 1)
                    G1, ok = _us._psd_floor_or_reject(G1)
                    if not ok:
                        if guard_stats is not None:
                            guard_stats["gamma_rejected"] = guard_stats.get("gamma_rejected", 0) + 1
                        new.append((mu0, D0, G0))
                    else:
                        new.append((mu1, D1, G1))
                self.params = new
        for j, c in enumerate(self.cats):
            lab = np.where(np.isnan(X[:, c]), -1, X[:, c]).astype(int)
            ok = lab >= 0
            for k in range(self.K):
                cnt = np.bincount(lab[ok], weights=Rk[k][ok], minlength=self.levels[j])
                self.pi[j][k] = (self.alpha[j] + cnt) / (self.alpha[j].sum() + cnt.sum())

    def logprior(self, X):
        lp = 0.0
        if self.cont:
            p = len(self.cont)
            for _, _, G in self.params:
                G = 0.5 * (np.asarray(G) + np.asarray(G).T)
                sign, logdet = np.linalg.slogdet(G)
                if sign <= 0:
                    return -np.inf
                lp += -0.5 * (self.nu + p + 1) * logdet - 0.5 * float(np.trace(np.linalg.solve(G, self.Psi)))
        for j in range(len(self.cats)):
            lp += sum(float(np.sum(self.alpha[j] * np.log(p))) for p in self.pi[j])
        return lp


# ── the model ────────────────────────────────────────────────────────────────
class FactorialMixture:
    """axes: list of dicts {cont: [cols], cats: [cols], levels: [L per cat], n_states, kind}."""

    def __init__(self, axes, tau=5.0, alpha_w=0.05, seed=0, n_mc=500, prior_n0_frac=0.05, prior_scale=0.1,
                 tau_by_sample=None):
        """tau_by_sample: optional {sample index: tau} overriding ``tau`` for
        those samples -- e.g. a large tau on P/LP and auxiliary pathogenic
        samples pushes their weight tables toward the product of their own
        per-axis marginals, so a defect on any one axis carries its
        pathogenic weight regardless of the other axes (the single-mechanism
        candidate); benign/population samples keep the default."""
        self.axes = [Axis(a.get("cont", []), a.get("cats", []), a.get("levels", []), a["n_states"],
                          a.get("kind", "msn"), a.get("prior_n0_frac", prior_n0_frac),
                          a.get("prior_scale", prior_scale)) for a in axes]
        self.tau, self.alpha_w, self.n_mc = float(tau), float(alpha_w), int(n_mc)
        self.tau_by_sample = {int(k): float(v) for k, v in (tau_by_sample or {}).items()}
        self.rng = np.random.default_rng(seed)
        self.np_rng = np.random.RandomState(seed)
        self.shape = tuple(ax.K for ax in self.axes)
        self.cells = list(itertools.product(*[range(k) for k in self.shape]))
        self.cell_idx = [[[ci for ci, c in enumerate(self.cells) if c[a] == k] for k in range(ax.K)]
                         for a, ax in enumerate(self.axes)]
        self.guard_stats = {}
        self.overshoot = None

    def _tau(self, s):
        return self.tau_by_sample.get(s, self.tau)

    def cell_logpdfs(self, X):
        A = [ax.logpdfs(X) for ax in self.axes]
        lp = np.zeros((len(self.cells), len(X)))
        for ci, c in enumerate(self.cells):
            for a, s in enumerate(c):
                lp[ci] += A[a][s]
        return lp

    def _posteriors(self, lp, SA, W):
        R = np.zeros_like(lp)
        for s in range(SA.shape[1]):
            rows = SA[:, s]
            if rows.any():
                with np.errstate(divide="ignore"):
                    num = lp[:, rows] + np.log(W[s])[:, None]
                R[:, rows] = np.exp(num - logsumexp(num, axis=0))
        return R

    def loglik(self, X, SA, W=None):
        W = self.W if W is None else W
        lp = self.cell_logpdfs(X)
        ll = 0.0
        for s in range(SA.shape[1]):
            rows = SA[:, s]
            if rows.any():
                with np.errstate(divide="ignore"):
                    ll += float(logsumexp(lp[:, rows] + np.log(W[s])[:, None], axis=0).sum())
        return ll

    def penalty(self, X, W=None):
        W = self.W if W is None else W
        pen = sum(ax.logprior(X) for ax in self.axes)
        pen += self.alpha_w * float(np.sum(np.log(np.maximum(W, 1e-300))))
        pen -= sum(self._tau(s) * total_correlation(W[s], self.shape) for s in range(W.shape[0]))
        return pen

    def objective(self, X, SA):
        return self.loglik(X, SA) + self.penalty(X)

    def _snapshot(self):
        return ([(list(ax.params), [list(p) for p in ax.pi]) for ax in self.axes], self.W.copy())

    def _restore(self, snap):
        axes, W = snap
        for ax, (params, pi) in zip(self.axes, axes):
            ax.params, ax.pi = params, pi
        self.W = W

    def fit(self, X, SA, max_iter=2000, tol=1e-8):
        SA = np.asarray(SA, dtype=bool)
        for ax in self.axes:
            ax.init(X, self.rng)
        S = SA.shape[1]
        self.W = np.full((S, len(self.cells)), 1.0 / len(self.cells))
        objs = [self.objective(X, SA)]
        for it in range(max_iter):
            snap = self._snapshot()
            R = self._posteriors(self.cell_logpdfs(X), SA, self.W)
            for a, ax in enumerate(self.axes):
                Rk = np.array([R[idx].sum(0) for idx in self.cell_idx[a]])
                ax.mstep(X, Rk, self.np_rng, n_mc=self.n_mc, guard_stats=self.guard_stats)
            # W step: E-step at (new theta, old W), then the penalised GEM update
            lp = self.cell_logpdfs(X)
            P = self._posteriors(lp, SA, self.W)
            W_new = np.zeros_like(self.W)
            for s in range(S):
                counts = P[:, SA[:, s]].sum(1) + self.alpha_w
                W_new[s] = weight_step(counts, self.W[s], self._tau(s), self.shape)
            self.W = W_new
            obj = self.objective(X, SA)
            rel = (obj - objs[-1]) / max(abs(objs[-1]), 1.0)
            if rel < -PLATEAU_REL_TOL:
                if -rel >= CATASTROPHIC_REL_TOL:
                    raise RuntimeError(f"factorial EM objective collapsed at iter {it} (rel {rel:.2e})")
                self._restore(snap)
                self.overshoot = (it, objs[-1] - obj, -rel)
                break
            objs.append(obj)
            if abs(rel) < tol:
                break
        self.objectives = np.array(objs)
        return self

    # ── evaluation helpers ──────────────────────────────────────────────────
    def log_mix(self, X, w):
        with np.errstate(divide="ignore"):
            return logsumexp(self.cell_logpdfs(X) + np.log(np.maximum(np.asarray(w), 1e-300))[:, None], axis=0)

    def axis_marginal_weights(self, s):
        return _marginals(self.W[s], self.shape)

    def axis_posteriors(self, X, w):
        lp = self.cell_logpdfs(X) + np.log(np.maximum(np.asarray(w), 1e-300))[:, None]
        R = np.exp(lp - logsumexp(lp, axis=0))
        return [np.array([R[idx].sum(0) for idx in self.cell_idx[a]]) for a in range(len(self.axes))]
