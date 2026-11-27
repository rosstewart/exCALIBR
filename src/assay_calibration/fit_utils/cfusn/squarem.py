"""SQUAREM acceleration of the EM in cfusn/fit.py::single_fit.

EXPERIMENTAL, opt-in (``em_accelerator="squarem"``; default off, so the plain
EM path is untouched). Varadhan & Roland (2008, Scand J Stat 35:335-353),
"SqS3" step length, with the safeguards of the reference SQUAREM package.

One cycle, with F the existing EM map (update_steps.em_iteration: E-step,
M-step with every guard, weight update) acting on theta = (component params,
mixing weights W, categorical probabilities):

    theta1 = F(theta0),  theta2 = F(theta1)
    r = theta1 - theta0,  v = theta2 - theta1 - r
    alpha = ||r|| / ||v||  clamped to [1, step_max]
    theta' = theta0 + 2 alpha r + alpha^2 v          (alpha = 1 gives theta2)
    theta'' = F(theta')                               (stabilising EM step)

The difference vectors live in an unconstrained parameterisation (pack /
unpack): mu and Delta as they are, Gamma through its Cholesky factor with a
log diagonal (so theta' has a positive-definite Gamma; a step long enough to
make it numerically indefinite is shortened), W rows and
categorical probabilities through their logs, renormalised on unpacking
(entries that are exactly zero stay zero).

Safeguards -- SQUAREM defers to the EM map, it never bypasses it:
  * theta'' is accepted only if its objective is >= theta2's, so accepted
    iterates are monotone in the penalised objective, exactly like EM;
    otherwise the cycle falls back to theta2 (the plain EM result).
  * Every accepted iterate is an output of F, i.e. of the M-step with all of
    its guards. If a guard fired on the stabilising step in a way that kept
    an extrapolated block (Gamma rejected, Delta cap bound, component frozen:
    a block of theta'' equal to the same block of theta' although plain EM
    moved it), the cycle is rejected, because that block never went through
    the M-step. Guard events of rejected steps are rolled back so they cannot
    trigger execute_fit_job's frozen-component exclusion.
  * theta' must give finite densities and a finite objective; otherwise alpha
    is shrunk toward 1 (up to 3 times) and then the cycle falls back to theta2.
  * step_max starts at 1 and is multiplied by 4 whenever the clamp binds
    (including at alpha = 1, before any extrapolation), divided by 4 (not
    below 1) when an extrapolation at the clamp is rejected -- the reference
    package's defaults.

Convergence: the fit stops when one whole cycle (theta0 to the next accepted
iterate, i.e. >= 2 EM steps) gains < 1e-8 relative -- the plain rule's
threshold on a longer step, so never looser than plain EM's per-step test
(testing single EM steps stopped SQUAREM early in slow regions). Decreases
use the plain rules on consecutive EM steps from accepted iterates: a relative
decrease > 1e-8 stops the fit (the caller returns the best iterate), >= 1e-1
fails it.
``max_em_iters`` counts F evaluations, including rejected stabilising steps.

Warm-up: plain EM steps until one gains < 1e-6 relative
(DEFAULT_WARMUP_REL_TOL, kwarg squarem_warmup_rel_tol), then SQUAREM cycles.

Supported: multivariate fits with q <= 2 (deterministic E-step), with strata,
sample weights (tau), missing readouts and categorical readouts; univariate
fits (EXPERIMENTAL, 11/26/26), including the univariate density-ratio
constraint. Refused: q > 2 (Monte Carlo E-step), multivariate constrained /
separation fits, dim_weights, weight_divergence and tied_population.

Univariate fits (UnivariatePacker): component params are the canonical
(a, loc, scale) tuples; the difference vectors use the alternate form
(loc, Delta, log Gamma), Gamma = scale^2 - Delta^2 > 0. A constrained fit's
EM map line-searches each block toward its M-step candidate while the
density-ratio constraint holds, starting from a feasible point; theta' must
therefore satisfy the constraint too (the ``valid`` callback), or alpha is
shrunk as for an invalid point. Force-Gaussian components (Delta = 0) keep
Delta = 0 (r = v = 0 there).
"""

import numpy as np

SUPPORTED = ("squarem",)

# Plain EM steps first, until one gains < this relative amount; extrapolation
# then starts inside the basin plain EM is heading for. Without it, early long
# steps (alpha up to ~700) moved fits into other, sometimes worse, slow regions:
# on 8 synthetic fits (test/fit_speed sweep, 2 datasets x 2 seeds x q1/q2)
# no warm-up ended below plain EM in 3/8, a 1e-5 warm-up in 1/8, 1e-6 in 0/8
# (with 1.1-12x fewer EM evaluations). Override: squarem_warmup_rel_tol.
DEFAULT_WARMUP_REL_TOL = 1e-6


def resolve(kwargs):
    """The requested accelerator name, or None for plain EM."""
    acc = kwargs.get("em_accelerator")
    if acc in (None, "none", "em", False):
        return None
    if acc not in SUPPORTED:
        raise ValueError(f"em_accelerator must be one of None, 'none', {SUPPORTED}; got {acc!r}")
    return acc


def check_supported(multivariate, latent_q, separation_active, dim_weighted_active,
                    weight_divergence, tied_population):
    why = []
    if latent_q is not None and latent_q > 2:
        why.append("latent_q > 2 (Monte Carlo E-step)")
    if separation_active:
        why.append("constrained / separation fits")
    if dim_weighted_active:
        why.append("dim_weights")
    if weight_divergence is not None:
        why.append("weight_divergence")
    if tied_population is not None:
        why.append("tied_population")
    if why:
        raise NotImplementedError("em_accelerator='squarem' does not support " + ", ".join(why))


# ── parameterisation ────────────────────────────────────────────────────────

def _chol_pack(G):
    L = np.linalg.cholesky(0.5 * (G + G.T))
    p = L.shape[0]
    il = np.tril_indices(p, -1)
    return np.concatenate([np.log(np.diag(L)), L[il]])


def _chol_unpack(x, p):
    L = np.zeros((p, p))
    L[np.diag_indices(p)] = np.exp(x[:p])
    L[np.tril_indices(p, -1)] = x[p:]
    G = L @ L.T
    return 0.5 * (G + G.T)


def _log_simplex(P, fixed):
    out = np.zeros(P.shape)
    free = ~fixed
    out[free] = np.log(P[free])
    return out


def _unlog_simplex(X, fixed, fill):
    """Rows of exp(X) renormalised; ``fixed`` entries take ``fill``."""
    P = np.where(fixed, fill, np.exp(X - np.max(np.where(fixed, -np.inf, X), axis=-1, keepdims=True)))
    free_mass = np.where(fixed, 0.0, P).sum(axis=-1, keepdims=True)
    fixed_mass = np.where(fixed, fill, 0.0).sum(axis=-1, keepdims=True)
    P = np.where(fixed, fill, P / np.maximum(free_mass, 1e-300) * np.maximum(1.0 - fixed_mass, 0.0))
    return P


class Packer:
    """Maps theta = (params, W, cat_probs) to a flat vector and back, for the
    three iterates of one cycle. Entries of W / cat_probs that are exactly
    zero in any of them are held at theta2's value (log undefined)."""

    def __init__(self, thetas):
        params0, W0, probs0 = thetas[0]
        self.K = len(params0)
        self.shapes = [(np.shape(mu), np.shape(D), np.shape(G)) for mu, D, G in params0]
        self.W_fixed = np.zeros(np.shape(W0), dtype=bool)
        for _, W, _ in thetas:
            self.W_fixed |= (np.asarray(W) <= 0)
        self.W_fill = np.where(self.W_fixed, np.asarray(thetas[-1][1]), 0.0)
        self.has_cat = probs0 is not None
        if self.has_cat:
            self.P_fixed = [np.zeros(np.shape(P), dtype=bool) for P in probs0]
            for _, _, probs in thetas:
                for j, P in enumerate(probs):
                    self.P_fixed[j] |= (np.asarray(P) <= 0)
            self.P_fill = [np.where(f, np.asarray(P), 0.0) for f, P in zip(self.P_fixed, thetas[-1][2])]

    def pack(self, theta):
        params, W, probs = theta
        parts = []
        for (mu, D, G), (smu, sD, sG) in zip(params, self.shapes):
            if np.shape(mu) != smu or np.shape(D) != sD or np.shape(G) != sG:
                raise ValueError("parameter shapes changed within a cycle")
            parts += [np.ravel(mu), np.ravel(D), _chol_pack(np.asarray(G, dtype=float))]
        parts.append(_log_simplex(np.asarray(W, dtype=float), self.W_fixed).ravel())
        if self.has_cat:
            for P, f in zip(probs, self.P_fixed):
                parts.append(_log_simplex(np.asarray(P, dtype=float), f).ravel())
        return np.concatenate(parts)

    def unpack(self, x):
        params, i = [], 0
        for smu, sD, sG in self.shapes:
            n = int(np.prod(smu)); mu = x[i:i + n].reshape(smu); i += n
            n = int(np.prod(sD)); D = x[i:i + n].reshape(sD); i += n
            p = sG[0]; n = p + p * (p - 1) // 2
            G = _chol_unpack(x[i:i + n], p); i += n
            params.append((mu.copy(), D.copy(), G))
        n = self.W_fixed.size
        W = _unlog_simplex(x[i:i + n].reshape(self.W_fixed.shape), self.W_fixed, self.W_fill); i += n
        probs = None
        if self.has_cat:
            probs = []
            for f, fill in zip(self.P_fixed, self.P_fill):
                n = f.size
                probs.append(_unlog_simplex(x[i:i + n].reshape(f.shape), f, fill)); i += n
        return params, W, probs


class UnivariatePacker:
    """Packer for univariate fits: params are canonical (a, loc, scale)
    tuples, packed as (loc, Delta, log Gamma) of the alternate form."""

    def __init__(self, thetas):
        from .density_utils import canonical_to_alternate
        self._c2a = canonical_to_alternate
        params0, W0, _ = thetas[0]
        self.K = len(params0)
        self.W_fixed = np.zeros(np.shape(W0), dtype=bool)
        for _, W, _ in thetas:
            self.W_fixed |= (np.asarray(W) <= 0)
        self.W_fill = np.where(self.W_fixed, np.asarray(thetas[-1][1]), 0.0)

    def pack(self, theta):
        params, W, _ = theta
        parts = []
        for a, loc, scale in params:
            loc_, D, G = self._c2a(a, loc, scale)
            if not G > 0:
                raise ValueError("non-positive Gamma")
            parts.append(np.array([loc_, D, np.log(G)]))
        parts.append(_log_simplex(np.asarray(W, dtype=float), self.W_fixed).ravel())
        return np.concatenate(parts)

    def unpack(self, x):
        params = []
        for k in range(self.K):
            loc, D, lG = x[3 * k:3 * k + 3]
            G = float(np.exp(lG))
            a = float(np.sign(D) * np.sqrt(D * D / G))
            params.append((a, float(loc), float(np.sqrt(G + D * D))))
        i = 3 * self.K
        n = self.W_fixed.size
        W = _unlog_simplex(x[i:i + n].reshape(self.W_fixed.shape), self.W_fixed, self.W_fill)
        return params, W, None


def _moved_blocks(theta0, theta2):
    """Per component, which of (mu, Delta, Gamma) plain EM changed."""
    out = []
    for (m0, D0, G0), (m2, D2, G2) in zip(theta0[0], theta2[0]):
        out.append((not np.array_equal(m0, m2), not np.array_equal(D0, D2),
                    not np.array_equal(G0, G2)))
    return out


def _guard_kept_extrapolated(theta_p, theta_pp, moved):
    """True if a block of theta'' equals the extrapolated block of theta'
    although plain EM moves that block, i.e. an M-step guard (Gamma reject,
    Delta cap, freeze) kept a value that never went through the M-step."""
    for (mp, Dp, Gp), (mpp, Dpp, Gpp), mv in zip(theta_p[0], theta_pp[0], moved):
        for a, b, m in ((mp, mpp, mv[0]), (Dp, Dpp, mv[1]), (Gp, Gpp, mv[2])):
            if m and np.array_equal(np.asarray(a, dtype=float).ravel(),
                                    np.asarray(b, dtype=float).ravel()):
                return True
    return False


# ── driver ──────────────────────────────────────────────────────────────────

class StopFit(Exception):
    """Raised by the driver's callbacks to end the EM loop with a reason."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def run(theta0, cache0, obj0, F, objective, density, record, count_only, budget_left,
        guard_stats=None, tol=1e-8, plateau_rel_tol=1e-8, overshoot_rel_tol=1e-5,
        catastrophic_rel_tol=1e-1, step_max0=1.0, mstep=4.0, max_backtrack=3,
        early_stopping=True, warmup_rel_tol=0.0, step_max_cap=np.inf,
        packer=Packer, valid=None, guard_check=True):
    """SQUAREM loop.

    F(theta, cache) -> (theta', ll, cache')  one EM iteration (counts as one).
    objective(theta, ll) -> penalised objective (single_fit's series).
    density(theta) -> (cache, ll)            log-pdf cache and LL at theta.
    record(theta, ll, obj)                   append an accepted F output.
    count_only()                             count an F evaluation not kept.
    budget_left() -> int                     F evaluations still allowed.

    warmup_rel_tol: plain EM steps until one gains < this relative amount
    (0 = no warm-up), so extrapolation starts inside the basin plain EM is
    heading for. step_max_cap: upper bound on step_max.
    packer: Packer (multivariate) or UnivariatePacker.
    valid(theta) -> bool: extra feasibility test for theta' (the univariate
    density-ratio constraint); False shrinks alpha like an invalid point.
    guard_check: reject a cycle whose stabilising step kept an extrapolated
    block (multivariate M-step guards). Off for univariate fits: there the
    only block-keeping step is the constraint line search, which from a
    feasible theta' is the constrained EM map itself.

    Returns (theta, stop_reason, stats); theta is the last accepted iterate.
    """
    stats = dict(cycles=0, accepted=0, fallback_objective=0, fallback_guard=0,
                 fallback_invalid=0, fallback_error=0, plain_alpha1=0, alpha_sum=0.0,
                 alpha_max=1.0)
    step_max = step_max0
    overshoot = None

    def _converged(o_prev, o_new):
        if not early_stopping:
            return
        with np.errstate(invalid="ignore"):
            if abs(o_new - o_prev) / abs(o_prev) < tol:
                raise StopFit("converged")

    def _pair(o_prev, o_new):
        """Plain-EM decrease rule on two consecutive F outputs that both
        started from accepted iterates. (Convergence is tested per cycle.)"""
        if o_new < o_prev:
            rel_dec = (o_prev - o_new) / max(abs(o_prev), 1e-300)
            if rel_dec >= catastrophic_rel_tol:
                raise ValueError(f"Likelihood decreased by {o_prev - o_new:.2e}")
            if rel_dec > plateau_rel_tol:
                raise StopFit("overshoot" if rel_dec >= overshoot_rel_tol else "revert")

    def _plain(theta, cache, obj):
        if budget_left() <= 0:
            raise StopFit("cap")
        theta_n, ll, cache_n = F(theta, cache)
        if np.isnan(ll) or np.isnan(np.asarray(theta_n[1])).any():
            raise ValueError("NaN in likelihoods or weights")
        obj_n = objective(theta_n, ll)
        record(theta_n, ll, obj_n)
        return theta_n, cache_n, obj_n

    theta = theta0
    try:
        stats["warmup_iters"] = 0
        if warmup_rel_tol > 0:
            while True:
                theta1, cache1, obj1 = _plain(theta0, cache0, obj0)
                theta = theta1
                stats["warmup_iters"] += 1
                _pair(obj0, obj1)
                _converged(obj0, obj1)
                done = abs(obj1 - obj0) / abs(obj0) < warmup_rel_tol
                theta0, cache0, obj0 = theta1, cache1, obj1
                if done:
                    break
        while True:
            stats["cycles"] += 1
            theta1, cache1, obj1 = _plain(theta0, cache0, obj0)
            theta = theta1
            _pair(obj0, obj1)
            theta2, cache2, obj2 = _plain(theta1, cache1, obj1)
            theta = theta2
            _pair(obj1, obj2)

            # ---- extrapolation ------------------------------------------
            accepted = False
            try:
                pk = packer([theta0, theta1, theta2])
                x0, x1, x2 = pk.pack(theta0), pk.pack(theta1), pk.pack(theta2)
                r = x1 - x0
                v = x2 - x1 - r
                nr, nv = float(np.linalg.norm(r)), float(np.linalg.norm(v))
                alpha = nr / nv if (nv > 0 and np.isfinite(nr) and np.isfinite(nv)) else 1.0
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                alpha = 1.0
            alpha = float(min(max(alpha, 1.0), step_max))
            cap_bound = alpha >= step_max

            if alpha > 1.0 and budget_left() > 0:
                theta_p = cache_p = None
                for _ in range(max_backtrack + 1):
                    try:
                        with np.errstate(all="ignore"):
                            cand = pk.unpack(x0 + 2.0 * alpha * r + alpha * alpha * v)
                            # L L' is PD in exact arithmetic but can be
                            # numerically indefinite after a long step.
                            if packer is Packer:
                                for _prm in cand[0]:
                                    np.linalg.cholesky(_prm[2])
                            if valid is not None and not valid(cand):
                                raise ValueError("extrapolated point infeasible")
                            cache_c, ll_c = density(cand)
                        if all(np.all(np.isfinite(b)) for prm in cand[0] for b in prm) \
                                and np.isfinite(ll_c):
                            theta_p, cache_p, obj_p = cand, cache_c, objective(cand, ll_c)
                            if np.isfinite(obj_p):
                                break
                            theta_p = None
                    except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError):
                        theta_p = None
                    alpha = 0.5 * (alpha + 1.0)
                    if alpha - 1.0 < 1e-3:
                        break
                if theta_p is None:
                    stats["fallback_invalid"] += 1
                else:
                    saved = None if guard_stats is None else dict(guard_stats)
                    try:
                        theta_pp, ll_pp, cache_pp = F(theta_p, cache_p)
                        obj_pp = objective(theta_pp, ll_pp)
                        err = (np.isnan(ll_pp) or np.isnan(np.asarray(theta_pp[1])).any()
                               or not np.isfinite(obj_pp))
                    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError):
                        err, theta_pp = True, None
                    if err:
                        stats["fallback_error"] += 1
                    elif guard_check and _guard_kept_extrapolated(theta_p, theta_pp,
                                                                  _moved_blocks(theta0, theta2)):
                        stats["fallback_guard"] += 1
                    elif obj_pp < obj2:
                        stats["fallback_objective"] += 1
                    else:
                        accepted = True
                    if not accepted:
                        count_only()
                        if guard_stats is not None:
                            guard_stats.clear()
                            guard_stats.update(saved)
            else:
                stats["plain_alpha1"] += 1

            # step_max adaptation (reference SQUAREM): grow while the clamp
            # binds (including at alpha = 1 before any extrapolation has been
            # tried), shrink back after a rejected extrapolation.
            rejected = (not accepted) and alpha > 1.0
            if cap_bound and not rejected:
                step_max = min(step_max * mstep, step_max_cap)
            elif cap_bound and rejected:
                step_max = max(step_max0, step_max / mstep)

            if accepted:
                stats["accepted"] += 1
                stats["alpha_sum"] += alpha
                stats["alpha_max"] = max(stats["alpha_max"], alpha)
                record(theta_pp, ll_pp, obj_pp)
                theta = theta_pp
                obj_start = obj0
                theta0, cache0, obj0 = theta_pp, cache_pp, obj_pp
            else:
                obj_start = obj0
                theta0, cache0, obj0 = theta2, cache2, obj2
            # Convergence on the whole cycle's gain. Testing single EM steps
            # (the plain rule) stops SQUAREM too early: an extrapolated point
            # can sit where one EM step gains < 1e-8 although a cycle still
            # gains far more (seen on a synthetic q=2 fit: stopped at 507
            # evaluations 5e-4 below where plain EM converged). A cycle spans
            # >= 2 EM steps, so this is never looser than the plain rule.
            _converged(obj_start, obj0)
    except StopFit as s:
        return theta, s.reason, stats
