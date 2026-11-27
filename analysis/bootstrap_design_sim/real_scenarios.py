"""Bootstrap-design simulation, Arm B: real-dataset scenarios.

Each simulated dataset is a real MV dataset (its production rows, readout
patterns, strata and sample memberships) with new scores drawn from a known
truth. The truth favours neither MV model: per class, each readout's scores
follow a Gaussian-kernel density fitted to that dataset's labelled controls,
and readouts are joined by a Gaussian copula with the class's own
between-readout correlation. The true density of any observed sub-vector x_S
is then closed form,
    f(x_S) = phi_{R_SS}(z_S) / prod_j phi(z_j) * prod_j f_j(x_j),
    z_j = Phi^-1(F_j(x_j)),
so every row's true log LR is exact.

Classes (from the real labels; rows labelled both P/LP and B/LB are left out
of the fit):
  P missense   P/LP, missense, no splice flag
  P null       other P/LP (PTV, splice-flagged, other); scored like LOF
  B            B/LB (all consequences); synonymous rows added if B/LB < 10
A P class with < MIN_CLASS rows borrows the pooled P/LP rows. Datasets whose
pooled P/LP or benign reference still has < MIN_CLASS rows are skipped (no
sign-safe way to borrow another dataset's scores).

Simulated rows:
  P/LP rows       their consequence class
  B/LB rows       benign
  rows in both    pathogenic with probability 0.5 (auxiliary; never evaluated)
  synonymous      benign (setting "synmis": 5% drawn as P null)
  gnomAD rows     hidden class, pathogenic with probability pi overall; with
                  coupling beta = 1 the probability follows the row's readout
                  pattern (P/LP share among labelled rows of that pattern,
                  shrunk to the average), as real coverage tracks domains
The oracle LR of a row is its consequence's P class vs B.

Workflow (paths are arguments; nothing here submits anything):
  1. dump: MV_SIM_DUMP=<dir> python hpc/prepare.py mv_all ... writes each
     dataset's real rows (hook in prepare._generate_bootstrap_fit_jobs).
  2. python -m analysis.bootstrap_design_sim.real_scenarios simulate <dump dir> <out dir> --reps 10
  3. bundles: MV_SIM_SCORES=<out dir>/<setting>/rep_<r> for prepare (scores
     swapped in before jobs are built) and for scoring (the same hook in the
     ms rebuild), so fitting and scoring are the production code.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

MIN_CLASS = 10
SETTINGS = {"main": dict(pi=0.25, beta=1.0, syn_path=0.0),
            "pi10": dict(pi=0.10, beta=1.0, syn_path=0.0),
            "indep": dict(pi=0.25, beta=0.0, syn_path=0.0),
            "synmis": dict(pi=0.25, beta=1.0, syn_path=0.05)}
ROLE = dict(P=0, B=1, g=2, s=3)


# ── hook used by hpc/prepare.py and the scoring rebuild ──────────────────

def _sample_matrix(ms):
    """(N, 4) bool over P/LP, B/LB, gnomAD, synonymous from the ms's named samples."""
    names = list(ms.sample_names)
    SA = np.asarray(ms.sample_assignments, bool)
    out = np.zeros((SA.shape[0], 4), bool)
    for j, n in enumerate(names):
        nl = str(n).lower()
        k = (0 if "patho" in nl and "benign" not in nl else 1 if "benign" in nl else
             3 if "synon" in nl else 2 if ("population" in nl or "gnomad" in nl) else None)
        if k is not None:
            out[:, k] |= SA[:, j]
    return out


def apply_hook(ms, dataset_label, annotations=None):
    """MV_SIM_DUMP=<dir>: write the real rows of `ms` (annotations: callable
    returning strata.row_annotations' DataFrame). MV_SIM_SCORES=<dir>: replace
    the scores with <dir>/<dataset_label>.npz's X (same rows; cells may only
    turn missing). Applied once per ms (cluster splits inherit it)."""
    if getattr(ms, "_sim_applied", False):
        return ms
    dump, sim = os.environ.get("MV_SIM_DUMP"), os.environ.get("MV_SIM_SCORES")
    if dump:
        ann = annotations() if annotations is not None else None
        X = np.asarray(ms.scores, float)
        np.savez_compressed(
            Path(dump) / f"{dataset_label}.npz", X=X, S=_sample_matrix(ms),
            consequence=np.asarray(ann["consequence"].astype(str) if ann is not None else ["missense"] * len(X), dtype=str),
            splice=np.asarray(ann["splice"].astype(bool) if ann is not None else np.zeros(len(X), bool)),
            dims=np.asarray([str(d) for d in ms.dataset_names]))
    if sim:
        f = Path(sim) / f"{dataset_label}.npz"
        if not f.exists():
            raise FileNotFoundError(f"MV_SIM_SCORES set but no simulated scores for {dataset_label}: {f}")
        Xs = np.load(f)["X"]
        X0 = np.asarray(ms.scores, float)
        if Xs.shape != X0.shape or (np.isnan(X0) & ~np.isnan(Xs)).any():
            raise ValueError(f"{dataset_label}: simulated scores do not match the real rows ({Xs.shape} vs {X0.shape})")
        ms._scores = Xs
        if getattr(ms, "_scores_matrix", None) is not None and np.shape(ms._scores_matrix) == Xs.shape:
            ms._scores_matrix = Xs
    ms._sim_applied = True
    return ms


def apply_hook_by_match(ms):
    """Scoring side (mv_analysis.config): the rebuilt ms has no dataset label,
    so find its simulated file in MV_SIM_SCORES by readout names and shape."""
    if getattr(ms, "_sim_applied", False):
        return ms
    sim = Path(os.environ["MV_SIM_SCORES"])
    dims = tuple(str(d) for d in ms.dataset_names)
    shape = np.shape(ms.scores)
    hits = []
    for f in sorted(sim.glob("*.npz")):
        z = np.load(f)
        if tuple(z["dims"]) == dims and z["X"].shape == shape:
            hits.append(f.stem)
    if len(hits) != 1:
        raise ValueError(f"MV_SIM_SCORES: {len(hits)} simulated files match dims {dims} and shape {shape}: {hits}")
    os.environ.pop("MV_SIM_DUMP", None)
    return apply_hook(ms, hits[0])


# ── truth: KDE marginals + Gaussian copula per class ─────────────────────

class Marginal:
    """Gaussian KDE (Silverman bandwidth) with exact pdf / cdf and a grid inverse."""

    def __init__(self, x):
        x = np.asarray(x, float)
        self.x = x
        sd = np.std(x, ddof=1) if len(x) > 1 else 1.0
        iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349 if len(x) > 1 else sd
        s = min(sd, iqr) if iqr > 0 else sd
        self.h = max(0.9 * (s if s > 0 else 1.0) * len(x) ** -0.2, 1e-6)
        g = np.linspace(x.min() - 5 * self.h, x.max() + 5 * self.h, 4096)
        self._g, self._G = g, self.cdf(g)

    def logpdf(self, v):
        v = np.asarray(v, float)
        out = np.empty(len(v))
        for i in range(0, len(v), 4096):
            d = (v[i:i + 4096, None] - self.x[None, :]) / self.h
            out[i:i + 4096] = np.logaddexp.reduce(norm.logpdf(d), axis=1) - np.log(len(self.x) * self.h)
        return out

    def cdf(self, v):
        v = np.asarray(v, float)
        out = np.empty(len(v))
        for i in range(0, len(v), 4096):
            out[i:i + 4096] = norm.cdf((v[i:i + 4096, None] - self.x[None, :]) / self.h).mean(axis=1)
        return out

    def ppf(self, u):
        return np.interp(u, self._G, self._g)


def _nearest_corr(R):
    w, V = np.linalg.eigh((R + R.T) / 2)
    R = (V * np.clip(w, 1e-3, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


class ClassDensity:
    def __init__(self, X, fallback_X, ref_X):
        """X: this class's rows (NaN = missing); fallback_X: rows used for a
        readout this class barely covers; ref_X: all rows (correlation fallback)."""
        D = X.shape[1]
        self.m = []
        for j in range(D):
            for src in (X, fallback_X, ref_X):
                v = src[:, j][~np.isnan(src[:, j])]
                if len(v) >= 5:
                    break
            self.m.append(Marginal(v))
        Z = self._z(X)
        Zr = self._z(ref_X)
        R = np.eye(D)
        for a in range(D):
            for b in range(a + 1, D):
                for ZZ in (Z, Zr):
                    ok = ~np.isnan(ZZ[:, a]) & ~np.isnan(ZZ[:, b])
                    if ok.sum() >= 10:
                        R[a, b] = R[b, a] = np.corrcoef(ZZ[ok, a], ZZ[ok, b])[0, 1]
                        break
        self.R = _nearest_corr(np.nan_to_num(R))

    def _z(self, X):
        Z = np.full(X.shape, np.nan)
        for j, m in enumerate(self.m):
            o = ~np.isnan(X[:, j])
            Z[o, j] = norm.ppf(np.clip(m.cdf(X[o, j]), 1e-12, 1 - 1e-12))
        return Z

    def sample(self, rng, n, R=None):
        R = self.R if R is None else R
        z = rng.multivariate_normal(np.zeros(len(self.m)), R, size=n)
        u = norm.cdf(z)
        return np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(self.m)])

    def logpdf(self, X, R=None):
        """Log density of each row's observed sub-vector."""
        R = self.R if R is None else R
        obs = ~np.isnan(X)
        out = np.full(len(X), np.nan)
        Z = self._z(X)
        lm = np.zeros(len(X))
        for j, m in enumerate(self.m):
            o = obs[:, j]
            lm[o] += m.logpdf(X[o, j])
        keys = {}
        for i, o in enumerate(obs):
            keys.setdefault(o.tobytes(), []).append(i)
        for k, idx in keys.items():
            d = np.flatnonzero(obs[idx[0]])
            if len(d) == 0:
                continue
            z = Z[np.ix_(idx, d)]
            Rs = R[np.ix_(d, d)]
            _, logdet = np.linalg.slogdet(Rs)
            q = np.einsum("ij,jk,ik->i", z, np.linalg.inv(Rs), z)
            lc = -0.5 * (logdet + q - (z ** 2).sum(axis=1))
            out[idx] = lc + lm[idx]
        return out


def load_dump(f):
    d = np.load(f, allow_pickle=True)  # our own dumps; early ones stored consequence as an object array
    return dict(X=d["X"], S=d["S"], consequence=np.asarray(d["consequence"], dtype=str),
                splice=np.asarray(d["splice"], bool), dims=np.asarray(d["dims"], dtype=str))


def classes(dd):
    S, csq, spl = dd["S"], dd["consequence"], dd["splice"]
    P, B = S[:, 0] & ~S[:, 1], S[:, 1] & ~S[:, 0]
    mis = (csq == "missense") & ~spl
    return dict(P_missense=P & mis, P_null=P & ~mis, B=B, both=S[:, 0] & S[:, 1])


def fit_truth(dd):
    X = dd["X"]
    c = classes(dd)
    Pall = c["P_missense"] | c["P_null"]
    ben = c["B"]
    if ben.sum() < MIN_CLASS:
        ben = ben | (dd["S"][:, 3] & ~dd["S"][:, 0])
    if Pall.sum() < MIN_CLASS or ben.sum() < MIN_CLASS:
        return None, dict(n_P=int(Pall.sum()), n_B=int(ben.sum()))
    ref = X[dd["S"][:, 2] | dd["S"][:, 3]] if (dd["S"][:, 2] | dd["S"][:, 3]).sum() >= 20 else X
    out = {"B": ClassDensity(X[ben], X[ben], ref)}
    for k in ("P_missense", "P_null"):
        rows = c[k] if c[k].sum() >= MIN_CLASS else Pall
        out[k] = ClassDensity(X[rows], X[Pall], ref)
    info = {k: int(v.sum()) for k, v in c.items()}
    info["P_missense_borrowed"] = bool(c["P_missense"].sum() < MIN_CLASS)
    info["P_null_borrowed"] = bool(c["P_null"].sum() < MIN_CLASS)
    info["benign_with_synonymous"] = bool(c["B"].sum() < MIN_CLASS)
    return out, info


def rng_for(dataset, setting, rep):
    return np.random.default_rng([zlib.crc32(dataset.encode()), zlib.crc32(setting.encode()), int(rep)])


def _gnomad_probs(dd, pi, beta, m=10.0):
    """Pathogenic probability per gnomAD row: pi on average, tilted by the
    P/LP share among labelled rows with the same readout pattern."""
    obs = ~np.isnan(dd["X"])
    pat = np.array([o.tobytes() for o in obs])
    c = classes(dd)
    P, B = c["P_missense"] | c["P_null"], c["B"]
    sbar = P.sum() / max(P.sum() + B.sum(), 1)
    sp = {}
    for k in set(pat):
        r = pat == k
        sp[k] = (P[r].sum() + m * sbar) / (P[r].sum() + B[r].sum() + m)
    t = np.array([logit(np.clip(sp[k], 1e-4, 1 - 1e-4)) - logit(sbar) for k in pat]) * beta
    g = dd["S"][:, 2] & ~dd["S"][:, 0] & ~dd["S"][:, 1]
    lo, hi = -20.0, 20.0
    for _ in range(80):
        a = (lo + hi) / 2
        if expit(a + t[g]).mean() < pi:
            lo = a
        else:
            hi = a
    return expit(a + t), g


def simulate_one(truth, dd, dataset, setting, rep):
    cfg = SETTINGS[setting]
    rng = rng_for(dataset, setting, rep)
    X0 = dd["X"]
    n = len(X0)
    c = classes(dd)
    mis = (dd["consequence"] == "missense") & ~dd["splice"]
    pg, g = _gnomad_probs(dd, cfg["pi"], cfg["beta"])
    path = np.zeros(n, bool)
    path[c["P_missense"] | c["P_null"]] = True
    path[c["both"]] = rng.random(c["both"].sum()) < 0.5
    path[g] = rng.random(g.sum()) < pg[g]
    syn = dd["S"][:, 3] & ~dd["S"][:, 0] & ~dd["S"][:, 1] & ~g
    path[syn] = rng.random(syn.sum()) < cfg["syn_path"]
    X = np.full(X0.shape, np.nan)
    for key, rows in (("B", ~path), ("P_missense", path & mis), ("P_null", path & ~mis)):
        if rows.any():
            X[rows] = truth[key].sample(rng, int(rows.sum()))
    X[np.isnan(X0)] = np.nan
    lp = np.where(mis, truth["P_missense"].logpdf(X), truth["P_null"].logpdf(X))
    true_llr = lp - truth["B"].logpdf(X)
    evaluated = (c["P_missense"] | c["P_null"] | c["B"])
    return dict(X=X, true_class=path.astype(np.int8), true_log_lr=true_llr, gnomad=g,
                evaluated=evaluated, prior=np.float64(cfg["pi"]), dims=dd["dims"])


def simulate(dump_dir, out_dir, reps, settings=("main",)):
    dump_dir, out_dir = Path(dump_dir), Path(out_dir)
    log = {}
    for f in sorted(dump_dir.glob("*.npz")):
        ds = f.stem
        dd = load_dump(f)
        truth, info = fit_truth(dd)
        log[ds] = info
        if truth is None:
            print(f"{ds}: skipped (P/LP {info['n_P']}, benign {info['n_B']} < {MIN_CLASS})")
            continue
        for s in settings:
            for r in range(reps):
                od = out_dir / s / f"rep_{r}"
                od.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(od / f"{ds}.npz", **simulate_one(truth, dd, ds, s, r))
        print(f"{ds}: {len(settings)} settings x {reps} replicates; {info}")
    with open(out_dir / "truth_log.json", "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["simulate"])
    ap.add_argument("dump_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--settings", nargs="+", default=["main"], choices=list(SETTINGS))
    a = ap.parse_args()
    simulate(a.dump_dir, a.out_dir, a.reps, a.settings)
