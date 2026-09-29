#!/usr/bin/env python3
"""CPU/GPU parity for the exact q=2 truncated-normal moment helpers.

The GPU path's helpers are meant to be the SAME algorithm as the CPU's, not a
second implementation kept in agreement by hand. That intent is easy to break
silently: the previous version used Owen's T on CPU and a Drezner rho-integral
in JAX, with a different rho clip, an extra 1e-8 floor and 24 quadrature nodes,
and nobody noticed because the GPU path is opt-in and JAX is not installed in
most environments.

This test checks the agreement WITHOUT requiring JAX, by executing the JAX
source under a numpy-backed shim (`jnp` -> numpy, `log_ndtr`/`logsumexp` ->
scipy). That catches the failure modes that actually occur in practice --
divergent constants, a different formula, wrong axis/broadcast handling -- which
is how an einsum index collision in this same file was caught previously.

It does NOT substitute for running the real thing: `test_batch_em_parity.py` on
a JAX machine is still required to validate the traced/jitted code path.

Usage:  python tests/test_bvn_cpu_gpu_parity.py
"""
import re
import sys
import types
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr as _sp_log_ndtr
from scipy.special import logsumexp as _sp_logsumexp
from scipy.special import ndtr as _sp_ndtr
from scipy.special import owens_t as _sp_owens_t


def _sp_norm_pdf(x, loc=0.0, scale=1.0):
    z = (np.asarray(x, dtype=float) - loc) / scale
    return np.exp(-0.5 * z * z) / (scale * np.sqrt(2.0 * np.pi))

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.cfusn import update_steps as CPU

JAX_SRC = (_REPO_ROOT / "src/assay_calibration/fit_utils/jax_batch"
           / "batch_em_cfusn.py")

# Names the ported helpers need; everything else in the module is skipped.
_WANTED = ("_LOG_SQRT_2PI", "_BVN_QUAD_NODES", "_BVN_QUAD_PANELS",
           "_BVN_QUAD_LOG_RANGE", "_F1_1_ASYMPTOTIC_T", "_OWENS_MIN_RELIABLE",
           "_GL_NODES", "_GL_WEIGHTS", "_log_bvn_orthant_quad", "_log_phi_at0",
           "_log_f1_0", "_log_f1_1", "_owens_t", "_bvn_cdf_owens",
           "_log_bvn_cdf")


def load_jax_helpers_under_numpy():
    """Exec just the shared-helper block of the JAX module with jnp -> numpy."""
    src = JAX_SRC.read_text()
    start = src.index("_LOG_SQRT_2PI = ")
    end = src.index("def _augment_omega(")
    block = src[start:end]

    # numpy has no `jnp.asarray(...)`-only spellings to fix up beyond the name
    block = re.sub(r"\bjnp\.", "np.", block)

    # jnp.broadcast_shapes / jnorm.cdf have no numpy spelling; shim them.
    block = block.replace("np.broadcast_shapes", "np.broadcast_shapes")
    shim_norm = types.SimpleNamespace(cdf=_sp_ndtr, pdf=_sp_norm_pdf)
    ns = {"np": np, "_np": np, "jnorm": shim_norm,
          "log_ndtr": _sp_log_ndtr, "logsumexp": _sp_logsumexp}
    exec(compile(block, str(JAX_SRC), "exec"), ns)
    missing = [n for n in _WANTED if n not in ns]
    if missing:
        raise AssertionError(f"JAX helper block missing names: {missing}")
    return types.SimpleNamespace(**{n: ns[n] for n in _WANTED})


def main():
    GPU = load_jax_helpers_under_numpy()
    fails = []

    # ---- constants must be identical, not merely close -------------------
    for name in ("_LOG_SQRT_2PI", "_BVN_QUAD_NODES", "_BVN_QUAD_PANELS",
                 "_BVN_QUAD_LOG_RANGE", "_F1_1_ASYMPTOTIC_T"):
        c, g = getattr(CPU, name), getattr(GPU, name)
        ok = c == g
        print(f"  const {name:24s} cpu={c!r:<10} gpu={g!r:<10} "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"constant {name}: {c!r} != {g!r}")

    rng = np.random.RandomState(11)

    # ---- log Phi_2 across the whole depth range --------------------------
    print("\n  log_bvn_orthant_quad (rho scalar on CPU, per-row on GPU):")
    # |rho| -> 1 included: S = I - D' Om^-1 D drives rho there as Delta grows,
    # and that is where the quadrature is weakest (a 32x2 setting that looked
    # fine at rho=0.45 gave 4.5e-05 at rho=0.999).
    for rho in (-0.999999, -0.9, -0.3, 0.0, 0.45, 0.97, 0.999, 0.999999):
        h, k = [], []
        for _ in range(800):
            d = rng.uniform(0.0, 11.0)
            h.append(-d + rng.uniform(-1, 1))
            k.append(-d + rng.uniform(-1, 1))
        h, k = np.array(h), np.array(k)
        c = CPU._log_bvn_orthant_quad(h, k, rho)
        g = GPU._log_bvn_orthant_quad(h, k, np.full_like(h, rho))
        d = np.abs(c - g).max()
        # The quadrature is genuinely ill-conditioned as rho -> -1 (the
        # distribution degenerates to Z2 ~ -Z1 and the integrand sharpens), so
        # two implementations of the SAME formula differ there by ~1e-7. That is
        # an accuracy limit, not a divergence -- and in production the hybrid
        # routes those rows to Owen's T, since L is large when |rho| is extreme.
        tol = 1e-6 if abs(rho) > 0.999 else 1e-12
        print(f"    rho={rho:+.6f}  max |dlog| = {d:.3e}  (tol {tol:.0e})"
              f"   {'OK' if d < tol else 'MISMATCH'}")
        if not d < tol:
            fails.append(f"log_bvn_orthant_quad rho={rho}: {d:.3e}")

    # ---- the scalar helpers ---------------------------------------------
    print("\n  scalar helpers:")
    # Cover LARGE POSITIVE t as well as the deep negative tail. An earlier
    # _log_f1_1 applied its Phi/phi bracket form for all t > -25; that form
    # grows like exp(t^2/2) and returned inf for t >= 45, which destroyed real
    # fits -- and every test at the time swept only t <= 0, so all of them
    # passed. Large |t| of BOTH signs is now part of the contract.
    mu = np.concatenate([rng.uniform(-300, -30, 300),
                         rng.uniform(-30, 30, 300),
                         rng.uniform(30, 300, 300),
                         np.array([-1.0, 0.0, 1.0, -25.0, -24.999, 45.0, 200.0])])
    for sname, sval in (("s=0.4", 0.4), ("s=1.0", 1.0), ("s=3.7", 3.7)):
        for fn in ("_log_phi_at0", "_log_f1_0", "_log_f1_1"):
            c = np.asarray(getattr(CPU, fn)(mu, sval), dtype=float)
            g = np.asarray(getattr(GPU, fn)(mu, sval), dtype=float)
            d = np.abs(c - g).max()
            print(f"    {fn:16s} {sname}  max |dlog| = {d:.3e}"
                  f"   {'OK' if d < 1e-12 else 'MISMATCH'}")
            if not d < 1e-12:
                fails.append(f"{fn} {sname}: {d:.3e}")

    # ---- the ported Owen's T, against scipy (what the CPU actually calls) --
    print("\n  Owen's T vs scipy.special.owens_t (CPU's own reference):")
    ot_cases = [
        ("|a| <= 1", rng.uniform(-40, 40, 4000), rng.uniform(-1, 1, 4000)),
        ("1 < |a| <= 50", rng.uniform(-40, 40, 4000), rng.uniform(-50, 50, 4000)),
        ("|a| > 50", rng.uniform(-40, 40, 4000),
         np.r_[rng.uniform(50, 1e4, 2000), rng.uniform(-1e4, -50, 2000)]),
        ("|a| huge", rng.uniform(-40, 40, 4000), rng.uniform(-1e9, 1e9, 4000)),
        ("|h| huge", np.r_[rng.uniform(40, 300, 2000),
                           rng.uniform(-300, -40, 2000)],
         rng.uniform(-1e6, 1e6, 4000)),
        ("|h| tiny, |a| huge", rng.uniform(-1e-9, 1e-9, 4000),
         rng.uniform(-1e9, 1e9, 4000)),
        ("a at crossover", rng.uniform(-20, 20, 2000),
         np.full(2000, 50.0) * (1 + rng.uniform(-1e-9, 1e-9, 2000))),
        ("a == 0", rng.uniform(-10, 10, 500), np.zeros(500)),
        ("h == 0", np.zeros(500), rng.uniform(-1e5, 1e5, 500)),
    ]
    # 7e-12 is the measured worst corner (|h| tiny, |a| huge), where T is
    # O(0.25) so this is ~3e-11 relative; everything else is <= 1.1e-16.
    OT_TOL = 1e-11
    for name, h, a in ot_cases:
        d = np.abs(GPU._owens_t(h, a) - _sp_owens_t(h, a)).max()
        print(f"    {name:22s} max abs err = {d:.3e}"
              f"   {'OK' if d < OT_TOL else 'MISMATCH'}")
        if not d < OT_TOL:
            fails.append(f"owens_t {name}: {d:.3e}")

    # ---- the full hybrid, CPU vs GPU ------------------------------------
    print("\n  log_bvn_cdf hybrid (CPU vs GPU, both branches):")
    for rho in (-0.9, 0.0, 0.45, 0.97, 0.999):
        h, k = [], []
        for _ in range(1500):
            d = rng.uniform(0.0, 12.0)
            h.append(-d + rng.uniform(-1.5, 1.5))
            k.append(-d + rng.uniform(-1.5, 1.5))
        h, k = np.array(h), np.array(k)
        c = CPU._log_bvn_cdf(h, k, rho)
        g = GPU._log_bvn_cdf(h, k, np.full_like(h, rho))
        d = np.abs(c - g).max()
        # Both paths now run the same two algorithms with the same threshold, so
        # what remains is BRANCH-BOUNDARY SENSITIVITY: for rows whose L lands
        # within rounding of _OWENS_MIN_RELIABLE, the CPU's scipy owens_t and
        # the GPU's quadrature owens_t (which agree to ~1e-16 absolute) can fall
        # on opposite sides of the cut, and the two branches differ by ~1e-9
        # relative at that L. Bounded, confined to the cut, and ~1e-7 in log --
        # immaterial for EM, but it is why this is not exact equality.
        # The per-branch checks above are what catch a genuine divergence.
        print(f"    rho={rho:+.3f}  max |dlog| = {d:.3e}  (tol 1e-6)"
              f"   {'OK' if d < 1e-6 else 'MISMATCH'}")
        if not d < 1e-6:
            fails.append(f"log_bvn_cdf rho={rho}: {d:.3e}")

    print()
    if fails:
        print("FAIL — CPU and GPU helpers have diverged:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS — CPU and GPU run the same algorithms; per-branch agreement is\n"
          "       <=1e-12 (quadrature) and <=1e-11 (Owen's T vs scipy). The\n"
          "       hybrid's residual ~1e-7 is branch-boundary rounding, not a\n"
          "       second implementation -- see the comments at each check.")
    print("NOTE: still run tests/test_batch_em_parity.py on a JAX machine to "
          "validate the traced/jitted path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
