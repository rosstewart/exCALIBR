"""Per-variant mechanism attribution for factorial fits (EXPERIMENTAL; report
only -- evidence is unchanged).

For each axis, the "defective" state is the one with the largest
P/LP : B/LB axis-marginal weight ratio. A variant's attribution on an axis is
its posterior probability of being in that axis's defective state under the
P/LP sample's weights (rows with no data on an axis keep the prior, and are
reported as unmeasured). A variant is flagged as a single-axis defect when
exactly one MEASURED axis has P(defective) >= hi and every other measured axis
has P(defective) <= lo.
"""
import numpy as np
import pandas as pd


def defective_states(model, p_idx, b_idx, eps=1e-6):
    out = []
    for a in range(len(model.axes)):
        mp = model.axis_marginal_weights(p_idx)[a]
        mb = model.axis_marginal_weights(b_idx)[a]
        out.append(int(np.argmax(np.log(mp + eps) - np.log(mb + eps))))
    return out


def attribution_table(model, X, ids, p_idx, b_idx, axis_names=None, hi=0.8, lo=0.2):
    """X in the model's column space (standardized if the model was)."""
    post = model.axis_posteriors(X, model.W[p_idx])
    dstates = defective_states(model, p_idx, b_idx)
    names = axis_names or [f"axis{a + 1}" for a in range(len(model.axes))]
    df = pd.DataFrame({"variant": np.asarray(ids)})
    measured = []
    for a, ax in enumerate(model.axes):
        cols = ax.cont + ax.cats
        m = ~np.all(np.isnan(X[:, cols]), axis=1)
        measured.append(m)
        df[f"P_defective_{names[a]}"] = np.where(m, post[a][dstates[a]], np.nan)
    P = np.array([df[f"P_defective_{n}"].to_numpy() for n in names])        # (A, N), NaN = unmeasured
    n_meas = np.sum(np.array(measured), axis=0)
    high = np.nan_to_num(P, nan=0.0) >= hi
    low_or_unmeasured = np.isnan(P) | (P <= lo)
    single = (high.sum(0) == 1) & (np.all(high | low_or_unmeasured, axis=0)) & (n_meas >= 2)
    df["n_axes_measured"] = n_meas
    df["single_axis_defect"] = single
    df["defective_axis"] = [names[int(np.argmax(np.nan_to_num(P[:, i], nan=-1)))] if single[i] else ""
                            for i in range(P.shape[1])]
    return df, dstates
