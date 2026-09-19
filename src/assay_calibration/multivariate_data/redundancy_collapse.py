"""Optional, opt-in redundancy collapse for highly-correlated assay
dimensions within a gene's MultiScoreset -- e.g. TP53's Kato_2003 8-assay
panel (pairwise r=0.6-0.88, all measuring the same underlying WT-p53
transactivation signal), collapsed to its top-2 principal components before
CFUSN fitting.

**Why this exists**: unweighted joint likelihoods let a large block of
near-duplicate dimensions dominate both the E-step (cluster assignment) and
M-step (parameter updates) relative to unique/orthogonal dimensions with
equal biological relevance but only 1 "vote." Collapsing a validated
redundant block frees up that dominance and (confirmed on real TP53 data)
substantially improves within-component skew capture and evidence recovery
for variants whose only real signal is in the now-less-diluted orthogonal
dimensions.

**Manual opt-in is the only mechanism validated safe enough for direct
production use.** Automatic threshold-based detection (both variants below)
is provided because it's a genuinely useful *diagnostic* tool for surfacing
candidate blocks for human review, but it is NOT safe as a fully automatic,
unsupervised trigger -- two independent real-data findings during
validation:

1. Connected-components chains transitively: dim A correlating with B, and
   B with C, merges A/B/C into one block even if A and C aren't correlated
   at all. On real TP53 data this merged 13-14 of 16 dimensions into one
   block at threshold 0.7-0.75 -- a far more aggressive, more dangerous
   collapse than the validated hand-picked 8-dim grouping.
   `detect_redundant_blocks_clique` (requiring every pair within a block to
   individually clear the threshold) fixes this specific failure mode.
2. Even with cliques, there is no single correlation threshold that is both
   safe (never collapses real complementary signal) and effective (catches
   real redundancy) across gene sets: TP53's real redundant pairs span
   r=0.6-0.88, but real, genuinely complementary LABEL-seq pairs (e.g. a
   gene's plain-abundance vs. HSP90-inhibitor-treated-abundance assay) span
   r=0.30-0.72 and can land INSIDE that same range (e.g. sos2 at 0.723,
   egfr at 0.684) -- a threshold low enough to auto-recover TP53's block
   would also risk auto-collapsing those real LABEL-seq pairs.

So: use `collapse_block` with an explicit, human-chosen `block_dims` (by
dimension name, resolved against the MultiScoreset's own `dataset_names`)
for anything that actually runs. Use `detect_redundant_blocks*` only to
generate candidates for a human to look at, never to auto-collapse without
review.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


# The validated hand-picked block referenced throughout this module's
# docstring: TP53's Kato_2003 8-assay panel (pairwise r=0.6-0.88, all
# measuring the same underlying WT-p53 transactivation signal), collapsed to
# its top-2 principal components. Given as SUFFIXES, not full dataset_names,
# because the two different TP53 MultiScoreset builders in hpc/prepare.py
# name these 8 dims differently: `--gene-set integrated` prefixes them
# ("TP53_Kato_2003_WAF1nWT"), while the dedicated `--gene-set tp53` builder
# uses the bare suffix ("WAF1nWT") -- confirmed by directly inspecting both
# builders' `ms.dataset_names`. Suffix-matching (below, in
# resolve_preset_block_names) lets one preset definition work against
# either naming convention instead of needing two presets for the same
# underlying assay panel. Not applied anywhere by default -- hpc/prepare.py's
# --redundancy-collapse-preset and the MV cockpit's equivalent flag both need
# this passed explicitly; see `collapse_block`'s module-level "manual opt-in
# only" rationale.
TP53_KATO_SUFFIXES = [
    "AIP1nWT", "BAXnWT", "GADD45nWT", "MDM2nWT",
    "NOXAnWT", "P53R2nWT", "WAF1nWT", "h1433snWT",
]
TP53_KATO_K = 2

# name -> (suffixes, k), looked up by hpc/prepare.py's
# --redundancy-collapse-preset and mv_analysis.config's equivalent cockpit
# helper, so both tools share one definition instead of duplicating the 8
# literal dimension-name variants.
PRESETS = {
    "tp53_kato_pca2": (TP53_KATO_SUFFIXES, TP53_KATO_K),
}


def resolve_preset_block_names(dataset_names: Sequence[str], suffixes: Sequence[str]) -> List[str]:
    """Resolve a preset's dimension-name SUFFIXES against one gene's actual
    `ms.dataset_names` (exact match OR "endswith" match, so both TP53
    MultiScoreset-builder naming conventions above resolve correctly).
    Returns [] (not an error) if fewer than len(suffixes) dims resolve --
    callers should treat that the same as "gene doesn't have this block,
    skip" (matching collapse_block's existing missing-dims skip logic)."""
    resolved = []
    for suffix in suffixes:
        matches = [n for n in dataset_names if n == suffix or n.endswith("_" + suffix)]
        if len(matches) != 1:
            return []
        resolved.append(matches[0])
    return resolved


def apply_preset(gene_ms_map: dict, preset: str, k_override: Optional[int] = None,
                  verbose: bool = True) -> dict:
    """Apply a named PRESETS entry to every gene's ms in `gene_ms_map` IN
    PLACE, resolving each preset's suffixes against that gene's own
    `dataset_names` (skipping, not erroring, genes that don't have the full
    block -- same convention as collapse_block's manual callers). Single
    shared implementation for hpc/prepare.py's --redundancy-collapse-preset
    and the MV cockpit's equivalent, so both tools apply the exact same
    definition instead of duplicating this per-gene resolve/skip/collapse
    loop. Returns gene_ms_map (mutated ms objects, same dict).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown redundancy-collapse preset {preset!r}. "
                          f"Available: {sorted(PRESETS.keys())}")
    suffixes, preset_k = PRESETS[preset]
    k = k_override if k_override is not None else preset_k
    for gene, ms in gene_ms_map.items():
        block_names = resolve_preset_block_names(ms.dataset_names, suffixes)
        if not block_names:
            if verbose:
                print(f"  [redundancy-collapse] {gene}: preset {preset!r} suffixes {suffixes} "
                      f"not all found in dataset_names, skipping")
            continue
        before = list(ms.dataset_names)
        collapse_block(ms, block_names, k)
        if verbose:
            print(f"  [redundancy-collapse] {gene}: preset {preset!r} collapsed {block_names} "
                  f"({len(block_names)} dims) -> {k} PC(s); "
                  f"{len(before)} -> {len(ms.dataset_names)} total dims")
    return gene_ms_map


# ──────────────────────────────────────────────────────────────────────
# Correlation graph / candidate-block detection (diagnostic use only --
# see module docstring; not safe as a fully automatic trigger)
# ──────────────────────────────────────────────────────────────────────

def pairwise_corr_matrix(scores: np.ndarray, min_co_observed: int = 20) -> np.ndarray:
    """(p, p) NaN-aware pairwise Pearson correlation matrix. NaN entries
    where fewer than min_co_observed rows have both dims observed."""
    N, p = scores.shape
    corr = np.full((p, p), np.nan)
    np.fill_diagonal(corr, 1.0)
    for i in range(p):
        for j in range(i + 1, p):
            x, y = scores[:, i], scores[:, j]
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() >= min_co_observed:
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                corr[i, j] = corr[j, i] = r
    return corr


def detect_redundant_blocks_connected(scores: np.ndarray, corr_threshold: float = 0.85,
                                       min_co_observed: int = 20) -> List[List[int]]:
    """Connected components of the graph where an edge connects dims i,j
    iff |corr(i,j)| > corr_threshold. DIAGNOSTIC ONLY -- see module
    docstring: this allows transitive chaining and can merge a much larger,
    more heterogeneous block than any single pair actually warrants.
    Returns a list of disjoint blocks (each a sorted list of >=2 dim
    indices)."""
    corr = pairwise_corr_matrix(scores, min_co_observed=min_co_observed)
    p = scores.shape[1]
    adj = np.abs(corr) > corr_threshold
    np.fill_diagonal(adj, False)
    adj = np.nan_to_num(adj, nan=0.0).astype(bool)

    visited = np.zeros(p, dtype=bool)
    blocks = []
    for i in range(p):
        if visited[i] or not adj[i].any():
            continue
        stack = [i]
        comp = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            visited[node] = True
            for n in np.where(adj[node])[0]:
                if n not in comp:
                    stack.append(int(n))
        if len(comp) >= 2:
            blocks.append(sorted(comp))
    return blocks


def _bron_kerbosch(R, P, X, adj, cliques):
    if not P and not X:
        if len(R) >= 2:
            cliques.append(sorted(R))
        return
    for v in list(P):
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
        P = P - {v}
        X = X | {v}


def detect_redundant_blocks_clique(scores: np.ndarray, corr_threshold: float = 0.85,
                                    min_co_observed: int = 20) -> List[List[int]]:
    """Maximal cliques of the thresholded correlation graph: every pair of
    dims WITHIN a returned block individually clears corr_threshold, not
    just a connected path -- avoids `detect_redundant_blocks_connected`'s
    transitive over-merging. DIAGNOSTIC ONLY (see module docstring): even
    with this fix, no single threshold is both safe and effective across
    gene sets, so this should surface candidates for human review, not
    auto-collapse. Returned cliques may overlap (a dim can appear in more
    than one maximal clique); the caller decides how to handle overlaps."""
    corr = pairwise_corr_matrix(scores, min_co_observed=min_co_observed)
    p = scores.shape[1]
    adj = {i: set() for i in range(p)}
    for i in range(p):
        for j in range(p):
            if i != j and np.isfinite(corr[i, j]) and abs(corr[i, j]) > corr_threshold:
                adj[i].add(j)
    cliques = []
    _bron_kerbosch(set(), set(range(p)), set(), adj, cliques)
    cliques.sort(key=len, reverse=True)
    return cliques


def choose_block_k(scores: np.ndarray, block_dims: Sequence[int],
                    variance_threshold: float = 0.87, min_complete_rows: int = 20) -> int:
    """Smallest k (1 <= k < len(block_dims)) whose top-k PCA components
    (fit on rows where the whole block is observed) explain >=
    variance_threshold of the block's own variance. Returns
    len(block_dims) (i.e. "don't collapse") if that bar can't be cleared
    below full dimensionality, or if there isn't enough complete data for a
    stable covariance estimate."""
    sub = scores[:, block_dims]
    complete_mask = ~np.isnan(sub).any(axis=1)
    block_size = len(block_dims)
    if complete_mask.sum() < max(min_complete_rows, block_size + 2):
        return block_size
    X = sub[complete_mask]
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total < 1e-12:
        return block_size
    cumvar = np.cumsum(eigvals) / total
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    return min(k, block_size)


# ──────────────────────────────────────────────────────────────────────
# The actual collapse mechanism -- this part is production-safe when given
# an explicit, human-chosen block_dims.
# ──────────────────────────────────────────────────────────────────────

class RedundancyCollapseTransform:
    """Fitted linear projection for one block: maps raw scores in
    `block_dims` to a k-dim collapsed representation (top-k PCA, fit on
    rows where the whole block is observed).

    `back_project_marginal` gives an APPROXIMATE per-raw-dimension view of
    fitted CFUSN component params for interpretability -- NOT a full
    reconstruction. PCA permanently discards the orthogonal-complement
    directions beyond the retained k components, and the fitted model never
    sees those directions; the back-projection reflects only the part of
    each raw dimension's behavior explained by the retained components.
    """

    def __init__(self, block_dims: Sequence[int], k: int, mean_: np.ndarray, evecs: np.ndarray):
        self.block_dims = list(block_dims)
        self.k = k
        self.mean_ = mean_
        self.evecs = evecs  # (block_size, k)

    @classmethod
    def fit(cls, scores: np.ndarray, block_dims: Sequence[int], k: int,
            min_complete_rows: int = 20) -> "RedundancyCollapseTransform":
        sub = scores[:, block_dims]
        complete_mask = ~np.isnan(sub).any(axis=1)
        if complete_mask.sum() < max(min_complete_rows, len(block_dims) + 2):
            raise ValueError(
                f"Not enough complete rows ({complete_mask.sum()}) to fit a "
                f"stable {k}-component projection for block {block_dims}"
            )
        X = sub[complete_mask]
        mean_ = X.mean(axis=0)
        Xc = X - mean_
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1][:k]
        evecs = eigvecs[:, order]
        return cls(block_dims, k, mean_, evecs)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Raw (N, p_full) scores -> (N, k) collapsed values. Rows where
        any block dim is NaN get NaN collapsed values (no imputation)."""
        sub = scores[:, self.block_dims]
        mask = ~np.isnan(sub).any(axis=1)
        out = np.full((scores.shape[0], self.k), np.nan)
        out[mask] = (sub[mask] - self.mean_) @ self.evecs
        return out

    def back_project_marginal(self, mu_k, delta_k, gamma_diag_k):
        """mu_k: (k,), delta_k: (k, q), gamma_diag_k: (k,) fitted params in
        collapsed space. Returns approximate per-raw-dim (mu, delta, var),
        shapes (block_size,), (block_size, q), (block_size,) -- see class
        docstring caveat."""
        mu_raw = self.mean_ + self.evecs @ mu_k
        delta_raw = self.evecs @ delta_k
        var_raw = np.einsum('ik,k,ik->i', self.evecs, gamma_diag_k, self.evecs)
        return mu_raw, delta_raw, var_raw


def collapse_block(ms, block_names: Sequence[str], k: int,
                    new_names: Optional[Sequence[str]] = None) -> RedundancyCollapseTransform:
    """Collapse one named block of dimensions on `ms` IN PLACE (mutates
    `ms._scores`, `ms.dataset_names`, and the other attributes derived from
    the score matrix's shape/content -- `ms.d` (backs the `n_assays`
    property), `ms._xlims` (backs `xlims`), `ms._missing` (backs
    `missing`). `.scores`/`.xlims`/`.missing` are all read-only
    `@property`s backed by private attributes on both MultiScoreset/
    BasicMultiScoreset, so those must be set directly, not through the
    properties. `sample_counts`/`_variants_kept` are derived from
    `sample_assignments` rather than `scores` and are unaffected.

    `block_names` : dimension names to collapse, resolved against
    `ms.dataset_names` -- order doesn't need to match the block's storage
    order, just needs to name existing dimensions.
    `k` : target dimensionality for this block (human-chosen; use
    `choose_block_k` as a starting suggestion, not an automatic decision).
    `new_names` : names for the k collapsed columns in the resulting
    `ms.dataset_names` (default: "{first_block_name}_PC1", "_PC2", ...).

    Returns the fitted `RedundancyCollapseTransform` (needed to score new
    variants consistently with what was fit, and for back-projection).
    """
    names = list(ms.dataset_names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing_names = [n for n in block_names if n not in name_to_idx]
    if missing_names:
        raise ValueError(f"collapse_block: dimension(s) not found in ms.dataset_names: {missing_names}")
    block_dims = [name_to_idx[n] for n in block_names]
    remaining_dims = [i for i in range(len(names)) if i not in block_dims]

    transform = RedundancyCollapseTransform.fit(ms.scores, block_dims, k)
    collapsed = transform.transform(ms.scores)
    new_scores = np.hstack([collapsed, ms.scores[:, remaining_dims]])

    if new_names is None:
        new_names = [f"{block_names[0]}_PC{i+1}" for i in range(k)]
    ms._scores = new_scores
    ms.dataset_names = list(new_names) + [names[i] for i in remaining_dims]
    ms.d = new_scores.shape[1]
    ms._missing = np.isnan(new_scores)
    ms._xlims = tuple(
        (float(np.nanmin(new_scores[:, d])), float(np.nanmax(new_scores[:, d])))
        for d in range(new_scores.shape[1])
    )

    return transform
