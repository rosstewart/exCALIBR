# Prior-estimation simulation (Reviewer 3, #6)

This simulation tests whether ExCALIBR's gnomAD-derived prior recovers the true
fraction of pathogenic population variants when an assay measures only one of
a gene's disease mechanisms. It also tests how label noise, mechanism-biased
P/LP sampling, assay separation and correlation between mechanisms move the
prior.

    python -m analysis.prior_simulation.simulate_priors [--n-jobs 64] [--no-fit]
    python -m analysis.prior_simulation.plot_priors

Outputs go to `/data/ross/assay_calibration/prior_simulation/`:

| File | Contents |
|---|---|
| `results.csv` | One row per simulation cell. |
| `fit_reps.csv` | Production-fit prior (median over bootstraps) for each replicate dataset. |
| `sanity_large_n.csv` | Fixed-component estimate with 2×10⁵ variants per sample. |
| `prior_simulation.{png,pdf}` | Figure. |

## Model

Each variant has latent effects (e1, e2) on mechanisms M1 and M2, drawn from
N(μ, Σ) with Σ = [[1, ρ], [ρ, 1]]. The class means are:

| Class | Mean (e1, e2) |
|---|---|
| Benign | (0, 0) |
| Pathogenic via M1 | (−Δ, −ρΔ) |
| Pathogenic via M2 | (−ρΔ, −Δ) |

25% of pathogenic population variants act via M1.

The three assays read:

| Assay | Score |
|---|---|
| A1 | e1 |
| A2 | e2 |
| A3 | min(e1, e2) |

The samples are:

| Sample | Base composition |
|---|---|
| gnomAD | pathogenic with probability π |
| P/LP | pathogenic |
| B/LB | benign |
| Synonymous | benign |

The base case is Δ = 4 and ρ = 0, with clean labels and the P/LP mechanism mix
equal to the population's.

## Estimators

Both estimators end in the production prior EM on gnomAD, with
`benign_method="avg"` and the production rule that discards a prior ≤ 0.001
or ≥ 0.999.

- **Fixed true components.** There is no mixture fit. The class densities are
  exact, and each sample's class weights come from weight-only EM. The prior
  is computed by `point_ranges.estimate_prior_from_class_densities`, the same
  function `get_fit_prior` calls. A built-in check confirms that this path
  equals `get_fit_prior` for Gaussian assays.
- **Production fit.** `Fit.run` with 2 Gaussian components
  (`force_gaussian`), 1 restart and 100 bootstraps, then `get_fit_prior` on
  the full dataset.

`expected_prior` is the infinite-control limit: the true class weights, with
the prior EM run on 2×10⁵ gnomAD variants.
