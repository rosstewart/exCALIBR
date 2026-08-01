← [Back to README](../README.md)

# Configuration Options

### Bootstrap parameters

```bash
# Interactive / exploratory (defaults)
python run_pipeline.py --dataset my.csv --name MyGene
# → 20 bootstraps × 8 fits

# Production quality
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 100
```

The calibration works by repeatedly refitting the model to resampled
("bootstrapped") versions of your data, then combining the results — this
is what makes the output stable and gives you a confidence range instead
of a single brittle fit. Two settings control how much of this resampling
is done:

- `--n-bootstraps`: how many resampled versions of the dataset get fit.
  This is the main quality/speed dial.
- `--fits-per-bootstrap`: for each resampled version, how many times the
  fitting is retried from a different random starting point (only the
  best-scoring attempt is kept). A secondary dial.

Turning either one up gives a more stable, reproducible result, at the
cost of more compute time.

#### Quality vs. speed presets

We tested how much the final result changes as `--n-bootstraps` is
lowered, using 87 real assay datasets (about 85,000 variants total) and
comparing each reduced run against a much larger (~1000-bootstrap) run of
the same dataset as a reference. For every variant, we recorded exactly
what happened to its evidence-point value (the signed number, e.g. `+2` or
`-3`, that the point-range calibration assigns it — positive numbers favor
pathogenic, negative numbers favor benign, `0` means no evidence either
way) compared to the reference run:

| `--n-bootstraps` | Unchanged | Shifted within the same evidence-strength band* | Shifted to a different band, same direction | Went from no evidence to some evidence | Went from some evidence back to none | Flipped from pathogenic-leaning to benign-leaning (or vice versa) |
|---|---|---|---|---|---|---|
| 20  | 79.1% | 10.3% | 7.7% | 2.6% | 0.4% | 0.0% |
| 50  | 83.4% | 7.9%  | 5.4% | 1.8% | 1.6% | 0.0% |
| 100 | 90.0% | 4.2%  | 3.5% | 1.3% | 1.0% | 0.0% |
| 250 | 92.9% | 2.8%  | 3.2% | 0.3% | 0.8% | 0.0% |
| 500 | 95.7% | 1.6%  | 1.6% | 0.1% | 0.9% | 0.0% |

\* Evidence-strength bands (same for pathogenic and benign direction): `1` = Supporting, `2`–`3` = Moderate, `4`–`7` = Strong, `8` = Very Strong.

The most reassuring number here: across every bootstrap count we tested,
**0.0% of variants flipped from pathogenic-leaning to benign-leaning or
vice versa** — the most severe kind of disagreement never happened, even
at the lowest bootstrap count. What does happen at low bootstrap counts is
smaller drift: a variant's point value nudging up or down (usually within
the same evidence-strength band, sometimes into a neighboring one), or
occasionally gaining/losing evidence entirely (moving to/from `0`).

Separately, we also compared each reduced run's overall ability to tell
already-labeled pathogenic and benign variants apart, and what fraction of
variants get a confident call at all (instead of an inconclusive result).
Both barely move for a typical dataset at any bootstrap count tested; for
the hardest ~10% of datasets, the confident-call fraction can differ by up
to ~9 percentage points at 20 bootstraps, dropping to ~2 by 500.

**Caveat:** this comparison only varies `--n-bootstraps`; it always used
`--fits-per-bootstrap 100`, not the pipeline's default of 8. The presets
below assume the two dials affect quality in a similar way, which we
believe is reasonable but have not separately confirmed.

Based on this, here are five presets to choose from:

| Preset | `--n-bootstraps` | `--fits-per-bootstrap` | What to expect |
|---|---|---|---|
| Light (default) | 20   | 8   | Fastest option, good for a first look. ~21% of variants get some kind of different point value than a much larger run (see table above for the breakdown); never a pathogenic-vs-benign flip. |
| Medium           | 100  | 8   | Noticeably more stable, still practical to run on a laptop/desktop. ~10% of variants shift. |
| Large            | 500  | 8   | Good for a result you plan to rely on. ~4% of variants shift. Best run on a shared server or cluster. |
| XL               | 1000 | 8   | Matches the bootstrap count used as the reference standard above, so drift should be minimal — but we only directly confirmed this at `--fits-per-bootstrap 100`, not 8. Needs a server/cluster. |
| Finest           | 1000 | 100 | The reference-quality configuration itself. Very slow — intended for a compute cluster, not a personal computer. |

```bash
# Light (default) — fast, exploratory
python run_pipeline.py --dataset my.csv --name MyGene

# Medium — better stability, still practical on a laptop/desktop
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 100 --fits-per-bootstrap 8

# Large
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 500 --fits-per-bootstrap 8

# XL
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 8

# Finest — the reference-quality configuration; run on a cluster
python run_pipeline.py --dataset my.csv --name MyGene \
    --n-bootstraps 1000 --fits-per-bootstrap 100
```

For Large/XL/Finest on many datasets, prefer the
[batch HPC workflow](batch-hpc-workflow.md) (SLURM array jobs, one dataset
per node) instead of running them one at a time on your own computer.

#### Speed estimates

How long a run takes mostly depends on two things: how big your preset is
(`--n-bootstraps × --fits-per-bootstrap`) and how many CPU cores your
computer can devote to the job (`--n-jobs`; use `--n-jobs -1` to use all
available cores). Almost all your CPU cores can work on this at the same
time, so more cores means a roughly proportional speedup.

We timed the example dataset (`example/MSH2_Jia_2021.csv`, 1579 variants)
at the Light preset (20 bootstraps × 8 fits) and scaled that measurement
up for the other presets, assuming the same proportional speedup on more
cores. Real times will vary by dataset size and computer, but this should
give a reasonable ballpark:

| Preset | 4 cores (typical laptop) | 16 cores (workstation) | 64 cores (server/cluster node) |
|---|---|---|---|
| Light  | ~2.5 hours | ~35 min | ~10 min |
| Medium | ~12 hours  | ~3 hours | ~45 min |
| Large  | ~2.5 days  | ~16 hours | ~4 hours |
| XL     | ~5 days    | ~1.3 days | ~8 hours |
| Finest | ~2 months  | ~16 days | ~4 days |

A few notes:
- These are for a single dataset, single gene/assay. If you're calibrating
  many datasets at once, use the [batch HPC workflow](batch-hpc-workflow.md)
  so datasets run in parallel across a cluster instead of one after another.
- Most of this time (well over 90%, in our test) goes to the bootstrap
  fitting step itself; the plotting/export steps that follow take well
  under a minute regardless of preset.
- Fitting only a 3-component model (the default) is slightly slower than
  fitting only a 2-component model (roughly 15-20% slower in our test) —
  fitting both (`--components 2 3`) takes about as long as the two added
  together.

You can reproduce or extend these measurements with
`tests/benchmark_run_pipeline_speed.py`.

### Component selection

```bash
# Default: 3-component only (assumed at least as good as 2c for most assays,
# not always true, but a reasonable default for typical usage)
python run_pipeline.py --dataset my.csv --name MyGene

# Fit both 2c and 3c and auto-select the better one
python run_pipeline.py --dataset my.csv --name MyGene --components 2 3

# Fit 5-component model
python run_pipeline.py --dataset my.csv --name MyGene --components 5
```

### Prior estimation

The "prior" is the estimated probability that a randomly chosen variant in
this gene is pathogenic. It's a key input to the ACMG evidence-strength
thresholds: a rarer-disease gene (lower prior) requires stronger assay
evidence to reach the same evidence tier than a gene where pathogenic
variants are more common. By default, the prior is estimated directly from
your data (via EM fitting to the pathogenic/benign score distributions) —
you don't need to supply one yourself.

```bash
# Empirical EM estimation (default) -- estimates the prior from your data
python run_pipeline.py --dataset my.csv --name MyGene

# Use 5th/95th percentile thresholds instead of the median-prior estimate
python run_pipeline.py --dataset my.csv --name MyGene --no-median-prior
```

**If you have domain knowledge about this gene's true prevalence of
pathogenic variants (e.g. from a published disease-prevalence estimate),
or you're not satisfied with the data-driven estimate** (e.g. too few
labeled controls to estimate it reliably), you can set it directly and
skip estimation entirely with `--manual-prior`:

```bash
# Manually set the prior, e.g. from an external estimate
python run_pipeline.py --dataset my.csv --name MyGene --manual-prior 0.001
```

`--manual-prior` takes a probability strictly between 0 and 1 and applies
to the whole run — every bootstrap fit's evidence thresholds are then
derived from this fixed value instead of a per-fit estimate.

#### Pathomechanism prior (advanced)

Some assays only detect *one* of several disease mechanisms for a gene —
for example, an assay that measures loss-of-function will correctly flag
LoF-pathogenic variants as abnormal, but a dominant-negative or
gain-of-function pathogenic variant might score exactly like a benign one,
simply because the assay isn't measuring the thing that makes it
pathogenic. Left alone, this "dilutes" the pathogenic sample with variants
that are truly pathogenic but invisible to this particular assay, which
can understate how strong the assay's pathogenic-direction evidence
really is for the variants it *can* detect.

`--pathomechanism-prior` addresses this by estimating what fraction of
your ClinVar-labeled pathogenic variants the assay actually appears to
detect, and computing a separate, undiluted pathogenic-direction
prior/evidence pair from just that subset. The benign-direction evidence
(a normal-looking score suggesting a variant is benign) is left completely
untouched, since that reasoning doesn't depend on disease mechanism.

```bash
# Enable the pathomechanism-aware pathogenic-direction prior
python run_pipeline.py --dataset my.csv --name MyGene --pathomechanism-prior
```

This only applies to datasets with a Pathogenic-labeled sample (PN or PU
sample-availability mode) and is mutually exclusive with the experimental
`--filter-pathogenic-sample-by-lr` flag. The estimated "fraction of
pathogenic variants this assay detects" and the resulting pathogenic-only
prior are reported in the output JSON as `PLP_frac_pathomechanism_measured`
and `pathomechanism_prior` respectively. Consider this flag if you suspect
your assay is mechanism-specific (e.g. it only captures loss-of-function)
for a gene where disease is caused by multiple distinct mechanisms.

### Benign sample method

```bash
--benign-method avg         # Average benign and synonymous (default when both exist)
--benign-method benign      # Use benign (ClinVar B/LB) only
--benign-method synonymous  # Use synonymous only
```

### Point-range postprocessing

```bash
# Default: enforce monotonicity + extend to score-axis limits
python run_pipeline.py --dataset my.csv --name MyGene

# Disable ALL postprocessing and see the raw, unprocessed LR-threshold-crossing
# intervals exactly as fitted -- a debugging/inspection tool, not something you
# need for bidirectional assays (see below, handled automatically)
python run_pipeline.py --dataset my.csv --name MyGene --no-postprocess

# Conservative (stricter) monotonicity enforcement
python run_pipeline.py --dataset my.csv --name MyGene --conservative-monotonicity
```

### Bidirectional assay auto-detection

Some assays (e.g. LoF/GoF in one assay) show pathogenic-leaning evidence on
**both** sides of a benign region — the standard single-direction
monotonicity/extend-to-limits postprocessing assumes one direction and is
inappropriate for these. `run_pipeline.py` and `run_igvf_batch.py`
**auto-detect this pattern by default** (`n_c >= 3` fits only), so you
should not need `--no-postprocess` (a separate, blunter "give me raw
output" flag, see above) for this case:

- For each bootstrap fit, mixture components are sorted along the score
  axis and labeled pathogenic-like/benign-like by comparing the pathogenic
  and benign samples' mixture weights (PU/NU sample-availability cases fall
  back to gnomAD in place of whichever of pathogenic/benign is unavailable).
  A fit is flagged if a benign-like component has a pathogenic-like
  component on each side.
- If a majority of bootstrap fits are flagged, standard monotonicity
  enforcement is skipped for that dataset's **pathogenic** tiers (each side
  of the benign region is independently re-nested and extended toward its
  own axis limit, honoring `--conservative-monotonicity` per side). Benign
  tiers are assumed to never be bidirectional: they still get cleaned up
  (noisy same-tier fragmentation merged in liberal mode; the standard
  strict "evidence goes back to indeterminate" removal in conservative
  mode), but are never extended to the axis limit.
- Otherwise, standard postprocessing runs unchanged.

```bash
# Default: auto-detection on
python run_pipeline.py --dataset my.csv --name MyGene --components 3

# Disable auto-detection; use standard postprocessing unconditionally
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional

# Disable auto-detection AND all postprocessing (old bidirectional-assay workflow)
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional --no-postprocess
```
