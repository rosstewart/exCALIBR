import numpy as np
from typing import Union


def _tavtigian_fails(C_vals: np.ndarray, prior: float, original: bool, strict: bool,
                      verbose: bool = False):
    """Total rule-violation count for each candidate C in C_vals -- the
    per-candidate objective get_tavtigian_constant minimizes. Factored out
    so it can be reused for both a coarse (wide-range, log-spaced) and a
    fine (dense integer) search pass; see get_tavtigian_constant.
    """
    pathogenic_posteriors = np.round(
        np.stack(
            list(map(lambda C: pathogenicRulesPosterior(C, prior, original), C_vals)),
            axis=0,
        ),
        3,
    )
    pathogenic_fails = np.sum(pathogenic_posteriors < 0.99, axis=1)

    likely_pathogenic_posteriors = np.round(
        np.stack(
            list(
                map(
                    lambda C: likelyPathogenicRulesPosterior(C, prior, original), C_vals
                )
            ),
            axis=0,
        ),
        3,
    )
    lp_mask = likely_pathogenic_posteriors < 0.90
    if strict:
        lp_mask = lp_mask | (likely_pathogenic_posteriors > 0.99)
    likely_pathogenic_fails = np.sum(lp_mask, axis=1)

    benign_posteriors = np.round(
        np.stack(list(map(lambda C: benignRulesPosterior(C, prior), C_vals)), axis=0), 3
    )
    benign_fails = np.sum(benign_posteriors > 0.01, axis=1)

    likely_benign_posteriors = np.round(
        np.stack(
            list(map(lambda C: likelybenignRulesPosterior(C, prior, original), C_vals)),
            axis=0,
        ),
        3,
    )

    lb_mask = likely_benign_posteriors > 0.10
    if strict:
        lb_mask = lb_mask | (likely_benign_posteriors < 0.01)
    likely_benign_fails = np.sum(lb_mask, axis=1)
    fails = (
        pathogenic_fails + likely_pathogenic_fails + benign_fails + likely_benign_fails
    )
    if verbose:
        star_idx = np.argmin(fails)
        print(likely_pathogenic_posteriors[star_idx])
        print(pathogenic_posteriors[star_idx])
        print(likely_benign_posteriors[star_idx])
        print(benign_posteriors[star_idx])
    return fails


def get_tavtigian_constant(prior: float, *args, **kwargs) -> Union[float, int]:
    original = kwargs.get("original", False)
    strict = kwargs.get("strict", False)
    verbose = kwargs.get("verbose", False)
    # Explicit C_max => legacy behavior: a single dense linear scan over
    # [1, C_max], exactly as before (kept for reproducibility/callers that
    # pin a specific search range).
    C_max = kwargs.get("C_max", None)

    if C_max is not None:
        C_vals = np.arange(1, C_max + 1)
        fails = _tavtigian_fails(C_vals, prior, original, strict, verbose)
        return C_vals[np.argmin(fails)]

    # No C_max given: adaptive two-stage search with no hard ceiling. A
    # dense linear scan up to a fixed cap (previously 100000) silently
    # truncates the true optimum for very low/high priors, where the
    # best-fit C* can run into the millions or more -- this showed up as
    # visibly wrong, capped-looking C*(p) behavior at the extremes of the
    # prior range. Instead: (1) a coarse log-spaced scan across a huge
    # range (1 to 10^15) to locate the right order of magnitude, then (2) a
    # dense integer scan in a window around that coarse optimum to pin down
    # the exact best integer, same precision as the old dense scan.
    coarse_vals = np.unique(np.round(np.logspace(0, 15, 600)).astype(np.int64))
    coarse_vals = coarse_vals[coarse_vals >= 1]
    coarse_fails = _tavtigian_fails(coarse_vals, prior, original, strict)
    best_i = int(np.argmin(coarse_fails))
    lo = coarse_vals[max(best_i - 1, 0)]
    hi = coarse_vals[min(best_i + 1, len(coarse_vals) - 1)]

    fine_vals = np.arange(max(1, lo), hi + 1)
    if len(fine_vals) > 250_000:
        # Window too wide for a dense integer scan (only possible far out
        # in the log grid, where neighboring points are themselves far
        # apart) -- fall back to a denser log-spaced sub-scan of the same
        # window instead of scanning every integer in it.
        fine_vals = np.unique(np.round(np.geomspace(max(1, lo), hi, 250_000)).astype(np.int64))
    fine_fails = _tavtigian_fails(fine_vals, prior, original, strict, verbose)
    return int(fine_vals[np.argmin(fine_fails)])


def pathogenicRulesPosterior(C: int, prior: float, original: bool) -> np.ndarray:
    fracs = [2**-3, 2**-2, 2**-1, 1]
    posterior = np.zeros(8)
    posterior[0] = locallrPlus2Posterior(C ** (np.dot([0, 0, 1, 1], fracs)), prior)
    posterior[1] = locallrPlus2Posterior(C ** (np.dot([0, 2, 0, 1], fracs)), prior)
    posterior[2] = locallrPlus2Posterior(C ** (np.dot([1, 1, 0, 1], fracs)), prior)
    posterior[3] = locallrPlus2Posterior(C ** (np.dot([2, 0, 0, 1], fracs)), prior)
    posterior[4] = locallrPlus2Posterior(C ** (np.dot([0, 3, 1, 0], fracs)), prior)
    posterior[5] = locallrPlus2Posterior(C ** (np.dot([2, 2, 1, 0], fracs)), prior)
    posterior[6] = locallrPlus2Posterior(C ** (np.dot([4, 1, 1, 0], fracs)), prior)
    if original:
        # The original rules consider 2 strong lines of evidence as pathogenic.
        posterior[7] = locallrPlus2Posterior(C ** (np.dot([0, 0, 2, 0], fracs)), prior)
    else:
        # The modified rules consider 1 moderate and 1 very strong lines as
        # pathogenic. This replaces the 2 strong lines of evidence rule, which
        # is moved to the likely pathogenic rules.
        posterior[7] = locallrPlus2Posterior(C ** (np.dot([0, 1, 0, 1], fracs)), prior)
    return posterior


def likelyPathogenicRulesPosterior(C: int, prior: float, original: bool) -> np.ndarray:
    fracs = [2**-3, 2**-2, 2**-1, 1]
    posterior = np.zeros(6)
    posterior[0] = locallrPlus2Posterior(C ** (np.dot([0, 1, 1, 0], fracs)), prior)
    posterior[1] = locallrPlus2Posterior(C ** (np.dot([2, 0, 1, 0], fracs)), prior)
    posterior[2] = locallrPlus2Posterior(C ** (np.dot([0, 3, 0, 0], fracs)), prior)
    posterior[3] = locallrPlus2Posterior(C ** (np.dot([2, 2, 0, 0], fracs)), prior)
    posterior[4] = locallrPlus2Posterior(C ** (np.dot([4, 1, 0, 0], fracs)), prior)
    if original:
        #  The original rules consider 1 moderate and 1 very strong evidence as likely pathogenic.
        posterior[5] = locallrPlus2Posterior(C ** (np.dot([0, 1, 0, 1], fracs)), prior)
    else:
        # The modified rules consider 2 strong evidence as likely pathogenic.
        # This replaces the 1 moderate and 1 very strong evidence rule, which
        # is moved to the pathogenic rules.
        posterior[5] = locallrPlus2Posterior(C ** (np.dot([0, 0, 2, 0], fracs)), prior)
    return posterior


def locallrPlus2Posterior(llrp: float, prior: float) -> float:
    return llrp * prior / ((llrp - 1) * prior + 1)


def benignRulesPosterior(C: float, prior: float):
    ## Computes the posterior values for the ç benign rules under
    ## Tavtigan's framework [1] for a candidate C (O_PVSt) and a prior.
    # [1] Tavtigian, Sean V., et al. "Modeling the ACMG/AMP variant classification
    # guidelines as a Bayesian classification framework." Genetics in Medicine
    # 20.9 (2018): 1054-1060.
    # Inputs:
    # C: (numeric scalar) The constant O_PVSt in [1].
    # prior: (numeric scalar in (0,1)) the proprtion of positives in the
    #   reference population.
    # original: (boolean, defualt false) set to true for using the original
    #   ACMG/AMP combing rules. Set to false if modified rules are to be used.
    #
    # Output:
    # posterior: (numeric vector of length 1 (number of benign rules))
    #   The posterior computed for each rule.

    # fracs contains the exponents of C to be used to compute the positive likelihood
    # ratio (lr+) for a single line of supporting, moderate, strong and very strong
    # evidence towards benignity. For example, the the lr+ for a single line of
    # supporting evidence is C**(-2**-3). Note that the fracs for benign evidence is
    # negative of that for pathogenic evidence as given in
    # 'pathogenicRulesPosterior' function. This allows cancelling pathogenic
    # and benign evidence of equal strength. However, no ACMG/AMP rule
    # explicitly combines them.
    fracs = -np.array([2**-3, 2**-2, 2**-1, 1])
    posterior = np.zeros(1)
    # The combined lr+ of a rule is obtained by doting the lr+ for evidences
    # supported in the rule. For example, for the rule requiring 2 strong
    # evidence, combined lr+ is C**(-2*2**-1).
    # The posterior for the rule is computed from the prior and the combined lr+.
    # The vector of counts below specify the number of evidence of each kind
    # supported in the rule. For example, the rule supports
    # and 2 strong evidence.
    posterior[0] = locallrPlus2Posterior(C ** (np.dot([0, 0, 2, 0], fracs)), prior)

    return posterior


def likelybenignRulesPosterior(e, alpha, original):
    ## Computes the posterior values for the ACMG/AMP likely benign rules under
    ## Tavtigan's framework [1] for a candidate C (O_PVSt) and a prior.
    # [1] Tavtigian, Sean V., et al. "Modeling the ACMG/AMP variant classification
    # guidelines as a Bayesian classification framework." Genetics in Medicine
    # 20.9 (2018): 1054-1060.
    # Inputs:
    # C: (numeric scalar) The constant O_PVSt in [1].
    # prior: (numeric scalar in (0,1)) the proprtion of positives in the
    #   reference population.
    # original: (boolean, defualt false) set to true for using the original
    #   ACMG/AMP combing rules. Set to false if modified rules are to be used.
    #
    # Output:
    # posterior: (numeric vector of length 1 (number of benign rules))
    #   The posterior computed for each rule.

    # fracs contains the exponents of C to be used to compute the positive likelihood
    # ratio (lr+) for a single line of supporting, moderate, strong and very strong
    # evidence towards benignity. For example, the the lr+ for a single line of
    # supporting evidence is C**(-2**-3).
    fracs = -np.array([2**-3, 2**-2, 2**-1, 1])
    v = np.zeros(2)
    # The combined lr+ of a rule is obtained by doting the lr+ for evidences
    # supported in the rule. For example, for the rule requiring 1 supporting and
    # 1 strong evidence, combined lr+ is C**(-2**-3 -2**-1).
    # The posterior for the rule is computed from the prior and the combined lr+.
    # The vector of counts below specify the number of evidence of each kind
    # supported in the rule. For example, the rule supports
    # and 1 supporting and 1 strong evidence.
    v[0] = locallrPlus2Posterior(e ** (np.dot([1, 0, 1, 0], fracs)), alpha)
    v[1] = locallrPlus2Posterior(e ** (np.dot([2, 0, 0, 0], fracs)), alpha)
    # if original:
    #     # The original rules consider 2 supporting lines of evidence as likely benign.
    #     # This rule is not included in the modified rules.

    # else:
    #     v = v[:1]
    return v
