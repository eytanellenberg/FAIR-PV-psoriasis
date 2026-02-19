import numpy as np
from scipy.stats import beta, lognorm

# -------------------------
# Core deterministic model
# -------------------------

def value_function(p0, rrs):
    """
    Absolute excess risk:
    v(S) = p0 * (RR_product - 1)
    """
    rr_product = np.prod(rrs)
    return p0 * (rr_product - 1)


def shapley_two_factors(p0, rr1, rr2):
    """
    Exact Shapley value for two factors
    """
    v_empty = 0
    v1 = value_function(p0, [rr1])
    v2 = value_function(p0, [rr2])
    v12 = value_function(p0, [rr1, rr2])

    phi1 = (v1 + (v12 - v2)) / 2
    phi2 = (v2 + (v12 - v1)) / 2

    total = v12

    rai1 = phi1 / total if total > 0 else 0
    rai2 = phi2 / total if total > 0 else 0

    return phi1, phi2, rai1, rai2


# -------------------------
# Monte Carlo engine
# -------------------------

def monte_carlo_simulation(
    p0_alpha,
    p0_beta,
    rr1_median,
    rr1_ci_low,
    rr1_ci_high,
    rr2_median,
    rr2_ci_low,
    rr2_ci_high,
    n=10000
):
    """
    Monte Carlo propagation as promised in the paper:
    - p0 ~ Beta(alpha, beta)
    - RR ~ LogNormal derived from 95% CI
    """

    # Convert CI to lognormal sigma
    def lognormal_params(median, low, high):
        sigma = (np.log(high) - np.log(low)) / (2 * 1.96)
        mu = np.log(median)
        return mu, sigma

    mu1, sigma1 = lognormal_params(rr1_median, rr1_ci_low, rr1_ci_high)
    mu2, sigma2 = lognormal_params(rr2_median, rr2_ci_low, rr2_ci_high)

    rai1_list = []

    for _ in range(n):
        p0 = beta.rvs(p0_alpha, p0_beta)
        rr1 = lognorm(s=sigma1, scale=np.exp(mu1)).rvs()
        rr2 = lognorm(s=sigma2, scale=np.exp(mu2)).rvs()

        _, _, rai1, _ = shapley_two_factors(p0, rr1, rr2)
        rai1_list.append(rai1)

    rai1_array = np.array(rai1_list)

    return {
        "median_RAI": np.median(rai1_array),
        "CI90_low": np.percentile(rai1_array, 5),
        "CI90_high": np.percentile(rai1_array, 95),
        "P_RAI_>50": np.mean(rai1_array > 0.5),
        "P_RAI_>=20": np.mean(rai1_array >= 0.2),
    }


if __name__ == "__main__":
    # Example test
    result = monte_carlo_simulation(
        p0_alpha=7,
        p0_beta=43,
        rr1_median=3.2,
        rr1_ci_low=2.1,
        rr1_ci_high=4.8,
        rr2_median=1.8,
        rr2_ci_low=1.2,
        rr2_ci_high=2.4,
        n=10000
    )

    print(result)
