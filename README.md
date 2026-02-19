# FAIR-PV-Psoriasis

Minimal reproducible implementation of the FAIR-PV framework.

This repository contains the exact quantitative engine described in:

Ellenberg E, Katorza E, Ramot Y,.
Refining pharmacovigilance in psoriasis: A quantitative evaluation of biologic-attributable risk using the FAIR framework.

## Mathematical Model

Absolute excess risk:

v(S) = p0 × (RR_product − 1)

Shapley value decomposition is applied to partition excess risk among competing factors.

Uncertainty propagation:

- p0 ~ Beta distribution
- RR ~ LogNormal distribution (derived from 95% CI)
- Monte Carlo simulation (n = 10,000)

Outputs:

- Risk Attribution Index (RAI)
- 90% credibility interval
- P(RAI > 50%)
- P(RAI ≥ 20%)

This repository contains only the minimal computational core.
