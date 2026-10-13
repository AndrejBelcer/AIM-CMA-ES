import numpy as np
import scipy.special as sp
from typing import Dict, Tuple

from src.implementations.cma_implementation.cma_es_constants import ComputePositiveWeights, ComputeMuEff, ComputeNegativeWeights, ComputeSigmaConstants, ComputeCConstants, ComputeChi

class CMAParameters:
    """
    Constants:
    (lambda -> lambd): Number of sampled points.
    (mu): Number of positively rewarded points.
    (rho = lambda - mu): Uses worst points to actively shrink bad directions.

    (CS1 = 1 - c_sigma): Precomputed constant for p_sigma update.
    (CS2 = sqrt(c_sigma * (2 - c_sigma) * mu_eff)): Precomputed constant for p_sigma update.
    (CS3 = c_sigma / d_sigma): Precomputed constant for sigma update.

    (CC1 = 1 - c_c): Precomputed constant for p_C update.
    (CC2 = sqrt(c_c * (2 - c_c) * mu_eff)): Precomputed constant for p_C update.
    (CC3 = 1 - c_1 - c_mu * sum_w)

    Used Constants in the Update:
    lambd
    mu
    rho
    w_pos
    w_neg
    CS1
    CS2
    CS3
    c_1
    c_mu
    CC1
    CC2
    CC3
    chi_n
    """
    def __init__(self, n: int, lambd: int, alpha_cov: float = 2.0):
        self.mu: int = max(1, lambd // 2)
        self.rho: int = lambd - self.mu
        self.w_pos: np.ndarray = ComputePositiveWeights(self.mu, lambd)
        mu_eff = ComputeMuEff(self.w_pos)

        c_sigma, d_sigma = ComputeSigmaConstants(n, mu_eff)
        self.CS1: float = 1 - c_sigma
        self.CS2: float = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
        self.CS3: float = c_sigma / d_sigma
        c_c, self.c_1, self.c_mu = ComputeCConstants(n, mu_eff, alpha_cov)
        
        self.K = np.sqrt((c_c * (2 - c_sigma)) / (c_sigma * (2 - c_c)))

        self.w_neg: np.ndarray = ComputeNegativeWeights(self.mu, lambd, mu_eff, self.c_1, self.c_mu, n)
        w_sum: float = np.sum(self.w_pos) + np.sum(self.w_neg)

        self.CC1 = 1 - c_c
        self.CC2 = np.sqrt(c_c * (2 - c_c) * mu_eff)
        self.CC3 = 1 - self.c_1 - self.c_mu * w_sum
        self.chi_n = ComputeChi(n)
