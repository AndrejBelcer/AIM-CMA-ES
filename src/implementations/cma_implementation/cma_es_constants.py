import numpy as np
import scipy.special as sp
from typing import Dict, Tuple

def ComputePositiveWeights(mu: int, lambd: int) -> np.ndarray:
    # (Eq 49 & 53)
    # Raw weights: log((lambda + 1) / 2) - log(i)
    i_pos = np.arange(1, mu + 1)
    w_pos_raw = np.log((lambd + 1) / 2.0) - np.log(i_pos)
    return w_pos_raw / np.sum(w_pos_raw)

def ComputeMuEff(w_pos: np.ndarray) -> float:
    # (Eq 8)
    mu_eff = 1.0 / np.sum(w_pos**2)
    return mu_eff

def ComputeNegativeWeights(mu: int, lambd: int, mu_eff: float, c_1: float, c_mu: float, n: int) -> np.ndarray:
    # (Eq 49)
    # i goes from mu + 1 to lambda
    i_neg = np.arange(mu + 1, lambd + 1)
    w_neg_raw = np.log((lambd + 1) / 2.0) - np.log(i_neg)
    
    # (Eq 50-52)
    # Calculate bounds to guarantee positive definiteness
    sum_abs_w_neg_raw = np.sum(np.abs(w_neg_raw))
    mu_eff_neg = sum_abs_w_neg_raw**2 / np.sum(w_neg_raw**2)
    
    alpha_mu_minus = 1 + c_1 / c_mu
    alpha_mueff_minus = 1 + 2 * mu_eff_neg / (mu_eff + 2)
    alpha_posdef_minus = (1 - c_1 - c_mu) / (n * c_mu)
    
    # (Eq 53)
    alpha_min = min(alpha_mu_minus, alpha_mueff_minus, alpha_posdef_minus)
    
    w_neg = w_neg_raw * (alpha_min / sum_abs_w_neg_raw)
    return w_neg

def ComputeSigmaConstants(n: int, mu_eff: float) -> Tuple[float, float]:
    """
    Returns: c_sigma, d_sigma
    """
    # (Eq 55)
    # Learning Rate
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    # Damping for step-size update
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma

    return c_sigma, d_sigma

def ComputeCConstants(n: int, mu_eff: float, alpha_cov: float) -> Tuple[float, float, float]:
    """
    Returns: c_c, c_1, c_mu
    """
    # (Eq 55-58)
    # Learning Rate
    c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
    # Rank 1 update
    c_1 = alpha_cov / ((n + 1.3)**2 + mu_eff)
    # Rank mu update
    c_mu = min(1 - c_1, alpha_cov * (0.25 + mu_eff + 1/mu_eff - 2) / ((n + 2)**2 + alpha_cov * mu_eff / 2))

    return c_c, c_1, c_mu

def ComputeChi(n: int) -> float:
    """
    Computes the expected length of a standard normally distributed vector in n dimensions.
    Formula: E||N(0,I)|| = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
    Approximation (Puiseux Series):
    √(n) * (1 - 1/(4*n) + 1/(32*n**2))
    """
    # Expectation of ||N(0,I)||
    result: float = 0.0
    
    if (n < 100):
        result = np.sqrt(2) * sp.gamma((n + 1) * 0.5) / sp.gamma(n * 0.5)
    else:
        result = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
    
    return result
