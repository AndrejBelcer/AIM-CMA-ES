import numpy as np
import threading
import json
import os
from typing import List, Tuple
from abc import ABC, abstractmethod

from src.implementations.cma_implementation.cma_es_parameters import CMAParameters

class CMAES_Base(ABC):
    def __init__(self, n: int, lambd: int, m0: np.ndarray, sigma0: float, C0: np.ndarray, max_gen: int, ftol: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        # Constants
        self.n = n
        self.lambd = lambd

        # Bound Handling
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.width = self.ub - self.lb
        self.two_width = 2.0 * self.width
        self.max_width_criteria = np.max(self.width) * 0.5
        
        # State
        self.m = m0.copy()
        self.sigma = sigma0
        self.C = C0.copy()
        
        # Evolution paths
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        
        # Matrix Decomposition
        self.C, self.B, self.D_diag, self.C_inv_half = self.decompose(self.C)


        # Control & Stagnation Tracking
        self.generation = 0
        self.max_gen = max_gen
        self.ftol = ftol
        self.stopped = False
        self.stop_string = "Active"

        self.params: CMAParameters = None
        
    
    def ask(self) -> np.ndarray:
        """Samples lambda points, handles reflection, and stores y_k."""
        self.generation += 1
        
        y = self.compute_y()
        x = self.m + self.sigma * y
        
        # Bound Reflection
        out_mask = (x < self.lb) | (x > self.ub)
        if np.any(out_mask):
            x_shifted = x - self.lb
            x_mod = np.abs(x_shifted) % self.two_width
            x_folded = np.minimum(x_mod, self.two_width - x_mod)
            x_fixed = x_folded + self.lb
            
            x = np.where(out_mask, x_fixed, x)
            # Re-calculate y to maintain the actual step taken
            y = (x - self.m) / self.sigma
            
        self._current_y = y 
        self._current_x = x
        return x

    @abstractmethod
    def tell(self, scores: np.ndarray) -> None:
        pass


    def compute_y(self) -> np.ndarray:
        # Y = B * D * Z
        z = np.random.standard_normal((self.lambd, self.n))
        return (self.B @ (self.D_diag[:, None] * z.T)).T


    def update_m(self, y_w: np.ndarray) -> None:
        self.m += self.sigma * y_w 


    def update_sigma(self, y_w: np.ndarray) -> None:
        self.ps = self.params.CS1 * self.ps + self.params.CS2 * (self.C_inv_half @ y_w)
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp(self.params.CS3 * ((ps_norm / self.params.chi_n) - 1))


    def update_C(self, y_w: np.ndarray, y_mu: np.ndarray, y_rho: np.ndarray) -> None:
        self.pc = self.params.CC1 * self.pc + self.params.CC2 * y_w
        rank_1 = self.params.c_1 * np.outer(self.pc, self.pc)
        
        rank_mu_pos = self.params.c_mu * (y_mu.T @ (self.params.w_pos[:, None] * y_mu))
        
        # Active CMA: Mahalanobis distance for negative updates
        z_rho = (self.C_inv_half @ y_rho.T).T
        dist_sq = np.maximum(np.sum(z_rho**2, axis=1), 1e-12)
        w_circ = self.params.w_neg * (self.n / dist_sq)
        
        rank_mu_neg = self.params.c_mu * (y_rho.T @ (w_circ[:, None] * y_rho))
        
        self.C = self.params.CC3 * self.C + rank_1 + rank_mu_pos + rank_mu_neg
        self.C = (self.C + self.C.T) * 0.5

    def decompose(self, C: np.ndarray, eigenvalue_floor:float = 1e-18) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vals, B = np.linalg.eigh(C)

        clipped = vals < eigenvalue_floor
        if np.any(clipped):
            # Apply the same eigenvalue repair to the covariance.
            B_clipped = B[:, clipped]
            increases = eigenvalue_floor - vals[clipped]

            C += (B_clipped * increases) @ B_clipped.T
            C = 0.5 * (C + C.T)

            vals[clipped] = eigenvalue_floor        

        D_diag = np.sqrt(vals)
        C_inv_half = (B * (1.0 / D_diag)) @ B.T

        return C, B, D_diag, C_inv_half
    

    def check_stop(self, pool_scores: np.ndarray) -> bool:
        """
        Default termination check.
        For derived classes, check_subclass_stop should be modified.
        """
        if (self.check_generation()):
            return True
        elif (self.check_condition_cov()):
            return True
        elif (self.check_tol_x_min()):
            return True
        elif (self.check_tol_x_max()):
            return True
        elif (self.check_subclass_stop(pool_scores)):
            return True
        else:
            return False
        
    def check_generation(self) -> bool:
        if (self.generation >= self.max_gen):
            self.stopped = True
            self.stop_string = "Ran out of allowed steps."
            return True
        else:
            return False
    
    def check_condition_cov(self) -> bool:
        if (self.D_diag[-1] > 1e7 * self.D_diag[0]):
            self.stopped = True
            self.stop_string = "Covariance matrix ill-conditioned."
            return True
        else:
            return False
        
    def check_tol_x_min(self) -> bool:
        if (self.sigma * self.D_diag[-1] < 1e-11):
            self.stopped = True
            self.stop_string = "Step size effectively zero (TolX)."
            return True
        else:
            return False
        
    def check_tol_x_max(self) -> bool:
        if (self.sigma * self.D_diag[-1] > self.max_width_criteria):
            self.stopped = True
            self.stop_string = "Diverged: Step size exceeds search space criteria."
            return True
        else:
            return False
        
    def check_subclass_stop(self, pool_scores: np.ndarray) -> bool:
        """
        The derived class specific termination options.
        """
        return False


    def warm_start_path(self, v_norm: np.ndarray, path_trust: float) -> None:
        """
        Injects kinematic momentum into the evolution paths.
        path_trust in [0, 1] scales the confidence. 1.0 = perfect neutrality (no sigma decay).
        """
        if path_trust <= 0.0 or np.linalg.norm(v_norm) < 1e-12:
            return
            
        # 1. Project physical velocity into isotropic space
        z = self.C_inv_half @ v_norm
        z_norm = np.linalg.norm(z)
        
        if z_norm < 1e-12:
            return
            
        # 2. Fetch constants
        chi_N = self.params.chi_n
        K = self.params.K
        
        # 3. Inject momentum
        self.ps = path_trust * chi_N * (z / z_norm)
        self.pc = path_trust * (chi_N / (K * z_norm)) * v_norm
