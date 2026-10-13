import numpy as np
import threading
import json
import os
from typing import List, Tuple

from src.implementations.cma_implementation.cma_es_base import CMAES_Base
from src.implementations.cma_implementation.cma_es_elite.cma_es_elite_parameters import ParameterManager_CMA_Elite, CMAParametersElite
from src.implementations.cma_implementation.cma_es_parameters import CMAParameters

class CMAES_Elite_Kernel(CMAES_Base):
    """
    CMA-ES Sampler: High-Performance Elitist Burst-Mode Implementation

    Deviations from standard CMA-ES (Hansen Tutorial 2016):
    1. Heaviside Removal: Removed `h_sigma` safety brake (h_sigma hardcoded to 1). Warm-started burst kernels are treated as mature; we want aggressive, continuous path updates.
    2. Hardcoded c_m = 1: Moves the mean exactly to the weighted center of successful steps.
    3. No lazy updates - always computes the whole update.
    4. Reflection Bounds: Uses reflections bounds for bound constraints. Re-calculates `y` for reflected points to preserve gradient integrity.
    5. Symmetry: Enforces C = (C + C.T) * 0.5 to prevent float drift.
    6. Custom stop conditions.
    7. Elitist: Keeps best points so far. Rewards [mu] points, penalizes [rho] points, lambda = mu + rho.
    On the first step, the number of samples is only [lambda].
    During the run (after the first step), the points ranked are of size: [epsilon + lambda], influencing the covariance as groups of sizes: [mu, epsilon, rho].
    """
    def __init__(self, n: int, lambd: int, m0: np.ndarray, sigma0: float, C0: np.ndarray, max_gen: int, ftol: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        super().__init__(n = n, lambd = lambd, m0= m0, sigma0=sigma0, C0=C0, max_gen=max_gen, ftol=ftol, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.params: CMAParameters = ParameterManager_CMA_Elite.get_constants(n, lambd)

        # Elites Pool:
        self.elites_x = np.empty((0, n))
        self.elites_y = np.empty((0, n))
        self.elites_scores = np.empty(0)
        
        # Stagnation Parameters:
        self.best_score: float = np.inf
        self.stagnation_max_count: int = max(2, max_gen // 6)
        self.stagnation_count: int = 0

        self.target: float = np.inf
        self.premature_max_count: int = max(2, max_gen // 6)
        self.premature_count: int = 0


    def tell(self, scores: np.ndarray) -> None:
        """Updates the distribution using epsilon-elitist selection."""
        if self.stopped:
            return
        
        # 1. Prepare Selection Pool
        pool_x = np.concatenate((self._current_x, self.elites_x), axis=0)
        pool_y = np.concatenate((self._current_y, self.elites_y), axis=0)
        pool_scores = np.concatenate((scores, self.elites_scores), axis=0)
        
        # 2. Sort the pool
        sort_idx = np.argsort(pool_scores)
        pool_x = pool_x[sort_idx]
        pool_y = pool_y[sort_idx]
        pool_scores = pool_scores[sort_idx]

        # Update elites immediately so they are available even if we terminate this step
        self.elites_x = pool_x[:self.params.epsilon]
        self.elites_y = pool_y[:self.params.epsilon]
        self.elites_scores = pool_scores[:self.params.epsilon]
        
        # 3. Termination Check
        if self.check_stop(pool_scores):
            return
        
        # 4. Extract points for updates
        y_mu = pool_y[:self.params.mu]
        y_rho = pool_y[-self.params.rho:]
        
        # 5. Compute y_w and Update Mean
        y_w = np.dot(self.params.w_pos, y_mu)
        
        # 6. Updates
        self.update_m(y_w)
        self.update_sigma(y_w)
        self.update_C(y_w, y_mu, y_rho)

        # 7. Finalize Matrix State
        self.C, self.B, self.D_diag, self.C_inv_half = self.decompose(self.C)
        
        # 8. Elite Maintenance: Fully Vectorized Re-anchoring
        self.elites_y = (self.elites_x - self.m) / self.sigma

    def check_subclass_stop(self, pool_scores: np.ndarray) -> bool:
        """Evaluates termination criteria based on the sorted pool scores."""
        current_best = pool_scores[0]
        
        # 1. Stagnation Check
        if current_best >= self.best_score:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.best_score = current_best
            
        if self.stagnation_count > self.stagnation_max_count:
            self.stopped = True
            self.stop_string = f"Best solution stagnated for {self.stagnation_max_count} generations."
            return True

        # 2. Tolerance Check: Higher vs Lowest comparison (within the "good" mu pool + 1)
        f_range = pool_scores[self.params.mu + 1] - current_best
        if f_range < self.ftol:
            self.stopped = True
            self.stop_string = "Kernel converged."
            return True
        
        # 3. Premature Concergence Check
        if current_best > 0:
            if (current_best < self.target):
                self.target = max(0, current_best * 0.99 - 0.001)
                self.premature_count = 0
            else:
                self.premature_count += 1
                if(self.premature_count > self.premature_max_count):
                    self.stop_string = "Kernel is converging in positive valued region."
                    self.stopped = True
                    return True

        return False
