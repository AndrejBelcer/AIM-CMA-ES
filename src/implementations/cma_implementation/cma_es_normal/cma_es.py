import numpy as np
import threading
import json
import os

from src.implementations.cma_implementation.cma_es_base import CMAES_Base
from src.implementations.cma_implementation.cma_es_normal.parameter_manager_normal import ParameterManager_CMA
from src.implementations.cma_implementation.cma_es_parameters import CMAParameters


class CMAES_Kernel(CMAES_Base):
    """
    CMA-ES Sampler: High-Performance Burst-Mode Implementation

    Deviations from standard CMA-ES (Hansen Tutorial 2016):
    1. Heaviside Removal: Removed `h_sigma` safety brake (h_sigma hardcoded to 1). Warm-started burst kernels are treated as mature; we want aggressive, continuous path updates.
    2. Hardcoded c_m = 1: Moves the mean exactly to the weighted center of successful steps.
    3. No lazy updates - always computes the whole update.
    4. Reflection Bounds: Uses reflections bounds for bound constraints. Re-calculates `y` for reflected points to preserve gradient integrity.
    5. Symmetry: Enforces C = (C + C.T) * 0.5 to prevent float drift.
    6. Custom stop conditions.
    """
    def __init__(self, n: int, lambd: int, m0: np.ndarray, sigma0: float, C0: np.ndarray, max_gen: int, ftol: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        super().__init__(n = n, lambd = lambd, m0= m0, sigma0=sigma0, C0=C0, max_gen=max_gen, ftol=ftol, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.params: CMAParameters = ParameterManager_CMA.get_constants(n, lambd)

        # Stagnation Parameters:
        self.best_score = np.inf
        self.stagnation_count = 0
        self.stagnation_max_count = 7 + n // 3

    def tell(self, scores: np.ndarray) -> None:
        """Updates the distribution using epsilon-elitist selection."""
        if self.stopped:
            return
         
        # 1. Sort the pool directly from the current state
        sort_idx = np.argsort(scores)
        ###pool_x = self._current_x[sort_idx]
        pool_y = self._current_y[sort_idx]
        pool_scores = scores[sort_idx]

        # 2. Termination Check
        if self.check_stop(pool_scores):
            return
        
        # 3. Extract points for updates
        y_mu = pool_y[:self.params.mu]
        y_rho = pool_y[-self.params.rho:]
        
        # 4. Compute y_w and Update Mean
        y_w = np.dot(self.params.w_pos, y_mu)
        
        # 5. Updates
        self.update_m(y_w)
        self.update_sigma(y_w)
        self.update_C(y_w, y_mu, y_rho)

        # 6. Finalize Matrix State
        self.C, self.B, self.D_diag, self.C_inv_half = self.decompose(self.C)


    def check_subclass_stop(self, pool_scores: np.ndarray) -> bool:
        # Stagnation Check
        current_best = pool_scores[0]
        if current_best >= self.best_score:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.best_score = current_best
            
        if self.stagnation_count >= self.stagnation_max_count:
            self.stopped = True
            self.stop_string = f"Best solution stagnated for {self.stagnation_max_count} generations."
            return True
            
        # Tolerance Check: Higher vs Lowest comparison (within the "good" mu pool + 1)
        f_range = pool_scores[self.params.mu + 1] - pool_scores[0]
        if f_range < self.ftol:
            self.stopped = True
            self.stop_string = "Kernel converged."
            return True
                
        return False
