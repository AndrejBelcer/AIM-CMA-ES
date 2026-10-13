import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.scalarizer_base import Scalarizer
from src.implementations.optimizer_implementations.aim_cma_es.enums import ScalarizationMode, CMA_Type
from src.implementations.optimizer_implementations.aim_cma_es.cma_component.cma_kernel import Kernel, KernelOriginalPure, KernelOriginalElite, KernelNormal, KernelElite, KernelLeapfrog, KernelSym
from src.implementations.optimizer_implementations.aim_cma_es.cma_component.cma_warm_starter import CmaWarmStartManager
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import CMAParameters

class CmaManager:
    """
    Manages the lifecycle of a single CMA-ES kernel.
    Handles data normalization (ROI-centric) and Warm Start orchestration.
    """
    def __init__(self, 
                 x_count: int, 
                 true_lower_bounds_normalized: np.ndarray,
                 true_upper_bounds_normalized: np.ndarray,
                 cma_parameters: CMAParameters
                ) -> None:
        
        self.x_count = x_count
        

        # True Bounds projected into the ROI-normalized space
        self.cma_lb = true_lower_bounds_normalized
        self.cma_ub = true_upper_bounds_normalized
        
        # CMA Parameters
        self.popsize: int = cma_parameters.popsize if cma_parameters.popsize else 4 + int(round(3 * np.log(x_count)))
        self.max_iter: int = np.ceil(cma_parameters.max_iter_const + np.sqrt(x_count) * cma_parameters.max_iter_per_sqrt_dim)
        self.f_tolerance: float = cma_parameters.convergence_tolerance
        self.cma_type = cma_parameters.cma_type
        self.global_exploration_sigma = cma_parameters.warm_start_parameters.global_exploration_sigma
        
        # O(1) Kernel Mapping
        self.kernel_map = {
            CMA_Type.CMA_ORIGINAL_PURE: KernelOriginalPure,
            CMA_Type.CMA_ORIGINAL_ELITE: KernelOriginalElite,
            CMA_Type.CMA_NORMAL: KernelNormal,
            CMA_Type.CMA_ELITE: KernelElite,
            CMA_Type.CMA_LEAPFROG: KernelLeapfrog,
            CMA_Type.CMA_SYM: KernelSym,
        }
        
        # State
        self.es_kernel: Optional[Kernel] = None
        self.active_scalarizer: Optional[Scalarizer] = None
        self.feasibility_runs: int = 3 # Starting feasibility run population increase factor - increases further with runs.
        
        # Initialize the CmaWarmStartManager
        self.warm_starter = CmaWarmStartManager(x_count=self.x_count, true_lower_bounds_normalized=true_lower_bounds_normalized, true_upper_bounds_normalized=true_upper_bounds_normalized, warm_start_parameters=cma_parameters.warm_start_parameters)


    def is_stop(self) -> bool:
        if self.es_kernel is None:
            return True
        return self.es_kernel.is_converged()
    
    
    def is_active(self) -> bool:
        return not self.is_stop()


    def deactivate(self):
        self.es_kernel = None
        self.active_scalarizer = None


    def get_best_point(self) -> Optional[np.ndarray]:
        """
        Returns the best point (xfavorite) in normalized space.
        """
        if self.es_kernel is None:
            return None
        x_norm_best = self.es_kernel.get_incumbent()
        if x_norm_best is not None:
            return x_norm_best
        return None

    def InitKernel(self, archive: InternalArchive, f_U: np.ndarray, scalarizer: Scalarizer, points_of_interest: List[np.ndarray]):
        self.active_scalarizer = scalarizer
        kernel_class = self.kernel_map[self.cma_type]
        used_popsize = self.popsize
        
        x_0, sigma, C, v_norm, path_trust = self.warm_starter.GetWarmStartValues(archive, f_U, scalarizer, points_of_interest)
        if self.active_scalarizer.mode == ScalarizationMode.FEASIBILITY:
            self.feasibility_runs += 1
            used_popsize = int(round(self.popsize * (1 + 0.5 * self.feasibility_runs)))  
        elif (self.active_scalarizer.mode == ScalarizationMode.EXP):
            used_popsize = 6 + 2 * self.x_count
            sigma = self.global_exploration_sigma

        self.es_kernel = kernel_class(x0 = x_0, sigma = sigma, C = C, popsize = used_popsize, max_iter = self.max_iter, tolfun = self.f_tolerance, lower_bounds = self.cma_lb, upper_bounds = self.cma_ub, v_norm = v_norm, path_trust =  path_trust)
        
    def ask(self) -> np.ndarray:
        """
        Returns candidates in normalized space.
        """
        return self.es_kernel.ask()

    def tell(self, x_norm: np.ndarray, f_raw: np.ndarray, g_raw: np.ndarray):
        """
        Accepts normalized space coordinates and updates kernel.
        """
        if self.es_kernel is None: return

        s_values = self.active_scalarizer.evaluate(f_raw, g_raw)
        
        self.es_kernel.tell(x_norm, s_values)
