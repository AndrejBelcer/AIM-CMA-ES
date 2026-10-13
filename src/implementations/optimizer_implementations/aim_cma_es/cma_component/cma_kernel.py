import numpy as np
import cma
from cma import CMAEvolutionStrategy
from typing import List, Tuple, Optional, Union, Dict, Any

from abc import ABC, abstractmethod

from src.implementations.cma_implementation.cma_es_base import CMAES_Base

from src.implementations.cma_implementation.cma_es_normal.cma_es import CMAES_Kernel
from src.implementations.cma_implementation.cma_es_elite.cma_es_elite import CMAES_Elite_Kernel
from src.implementations.cma_implementation.cma_es_variants.cma_es_leapfrog import CMAES_Leapfrog_Kernel
from src.implementations.cma_implementation.cma_es_variants.cma_es_sym import CMAES_Sym_Kernel


    
class Kernel(ABC):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        pass
        
    @abstractmethod
    def ask(self) -> np.ndarray:
        pass

    @abstractmethod
    def tell(self, x_norm: np.ndarray, s_values: np.ndarray):
        pass

    @abstractmethod
    def is_converged(self) -> bool:
        pass

    @abstractmethod
    def get_stop_reason(self) -> str:
        pass
    
    @abstractmethod
    def get_distribution_data(self) -> Dict:
        pass

    @abstractmethod
    def get_incumbent(self) -> Optional[np.ndarray]:
        pass

class KernelPyCMA(Kernel):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, inopts: Dict):
        self.dim = len(x0)
        self._has_crashed = False
        self._crash_reason = ""

        # Keep fallbacks in case of immediate or subsequent crashes
        self.last_sigma = sigma
        self.last_C = C.copy() if C is not None else np.eye(self.dim)
        self.last_favorite = x0.copy()

        self.es: CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0, sigma, inopts = inopts)
        
        # Inject Covariance
        det = np.linalg.det(C)
        if det > 1e-16:
            # Normalize determinant to 1
            C_norm = C / (det ** (1/self.dim))
            self.es.sm.C = C_norm
            self.es.sm.update_now(lazy_update_gap=0)

    def ask(self) -> np.ndarray:
        if self._has_crashed:
            return np.empty((0, self.dim))
        try:
            candidates = np.array(self.es.ask())
            return candidates
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during ask: {str(e)}"
            return np.empty((0, self.dim))

    def tell(self, x_norm: np.ndarray, s_values: np.ndarray):
        if self._has_crashed:
            return
        try:
            self.es.tell(x_norm, s_values, check_points=True)
            # Update fallbacks on success
            self.last_sigma = self.es.sigma
            self.last_C = self.es.C.copy()
            if self.es.result and self.es.result.xfavorite is not None:
                self.last_favorite = self.es.result.xfavorite.copy()
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during tell: {str(e)}"

    def is_converged(self) -> bool:
        if self._has_crashed:
            return True
        try:
            return bool(self.es.stop())
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during convergence check: {str(e)}"
            return True

    def get_stop_reason(self) -> str:
        if self._has_crashed:
            return f"Crashed ({self._crash_reason})"
        try:
            stop_dict = self.es.stop()
            if not stop_dict:
                return "Active"
            return ", ".join(stop_dict.keys())
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during stop reason check: {str(e)}"
            return f"Crashed ({self._crash_reason})"
    
    def get_distribution_data(self) -> Dict:
        if self._has_crashed:
            return {
                'sigma': self.last_sigma,
                'C': self.last_C
            }
        try:
            return {
                'sigma': self.es.sigma,
                'C': self.es.C.copy()
            }
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during distribution retrieval: {str(e)}"
            return {
                'sigma': self.last_sigma,
                'C': self.last_C
            }

    def get_incumbent(self) -> Optional[np.ndarray]:
        if self._has_crashed:
            return self.last_favorite
        try:
            return self.es.result.xfavorite
        except Exception as e:
            self._has_crashed = True
            self._crash_reason = f"Crash during incumbent retrieval: {str(e)}"
            return self.last_favorite



class KernelOriginalPure(KernelPyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        inopts: Dict = {
            'popsize': popsize,
            'maxiter': max_iter,
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'BoundaryHandler': cma.BoundTransform,
            'verbose': -9,
            'CMA_diagonal': False,
            'CMA_active': True,
            'tolfun': tolfun,
            'CMA_elitist': False,
        }
        
        super().__init__(x0 = x0, sigma = sigma, C = C, inopts = inopts)
        

class KernelOriginalElite(KernelPyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        inopts: Dict = {
            'popsize': popsize,
            'maxiter': max_iter,
            'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
            'BoundaryHandler': cma.BoundTransform,
            'verbose': -9,
            'CMA_diagonal': False,
            'CMA_active': True,
            'tolfun': tolfun,
            'CMA_elitist': True
        }
        
        super().__init__(x0 = x0, sigma = sigma, C = C, inopts = inopts)
        
        
class KernelMyCMA(Kernel):
    def __init__(self, cma_es: CMAES_Base, v_norm: np.ndarray, path_trust: float):
        self.es: CMAES_Base = cma_es
        self.es.warm_start_path(v_norm, path_trust)

    def ask(self) -> np.ndarray:
        candidates = self.es.ask()
        return candidates

    def tell(self, x_norm: np.ndarray, s_values: np.ndarray):
        self.es.tell(s_values)

    def is_converged(self) -> bool:
        return self.es.stopped

    def get_stop_reason(self) -> str:
        if not self.es.stopped:
            return "Active"
        else:
            return self.es.stop_string
        
    def get_distribution_data(self) -> Dict:
        return {
            'sigma': self.es.sigma,
            'C': self.es.C.copy()
        }

    def get_incumbent(self) -> Optional[np.ndarray]:
        return self.es.m
        

class KernelNormal(KernelMyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        super().__init__(cma_es = CMAES_Kernel(len(x0), popsize, x0, sigma, C, max_iter, tolfun, lower_bounds, upper_bounds), v_norm = v_norm, path_trust = path_trust)

class KernelElite(KernelMyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        super().__init__(cma_es = CMAES_Elite_Kernel(len(x0), popsize, x0, sigma, C, max_iter, tolfun, lower_bounds, upper_bounds), v_norm = v_norm, path_trust = path_trust)
        
class KernelLeapfrog(KernelMyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        super().__init__(cma_es = CMAES_Leapfrog_Kernel(len(x0), popsize, x0, sigma, C, max_iter, tolfun, lower_bounds, upper_bounds), v_norm = v_norm, path_trust = path_trust)
        
class KernelSym(KernelMyCMA):
    def __init__(self, x0: np.ndarray, sigma: float, C: np.ndarray, popsize: int, max_iter: int, tolfun: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, v_norm: np.ndarray, path_trust: float):
        super().__init__(cma_es = CMAES_Sym_Kernel(len(x0), popsize, x0, sigma, C, max_iter, tolfun, lower_bounds, upper_bounds), v_norm = v_norm, path_trust = path_trust)
