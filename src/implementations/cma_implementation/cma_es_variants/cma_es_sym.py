import numpy as np
import threading
import json
import os
from typing import List, Tuple

from src.implementations.cma_implementation.cma_es_elite.cma_es_elite import CMAES_Elite_Kernel
from src.implementations.cma_implementation.cma_es_elite.cma_es_elite_parameters import ParameterManager_CMA_Elite


class CMAES_Sym_Kernel(CMAES_Elite_Kernel):
    """
    Choosable c_m, with a default of c_m = 1.7.
    Generates symmetric distribution.
    """
    def __init__(self, n: int, lambd: int, m0: np.ndarray, sigma0: float, C0: np.ndarray, max_gen: int, ftol: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, c_m: float = 1.7):
        if (lambd % 2 != 0):
            lambd += 1
        
        super().__init__(n = n, lambd = lambd, m0= m0, sigma0=sigma0, C0=C0, max_gen=max_gen, ftol=ftol, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.sample_size = lambd // 2
        self.c_m = c_m

    def compute_y(self) -> np.ndarray:
        # Y = B * D * Z
        z = np.random.standard_normal((self.sample_size, self.n))
        z = np.concatenate((z, -z), axis=0)
        return (self.B @ (self.D_diag[:, None] * z.T)).T

    def update_m(self, y_w: np.ndarray) -> None:
        self.m += self.c_m * self.sigma * y_w
