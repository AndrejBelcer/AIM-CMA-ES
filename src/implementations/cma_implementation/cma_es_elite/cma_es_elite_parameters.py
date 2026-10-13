import numpy as np
import scipy.special as sp
from typing import Dict, Tuple

from src.implementations.cma_implementation.cma_es_parameters import CMAParameters


class ParameterManager_CMA_Elite:
    """
    Pre-computes and caches all CMA-ES constants based on n and lambda.
    Follows the 2016 default parameter settings (Table 1 in Tutorial).
    """
    _cache: Dict[Tuple[int, int], CMAParametersElite] = {}

    @classmethod
    def get_constants(cls, n: int, lambd: int) -> CMAParametersElite:
        key = (n, lambd)
        if key not in cls._cache:
            cls._cache[key] = CMAParametersElite(n, lambd)
        return cls._cache[key]



class CMAParametersElite(CMAParameters):
    def __init__(self, n: int, lambd: int, alpha_cov: float = 2.0):
        super().__init__(n = n, lambd=lambd, alpha_cov=alpha_cov)
        self.epsilon: int = max(1, self.mu // 2)

        

