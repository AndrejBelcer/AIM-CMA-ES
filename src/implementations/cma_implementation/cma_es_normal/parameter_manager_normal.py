import numpy as np
import scipy.special as sp
from typing import Dict, Tuple

from src.implementations.cma_implementation.cma_es_parameters import CMAParameters

class ParameterManager_CMA:
    """
    Pre-computes and caches all CMA-ES constants based on n and lambda.
    Follows the 2016 default parameter settings (Table 1 in Tutorial).
    """
    _cache: Dict[Tuple[int, int], CMAParameters] = {}

    @classmethod
    def get_constants(cls, n: int, lambd: int) -> CMAParameters:
        key = (n, lambd)
        if key not in cls._cache:
            cls._cache[key] = CMAParameters(n, lambd)
        return cls._cache[key]
