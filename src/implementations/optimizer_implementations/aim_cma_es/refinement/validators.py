import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type
from scipy.spatial.distance import cdist

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinementParameters

from src.implementations.optimizer_implementations.aim_cma_es.refinement.local_region import LocalRegion


class IValidator(ABC):
    def __init__(self, refinement_parameters: RefinementParameters, x_count: int, f_count: int):
        self.refinement_parameters = refinement_parameters
        self.x_count = x_count
        self.f_count = f_count
        
    @abstractmethod
    def validate(self, candidates: np.ndarray, region: LocalRegion, archive: InternalArchive) -> np.ndarray:
        pass


class NoOpValidator(IValidator):
    """Pass-through for testing without validation."""
    def validate(self, candidates: np.ndarray, region: LocalRegion, archive: InternalArchive) -> np.ndarray:
        return candidates
