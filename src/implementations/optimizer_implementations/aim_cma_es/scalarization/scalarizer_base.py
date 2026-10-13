import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from src.implementations.optimizer_implementations.aim_cma_es.enums import CVType, ScalarizationMode
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.constraint_violation import ConstraintViolation
from src.implementations.optimizer_implementations.aim_cma_es.state_tracker import StateTracker
from src.utilities.indicators.candidate_indicator import CandidateIndicator

class Scalarizer(ABC):
    """
    Abstract Base Class for Scalarization Strategies.
    
    Role:
    - Holds the frozen state of the optimization (bounds, front, etc.) to ensure 
      that the fitness landscape remains stationary during a single CMA-ES kernel run.
    - Handles the normalization of raw objectives into the [0, 1] space.
    - Manages constraint violation (CV) calculations.
    """
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, scalarization_mode: ScalarizationMode):
        self.f_L = state.f_L
        self.f_D = state.f_D
        self.g_divisor = g_divisor
        self.f_count = len(self.f_L)            
        self.cv_indicator = ConstraintViolation(cv_type)
        self.mode: ScalarizationMode = scalarization_mode
        
        self.indicator: Optional[CandidateIndicator] = None # Can be set by the derived class

    def normalize(self, f_raw: np.ndarray) -> np.ndarray:
        """Transforms raw objectives to normalized [0,1] space (relative to frozen bounds)."""
        return (f_raw - self.f_L) / self.f_D

    def get_cv(self, g_raw: np.ndarray) -> np.ndarray:
        """Computes Constraint Violation using the configured strategy."""
        return self.cv_indicator.compute(g_raw, self.g_divisor)

    @abstractmethod
    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        """
        Computes the scalar fitness s(x) for the batch.
        Minimization: Lower is better.
        """
        pass

    @abstractmethod
    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        """
        Used for the warm starts.
        Computes the score, with no feasibility penalty.
        For the scoring, all points are evaluated as if dominated.
        """
        pass
    
    @abstractmethod
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        """
        Used for the warm starts.
        For the scoring, all points are evaluated as if dominated.
        Computes the score with feasibility penalty.
        """
        pass
    
    @abstractmethod
    def get_string_description(self) -> str:
        """
        Returns the type of goal the current scalarization is chasing.
        """
        pass
