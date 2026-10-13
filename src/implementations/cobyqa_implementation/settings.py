import numpy as np
from typing import Optional
from enum import Enum, auto

class OptimizerState(Enum):
    """
    State machine for the COBYQA Ask-and-Tell interface.
    """
    INIT = auto()           # Generating initial NPT points
    TR_STEP = auto()        # Standard Trust-Region step
    SOC_STEP = auto()       # Second-Order Correction step
    GEOMETRY_STEP = auto()  # Geometry-improving step
    DONE = auto()           # Optimization finished


class COBYQAParameters:
    """
    Hyperparameters and internal constants for the COBYQA solver.
    """
    def __init__(self,
                 rhobeg: float = 1.0,
                 rhoend: float = 1e-6,
                 npt: Optional[int] = None,
                 max_iter: int = 10000,
                 feasibility_tol: float = 1e-8,
                 
                 # Trust Region Radius Update Factors
                 decrease_radius_factor: float = 0.5,
                 increase_radius_factor: float = 1.41421356, # sqrt(2)
                 increase_radius_threshold: float = 2.0,
                 decrease_radius_threshold: float = 1.4,
                 
                 # Resolution Update Factors
                 decrease_resolution_factor: float = 0.1,
                 large_resolution_threshold: float = 250.0,
                 moderate_resolution_threshold: float = 16.0,
                 
                 # Reduction Ratio Thresholds
                 low_ratio: float = 0.1,
                 high_ratio: float = 0.7,
                 very_low_ratio: float = 0.01,
                 
                 # Penalty Parameters
                 penalty_increase_threshold: float = 1.5,
                 penalty_increase_factor: float = 2.0,
                 
                 # Step and Geometry Thresholds
                 short_step_threshold: float = 0.5,
                 low_radius_factor: float = 0.1,
                 byrd_omojokun_factor: float = 0.8,
                 threshold_ratio_constraints: float = 2.0,
                 large_shift_factor: float = 10.0,
                 large_gradient_factor: float = 10.0,
                 resolution_factor: float = 2.0,
                 
                 # Algorithmic toggles
                 improve_tcg: bool = True):
        
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.npt = npt
        self.max_iter = max_iter
        self.feasibility_tol = feasibility_tol
        
        self.decrease_radius_factor = decrease_radius_factor
        self.increase_radius_factor = increase_radius_factor
        self.increase_radius_threshold = increase_radius_threshold
        self.decrease_radius_threshold = decrease_radius_threshold
        
        self.decrease_resolution_factor = decrease_resolution_factor
        self.large_resolution_threshold = large_resolution_threshold
        self.moderate_resolution_threshold = moderate_resolution_threshold
        
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.very_low_ratio = very_low_ratio
        
        self.penalty_increase_threshold = penalty_increase_threshold
        self.penalty_increase_factor = penalty_increase_factor
        
        self.short_step_threshold = short_step_threshold
        self.low_radius_factor = low_radius_factor
        self.byrd_omojokun_factor = byrd_omojokun_factor
        self.threshold_ratio_constraints = threshold_ratio_constraints
        self.large_shift_factor = large_shift_factor
        self.large_gradient_factor = large_gradient_factor
        self.resolution_factor = resolution_factor
        self.improve_tcg = improve_tcg
