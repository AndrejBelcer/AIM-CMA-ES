import numpy as np
from dataclasses import dataclass

@dataclass
class ManifoldModel:
    """
    The mathematical output of the fitting stage.
    """
    origin: np.ndarray          # The exact center of the sampling volume in D-space
    basis: np.ndarray           # D x D orthogonal matrix (first M-1 are tangents, rest are normals)
    tangent_bounds_min: np.ndarray # Bounds along the tangent axes
    tangent_bounds_max: np.ndarray 
    normal_scales: np.ndarray   # The thickness/spread along each normal axis
