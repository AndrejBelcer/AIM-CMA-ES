import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinerParameters
from src.implementations.optimizer_implementations.aim_cma_es.refiner.manifold_model import ManifoldModel


class ISampler(ABC):
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        self.refiner_parameters = refiner_parameters
        self.x_count = x_count
        self.f_count = f_count

    @abstractmethod
    def sample(self, model: ManifoldModel) -> np.ndarray:
        pass

    # ==========================================
    # MATH HELPERS (Clean, centralized geometry)
    # ==========================================

    def _project_to_ambient_space(self, model: ManifoldModel, normalized_coords: np.ndarray) -> np.ndarray:
        """
        Maps normalized tangent coordinates (e.g. [-0.5, 0.5]) into the ambient search space (D).
        """
        n_points = len(normalized_coords)
        n_tangents = len(model.tangent_bounds_min)
        
        if n_tangents == 0:
            return np.tile(model.origin, (n_points, 1))

        # 1. Scale coordinates from normalized box to actual physical span
        t_center = 0.5 * (model.tangent_bounds_min + model.tangent_bounds_max)
        t_span = np.maximum(model.tangent_bounds_max - model.tangent_bounds_min, 1e-6)
        
        actual_tangents = min(n_tangents, normalized_coords.shape[1])
        scaled_coords = t_center[:actual_tangents] + (normalized_coords[:, :actual_tangents] * t_span[:actual_tangents])

        # 2. Project from Tangent Space -> Ambient X-Space
        V_tangent = model.basis[:actual_tangents]
        P_base = model.origin + (scaled_coords @ V_tangent)
        return P_base

    def _generate_orthogonal_noise(self, model: ManifoldModel, n_points: int) -> np.ndarray:
        """
        Generates Gaussian noise vectors strictly orthogonal to the fitted tangent space,
        scaled anisotropically per normal direction.
        """
        n_tangents = len(model.tangent_bounds_min)
        V_normal = model.basis[n_tangents:]
        
        if len(V_normal) == 0:
            return np.zeros((n_points, self.x_count))

        # model.normal_scales already contains the per-direction bounds/std-devs
        noise_coeffs = np.random.normal(0, 1, size=(n_points, len(V_normal))) * model.normal_scales
        
        return noise_coeffs @ V_normal


# ==========================================
# GRID SAMPLERS
# ==========================================

class GridSampler(ISampler):
    """
    Standard Grid: Maps a regular mesh onto the shifted tangent space and adds orthogonal noise.
    """
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        super().__init__(refiner_parameters, x_count, f_count)
        
        self.grid_1d = np.linspace(-0.5 - refiner_parameters.extension, 0.5 + refiner_parameters.extension, refiner_parameters.n_samples_per_gap)
        mesh = np.meshgrid(*[self.grid_1d for _ in range(self.f_count - 1)], indexing='ij')
        self.grid_coeffs = np.column_stack([m.ravel() for m in mesh])
        self.n_grid_points = len(self.grid_coeffs)

    def sample(self, model: ManifoldModel) -> np.ndarray:
        P_base = self._project_to_ambient_space(model, self.grid_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_grid_points)
        return P_base + noise


class DoubleGridSampler(GridSampler):
    """
    Hybrid Grid: Generates TWO sets of candidates.
    1. Exact points on the shifted line (Exploitation / Micro-convergence).
    2. Points with orthogonal noise (Exploration / Thickness).
    """
    def sample(self, model: ManifoldModel) -> np.ndarray:
        P_base = self._project_to_ambient_space(model, self.grid_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_grid_points)
        
        if np.all(noise == 0):
            return P_base

        return np.vstack([P_base, P_base + noise])


# ==========================================
# RANDOM SAMPLERS
# ==========================================

class RandomSampler(ISampler):
    """
    Standard Uniform Random: Safe for high-dimensional objective spaces.
    """
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        super().__init__(refiner_parameters, x_count, f_count)
        self.n_samples = 150
        self.extension = refiner_parameters.extension

    def _get_random_coeffs(self) -> np.ndarray:
        actual_tangents = max(1, self.f_count - 1)
        return np.random.uniform(low=-0.5 - self.extension, high=0.5 + self.extension, size=(self.n_samples, actual_tangents))

    def sample(self, model: ManifoldModel) -> np.ndarray:
        random_coeffs = self._get_random_coeffs()
        P_base = self._project_to_ambient_space(model, random_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_samples)
        return P_base + noise


class DoubleRandomSampler(RandomSampler):
    """
    Hybrid Random: Returns exact random points on the shifted tangent space AND noisy variations.
    """
    def sample(self, model: ManifoldModel) -> np.ndarray:
        random_coeffs = self._get_random_coeffs()
        P_base = self._project_to_ambient_space(model, random_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_samples)

        if np.all(noise == 0):
            return P_base

        return np.vstack([P_base, P_base + noise])
