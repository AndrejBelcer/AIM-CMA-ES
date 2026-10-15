import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinementParameters
from src.implementations.optimizer_implementations.aim_cma_es.refinement.manifold_model import ManifoldModel


class ISampler(ABC):
    def __init__(self, refinement_parameters: RefinementParameters, x_count: int, f_count: int):
        self.refinement_parameters = refinement_parameters
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
        tangent_basis = model.basis[:actual_tangents]
        model_points = model.origin + (scaled_coords @ tangent_basis)
        return model_points

    def _generate_orthogonal_noise(self, model: ManifoldModel, n_points: int) -> np.ndarray:
        """
        Generates Gaussian noise vectors strictly orthogonal to the fitted tangent space,
        scaled anisotropically per normal direction.
        """
        n_tangents = len(model.tangent_bounds_min)
        normal_basis = model.basis[n_tangents:]
        
        if len(normal_basis) == 0:
            return np.zeros((n_points, self.x_count))

        # model.normal_scales already contains the per-direction bounds/std-devs
        noise_coeffs = np.random.normal(0, 1, size=(n_points, len(normal_basis))) * model.normal_scales
        
        return noise_coeffs @ normal_basis


# ==========================================
# GRID SAMPLERS
# ==========================================

class GridSampler(ISampler):
    """
    Standard Grid: Maps a regular mesh onto the shifted tangent space and adds orthogonal noise.
    """
    def __init__(self, refinement_parameters: RefinementParameters, x_count: int, f_count: int):
        super().__init__(refinement_parameters, x_count, f_count)
        
        self.grid_1d = np.linspace(-0.5 - refinement_parameters.tangent_grid_extension, 0.5 + refinement_parameters.tangent_grid_extension, refinement_parameters.grid_points_per_tangent_axis)
        mesh = np.meshgrid(*[self.grid_1d for _ in range(self.f_count - 1)], indexing='ij')
        self.grid_coeffs = np.column_stack([m.ravel() for m in mesh])
        self.n_grid_points = len(self.grid_coeffs)

    def sample(self, model: ManifoldModel) -> np.ndarray:
        model_points = self._project_to_ambient_space(model, self.grid_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_grid_points)
        return model_points + noise


class ModelAndPerturbedGridSampler(GridSampler):
    """
    Hybrid Grid: Generates TWO sets of candidates.
    1. Model samples on the fitted tangent space.
    2. Perturbed model samples with orthogonal noise.
    """
    def sample(self, model: ManifoldModel) -> np.ndarray:
        model_points = self._project_to_ambient_space(model, self.grid_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_grid_points)
        
        if np.all(noise == 0):
            return model_points

        return np.vstack([model_points, model_points + noise])


# ==========================================
# RANDOM SAMPLERS
# ==========================================

class RandomSampler(ISampler):
    """
    Standard Uniform Random: Safe for high-dimensional objective spaces.
    """
    def __init__(self, refinement_parameters: RefinementParameters, x_count: int, f_count: int):
        super().__init__(refinement_parameters, x_count, f_count)
        self.n_samples = 150
        self.tangent_grid_extension = refinement_parameters.tangent_grid_extension

    def _get_random_coeffs(self) -> np.ndarray:
        actual_tangents = max(1, self.f_count - 1)
        return np.random.uniform(low=-0.5 - self.tangent_grid_extension, high=0.5 + self.tangent_grid_extension, size=(self.n_samples, actual_tangents))

    def sample(self, model: ManifoldModel) -> np.ndarray:
        random_coeffs = self._get_random_coeffs()
        model_points = self._project_to_ambient_space(model, random_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_samples)
        return model_points + noise


class ModelAndPerturbedRandomSampler(RandomSampler):
    """
    Hybrid Random: Returns exact random points on the shifted tangent space AND noisy variations.
    """
    def sample(self, model: ManifoldModel) -> np.ndarray:
        random_coeffs = self._get_random_coeffs()
        model_points = self._project_to_ambient_space(model, random_coeffs)
        noise = self._generate_orthogonal_noise(model, self.n_samples)

        if np.all(noise == 0):
            return model_points

        return np.vstack([model_points, model_points + noise])
