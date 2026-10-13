import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from scipy.linalg import svd, eigh, qr
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize

from moocore import is_nondominated

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinerParameters
from src.implementations.optimizer_implementations.aim_cma_es.refiner.local_bubble import LocalBubble
from src.implementations.optimizer_implementations.aim_cma_es.refiner.manifold_model import ManifoldModel


class IManifoldFitter(ABC):
    """
    Abstract base class for all manifold fitters.
    Contains core geometric helper methods shared across implementations.
    """
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        self.refiner_parameters = refiner_parameters
        self.x_count = x_count
        self.f_count = f_count

    @abstractmethod
    def fit(self, bubble: LocalBubble, archive: InternalArchive) -> Optional[ManifoldModel]:
        pass

    # ==========================================
    # GEOMETRIC & MATHEMATICAL HELPERS
    # ==========================================

    def _compute_base_svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts mean center, full basis, tangents, and normals.
        Uses full_matrices=True to guarantee a complete DxD basis even if N < D.
        """
        n_tangent = min(self.f_count - 1, self.x_count)
        mu = np.mean(X, axis=0)
        centered = X - mu
        _, _, Vh = svd(centered, full_matrices=True)
        
        V_tangent = Vh[:n_tangent]
        V_normal = Vh[n_tangent:]
        return mu, Vh, V_tangent, V_normal

    def _compute_base_svd_medoid(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts medoid center, full basis, tangents, and normals.
        Ensures the anchor point is strictly on the empirical manifold.
        """
        n_tangent = min(self.f_count - 1, self.x_count)
        mu_arithmetic = np.mean(X, axis=0)
        
        dists_sq = np.sum((X - mu_arithmetic) ** 2, axis=1)
        medoid_idx = np.argmin(dists_sq)
        medoid = X[medoid_idx]
        
        centered = X - medoid
        _, _, Vh = svd(centered, full_matrices=True)
        
        V_tangent = Vh[:n_tangent]
        V_normal = Vh[n_tangent:]
        return medoid, Vh, V_tangent, V_normal

    def _compute_spatial_bounds(self, X: np.ndarray, center: np.ndarray, tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(tangents) == 0:
            return np.zeros(0), np.zeros(0)
        coords = (X - center) @ tangents.T
        return np.min(coords, axis=0), np.max(coords, axis=0)

    def _compute_percentile_bounds(self, X: np.ndarray, center: np.ndarray, tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(tangents) == 0:
            return np.zeros(0), np.zeros(0)
        coords = (X - center) @ tangents.T
        return np.percentile(coords, 5, axis=0), np.percentile(coords, 95, axis=0)

    def _compute_noise_scale_vector(self, X: np.ndarray, center: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        Calculates orthogonal spread (thickness) per normal direction.
        """
        if len(normals) == 0:
            return np.zeros(0)
        normal_coords = (X - center) @ normals.T
        normal_spreads = np.std(normal_coords, axis=0)
        # Apply noise factor and a safety floor
        return np.maximum(normal_spreads * self.refiner_parameters.noise_factor, 1e-12)



class StandardSvdFitter(IManifoldFitter):
    def fit(self, bubble: LocalBubble, archive: InternalArchive) -> Optional[ManifoldModel]:
        X_E = bubble.get_X_E(archive)
        if len(X_E) < 2:
            return None
        return self._compute_model(X_E)

    def _compute_model(self, X_E: np.ndarray) -> ManifoldModel:
        center, basis, tangents, normals = self._compute_base_svd(X_E)
        bounds_min, bounds_max = self._compute_percentile_bounds(X_E, center, tangents)
        normal_scales = self._compute_noise_scale_vector(X_E, center, normals)
        
        return ManifoldModel(origin=center, basis=basis, tangent_bounds_min=bounds_min, tangent_bounds_max=bounds_max, normal_scales=normal_scales)


class TrimmedSvdFitter(StandardSvdFitter):
    def fit(self, bubble: LocalBubble, archive: InternalArchive) -> Optional[ManifoldModel]:
        X_E = bubble.get_X_E(archive)
        if len(X_E) < 3:
            return super().fit(bubble, archive)
        center_init, _, _, normals_init = self._compute_base_svd(X_E)
        if len(normals_init) > 0:
            normal_projections = (X_E - center_init) @ normals_init.T
            ortho_dists_sq = np.sum(normal_projections ** 2, axis=1)
        else:
            ortho_dists_sq = np.zeros(len(X_E))
        trim_fraction = 0.20
        n_keep = max(2, int(len(X_E) * (1.0 - trim_fraction)))
        keep_indices = np.argpartition(ortho_dists_sq, n_keep - 1)[:n_keep]
        return self._compute_model(X_E[keep_indices])

class RansacSvdFitter(StandardSvdFitter):
    """
    Random Sample Consensus SVD: Identifies largest inlier consensus set aligned to a line/plane.
    """
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        super().__init__(refiner_parameters, x_count, f_count)
        self.n_iterations = 50
        self.threshold_fraction = 0.05

    def fit(self, bubble: LocalBubble, archive: InternalArchive) -> Optional[ManifoldModel]:
        X_E = bubble.get_X_E(archive)
        n_points = len(X_E)
        n_tangent = min(self.f_count - 1, self.x_count)
        min_samples = n_tangent + 1

        if n_points <= min_samples:
            return super().fit(bubble, archive)

        threshold_sq = (self.threshold_fraction * bubble.radius) ** 2
        best_inlier_count = 0
        best_inlier_mask = None

        for _ in range(self.n_iterations):
            sample_indices = np.random.choice(n_points, size=min_samples, replace=False)
            X_sample = X_E[sample_indices]

            mu_sample, _, V_tangent, _ = self._compute_base_svd(X_sample)

            centered_all = X_E - mu_sample
            if len(V_tangent) > 0:
                projection = (centered_all @ V_tangent.T) @ V_tangent
                orthogonal_vectors = centered_all - projection
            else:
                orthogonal_vectors = centered_all

            dists_sq = np.einsum('ij,ij->i', orthogonal_vectors, orthogonal_vectors)
            inlier_mask = dists_sq <= threshold_sq
            inlier_count = np.sum(inlier_mask)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask
                if inlier_count > 0.9 * n_points:
                    break

        if best_inlier_count < min_samples or best_inlier_mask is None:
            return super().fit(bubble, archive)

        # Fit final model purely on the inliers
        X_inliers = X_E[best_inlier_mask]
        center, basis, tangents, normals = self._compute_base_svd(X_inliers)
        bounds_min, bounds_max = self._compute_percentile_bounds(X_inliers, center, tangents)
        normal_scales = self._compute_noise_scale_vector(X_inliers, center, normals)
        
        return ManifoldModel(origin=center, basis=basis, tangent_bounds_min=bounds_min, tangent_bounds_max=bounds_max, normal_scales=normal_scales)
