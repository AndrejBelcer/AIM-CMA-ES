import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinementParameters
from src.implementations.optimizer_implementations.aim_cma_es.refinement.local_region import LocalRegion

class ILocalityGatherer(ABC):
    def __init__(self, refinement_parameters: RefinementParameters, x_count: int, f_count: int):
        self.refinement_parameters = refinement_parameters
        self.x_count = x_count
        self.f_count = f_count

    def get_number_of_regions(self) -> int:
        return np.random.randint(low = self.refinement_parameters.min_local_regions, high = self.refinement_parameters.max_local_regions + 1 , size = None)

    @abstractmethod
    def gather(self, archive: InternalArchive, front_indices: np.ndarray, f_L: np.ndarray, f_D: np.ndarray) -> List[LocalRegion]:
        pass

class FpsUniformRadiusGatherer(ILocalityGatherer):
    """
    Uses Farthest Point Sampling (FPS) on the empirical set in normalized decision space to select centers.
    Calculates a common radius that covers the empirical set around those centers.
    Creates LocalRegions containing empirical-set, infeasible, and dominated archive indices.
    """
    def gather(self, archive, front_indices, f_L, f_D):
        n_front = len(front_indices)
        if n_front == 0:
            return []

        X_norm = archive.get_norm_data()
        empirical_set_X_normalized = X_norm[front_indices]

        actual_n_regions = min(self.get_number_of_regions(), n_front)
        actual_n_targeted = min(self.refinement_parameters.local_regions_per_pass, actual_n_regions)

        center_local_indices, global_radius = self._fps_and_radius(empirical_set_X_normalized, actual_n_regions)
        targeted_local_indices = np.random.choice(center_local_indices, size=actual_n_targeted, replace=False)
        targeted_centers_X = empirical_set_X_normalized[targeted_local_indices]

        search_radius = (global_radius + 1e-6) * self.refinement_parameters.local_region_radius_factor

        regions: List[LocalRegion] = []

        for center_x in targeted_centers_X:
            # Query only the empirical-set subset, without scanning the full archive.
            ball_local_indices = self._query_ball_subset(empirical_set_X_normalized, center_x, search_radius)

            if len(ball_local_indices) == 0:
                continue

            # Map local indices (relative to empirical_set_X_normalized) → archive-level indices
            empirical_set_indices = front_indices[ball_local_indices]

            region = LocalRegion(center_x=center_x, radius=search_radius, empirical_set_indices=empirical_set_indices)
            regions.append(region)

        return regions


    @staticmethod
    def _query_ball_subset(X_subset: np.ndarray, c_norm: np.ndarray, r: float) -> np.ndarray:
        """
        Fast ball query on a small numpy array. No archive needed.
        Uses bounding-box pruning followed by squared-distance filtering.
        """
        c = np.asarray(c_norm, dtype=np.float64).ravel()
        D = c.shape[0]
        r_sq = r * r

        lb = c - r
        ub = c + r

        vals_0 = X_subset[:, 0]
        mask = (vals_0 >= lb[0]) & (vals_0 <= ub[0])
        candidates = np.nonzero(mask)[0]

        for d in range(1, D):
            if len(candidates) == 0:
                break
            vals_d = X_subset[candidates, d]
            mask_d = (vals_d >= lb[d]) & (vals_d <= ub[d])
            candidates = candidates[mask_d]

        if len(candidates) == 0:
            return np.array([], dtype=int)

        diff = X_subset[candidates] - c
        sq_dists = np.einsum('ij,ij->i', diff, diff)
        return candidates[sq_dists <= r_sq]

    def _fps_and_radius(self, empirical_set_X_normalized: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        """
        Runs Farthest Point Sampling to find 'k' centers spread across the manifold.
        Returns the local indices of the centers and the maximum distance from any point to its nearest center.
        """
        n_points = len(empirical_set_X_normalized)
        centers = np.zeros(k, dtype=int)
        
        # Start with index 0
        centers[0] = 0
        
        # Track the squared distance from each point to its nearest center
        min_dists_sq = np.sum((empirical_set_X_normalized - empirical_set_X_normalized[centers[0]])**2, axis=1)
        
        for i in range(1, k):
            # The next center is the point farthest from all existing centers
            next_center = np.argmax(min_dists_sq)
            centers[i] = next_center
            
            # Update the minimum distances
            dist_to_new_center_sq = np.sum((empirical_set_X_normalized - empirical_set_X_normalized[next_center])**2, axis=1)
            min_dists_sq = np.minimum(min_dists_sq, dist_to_new_center_sq)
            
        # The global covering radius is the maximum of the minimum distances
        global_radius = np.sqrt(np.max(min_dists_sq))
        
        # Safety bound to prevent radius from collapsing to 0 in edge cases
        global_radius = max(float(global_radius), 1e-4)
        
        return centers, global_radius
