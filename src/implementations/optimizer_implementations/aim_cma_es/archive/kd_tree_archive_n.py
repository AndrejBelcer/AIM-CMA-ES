from typing import Any, Tuple, List, Optional
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import ArchiveParameters


class KDTreeArchiveN(InternalArchive):
    """
    Uses KDTree with lazy rebuilds for k-NN queries.
    Uses dimension-pruning bounding box method for fast ball searches.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, capacity: int, parameters: ArchiveParameters) -> None:
        super().__init__(x_count, f_count, g_count, lower_bounds, upper_bounds, capacity)

        self.checkpoint_interval: int = parameters.checkpoint_interval
        self.leaf_size: int = parameters.kd_leaf_size

        self._kdtree: Optional[KDTree] = None
        self._kdtree_size: int = 0


    def _check_and_rebuild_tree(self):
        if self.n_size == 0:
            return
            
        unindexed_count = self.n_size - self._kdtree_size
        if self._kdtree is None or unindexed_count >= self.checkpoint_interval:
            self._kdtree = KDTree(self._X_norm_arr[:self.n_size], leafsize=self.leaf_size)
            self._kdtree_size = self.n_size


    def query_k_nearest(self, c_norm: np.ndarray, k: int, distance_upper_bound: float, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        c_norm_2d = np.atleast_2d(c_norm)
        M = c_norm_2d.shape[0]
        
        if self.n_size == 0:
            return np.empty((M, 0)), np.empty((M, 0), dtype=int)
            
        self._check_and_rebuild_tree()
        K_tree = self._kdtree_size
        N_total = self.n_size
        actual_k = min(N_total, k)
        
        if K_tree > 0:
            tree_k = min(K_tree, actual_k)
            d_tree, i_tree = self._kdtree.query(c_norm_2d, k=tree_k, eps=eps, distance_upper_bound=distance_upper_bound)
            d_tree = np.reshape(d_tree, (M, tree_k))
            i_tree = np.reshape(i_tree, (M, tree_k))
        else:
            d_tree = np.empty((M, 0))
            i_tree = np.empty((M, 0), dtype=int)
            
        if N_total > K_tree:
            buffer_norm = self._X_norm_arr[K_tree:N_total]
            d_buf = cdist(c_norm_2d, buffer_norm, metric='euclidean')
            i_buf = np.tile(np.arange(K_tree, N_total), (M, 1))
            
            if distance_upper_bound != np.inf:
                out_of_bounds = d_buf > distance_upper_bound
                d_buf[out_of_bounds] = np.inf
                i_buf[out_of_bounds] = N_total
        else:
            d_buf = np.empty((M, 0))
            i_buf = np.empty((M, 0), dtype=int)
            
        combined_dists = np.hstack([d_tree, d_buf])
        combined_indices = np.hstack([i_tree, i_buf])
        
        if combined_dists.shape[1] <= actual_k:
            sort_mask = np.argsort(combined_dists, axis=1)
            final_dists = np.take_along_axis(combined_dists, sort_mask, axis=1)
            final_indices = np.take_along_axis(combined_indices, sort_mask, axis=1)
        else:
            part_mask = np.argpartition(combined_dists, actual_k - 1, axis=1)[:, :actual_k]
            part_dists = np.take_along_axis(combined_dists, part_mask, axis=1)
            part_indices = np.take_along_axis(combined_indices, part_mask, axis=1)
            
            sort_mask = np.argsort(part_dists, axis=1)
            final_dists = np.take_along_axis(part_dists, sort_mask, axis=1)
            final_indices = np.take_along_axis(part_indices, sort_mask, axis=1)
            
        invalid_mask = np.isinf(final_dists)
        final_indices[invalid_mask] = -1
            
        return final_dists, final_indices
