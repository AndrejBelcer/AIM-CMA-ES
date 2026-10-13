from typing import Any, Tuple, List, Optional
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import ArchiveParameters


class KDTreeArchive(InternalArchive):
    """
    Uses scipy.spatial.KDTree with lazy rebuilds for both ball and k-NN queries.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, capacity: int, parameters: ArchiveParameters) -> None:
        super().__init__(x_count, f_count, g_count, lower_bounds, upper_bounds, capacity)

        self.checkpoint_interval: int = parameters.checkpoint_interval
        self.leaf_size: int = parameters.kd_leaf_size

        # Hybrid Spatial Indexing State
        self._kdtree: Optional[KDTree] = None
        self._kdtree_size: int = 0


    def _check_and_rebuild_tree(self):
        """
        Lazily builds or rebuilds the spatial KDTree if the number of unindexed 
        points (points added since the last rebuild) exceeds the checkpoint interval.
        """
        if self.n_size == 0:
            return
            
        unindexed_count = self.n_size - self._kdtree_size
        if self._kdtree is None or unindexed_count >= self.checkpoint_interval:
            self._kdtree = KDTree(self._X_norm_arr[:self.n_size], leafsize=self.leaf_size)
            self._kdtree_size = self.n_size


    def query_ball_point(self, c_norm: np.ndarray, r: float) -> np.ndarray:
        if self.n_size == 0:
            return np.empty(0, dtype=int)
            
        self._check_and_rebuild_tree()
        K = self._kdtree_size
        N = self.n_size
        
        parts = []
        
        # 1. Query the indexed tree
        if K > 0:
            tree_list = self._kdtree.query_ball_point(c_norm, r)
            if tree_list:
                parts.append(np.array(tree_list, dtype=int))
            
        # 2. Query the unindexed buffer
        if N > K:
            buffer_norm = self._X_norm_arr[K:N]
            sq_dists = np.sum((buffer_norm - c_norm)**2, axis=1)
            buf_indices = np.where(sq_dists <= r**2)[0] + K
            if buf_indices.size > 0:
                parts.append(buf_indices)
            
        if not parts:
            return np.empty(0, dtype=int)
        elif len(parts) == 1:
            return parts[0]
        else:
            return np.concatenate(parts)


    def query_k_nearest(self, c_norm: np.ndarray, k: int, distance_upper_bound: float, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the exact 'k' nearest neighbors. 
        Missing neighbors (due to distance_upper_bound) are returned with distance np.inf and index -1.
        Always returns 2D arrays of shape (M, k).
        """
        # Force 2D for vectorization
        c_norm_2d = np.atleast_2d(c_norm)
        M = c_norm_2d.shape[0]
        
        if self.n_size == 0:
            return np.empty((M, 0)), np.empty((M, 0), dtype=int)
            
        self._check_and_rebuild_tree()
        K_tree = self._kdtree_size
        N_total = self.n_size
        actual_k = min(N_total, k)
        
        # 1. Query the indexed tree
        if K_tree > 0:
            tree_k = min(K_tree, actual_k)
            d_tree, i_tree = self._kdtree.query(c_norm_2d, k=tree_k, eps=eps, distance_upper_bound=distance_upper_bound)
            
            # EXPLICIT RESHAPE to prevent Scipy/NumPy squeezing errors when k=1 or M=1
            d_tree = np.reshape(d_tree, (M, tree_k))
            i_tree = np.reshape(i_tree, (M, tree_k))
        else:
            d_tree = np.empty((M, 0))
            i_tree = np.empty((M, 0), dtype=int)
            
        # 2. Query the unindexed buffer
        if N_total > K_tree:
            buffer_norm = self._X_norm_arr[K_tree:N_total]
            d_buf = cdist(c_norm_2d, buffer_norm, metric='euclidean')
            i_buf = np.tile(np.arange(K_tree, N_total), (M, 1))
            
            # Apply distance upper bound to buffer
            if distance_upper_bound != np.inf:
                out_of_bounds = d_buf > distance_upper_bound
                d_buf[out_of_bounds] = np.inf
                i_buf[out_of_bounds] = N_total # Temporary padding
        else:
            d_buf = np.empty((M, 0))
            i_buf = np.empty((M, 0), dtype=int)
            
        # 3. Combine Tree and Buffer results
        combined_dists = np.hstack([d_tree, d_buf])
        combined_indices = np.hstack([i_tree, i_buf])
        
        # 4. Extract top 'actual_k'
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
            
        # 5. SAFETY: Replace any out-of-bounds indices with -1
        invalid_mask = np.isinf(final_dists)
        final_indices[invalid_mask] = -1
            
        return final_dists, final_indices
