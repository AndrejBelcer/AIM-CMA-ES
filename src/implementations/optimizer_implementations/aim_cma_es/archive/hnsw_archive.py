from typing import Tuple, List, Optional
import numpy as np
import hnswlib

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import ArchiveParameters

class HNSWArchive(InternalArchive):
    """
    Uses HNSW (Hierarchical Navigable Small World) for ultra-fast high-dimensional queries.
    Note: HNSW is an Approximate Nearest Neighbor (ANN) algorithm.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, capacity: int, parameters: ArchiveParameters) -> None:
        super().__init__(x_count, f_count, g_count, lower_bounds, upper_bounds, capacity)

        # Initialize HNSW Index
        self.index = hnswlib.Index(space='l2', dim=x_count)
        
        # M=16, ef_construction=200 are excellent defaults for 20-40D.
        self.index.init_index(max_elements=capacity, ef_construction=200, M=16)
        
        # Default ef controls query time vs accuracy. 
        self.default_ef = parameters.default_ef
        self.index.set_ef(self.default_ef) 
        
        self._indexed_size = 0


    def _resize(self):
        """Override resize to also resize the HNSW index."""
        super()._resize()
        self.index.resize_index(self.capacity)


    def add(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        """Override add to push points to the HNSW index incrementally."""
        start_idx = self.n_size
        super().add(x, f, g) # Calls the parent add, updates self.n_size
        
        n_new = len(x)
        if n_new == 0:
            return
        
        # Add to HNSW index. We pass the normalized data and their corresponding integer IDs.
        new_norm_data = self._X_norm_arr[start_idx : self.n_size]
        ids = np.arange(start_idx, self.n_size)
        
        self.index.add_items(new_norm_data, ids)
        self._indexed_size = self.n_size


    def query_k_nearest(self, c_norm: np.ndarray, k: int, distance_upper_bound: float, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the 'k' nearest neighbors using HNSW.
        """
        c_norm_2d = np.atleast_2d(c_norm)
        M = c_norm_2d.shape[0]
        
        if self.n_size == 0:
            return np.empty((M, 0)), np.empty((M, 0), dtype=int)
            
        actual_k = min(self.n_size, k)
        current_ef = self.index.ef
        
        try:
            # Safely increase ef if k is larger than current ef to prevent C++ crash
            if actual_k > current_ef:
                self.index.set_ef(actual_k)
                
            # Query HNSW. Returns labels (indices) as uint64 and squared distances.
            labels, sq_dists = self.index.knn_query(c_norm_2d, k=actual_k)
            
        finally:
            # Always restore ef to prevent slowing down future queries
            if self.index.ef != current_ef:
                self.index.set_ef(current_ef)
        
        # Cast labels from uint64 to int64 so we can assign -1 to them
        labels = labels.astype(np.int64)
        
        # Convert squared L2 to standard Euclidean distance
        dists = np.sqrt(sq_dists)
        
        # Apply distance upper bound
        if distance_upper_bound != np.inf:
            invalid_mask = dists > distance_upper_bound
            dists[invalid_mask] = np.inf
            labels[invalid_mask] = -1
            
        return dists, labels
