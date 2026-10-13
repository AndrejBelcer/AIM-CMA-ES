from typing import Tuple, List, Optional
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular
from abc import ABC, abstractmethod

import numba as nb

@nb.njit(cache=True, fastmath=True)
def _numba_early_abandon_search(X: np.ndarray, c: np.ndarray, r_sq: float) -> np.ndarray:
    """
    Highly optimized, cache-friendly Numba implementation.
    Iterates row-by-row (C-contiguous) for maximum memory bandwidth.
    """
    N, D = X.shape
    out_indices = np.empty(N, dtype=np.int64)
    count = 0
    
    for i in range(N):
        dist_sq = 0.0
        survived = True
        for d in range(D):
            diff = X[i, d] - c[d]
            dist_sq += diff * diff
            
            # Early abandonment
            if dist_sq > r_sq:
                survived = False
                break
                
        if survived:
            out_indices[count] = i
            count += 1
            
    return out_indices[:count]


class InternalArchive(ABC):
    """
    High-performance, pre-allocated storage for solver history.
    
    Optimization:
    - Pre-allocation: Avoids O(N^2) memory copying from repeated np.vstack calls.
    - Views: Returns slices of the main array to avoid copying data.
    - Normalization: Handles duplicate checking in [0,1] space.
    - Feasibility Caching: Tracks feasibility state on insertion for O(1) queries.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, capacity: int) -> None:
        self.x_count = x_count
        self.f_count = f_count
        self.g_count = g_count
        
        # Bounds for Normalization
        self.lb = lower_bounds
        self.ub = upper_bounds
        
        # Pre-calculate range
        self.range = self.ub - self.lb
        self.range[self.range < 1e-9] = 1.0
        
        # State
        self.n_size = 0
        self.capacity = capacity
        
        # Pre-allocate contiguous memory
        self._X_arr = np.empty((capacity, x_count), dtype=np.float64)
        # Store normalized coordinates directly to prevent redundant calculations
        self._X_norm_arr = np.empty((capacity, x_count), dtype=np.float64)
        self._F_arr = np.empty((capacity, f_count), dtype=np.float64)
        self._G_arr = np.empty((capacity, g_count), dtype=np.float64)
        
        # Pre-allocate feasibility tracking for O(1) retrieval
        self._feas_arr = np.empty(capacity, dtype=bool)
        self.n_feasible = 0


    def _resize(self):
        """Doubles capacity if full."""
        new_cap = self.capacity * 2
        print(f"[InternalArchive] Resizing storage: {self.capacity} -> {new_cap}")
        
        # Resize X
        new_X = np.empty((new_cap, self.x_count), dtype=np.float64)
        new_X[:self.capacity] = self._X_arr
        self._X_arr = new_X

        # Resize X_norm
        new_X_norm = np.empty((new_cap, self.x_count), dtype=np.float64)
        new_X_norm[:self.capacity] = self._X_norm_arr
        self._X_norm_arr = new_X_norm
        
        # Resize F
        new_F = np.empty((new_cap, self.f_count), dtype=np.float64)
        new_F[:self.capacity] = self._F_arr
        self._F_arr = new_F
        
        # Resize G
        new_G = np.empty((new_cap, self.g_count), dtype=np.float64)
        new_G[:self.capacity] = self._G_arr
        self._G_arr = new_G
        
        # Resize Feasibility Array
        new_feas = np.empty(new_cap, dtype=bool)
        new_feas[:self.capacity] = self._feas_arr
        self._feas_arr = new_feas
        
        self.capacity = new_cap


    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lb) / self.range


    def add(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        """
        Adds a batch of evaluations.
        Input shapes: x(N, D), f(N, M), g(N, P)
        """
        n_new = len(x)
        if n_new == 0: return

        # Check capacity
        if self.n_size + n_new > self.capacity:
            self._resize()

        # Sanitize F (NaN -> Inf)
        f_clean = np.where(np.isnan(f), np.inf, f)
        
        # Compute normalized coordinate representations immediately
        x_norm = self._normalize(x)
        
        # Compute feasibility mask for the incoming batch
        if self.g_count > 0:
            is_feas = np.all(g <= 0, axis=1)
        else:
            is_feas = np.ones(n_new, dtype=bool)
        
        # Direct write to memory (Zero Copy overhead)
        start = self.n_size
        end = start + n_new
        
        self._X_arr[start:end] = x
        self._X_norm_arr[start:end] = x_norm
        self._F_arr[start:end] = f_clean
        self._G_arr[start:end] = g
        self._feas_arr[start:end] = is_feas
        
        self.n_size = end
        self.n_feasible += np.sum(is_feas)

    def get_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns VIEWS of the active data."""
        return (self._X_arr[:self.n_size], self._F_arr[:self.n_size], self._G_arr[:self.n_size])

    def get_norm_data(self) -> np.ndarray:
        """Returns a view of the active normalized coordinate data."""
        return self._X_norm_arr[:self.n_size]

    def get_feasible_mask(self) -> np.ndarray:
        return self._feas_arr[:self.n_size]

    def get_values_by_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._F_arr[indices], self._G_arr[indices]
    
    def get_norm_x_by_indices(self, indices: np.ndarray) -> np.ndarray:
        return self._X_norm_arr[indices]
    
    def get_x_by_indices(self, indices: np.ndarray) -> np.ndarray:
        return self._X_arr[indices]

    def get_f_by_indices(self, indices: np.ndarray) -> np.ndarray:
        return self._F_arr[indices]

    def get_g_by_indices(self, indices: np.ndarray) -> np.ndarray:
        return self._G_arr[indices]

    def get_sanitized_f_by_indices(self, indices: np.ndarray, f_U: np.ndarray) -> np.ndarray:
        """Sanitizes only the requested indices, saving massive overhead."""
        F_view = self._F_arr[indices]
        F_sanitized = F_view.copy()
        mask_inf = np.isinf(F_sanitized)
        
        if np.any(mask_inf):
            for i in range(self.f_count):
                penalty = 2.0 * max(0.1, f_U[i])
                F_sanitized[mask_inf[:, i], i] = penalty
                
        return F_sanitized

    def get_sanitized_data(self, f_U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy method for full archive sanitization."""
        F_sanitized = self.get_sanitized_f_by_indices(np.arange(self.n_size), f_U)
        return self._X_arr[:self.n_size], F_sanitized, self._G_arr[:self.n_size]

    def check_duplicates_with_indices(self, x_batch: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized duplicate check cleanly wrapping query_k_nearest.
        """
        n_batch = len(x_batch)
        if self.n_size == 0:
            return np.ones(n_batch, dtype=bool), np.full(n_batch, -1, dtype=int)

        X_batch_norm = self._normalize(x_batch)
        
        # Query the single closest neighbor within the tolerance
        dists, indices = self.query_k_nearest(X_batch_norm, k=1, distance_upper_bound=tol)
        
        # query_k_nearest returns shape (N, k). Since k=1, we flatten (N, 1) to (N,)
        dists = dists.ravel()
        indices = indices.ravel()
        
        # If distance is <= tol, it's a duplicate (not novel)
        is_novel = dists > tol
        
        # For novel points, we don't want to return an index, so we set it to -1
        result_indices = np.where(is_novel, -1, indices)
        
        return is_novel, result_indices


    #########
    # Overridable Methods
    #########
    def query_ball_points(self, centers_norm: np.ndarray, r: float) -> List[np.ndarray]:
        """
        Finds all indices in the archive within radius 'r' for multiple centers.
        `centers_norm` is a 2D array of shape (K, D).
        Returns a list of length K, where each element is a 1D numpy array of integer indices.
        """
        ####
        # Skip safety check for speed.
        ####
        #if self.n_size == 0 or len(centers_norm) == 0:
        #    return [np.empty(0, dtype=int) for _ in range(len(centers_norm))]
        centers = np.atleast_2d(centers_norm)
        return [self.query_ball_point(c, r) for c in centers]


    
    def query_ball_point(self, c_norm: np.ndarray, r: float, use_numba: bool = False) -> np.ndarray:
        """
        Finds all indices in the archive that lie within a normalized Euclidean radius 'r'.
        Branches based on 'r' to avoid useless pruning overhead for large radii.
        """
        if self.n_size == 0:
            return np.empty(0, dtype=int)
            
        c_norm_1d = np.asarray(c_norm, dtype=np.float64).ravel()
        r_sq = r * r
        X_active = self._X_norm_arr[:self.n_size]
        
        # Threshold for branching. 
        # At r > 0.3 in high dimensions, the bounding box covers too much space.
        if r > 0.3:
            return self._query_blas(X_active, c_norm_1d, r_sq)
        else:
            if use_numba:
                return _numba_early_abandon_search(X_active, c_norm_1d, r_sq)
            else:
                return self._query_prune(X_active, c_norm_1d, r, r_sq)

    def _query_blas(self, X_active: np.ndarray, c_norm_1d: np.ndarray, r_sq: float) -> np.ndarray:
        """
        Pure BLAS implementation for large r. 
        Optimized for streaming massive arrays through the CPU.
        """
        # ||c||^2
        c_sq = np.dot(c_norm_1d, c_norm_1d)
        
        # X @ c (Multi-threaded BLAS GEMV)
        X_dot_c = X_active @ c_norm_1d
        
        # ||X||^2 computed live (Memory bandwidth bound, but fast enough for < 10^7)
        X_sq = np.einsum('ij,ij->i', X_active, X_active) 
        
        # Full exact squared distances: ||X||^2 - 2Xc + ||c||^2
        sq_dists = X_sq - (2.0 * X_dot_c) + c_sq
        
        # Clip floating point noise (e.g., -1e-16 becomes 0.0)
        np.maximum(sq_dists, 0.0, out=sq_dists)
        
        return np.nonzero(sq_dists <= r_sq)[0]

    def _query_prune(self, X_active: np.ndarray, c_norm_1d: np.ndarray, r: float, r_sq: float) -> np.ndarray:
        """
        Pure NumPy dimension-by-dimension pruning for small r.
        Avoids reading the full array by aggressively shrinking the candidate pool.
        """
        D = c_norm_1d.shape[0]
        
        # Bounding box limits
        lb = c_norm_1d - r
        ub = c_norm_1d + r
        
        # --- Step 1: Dimension 0 (Full array read, creates initial mask) ---
        vals_0 = X_active[:, 0]
        mask = (vals_0 >= lb[0]) & (vals_0 <= ub[0])
        candidates = np.nonzero(mask)[0]
        
        # --- Step 2: Dimension-by-Dimension Pruning ---
        for d in range(1, D):
            if candidates.size == 0:
                return np.empty(0, dtype=int)
            
            # Extract only the surviving candidates for dimension d
            vals_d = X_active[candidates, d]
            
            # Apply bounding box
            mask_d = (vals_d >= lb[d]) & (vals_d <= ub[d])
            candidates = candidates[mask_d]

        if candidates.size == 0:
            return np.empty(0, dtype=int)

        # --- Step 3: Exact L2 Distance on Survivors ---
        survivors_X = X_active[candidates]
        diff = survivors_X - c_norm_1d
        
        # np.einsum avoids allocating a large array for diff**2
        sq_dists = np.einsum('ij,ij->i', diff, diff)
        
        return candidates[sq_dists <= r_sq]




    #########
    # Abstract Methods
    #########
    @abstractmethod
    def query_k_nearest(self, c_norm: np.ndarray, k: int, distance_upper_bound: float, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the exact 'k' nearest neighbors. 
        Missing neighbors (due to distance_upper_bound) are returned with distance np.inf and index -1.
        Always returns 2D arrays of shape (M, k).
        """
        pass
    
    
