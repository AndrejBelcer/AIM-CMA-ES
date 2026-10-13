import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Union

from moocore import is_nondominated

from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import minimum_spanning_tree


from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive


class StateTracker:
    """
    Manages the Live and Frozen states of the optimization process.
    Handles incremental non-dominated sorting and normalization bounds.
    
    Optimization:
    - Caches front objective coordinates (live_front_F) and sorted representations
      to achieve O(1) reads, completely eliminating redundant archive fetches and sorting overhead.
    """
    def __init__(self, f_count: int, x_count: int):
        self.f_count = f_count
        self.x_count = x_count
        
        # --- Live State (Global Extremes) ---
        self.f_min = np.full(f_count, np.inf)
        self.f_max = np.full(f_count, -np.inf)
        
        # --- Live State (Feasible Extremes) ---
        self.feas_f_min = np.full(f_count, np.inf)
        self.feas_f_max = np.full(f_count, -np.inf)
        
        # --- Live State (Empirical Front Extremes) ---
        self.live_front_f_L = np.zeros(f_count)
        self.live_front_f_U = np.ones(f_count)
        self.live_best_corners = np.full(f_count, np.inf)
        
        self.empirical_front_indices: np.ndarray = np.array([], dtype=int)

        # --- CACHED STATE FOR O(1) LIFECYCLE ACCESS ---
        self._live_front_F = np.empty((0, f_count))
        self._live_front_F_sorted_2d = np.empty((0, f_count))

        # Buffer containing newly evaluated indices that are eligible for NDS processing
        self.pending_feasible_indices: List[int] = []
        
        # --- Frozen State ---
        self.f_L = np.zeros(f_count)
        self.f_U = np.ones(f_count)
        self.f_D = np.ones(f_count)
        self.frozen_front_X = np.empty((0, x_count))
        self.frozen_front_F = np.empty((0, f_count))
        self.frozen_best_corners = np.full(f_count, np.inf)
        
        # Structured 2D array for vectorized spatial pruning for target gap
        self.tabu_array: np.ndarray = np.empty((0, f_count))

    def update_live_state(self, new_indices: np.ndarray, new_f: np.ndarray, new_g: np.ndarray, archive: InternalArchive):
        """Updates global extremes and the Empirical Front immediately (Synchronous)."""
        self.buffer_live_state(new_indices, new_f, new_g, archive)
        self.finalize_live_state(archive)


    def buffer_live_state(self, new_indices: np.ndarray, new_f: np.ndarray, new_g: np.ndarray, archive: InternalArchive):
        """
        Updates the global and feasible extrema immediately, but buffers eligible indices 
        for deferred Non-Dominated Sort runs. 
        
        Applies a fast Pareto dominance pre-filter to discard points that are 
        already dominated by the current empirical front.
        """
        # 1. Update Global Extremes (Always keep boundaries fresh)
        for i in range(self.f_count):
            col = new_f[:, i]
            valid = np.isfinite(col)
            if np.any(valid):
                self.f_min[i] = min(self.f_min[i], np.min(col[valid]))
                self.f_max[i] = max(self.f_max[i], np.max(col[valid]))
        
        # 2. Extract feasible candidates
        if archive.g_count > 0:
            is_new_feas = np.all(new_g <= 0, axis=1)
        else:
            is_new_feas = np.ones(len(new_f), dtype=bool)
            
        if not np.any(is_new_feas):
            return
            
        new_feas_indices = new_indices[is_new_feas]
        new_feas_f = new_f[is_new_feas]

        # Update Feasible Extremes incrementally
        self.feas_f_min = np.minimum(self.feas_f_min, np.min(new_feas_f, axis=0))
        self.feas_f_max = np.maximum(self.feas_f_max, np.max(new_feas_f, axis=0))

        # 3. If the front is empty, buffer everything and exit
        if len(self.empirical_front_indices) == 0:
            self.pending_feasible_indices.extend(new_feas_indices.tolist())
            return

        # 4. Pareto Pre-Filter (Using highly efficient pre-fetched/pre-sorted cached arrays)
        if self.f_count == 2:
            # 2D Path: O(log K) binary search using pre-sorted cache. Zero sorting overhead here!
            idx = np.searchsorted(self._live_front_F_sorted_2d[:, 0], new_feas_f[:, 0])
            is_dominated = (idx > 0) & (new_feas_f[:, 1] >= self._live_front_F_sorted_2d[idx - 1, 1])
            
        else:
            # N-D Path: Vectorized dominance check directly against cached contiguous array. Zero fetch overhead!
            is_worse_or_equal = new_feas_f[:, np.newaxis, :] >= self._live_front_F[np.newaxis, :, :]
            is_dominated = np.any(np.all(is_worse_or_equal, axis=2), axis=1)

        # 5. Buffer only the non-dominated candidates
        novel_feas_indices = new_feas_indices[~is_dominated]
        if len(novel_feas_indices) > 0:
            self.pending_feasible_indices.extend(novel_feas_indices.tolist())


    def finalize_live_state(self, archive: InternalArchive):
        """
        Merges and sorts all accumulated indices in the pending buffer into the active
        Empirical Front in a single batch. Clears the buffer upon completion.
        
        Optimization:
        - Only queries the archive for the NEW pending points.
        - Stacks new points with the cached front, keeping operations local and ultra-fast.
        """
        if len(self.pending_feasible_indices) == 0:
            return

        # 1. Clean and deduplicate pending indices
        pending_arr = np.unique(self.pending_feasible_indices)
        
        # Filter out any that are already in the empirical front (safety margin)
        pending_arr = np.setdiff1d(pending_arr, self.empirical_front_indices, assume_unique=True)
        
        # Clear the index accumulation buffer
        self.pending_feasible_indices.clear()
        
        if len(pending_arr) == 0:
            return

        # 2. Reconstruct global index map in a reliable, contiguous stacking order
        candidate_indices = np.concatenate([self.empirical_front_indices, pending_arr])

        # 3. Fetch ONLY the newly added F values from the archive
        pending_F = archive.get_f_by_indices(pending_arr)
        
        # 4. Stack cached front values with pending values. Zero redundant archive fetches!
        candidate_F = np.vstack([self._live_front_F, pending_F])
        
        # 5. Run the blazing fast pre-compiled sweep
        is_nondom = is_nondominated(candidate_F, keep_weakly=False)
        
        self.empirical_front_indices = candidate_indices[is_nondom].astype(int)

        # 6. Update Cache
        self._live_front_F = candidate_F[is_nondom]
        
        if self.f_count == 2:
            sort_idx = np.argsort(self._live_front_F[:, 0])
            self._live_front_F_sorted_2d = self._live_front_F[sort_idx]

        # 7. Extract best objective values representing front corners and cache front bounds
        if len(self._live_front_F) > 0:
            self.live_best_corners = np.min(self._live_front_F, axis=0)
            self.live_front_f_L = np.min(self._live_front_F, axis=0)
            self.live_front_f_U = np.max(self._live_front_F, axis=0)


    def GetNormalizationBounds(self, archive: InternalArchive) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves f_L, f_U and f_D based on the '2N' rule in O(1) time using cached boundaries.
        """
        if np.any(np.isinf(self.f_min)):
            return np.zeros(self.f_count), np.ones(self.f_count), np.ones(self.f_count)

        n_front = len(self.empirical_front_indices)
        threshold = 2 * self.f_count
        
        if n_front >= threshold:
            f_L = self.live_front_f_L.copy()
            f_U_base = self.live_front_f_U.copy()
        elif archive.n_feasible >= threshold:
            f_L = self.feas_f_min.copy()
            f_U_base = self.feas_f_max.copy()
        else:
            f_L = self.f_min.copy()
            f_U_base = self.f_max.copy()

        raw_range = f_U_base - f_L       
        f_D = np.maximum(raw_range, 1e-8)
        f_U = f_L + f_D
        
        return f_L, f_U, f_D


    def freeze_state(self, archive: InternalArchive):
        """Snapshots the current state into the frozen variables."""
        f_L, f_U, f_D = self.GetNormalizationBounds(archive)
        
        self.f_L = f_L
        self.f_U = f_U
        self.f_D = f_D
        self.frozen_best_corners = self.live_best_corners.copy()
        
        if len(self.empirical_front_indices) > 0:
            # We already have the front F values cached! We only need X from the archive.
            self.frozen_front_X = archive.get_x_by_indices(self.empirical_front_indices).copy()
            self.frozen_front_F = self._live_front_F.copy()
        else:
            self.frozen_front_X = np.empty((0, self.x_count))
            self.frozen_front_F = np.empty((0, self.f_count))


    def add_tabu_point(self, f: np.ndarray) -> None:
        self.tabu_array = np.vstack([self.tabu_array, f])


    def empty_tabu_array(self) -> None:
        self.tabu_array = np.empty((0, self.f_count))


    def GetGapTarget(self, tabu_tolerance: float) -> Tuple[Optional[np.ndarray], List[np.ndarray], bool]:
        """
        Calculates the largest gap in the Frozen Front.
        - For 2D: Uses an efficient O(N log N) sorting-based approach.
        - For >2D: Uses a robust O(N^2) Minimum Spanning Tree approach.
        Returns: (Target_F_Real, [X_Raw_A, X_Raw_B], All_Were_Tabu_Flag)
        """
        F_front = self.frozen_front_F
        X_front = self.frozen_front_X
        
        if len(F_front) < 2:
            return None, [], False
            
        # Normalize Front for gap calculation
        F_norm = (F_front - self.f_L) / self.f_D
        
        # --- Select Edges (Gaps) ---
        if self.f_count == 2:
            # --- 2D Optimized Path (O(N log N)) ---
            sort_indices = np.argsort(F_norm[:, 0])
            F_sorted = F_norm[sort_indices]
            
            # Vectorized coordinate deltas (slice subtraction is faster than np.diff)
            diffs = F_sorted[1:] - F_sorted[:-1]
            
            # Potential HV = Delta f1 * |Delta f2|
            # (np.abs handles potential slight numerical noise in empirical fronts)
            areas = diffs[:, 0] * np.abs(diffs[:, 1])
            
            # Vectorized edge sorting by largest area (view reversal [::-1] avoids memory copies)
            edge_order = np.argsort(areas)[::-1]
            original_r_indices = sort_indices[edge_order]
            original_c_indices = sort_indices[edge_order + 1]
        else:
            # --- N-D Path (O(N^2)) ---
            dists_pdist = np.maximum(pdist(F_norm), 1e-9)
            mst = minimum_spanning_tree(squareform(dists_pdist))
            rows, cols = mst.nonzero()
            
            # Vectorized edge extraction (Replaces Python loop and .sort())
            mask = rows < cols
            r_indices = rows[mask]
            c_indices = cols[mask]
            
            # Compute distances for the edges
            y_r = F_norm[r_indices]
            y_c = F_norm[c_indices]
            dists = np.linalg.norm(y_r - y_c, axis=1)
            
            edge_order = np.argsort(dists)[::-1]
            original_r_indices = r_indices[edge_order]
            original_c_indices = c_indices[edge_order]

        # --- Process Edges to Find a Non-Tabu Target ---
        num_edges = len(original_r_indices)
        if num_edges == 0:
            return None, [], False
            
        has_tabu = len(self.tabu_array) > 0
        if has_tabu:
            tabu_norm_array = (self.tabu_array - self.f_L) / self.f_D
        
        # We process edges in batches to avoid wasting computation if an early edge is valid,
        # while still benefiting from blazing fast C-level vectorized operations.
        BATCH_SIZE = 256
        
        for start_idx in range(0, num_edges, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_edges)
            
            batch_r = original_r_indices[start_idx:end_idx]
            batch_c = original_c_indices[start_idx:end_idx]
            
            y_a = F_norm[batch_r]
            y_b = F_norm[batch_c]

            # --- Analytic Right Triangle Geometry (Fully Vectorized) ---
            y_m = np.minimum(y_a, y_b)
            vec_a = y_a - y_m
            vec_b = y_b - y_m
            len_a_sq = np.sum(vec_a**2, axis=1)
            len_b_sq = np.sum(vec_b**2, axis=1)
            
            cond = len_a_sq >= len_b_sq
            scalar = np.zeros(end_idx - start_idx)
            
            # Safe division masks
            mask_a = cond & (len_a_sq > 1e-12)
            mask_b = (~cond) & (len_b_sq > 1e-12)
            
            scalar[mask_a] = (len_a_sq[mask_a] - len_b_sq[mask_a]) / (2 * len_a_sq[mask_a])
            scalar[mask_b] = (len_b_sq[mask_b] - len_a_sq[mask_b]) / (2 * len_b_sq[mask_b])
            
            target_norm = np.where(cond[:, np.newaxis], y_m + scalar[:, np.newaxis] * vec_a, y_m + scalar[:, np.newaxis] * vec_b)
            
            # --- Vectorized Tabu Check ---
            is_tabu = np.zeros(end_idx - start_idx, dtype=bool)
            if has_tabu:
                # cdist is highly optimized in C and computes all pairwise distances instantly
                dists_to_tabu = cdist(target_norm, tabu_norm_array)
                is_tabu = np.any(dists_to_tabu < tabu_tolerance, axis=1)
            
            # Find the first target in this batch that is NOT tabu
            valid_indices = np.where(~is_tabu)[0]
            
            if len(valid_indices) > 0:
                first_valid = valid_indices[0]
                best_r = batch_r[first_valid]
                best_c = batch_c[first_valid]
                best_target_norm = target_norm[first_valid]
                
                target_real = self.f_L + best_target_norm * self.f_D
                return target_real, [X_front[best_r], X_front[best_c]], False
        
        # If we exhausted all batches and found nothing, all were tabu
        return None, [], True



    def CheckCornerImprovement(self, f_significant_change: float) -> List[int]:
        """
        Compares the live corners against the frozen corners.
        Returns a list of objective indices that improved significantly.
        """
        raw_improvement = self.frozen_best_corners - self.live_best_corners
        norm_improvement = raw_improvement / self.f_D
        
        improved_indices: list[int] = []
        for i in range(self.f_count):
            imp = norm_improvement[i]
            
            if imp >= f_significant_change:
                improved_indices.append(i)
                
        return improved_indices

    def DebugCornerImprovement(self, f_significant_change: float, active_target_axis: Optional[int] = None) -> List[str]:
        raw_improvement = self.frozen_best_corners - self.live_best_corners
        norm_improvement = raw_improvement / self.f_D

        results: list[str] = []

        for i in range(self.f_count):
            imp = norm_improvement[i]

            if(active_target_axis is not None):
                if(i == active_target_axis):
                    if imp >= f_significant_change:
                        results.append(f"[Target Corner {i}] Improved significantly by {imp:.2e}.")
                    elif imp > 0:
                        results.append(f"[Target Corner {i}] Improved by {imp:.2e}.")
                    else:
                        print(f"[Target Corner {i}] Failed to improve target.")
                   
            else:
                results.append(f"[Side Effect] Corner {i} improved by {imp:.2e}.")
                        
        return results
