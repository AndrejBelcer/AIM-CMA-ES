import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from scipy.spatial.distance import cdist

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.scalarizer_base import Scalarizer
from src.implementations.optimizer_implementations.aim_cma_es.enums import CVType, ScalarizationMode
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import WarmStartParameters

class CmaWarmStartManager:
    """
    Handles the logic for initializing CMA-ES from existing data.
    Operates strictly in the normalized ROI space.
    
    Logic:
    - Grid Reduction: Filters density to avoid bias using a pre-computed distance matrix.
    - Selection: Picks 'n_warm_start' points based on strategy (Full vs Partial).
    - Ranking: Sorts points (Feasible > Infeasible) and identifies ties.
    - Mixture Covariance: Computes both a weighted positional covariance and a velocity vector (gradient of improvement).
      Normalizes both to have Trace=N, mixes them and applies shrinkage.
    - Initialization: Shifts the starting mean along the velocity vector and extracts the initial step-size (sigma) from the weighted positional variance.
    """
    def __init__(self, x_count: int, true_lower_bounds_normalized: np.ndarray, true_upper_bounds_normalized: np.ndarray, warm_start_parameters: WarmStartParameters):
        self.x_count = x_count

        # True Bounds projected into the ROI-normalized space
        self.true_lower_bounds_normalized = true_lower_bounds_normalized
        self.true_upper_bounds_normalized = true_upper_bounds_normalized
        
        # Extract parameters
        self.sigma_factor = warm_start_parameters.sigma_factor
        
        
        self.velocity_contribution_feasibility, self.shrinkage_factor_feasibility = self.get_factors(warm_start_parameters.feasability_covariance.velocity_contribution.d2_value, warm_start_parameters.feasability_covariance.velocity_contribution.d40_value, warm_start_parameters.feasability_covariance.shrinkage.d2_value, warm_start_parameters.feasability_covariance.shrinkage.d40_value)
        self.feasibility_path = self.parameter_scaling(warm_start_parameters.feasability_covariance.path_trust.d2_value, warm_start_parameters.feasability_covariance.path_trust.d40_value)
        
        self.velocity_contribution_corner, self.shrinkage_factor_corner = self.get_factors(warm_start_parameters.corner_covariance.velocity_contribution.d2_value, warm_start_parameters.corner_covariance.velocity_contribution.d40_value, warm_start_parameters.corner_covariance.shrinkage.d2_value, warm_start_parameters.corner_covariance.shrinkage.d40_value)
        self.corner_path = self.parameter_scaling(warm_start_parameters.corner_covariance.path_trust.d2_value, warm_start_parameters.corner_covariance.path_trust.d40_value)
        
        self.velocity_contribution_gap, self.shrinkage_factor_gap = self.get_factors(warm_start_parameters.gap_covariance.velocity_contribution.d2_value, warm_start_parameters.gap_covariance.velocity_contribution.d40_value, warm_start_parameters.gap_covariance.shrinkage.d2_value, warm_start_parameters.gap_covariance.shrinkage.d40_value)
        self.gap_path = self.parameter_scaling(warm_start_parameters.gap_covariance.path_trust.d2_value, warm_start_parameters.gap_covariance.path_trust.d40_value)
    
        self.velocity_contribution_uhvi, self.shrinkage_factor_uhvi = self.get_factors(warm_start_parameters.uhvi_covariance.velocity_contribution.d2_value, warm_start_parameters.uhvi_covariance.velocity_contribution.d40_value, warm_start_parameters.uhvi_covariance.shrinkage.d2_value, warm_start_parameters.uhvi_covariance.shrinkage.d40_value)
        self.uhvi_path = self.parameter_scaling(warm_start_parameters.uhvi_covariance.path_trust.d2_value, warm_start_parameters.uhvi_covariance.path_trust.d40_value)
        
        self.velocity_contribution_wsum, self.shrinkage_factor_wsum = self.get_factors(warm_start_parameters.wsum_covariance.velocity_contribution.d2_value, warm_start_parameters.wsum_covariance.velocity_contribution.d40_value, warm_start_parameters.wsum_covariance.shrinkage.d2_value, warm_start_parameters.wsum_covariance.shrinkage.d40_value)
        self.wsum_path = self.parameter_scaling(warm_start_parameters.wsum_covariance.path_trust.d2_value, warm_start_parameters.wsum_covariance.path_trust.d40_value)

        
        self.initial_sigma_move = warm_start_parameters.initial_sigma_move
        self.min_sigma = warm_start_parameters.min_sigma
        self.max_sigma = warm_start_parameters.max_sigma
        self.sigma_fallback = warm_start_parameters.sigma_fallback
        self.tie_tolerance = warm_start_parameters.tie_tolerance
        
        # Heuristic for covariance estimation samples
        # Covariance parameters: D*(D+1)/2.
        self.n_warm_start: int = int(3 * x_count)
        
        # Radius values (Scaled by dimensionality for most)
        self.r_feasibility: float = warm_start_parameters.r_feasibility_perx * np.sqrt(self.x_count)
        self.r_uhvi: float = warm_start_parameters.r_uhvi_perx * np.sqrt(self.x_count)
        self.r_corner: float = warm_start_parameters.r_corner_perx * np.sqrt(self.x_count)
        self.r_exploration: float = warm_start_parameters.r_exploration_perx * np.sqrt(self.x_count)
        
        # Maximum number of points to extract from the archive before density reduction
        self.min_points = int(round(self.n_warm_start * 2.5))
        self.max_points_in_hypersphere = warm_start_parameters.max_points_in_hypersphere


    def parameter_scaling(self, value_d2: float, value_d40: float) -> float:
        """
        Scales the parameters based on the d2 and d40 value points.
        Minimum and maximum are 0.0 and 1.0.
        """
        t: float = (self.x_count - 2) / (40 - 2)
        sq_t: float = np.sqrt(t)
        
        value: float = value_d2 + (value_d40 - value_d2) * sq_t
        return np.clip(value, 0.0, 1.0)


    def get_factors(self, velocity_d2: float, velocity_d40: float, shrinkage_d2: float, shrinkage_d40: float) -> Tuple[float, float]:
        shrinkage = self.parameter_scaling(shrinkage_d2, shrinkage_d40)
        velocity = np.clip(self.parameter_scaling(velocity_d2, velocity_d40), 0.0, 1.0 - shrinkage)
        return velocity, shrinkage
        
   
    def _get_ranks(self, scores: np.ndarray, offset: int) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=int)
            
        ranks = np.zeros(len(scores), dtype=int)
        
        # NEW: Make tie tolerance relative to the score spread
        score_spread = np.max(scores) - np.min(scores)
        # If spread is 0, everything is a tie anyway. Otherwise, use a percentage (e.g., 0.1% of spread)
        dynamic_tie_tolerance = self.tie_tolerance * score_spread

        group_start_score = scores[0]
        group_rank = offset + 1
        ranks[0] = group_rank
        
        for i in range(1, len(scores)):
            if scores[i] - group_start_score <= dynamic_tie_tolerance:
                ranks[i] = group_rank
            else:
                group_start_score = scores[i]
                group_rank = offset + i + 1
                ranks[i] = group_rank
        return ranks

    def _compute_weights(self, number_of_points: int, ranking_list: np.ndarray) -> np.ndarray:
        """
        Computes logarithmic weights for CMA-ES covariance estimation.
        Tied ranks receive the mathematical average of the weights for their respective positions,
        ensuring that the search direction isn't biased by arbitrary sorting of identical values.
        """
        weights = np.log(number_of_points + 2.5) - np.log(np.arange(1, number_of_points + 1))
        weights = np.maximum(weights, 0.0)
        
        unique_ranks = np.unique(ranking_list)
        for rank in unique_ranks:
            mask = (ranking_list == rank)
            weights[mask] = np.mean(weights[mask])
            
        weights /= np.sum(weights)
        return weights
    
    
    def compute_initial_shift(self, c: np.ndarray, v_norm: np.ndarray, C: np.ndarray, sigma: float, beta: float) -> np.ndarray:
        """
        Computes the shifted starting mean x_0 such that the Mahalanobis distance 
        from x_0 back to c is exactly 'beta' standard deviations.
        """
        # 1. Compute C^{-1/2} using eigendecomposition for numerical stability
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # Clip eigenvalues to prevent division by zero or negative roots due to float inaccuracies
        eigenvalues = np.maximum(eigenvalues, 1e-14)
        
        D_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        C_inv_half = eigenvectors @ D_inv @ eigenvectors.T
        
        # 2. Project velocity into isotropic space
        z = C_inv_half @ v_norm
        z_norm = np.linalg.norm(z)
        
        # 3. Compute shift magnitude (alpha)
        if z_norm < 1e-12:
            alpha = 0.0  # Fallback if velocity is effectively zero
        else:
            alpha = beta * (sigma / z_norm)
            
        # 4. Shift the mean
        x_0 = c + alpha * v_norm
        
        return x_0
    
    

    def _compute_warm_start(self, x_0: np.ndarray, X_pool_norm: np.ndarray, ranking_list: np.ndarray, velocity_contribution: float, shrinkage_factor: float) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Computes the gradient-aligned covariance matrix, step size, and shifted mean.
        Uses a trace-normalized mixture of positional covariance and velocity covariance.
        Returns [x0, sigma, C, v_norm] for the start of the search.
        """
        n, d = X_pool_norm.shape
        used_sigma_factor = self.sigma_factor

        # Fallback for tiny pools
        if n < 2:
            fallback_x0 = X_pool_norm[0] if n == 1 else np.full(d, 0.5)
            fallback_x0 = np.clip(fallback_x0, self.true_lower_bounds_normalized, self.true_upper_bounds_normalized)
            return fallback_x0, self.sigma_fallback, np.eye(d), np.zeros(d)
            
        elif n < self.n_warm_start:
            used_sigma_factor *= np.sqrt(self.n_warm_start / n)

        # 1. Weights (accounting for ties)
        weights = self._compute_weights(n, ranking_list)

        # 2. Extract Velocity (Gradient of Improvement)
        m_unweighted = np.mean(X_pool_norm, axis=0)
        m_weighted = np.average(X_pool_norm, axis=0, weights=weights)
        
        v = m_weighted - m_unweighted
        v_len = np.linalg.norm(v)

        local_scale = np.max(np.std(X_pool_norm, axis=0))
        v_norm_successful: bool = False
        
        # 3. Construct Velocity Covariance
        if v_len > (1e-6 * local_scale) and v_len > 1e-15:
            v_norm = v / v_len
            C_V = np.outer(v_norm, v_norm)
            v_norm_successful = True
        else:
            v_norm = np.zeros(d)
            C_V = np.eye(d) / d  # Trace = 1 fallback

        # 4. Compute Weighted Positional Covariance
        try:
            C_W = np.cov(X_pool_norm, rowvar=False, aweights=weights)
        except:
            C_W = np.eye(d) * 1e-4

        C_W += np.eye(d) * 1e-9  # Small regularization for numerical stability

        # 5. Trace Normalization (Scale both to Trace = d)
        tr_W = np.trace(C_W)
        tr_V = np.trace(C_V)
        
        C_W_norm = C_W * (d / tr_W) if tr_W > 1e-12 else np.eye(d)
        C_V_norm = C_V * (d / tr_V) if tr_V > 1e-12 else np.eye(d)

        # 6. Mixture and Shrinkage
        C_final = (1.0 - velocity_contribution - shrinkage_factor) * C_W_norm + velocity_contribution * C_V_norm + shrinkage_factor * np.eye(d)

        # 7. Sigma Estimation
        if tr_W < 1e-16:
            sigma_w = 1e-4
        else:
            sigma_w = np.sqrt(tr_W / d)

        # Scale and clip the final sigma for CMA-ES
        sigma_final = np.clip(sigma_w * used_sigma_factor, self.min_sigma, self.max_sigma)

        # 8. Shift Mean using the Mahalanobis reachability logic
        if (v_norm_successful):
            x_shifted = self.compute_initial_shift(c=x_0, v_norm=v_norm, C=C_final, sigma=sigma_final, beta=self.initial_sigma_move)
        else:
            x_shifted = x_0

        # Clip against the true bounds
        x_shifted_clipped = np.clip(x_shifted, self.true_lower_bounds_normalized, self.true_upper_bounds_normalized)

        return x_shifted_clipped, sigma_final, C_final, v_norm
    
    
    
    

    def _get_points_in_hypersphere(self, archive: InternalArchive, c: np.ndarray, r: float, min_points: int) -> Tuple[np.ndarray, float]:
        """
        Finds indices of points within radius r of center c using the KDTree.
        If fewer than min_points are found, queries the exact min_points nearest neighbors.
        If more than max_points_in_hypersphere are found, returns a random subsample.
        Returns the points and the radius used.
        """
        # 1. O(log N) Spatial Query
        indices_in_radius = archive.query_ball_point(c, r)
        n_in_radius = indices_in_radius.size
        
        if n_in_radius >= min_points:
            # Safety Cap: Prevent massive RAM allocation in later steps
            if n_in_radius > self.max_points_in_hypersphere:
                return np.random.choice(indices_in_radius, size=self.max_points_in_hypersphere, replace=False), r
            return indices_in_radius, r
        
        # 2. Fallback: O(log N) Spatial Query for the exact 'min_points' nearest neighbors
        # Reduce the number of queried points, so that we don't expand the space too much
        _, closest_indices = archive.query_k_nearest(c, min_points // 2, distance_upper_bound=np.sqrt(self.x_count))
        closest_indices = closest_indices.ravel()
        # Remove invalid (-1) indices
        closest_indices_filtered = closest_indices[closest_indices != -1]
        # Combine results
        candidate_indices = np.unique(np.concatenate([indices_in_radius, closest_indices_filtered]))
        # Get coordinates of the candidates
        X_candidates = archive.get_norm_data()[candidate_indices]
        # Find the maximum distance from the center 'c'
        dists = np.linalg.norm(X_candidates - c, axis=1)
        effective_r = max(r, np.max(dists))
        
        return candidate_indices, effective_r

    def _compute_appropriate_distance(self, n_points: int, r: float) -> float:
        """
        Computes the resolution distance d such that N spheres of radius d 
        occupy exactly 30% of the volume of the search sphere of radius r.
        Formula: d = r * (0.3 / N)^(1/D)
        """
        if n_points < 2:
            return r
        
        used_dim = min(self.x_count, 4.0) 
        d = r * (0.3 / n_points)**(1.0 / used_dim)
        return d


    def _get_reduced_ranked_points(self, indices: np.ndarray, X_norm: np.ndarray, F_sub: np.ndarray, G_sub: np.ndarray, is_feas_mask: np.ndarray, d: float, scalarizer: Scalarizer) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedily reduces density by distance 'd' and returns ranked indices/ranks.
        Strictly prioritizes Feasible points over Infeasible ones.
        Uses pre-fetched subsets and O(1) feasibility masks for maximum performance.
        """
        if len(indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # 1. Split into Feasible and Infeasible sets using the pre-computed mask
        idx_feas = indices[is_feas_mask]
        idx_infeas = indices[~is_feas_mask]
        
        # 2. Score and Sort Feasible points
        if len(idx_feas) > 0:
            # Evaluate only the feasible subset
            scores_feas = scalarizer.evaluate_archive(F_sub[is_feas_mask], G_sub[is_feas_mask])
            order_feas = np.argsort(scores_feas)
            sorted_feas = idx_feas[order_feas]
            sorted_scores_feas = scores_feas[order_feas]
        else:
            sorted_feas = np.array([], dtype=int)
            sorted_scores_feas = np.array([], dtype=float)

        # 3. Score and Sort Infeasible points
        if len(idx_infeas) > 0:
            # Evaluate only the infeasible subset
            cv_infeas = scalarizer.get_cv(G_sub[~is_feas_mask])
            order_infeas = np.argsort(cv_infeas)
            sorted_infeas = idx_infeas[order_infeas]
            sorted_cv_infeas = cv_infeas[order_infeas]
        else:
            sorted_infeas = np.array([], dtype=int)
            sorted_cv_infeas = np.array([], dtype=float)

        # 4. Concatenate (Feasible strictly before Infeasible)
        sorted_indices = np.concatenate([sorted_feas, sorted_infeas]).astype(int)
        sorted_scores = np.concatenate([sorted_scores_feas, sorted_cv_infeas])
        
        # Track the boundary between feasible and infeasible in the sorted arrays
        n_feas_total = len(sorted_feas)

        # 5. Greedy Distance Reduction (LAZY EVALUATION)
        # Pre-extract the coordinates for the sorted candidates.
        X_sorted = X_norm[sorted_indices]
        d_sq = d**2 
        
        # Boolean mask to track which points are still valid (not too close to a selected point)
        valid_mask = np.ones(len(sorted_indices), dtype=bool)
        
        selected_indices = []
        selected_scores = []
        n_feas_selected = 0  # Counter for how many feasible points actually get selected
        
        for i in range(len(sorted_indices)):
            if len(selected_indices) >= self.n_warm_start:
                break
                
            if not valid_mask[i]:
                continue
                
            # Point i is valid and selected
            selected_indices.append(sorted_indices[i])
            selected_scores.append(sorted_scores[i])
            
            # If the index is from the feasible half, increment our counter
            if i < n_feas_total:
                n_feas_selected += 1
            
            # Invalidate future points lazily
            # Only check points after 'i' that are STILL valid
            valid_after_i = valid_mask[i+1:]
            if not np.any(valid_after_i):
                continue  # No more valid points left to invalidate
                
            # Get the absolute indices of the remaining valid candidates
            check_indices = (i + 1) + np.where(valid_after_i)[0]
            
            # Compute distances from the newly selected point to the remaining valid candidates
            # X_sorted[i:i+1] shape is (1, D), X_sorted[check_indices] shape is (V, D)
            dists_sq = np.sum((X_sorted[check_indices] - X_sorted[i])**2, axis=1)

            
            # Find which ones are too close and invalidate them
            too_close_mask = dists_sq < d_sq
            too_close_indices = check_indices[too_close_mask]
            valid_mask[too_close_indices] = False

        # 6. Compute Ranks
        final_indices = np.array(selected_indices, dtype=int)
        final_scores = np.array(selected_scores, dtype=float)
        
        # Because feasible points were strictly before infeasible points in sorted_indices,
        # they are also strictly before infeasible points in final_scores.
        ranks_f = self._get_ranks(final_scores[:n_feas_selected], offset=0)
        ranks_i = self._get_ranks(final_scores[n_feas_selected:], offset=n_feas_selected)
        
        if len(final_indices) > 0:
            final_ranks = np.concatenate([ranks_f, ranks_i])
        else:
            final_ranks = np.array([], dtype=int)
        
        return final_indices, final_ranks


    def GetWarmStartValues(self, archive: InternalArchive, f_U: np.ndarray, scalarizer: Scalarizer, points_of_interest: List[np.ndarray]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        """
        Main entry point for CMA-ES Warm Start.
        Returns [m, sigma, C, v_norm, path_trust]
        """
        mode = scalarizer.mode
        
        # UHVI defaults: Random start inside the ROI. 
        c = np.random.rand(self.x_count)
        r: float = self.r_uhvi
        d: float = 0.0
        velocity_contribution: float = self.velocity_contribution_uhvi
        shrinkage_factor: float = self.shrinkage_factor_uhvi
        path_trust: float = self.uhvi_path
        

        if (mode == ScalarizationMode.FEASIBILITY):
            r = self.r_feasibility
            velocity_contribution = self.velocity_contribution_feasibility
            shrinkage_factor = self.shrinkage_factor_feasibility
            path_trust = self.feasibility_path
        elif (mode == ScalarizationMode.CORNER):
            c = points_of_interest[0]
            r = self.r_corner
            velocity_contribution = self.velocity_contribution_corner
            shrinkage_factor = self.shrinkage_factor_corner
            path_trust = self.corner_path
        elif (mode == ScalarizationMode.GAP):
            p1, p2 = points_of_interest[0], points_of_interest[1]
            c = (p1 + p2) / 2.0
            L: float = np.linalg.norm(p1 - p2)
            r = 1.5 * L
            d = 0.334 * L 
            velocity_contribution = self.velocity_contribution_gap
            shrinkage_factor = self.shrinkage_factor_gap
            path_trust = self.gap_path
        elif (mode == ScalarizationMode.EXP):
            velocity_contribution = self.velocity_contribution_wsum
            shrinkage_factor = self.shrinkage_factor_wsum
            r = self.r_exploration
            c = points_of_interest[0]
            path_trust = self.wsum_path

                
        # 2. Filter by Hypersphere using the Archive's KDTree
        candidate_indices, effective_r = self._get_points_in_hypersphere(archive, c, r, min_points=self.min_points)
        
        # 3. Compute Resolution Distance
        d_max = self._compute_appropriate_distance(len(candidate_indices), effective_r)
        
        if d == 0.0:
            d = d_max
        else:
            d = min(d, d_max)
        
        # 4. Lazy Loading & O(1) Feasibility Check
        # Fetch ONLY the data for the candidate points, saving massive memory and compute overhead.
        F_sub = archive.get_sanitized_f_by_indices(candidate_indices, f_U)
        G_sub = archive.get_g_by_indices(candidate_indices)
        is_feas_mask = archive.get_feasible_mask()[candidate_indices]
        
        # 5. Density Reduction and Ranking
        X_norm = archive.get_norm_data()
        final_indices, ranking_list = self._get_reduced_ranked_points(candidate_indices, X_norm, F_sub, G_sub, is_feas_mask, d, scalarizer)
        
        # 6. Compute Final CMA Parameters
        X_pool = X_norm[final_indices]
        
        
        m, sigma, C, v_norm = self._compute_warm_start(x_0 = c, X_pool_norm = X_pool, ranking_list = ranking_list, velocity_contribution = velocity_contribution, shrinkage_factor = shrinkage_factor)
        return (m, sigma, C, v_norm, path_trust)
