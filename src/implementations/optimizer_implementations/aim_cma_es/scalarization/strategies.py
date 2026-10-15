import numpy as np
from scipy.special import softplus
from typing import Optional

from src.utilities.indicators.HVO.candidate_gap_optimized import Candidate_GapHV_Optimized
from src.utilities.indicators.HVO.candidate_uhvi_optimized import Candidate_UHVI_Optimized

from src.implementations.optimizer_implementations.aim_cma_es.scalarization.scalarizer_base import Scalarizer
from src.implementations.optimizer_implementations.aim_cma_es.enums import CVType, ScalarizationMode
from src.implementations.optimizer_implementations.aim_cma_es.state_tracker import StateTracker

# ==============================================================================
# STRATEGY IMPLEMENTATIONS
# ==============================================================================

class FeasibilityScalarizer(Scalarizer):
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.FEASIBILITY)

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = np.zeros(len(f_raw))
        
        # 1. Check Feasibility
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0, axis=1)
        else:
            is_infeas = np.zeros(len(f_raw), dtype=bool)
            
        is_feas = ~is_infeas

        # 2. Infeasible: Return CV
        if np.any(is_infeas):
            scores[is_infeas] = self.get_cv(g_raw[is_infeas])

        # 3. Feasible: Return -Prod(max(1e-9, 1.0 - y'))
        if np.any(is_feas):
            f_feas = f_norm[is_feas]
            # Term: 1.0 - y'
            terms = 1.0 - f_feas
            # Clamp to ensure positivity and prevent sign flipping
            terms = np.maximum(terms, 1e-9)
            # Product and Negate
            scores[is_feas] = -np.prod(terms, axis=1)

        return scores

    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.get_cv(g_raw)

    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.get_cv(g_raw)

    def get_string_description(self):
        string_array = np.array2string(self.g_divisor, formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"Feasibility search. Current g_divisor: {string_array}"


class CornerSearchScalarizer(Scalarizer):
    """Corner Search: minimize one normalized objective."""

    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, objective_index: int):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.CORNER_SEARCH)
        self.objective_index = objective_index

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = f_norm[:, self.objective_index]     # view into f_norm (shape (N,))

        N = scores.shape[0]
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0.0, axis=1)
        else:
            is_infeas = np.zeros(N, dtype=bool)

        # Only work on infeasible rows (K elements)
        if is_infeas.any():
            cv_vals = self.get_cv(g_raw[is_infeas])   # shape (K,)
            # work on a small length-K temporary (boolean indexing returns a copy)
            tmp = scores[is_infeas].copy()            # tmp shape (K,)
            # clamp to [-1, +inf] (i.e. set any values < -1 to -1)
            np.maximum(tmp, -1.0, out=tmp)            # in-place on tmp
            # add cv in-place
            tmp += cv_vals                            # in-place addition
            # apply softplus in-place on tmp (softplus supports out=)
            softplus(tmp, out=tmp)
            # write back to s for those indices (overwriting f_norm)
            scores[is_infeas] = tmp

        return scores

    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = f_norm[:, self.objective_index]
        return scores
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)
        

    def get_string_description(self):
        return f"Corner Search, objective [{self.objective_index}]."


class ExploitationScalarizer(Scalarizer):
    """Exploitation: target a front gap using the article's gap/HVI scalarization."""

    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, target_F_raw: np.ndarray, hv_reference_factor: float = 1.2):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.EXPLOITATION)
        
        # Keep raw value for possible tabu
        self.target_F_raw = target_F_raw        
        self.indicator = Candidate_GapHV_Optimized(self.normalize(state.frozen_front_F), np.ones(self.f_count) * hv_reference_factor, self.normalize(target_F_raw))

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = self.indicator.calculate_candidate_score(f_norm)

        N = scores.shape[0]
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0.0, axis=1)
        else:
            is_infeas = np.zeros(N, dtype=bool)

        if is_infeas.any():
            cv_vals = self.get_cv(g_raw[is_infeas]) 
            tmp = scores[is_infeas].copy()
            np.maximum(tmp, -1.0, out=tmp)
            tmp += cv_vals
            softplus(tmp, out=tmp)
            scores[is_infeas] = tmp

        return scores
    
    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        return self.indicator.calculate_archive_scores(f_norm)
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        string_array = np.array2string(self.target_F_raw, formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"Exploitation; gap target: {string_array}."


class GlobalExplorationWeightedSumScalarizer(Scalarizer):
    """
    Weighted Sum Scalarizer for global exploration.
    Normalizes objectives and applies a linear weight vector.
    Subtracts the best weighted sum value on the frozen front, so that the current best point has score 0 and improvements are negative.
    """
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, weights: np.ndarray, frozen_front_F_raw: np.ndarray):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.GLOBAL_EXPLORATION)
        self.weights = np.asarray(weights, dtype=float)
        self.front_score_offset = float(np.min(self.unshifted_weighted_sum(frozen_front_F_raw)))

    def unshifted_weighted_sum(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the weighted scalarization on raw objectives.
        Normalizes f, then applies weights.
        Returns: w^T * f_norm, shape (N,)
        """
        f_norm = self.normalize(f_raw)
        return f_norm @ self.weights

    def shifted_weighted_sum(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the weighted scalarization, shifted so that the best point on the frozen front has score 0.
        Improvements over the current front are negative.
        Returns: w^T * f_norm - front_score_offset, shape (N,)
        """
        return self.unshifted_weighted_sum(f_raw) - self.front_score_offset

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        scores = self.shifted_weighted_sum(f_raw)

        N = scores.shape[0]
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0.0, axis=1)
        else:
            is_infeas = np.zeros(N, dtype=bool)

        if is_infeas.any():
            cv_vals = self.get_cv(g_raw[is_infeas])
            tmp = scores[is_infeas].copy()
            np.maximum(tmp, -1.0, out=tmp)
            tmp += cv_vals
            softplus(tmp, out=tmp)
            scores[is_infeas] = tmp

        return scores

    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.shifted_weighted_sum(f_raw)
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        w_str = np.array2string(self.weights, formatter={'float_kind': lambda x: f"{x:.3f}"})
        return f"Global Exploration (weighted sum), weights: w = {w_str}. Front score offset: {self.front_score_offset:.4e}"


class GlobalExplorationChebyshevScalarizer(Scalarizer):
    """
    Augmented Chebyshev Scalarizer for global exploration.
    Normalizes objectives and applies a Chebyshev weight vector.
    Subtracts the best value on the frozen front, so that the current best point has score 0 and improvements are negative.
    Uses an augmented term (augmentation_coefficient) to guarantee strict Pareto compliance even in flat regions.
    """
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, weights: np.ndarray, frozen_front_F_raw: np.ndarray, augmentation_coefficient: float = 0.01):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.GLOBAL_EXPLORATION)
        self.weights = np.asarray(weights, dtype=float)
        self.augmentation_coefficient = augmentation_coefficient  # Set before evaluating the unshifted score.
        self.front_score_offset = float(np.min(self.unshifted_augmented_chebyshev(frozen_front_F_raw)))

    def unshifted_augmented_chebyshev(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the Augmented Chebyshev scalarization on raw objectives.
        Normalizes f, then applies weights.
        Returns: max(w * f_norm) + augmentation_coefficient * sum(w * f_norm), shape (N,)
        """
        f_norm = self.normalize(f_raw)
        
        # Element-wise multiplication (NOT dot product). 
        # f_norm is (N, M) and weights is (M,), so this broadcasts perfectly.
        weighted_f = f_norm * self.weights
        
        # The Chebyshev max term (L_infinity norm without absolute value)
        max_term = np.max(weighted_f, axis=1)
        
        # The augmented linear term to ensure strict Pareto compliance
        sum_term = np.sum(weighted_f, axis=1)
        
        return max_term + self.augmentation_coefficient * sum_term

    def shifted_augmented_chebyshev(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the Chebyshev scalarization, shifted so that the best point on the frozen front has score 0.
        Improvements over the current front are negative.
        Returns: raw_score - front_score_offset, shape (N,)
        """
        return self.unshifted_augmented_chebyshev(f_raw) - self.front_score_offset

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        scores = self.shifted_augmented_chebyshev(f_raw)

        N = scores.shape[0]
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0.0, axis=1)
        else:
            is_infeas = np.zeros(N, dtype=bool)

        if is_infeas.any():
            cv_vals = self.get_cv(g_raw[is_infeas])
            tmp = scores[is_infeas].copy()
            np.maximum(tmp, -1.0, out=tmp)
            tmp += cv_vals
            softplus(tmp, out=tmp)
            scores[is_infeas] = tmp

        return scores

    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.shifted_augmented_chebyshev(f_raw)
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        string_array = np.array2string(self.weights, formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"Global Exploration (augmented Chebyshev), weights: {string_array}."
        

class LocalExplorationScalarizer(Scalarizer):
    """Local Exploration using uncrowded hypervolume improvement (UHVI)."""

    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, hv_reference_factor: float = 1.2):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.LOCAL_EXPLORATION)
        
        self.indicator = Candidate_UHVI_Optimized(self.normalize(state.frozen_front_F), np.ones(self.f_count) * hv_reference_factor)

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = self.indicator.calculate_candidate_score(f_norm)

        N = scores.shape[0]
        if g_raw.shape[1] > 0:
            is_infeas = np.any(g_raw > 0.0, axis=1)
        else:
            is_infeas = np.zeros(N, dtype=bool)

        if is_infeas.any():
            cv_vals = self.get_cv(g_raw[is_infeas]) 
            tmp = scores[is_infeas].copy()
            np.maximum(tmp, -1.0, out=tmp)
            tmp += cv_vals
            softplus(tmp, out=tmp)
            scores[is_infeas] = tmp

        return scores

    def evaluate_archive(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        return self.indicator.calculate_archive_scores(f_norm)

    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        return f"Local Exploration (UHVI). Current empirical HV: {self.indicator.frozen_score}"


    
