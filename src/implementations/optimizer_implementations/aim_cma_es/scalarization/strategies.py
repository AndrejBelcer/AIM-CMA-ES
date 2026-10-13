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


class CornerScalarizer(Scalarizer):
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, axis: int):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.CORNER)
        self.axis = axis

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        f_norm = self.normalize(f_raw)
        scores = f_norm[:, self.axis]     # view into f_norm (shape (N,))

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
        scores = f_norm[:, self.axis]
        return scores
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)
        

    def get_string_description(self):
        return f"Corner [{self.axis}] search."


class GapScalarizer(Scalarizer):
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, target_raw: np.ndarray, hv_reference_factor: float = 1.2):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.GAP)
        
        # Keep raw value for possible tabu
        self.target_raw = target_raw        
        self.indicator = Candidate_GapHV_Optimized(self.normalize(state.frozen_front_F), np.ones(self.f_count) * hv_reference_factor, self.normalize(target_raw))

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
        string_array = np.array2string(self.target_raw, formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"Gap {string_array} fill."


class WSumScalarizer(Scalarizer):
    """
    Weighted Sum Scalarizer for global exploration.
    Normalizes objectives and applies a linear weight vector.
    Subtracts the best weighted sum value on the frozen front, so that the current best point has score 0 and improvements are negative.
    """
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, weights: np.ndarray, current_front: np.ndarray):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.EXP)
        self.weights = np.asarray(weights, dtype=float)
        self.best_value = float(np.min(self.weight_sum_raw(current_front)))

    def weight_sum_raw(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the weighted scalarization on raw objectives.
        Normalizes f, then applies weights.
        Returns: w^T * f_norm, shape (N,)
        """
        f_norm = self.normalize(f_raw)
        return f_norm @ self.weights

    def weight_sum(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the weighted scalarization, shifted so that the best point on the frozen front has score 0.
        Improvements over the current front are negative.
        Returns: w^T * f_norm - best_value, shape (N,)
        """
        return self.weight_sum_raw(f_raw) - self.best_value

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        scores = self.weight_sum(f_raw)

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
        return self.weight_sum(f_raw)
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        w_str = np.array2string(self.weights, formatter={'float_kind': lambda x: f"{x:.3f}"})
        return f"Weighted Sum Scalarization with weights: w = {w_str}. Best front value: {self.best_value:.4e}"


class CSumScalarizer(Scalarizer):
    """
    Augmented Chebyshev Scalarizer for global exploration.
    Normalizes objectives and applies a Chebyshev weight vector.
    Subtracts the best value on the frozen front, so that the current best point has score 0 and improvements are negative.
    Uses an augmented term (rho) to guarantee strict Pareto compliance even in flat regions.
    """
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, weights: np.ndarray, current_front: np.ndarray, rho: float = 0.01):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.EXP)
        self.weights = np.asarray(weights, dtype=float)
        self.rho = rho  # Save rho before calling weight_sum_raw
        self.best_value = float(np.min(self.cheb_weight_sum_raw(current_front)))

    def cheb_weight_sum_raw(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the Augmented Chebyshev scalarization on raw objectives.
        Normalizes f, then applies weights.
        Returns: max(w * f_norm) + rho * sum(w * f_norm), shape (N,)
        """
        f_norm = self.normalize(f_raw)
        
        # Element-wise multiplication (NOT dot product). 
        # f_norm is (N, M) and weights is (M,), so this broadcasts perfectly.
        weighted_f = f_norm * self.weights
        
        # The Chebyshev max term (L_infinity norm without absolute value)
        max_term = np.max(weighted_f, axis=1)
        
        # The augmented linear term to ensure strict Pareto compliance
        sum_term = np.sum(weighted_f, axis=1)
        
        return max_term + self.rho * sum_term

    def cheb_weight_sum(self, f_raw: np.ndarray) -> np.ndarray:
        """
        Computes the Chebyshev scalarization, shifted so that the best point on the frozen front has score 0.
        Improvements over the current front are negative.
        Returns: raw_score - best_value, shape (N,)
        """
        return self.cheb_weight_sum_raw(f_raw) - self.best_value

    def evaluate(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        scores = self.cheb_weight_sum(f_raw)

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
        return self.cheb_weight_sum(f_raw)
    
    def evaluate_archive_full(self, f_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        return self.evaluate(f_raw, g_raw)

    def get_string_description(self):
        string_array = np.array2string(self.weights, formatter={'float_kind': lambda x: f"{x:.2e}"})
        return f"Chebyshew scalarization: [{string_array}] weight."
        

class UHVIScalarizer(Scalarizer):
    def __init__(self, state: StateTracker, cv_type: CVType, g_divisor: np.ndarray, hv_reference_factor: float = 1.2):
        super().__init__(state, cv_type, g_divisor, ScalarizationMode.UHVI)
        
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
        return f"UHVI initialized. Current empirical HV: {self.indicator.frozen_score}"


    
