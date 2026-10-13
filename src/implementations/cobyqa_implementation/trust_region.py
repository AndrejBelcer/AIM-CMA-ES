import numpy as np
from scipy.optimize import lsq_linear
from typing import Tuple

from src.implementations.cobyqa_implementation.settings import COBYQAParameters
from src.implementations.cobyqa_implementation.models import ModelManager
from src.implementations.cobyqa_implementation.subsolvers import normal_byrd_omojokun, tangential_byrd_omojokun_unconstrained, tangential_byrd_omojokun_constrained, cauchy_geometry, spider_geometry, qr_tangential_byrd_omojokun, get_arrays_tol

TINY = np.finfo(float).tiny
EPS = np.finfo(float).eps

class TrustRegionState:
    """
    Manages the trust-region framework, penalty parameters, Lagrange multipliers, 
    and step generation for COBYQA.
    """
    def __init__(self, models: ModelManager, lower_bounds: np.ndarray, upper_bounds: np.ndarray, params: COBYQAParameters):
        self.models = models
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.params = params
        
        self.n = models.n
        self.g_count = models.g_count
        self.h_count = models.h_count
        self.has_constraints = (self.g_count > 0) or (self.h_count > 0)
        
        self.penalty = 0.0
        self.resolution = params.rhobeg
        self._radius = self.resolution
        
        self.lm_g = np.zeros(self.g_count)
        self.lm_h = np.zeros(self.h_count)
        
        self.best_index = 0
        self.set_best_index()
        self.set_multipliers(self.x_best)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = value
        if self._radius <= self.params.decrease_radius_threshold * self.resolution:
            self._radius = self.resolution

    @property
    def x_best(self) -> np.ndarray:
        return self.models.interpolation.point(self.best_index)

    @property
    def f_best(self) -> float:
        return self.models.f_vals[self.best_index]

    @property
    def g_best(self) -> np.ndarray:
        return self.models.g_vals[self.best_index, :] if self.g_count > 0 else np.array([])

    @property
    def h_best(self) -> np.ndarray:
        return self.models.h_vals[self.best_index, :] if self.h_count > 0 else np.array([])

    # ==========================================
    # Merit Function and SQP Approximations
    # ==========================================

    def maxcv(self, x: np.ndarray, g_val: np.ndarray, h_val: np.ndarray) -> float:
        """Calculates the maximum constraint violation."""
        v = 0.0
        if self.g_count > 0:
            v = max(v, np.max(np.maximum(g_val, 0.0), initial=0.0))
        if self.h_count > 0:
            v = max(v, np.max(np.abs(h_val), initial=0.0))
        return v

    def merit(self, x: np.ndarray, f_val: float, g_val: np.ndarray, h_val: np.ndarray) -> float:
        """Evaluates the L1 merit function."""
        m_val = f_val
        if self.penalty > 0.0:
            viol = 0.0
            if self.g_count > 0:
                viol += np.sum(np.maximum(g_val, 0.0))
            if self.h_count > 0:
                viol += np.sum(np.abs(h_val))
            m_val += self.penalty * viol
        return m_val

    def lag_model_hess_prod(self, v: np.ndarray) -> np.ndarray:
        """Right product of the Lagrangian Hessian with a vector."""
        res = self.models.f_hess_prod(v)
        if self.g_count > 0:
            res += self.lm_g @ self.models.g_hess_prod(v)
        if self.h_count > 0:
            res += self.lm_h @ self.models.h_hess_prod(v)
        return res

    def sqp_fun(self, step: np.ndarray) -> float:
        """Quadratic approximation of the objective function for the SQP subproblem."""
        return step @ (self.models.f_grad(self.x_best) + 0.5 * self.lag_model_hess_prod(step))

    def sqp_g(self, step: np.ndarray) -> np.ndarray:
        """Linearization of the inequality constraints."""
        if self.g_count == 0: return np.array([])
        return self.models.g(self.x_best) + self.models.g_grad(self.x_best) @ step

    def sqp_h(self, step: np.ndarray) -> np.ndarray:
        """Linearization of the equality constraints."""
        if self.h_count == 0: return np.array([])
        return self.models.h(self.x_best) + self.models.h_grad(self.x_best) @ step

    def get_constraint_linearizations(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns A_ineq, b_ineq, A_eq, b_eq for the linearizations at x."""
        A_ineq = self.models.g_grad(x) if self.g_count > 0 else np.empty((0, self.n))
        b_ineq = -self.models.g(x) if self.g_count > 0 else np.empty(0)
        A_eq = self.models.h_grad(x) if self.h_count > 0 else np.empty((0, self.n))
        b_eq = -self.models.h(x) if self.h_count > 0 else np.empty(0)
        return A_ineq, b_ineq, A_eq, b_eq

    # ==========================================
    # Step Generation
    # ==========================================

    def get_trust_region_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the normal and tangential steps using the Byrd-Omojokun approach."""
        A_ineq, b_ineq, A_eq, b_eq = self.get_constraint_linearizations(self.x_best)
        xl = self.lower_bounds - self.x_best
        xu = self.upper_bounds - self.x_best

        # 1. Normal Step (to minimize constraint violation)
        if self.has_constraints:
            radius_normal = self.params.byrd_omojokun_factor * self.radius
            normal_step = normal_byrd_omojokun(A_ineq, b_ineq, A_eq, b_eq, xl, xu, radius_normal, self.params.improve_tcg)
        else:
            normal_step = np.zeros(self.n)

        # 2. Tangential Step (to minimize objective while staying in null space of active constraints)
        radius_tangential = np.sqrt(max(self.radius**2.0 - normal_step @ normal_step, 0.0))
        xl -= normal_step
        xu -= normal_step
        
        g_best_grad = self.models.f_grad(self.x_best) + self.lag_model_hess_prod(normal_step)
        
        if self.has_constraints:
            b_ineq = np.maximum(b_ineq - A_ineq @ normal_step, 0.0)
            tangential_step = tangential_byrd_omojokun_constrained(
                g_best_grad, self.lag_model_hess_prod, xl, xu, A_ineq, b_ineq, A_eq, radius_tangential, self.params.improve_tcg
            )
        else:
            tangential_step = tangential_byrd_omojokun_unconstrained(
                g_best_grad, self.lag_model_hess_prod, xl, xu, radius_tangential, self.params.improve_tcg
            )

        return normal_step, tangential_step

    def get_second_order_correction_step(self, step: np.ndarray) -> np.ndarray:
        """Computes a second-order correction step to combat the Maratos effect."""
        A_ineq, b_ineq, A_eq, b_eq = self.get_constraint_linearizations(self.x_best)
        xl = self.lower_bounds - self.x_best
        xu = self.upper_bounds - self.x_best
        radius = np.linalg.norm(step)
        return normal_byrd_omojokun(A_ineq, b_ineq, A_eq, b_eq, xl, xu, radius, self.params.improve_tcg)

    def get_geometry_step(self, k_new: int) -> np.ndarray:
        """Computes a step to improve the geometry of the interpolation set."""
        # Build the k_new-th Lagrange polynomial
        coord_vec = np.squeeze(np.eye(1, self.models.npt, k_new))
        
        # We need the gradient and curvature of the Lagrange polynomial
        rhs = np.block([[coord_vec, np.zeros(self.n + 1)]]).T
        from .models import solve_interpolation_systems
        x_sol, _ = solve_interpolation_systems(self.models.interpolation, rhs)
        g_lag = x_sol[self.models.npt + 1:, 0]
        i_hess_lag = x_sol[:self.models.npt, 0]
        
        def lag_curv(v):
            return i_hess_lag @ (self.models.interpolation.xpt.T @ v) ** 2.0

        xl = self.lower_bounds - self.x_best
        xu = self.upper_bounds - self.x_best

        # 1. Simple constrained Cauchy step
        step = cauchy_geometry(0.0, g_lag, lag_curv, xl, xu, self.radius)
        sigma = self.models.determinants(self.x_best + step, k_new)

        # 2. Spider geometry step
        xpt = self.models.interpolation.xpt - self.models.interpolation.xpt[:, self.best_index, np.newaxis]
        xpt[:, [0, self.best_index]] = xpt[:, [self.best_index, 0]]
        step_alt = spider_geometry(0.0, g_lag, lag_curv, xpt[:, 1:], xl, xu, self.radius)
        sigma_alt = self.models.determinants(self.x_best + step_alt, k_new)
        
        if abs(sigma_alt) > abs(sigma):
            step = step_alt
            sigma = sigma_alt

        # 3. Tangential Cauchy step (if constrained)
        if self.has_constraints:
            A_ineq, b_ineq, A_eq, b_eq = self.get_constraint_linearizations(self.x_best)
            tol_bd = get_arrays_tol(xl, xu)
            tol_ub = get_arrays_tol(b_ineq) if self.g_count > 0 else 0.0
            
            free_xl = xl <= -tol_bd
            free_xu = xu >= tol_bd
            free_ub = b_ineq >= tol_ub if self.g_count > 0 else np.empty(0, dtype=bool)

            n_act, q = qr_tangential_byrd_omojokun(A_ineq, A_eq, free_xl, free_xu, free_ub)
            g_lag_proj = q[:, n_act:] @ (q[:, n_act:].T @ g_lag)
            norm_g_lag_proj = np.linalg.norm(g_lag_proj)
            
            if 0 < n_act < self.n and norm_g_lag_proj > TINY * self.radius:
                step_alt = (self.radius / norm_g_lag_proj) * g_lag_proj
                if lag_curv(step_alt) < 0.0:
                    step_alt = -step_alt

                # Check constraint violation of this step
                cbd = np.block([xl - step_alt, step_alt - xu])
                cub = A_ineq @ step_alt - b_ineq if self.g_count > 0 else np.array([])
                ceq = A_eq @ step_alt - b_eq if self.h_count > 0 else np.array([])
                
                maxcv_val = max(np.max(array, initial=0.0) for array in [cbd, cub, np.abs(ceq)])
                
                tol = np.max(np.abs(step_alt[~free_xl]), initial=0.0)
                tol = np.max(np.abs(step_alt[~free_xu]), initial=tol)
                if self.g_count > 0:
                    tol = np.max(np.abs(A_ineq[~free_ub, :] @ step_alt), initial=tol)
                tol = min(10.0 * tol, 1e-2 * np.linalg.norm(step_alt))
                
                if maxcv_val <= tol:
                    sigma_alt = self.models.determinants(self.x_best + step_alt, k_new)
                    if abs(sigma_alt) >= 0.1 * abs(sigma):
                        step = np.clip(step_alt, xl, xu)

        return step

    # ==========================================
    # State Updates
    # ==========================================

    def set_best_index(self):
        """Finds the index of the best point in the interpolation set based on the merit function."""
        best_idx = self.best_index
        m_best = self.merit(self.x_best, self.f_best, self.g_best, self.h_best)
        r_best = self.maxcv(self.x_best, self.g_best, self.h_best)
        
        tol = 10.0 * EPS * max(self.n, self.models.npt) * max(abs(m_best), 1.0)
        
        for k in range(self.models.npt):
            if k != self.best_index:
                x_val = self.models.interpolation.point(k)
                g_val = self.models.g_vals[k, :] if self.g_count > 0 else np.array([])
                h_val = self.models.h_vals[k, :] if self.h_count > 0 else np.array([])
                
                m_val = self.merit(x_val, self.models.f_vals[k], g_val, h_val)
                r_val = self.maxcv(x_val, g_val, h_val)
                
                if m_val < m_best or (m_val < m_best + tol and r_val < r_best):
                    best_idx = k
                    m_best = m_val
                    r_best = r_val
                    
        self.best_index = best_idx

    def get_index_to_remove(self, x_new: np.ndarray = None) -> Tuple[int, float]:
        """Finds the index of the interpolation point to remove."""
        dist_sq = np.sum((self.models.interpolation.xpt - self.models.interpolation.xpt[:, self.best_index, np.newaxis]) ** 2.0, axis=0)
        
        if x_new is None:
            sigma = 1.0
            weights = dist_sq
        else:
            sigma = self.models.determinants(x_new)
            weights = np.maximum(1.0, dist_sq / max(self.params.low_radius_factor * self.radius, self.resolution) ** 2.0) ** 3.0
            weights[self.best_index] = -1.0  # Never remove the best point
            
        k_max = np.argmax(weights * np.abs(sigma))
        return k_max, np.sqrt(dist_sq[k_max])

    def get_reduction_ratio(self, step: np.ndarray, f_val: float, g_val: np.ndarray, h_val: np.ndarray) -> float:
        """Calculates the ratio of actual reduction to predicted reduction."""
        merit_old = self.merit(self.x_best, self.f_best, self.g_best, self.h_best)
        merit_new = self.merit(self.x_best + step, f_val, g_val, h_val)
        
        merit_model_old = self.merit(self.x_best, 0.0, self.models.g(self.x_best), self.models.h(self.x_best))
        merit_model_new = self.merit(self.x_best + step, self.sqp_fun(step), self.sqp_g(step), self.sqp_h(step))
        
        if abs(merit_model_old - merit_model_new) > TINY * abs(merit_old - merit_new):
            return (merit_old - merit_new) / abs(merit_model_old - merit_model_new)
        return -1.0

    def update_radius(self, step: np.ndarray, ratio: float):
        """Updates the trust-region radius based on the reduction ratio."""
        s_norm = np.linalg.norm(step)
        if ratio <= self.params.low_ratio:
            self.radius *= self.params.decrease_radius_factor
        elif ratio <= self.params.high_ratio:
            self.radius = max(self.params.decrease_radius_factor * self.radius, s_norm)
        else:
            self.radius = min(
                self.params.increase_radius_factor * self.radius,
                max(self.params.decrease_radius_factor * self.radius, self.params.increase_radius_threshold * s_norm)
            )

    def enhance_resolution(self):
        """Reduces the resolution (rho) when the trust region becomes too small."""
        if self.params.large_resolution_threshold * self.params.rhoend < self.resolution:
            self.resolution *= self.params.decrease_resolution_factor
        elif self.params.moderate_resolution_threshold * self.params.rhoend < self.resolution:
            self.resolution = np.sqrt(self.resolution * self.params.rhoend)
        else:
            self.resolution = self.params.rhoend

        self._radius = max(self.params.decrease_radius_factor * self._radius, self.resolution)

    # ==========================================
    # Penalty and Multipliers
    # ==========================================

    def increase_penalty(self, step: np.ndarray) -> bool:
        """Increases the penalty parameter if the SQP step predicts a large constraint improvement."""
        if not self.has_constraints:
            return True
            
        A_ineq, b_ineq, A_eq, b_eq = self.get_constraint_linearizations(self.x_best)
        
        viol_old = 0.0
        viol_new = 0.0
        if self.g_count > 0:
            viol_old += np.linalg.norm(np.maximum(0.0, -b_ineq))
            viol_new += np.linalg.norm(np.maximum(0.0, A_ineq @ step - b_ineq))
        if self.h_count > 0:
            viol_old += np.linalg.norm(b_eq)
            viol_new += np.linalg.norm(A_eq @ step - b_eq)
            
        viol_diff = max(viol_old - viol_new, 0.0)
        sqp_val = self.sqp_fun(step)

        threshold = np.linalg.norm(np.block([self.lm_g, self.lm_h]))
        if abs(viol_diff) > TINY * abs(sqp_val):
            threshold = max(threshold, sqp_val / viol_diff)
            
        best_index_save = self.best_index
        if self.penalty <= self.params.penalty_increase_threshold * threshold:
            self.penalty = max(self.params.penalty_increase_factor * threshold, 1.0)
            self.set_best_index()
            
        return best_index_save == self.best_index

    def decrease_penalty(self):
        """Decreases the penalty parameter if constraints are well satisfied."""
        if not self.has_constraints:
            return
            
        # Calculate max constraint values over the interpolation set
        r_val = []
        if self.g_count > 0:
            r_val.append(self.models.g_vals)
        if self.h_count > 0:
            r_val.append(self.models.h_vals)
            r_val.append(-self.models.h_vals)
            
        r_val = np.block(r_val)
        c_min = np.nanmin(r_val, axis=0)
        c_max = np.nanmax(r_val, axis=0)
        
        indices = c_min < self.params.threshold_ratio_constraints * c_max
        if np.any(indices):
            f_min = np.nanmin(self.models.f_vals)
            f_max = np.nanmax(self.models.f_vals)
            c_min_neg = np.minimum(0.0, c_min[indices])
            c_diff = np.min(c_max[indices] - c_min_neg)
            penalty = (f_max - f_min) / c_diff if c_diff > TINY * (f_max - f_min) else np.inf
        else:
            penalty = 0.0
            
        self.penalty = min(self.penalty, penalty)
        self.set_best_index()

    def set_multipliers(self, x: np.ndarray):
        """Estimates the Lagrange multipliers by solving a bounded least-squares problem."""
        if not self.has_constraints:
            return
            
        incl_xl = self.lower_bounds >= x
        incl_xu = self.upper_bounds <= x
        incl_g = self.g_best >= 0.0 if self.g_count > 0 else np.empty(0, dtype=bool)
        
        m_xl = np.count_nonzero(incl_xl)
        m_xu = np.count_nonzero(incl_xu)
        m_g = np.count_nonzero(incl_g)

        if m_g + self.h_count > 0:
            identity = np.eye(self.n)
            blocks = [-identity[incl_xl, :], identity[incl_xu, :]]
            
            if self.g_count > 0:
                blocks.append(self.models.g_grad(x)[incl_g, :])
            if self.h_count > 0:
                blocks.append(self.models.h_grad(x))
                
            c_jac = np.r_[tuple(blocks)]
            g_best = self.models.f_grad(x)
            
            xl_lm = np.full(c_jac.shape[0], -np.inf)
            xl_lm[: m_xl + m_xu + m_g] = 0.0
            
            res = lsq_linear(c_jac.T, -g_best, bounds=(xl_lm, np.inf), method="bvls")

            if self.g_count > 0:
                self.lm_g[incl_g] = res.x[m_xl + m_xu : m_xl + m_xu + m_g]
                self.lm_g[~incl_g] = 0.0
            if self.h_count > 0:
                self.lm_h[:] = res.x[m_xl + m_xu + m_g :]
