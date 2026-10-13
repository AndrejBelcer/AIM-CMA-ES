import numpy as np
from scipy.linalg import eigh

EPS = np.finfo(float).eps

def solve_interpolation_systems(interpolation, rhs: np.ndarray):
    """
    Solves the interpolation systems using Eigen decomposition.
    Returns (solutions, ill_conditioned_flag).
    """
    # FAST CACHE CHECK: O(1) integer comparison instead of O(N) array_equal
    _cache = interpolation._lhs_cache
    if _cache is not None and _cache["counter"] == interpolation._update_counter:
        return _cache["H"] @ rhs, _cache["ill_conditioned"]

    n, npt = interpolation.xpt.shape
    scale = np.max(np.linalg.norm(interpolation.xpt, axis=0), initial=EPS)
    xpt_scale = interpolation.xpt / scale

    a = np.zeros((npt + n + 1, npt + n + 1))
    a[:npt, :npt] = 0.5 * (xpt_scale.T @ xpt_scale) ** 2.0
    a[:npt, npt] = 1.0
    a[:npt, npt + 1:] = xpt_scale.T
    a[npt, :npt] = 1.0
    a[npt + 1:, :npt] = xpt_scale

    right_scaling = np.empty(npt + n + 1)
    right_scaling[:npt] = 1.0 / scale**2.0
    right_scaling[npt] = scale**2.0
    right_scaling[npt + 1:] = scale

    eig_values, eig_vectors = eigh(a, check_finite=False)
    
    large_eig_values = np.abs(eig_values) > EPS
    V = eig_vectors[:, large_eig_values]
    inv_L = 1.0 / eig_values[large_eig_values]
    
    # PRECOMPUTE THE INVERSE MATRIX: H = diag(R) @ V @ diag(inv_L) @ V.T @ diag(R)
    # This reduces 3 matrix multiplications down to 1 for all future cache hits.
    M = (V * inv_L) @ V.T
    H = M * right_scaling[:, np.newaxis] * right_scaling[np.newaxis, :]
    ill_cond = not np.all(large_eig_values)
    
    interpolation._lhs_cache = {
        "counter": interpolation._update_counter,
        "H": H,
        "ill_conditioned": ill_cond,
    }

    return H @ rhs, ill_cond


class Interpolation:
    def __init__(self, x0: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray, rhobeg: float, npt: int):
        self.n = len(x0)
        self.npt = npt
        self._lhs_cache = None
        self._update_counter = 0  # Used for O(1) cache invalidation
        
        self.x_base = np.copy(x0)
        
        very_close_xl = self.x_base <= lower_bounds + 0.5 * rhobeg
        self.x_base[very_close_xl] = lower_bounds[very_close_xl]
        
        close_xl = (lower_bounds + 0.5 * rhobeg < self.x_base) & (self.x_base <= lower_bounds + rhobeg)
        self.x_base[close_xl] = np.minimum(lower_bounds[close_xl] + rhobeg, upper_bounds[close_xl])
        
        very_close_xu = self.x_base >= upper_bounds - 0.5 * rhobeg
        self.x_base[very_close_xu] = upper_bounds[very_close_xu]
        
        close_xu = (self.x_base < upper_bounds - 0.5 * rhobeg) & (upper_bounds - rhobeg <= self.x_base)
        self.x_base[close_xu] = np.maximum(upper_bounds[close_xu] - rhobeg, lower_bounds[close_xu])

        self.xpt = np.zeros((self.n, self.npt))
        for k in range(1, self.npt):
            if k <= self.n:
                self.xpt[k - 1, k] = -rhobeg if very_close_xu[k - 1] else rhobeg
            elif k <= 2 * self.n:
                if very_close_xl[k - self.n - 1]:
                    self.xpt[k - self.n - 1, k] = 2.0 * rhobeg
                elif very_close_xu[k - self.n - 1]:
                    self.xpt[k - self.n - 1, k] = -2.0 * rhobeg
                else:
                    self.xpt[k - self.n - 1, k] = -rhobeg
            else:
                spread = (k - self.n - 1) // self.n
                k1 = k - (1 + spread) * self.n - 1
                k2 = (k1 + spread) % self.n
                self.xpt[k1, k] = self.xpt[k1, k1 + 1]
                self.xpt[k2, k] = self.xpt[k2, k2 + 1]

    def point(self, k: int) -> np.ndarray:
        return self.x_base + self.xpt[:, k]


class QuadraticModel:
    """Single Quadratic model (used for the objective function f)."""
    def __init__(self, interpolation: Interpolation, values: np.ndarray):
        self.n = interpolation.n
        self.npt = interpolation.npt
        self.const, self.grad_vec, self.i_hess, _ = self._get_model(interpolation, values)
        self.e_hess = np.zeros((self.n, self.n))

    def __call__(self, x: np.ndarray, interpolation: Interpolation) -> float:
        x_diff = x - interpolation.x_base
        return (self.const + self.grad_vec @ x_diff + 
                0.5 * (self.i_hess @ (interpolation.xpt.T @ x_diff) ** 2.0 + x_diff @ self.e_hess @ x_diff))

    def grad(self, x: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        x_diff = x - interpolation.x_base
        return self.grad_vec + self.hess_prod(x_diff, interpolation)

    def hess(self, interpolation: Interpolation) -> np.ndarray:
        return self.e_hess + interpolation.xpt @ (self.i_hess[:, np.newaxis] * interpolation.xpt.T)

    def hess_prod(self, v: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        return self.e_hess @ v + interpolation.xpt @ (self.i_hess * (interpolation.xpt.T @ v))

    def curv(self, v: np.ndarray, interpolation: Interpolation) -> float:
        return v @ self.e_hess @ v + self.i_hess @ (interpolation.xpt.T @ v) ** 2.0

    def update(self, interpolation: Interpolation, k_new: int, dir_old: np.ndarray, values_diff: np.ndarray) -> bool:
        self.e_hess += self.i_hess[k_new] * np.outer(dir_old, dir_old)
        self.i_hess[k_new] = 0.0
        const, grad, i_hess, ill_conditioned = self._get_model(interpolation, values_diff)
        self.const += const
        self.grad_vec += grad
        self.i_hess += i_hess
        return ill_conditioned

    def shift_x_base(self, interpolation: Interpolation, new_x_base: np.ndarray):
        self.const = self(new_x_base, interpolation)
        self.grad_vec = self.grad(new_x_base, interpolation)
        shift = new_x_base - interpolation.x_base
        update = np.outer(shift, (interpolation.xpt - 0.5 * shift[:, np.newaxis]) @ self.i_hess)
        self.e_hess += update + update.T

    @staticmethod
    def _get_model(interpolation: Interpolation, values: np.ndarray):
        n, npt = interpolation.n, interpolation.npt
        rhs = np.block([[values, np.zeros(n + 1)]]).T
        x, ill_conditioned = solve_interpolation_systems(interpolation, rhs)
        return x[npt, 0], x[npt + 1:, 0], x[:npt, 0], ill_conditioned


class VectorizedQuadraticModel:
    """
    Vectorized Quadratic model (used for constraints g and h).
    Evaluates 'm' models simultaneously using 3D tensors, eliminating Python loops.
    """
    def __init__(self, interpolation: Interpolation, values: np.ndarray):
        self.n = interpolation.n
        self.npt = interpolation.npt
        self.m = values.shape[1]
        
        const, grad_vec, i_hess, _ = self._get_model(interpolation, values)
        self.const = const          # (m,)
        self.grad_vec = grad_vec    # (m, n)
        self.i_hess = i_hess        # (m, npt)
        self.e_hess = np.zeros((self.m, self.n, self.n))

    def __call__(self, x: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        x_diff = x - interpolation.x_base
        term1 = self.i_hess @ ((interpolation.xpt.T @ x_diff) ** 2.0)
        term2 = np.einsum('mij,i,j->m', self.e_hess, x_diff, x_diff)
        return self.const + self.grad_vec @ x_diff + 0.5 * (term1 + term2)

    def grad(self, x: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        x_diff = x - interpolation.x_base
        return self.grad_vec + self.hess_prod(x_diff, interpolation)

    def hess_prod(self, v: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        term1 = np.einsum('mij,j->mi', self.e_hess, v)
        term2 = (self.i_hess * (interpolation.xpt.T @ v)) @ interpolation.xpt.T
        return term1 + term2

    def curv(self, v: np.ndarray, interpolation: Interpolation) -> np.ndarray:
        term1 = np.einsum('mij,i,j->m', self.e_hess, v, v)
        term2 = self.i_hess @ ((interpolation.xpt.T @ v) ** 2.0)
        return term1 + term2

    def update(self, interpolation: Interpolation, k_new: int, dir_old: np.ndarray, values_diff: np.ndarray) -> bool:
        outer_dir = np.outer(dir_old, dir_old)
        self.e_hess += self.i_hess[:, k_new, np.newaxis, np.newaxis] * outer_dir
        self.i_hess[:, k_new] = 0.0

        const, grad, i_hess, ill_conditioned = self._get_model(interpolation, values_diff)
        self.const += const
        self.grad_vec += grad
        self.i_hess += i_hess
        return ill_conditioned

    def shift_x_base(self, interpolation: Interpolation, new_x_base: np.ndarray):
        self.const = self(new_x_base, interpolation)
        self.grad_vec = self.grad(new_x_base, interpolation)
        shift = new_x_base - interpolation.x_base
        
        temp = (interpolation.xpt - 0.5 * shift[:, np.newaxis]) @ self.i_hess.T
        update = np.einsum('i,jm->mij', shift, temp)
        self.e_hess += update + np.einsum('mij->mji', update)

    @staticmethod
    def _get_model(interpolation: Interpolation, values: np.ndarray):
        n, npt = interpolation.n, interpolation.npt
        m = values.shape[1]
        rhs = np.vstack([values, np.zeros((n + 1, m))])
        x, ill_conditioned = solve_interpolation_systems(interpolation, rhs)
        return x[npt, :], x[npt + 1:, :].T, x[:npt, :].T, ill_conditioned


class ModelManager:
    def __init__(self, interpolation: Interpolation, f_vals: np.ndarray, g_vals: np.ndarray, h_vals: np.ndarray):
        self.interpolation = interpolation
        self.n = interpolation.n
        self.npt = interpolation.npt
        
        self.g_count = g_vals.shape[1] if g_vals.size > 0 else 0
        self.h_count = h_vals.shape[1] if h_vals.size > 0 else 0

        self.f_vals = np.copy(f_vals)
        self.g_vals = np.copy(g_vals) if self.g_count > 0 else np.empty((self.npt, 0))
        self.h_vals = np.copy(h_vals) if self.h_count > 0 else np.empty((self.npt, 0))

        self.f_model = QuadraticModel(self.interpolation, self.f_vals)
        if self.g_count > 0:
            self.g_models = VectorizedQuadraticModel(self.interpolation, self.g_vals)
        if self.h_count > 0:
            self.h_models = VectorizedQuadraticModel(self.interpolation, self.h_vals)

    def update_interpolation(self, k_new: int, x_new: np.ndarray, f_val: float, g_val: np.ndarray, h_val: np.ndarray) -> bool:
        f_diff = np.zeros(self.npt)
        f_diff[k_new] = f_val - self.f_model(x_new, self.interpolation)
        
        if self.g_count > 0:
            g_diff = np.zeros((self.npt, self.g_count))
            g_diff[k_new, :] = g_val - self.g_models(x_new, self.interpolation)
            
        if self.h_count > 0:
            h_diff = np.zeros((self.npt, self.h_count))
            h_diff[k_new, :] = h_val - self.h_models(x_new, self.interpolation)

        self.f_vals[k_new] = f_val
        if self.g_count > 0: self.g_vals[k_new, :] = g_val
        if self.h_count > 0: self.h_vals[k_new, :] = h_val

        dir_old = np.copy(self.interpolation.xpt[:, k_new])
        self.interpolation.xpt[:, k_new] = x_new - self.interpolation.x_base
        self.interpolation._update_counter += 1  # INVALIDATE CACHE

        ill_cond = self.f_model.update(self.interpolation, k_new, dir_old, f_diff)
        if self.g_count > 0:
            ill_cond = ill_cond or self.g_models.update(self.interpolation, k_new, dir_old, g_diff)
        if self.h_count > 0:
            ill_cond = ill_cond or self.h_models.update(self.interpolation, k_new, dir_old, h_diff)
            
        return ill_cond

    def shift_x_base(self, new_x_base: np.ndarray):
        self.f_model.shift_x_base(self.interpolation, new_x_base)
        if self.g_count > 0: self.g_models.shift_x_base(self.interpolation, new_x_base)
        if self.h_count > 0: self.h_models.shift_x_base(self.interpolation, new_x_base)

        shift = new_x_base - self.interpolation.x_base
        self.interpolation.x_base += shift
        self.interpolation.xpt -= shift[:, np.newaxis]
        self.interpolation._update_counter += 1  # INVALIDATE CACHE

    def determinants(self, x_new: np.ndarray, k_new: int = None):
        shift = x_new - self.interpolation.x_base
        new_col = np.empty((self.npt + self.n + 1, 1))
        new_col[:self.npt, 0] = 0.5 * (self.interpolation.xpt.T @ shift) ** 2.0
        new_col[self.npt, 0] = 1.0
        new_col[self.npt + 1:, 0] = shift
        
        inv_new_col = solve_interpolation_systems(self.interpolation, new_col)[0]
        beta = 0.5 * (shift @ shift) ** 2.0 - new_col[:, 0] @ inv_new_col[:, 0]

        if k_new is None:
            coord_vec = np.eye(self.npt + self.n + 1, self.npt)
            alpha = np.diag(solve_interpolation_systems(self.interpolation, coord_vec)[0])
            tau = inv_new_col[:self.npt, 0]
        else:
            coord_vec = np.eye(self.npt + self.n + 1, 1, -k_new)
            alpha = solve_interpolation_systems(self.interpolation, coord_vec)[0][k_new, 0]
            tau = inv_new_col[k_new, 0]
            
        return alpha * beta + tau**2.0

    # --- Vectorized Model Evaluations ---
    def f(self, x: np.ndarray) -> float: return self.f_model(x, self.interpolation)
    def f_grad(self, x: np.ndarray) -> np.ndarray: return self.f_model.grad(x, self.interpolation)
    def f_hess_prod(self, v: np.ndarray) -> np.ndarray: return self.f_model.hess_prod(v, self.interpolation)

    def g(self, x: np.ndarray) -> np.ndarray:
        return self.g_models(x, self.interpolation) if self.g_count > 0 else np.array([])
    def g_grad(self, x: np.ndarray) -> np.ndarray:
        return self.g_models.grad(x, self.interpolation) if self.g_count > 0 else np.empty((0, self.n))
    def g_hess_prod(self, v: np.ndarray) -> np.ndarray:
        return self.g_models.hess_prod(v, self.interpolation) if self.g_count > 0 else np.empty((0, self.n))

    def h(self, x: np.ndarray) -> np.ndarray:
        return self.h_models(x, self.interpolation) if self.h_count > 0 else np.array([])
    def h_grad(self, x: np.ndarray) -> np.ndarray:
        return self.h_models.grad(x, self.interpolation) if self.h_count > 0 else np.empty((0, self.n))
    def h_hess_prod(self, v: np.ndarray) -> np.ndarray:
        return self.h_models.hess_prod(v, self.interpolation) if self.h_count > 0 else np.empty((0, self.n))
