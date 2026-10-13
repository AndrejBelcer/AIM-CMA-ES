import numpy as np

from src.support_code.cobyqa.settings import Options, Constants, DEFAULT_OPTIONS, DEFAULT_CONSTANTS
from src.support_code.cobyqa.models import Models, Interpolation, Quadratic
from src.support_code.cobyqa.framework import TrustRegion

# ==========================================
# 1. Mock Classes to Bypass Original Evals
# ==========================================

class MockLinear:
    """Provides empty linear constraints to keep TrustRegion math happy."""
    def __init__(self, n):
        self.n = n
        self.pcs = []
        
    @property
    def a_ub(self): return np.empty((0, self.n))
    @property
    def b_ub(self): return np.empty(0)
    @property
    def a_eq(self): return np.empty((0, self.n))
    @property
    def b_eq(self): return np.empty(0)

class MockBounds:
    def __init__(self, xl, xu):
        self.xl = np.array(xl, dtype=float)
        self.xu = np.array(xu, dtype=float)
        self.is_feasible = True
        
    def project(self, x):
        return np.clip(x, self.xl, self.xu)

class MockProblem:
    """Fools the TrustRegion into thinking it has a standard Problem object."""
    def __init__(self, x0, xl, xu, m_cub):
        self.x0 = np.array(x0, dtype=float)
        self.n = len(self.x0)
        self.n_orig = self.n
        self.bounds = MockBounds(xl, xu)
        self.linear = MockLinear(self.n)
        
        # Count finite bounds
        self.m_bounds = np.count_nonzero(self.bounds.xl > -np.inf) + np.count_nonzero(self.bounds.xu < np.inf)
        self.m_linear_ub = 0
        self.m_linear_eq = 0
        self.m_nonlinear_ub = m_cub
        self.m_nonlinear_eq = 0
        self.type = "nonlinearly constrained" if m_cub > 0 else "bound-constrained"
        self.is_feasibility = False
        
    def build_x(self, x):
        return self.bounds.project(x)
        
    def maxcv(self, x, cub_val, ceq_val):
        if len(cub_val) == 0: return 0.0
        return np.max(np.maximum(cub_val, 0.0), initial=0.0)
        
    def violation(self, x, cub_val, ceq_val):
        return np.maximum(cub_val, 0.0) if len(cub_val) > 0 else np.array([])

class AskTellModels(Models):
    """Subclasses Models to bypass the internal evaluation loop in __init__."""
    def __init__(self, pb, options, penalty, initial_f, initial_g):
        self._debug = options.get(Options.DEBUG.value, False)
        self._interpolation = Interpolation(pb, options)
        
        self._fun_val = np.array(initial_f, dtype=float)
        self._cub_val = np.array(initial_g, dtype=float)
        self._ceq_val = np.empty((options[Options.NPT.value], 0))
        
        self._fun = Quadratic(self.interpolation, self._fun_val, self._debug)
        self._cub = np.empty(self.m_nonlinear_ub, dtype=object)
        self._ceq = np.empty(0, dtype=object)
        
        for i in range(self.m_nonlinear_ub):
            self._cub[i] = Quadratic(self.interpolation, self._cub_val[:, i], self._debug)

class AskTellTrustRegion(TrustRegion):
    """Subclasses TrustRegion to inject our AskTellModels."""
    def __init__(self, pb, options, constants, initial_f, initial_g):
        self._penalty = 0.0
        self._pb = pb
        self._models = AskTellModels(self._pb, options, self.penalty, initial_f, initial_g)
        self._constants = constants
        
        self._best_index = 0
        self.set_best_index()
        
        self._lm_linear_ub = np.zeros(0)
        self._lm_linear_eq = np.zeros(0)
        self._lm_nonlinear_ub = np.zeros(self.m_nonlinear_ub)
        self._lm_nonlinear_eq = np.zeros(0)
        self.set_multipliers(self.x_best)
        
        self._resolution = options[Options.RHOBEG.value]
        self._radius = self.resolution

# ==========================================
# 2. The Main Ask/Tell Optimizer Class
# ==========================================

class COBYQA:
    def __init__(self, x0: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray, rhobeg: float = 0.1, rhoend: float = 1e-6, npt: int = None):
        self.x0 = np.array(x0, dtype=float)
        self.n = len(self.x0)
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        
        # Setup Options
        self.options = dict(DEFAULT_OPTIONS)
        self.options[Options.RHOBEG.value] = rhobeg
        self.options[Options.RHOEND.value] = rhoend
        self.options[Options.NPT.value] = npt if npt is not None else 2 * self.n + 1
        self.options[Options.DEBUG.value] = False
        self.constants = dict(DEFAULT_CONSTANTS)
        
        self.optimizer_running = True
        
        # Initialize the generator state machine
        self._gen = self._optimizer_logic()
        self._next_ask = next(self._gen)
        
    def ask(self) -> np.ndarray:
        """Returns points to evaluate. Shape: (Batch_Size, D)"""
        if not self.optimizer_running:
            return np.empty((0, self.n))
        return self._next_ask
        
    def tell(self, x: np.ndarray, s: np.ndarray, g: np.ndarray):
        """Receives evaluated points and advances the state machine."""
        if not self.optimizer_running:
            return
        try:
            # Send the scalarized objective 's' and constraints 'g' back into the generator
            self._next_ask = self._gen.send((s, g))
        except StopIteration:
            self.optimizer_running = False

    def _optimizer_logic(self):
        """
        The core generator. It yields `x` whenever it needs an evaluation, 
        and receives `(s, g)` when `.send()` is called by `tell()`.
        """
        npt = self.options[Options.NPT.value]
        
        # --- PHASE 1: INITIALIZATION ---
        pb_temp = MockProblem(self.x0, self.lower_bounds, self.upper_bounds, 0)
        interp = Interpolation(pb_temp, self.options)
        
        # 1a. Ask for x0
        s0, g0 = yield interp.point(0).reshape(1, -1)
        s0 = np.atleast_1d(s0)[0]
        g0 = np.atleast_2d(g0)[0]
        m_cub = len(g0)
        
        fun_val = np.full(npt, np.nan)
        cub_val = np.full((npt, m_cub), np.nan)
        fun_val[0] = s0
        cub_val[0, :] = g0
        
        # 1b. Ask for the remaining NPT-1 points as a single batch!
        x_rem = np.array([interp.point(k) for k in range(1, npt)])
        s_rem, g_rem = yield x_rem
        s_rem = np.atleast_1d(s_rem)
        g_rem = np.atleast_2d(g_rem)
        
        for k in range(1, npt):
            fun_val[k] = s_rem[k-1]
            cub_val[k, :] = g_rem[k-1]
            
        # --- PHASE 2: BUILD FRAMEWORK ---
        pb = MockProblem(self.x0, self.lower_bounds, self.upper_bounds, m_cub)
        framework = AskTellTrustRegion(pb, self.options, self.constants, fun_val, cub_val)
        
        # --- PHASE 3: MAIN OPTIMIZATION LOOP ---
        n_short_steps = 0
        n_very_short_steps = 0
        n_alt_models = 0
        
        while True:
            # Shift base if needed
            if np.linalg.norm(framework.x_best - framework.models.interpolation.x_base) >= self.constants[Constants.LARGE_SHIFT_FACTOR.value] * framework.radius:
                framework.shift_x_base(self.options)
                
            radius_save = framework.radius
            normal_step, tangential_step = framework.get_trust_region_step(self.options)
            step = normal_step + tangential_step
            s_norm = np.linalg.norm(step)
            
            enhance_resolution = False
            improve_geometry = False
            k_new = None
            
            # Check for short step (No evaluation needed)
            if s_norm <= self.constants[Constants.SHORT_STEP_THRESHOLD.value] * framework.resolution:
                framework.radius *= self.constants[Constants.DECREASE_RESOLUTION_FACTOR.value]
                if radius_save > framework.resolution:
                    n_short_steps = 0
                    n_very_short_steps = 0
                else:
                    n_short_steps += 1
                    n_very_short_steps += 1
                    if s_norm > 0.1 * framework.resolution:
                        n_very_short_steps = 0
                
                enhance_resolution = n_short_steps >= 5 or n_very_short_steps >= 3
                if not enhance_resolution:
                    k_new, dist_new = framework.get_index_to_remove()
                    improve_geometry = dist_new > max(framework.radius, self.constants[Constants.RESOLUTION_FACTOR.value] * framework.resolution)
            
            else:
                same_best = framework.increase_penalty(step)
                if same_best:
                    # --- EVALUATE TRIAL STEP ---
                    x_eval = framework.x_best + step
                    s_val, g_val = yield x_eval.reshape(1, -1)
                    s_val = np.atleast_1d(s_val)[0]
                    g_val = np.atleast_2d(g_val)[0]
                    
                    # --- CHECK SECOND ORDER CORRECTION (SOC) ---
                    merit_old = framework.merit(framework.x_best, framework.fun_best, framework.cub_best, framework.ceq_best)
                    merit_new = framework.merit(x_eval, s_val, g_val, np.array([]))
                    
                    if m_cub > 0 and merit_new > merit_old and np.linalg.norm(normal_step) > self.constants[Constants.BYRD_OMOJOKUN_FACTOR.value]**2 * framework.radius:
                        soc_step = framework.get_second_order_correction_step(step, self.options)
                        if np.linalg.norm(soc_step) > 0:
                            step += soc_step
                            x_eval = framework.x_best + step
                            
                            # --- EVALUATE SOC STEP ---
                            s_val, g_val = yield x_eval.reshape(1, -1)
                            s_val = np.atleast_1d(s_val)[0]
                            g_val = np.atleast_2d(g_val)[0]
                            
                    ratio = framework.get_reduction_ratio(step, s_val, g_val, np.array([]))
                    k_new = framework.get_index_to_remove(x_eval)[0]
                    ill_cond = framework.models.update_interpolation(k_new, x_eval, s_val, g_val, np.array([]))
                    framework.set_best_index()
                    framework.update_radius(step, ratio)
                    
                    # Alt models check
                    if framework.radius <= framework.resolution:
                        if ratio >= self.constants[Constants.VERY_LOW_RATIO.value]:
                            n_alt_models = 0
                        else:
                            n_alt_models += 1
                            grad = framework.models.fun_grad(framework.x_best)
                            grad_alt = framework.models.fun_alt_grad(framework.x_best)
                            if np.linalg.norm(grad) < self.constants[Constants.LARGE_GRADIENT_FACTOR.value] * np.linalg.norm(grad_alt):
                                n_alt_models = 0
                            if n_alt_models >= 3:
                                framework.models.reset_models()
                                n_alt_models = 0
                                
                    framework.set_multipliers(framework.x_best + step)
                    
                    k_new, dist_new = framework.get_index_to_remove()
                    improve_geometry = ill_cond or (ratio <= self.constants[Constants.LOW_RATIO.value] and dist_new > max(framework.radius, self.constants[Constants.RESOLUTION_FACTOR.value] * framework.resolution))
                    enhance_resolution = (radius_save <= framework.resolution and ratio <= self.constants[Constants.LOW_RATIO.value] and not improve_geometry)
                    
                else:
                    enhance_resolution = False
                    improve_geometry = False
                    
            # --- APPLY RESOLUTION / GEOMETRY FIXES ---
            if enhance_resolution:
                if framework.resolution <= self.options[Options.RHOEND.value]:
                    break # Trust region reached minimum size. Optimization successful!
                framework.enhance_resolution(self.options)
                framework.decrease_penalty()
                
            if improve_geometry:
                step = framework.get_geometry_step(k_new, self.options)
                x_eval = framework.x_best + step
                
                # --- EVALUATE GEOMETRY STEP ---
                s_val, g_val = yield x_eval.reshape(1, -1)
                s_val = np.atleast_1d(s_val)[0]
                g_val = np.atleast_2d(g_val)[0]
                
                framework.models.update_interpolation(k_new, x_eval, s_val, g_val, np.array([]))
                framework.set_best_index()
