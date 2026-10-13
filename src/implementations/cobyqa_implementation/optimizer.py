import numpy as np
from typing import Tuple, Optional

from src.implementations.cobyqa_implementation.settings import COBYQAParameters
from src.implementations.cobyqa_implementation.models import Interpolation, ModelManager
from src.implementations.cobyqa_implementation.trust_region import TrustRegionState

class COBYQAOptimizer:
    """
    Ask-and-Tell optimizer for COBYQA.
    """
    def __init__(self, 
                 x0: np.ndarray, 
                 lower_bounds: np.ndarray, 
                 upper_bounds: np.ndarray, 
                 g_count: int = 0, 
                 h_count: int = 0, 
                 params: COBYQAParameters = COBYQAParameters()):
        
        self.x0 = np.asarray(x0, dtype=float)
        self.n = len(self.x0)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        
        self.g_count = g_count
        self.h_count = h_count
        self.params = params
        
        if self.params.npt is None:
            self.params.npt = 2 * self.n + 1
            
        self.running = True
        self.success = False
        self.best_x = np.copy(self.x0)
        self.best_f = np.inf
        
        # Initialize the generator state machine
        self._gen = self._optimizer_logic()
        self._next_ask = next(self._gen)

    def ask(self) -> np.ndarray:
        """
        Returns the next point(s) to evaluate.
        Shape is (NPT, D) for the first call, and (1, D) for all subsequent calls.
        """
        if not self.running:
            return np.empty((0, self.n))
        return self._next_ask

    def tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray = None, h: np.ndarray = None):
        """
        Receives evaluated points and advances the optimizer state.
        """
        if not self.running:
            return

        # Format inputs
        f = np.atleast_1d(f)
        g = np.atleast_2d(g) if g is not None and self.g_count > 0 else np.empty((len(f), 0))
        h = np.atleast_2d(h) if h is not None and self.h_count > 0 else np.empty((len(f), 0))

        try:
            # Send the evaluations back into the generator
            self._next_ask = self._gen.send((f, g, h))
        except StopIteration:
            self.running = False

    def stop(self) -> bool:
        """Returns True if the optimization has finished."""
        return not self.running

    def _optimizer_logic(self):
        """
        The core generator. It yields `x` whenever it needs an evaluation, 
        and receives `(f, g, h)` when `.send()` is called by `tell()`.
        """
        # ==========================================
        # PHASE 1: INITIALIZATION
        # ==========================================
        interp = Interpolation(self.x0, self.lower_bounds, self.upper_bounds, self.params.rhobeg, self.params.npt)
        
        # Ask for all NPT points at once (Batch Evaluation)
        x_init = np.array([interp.point(k) for k in range(self.params.npt)])
        f_init, g_init, h_init = yield x_init
        
        # Track the best point seen so far (for user convenience)
        best_idx = np.argmin(f_init)
        self.best_x = x_init[best_idx]
        self.best_f = f_init[best_idx]

        # Build the models and trust region state
        models = ModelManager(interp, f_init, g_init, h_init)
        tr = TrustRegionState(models, self.lower_bounds, self.upper_bounds, self.params)

        # ==========================================
        # PHASE 2: MAIN OPTIMIZATION LOOP
        # ==========================================
        n_short_steps = 0
        n_very_short_steps = 0
        n_alt_models = 0
        
        while True:
            # 1. Shift base if we have moved too far
            if np.linalg.norm(tr.x_best - models.interpolation.x_base) >= self.params.large_shift_factor * tr.radius:
                models.shift_x_base(np.copy(tr.x_best))

            # 2. Calculate Trust Region Step
            radius_save = tr.radius
            normal_step, tangential_step = tr.get_trust_region_step()
            step = normal_step + tangential_step
            s_norm = np.linalg.norm(step)

            enhance_resolution = False
            improve_geometry = False
            k_new = None

            # 3. Check for Short Step (Skip evaluation if too short)
            if s_norm <= self.params.short_step_threshold * tr.resolution:
                tr.radius *= self.params.decrease_resolution_factor
                if radius_save > tr.resolution:
                    n_short_steps = 0
                    n_very_short_steps = 0
                else:
                    n_short_steps += 1
                    n_very_short_steps += 1
                    if s_norm > 0.1 * tr.resolution:
                        n_very_short_steps = 0
                
                enhance_resolution = (n_short_steps >= 5) or (n_very_short_steps >= 3)
                if not enhance_resolution:
                    k_new, dist_new = tr.get_index_to_remove()
                    improve_geometry = dist_new > max(tr.radius, self.params.resolution_factor * tr.resolution)
            
            # 4. Normal Step Evaluation
            else:
                same_best = tr.increase_penalty(step)
                if same_best:
                    x_eval = np.clip(tr.x_best + step, self.lower_bounds, self.upper_bounds)
                    
                    # --- YIELD 1: EVALUATE TRIAL STEP ---
                    f_val, g_val, h_val = yield x_eval.reshape(1, -1)
                    f_val, g_val, h_val = f_val[0], g_val[0], h_val[0]
                    
                    if f_val < self.best_f:
                        self.best_f = f_val
                        self.best_x = x_eval

                    # --- SECOND ORDER CORRECTION (SOC) ---
                    merit_old = tr.merit(tr.x_best, tr.f_best, tr.g_best, tr.h_best)
                    merit_new = tr.merit(x_eval, f_val, g_val, h_val)
                    
                    if tr.has_constraints and merit_new > merit_old and np.linalg.norm(normal_step) > (self.params.byrd_omojokun_factor**2) * tr.radius:
                        soc_step = tr.get_second_order_correction_step(step)
                        if np.linalg.norm(soc_step) > 0:
                            step += soc_step
                            x_eval = np.clip(tr.x_best + step, self.lower_bounds, self.upper_bounds)
                            
                            # --- YIELD 2: EVALUATE SOC STEP ---
                            f_val, g_val, h_val = yield x_eval.reshape(1, -1)
                            f_val, g_val, h_val = f_val[0], g_val[0], h_val[0]
                            
                            if f_val < self.best_f:
                                self.best_f = f_val
                                self.best_x = x_eval

                    # --- UPDATE MODELS AND RADIUS ---
                    ratio = tr.get_reduction_ratio(step, f_val, g_val, h_val)
                    k_new = tr.get_index_to_remove(x_eval)[0]
                    ill_cond = models.update_interpolation(k_new, x_eval, f_val, g_val, h_val)
                    
                    tr.set_best_index()
                    tr.update_radius(step, ratio)

                    # --- ALT MODELS CHECK ---
                    if tr.radius <= tr.resolution:
                        if ratio >= self.params.very_low_ratio:
                            n_alt_models = 0
                        else:
                            n_alt_models += 1
                            grad = models.f_grad(tr.x_best)
                            # Quick alt gradient check using current values
                            from .models import QuadraticModel
                            alt_model = QuadraticModel(models.interpolation, models.f_vals)
                            grad_alt = alt_model.grad(tr.x_best, models.interpolation)
                            
                            if np.linalg.norm(grad) < self.params.large_gradient_factor * np.linalg.norm(grad_alt):
                                n_alt_models = 0
                            if n_alt_models >= 3:
                                # Reset models to alternative
                                models.f_model = alt_model
                                for i in range(models.g_count):
                                    models.g_models[i] = QuadraticModel(models.interpolation, models.g_vals[:, i])
                                for i in range(models.h_count):
                                    models.h_models[i] = QuadraticModel(models.interpolation, models.h_vals[:, i])
                                n_alt_models = 0

                    tr.set_multipliers(tr.x_best + step)
                    
                    k_new, dist_new = tr.get_index_to_remove()
                    improve_geometry = ill_cond or (ratio <= self.params.low_ratio and dist_new > max(tr.radius, self.params.resolution_factor * tr.resolution))
                    enhance_resolution = (radius_save <= tr.resolution and ratio <= self.params.low_ratio and not improve_geometry)
                    
                else:
                    enhance_resolution = False
                    improve_geometry = False

            # 5. Apply Resolution / Geometry Fixes
            if enhance_resolution:
                if tr.resolution <= self.params.rhoend:
                    self.success = True
                    break # Optimization finished successfully
                
                tr.enhance_resolution()
                tr.decrease_penalty()
                
            if improve_geometry:
                step = tr.get_geometry_step(k_new)
                x_eval = np.clip(tr.x_best + step, self.lower_bounds, self.upper_bounds)
                
                # --- YIELD 3: EVALUATE GEOMETRY STEP ---
                f_val, g_val, h_val = yield x_eval.reshape(1, -1)
                f_val, g_val, h_val = f_val[0], g_val[0], h_val[0]
                
                if f_val < self.best_f:
                    self.best_f = f_val
                    self.best_x = x_eval
                
                models.update_interpolation(k_new, x_eval, f_val, g_val, h_val)
                tr.set_best_index()
