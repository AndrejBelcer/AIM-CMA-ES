import numpy as np
import math
from typing import Optional, List, Any
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
from scipy.stats.qmc import LatinHypercube
from itertools import combinations

from src.implementations.optimizer_implementations.aim_cma_es.enums import InitializerPhase
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import InitializationParameters
from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.state_tracker import StateTracker

class Initializer(ABC):
    """
    Diverse Initialization Procedure:
    1. Generates an initial LHS batch to explore the landscape.
    2. Generates N random weight vectors.
    3. Uses the Hungarian algorithm to optimally pair each LHS point to a weight vector.
    4. Runs a local optimizer (e.g., COBYQA) for each pair to find local Pareto basins.
    5. Re-evaluates the Hungarian matching dynamically after each run.
    6. Concludes with a final LHS batch to fill remaining gaps.
    """
    def __init__(self, 
                 x_count: int, 
                 f_count: int,
                 g_count: int,
                 initialization_parameters: InitializationParameters) -> None:
        
        self.x_count = x_count
        self.f_count = f_count
        self.g_count = g_count
        
        self.use_L2_weight_normalization = initialization_parameters.use_L2_weight_normalization
        self.seed = initialization_parameters.seed

        # Budgets
        self.initial_lhs_size = math.ceil(initialization_parameters.initial_lhs_size_factor * math.sqrt(self.x_count) * math.sqrt(self.f_count))
        self.final_lhs_size = math.ceil(initialization_parameters.final_lhs_size_factor * math.sqrt(self.x_count))
        self.optimizer_evaluation_budget = math.ceil((2 * self.x_count + 1) + (initialization_parameters.additional_optimizer_evaluations_per_variable * self.x_count))
        
        # State Machine
        self.active_optimizer: Any = None
        self.finished: bool = False
        self.phase: InitializerPhase = InitializerPhase.LHS_SEEDING
        
        # Data Storage
        self.initial_lhs_X = np.empty((0, self.x_count))
        self.initial_lhs_F = np.empty((0, self.f_count))
        self.initial_lhs_G = np.empty((0, self.g_count))
        
        self.remaining_lhs_indices: List[int] = []
        self.remaining_weights: List[np.ndarray] = []
        
        # Active Optimizer Context
        self.active_weight: Optional[np.ndarray] = None
        self.active_f_L: Optional[np.ndarray] = None
        self.active_f_D: Optional[np.ndarray] = None
        self.optimizer_evaluations_used = 0

    @property
    def is_optimizer_active(self) -> bool:
        return self.active_optimizer is not None and self._is_optimizer_running()

    @property
    def has_remaining_points(self) -> bool:
        return len(self.remaining_lhs_indices) > 0

    def transition_to_final_lhs(self):
        self.phase = InitializerPhase.LHS_FINAL

    def _generate_lhs(self, n_points: int) -> np.ndarray:
        sampler = LatinHypercube(d=self.x_count, scramble=True, seed=self.seed)
        self.seed += 1
        return sampler.random(n=n_points)

    def generate_weights(self, n_weights_requested: int, n_obj: int, use_L2_weight_normalization: bool = False, add_noise: bool = True, seed: int = 42) -> np.ndarray:
        """
        Generates a uniformly distributed set of weight vectors using the Das-Dennis approach.
        (Simplex Lattice) method, adjusted for an exact requested budget.
        """
        if n_obj == 1:
            return np.array([[1.0]])

        # 1. Find the required number of partitions (p) to get AT LEAST n_weights_requested
        p = 1
        while math.comb(n_obj + p - 1, p) < n_weights_requested:
            p += 1

        # 2. Generate the exact Das-Dennis grid using "Stars and Bars" combinatorics
        n_positions = p + n_obj - 1
        weights = []
        
        # We place (n_obj - 1) dividers among (p + n_obj - 1) positions
        for dividers in combinations(range(n_positions), n_obj - 1):
            extended_dividers = [-1] + list(dividers) + [n_positions]
            weight = []
            for i in range(n_obj):
                # The value is the number of "stars" between the dividers, divided by p
                val = extended_dividers[i+1] - extended_dividers[i] - 1
                weight.append(val / p)
            weights.append(weight)
            
        weights = np.array(weights)

        rng = np.random.default_rng(seed)

        # 3. Randomly drop excess weights to match the exact requested budget
        if len(weights) > n_weights_requested:
            indices = rng.choice(len(weights), size=n_weights_requested, replace=False)
            weights = weights[indices]

        # 4. Add slight noise (Jitter)
        if add_noise:
            # Maximum noise is 10% of the grid step size to maintain uniformity
            max_noise = (1.0 / p) * 0.1 
            # Add uniform noise
            noise = rng.uniform(0, max_noise, size=weights.shape)
            weights += noise

        # 5. Normalize the weights
        if use_L2_weight_normalization:
            # Normalize to L2 sphere (sum of squares = 1)
            norms = np.linalg.norm(weights, axis=1, keepdims=True)
        else:
            # Normalize to L1 simplex (sum = 1)
            norms = np.sum(weights, axis=1, keepdims=True)
            
        norms[norms == 0] = 1e-9 # Prevent division by zero
        weights = weights / norms

        return weights


    def _generate_and_shuffle_weight_pool(self):
        self.seed += 1
        rng = np.random.default_rng(self.seed)
        
        weights = self.generate_weights(n_weights_requested=self.initial_lhs_size, n_obj=self.f_count, use_L2_weight_normalization=self.use_L2_weight_normalization, add_noise=True, seed=self.seed)

        rng.shuffle(weights)
        self.remaining_weights = list(weights)

    def ask(self) -> np.ndarray:
        if self.finished:
            return np.empty((0, self.x_count))
            
        if self.phase == InitializerPhase.LHS_SEEDING:
            return self._generate_lhs(self.initial_lhs_size)    
        
        elif self.phase == InitializerPhase.OPTIMIZER:
            if self.is_optimizer_active:
                x_ask = self._ask_optimizer()
                if len(x_ask) == 0:
                    self.active_optimizer = None
                return x_ask
            return np.empty((0, self.x_count))
            
        elif self.phase == InitializerPhase.LHS_FINAL:
            self.finished = True
            return self._generate_lhs(self.final_lhs_size)

        return np.empty((0, self.x_count))

    def init_optimizer(self, archive: InternalArchive, state: StateTracker, g_divisor: np.ndarray):
        # 1. Get current normalization bounds
        f_L, _, f_D = state.GetNormalizationBounds(archive)
        
        # 2. Build Cost Matrix for remaining points
        n_rem = len(self.remaining_lhs_indices)
        C = np.zeros((n_rem, n_rem))
        
        F_rem = self.initial_lhs_F[self.remaining_lhs_indices]
        G_rem = self.initial_lhs_G[self.remaining_lhs_indices]
        
        F_norm = (F_rem - f_L) / f_D
        F_norm = np.where(np.isinf(F_norm) | np.isnan(F_norm), 1e6, F_norm)
        
        for i, w in enumerate(self.remaining_weights):
            s = np.sum(w * F_norm, axis=1)
            if self.g_count > 0:
                penalty = 10.0 * np.sum(np.maximum(G_rem / g_divisor, 0.0), axis=1)
                s += penalty
            C[:, i] = s
            
        # 3. Hungarian Match
        row_ind, col_ind = linear_sum_assignment(C)
        
        # 4. Pick the first match
        best_row = row_ind[0]
        best_col = col_ind[0]
        
        chosen_lhs_idx = self.remaining_lhs_indices.pop(best_row)
        chosen_weight = self.remaining_weights.pop(best_col)
        
        x0 = self.initial_lhs_X[chosen_lhs_idx]
        
        # 5. Initialize the specific optimizer (Implemented by subclass)
        self.active_optimizer = self._create_optimizer(x0)
        
        self.active_weight = chosen_weight
        self.active_f_L = f_L
        self.active_f_D = f_D
        self.optimizer_evaluations_used = 0

    def tell(self, x_norm: np.ndarray, f: np.ndarray, g: np.ndarray):
        if len(x_norm) == 0:
            return
        
        if self.phase == InitializerPhase.LHS_SEEDING:
            self.initial_lhs_X = x_norm.copy()
            self.initial_lhs_F = f.copy()
            self.initial_lhs_G = g.copy()
            
            self._generate_and_shuffle_weight_pool()
            self.remaining_lhs_indices = list(range(self.initial_lhs_size))
            self.phase = InitializerPhase.OPTIMIZER
            
        elif self.phase == InitializerPhase.OPTIMIZER and self.is_optimizer_active:
            # Compute scalarized objective for the local optimizer
            f_norm = (f - self.active_f_L) / self.active_f_D
            s = np.sum(self.active_weight * f_norm, axis=1)
            
            self._tell_optimizer(x_norm, s, g)
            self.optimizer_evaluations_used += len(x_norm)
            
            if not self._is_optimizer_running() or self.optimizer_evaluations_used >= self.optimizer_evaluation_budget:
                self.active_optimizer = None

    # ==========================================================================
    # Abstract Methods (To be implemented by specific optimizers)
    # ==========================================================================

    @abstractmethod
    def _create_optimizer(self, x0: np.ndarray) -> Any:
        """Instantiates and returns the local optimizer object."""
        pass

    @abstractmethod
    def _is_optimizer_running(self) -> bool:
        """Returns True if the local optimizer is still active/hasn't converged."""
        pass

    @abstractmethod
    def _ask_optimizer(self) -> np.ndarray:
        """Queries the local optimizer for the next candidate(s)."""
        pass

    @abstractmethod
    def _tell_optimizer(self, x_norm: np.ndarray, s: np.ndarray, g: np.ndarray):
        """Feeds the scalarized results back to the local optimizer."""
        pass
