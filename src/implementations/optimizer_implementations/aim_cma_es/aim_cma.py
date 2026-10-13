import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any

from src.core.optimizer import Optimizer
from src.core.event_log import Event
from src.implementations.optimizer_implementations.aim_cma_es.state_tracker import StateTracker
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.scalarizer_base import Scalarizer
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.strategies import FeasibilityScalarizer, CornerScalarizer, GapScalarizer, UHVIScalarizer, WSumScalarizer, CSumScalarizer


from src.implementations.optimizer_implementations.aim_cma_es.enums import Archive_Type, Initializer_Type, CMA_Type, SolverPhase, ScalarizationMode, InitilizerPhase
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import ArchiveParameters, CMAParameters, RefinerParameters, InitializationParameters

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.kd_tree_archive import KDTreeArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.kd_tree_archive_n import KDTreeArchiveN
from src.implementations.optimizer_implementations.aim_cma_es.archive.ball_tree_archive import BallTreeArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.hnsw_archive import HNSWArchive

from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer import Initializer
from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer_cobyqa import InitializerCOBYQA
from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer_cobyqa_official import InitializerCOBYQA_O

from src.implementations.optimizer_implementations.aim_cma_es.cma_component.cma_manager import CmaManager

from src.implementations.optimizer_implementations.aim_cma_es.refiner.refiner import Refiner




class AimCma(Optimizer):
    def __init__(self,
                 solver_name: str = "AIM-CMA-ES",
                 # --- Sampling Tolerance ---
                 x_tolerance: float = 1e-4,
                 # --- Configurations ---
                 archive_parameters: ArchiveParameters = ArchiveParameters(),
                 initialization_parameters: InitializationParameters = InitializationParameters(),
                 cma_parameters: CMAParameters = CMAParameters(),
                 refiner_parameters: RefinerParameters = RefinerParameters(),
                ) -> None:
        
        super().__init__(solver_name)
        
        self.x_tolerance = x_tolerance
        
        self.archive_parameters = archive_parameters
        self.initialization_parameters = initialization_parameters
        self.cma_parameters = cma_parameters
        self.refiner_parameters = refiner_parameters
        
        self.refiner_activation_at_count = refiner_parameters.refiner_activation_at_count
        self.refiner_L_loops = refiner_parameters.refiner_L_loops

        # The phase set during ask, so that tell function can know the current phase
        self.current_phase: SolverPhase = SolverPhase.NONE
        # The phase for setting up next phase in some cases
        self.next_phase: SolverPhase = SolverPhase.INITIALIZATION

        # CMA ES exploration parameters
        self.cma_exploration_phase_remaining: int = 0 # Flag for if the exploration should be run
        self.exploration_weights: List[np.ndarray] = []
        self.current_count_exploration_kernels: int = 0 # How many exploration kernels should be run when refiner finishes
        # Chance of random (UHVI) start
        self.uhvi_start_chance: float = 0.0

        # Logic Queues & Counters
        self.corner_queue: List[int] = []
        self.pending_cma_best: Optional[np.ndarray] = None
        
        self.refiner_failure_counter = 0
        self.refiner_runs_counter = 0
        
        self._reconstruction_data: Dict = {}
        
        # Debug options
        self._is_debug_run = False
        self._current_debug_event: Dict[str, Any] = {}
        

    def setup(self, expected_number_of_evaluations: int, n_vars: int, n_obj: int, n_constr: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, region_of_interest: np.ndarray):
        self.x_count = n_vars
        self.f_count = n_obj
        self.g_count = n_constr
        
        self.true_lower_bounds = lower_bounds
        self.true_upper_bounds = upper_bounds
        self.roi_lower_bounds, self.roi_upper_bounds = region_of_interest
        self.roi_range = np.maximum(self.roi_upper_bounds - self.roi_lower_bounds, 1e-9)
        
        true_lower_bounds_normalized: np.ndarray = self._normalize_x(self.true_lower_bounds)
        true_upper_bounds_normalized: np.ndarray = self._normalize_x(self.true_upper_bounds)
        
        self.corner_queue = list(range(self.f_count))

        # Prepare lookups
        archive_map = {
            Archive_Type.KDTREE: KDTreeArchive,
            Archive_Type.KDTREEN: KDTreeArchiveN,
            Archive_Type.BALLTREE: BallTreeArchive,
            Archive_Type.HNSW: HNSWArchive,
        }
        
        initializer_map = {
            Initializer_Type.COBYQA: InitializerCOBYQA,
            Initializer_Type.COBYQAOFF: InitializerCOBYQA_O,
        }
        
        
        # Initialize Components
        try:
            self.internal_archive: InternalArchive = archive_map[self.archive_parameters.archive_type](x_count=n_vars, f_count=n_obj, g_count=n_constr, lower_bounds=self.roi_lower_bounds, upper_bounds=self.roi_upper_bounds, capacity=expected_number_of_evaluations + 100, parameters = self.archive_parameters)
            self.initializer: Initializer = initializer_map[self.initialization_parameters.initializer_type](x_count=n_vars, f_count=n_obj, g_count=n_constr, initialization_parameters=self.initialization_parameters)
        except KeyError as e:
            raise NotImplementedError(f"Option not found in registry: {e}")

        self.state = StateTracker(f_count=n_obj, x_count=n_vars)

        self.cma_manager = CmaManager(x_count=n_vars, true_lower_bounds_normalized=true_lower_bounds_normalized, true_upper_bounds_normalized=true_upper_bounds_normalized, cma_parameters=self.cma_parameters)
        self.refiner = Refiner(x_count=n_vars, f_count=n_obj, g_count=n_constr, refiner_parameters=self.refiner_parameters)

        self.active_x_tolerance = self.x_tolerance * np.sqrt(n_vars)
        self.current_phase = SolverPhase.NONE
        self.next_phase: SolverPhase = SolverPhase.INITIALIZATION
        self.cma_exploration_phase_remaining = 0
        self.current_count_exploration_kernels: int = self.cma_parameters.starting_exploration_kernels
        self.exploration_weights = []
        self.uhvi_start_chance = self.cma_parameters.uhvi_start_chance_initial
        self._is_debug_run = False

    # ==========================================================================
    # ASK PIPELINE
    # ==========================================================================

    def ask(self) -> np.ndarray:
        """
        Unified Ask Pipeline:
        1. Determine Phase & Generate Raw Candidates
        2. Clip to True Bounds
        3. Check Duplicates
        4. Store Reconstruction Data
        5. Return Novel Points
        """
        # 1. Determine Phase & Generate
        candidates = self.DeterminePhaseAndAsk()
        
        if len(candidates) == 0:
            return candidates

        # 2. Global Clip to True Bounds
        candidates = np.clip(candidates, self.true_lower_bounds, self.true_upper_bounds)

        # 3. Duplicate Checking
        is_novel, archive_indices = self.internal_archive.check_duplicates_with_indices(candidates, self.active_x_tolerance)
        
        # 4. Store Reconstruction Data
        self._reconstruction_data = {
            'full_X': candidates,
            'is_novel': is_novel,
            'archive_indices': archive_indices
        }
        
        # 5. Return Novel
        return candidates[is_novel]
    
    def debug_ask(self) -> tuple[np.ndarray, Event]:
        self._is_debug_run = True
        self._current_debug_event = {}
        
        candidates = self.ask()
        
        # Gather Ask-specific debug data
        self._current_debug_event["phase"] = self.current_phase.value
        if 'is_novel' in self._reconstruction_data:
            self._current_debug_event["duplicates_prevented"] = int(np.sum(~self._reconstruction_data['is_novel']))
            
        if self.cma_manager.is_active():
            self._current_debug_event["cma_distribution"] = self.cma_manager.es_kernel.get_distribution_data()
            
        event = Event(self._current_debug_event)
        self._is_debug_run = False
        return candidates, event


    def DeterminePhaseAndAsk(self) -> np.ndarray:
        """Routes to the correct component based on budgets and state."""
        # Priority 0: Pending Incumbent
        if self.pending_cma_best is not None:
            self.current_phase = SolverPhase.CMA
            candidates = self._denormalize_x(self.pending_cma_best.reshape(1, -1))
            self.pending_cma_best = None
            return candidates
        
        # Priority 1: Initialization Phase
        if not self.initializer.finished:
            self.current_phase = SolverPhase.INITIALIZATION
            return self._ask_initializer()

        # Priority 2: Refiner Phase
        if self.next_phase == SolverPhase.REFINER:
            self.current_phase = SolverPhase.REFINER
            return self._ask_refiner()

        # Priority 3: CMA Phase
        self.current_phase = SolverPhase.CMA
        return self._ask_cma()
    
    
    def _ask_initializer(self) -> np.ndarray:
        # Check if we are in the optimizer phase but don't have an active optimizer running
        if self.initializer.phase == InitilizerPhase.OPTIMIZER and not self.initializer.is_optimizer_active:
            
            if self.initializer.has_remaining_points:
                g_divisor = self.compute_g_scales_from_archive()
                self.initializer.init_optimizer(self.internal_archive, self.state, g_divisor)
            else:
                # Out of points, transition to LHS_FINAL
                self.initializer.transition_to_final_lhs()

        candidates_norm = self.initializer.ask()
        return self._denormalize_x(candidates_norm)
        


    def _ask_cma(self) -> np.ndarray:
        if (self.cma_manager.is_active() == False):
            self.state.freeze_state(self.internal_archive)
            scalarizer, points_of_interest = self.CreateScalarizer()
            
            if(self._is_debug_run):
                self._current_debug_event["Kernel Start"] = "[CMA] Starting New Kernel."
                self._current_debug_event["Scalarization"] = scalarizer.get_string_description()

            _, f_U, _ = self.state.GetNormalizationBounds(self.internal_archive)

            # Pass the raw internal archive directly to utilize tree queries during warm starts
            self.cma_manager.InitKernel(self.internal_archive, f_U, scalarizer=scalarizer, points_of_interest=points_of_interest)
            
        return self._denormalize_x(self.cma_manager.ask())
        
    def _ask_refiner(self) -> np.ndarray:
        f_L, _, f_D = self.state.GetNormalizationBounds(self.internal_archive)
        candidates = self.refiner.get_candidates(archive=self.internal_archive, front_indices=self.state.empirical_front_indices, f_L = f_L, f_D = f_D)
        return self._denormalize_x(candidates)


    # ==========================================================================
    # TELL PIPELINE
    # ==========================================================================


    def tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        """
        Unified Tell Pipeline:
        1. Add Novel to Archive
        2. Update StateTracker
        3. Reconstruct Full Batch
        4. Update Active Component (CMA)
        """
        # 1. Add Novel points
        if len(x) > 0:
            start_idx = self.internal_archive.n_size
            self.internal_archive.add(x, f, g)
            end_idx = self.internal_archive.n_size
            
            new_indices = np.arange(start_idx, end_idx)
            
            # 2. Update StateTracker (Buffered during CMA stage to delay NDS overhead)
            self.state.buffer_live_state(new_indices, f, g, self.internal_archive)
            if(self.current_phase == SolverPhase.REFINER):
                self.refiner_runs_counter += 1
                if self.refiner_runs_counter >= self.refiner_L_loops:
                    self.refiner_runs_counter = 0
                    self.next_phase = SolverPhase.CMA
                    self.current_count_exploration_kernels = min(self.current_count_exploration_kernels + self.cma_parameters.step_exploration_kernels, self.cma_parameters.end_exploration_kernels)
                    self.cma_exploration_phase_remaining = self.current_count_exploration_kernels
                    self.uhvi_start_chance = self.cma_parameters.uhvi_start_chance_final
                # Check for corner improvement.
                if(self._is_debug_run):
                    self._current_debug_event["Corner Improvement"] = self.state.DebugCornerImprovement(f_significant_change=self.cma_parameters.f_significant_change)
                list_of_improvements: list[int] = self.state.CheckCornerImprovement(f_significant_change=self.cma_parameters.f_significant_change)
                for i in list_of_improvements:
                        if i not in self.corner_queue:
                            self.corner_queue.append(i)
            
            
            if (not self.cma_manager.is_active()):
                self.state.finalize_live_state(self.internal_archive)

        # 3. Reconstruct Full Batch
        if not ((self.current_phase == SolverPhase.INITIALIZATION) or (self.current_phase == SolverPhase.CMA)):
            return

        full_X = self._reconstruction_data['full_X']
        is_novel = self._reconstruction_data['is_novel']
        archive_indices = self._reconstruction_data['archive_indices']
        
        n_full = len(full_X)
        full_F = np.empty((n_full, self.f_count))
        full_G = np.empty((n_full, self.g_count))
        
        if np.any(is_novel):
            full_F[is_novel] = f
            full_G[is_novel] = g
            
        if np.any(~is_novel):
            dup_indices = archive_indices[~is_novel]
            f_cached, g_cached = self.internal_archive.get_values_by_indices(dup_indices)
            full_F[~is_novel] = f_cached
            full_G[~is_novel] = g_cached
            
        self._reconstruction_data = {}

        # 4. Update Active Component
        if self.current_phase == SolverPhase.INITIALIZATION:
            self._tell_initializer(full_X, full_F, full_G)
        elif self.cma_manager.is_active():
            self._tell_cma(full_X, full_F, full_G)

    def debug_tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray) -> Event:
        self._is_debug_run = True
        self._current_debug_event = {}
        
        self.tell(x, f, g)
        
        event = Event(self._current_debug_event)
        self._is_debug_run = False
        return event

    def _tell_initializer(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        f_clean = f.copy()
        mask_inf = np.isinf(f_clean)
        if np.any(mask_inf):
            for i in range(self.f_count):
                penalty = 2.0 * max(0.1, self.state.f_U[i])
                f_clean[mask_inf[:, i], i] = penalty

        x_normalized = self._normalize_x(x)
        self.initializer.tell(x_normalized, f_clean, g)


    def _tell_cma(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        f_clean = f.copy()
        mask_inf = np.isinf(f_clean)
        if np.any(mask_inf):
            for i in range(self.f_count):
                penalty = 2.0 * max(0.1, self.state.f_U[i])
                f_clean[mask_inf[:, i], i] = penalty

        x_norm = self._normalize_x(x)
        self.cma_manager.tell(x_norm, f_clean, g)
        
        if self.cma_manager.is_stop():
            self.pending_cma_best = self.cma_manager.get_best_point()
            self._handle_kernel_stop()
            self.cma_manager.deactivate()

    # ==========================================================================
    # KERNEL STOP & SCALARIZATION LOGIC
    # ==========================================================================

    def _handle_kernel_stop(self):
        # Process and sort any buffered evaluations before querying front metrics
        self.state.finalize_live_state(self.internal_archive)
        
        if(self._is_debug_run):
            stop_reason = self.cma_manager.es_kernel.get_stop_reason()
            self._current_debug_event["Kernel Stopped"] = f"Kernel Stop Reason: [{stop_reason}]"

        scalarizer = self.cma_manager.active_scalarizer
        if scalarizer is None:
            return
        mode = scalarizer.mode
        
        if mode == ScalarizationMode.FEASIBILITY:
            return

        # --- GAP / UHVI Logic ---
        if mode in [ScalarizationMode.GAP, ScalarizationMode.UHVI]:
            hv_start = scalarizer.indicator.frozen_score
            live_indices = self.state.empirical_front_indices
            raw_live_front, _ = self.internal_archive.get_values_by_indices(live_indices)
            norm_live_front = scalarizer.normalize(raw_live_front)
            hv_end = scalarizer.indicator.calculate_front_score(norm_live_front)

            gain = hv_end - hv_start
            
            if mode == ScalarizationMode.GAP:
                if gain < self.cma_parameters.f_HV_tolerance:                    
                    self.state.add_tabu_point(scalarizer.target_raw)
                    if(self._is_debug_run):
                        self._current_debug_event["Tabu Add"] = f"Gap Fill Failed. Adding target to Tabu."
                        self._current_debug_event["Tabu List"] = self.state.tabu_array.copy()
                    
                    # Refiner stage counter
                    self.refiner_failure_counter += 1
                    if self.refiner_failure_counter >= self.refiner_activation_at_count:
                        self.refiner_failure_counter = 0
                        # Permanent tolerance reduction
                        self.active_x_tolerance = max(1e-6, self.active_x_tolerance * 0.5)
                        self.next_phase = SolverPhase.REFINER
                        if(self._is_debug_run):
                            self._current_debug_event["Refiner Start"] = "Refiner started."

        # --- CORNER RE-QUEUE LOGIC ---
        if(self._is_debug_run):
            self._current_debug_event["Corner Improvement"] = self.state.DebugCornerImprovement(f_significant_change=self.cma_parameters.f_significant_change)
        list_of_improvements: list[int] = self.state.CheckCornerImprovement(f_significant_change=self.cma_parameters.f_significant_change)
        for i in list_of_improvements:
            if not((mode == ScalarizationMode.CORNER and scalarizer.axis == i)):
                if i not in self.corner_queue:
                    self.corner_queue.append(i)

    def compute_g_scales_from_archive(self, percentile: float = 75.0, min_scale: float = 1e-9, min_pos_count: int = 10) -> np.ndarray:
        # Short-circuit for unconstrained problems
        if self.g_count == 0:
            return np.empty(0, dtype=float)

        # 1. Get the mask of infeasible points (O(1) retrieval from our new cache)
        feas_mask = self.internal_archive.get_feasible_mask()
        infeas_mask = ~feas_mask
        
        # 2. If all points are feasible, return default scales of 1.0
        if not np.any(infeas_mask):
            return np.ones(self.g_count, dtype=float)
            
        # 3. Extract G values ONLY for infeasible points
        _, _, G_all = self.internal_archive.get_raw_data()
        G_infeas = G_all[infeas_mask]
        
        scales = np.empty(self.g_count, dtype=float)

        # 4. Compute statistics per constraint
        for j in range(self.g_count):
            col = G_infeas[:, j]
            # A point might be infeasible overall, but feasible for THIS specific constraint
            pos_vals = col[col > 0.0]

            if pos_vals.size == 0:
                scales[j] = 1.0
                continue

            if pos_vals.size < min_pos_count:
                baseline = float(np.max(pos_vals))
            else:
                baseline = float(np.percentile(pos_vals, percentile))

            scales[j] = baseline if baseline > min_scale else min_scale

        return scales


    def GenerateExplorationWeights(self, number_of_weights: int) -> List[np.ndarray]:
        """
        Generates stratified weights on the (M-1)-simplex.
        
        Bi-objective: stratified sampling on [ε, 1-ε] for α, where w = (α, 1-α).
        Each stratum contributes exactly one random sample, guaranteeing coverage
        across the full weight range while remaining stochastic.
        
        M > 2: Dirichlet sampling on the simplex (uniform, not stratified).
        """
        epsilon = 0.03  # Minimum weight component — avoids pure corner scalarizations
        
        if self.f_count == 2:
            # --- Bi-objective: Stratified 1-simplex ---
            weights = []
            for i in range(number_of_weights):
                low = epsilon + (1.0 - 2.0 * epsilon) * i / number_of_weights
                high = epsilon + (1.0 - 2.0 * epsilon) * (i + 1) / number_of_weights
                alpha = np.random.uniform(low, high)
                weights.append(np.array([alpha, 1.0 - alpha]))
            return weights
        else:
            # --- M > 2: Dirichlet on the simplex ---
            weights = []
            for _ in range(number_of_weights):
                w = np.random.dirichlet(np.ones(self.f_count))
                # Enforce minimum weight and re-normalize
                w = np.maximum(w, epsilon)
                w /= np.sum(w)
                weights.append(w)
            return weights
        

    def CreateScalarizer(self) -> Tuple[Scalarizer, List[np.ndarray]]:
        # Compute G-Scales exactly once per kernel initialization
        g_divisor = self.compute_g_scales_from_archive()
        
        # 0. Exploration Mode 
        if (self.cma_exploration_phase_remaining > 0):
            self.cma_exploration_phase_remaining -= 1
            return self.CreateWSUMScalarizer(g_divisor)

        n_front = len(self.state.frozen_front_X)
        # 1. Feasibility Mode
        if n_front == 0:
            return FeasibilityScalarizer(self.state, self.cma_parameters.cv_type, g_divisor), []

        # 2. Corner Mode
        if n_front < self.f_count and len(self.corner_queue) == 0:
            if(self._is_debug_run):
                self._current_debug_event["Small Front"] = "Front too small for gaps. Refilling corner queue."
            self.corner_queue = list(range(self.f_count))
        
        if self.corner_queue:
            return self.CreateCornerScalarizer(g_divisor)

        # 3. Gap Mode
        if np.random.rand() > self.uhvi_start_chance:
            gap_target, gap_points_raw, all_tabu = self.state.GetGapTarget(self.cma_parameters.f_Tabu_tolerance)
            
            if(all_tabu):
                if(self._is_debug_run):
                    self._current_debug_event["Tabu Full"] = "All gaps Tabu. Clearing list and running UHVI."
                self.state.empty_tabu_array()
            
            if (gap_target is not None):
                points_of_interest = [self._normalize_x(p) for p in gap_points_raw]
                if (self._is_debug_run):
                    self._current_debug_event["Gap_Target"] = gap_target.copy()
                return GapScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, gap_target, hv_reference_factor=self.cma_parameters.hv_reference_factor), points_of_interest

        # 4. UHVI Mode            
        return UHVIScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, hv_reference_factor=self.cma_parameters.hv_reference_factor), []


    def CreateWSUMScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:      
        # Replenish weight list if empty
        if len(self.exploration_weights) == 0:
            n_weights = 3 * self.f_count
            self.exploration_weights = self.GenerateExplorationWeights(n_weights)
        
        # 1. Pick a random starting point in normalized [0,1]^D ROI space
        c = np.random.rand(self.x_count)
        
        # 2. Find the 5 nearest archive points (in normalized coordinate space)
        k_neighbors = min(5, self.internal_archive.n_size)
        _, indices = self.internal_archive.query_k_nearest(c.reshape(1, -1), k=k_neighbors, distance_upper_bound=np.sqrt(self.x_count))
        neighbor_indices = indices.ravel()
        neighbor_indices = neighbor_indices[neighbor_indices >= 0]  # Filter missing
        
        # 3. Get raw objectives and normalize using frozen bounds
        f_neighbors = self.internal_archive.get_f_by_indices(neighbor_indices)
        f_L, _, f_D = self.state.GetNormalizationBounds(self.internal_archive)
        f_norm_neighbors = (f_neighbors - f_L) / f_D  # shape (k, M)
        
        # 4. For each remaining weight, compute median weighted sum across neighbors
        best_weight_idx = -1
        best_median_score = np.inf
        
        for i, w in enumerate(self.exploration_weights):
            scores = f_norm_neighbors @ w  # shape (k,)
            median_score = np.median(scores)
            if median_score < best_median_score:
                best_median_score = median_score
                best_weight_idx = i
        
        # 5. Pick the best weight and remove it from the list
        best_weight = self.exploration_weights.pop(best_weight_idx)
        
        if self._is_debug_run:
            self._current_debug_event["Exploration"] = (
                f"WSUM cold start. Weight: {np.array2string(best_weight, formatter={'float_kind': lambda x: f'{x:.3f}'})}, "
                f"Median score: {best_median_score:.4e}"
            )
        
        # 6. Return WSumScalarizer with that weight, and [c] as the cold-start center
        return WSumScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, best_weight, self.state.frozen_front_F), [c]


    def CreateCSUMScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:      
        # Replenish weight list if empty
        if len(self.exploration_weights) == 0:
            n_weights = 3 * self.f_count
            self.exploration_weights = self.GenerateExplorationWeights(n_weights)
        
        # 1. Pick a random starting point in normalized [0,1]^D ROI space
        c = np.random.rand(self.x_count)
        
        # 2. Find the 5 nearest archive points (in normalized coordinate space)
        k_neighbors = min(5, self.internal_archive.n_size)
        _, indices = self.internal_archive.query_k_nearest(c.reshape(1, -1), k=k_neighbors, distance_upper_bound=np.sqrt(self.x_count))
        neighbor_indices = indices.ravel()
        neighbor_indices = neighbor_indices[neighbor_indices >= 0]  # Filter missing
        
        # 3. Get raw objectives and normalize using frozen bounds
        f_neighbors = self.internal_archive.get_f_by_indices(neighbor_indices)
        f_L, _, f_D = self.state.GetNormalizationBounds(self.internal_archive)
        f_norm_neighbors = (f_neighbors - f_L) / f_D  # shape (k, M)
        
        # 4. For each remaining weight, compute median Augmented Chebyshev score across neighbors
        best_weight_idx = -1
        best_median_score = np.inf
        rho = 0.01  # Must match the default rho in CSumScalarizer
        
        for i, w in enumerate(self.exploration_weights):
            # --- FIXED SCORES COMPUTATION ---
            weighted_f = f_norm_neighbors * w               # Element-wise multiplication, shape (k, M)
            max_term = np.max(weighted_f, axis=1)           # Chebyshev max term, shape (k,)
            sum_term = np.sum(weighted_f, axis=1)           # Augmented linear term, shape (k,)
            scores = max_term + rho * sum_term              # Final score, shape (k,)
            # --------------------------------
            
            median_score = np.median(scores)
            if median_score < best_median_score:
                best_median_score = median_score
                best_weight_idx = i
        
        # 5. Pick the best weight and remove it from the list
        best_weight = self.exploration_weights.pop(best_weight_idx)
        
        if self._is_debug_run:
            self._current_debug_event["Exploration"] = (
                f"CSUM cold start. Weight: {np.array2string(best_weight, formatter={'float_kind': lambda x: f'{x:.3f}'})}, "
                f"Median score: {best_median_score:.4e}"
            )
        
        # 6. Return CSumScalarizer with that weight, and [c] as the cold-start center
        return CSumScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, best_weight, self.state.frozen_front_F, rho=rho), [c]


    def CreateR2Scalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:
        # 1. Pick a random starting point in normalized [0,1]^D ROI space
        c = np.random.rand(self.x_count)

        if self._is_debug_run:
            self._current_debug_event["Exploration"] = (f"R2 cold start.")
        
        # 2. Return R2Scalarizer and [c] as the cold-start center
        return R2Scalarizer(state=self.state, cv_type=self.cma_parameters.cv_type, g_divisor = g_divisor), [c]
    
    def CreateEXPScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:
        c = np.random.rand(self.x_count)
        return ExpUHVIScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, hv_reference_factor=self.cma_parameters.hv_reference_factor), [c]
    
    
    def CreateCornerScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:
        idx = self.corner_queue.pop(0)
        front_F = self.state.frozen_front_F
        front_X = self.state.frozen_front_X
        
        best_row_idx = np.argmin(front_F[:, idx])
        raw_corner_x = front_X[best_row_idx]
        
        points_of_interest = [self._normalize_x(raw_corner_x)]
        return CornerScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, idx), points_of_interest


    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.roi_lower_bounds) / self.roi_range

    def _denormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        dernomalized = self.roi_lower_bounds + x_norm * self.roi_range
        dernomalized = np.clip(dernomalized, self.true_lower_bounds, self.true_upper_bounds)
        return dernomalized
