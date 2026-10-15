import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any

from src.core.optimizer import Optimizer
from src.core.event_log import Event
from src.implementations.optimizer_implementations.aim_cma_es.state_tracker import StateTracker
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.scalarizer_base import Scalarizer
from src.implementations.optimizer_implementations.aim_cma_es.scalarization.strategies import FeasibilityScalarizer, CornerSearchScalarizer, ExploitationScalarizer, LocalExplorationScalarizer, GlobalExplorationWeightedSumScalarizer, GlobalExplorationChebyshevScalarizer


from src.implementations.optimizer_implementations.aim_cma_es.enums import Archive_Type, Initializer_Type, CMA_Type, SolverPhase, ScalarizationMode, InitializerPhase
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import ArchiveParameters, CMAComponentParameters, RefinementParameters, InitializationParameters

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.kd_tree_archive import KDTreeArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.kd_tree_archive_n import KDTreeArchiveN
from src.implementations.optimizer_implementations.aim_cma_es.archive.ball_tree_archive import BallTreeArchive
from src.implementations.optimizer_implementations.aim_cma_es.archive.hnsw_archive import HNSWArchive

from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer import Initializer
from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer_cobyqa import InitializerCOBYQA
from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer_cobyqa_official import InitializerCOBYQA_O

from src.implementations.optimizer_implementations.aim_cma_es.cma_component.cma_manager import CmaManager

from src.implementations.optimizer_implementations.aim_cma_es.refinement.refinement import Refinement




class AimCma(Optimizer):
    def __init__(self,
                 solver_name: str = "AIM-CMA-ES",
                 # --- Sampling Tolerance ---
                 duplicate_distance_factor: float = 1e-4,
                 # --- Configurations ---
                 archive_parameters: ArchiveParameters = ArchiveParameters(),
                 initialization_parameters: InitializationParameters = InitializationParameters(),
                 cma_parameters: CMAComponentParameters = CMAComponentParameters(),
                 refinement_parameters: RefinementParameters = RefinementParameters(),
                ) -> None:
        
        super().__init__(solver_name)
        
        self.duplicate_distance_factor = duplicate_distance_factor
        
        self.archive_parameters = archive_parameters
        self.initialization_parameters = initialization_parameters
        self.cma_parameters = cma_parameters
        self.refinement_parameters = refinement_parameters
        
        self.refinement_trigger_low_gain_count = refinement_parameters.refinement_trigger_low_gain_count
        self.refinement_passes_per_run = refinement_parameters.refinement_passes_per_run

        # The phase set during ask, so that tell function can know the current phase
        self.current_phase: SolverPhase = SolverPhase.NONE
        # The phase for setting up next phase in some cases
        self.next_phase: SolverPhase = SolverPhase.INITIALIZATION

        # Global Exploration scheduling state
        self.global_exploration_runs_remaining: int = 0 # Number of pending Global Exploration runs
        self.global_exploration_weight_pool: List[np.ndarray] = []
        self.global_exploration_runs_per_cycle: int = 0 # Global Exploration runs to schedule after Refinement
        # Probability of selecting Local Exploration instead of Exploitation
        self.local_exploration_probability: float = 0.0

        # Logic Queues & Counters
        self.corner_search_queue: List[int] = []
        self.pending_cma_best: Optional[np.ndarray] = None
        
        self.low_gain_exploitation_count = 0
        self.refinement_pass_count = 0
        
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
        
        self.corner_search_queue = list(range(self.f_count))

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
        self.refinement = Refinement(x_count=n_vars, f_count=n_obj, g_count=n_constr, refinement_parameters=self.refinement_parameters)

        self.duplicate_distance_tolerance = self.duplicate_distance_factor * np.sqrt(n_vars)
        self.current_phase = SolverPhase.NONE
        self.next_phase: SolverPhase = SolverPhase.INITIALIZATION
        # Initial Global Exploration remains unscheduled, regardless of the cycle baseline.
        self.global_exploration_runs_remaining = 0
        self.global_exploration_runs_per_cycle: int = self.cma_parameters.global_exploration_cycle_base_runs
        self.global_exploration_weight_pool = []
        self.local_exploration_probability = self.cma_parameters.initial_local_exploration_probability
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
        is_novel, archive_indices = self.internal_archive.check_duplicates_with_indices(candidates, self.duplicate_distance_tolerance)
        
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

        # Priority 2: Refinement Phase
        if self.next_phase == SolverPhase.REFINEMENT:
            self.current_phase = SolverPhase.REFINEMENT
            return self._ask_refinement()

        # Priority 3: CMA Phase
        self.current_phase = SolverPhase.CMA
        return self._ask_cma()
    
    
    def _ask_initializer(self) -> np.ndarray:
        # Check if we are in the optimizer phase but don't have an active optimizer running
        if self.initializer.phase == InitializerPhase.OPTIMIZER and not self.initializer.is_optimizer_active:
            
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
        
    def _ask_refinement(self) -> np.ndarray:
        f_L, _, f_D = self.state.GetNormalizationBounds(self.internal_archive)
        candidates = self.refinement.get_candidates(archive=self.internal_archive, front_indices=self.state.empirical_front_indices, f_L = f_L, f_D = f_D)
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
            # Preserve reported behavior: only batches with new evaluations count.
            if(self.current_phase == SolverPhase.REFINEMENT):
                self.refinement_pass_count += 1
                if self.refinement_pass_count >= self.refinement_passes_per_run:
                    self.refinement_pass_count = 0
                    self.next_phase = SolverPhase.CMA
                    self.global_exploration_runs_per_cycle = min(self.global_exploration_runs_per_cycle + self.cma_parameters.global_exploration_runs_increment, self.cma_parameters.max_global_exploration_runs)
                    self.global_exploration_runs_remaining = self.global_exploration_runs_per_cycle
                    self.local_exploration_probability = self.cma_parameters.final_local_exploration_probability
                # Check for corner improvement.
                if(self._is_debug_run):
                    self._current_debug_event["Corner Improvement"] = self.state.DebugCornerImprovement(corner_improvement_tolerance=self.cma_parameters.corner_improvement_tolerance)
                list_of_improvements: list[int] = self.state.CheckCornerImprovement(corner_improvement_tolerance=self.cma_parameters.corner_improvement_tolerance)
                for i in list_of_improvements:
                        if i not in self.corner_search_queue:
                            self.corner_search_queue.append(i)
            
            
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

        # --- Exploitation / Local Exploration hypervolume gain ---
        if mode in [ScalarizationMode.EXPLOITATION, ScalarizationMode.LOCAL_EXPLORATION]:
            hv_start = scalarizer.indicator.frozen_score
            live_indices = self.state.empirical_front_indices
            raw_live_front, _ = self.internal_archive.get_values_by_indices(live_indices)
            norm_live_front = scalarizer.normalize(raw_live_front)
            hv_end = scalarizer.indicator.calculate_front_score(norm_live_front)

            gain = hv_end - hv_start
            
            if mode == ScalarizationMode.EXPLOITATION:
                if gain < self.cma_parameters.exploitation_hv_gain_tolerance:                    
                    self.state.add_exploitation_tabu_target(scalarizer.target_F_raw)
                    if(self._is_debug_run):
                        self._current_debug_event["Tabu Add"] = f"Low-gain Exploitation run. Adding target to tabu."
                        self._current_debug_event["Tabu List"] = self.state.exploitation_tabu_targets_F_raw.copy()
                    
                    # Accumulate low-gain Exploitation runs; successes do not reset this count.
                    self.low_gain_exploitation_count += 1
                    if self.low_gain_exploitation_count >= self.refinement_trigger_low_gain_count:
                        self.low_gain_exploitation_count = 0
                        # Permanent tolerance reduction
                        self.duplicate_distance_tolerance = max(1e-6, self.duplicate_distance_tolerance * 0.5)
                        self.next_phase = SolverPhase.REFINEMENT
                        if(self._is_debug_run):
                            self._current_debug_event["Refinement Start"] = "Refinement started."

        # --- CORNER RE-QUEUE LOGIC ---
        if(self._is_debug_run):
            self._current_debug_event["Corner Improvement"] = self.state.DebugCornerImprovement(corner_improvement_tolerance=self.cma_parameters.corner_improvement_tolerance)
        list_of_improvements: list[int] = self.state.CheckCornerImprovement(corner_improvement_tolerance=self.cma_parameters.corner_improvement_tolerance)
        for i in list_of_improvements:
            if not((mode == ScalarizationMode.CORNER_SEARCH and scalarizer.objective_index == i)):
                if i not in self.corner_search_queue:
                    self.corner_search_queue.append(i)

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


    def GenerateGlobalExplorationWeights(self, number_of_weights: int) -> List[np.ndarray]:
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
        
        # 0. Scheduled Global Exploration 
        if (self.global_exploration_runs_remaining > 0):
            self.global_exploration_runs_remaining -= 1
            return self.CreateGlobalExplorationChebyshevScalarizer(g_divisor)

        n_front = len(self.state.frozen_front_X)
        # 1. Feasibility Mode
        if n_front == 0:
            return FeasibilityScalarizer(self.state, self.cma_parameters.cv_type, g_divisor), []

        # 2. Corner Search
        if n_front < self.f_count and len(self.corner_search_queue) == 0:
            if(self._is_debug_run):
                self._current_debug_event["Small Front"] = "Front too small for gaps. Refilling corner queue."
            self.corner_search_queue = list(range(self.f_count))
        
        if self.corner_search_queue:
            return self.CreateCornerSearchScalarizer(g_divisor)

        # 3. Exploitation
        if np.random.rand() > self.local_exploration_probability:
            exploitation_target_F_raw, gap_endpoints_X_raw, all_tabu = self.state.GetExploitationGapTarget(self.cma_parameters.exploitation_tabu_distance_tolerance)
            
            if(all_tabu):
                if(self._is_debug_run):
                    self._current_debug_event["Tabu Full"] = "All gaps tabu. Clearing list and running Local Exploration (UHVI)."
                self.state.clear_exploitation_tabu_targets()
            
            if (exploitation_target_F_raw is not None):
                points_of_interest = [self._normalize_x(p) for p in gap_endpoints_X_raw]
                if (self._is_debug_run):
                    self._current_debug_event["Exploitation_Target"] = exploitation_target_F_raw.copy()
                return ExploitationScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, exploitation_target_F_raw, hv_reference_factor=self.cma_parameters.hv_reference_factor), points_of_interest

        # 4. Local Exploration (UHVI)            
        return LocalExplorationScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, hv_reference_factor=self.cma_parameters.hv_reference_factor), []


    def CreateGlobalExplorationWeightedSumScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:      
        # Replenish weight list if empty
        if len(self.global_exploration_weight_pool) == 0:
            n_weights = 3 * self.f_count
            self.global_exploration_weight_pool = self.GenerateGlobalExplorationWeights(n_weights)
        
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
        
        for i, w in enumerate(self.global_exploration_weight_pool):
            scores = f_norm_neighbors @ w  # shape (k,)
            median_score = np.median(scores)
            if median_score < best_median_score:
                best_median_score = median_score
                best_weight_idx = i
        
        # 5. Pick the best weight and remove it from the list
        best_weight = self.global_exploration_weight_pool.pop(best_weight_idx)
        
        if self._is_debug_run:
            self._current_debug_event["Global Exploration"] = (
                f"Global Exploration (weighted sum) start. Weight: {np.array2string(best_weight, formatter={'float_kind': lambda x: f'{x:.3f}'})}, "
                f"Median score: {best_median_score:.4e}"
            )
        
        # 6. Return GlobalExplorationWeightedSumScalarizer with that weight, and [c] as the cold-start center
        return GlobalExplorationWeightedSumScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, best_weight, self.state.frozen_front_F), [c]


    def CreateGlobalExplorationChebyshevScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:      
        # Replenish weight list if empty
        if len(self.global_exploration_weight_pool) == 0:
            n_weights = 3 * self.f_count
            self.global_exploration_weight_pool = self.GenerateGlobalExplorationWeights(n_weights)
        
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
        augmentation_coefficient = 0.01  # Must match the default augmentation coefficient in GlobalExplorationChebyshevScalarizer
        
        for i, w in enumerate(self.global_exploration_weight_pool):
            # --- FIXED SCORES COMPUTATION ---
            weighted_f = f_norm_neighbors * w               # Element-wise multiplication, shape (k, M)
            max_term = np.max(weighted_f, axis=1)           # Chebyshev max term, shape (k,)
            sum_term = np.sum(weighted_f, axis=1)           # Augmented linear term, shape (k,)
            scores = max_term + augmentation_coefficient * sum_term              # Final score, shape (k,)
            # --------------------------------
            
            median_score = np.median(scores)
            if median_score < best_median_score:
                best_median_score = median_score
                best_weight_idx = i
        
        # 5. Pick the best weight and remove it from the list
        best_weight = self.global_exploration_weight_pool.pop(best_weight_idx)
        
        if self._is_debug_run:
            self._current_debug_event["Global Exploration"] = (
                f"Global Exploration (augmented Chebyshev) start. Weight: {np.array2string(best_weight, formatter={'float_kind': lambda x: f'{x:.3f}'})}, "
                f"Median score: {best_median_score:.4e}"
            )
        
        # 6. Return GlobalExplorationChebyshevScalarizer with that weight, and [c] as the cold-start center
        return GlobalExplorationChebyshevScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, best_weight, self.state.frozen_front_F, augmentation_coefficient=augmentation_coefficient), [c]


    
    
    
    def CreateCornerSearchScalarizer(self, g_divisor: np.ndarray) -> Tuple[Scalarizer, List[np.ndarray]]:
        idx = self.corner_search_queue.pop(0)
        front_F = self.state.frozen_front_F
        front_X = self.state.frozen_front_X
        
        best_row_idx = np.argmin(front_F[:, idx])
        raw_corner_x = front_X[best_row_idx]
        
        points_of_interest = [self._normalize_x(raw_corner_x)]
        return CornerSearchScalarizer(self.state, self.cma_parameters.cv_type, g_divisor, idx), points_of_interest


    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.roi_lower_bounds) / self.roi_range

    def _denormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        denormalized = self.roi_lower_bounds + x_norm * self.roi_range
        denormalized = np.clip(denormalized, self.true_lower_bounds, self.true_upper_bounds)
        return denormalized
