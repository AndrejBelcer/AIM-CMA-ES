import numpy as np

from typing import List, Tuple, Optional, Union, Dict, Any

from src.implementations.optimizer_implementations.aim_cma_es.enums import Archive_Type, Initializer_Type, CMA_Type, CVType
from src.implementations.optimizer_implementations.aim_cma_es.enums import LocalityStrategy, SparsifierStrategy, FitterStrategy, SamplerStrategy, ValidatorStrategy

class DimensionalParameter:
    def __init__(self, d2_value:float= 0.3, d40_value: float = 0.3, forced_value: Optional[float] = None):
        if (forced_value is None):
            self.d2_value = d2_value
            self.d40_value = d40_value
        else:
            self.d2_value = forced_value
            self.d40_value = forced_value


class ArchiveParameters:
    def __init__(self,
                 archive_type: Archive_Type = Archive_Type.HNSW,
                 checkpoint_interval: int = 1000,
                 kd_leaf_size: int = 15,
                 ball_leaf_size: int = 40,
                 default_ef: int = 100
                 ) -> None:
            self.archive_type = archive_type
            # Generic Tree parameters
            self.checkpoint_interval = checkpoint_interval
            # KD parameters
            self.kd_leaf_size = kd_leaf_size
            # Ball parameters
            self.ball_leaf_size = ball_leaf_size
            # HNSW parameters
            self.default_ef = default_ef


class InitializationParameters:
    def __init__(self,
                 initial_points_multiplier: int = 4, # * sqrt(f(=2)) * sqrt(x)
                 allowed_evaluations_per_dim: int = 5,
                 final_lhs: int = 25,
                 use_L2: bool = False,
                 rhobeg: float = 0.1,
                 rhoend: float = 5e-3,
                 seed: Optional[int] = None,
                 initializer_type: Initializer_Type = Initializer_Type.COBYQA
                 ) -> None:
        self.initial_points_multiplier = initial_points_multiplier
        self.allowed_evaluations_per_dim = allowed_evaluations_per_dim
        self.final_lhs = final_lhs
        self.use_L2 = use_L2
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.seed = seed if (seed is not None) else np.random.randint(1, 5555)
        self.initializer_type = initializer_type


class CovarianceWarmParameters():
    def __init__(self,
                 velocity_contribution: DimensionalParameter = DimensionalParameter(d2_value=0.6, d40_value=0.4),
                 shrinkage: DimensionalParameter = DimensionalParameter(d2_value=0.2, d40_value=0.4),
                 path_trust: DimensionalParameter = DimensionalParameter(d2_value=0.8, d40_value=0.6)
                 ) -> None:
        self.velocity_contribution = velocity_contribution
        self.shrinkage = shrinkage
        self.path_trust = path_trust


class WarmStartParameters:
    def __init__(self, 
                 min_sigma: float = 0.005,
                 max_sigma: float = 0.2,
                 sigma_fallback: float = 0.11,
                 exploration_sigma: float = 0.15,
                 feasability_covariance: CovarianceWarmParameters = CovarianceWarmParameters(velocity_contribution=DimensionalParameter(d2_value=0.5, d40_value=0.4), shrinkage=DimensionalParameter(d2_value=0.3, d40_value=0.4)),
                 corner_covariance: CovarianceWarmParameters = CovarianceWarmParameters(velocity_contribution=DimensionalParameter(d2_value=0.5, d40_value=0.4), shrinkage=DimensionalParameter(d2_value=0.3, d40_value=0.4)),
                 gap_covariance: CovarianceWarmParameters = CovarianceWarmParameters(velocity_contribution=DimensionalParameter(d2_value=0.5, d40_value=0.4), shrinkage=DimensionalParameter(d2_value=0.3, d40_value=0.4)),
                 uhvi_covariance: CovarianceWarmParameters = CovarianceWarmParameters(velocity_contribution=DimensionalParameter(d2_value=0.4, d40_value=0.2), shrinkage=DimensionalParameter(d2_value=0.3, d40_value=0.4)),
                 wsum_covariance: CovarianceWarmParameters = CovarianceWarmParameters(velocity_contribution=DimensionalParameter(d2_value=0.2, d40_value=0.1), shrinkage=DimensionalParameter(d2_value=0.7, d40_value=0.8), path_trust = DimensionalParameter(d2_value=0.4, d40_value=0.2)),
                 initial_sigma_move: float = 1.0,
                 sigma_factor: float = 0.8,
                 tie_tolerance: float = 1e-3,
                 max_points_in_hypersphere: int = 10000,
                 r_feasibility_perx: float = 0.2,
                 r_uhvi_perx: float = 0.15,
                 r_corner_perx: float = 0.04,
                 r_exploration_perx: float = 0.2
                ) -> None:
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_fallback = sigma_fallback
        self.exploration_sigma = exploration_sigma

        self.feasability_covariance = feasability_covariance
        self.corner_covariance = corner_covariance
        self.gap_covariance = gap_covariance
        self.uhvi_covariance = uhvi_covariance
        self.wsum_covariance = wsum_covariance

        self.initial_sigma_move = initial_sigma_move
        self.sigma_factor = sigma_factor
        self.tie_tolerance = tie_tolerance
        self.max_points_in_hypersphere = max_points_in_hypersphere
        self.r_feasibility_perx = r_feasibility_perx
        self.r_uhvi_perx = r_uhvi_perx
        self.r_corner_perx = r_corner_perx
        self.r_exploration_perx = r_exploration_perx


class CMAParameters:
    def __init__(self, 
                 popsize: Optional[int] = None,
                 max_iter_const: int = 10,
                 max_iter_per_sqrt_dim: int = 25,
                 uhvi_start_chance_initial: float = 0.25,
                 uhvi_start_chance_final: float = 0.5,
                 hv_reference_factor: float = 10.0,
                 convergence_tolerance: float = 1e-8,
                 f_HV_tolerance: float = 2e-4,              # Min HV gain to consider gap fill successful
                 f_Tabu_tolerance: float = 7e-3,            # Min distance to consider a gap target 'new'
                 f_significant_change: float = 1e-5,        # Improvement required to re-queue a corner
                 starting_exploration_kernels: int = 0,     # How many exploration kernels to start with
                 end_exploration_kernels: int = 7,         # The cap for the number of exploration kernels
                 step_exploration_kernels: int = 2,         # How quicly the exploration kernel count increases
                 warm_start_parameters: WarmStartParameters = WarmStartParameters(),
                 cv_type: CVType = CVType.LOGSUMEXP,        # Scalarization for constraints
                 cma_type: CMA_Type = CMA_Type.CMA_SYM
                ) -> None:
        self.popsize = popsize
        self.max_iter_const = max_iter_const
        self.max_iter_per_sqrt_dim = max_iter_per_sqrt_dim
        self.uhvi_start_chance_initial = uhvi_start_chance_initial
        self.uhvi_start_chance_final = uhvi_start_chance_final
        self.hv_reference_factor = hv_reference_factor
        self.convergence_tolerance = convergence_tolerance
        self.f_HV_tolerance = f_HV_tolerance
        self.f_Tabu_tolerance = f_Tabu_tolerance
        self.f_significant_change = f_significant_change
        self.starting_exploration_kernels = starting_exploration_kernels
        self.end_exploration_kernels = end_exploration_kernels
        self.step_exploration_kernels = step_exploration_kernels
        self.warm_start_parameters = warm_start_parameters
        self.cv_type = cv_type
        self.cma_type = cma_type


class RefinerParameters:
    def __init__(self,
                # Solver Parameters
                refiner_activation_at_count: int = 3,
                refiner_L_loops: int = 9,
                
                # Hyperparameters
                n_regions_min: int = 7,
                n_regions_max: int = 12,
                n_targeted_regions: int = 5,
                n_samples_per_gap: int = 20,
                extension: float = 0.25,
                noise_factor: float = 0.3,
                bubble_factor: float = 1.2,

                # Strategies
                locality_strategy: LocalityStrategy = LocalityStrategy.FPS_UNIFORM_RADIUS,
                sparsifier_strategy: SparsifierStrategy = SparsifierStrategy.RVOXEL,
                fitter_strategy: FitterStrategy = FitterStrategy.RANSAC,
                sampler_strategy: SamplerStrategy = SamplerStrategy.DOUBLE_GRID,
                validator_strategy: ValidatorStrategy = ValidatorStrategy.NONE,
                ) -> None:

        # Solver Parameters
        self.refiner_activation_at_count: int = refiner_activation_at_count
        self.refiner_L_loops: int = refiner_L_loops

        # Hyperparameters
        self.n_regions_min: int = n_regions_min
        self.n_regions_max: int = n_regions_max
        self.n_targeted_regions: int = n_targeted_regions
        self.n_samples_per_gap: int = n_samples_per_gap
        self.extension: float = extension
        self.noise_factor: float = noise_factor
        self.bubble_factor: float = bubble_factor

        # Strategies
        self.locality_strategy: LocalityStrategy = locality_strategy
        self.sparsifier_strategy: SparsifierStrategy = sparsifier_strategy
        self.fitter_strategy: FitterStrategy = fitter_strategy
        self.sampler_strategy: SamplerStrategy = sampler_strategy
        self.validator_strategy: ValidatorStrategy = validator_strategy
