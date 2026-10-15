import numpy as np

from typing import List, Tuple, Optional, Union, Dict, Any

from src.implementations.optimizer_implementations.aim_cma_es.enums import Archive_Type, Initializer_Type, CMA_Type, CVType
from src.implementations.optimizer_implementations.aim_cma_es.enums import LocalityStrategy, SparsifierStrategy, FitterStrategy, SamplerStrategy, ValidatorStrategy

class DimensionScaledParameter:
    """Values interpolated between dimensions 2 and 40, or held constant."""

    def __init__(self, value_at_dimension_2:float= 0.3, value_at_dimension_40: float = 0.3, constant_value: Optional[float] = None):
        if (constant_value is None):
            self.value_at_dimension_2 = value_at_dimension_2
            self.value_at_dimension_40 = value_at_dimension_40
        else:
            self.value_at_dimension_2 = constant_value
            self.value_at_dimension_40 = constant_value


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
                 initial_lhs_size_factor: int = 4, # * sqrt(f(=2)) * sqrt(x)
                 additional_optimizer_evaluations_per_variable: int = 5,
                 final_lhs_size_factor: int = 25,
                 use_L2_weight_normalization: bool = False,
                 initial_trust_region_radius: float = 0.1,
                 final_trust_region_radius: float = 5e-3,
                 seed: Optional[int] = None,
                 initializer_type: Initializer_Type = Initializer_Type.COBYQA
                 ) -> None:
        self.initial_lhs_size_factor = initial_lhs_size_factor
        self.additional_optimizer_evaluations_per_variable = additional_optimizer_evaluations_per_variable
        self.final_lhs_size_factor = final_lhs_size_factor
        self.use_L2_weight_normalization = use_L2_weight_normalization
        self.initial_trust_region_radius = initial_trust_region_radius
        self.final_trust_region_radius = final_trust_region_radius
        self.seed = seed if (seed is not None) else np.random.randint(1, 5555)
        self.initializer_type = initializer_type


class CovarianceWarmStartParameters():
    """Directional weight, isotropic weight, and evolution-path trust."""

    def __init__(self,
                 direction_weight: DimensionScaledParameter = DimensionScaledParameter(value_at_dimension_2=0.6, value_at_dimension_40=0.4),
                 isotropic_weight: DimensionScaledParameter = DimensionScaledParameter(value_at_dimension_2=0.2, value_at_dimension_40=0.4),
                 path_trust: DimensionScaledParameter = DimensionScaledParameter(value_at_dimension_2=0.8, value_at_dimension_40=0.6)
                 ) -> None:
        self.direction_weight = direction_weight
        self.isotropic_weight = isotropic_weight
        self.path_trust = path_trust


class WarmStartParameters:
    def __init__(self, 
                 min_warm_start_sigma: float = 0.005,
                 max_warm_start_sigma: float = 0.2,
                 fallback_warm_start_sigma: float = 0.11,
                 global_exploration_sigma: float = 0.15,
                 feasibility_covariance: CovarianceWarmStartParameters = CovarianceWarmStartParameters(direction_weight=DimensionScaledParameter(value_at_dimension_2=0.5, value_at_dimension_40=0.4), isotropic_weight=DimensionScaledParameter(value_at_dimension_2=0.3, value_at_dimension_40=0.4)),
                 corner_search_covariance: CovarianceWarmStartParameters = CovarianceWarmStartParameters(direction_weight=DimensionScaledParameter(value_at_dimension_2=0.5, value_at_dimension_40=0.4), isotropic_weight=DimensionScaledParameter(value_at_dimension_2=0.3, value_at_dimension_40=0.4)),
                 exploitation_covariance: CovarianceWarmStartParameters = CovarianceWarmStartParameters(direction_weight=DimensionScaledParameter(value_at_dimension_2=0.5, value_at_dimension_40=0.4), isotropic_weight=DimensionScaledParameter(value_at_dimension_2=0.3, value_at_dimension_40=0.4)),
                 local_exploration_covariance: CovarianceWarmStartParameters = CovarianceWarmStartParameters(direction_weight=DimensionScaledParameter(value_at_dimension_2=0.4, value_at_dimension_40=0.2), isotropic_weight=DimensionScaledParameter(value_at_dimension_2=0.3, value_at_dimension_40=0.4)),
                 global_exploration_covariance: CovarianceWarmStartParameters = CovarianceWarmStartParameters(direction_weight=DimensionScaledParameter(value_at_dimension_2=0.2, value_at_dimension_40=0.1), isotropic_weight=DimensionScaledParameter(value_at_dimension_2=0.7, value_at_dimension_40=0.8), path_trust = DimensionScaledParameter(value_at_dimension_2=0.4, value_at_dimension_40=0.2)),
                 initial_mean_shift_factor: float = 1.0,
                 warm_start_sigma_factor: float = 0.8,
                 rank_tie_tolerance_fraction: float = 1e-3,
                 max_neighborhood_points: int = 10000,
                 feasibility_radius_factor: float = 0.2,
                 local_exploration_radius_factor: float = 0.15,
                 corner_search_radius_factor: float = 0.04,
                 global_exploration_radius_factor: float = 0.2
                ) -> None:
        self.min_warm_start_sigma = min_warm_start_sigma
        self.max_warm_start_sigma = max_warm_start_sigma
        self.fallback_warm_start_sigma = fallback_warm_start_sigma
        self.global_exploration_sigma = global_exploration_sigma

        self.feasibility_covariance = feasibility_covariance
        self.corner_search_covariance = corner_search_covariance
        self.exploitation_covariance = exploitation_covariance
        self.local_exploration_covariance = local_exploration_covariance
        self.global_exploration_covariance = global_exploration_covariance

        self.initial_mean_shift_factor = initial_mean_shift_factor
        self.warm_start_sigma_factor = warm_start_sigma_factor
        self.rank_tie_tolerance_fraction = rank_tie_tolerance_fraction
        self.max_neighborhood_points = max_neighborhood_points
        self.feasibility_radius_factor = feasibility_radius_factor
        self.local_exploration_radius_factor = local_exploration_radius_factor
        self.corner_search_radius_factor = corner_search_radius_factor
        self.global_exploration_radius_factor = global_exploration_radius_factor


class CMAComponentParameters:
    """User settings for AIM's scalarized CMA-ES components and their scheduling.

    The separate cma_implementation.CMAParameters computes internal CMA constants.
    """

    def __init__(self, 
                 population_size: Optional[int] = None,
                 max_generations_base: int = 10,
                 max_generations_per_sqrt_dimension: int = 18,
                 initial_local_exploration_probability: float = 0.25,
                 final_local_exploration_probability: float = 0.5,
                 hv_reference_factor: float = 10.0,
                 scalarized_improvement_tolerance: float = 1e-8,
                 exploitation_hv_gain_tolerance: float = 2e-4,  # Minimum normalized HV gain for Exploitation.
                 exploitation_tabu_distance_tolerance: float = 7e-3,  # Normalized gap-target exclusion distance.
                 corner_improvement_tolerance: float = 1e-5,  # Improvement required to re-queue Corner Search.
                 global_exploration_cycle_base_runs: int = 0,  # Cycle baseline; initial pending runs remain zero.
                 max_global_exploration_runs: int = 7,  # Cap on scheduled Global Exploration runs per cycle.
                 global_exploration_runs_increment: int = 2,  # Increase after each completed Refinement run.
                 warm_start_parameters: WarmStartParameters = WarmStartParameters(),
                 cv_type: CVType = CVType.LOGSUMEXP,        # Scalarization for constraints
                 cma_type: CMA_Type = CMA_Type.CMA_SYM
                ) -> None:
        self.population_size = population_size
        self.max_generations_base = max_generations_base
        self.max_generations_per_sqrt_dimension = max_generations_per_sqrt_dimension
        self.initial_local_exploration_probability = initial_local_exploration_probability
        self.final_local_exploration_probability = final_local_exploration_probability
        self.hv_reference_factor = hv_reference_factor
        self.scalarized_improvement_tolerance = scalarized_improvement_tolerance
        self.exploitation_hv_gain_tolerance = exploitation_hv_gain_tolerance
        self.exploitation_tabu_distance_tolerance = exploitation_tabu_distance_tolerance
        self.corner_improvement_tolerance = corner_improvement_tolerance
        self.global_exploration_cycle_base_runs = global_exploration_cycle_base_runs
        self.max_global_exploration_runs = max_global_exploration_runs
        self.global_exploration_runs_increment = global_exploration_runs_increment
        self.warm_start_parameters = warm_start_parameters
        self.cv_type = cv_type
        self.cma_type = cma_type


class RefinementParameters:
    """Local Pareto-set modeling and sampling settings for Refinement."""

    def __init__(self,
                # Solver Parameters
                refinement_trigger_low_gain_count: int = 3,
                refinement_passes_per_run: int = 9,
                
                # Hyperparameters
                min_local_regions: int = 7,
                max_local_regions: int = 12,
                local_regions_per_pass: int = 5,
                grid_points_per_tangent_axis: int = 20,
                tangent_grid_extension: float = 0.25,
                orthogonal_noise_factor: float = 0.3,
                local_region_radius_factor: float = 1.2,

                # Strategies
                locality_strategy: LocalityStrategy = LocalityStrategy.FPS_UNIFORM_RADIUS,
                sparsifier_strategy: SparsifierStrategy = SparsifierStrategy.RVOXEL,
                fitter_strategy: FitterStrategy = FitterStrategy.RANSAC,
                sampler_strategy: SamplerStrategy = SamplerStrategy.DOUBLE_GRID,
                validator_strategy: ValidatorStrategy = ValidatorStrategy.NONE,
                ) -> None:

        # Solver Parameters
        self.refinement_trigger_low_gain_count: int = refinement_trigger_low_gain_count
        self.refinement_passes_per_run: int = refinement_passes_per_run

        # Hyperparameters
        self.min_local_regions: int = min_local_regions
        self.max_local_regions: int = max_local_regions
        self.local_regions_per_pass: int = local_regions_per_pass
        self.grid_points_per_tangent_axis: int = grid_points_per_tangent_axis
        self.tangent_grid_extension: float = tangent_grid_extension
        self.orthogonal_noise_factor: float = orthogonal_noise_factor
        self.local_region_radius_factor: float = local_region_radius_factor

        # Strategies
        self.locality_strategy: LocalityStrategy = locality_strategy
        self.sparsifier_strategy: SparsifierStrategy = sparsifier_strategy
        self.fitter_strategy: FitterStrategy = fitter_strategy
        self.sampler_strategy: SamplerStrategy = sampler_strategy
        self.validator_strategy: ValidatorStrategy = validator_strategy
