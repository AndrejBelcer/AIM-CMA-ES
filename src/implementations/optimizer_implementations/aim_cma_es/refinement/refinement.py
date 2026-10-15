import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.refinement.local_region import LocalRegion
from src.implementations.optimizer_implementations.aim_cma_es.refinement.manifold_model import ManifoldModel

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinementParameters
from src.implementations.optimizer_implementations.aim_cma_es.enums import LocalityStrategy, SparsifierStrategy, FitterStrategy, SamplerStrategy, ValidatorStrategy


from src.implementations.optimizer_implementations.aim_cma_es.refinement.locality_gatherers import ILocalityGatherer, FpsUniformRadiusGatherer
from src.implementations.optimizer_implementations.aim_cma_es.refinement.sparsifiers import ISparsifier, VoxelSparsifier, RandomVoxelSparsifier, NoOpSparsifier
from src.implementations.optimizer_implementations.aim_cma_es.refinement.manifold_fitters import IManifoldFitter, StandardSvdFitter, TrimmedSvdFitter, RansacSvdFitter
from src.implementations.optimizer_implementations.aim_cma_es.refinement.samplers import ISampler, GridSampler, ModelAndPerturbedGridSampler, RandomSampler
from src.implementations.optimizer_implementations.aim_cma_es.refinement.validators import IValidator, NoOpValidator



LOCALITY_REGISTRY: Dict[LocalityStrategy, Type[ILocalityGatherer]] = {
    LocalityStrategy.FPS_UNIFORM_RADIUS: FpsUniformRadiusGatherer,
}

SPARSIFIER_REGISTRY: Dict[SparsifierStrategy, Type[ISparsifier]] = {
    SparsifierStrategy.NONE: NoOpSparsifier,
    SparsifierStrategy.VOXEL: VoxelSparsifier,
    SparsifierStrategy.RVOXEL: RandomVoxelSparsifier,
}

FITTER_REGISTRY: Dict[FitterStrategy, Type[IManifoldFitter]] = {
    FitterStrategy.STANDARD_SVD: StandardSvdFitter,
    FitterStrategy.TRIMMED_SVD: TrimmedSvdFitter,
    FitterStrategy.RANSAC: RansacSvdFitter,
}

SAMPLER_REGISTRY: Dict[SamplerStrategy, Type[ISampler]] = {
    SamplerStrategy.GRID: GridSampler,
    SamplerStrategy.DOUBLE_GRID: ModelAndPerturbedGridSampler,
    SamplerStrategy.RANDOM: RandomSampler,
}

VALIDATOR_REGISTRY: Dict[ValidatorStrategy, Type[IValidator]] = {
    ValidatorStrategy.NONE: NoOpValidator,
}

# ==========================================
# 6. THE MAIN REFINEMENT PIPELINE
# ==========================================

class Refinement:
    """
    The orchestrator. It initializes the chosen strategies and runs the 5-stage pipeline.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, refinement_parameters: RefinementParameters) -> None:
        
        self.x_count = x_count
        self.f_count = f_count
        self.g_count = g_count
        self.refinement_parameters = refinement_parameters

        # Initialize the pipeline components using the registries
        try:
            self.gatherer = LOCALITY_REGISTRY[refinement_parameters.locality_strategy](refinement_parameters, x_count, f_count)
            self.sparsifier = SPARSIFIER_REGISTRY[refinement_parameters.sparsifier_strategy](refinement_parameters, x_count, f_count)
            self.fitter = FITTER_REGISTRY[refinement_parameters.fitter_strategy](refinement_parameters, x_count, f_count)
            self.sampler = SAMPLER_REGISTRY[refinement_parameters.sampler_strategy](refinement_parameters, x_count, f_count)
            self.validator = VALIDATOR_REGISTRY[refinement_parameters.validator_strategy](refinement_parameters, x_count, f_count)
        except KeyError as e:
            raise NotImplementedError(f"Strategy not found in registry: {e}")

    def get_candidates(self, archive: InternalArchive, front_indices: np.ndarray, f_L: np.ndarray, f_D: np.ndarray) -> np.ndarray:
        """
        Executes the 5-stage Refinement pipeline.
        """
        if len(front_indices) < 2:
            return np.empty((0, self.x_count))

        # Stage 1: Gather Local Regions
        regions = self.gatherer.gather(archive, front_indices, f_L, f_D)
        
        all_candidates = []

        for region in regions:
            # Stage 2: Sparsify (Reduce Density Bias)
            sparse_region = self.sparsifier.sparsify(region, archive, f_L, f_D)
            
            # Stage 3: Fit Manifold
            model = self.fitter.fit(sparse_region, archive)
            if model is None:
                continue # Skip if fitting failed (e.g., not enough points)
                
            # Stage 4: Sample Candidates
            raw_candidates = self.sampler.sample(model)
            if len(raw_candidates) == 0:
                continue
                
            # Stage 5: Validate (Sanity Check against boundaries)
            valid_candidates = self.validator.validate(raw_candidates, sparse_region, archive)
            
            if len(valid_candidates) > 0:
                all_candidates.append(valid_candidates)

        # Combine all valid candidates
        if not all_candidates:
            return np.empty((0, self.x_count))

        return np.vstack(all_candidates)
