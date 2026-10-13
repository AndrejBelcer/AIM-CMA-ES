import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.refiner.local_bubble import LocalBubble
from src.implementations.optimizer_implementations.aim_cma_es.refiner.manifold_model import ManifoldModel

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinerParameters
from src.implementations.optimizer_implementations.aim_cma_es.enums import LocalityStrategy, SparsifierStrategy, FitterStrategy, SamplerStrategy, ValidatorStrategy


from src.implementations.optimizer_implementations.aim_cma_es.refiner.locality_gatherers import ILocalityGatherer, FpsUniformRadiusGatherer
from src.implementations.optimizer_implementations.aim_cma_es.refiner.sparsifiers import ISparsifier, VoxelSparsifier, RandomVoxelSparsifier, NoOpSparsifier
from src.implementations.optimizer_implementations.aim_cma_es.refiner.manifold_fitters import IManifoldFitter, StandardSvdFitter, TrimmedSvdFitter, RansacSvdFitter
from src.implementations.optimizer_implementations.aim_cma_es.refiner.samplers import ISampler, GridSampler, DoubleGridSampler, RandomSampler
from src.implementations.optimizer_implementations.aim_cma_es.refiner.validators import IValidator, NoOpValidator



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
    SamplerStrategy.DOUBLE_GRID: DoubleGridSampler,
    SamplerStrategy.RANDOM: RandomSampler,
}

VALIDATOR_REGISTRY: Dict[ValidatorStrategy, Type[IValidator]] = {
    ValidatorStrategy.NONE: NoOpValidator,
}

# ==========================================
# 6. THE MAIN REFINER PIPELINE
# ==========================================

class Refiner:
    """
    The orchestrator. It initializes the chosen strategies and runs the 5-stage pipeline.
    """
    def __init__(self, x_count: int, f_count: int, g_count: int, refiner_parameters: RefinerParameters) -> None:
        
        self.x_count = x_count
        self.f_count = f_count
        self.g_count = g_count
        self.refiner_parameters = refiner_parameters

        # Initialize the pipeline components using the registries
        try:
            self.gatherer = LOCALITY_REGISTRY[refiner_parameters.locality_strategy](refiner_parameters, x_count, f_count)
            self.sparsifier = SPARSIFIER_REGISTRY[refiner_parameters.sparsifier_strategy](refiner_parameters, x_count, f_count)
            self.fitter = FITTER_REGISTRY[refiner_parameters.fitter_strategy](refiner_parameters, x_count, f_count)
            self.sampler = SAMPLER_REGISTRY[refiner_parameters.sampler_strategy](refiner_parameters, x_count, f_count)
            self.validator = VALIDATOR_REGISTRY[refiner_parameters.validator_strategy](refiner_parameters, x_count, f_count)
        except KeyError as e:
            raise NotImplementedError(f"Strategy not found in registry: {e}")

    def get_candidates(self, archive: InternalArchive, front_indices: np.ndarray, f_L: np.ndarray, f_D: np.ndarray) -> np.ndarray:
        """
        Executes the 5-stage Refiner pipeline.
        """
        if len(front_indices) < 2:
            return np.empty((0, self.x_count))

        # Stage 1: Gather Local Bubbles
        bubbles = self.gatherer.gather(archive, front_indices, f_L, f_D)
        
        all_candidates = []

        for bubble in bubbles:
            # Stage 2: Sparsify (Reduce Density Bias)
            sparse_bubble = self.sparsifier.sparsify(bubble, archive, f_L, f_D)
            
            # Stage 3: Fit Manifold
            model = self.fitter.fit(sparse_bubble, archive)
            if model is None:
                continue # Skip if fitting failed (e.g., not enough points)
                
            # Stage 4: Sample Candidates
            raw_candidates = self.sampler.sample(model)
            if len(raw_candidates) == 0:
                continue
                
            # Stage 5: Validate (Sanity Check against boundaries)
            valid_candidates = self.validator.validate(raw_candidates, sparse_bubble, archive)
            
            if len(valid_candidates) > 0:
                all_candidates.append(valid_candidates)

        # Combine all valid candidates
        if not all_candidates:
            return np.empty((0, self.x_count))

        return np.vstack(all_candidates)
