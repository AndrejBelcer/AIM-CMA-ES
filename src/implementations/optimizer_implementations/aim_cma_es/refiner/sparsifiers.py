import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import RefinerParameters


from src.implementations.optimizer_implementations.aim_cma_es.refiner.local_bubble import LocalBubble

class ISparsifier(ABC):
    def __init__(self, refiner_parameters: RefinerParameters, x_count: int, f_count: int):
        self.refiner_parameters = refiner_parameters
        self.x_count = x_count
        self.f_count = f_count

    @abstractmethod
    def sparsify(self, bubble: LocalBubble, archive: InternalArchive, f_L: np.ndarray, f_D: np.ndarray) -> LocalBubble:
        pass

class VoxelSparsifier(ISparsifier):
    """
    Destroys density bias (convergence streaks) by snapping points to a grid 
    and keeping only one point per voxel. 
    Operates in O(N) time and scales with the intrinsic dimensionality (M-1).
    """
    def sparsify(self, bubble: LocalBubble, archive: InternalArchive, f_L: np.ndarray, f_D: np.ndarray) -> LocalBubble:
        X_E = bubble.get_X_E(archive)
        n_points = len(X_E)
        target_k = 11 * (self.f_count - 1)
        if n_points <= target_k:
            return bubble

        intrinsic_dim = max(1.0, float(self.f_count - 1))
        diameter = 2.0 * bubble.radius
        voxel_size = max(diameter / (target_k ** (1.0 / intrinsic_dim)), 1e-6)

        # 1. Ensure we use a fixed integer type (e.g., int32)
        voxel_indices = np.round(X_E / voxel_size).astype(np.int32)
        
        # 2. THE TRICK: View the 2D array of ints as a 1D array of raw bytes
        # Must be contiguous in memory for the view to work
        contiguous_voxels = np.ascontiguousarray(voxel_indices)
        void_dt = np.dtype((np.void, contiguous_voxels.dtype.itemsize * contiguous_voxels.shape[1]))
        void_view = contiguous_voxels.view(void_dt).ravel()
        
        # 3. Run 1D unique (Blisteringly fast)
        _, unique_local_indices = np.unique(void_view, return_index=True)

        new_elite_indices = bubble.elite_indices[unique_local_indices]

        return LocalBubble(center_x=bubble.center_x, radius=bubble.radius, elite_indices=new_elite_indices, infeasible_indices=bubble.infeasible_indices, dominated_indices=bubble.dominated_indices)

class RandomVoxelSparsifier(ISparsifier):
    """
    Voxel sparsifier with:
    1. Uniform random point selection per voxel (via random permutation).
    2. Random grid jitter (shifts the voxel origin so boundaries aren't rigid).
    """
    def sparsify(self, bubble: LocalBubble, archive: InternalArchive, f_L: np.ndarray, f_D: np.ndarray) -> LocalBubble:
        X_E = bubble.get_X_E(archive)
        n_points = len(X_E)
        target_k = 11 * (self.f_count - 1)
        if n_points <= target_k:
            return bubble

        intrinsic_dim = max(1.0, float(self.f_count - 1))
        diameter = 2.0 * bubble.radius
        voxel_size = max(diameter / (target_k ** (1.0 / intrinsic_dim)), 1e-6)

        # 1. Add random grid jitter (prevents getting stuck on fixed voxel borders)
        jitter = np.random.uniform(0.0, voxel_size, size=(1, self.x_count))
        voxel_indices = np.round((X_E + jitter) / voxel_size).astype(np.int32)
        
        # 2. View as 1D contiguous byte array for fast hashing
        contiguous_voxels = np.ascontiguousarray(voxel_indices)
        void_dt = np.dtype((np.void, contiguous_voxels.dtype.itemsize * contiguous_voxels.shape[1]))
        void_view = contiguous_voxels.view(void_dt).ravel()
        
        # 3. THE SHUFFLE TRICK: Permute before calling np.unique
        perm = np.random.permutation(n_points)
        shuffled_void = void_view[perm]
        
        # unique picks the FIRST occurrence in the SHUFFLED array -> Uniformly random selection!
        _, unique_shuffled_indices = np.unique(shuffled_void, return_index=True)
        
        # Map back to original indices
        selected_local_indices = perm[unique_shuffled_indices]
        new_elite_indices = bubble.elite_indices[selected_local_indices]

        return LocalBubble(center_x=bubble.center_x, radius=bubble.radius, elite_indices=new_elite_indices, infeasible_indices=bubble.infeasible_indices, dominated_indices=bubble.dominated_indices)


class NoOpSparsifier(ISparsifier):
    """Pass-through for testing without sparsification."""
    def sparsify(self, bubble: LocalBubble, archive: InternalArchive, f_L: np.ndarray, f_D: np.ndarray) -> LocalBubble:
        return bubble

