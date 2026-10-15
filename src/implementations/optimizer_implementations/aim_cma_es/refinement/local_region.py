import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Type

from src.implementations.optimizer_implementations.aim_cma_es.archive.internal_archive import InternalArchive

@dataclass
class LocalRegion:
    """
    Holds the context of a local region. 
    Stores indices to avoid copying memory until explicitly requested.
    """
    center_x: np.ndarray
    radius: float
    empirical_set_indices: np.ndarray  # Always populated by gatherer

    # Lazy: None means "not yet fetched". Set to empty array once fetched.
    infeasible_indices: Optional[np.ndarray] = None
    dominated_indices: Optional[np.ndarray] = None
    dominated_F: Optional[np.ndarray] = None

    def get_empirical_set_X_normalized(self, archive: InternalArchive) -> np.ndarray:
        return archive.get_norm_x_by_indices(self.empirical_set_indices)

    def get_infeasible_X_normalized(self, archive: InternalArchive) -> np.ndarray:
        self._ensure_other_region_points(archive)
        if len(self.infeasible_indices) == 0:
            return np.empty((0, archive.x_count))
        return archive.get_norm_x_by_indices(self.infeasible_indices)

    def get_dominated_X_normalized(self, archive: InternalArchive) -> np.ndarray:
        self._ensure_other_region_points(archive)
        if len(self.dominated_indices) == 0:
            return np.empty((0, archive.x_count))
        return archive.get_norm_x_by_indices(self.dominated_indices)

    def _ensure_other_region_points(self, archive: InternalArchive) -> None:
        """Fetch other region points once, when infeasible or dominated solutions are requested."""
        if self.infeasible_indices is not None:
            return  # Already cached

        # Now returns a numpy array directly!
        ball_indices = archive.query_ball_point(self.center_x, r=self.radius)
        
        if ball_indices.size == 0:
            self.infeasible_indices = np.empty(0, dtype=int)
            self.dominated_indices = np.empty(0, dtype=int)
            return

        is_feas = archive.get_feasible_mask()[ball_indices]
        is_empirical_set_point = np.isin(ball_indices, self.empirical_set_indices)

        self.infeasible_indices = ball_indices[~is_feas]
        self.dominated_indices = ball_indices[is_feas & ~is_empirical_set_point]
        
    def get_dominated_F_raw(self, archive: InternalArchive) -> np.ndarray:
        self._ensure_other_region_points(archive)
        if self.dominated_F is None:
            if len(self.dominated_indices) == 0:
                self.dominated_F = np.empty((0, archive.f_count))
            else:
                self.dominated_F = archive.get_f_by_indices(self.dominated_indices)
        return self.dominated_F
