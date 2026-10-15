"""Optimized directed gap hypervolume scalarization."""

import numpy as np
from numba import njit

from src.utilities.indicators.HVO.candidate_hvo_common import OptimizedHVBase, _hv_improvement_2d, _points_array, _softplus_reward


@njit(cache=True, nogil=True)
def _gap_distances(points, gap, offset):
    scores = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        squared = 0.0
        for j in range(len(gap)):
            difference = points[i, j] - gap[j]
            squared += difference * difference
        scores[i] = np.sqrt(squared) - offset
    return scores


@njit(cache=True, nogil=True)
def _gap_scores_2d(points, xs, ys, neg_ys, edges, prefix, ref, gap, offset):
    scores = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        a, b = points[i, 0], points[i, 1]
        start = np.searchsorted(xs, a, side='right')
        if start > 0 and ys[start - 1] <= b:
            dx, dy = a - gap[0], b - gap[1]
            scores[i] = max(0.0, np.sqrt(dx * dx + dy * dy) - offset)
        elif a >= ref[0] or b >= ref[1]:
            scores[i] = _softplus_reward(points[i], ref)
        else:
            scores[i] = -_hv_improvement_2d(a, b, start, xs, ys, neg_ys, edges, prefix, ref)
    return scores


class Candidate_GapHV_Optimized(OptimizedHVBase):
    """Directed gap penalty, exact HV improvement, and legacy fallback.

    Dominated candidates receive L2(candidate - gap) minus gap_offset, where
    gap_offset is the gap's distance to the frozen dominated region. Archive
    warm starts use unshifted L2 distance, including infeasible candidates.
    """

    def __init__(self, frozen_front, ref_point=None, gap_point=None,
                 compute_frozen_score=True):
        super().__init__(frozen_front, ref_point, compute_frozen_score)

        gap = np.asarray(gap_point, dtype=np.float64)
        if gap.shape != (self.dim,) or not np.all(np.isfinite(gap)):
            raise ValueError('gap_point must be a finite M-vector.')
        if np.any(np.all(self.frozen_front <= gap, axis=1)):
            raise ValueError('gap_point must be undominated by the frozen front.')

        self.gap_point = np.array(gap, copy=True)
        self.gap_point.setflags(write=False)
        self.gap_offset = float(np.min(np.linalg.norm(np.maximum(self.frozen_front - gap, 0.0), axis=1)))
        self._penalty_impl = self._gap_penalty
        self._prepare_working_front(self.frozen_front)

    def _gap_penalty(self, points):
        return np.maximum(0.0, _gap_distances(points, self.gap_point, self.gap_offset))

    def _calculate_candidates_2d(self, points):
        return _gap_scores_2d(points, self._xs, self._ys, self._neg_ys, self._edges, self._prefix, self.ref_point, self.gap_point, self.gap_offset)

    def calculate_archive_scores(self, archive_points, contains_infeasible=False):
        """Return unshifted L2 to the target; feasibility does not alter gap ranking."""
        points = _points_array(archive_points, self.dim, name='archive_points')
        return _gap_distances(points, self.gap_point, 0.0)
