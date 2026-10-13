"""Shared machinery for optimized hypervolume scalarizations."""

import numpy as np
from moocore import filter_dominated, hypervolume
from numba import njit

from src.utilities.indicators.candidate_indicator import CandidateIndicator


@njit(cache=True, nogil=True)
def _softplus_reward(point, ref):
    value = 0.0
    for j in range(len(ref)):
        value += ref[j] - point[j]
    if value > 0.0:
        return -(value + np.log1p(np.exp(-value)))
    return -np.log1p(np.exp(value))


@njit(cache=True, nogil=True)
def _is_dominated_nd(point, front):
    """Weak dominance, including equality with a stored point."""
    for k in range(len(front)):
        dominated = True
        for j in range(front.shape[1]):
            if front[k, j] > point[j]:
                dominated = False
                break
        if dominated:
            return True
    return False


@njit(cache=True, nogil=True)
def _classify_nd(points, front, ref):
    """Return 1 for dominated, 0 for inside-HV, and -1 for fallback."""
    result = np.empty(len(points), dtype=np.int8)
    for i in range(len(points)):
        if _is_dominated_nd(points[i], front):
            result[i] = 1
        else:
            result[i] = 0
            for j in range(len(ref)):
                if points[i, j] >= ref[j]:
                    result[i] = -1
                    break
    return result


@njit(cache=True, nogil=True)
def _hv_improvement_2d(a, b, start, xs, ys, neg_ys, edges, prefix, ref):
    """Exact staircase integral with O(log K) lookup and O(1) area retrieval."""
    stop = np.searchsorted(neg_ys, -b, side='left')
    height = ref[1] if start == 0 else ys[start - 1]
    area = (edges[start] - a) * (height - b)
    if stop > start:
        strip_area = prefix[stop] - prefix[start]
        rectangle = b * (edges[stop] - edges[start])
        extra = strip_area - rectangle
        error_scale = abs(prefix[stop]) + abs(prefix[start]) + abs(rectangle)
        if extra <= 1e-12 * error_scale:
            extra = 0.0
            for k in range(start, stop):
                extra += (edges[k + 1] - edges[k]) * (ys[k] - b)
        area += extra
    return area


def _points_array(points, dim=None, *, name='points', allow_empty=True):
    result = np.asarray(points, dtype=np.float64)
    if result.ndim == 1 and result.size == 0 and dim is not None:
        result = result.reshape(0, dim)
    if result.ndim != 2 or (dim is not None and result.shape[1] != dim):
        raise ValueError(f'{name} must have shape (N, {dim if dim else "M"}).')
    if not allow_empty and len(result) == 0:
        raise ValueError(f'{name} must contain at least one point.')
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain only finite objective values.')
    return np.ascontiguousarray(result)


def _clean_front(front):
    """Remove redundant rows and establish increasing-first-objective 2D order."""
    result = filter_dominated(front)
    if front.shape[1] == 2:
        result = result[np.argsort(result[:, 0], kind='stable')]
    return np.array(result, dtype=np.float64, order='C', copy=True)


class OptimizedHVBase(CandidateIndicator):
    """Shared frozen-front hypervolume implementation."""

    def __init__(self, frozen_front, ref_point=None, compute_frozen_score=True):
        front = _points_array(frozen_front, name='frozen_front', allow_empty=False)
        dim = front.shape[1]
        if dim < 2:
            raise ValueError('At least two objectives are required.')

        ref = np.asarray(10.0 if ref_point is None else ref_point, dtype=np.float64)
        if ref.ndim == 0:
            ref = np.full(dim, float(ref))
        if ref.shape != (dim,) or not np.all(np.isfinite(ref)):
            raise ValueError('ref_point must be a finite scalar or an M-vector.')
        if np.any(front >= ref):
            raise ValueError('The frozen front must be strictly inside ref_point.')

        self.ref_point = np.array(ref, copy=True)
        self.ref_point.setflags(write=False)
        super().__init__(front, compute_frozen_score=compute_frozen_score)

    def initialize_frozen_front(self, frozen_front):
        return _clean_front(frozen_front)

    def _prepare_working_front(self, front):
        self.frozen_front = np.ascontiguousarray(front)
        self.frozen_front.setflags(write=False)

        if self.dim == 2:
            self._xs = np.ascontiguousarray(front[:, 0])
            self._ys = np.ascontiguousarray(front[:, 1])
            self._neg_ys = -self._ys
            self._edges = np.concatenate((self._xs, self.ref_point[:1]))
            self._prefix = np.concatenate(([0.0], np.cumsum(np.diff(self._edges) * self._ys)))
            self._candidate_impl = self._calculate_candidates_2d
        else:
            self._working_hv = float(hypervolume(front, self.ref_point))
            self._candidate_impl = self._calculate_candidates_nd

    def calculate_candidate_score(self, candidates):
        points = _points_array(candidates, self.dim, name='candidates')
        return self._candidate_impl(points)

    def calculate_front_score(self, front):
        """Score the supplied complete front without thinning or cached geometry."""
        points = _points_array(front, len(self.ref_point), name='front')
        if len(points) == 0:
            return 0.0

        score = float(hypervolume(points, self.ref_point))
        outside = ~np.all(points < self.ref_point, axis=1)
        if np.any(outside):
            score += float(np.sum(np.logaddexp(0.0, np.sum(self.ref_point - points[outside], axis=1))))
        return score

    def _calculate_candidates_nd(self, points):
        scores = np.empty(len(points), dtype=np.float64)
        if len(points) == 0:
            return scores

        status = _classify_nd(points, self.frozen_front, self.ref_point)
        dominated = status == 1
        if np.any(dominated):
            scores[dominated] = self._penalty_impl(points[dominated])

        outside = status == -1
        if np.any(outside):
            scores[outside] = -np.logaddexp(0.0, np.sum(self.ref_point - points[outside], axis=1))

        indices = np.flatnonzero(status == 0)
        if len(indices):
            augmented = np.empty((len(self.frozen_front) + 1, self.dim))
            augmented[:-1] = self.frozen_front
            for i in indices:
                augmented[-1] = points[i]
                improvement = (float(hypervolume(augmented, self.ref_point)) - self._working_hv)
                if improvement <= 1e-12 * abs(self._working_hv):
                    overlap_front = np.maximum(self.frozen_front, points[i])
                    improvement = float(np.prod(self.ref_point - points[i])) - float(hypervolume(overlap_front, self.ref_point))
                scores[i] = -max(0.0, improvement)
        return scores
