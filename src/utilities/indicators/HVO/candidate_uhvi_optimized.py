"""Optimized UHVI with a frozen, optionally thinned empirical front.

Distance is measured to the unbounded empirical non-domination boundary, not to
stored points and not to a boundary clipped by the hypervolume reference.
"""

import numpy as np
from moocore import is_nondominated
from numba import njit
from scipy.spatial import cKDTree

from src.utilities.indicators.HVO.candidate_hvo_common import OptimizedHVBase, _clean_front, _hv_improvement_2d, _is_dominated_nd, _points_array, _softplus_reward


@njit(cache=True, nogil=True)
def _distance_2d(a, b, xs, ys):
    """Distance to lower-left boxes bounded by the staircase corners.

    The two outer boxes have one unbounded coordinate. For dominated points,
    distance to this closed undominated region equals distance to its boundary.
    """
    dx = max(a - xs[0], 0.0)
    dy = max(b - ys[-1], 0.0)
    best = min(dx * dx, dy * dy)
    for k in range(1, len(xs)):
        dx = max(a - xs[k], 0.0)
        if dx * dx >= best:
            continue
        dy = max(b - ys[k - 1], 0.0)
        squared = dx * dx + dy * dy
        if squared < best:
            best = squared
    return np.sqrt(best)


@njit(cache=True, nogil=True)
def _uhvi_scores_2d(points, xs, ys, neg_ys, edges, prefix, ref):
    scores = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        a, b = points[i, 0], points[i, 1]
        start = np.searchsorted(xs, a, side='right')
        if start > 0 and ys[start - 1] <= b:
            scores[i] = _distance_2d(a, b, xs, ys)
        elif a >= ref[0] or b >= ref[1]:
            scores[i] = _softplus_reward(points[i], ref)
        else:
            scores[i] = -_hv_improvement_2d(a, b, start, xs, ys, neg_ys, edges, prefix, ref)
    return scores


@njit(cache=True, nogil=True)
def _archive_2d(points, xs, ys):
    scores = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        scores[i] = _distance_2d(points[i, 0], points[i, 1], xs, ys)
    return scores


@njit(cache=True, nogil=True)
def _archive_checked_2d(points, xs, ys):
    scores = np.zeros(len(points), dtype=np.float64)
    for i in range(len(points)):
        a, b = points[i, 0], points[i, 1]
        start = np.searchsorted(xs, a, side='right')
        if start > 0 and ys[start - 1] <= b:
            scores[i] = _distance_2d(a, b, xs, ys)
    return scores


@njit(cache=True, nogil=True)
def _distances_boxes(points, upper_bounds):
    scores = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        best = np.inf
        for k in range(len(upper_bounds)):
            squared = 0.0
            for j in range(points.shape[1]):
                difference = max(points[i, j] - upper_bounds[k, j], 0.0)
                squared += difference * difference
                if squared >= best:
                    break
            if squared < best:
                best = squared
                if best == 0.0:
                    break
        scores[i] = np.sqrt(best)
    return scores


def _build_upper_bounds(front, limit):
    """Decompose the closed undominated region into boxes (-inf, upper].

    Inserting p splits each box whose upper corner is strictly worse than p.
    Contained boxes are redundant. Infinite coordinates preserve extreme rays.
    Return None if decomposition exceeds its budget; evaluation then uses an
    exact branch-and-bound distance, selected once at initialization.
    """
    if limit == 0:
        return None
    dim = front.shape[1]
    bounds = np.full((1, dim), np.inf)
    coordinates = [np.unique(front[:, j]) for j in range(dim)]
    for point in front:
        split = np.all(point < bounds, axis=1)
        if not np.any(split):
            continue
        count = int(np.sum(split))
        if len(bounds) + count * (dim - 1) > limit * dim:
            return None
        children = np.repeat(bounds[split], dim, axis=0)
        for j in range(dim):
            children[j::dim, j] = point[j]
        bounds = np.vstack((bounds[~split], children))
        # moocore's dominance routines expect finite coordinates. Rank encoding
        # preserves their ordering and represents +inf by one extra finite rank.
        ranks = np.empty_like(bounds)
        for j in range(dim):
            ranks[:, j] = np.searchsorted(coordinates[j], bounds[:, j])
        bounds = bounds[is_nondominated(-ranks)]
        if len(bounds) > limit:
            return None
    return np.ascontiguousarray(bounds)


@njit(cache=True, nogil=True)
def _distance_search(point, projected, front, best):
    """Exact depth-first search without recursion or a complete decomposition.

    The explicit stack avoids exhausting the native call stack on large fronts.
    Each path lowers coordinates through finitely many front thresholds.
    """
    pending = [projected]
    while len(pending):
        current = pending.pop()
        squared = 0.0
        for j in range(len(point)):
            squared += (point[j] - current[j]) ** 2
        if squared >= best:
            continue
        violation = -1
        for k in range(len(front)):
            strictly_worse = True
            for j in range(len(point)):
                if current[j] <= front[k, j]:
                    strictly_worse = False
                    break
            if strictly_worse:
                violation = k
                break
        if violation < 0:
            best = squared
            continue
        # Escaping this orthant requires one coordinate on its boundary.
        costs = np.empty(len(point))
        for j in range(len(point)):
            costs[j] = squared - (point[j] - current[j]) ** 2 + (point[j] - front[violation, j]) ** 2
        # LIFO ordering explores the smallest lower bound first.
        for j in np.argsort(costs)[::-1]:
            if costs[j] < best:
                child = current.copy()
                child[j] = front[violation, j]
                pending.append(child)
    return best


@njit(cache=True, nogil=True)
def _distances_search(points, front, ideal):
    scores = np.empty(len(points))
    for i in range(len(points)):
        # Moving one coordinate to the ideal always reaches a boundary.
        best = np.inf
        for j in range(points.shape[1]):
            delta = max(points[i, j] - ideal[j], 0.0)
            best = min(best, delta * delta)
        scores[i] = np.sqrt(_distance_search(points[i], points[i].copy(), front, best))
    return scores


@njit(cache=True, nogil=True)
def _dominated_mask_nd(points, front):
    mask = np.empty(len(points), dtype=np.bool_)
    for i in range(len(points)):
        mask[i] = _is_dominated_nd(points[i], front)
    return mask


@njit(cache=True, nogil=True)
def _thin_2d_pass(front, target):
    n = len(front)
    distances = np.empty(n - 1)
    for i in range(n - 1):
        dx = front[i + 1, 0] - front[i, 0]
        dy = front[i + 1, 1] - front[i, 1]
        distances[i] = dx * dx + dy * dy
    used = np.zeros(n, dtype=np.bool_)
    merge_left = np.zeros(n, dtype=np.bool_)
    used[0] = used[-1] = True
    merged = 0
    for i in np.argsort(distances):
        if not used[i] and not used[i + 1]:
            used[i] = used[i + 1] = True
            merge_left[i] = True
            merged += 1
            if merged == n - target:
                break
    result = np.empty((n - merged, 2))
    i, output = 0, 0
    while i < n:
        if merge_left[i]:
            result[output, 0] = front[i, 0]
            result[output, 1] = front[i + 1, 1]
            i += 2
        else:
            result[output] = front[i]
            i += 1
        output += 1
    return result


@njit(cache=True, nogil=True)
def _merge_nd_pairs(front, left, right, protected, target):
    used = protected.copy()
    removed = np.zeros(len(front), dtype=np.bool_)
    result = front.copy()
    merged = 0
    for k in range(len(left)):
        i, j = left[k], right[k]
        if not used[i] and not used[j]:
            used[i] = used[j] = True
            removed[j] = True
            for m in range(front.shape[1]):
                result[i, m] = min(front[i, m], front[j, m])
            merged += 1
            if merged == len(front) - target:
                break
    return result[~removed]


def _thin_nd_pass(front, target):
    protected = np.zeros(len(front), dtype=bool)
    protected[np.argmin(front, axis=0)] = True
    eligible = np.flatnonzero(~protected)
    if len(eligible) < 2:
        return front
    # Nearest-neighbor pairs avoid the legacy K-by-K distance matrix. Rebuild
    # each pass. Minimum merges are unchanged, but pair choices can differ.
    distances, neighbors = cKDTree(front[eligible]).query(front[eligible], k=2)
    order = np.argsort(distances[:, 1], kind='stable')
    result = _merge_nd_pairs(front, eligible[order], eligible[neighbors[order, 1]], protected, target)
    return _clean_front(result)


class Candidate_UHVI_Optimized(OptimizedHVBase):
    """UHVI relative to an optionally thinned, frozen empirical front.

    max_size_per_dim=1024 caps the working front at 2048 points in 2D; None
    disables thinning. Coordinate-wise-minimum merges are optimistic and can
    suppress improvements relative to the original front. Dominance, candidate
    HV and distance ALL use the same approximation.

    frozen_score, when requested, reports the original full front, which is not
    retained. calculate_front_score always scores its supplied input.

    max_distance_boxes bounds ND distance preprocessing. If exceeded, an exact
    branch-and-bound distance is selected instead. Higher-dimensional distance
    can remain expensive; this changes time/memory, not the distance definition.
    """

    def __init__(self, frozen_front, ref_point=None, compute_frozen_score=True,
                 max_size_per_dim=1024, max_distance_boxes=20000):
        if max_size_per_dim is not None and (not isinstance(max_size_per_dim, (int, np.integer)) or max_size_per_dim < 2):
            raise ValueError('max_size_per_dim must be an integer >= 2, or None.')
        if not isinstance(max_distance_boxes, (int, np.integer)) or max_distance_boxes < 0:
            raise ValueError('max_distance_boxes must be a nonnegative integer.')
        super().__init__(frozen_front, ref_point, compute_frozen_score)
        self.max_front_size = (None if max_size_per_dim is None else max_size_per_dim * self.dim)
        front = self.frozen_front
        thin_pass = _thin_2d_pass if self.dim == 2 else _thin_nd_pass
        if self.max_front_size is not None:
            while len(front) > self.max_front_size:
                reduced = thin_pass(front, self.max_front_size)
                if len(reduced) >= len(front):
                    raise RuntimeError('Front thinning made no progress.')
                front = reduced
        self._prepare_working_front(front)
        if self.dim == 2:
            self._archive_impl = self._archive_distances_2d
            self._archive_checked_impl = self._archive_checked_distances_2d
        else:
            self._upper_bounds = _build_upper_bounds(front, max_distance_boxes)
            if self._upper_bounds is None:
                self._ideal = np.min(front, axis=0)
                self._penalty_impl = self._distance_search_nd
            else:
                self._penalty_impl = self._distance_boxes_nd
            self._archive_impl = self._penalty_impl
            self._archive_checked_impl = self._archive_checked_nd

    def _calculate_candidates_2d(self, points):
        return _uhvi_scores_2d(points, self._xs, self._ys, self._neg_ys, self._edges, self._prefix, self.ref_point)

    def _distance_boxes_nd(self, points):
        return _distances_boxes(points, self._upper_bounds)

    def _distance_search_nd(self, points):
        return _distances_search(points, self.frozen_front, self._ideal)

    def _archive_distances_2d(self, points):
        return _archive_2d(points, self._xs, self._ys)

    def _archive_checked_distances_2d(self, points):
        return _archive_checked_2d(points, self._xs, self._ys)

    def _archive_checked_nd(self, points):
        scores = np.zeros(len(points))
        mask = _dominated_mask_nd(points, self.frozen_front)
        if np.any(mask):
            scores[mask] = self._penalty_impl(points[mask])
        return scores

    def calculate_archive_scores(self, archive_points, containts_infeasible=False):
        """Warm-start distance only; undominated infeasible points get zero.

        False promises dominance/equality, allowing the dominance check to be
        skipped. The parameter spelling follows the project's updated API.
        """
        points = _points_array(archive_points, self.dim, name='archive_points')
        implementation = self._archive_checked_impl if containts_infeasible else self._archive_impl
        return implementation(points)
