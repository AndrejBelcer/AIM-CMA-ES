import numpy as np
from abc import ABC, abstractmethod

class CandidateIndicator(ABC):
    """
    Computes Indicator contributions of candidate points relative to a Fixed (Frozen) Front.
    Scores are always returned negative, because minimization of score is assumed.
    The provided front must include only non-dominated points.
    """
    def __init__(self, frozen_front: np.ndarray, compute_frozen_score: bool = True):
        """
        Args:
            frozen_front (np.ndarray): Shape (K, M). The empirical Pareto front.
            ref_point (np.ndarray): Shape (M,). The reference point for HV.
        """
        self.frozen_front: np.ndarray = self.initialize_frozen_front(frozen_front)
        self.dim = self.frozen_front.shape[1]
        
        self.frozen_score: float = 0.0
        if (compute_frozen_score):
            self.frozen_score = self.calculate_front_score(self.frozen_front)
        
        
    @abstractmethod
    def calculate_candidate_score(self, candidates: np.ndarray) -> np.ndarray:
        """
        Returns the scores for each candidate.
        In general, computes the difference to the score: S(F) - S(F + point)
        Lower (negative) is better.
        """
        Exception("Must be implemented by derived class")

    @abstractmethod
    def calculate_archive_scores(self, archive_points: np.ndarray) -> np.ndarray:
        """
        Returns the scores for each candidate from the archive.
        All should be expected to be dominated, so some checks can be quickened compared to calculate_candidate_score.
        This supports warm starts by finding the best points in the archive.
        """
        Exception("Must be implemented by derived class")

    @abstractmethod
    def calculate_front_score(self, front: np.ndarray) -> float:
        """
        Returns the value score of the front.
        The higher the number, the better.
        """
        Exception("Must be implemented by derived class")

    def initialize_frozen_front(self, frozen_front: np.ndarray) -> np.ndarray:
        """
        Can be overridden by a derived class.
        For example if points need to be first sorted, or anything else.
        """
        return frozen_front
