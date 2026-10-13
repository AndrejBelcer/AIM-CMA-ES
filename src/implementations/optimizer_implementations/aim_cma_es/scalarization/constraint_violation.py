import numpy as np
from scipy.special import logsumexp

from src.implementations.optimizer_implementations.aim_cma_es.enums import CVType

class ConstraintViolation:
    """
    Computes scalar constraint violation metrics.
    Supports multiple aggregation strategies.
    """
    def __init__(self, cv_type: CVType = CVType.LINEAR):
        self.cv_type = cv_type
        self.alpha = 3.5

    def compute(self, g_batch: np.ndarray, g_divisor: np.ndarray) -> np.ndarray:
        """
        Args:
            g_batch: (N, M) Raw constraint values.
            g_max: (M,) Maximum observed violations (for normalization).
        Returns:
            (N,) Scalar violation scores (0.0 if feasible).
        """
        if g_batch.shape[1] == 0:
            return np.zeros(len(g_batch))

        # 1. Normalize and Clamp (Only positive violations matter)
        # z_n = max(0, g) / max(g_max, 1e-9)
        g_pos = np.maximum(0.0, g_batch)
        z_n = g_pos / g_divisor

        # 2. Aggregate
        if self.cv_type == CVType.LINEAR:
            # Sum of normalized violations
            return np.sum(z_n, axis=1)
            
        elif self.cv_type == CVType.QUADRATIC:
            # Sum of squared normalized violations
            return np.sum(z_n**2, axis=1)
            
        elif self.cv_type == CVType.CHEBYSHEV:
            # Max normalized violation
            return np.max(z_n, axis=1)
            
        elif self.cv_type == CVType.LOGSUMEXP:
            # Smooth Max: (1/alpha) * ln(sum(exp(alpha * z_n)))
            return logsumexp(self.alpha * z_n, axis=1) / self.alpha
            
        else:
            # Fallback
            return np.sum(z_n, axis=1)
