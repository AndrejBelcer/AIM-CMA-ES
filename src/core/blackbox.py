import os
import numpy as np
import copy
from typing import List, Tuple, Optional, Callable, Dict
from abc import ABC, abstractmethod

from src.core.paths import SOLVER_DEBUG_DIR, SOLVER_RESULTS_DIR, COBI_PROBLEMS_DIR

class Blackbox(ABC):
    """
    Base class for problems. Handles bounds and counting evaluations.
    Defines the dimension of the search space (x_count), number of objectives (f_count) and number of constraints (g_count).
    Also defines the "hard bounds" where sampling is possible (lower_bounds and upper_bounds).
    And defines the "region of interest", which is a region where (with a high probability) the global optimal solutions lie.
    """
    def __init__(self, problem_name: str, x_count: int, f_count: int, g_count: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, region_of_interest_lower: Optional[np.ndarray] = None, region_of_interest_upper: Optional[np.ndarray] = None):
        # Problem metadata
        self.problem_name = problem_name

        # Problem dimensionality
        self.x_count = x_count
        self.f_count = f_count
        self.g_count = g_count

        # Problem bounds
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Region of interest assignment and checks
        self.region_of_interest_lower: np.ndarray = lower_bounds
        self.region_of_interest_upper: np.ndarray = upper_bounds
        
        if((region_of_interest_lower is not None) and (region_of_interest_upper is not None)):
            self.region_of_interest_lower: np.ndarray = region_of_interest_lower
            self.region_of_interest_upper: np.ndarray = region_of_interest_upper

        # Check bounds
        if (self.lower_bounds.shape != (self.x_count,)):
            raise ValueError(f"Lower bounds shape {self.lower_bounds.shape} does not match x_count {self.x_count}")
        if (self.upper_bounds.shape != (self.x_count,)):
            raise ValueError(f"Upper bounds shape {self.upper_bounds.shape} does not match x_count {self.x_count}")
        if (self.region_of_interest_lower.shape != (self.x_count,)):
            raise ValueError(f"Lower bounds shape {self.region_of_interest_lower.shape} does not match x_count {self.x_count}")
        if (self.region_of_interest_upper.shape != (self.x_count,)):
            raise ValueError(f"Upper bounds shape {self.region_of_interest_upper.shape} does not match x_count {self.x_count}")

    @property
    def x_bounds(self) -> np.ndarray:
        """
        Returns the bounds as a (D, 2) array where [:, 0] is lower 
        and [:, 1] is upper. Useful for analysis and plotting.
        """
        return np.column_stack([self.lower_bounds, self.upper_bounds])
    
    @property
    def region_of_interest(self) -> np.ndarray:
        """
        Returns the ROI bounds as a (2, D) array where [0] is lower 
        and [1] is upper. This allows for easy unpacking: lb, ub = bounds.
        """
        return np.array([self.region_of_interest_lower, self.region_of_interest_upper])


    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Public interface. Handles input shaping and counting.
        Input: x (N, D) or (D,) or possibly empty.
        Output: f (N, M), g (N, C)
        """
        x = np.asarray(x)

        # If the user passed a 1D vector (single point)
        if x.ndim == 1 and x.size != 0:
            x = x.reshape(1, -1)

        # If empty (size == 0) force shape to (0, D)
        if x.size == 0:
            x = np.empty((0, self.x_count))

        n_points = x.shape[0]

        # If no points, return properly shaped empty outputs right away
        if n_points == 0:
            f_empty = np.empty((0, self.f_count))
            g_empty = np.empty((0, self.g_count)) if self.g_count > 0 else np.empty((0, 0))
            return f_empty, g_empty

        f, g = self._evaluate_batch(x)

        if g is None:
            g = np.empty((n_points, 0))

        return f, g

    @abstractmethod
    def _evaluate_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_true_front(self) -> Optional[np.ndarray]:
        Warning("True front method not overwritten")
        return None

    def get_true_set(self) -> Optional[np.ndarray]:
        Warning("True set method not overwritten")
        return None
    
    def get_nadir_point(self) -> Optional[np.ndarray]:
        Warning("Nadir point method not overwritten")
        return None

    # --- Helper to Save Metadata ---
    def save_problem_metadata(self, debug_run: bool = False):
        """
        Checks if the blackbox has a True Front/Set. 
        If so, saves 'Solutions.npz' to the problem directory.
        """
        target_dir = os.path.join(SOLVER_RESULTS_DIR, self.problem_name) if not debug_run else os.path.join(SOLVER_DEBUG_DIR, self.problem_name)
        os.makedirs(target_dir, exist_ok=True)
        
        tf = self.get_true_front()
        ts = self.get_true_set()
        
        # Only save if we actually have data
        if tf is not None or ts is not None:
            filepath = os.path.join(target_dir, "Solutions.npz")
            
            # Handle cases where one might be None but the other isn't
            save_dict = {}
            if tf is not None: save_dict['TrueFront'] = tf
            if ts is not None: save_dict['TrueSet'] = ts
            
            np.savez_compressed(filepath, **save_dict)
            print(f"Problem metadata (True Front/Set) saved to: {filepath}")
        else:
            print(f"No True Front/Set available for {self.problem_name}. Skipping metadata save.")



    # ==========================================
    # Clone method for safe duplication
    # ==========================================
    def clone(self) -> Blackbox:
        """
        Returns a clean copy of the blackbox. 
        By default, uses deepcopy. Classes with C-pointers (like COCO) should override this.
        """
        return copy.deepcopy(self)
