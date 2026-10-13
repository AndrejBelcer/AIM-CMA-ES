import numpy as np
from typing import Tuple

import cocoex

from src.core.blackbox import Blackbox


class CocoBlackbox(Blackbox):
    """
    Wrapper for COCO (COmparing Continuous Optimizers) benchmark problems.
    Interfaces with the COCO C-backend to evaluate points and track constraints.
    """
    def __init__(
        self, 
        suite_name: str = "bbob-biobj", 
        function_id: int = 1, 
        dimension: int = 2, 
        instance_id: int = 1, 
        region_bound_fallback: float = 100.0, 
        region_of_interest: float = 5.0,
        force_region_fallback: bool = True
    ):
        # 1. Initialize Suite and Problem via COCO API
        self.suite = cocoex.Suite(suite_name, "", "")
        self.suite_name = suite_name
        self.function_id = function_id
        self.dimension = dimension
        self.instance_id = instance_id
        self.region_bound_fallback = region_bound_fallback
        self.region_of_interest_val = region_of_interest
        self.force_region_fallback = force_region_fallback
        
        self.problem = self.suite.get_problem_by_function_dimension_instance(function_id, dimension, instance_id)
        
        if self.problem is None:
            raise ValueError(f"COCO problem not found: {suite_name} f{function_id} d{dimension} i{instance_id}")

        # 2. Extract Metadata and Define Bounds
        p_name = f"Coco_{suite_name}_f{function_id}_d{dimension}_i{instance_id}"
        
        # Check if bounds are defined:
                # Bounds: Try to get from problem, fallback to -5, 5 if missing
        if ((hasattr(self.problem, 'lower_bounds')) and (self.problem.lower_bounds is not None) and (not self.force_region_fallback)):
            lb = self.problem.lower_bounds
            ub = self.problem.upper_bounds
        else:
            lb = np.ones(dimension) * -region_bound_fallback
            ub = np.ones(dimension) * region_bound_fallback
            
        # Region of Interest (ROI) defines where the global optimum is thought to be.
        roi_lb = np.full(dimension, -region_of_interest)
        roi_ub = np.full(dimension, region_of_interest)

        f_count = self.problem.number_of_objectives
        g_count = self.problem.number_of_constraints

        # Initialize the parent Blackbox class with the defined bounds
        super().__init__(problem_name=p_name, x_count=dimension, f_count=f_count, g_count=g_count, lower_bounds=lb, upper_bounds=ub, region_of_interest_lower=roi_lb, region_of_interest_upper=roi_ub)

    def _evaluate_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates a batch of points. COCO's C-backend natively evaluates one point at a time,
        so we iterate through the batch and collect the objective and constraint values.
        """
        n = x.shape[0]
        f_list = []
        g_list = []
        
        # COCO problems are typically single-point evaluators
        for i in range(n):
            row = x[i]
            f_val = self.problem(row)
            f_list.append(f_val)
            
            if self.g_count > 0:
                g_val = self.problem.constraint(row)
                g_list.append(g_val)
        
        f = np.array(f_list)
        g = np.array(g_list) if self.g_count > 0 else None
        return f, g

    # ==========================================
    # Safe Duplication and Multiprocessing
    # Override clone to bypass C-pointers
    # ==========================================
    def clone(self) -> CocoBlackbox:
        """
        Overrides the default deepcopy to safely re-instantiate the COCO C-backend objects.
        Standard deepcopy fails on C-pointers.
        """
        return CocoBlackbox(suite_name=self.suite_name, function_id=self.function_id, dimension=self.dimension, instance_id=self.instance_id, region_bound_fallback=self.region_bound_fallback, region_of_interest=self.region_of_interest_val, force_region_fallback = self.force_region_fallback)

    # ==========================================
    # MULTIPROCESSING FIX: __getstate__ and __setstate__
    # ==========================================
    def __getstate__(self):
        """
        Called when ProcessPoolExecutor serializes this object to send to a worker process.
        We must remove the un-picklable C-pointers before sending.
        """
        state = self.__dict__.copy()
        # Delete the C-pointers from the state being sent across cores
        if 'suite' in state:
            del state['suite']
        if 'problem' in state:
            del state['problem']
        return state

    def __setstate__(self, state):
        """
        Called when the worker process deserializes the object.
        We re-instantiate the C-pointers locally on this specific CPU core.
        """
        self.__dict__.update(state)
        
        # Re-create the C-pointers safely on the new CPU core
        self.suite = cocoex.Suite(self.suite_name, "", "")
        self.problem = self.suite.get_problem_by_function_dimension_instance(self.function_id, self.dimension, self.instance_id)
