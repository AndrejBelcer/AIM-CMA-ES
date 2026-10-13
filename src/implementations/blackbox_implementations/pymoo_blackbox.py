import numpy as np
import os
from typing import List, Tuple, Optional, Callable, Dict
from pymoo.core.problem import Problem

from src.core.blackbox import Blackbox



class PymooBlackbox(Blackbox):
    def __init__(self, problem: Problem, name_override: Optional[str] = None):
        xl, xu = problem.bounds()

        # Pymoo usually combines constraints. We assume inequality constraints.
        g_count = problem.n_ieq_constr if hasattr(problem, 'n_ieq_constr') else 0
        
        # Determine Name
        if name_override:
            p_name = name_override
        elif hasattr(problem, 'name'):
            p_name = problem.name() if callable(problem.name) else problem.name
            p_name = f"{p_name}_pymoo"
        else:
            p_name = "Unknown_pymoo"

        super().__init__(problem_name=p_name, x_count=problem.n_var, f_count=problem.n_obj, g_count=g_count, lower_bounds=xl, upper_bounds=xu)
        self.problem = problem

    def _evaluate_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = self.problem.evaluate(x, return_values_of=["F", "G"])
        f = out[0]
        g = out[1] if len(out) > 1 else None
        return f, g

    def get_true_front(self) -> Optional[np.ndarray]:
        try:
            pf = self.problem.pareto_front()
            return pf
        except Exception:
            return None

    def get_true_set(self) -> Optional[np.ndarray]:
        try:
            ps = self.problem.pareto_set()
            return ps
        except Exception:
            return None
        
    def get_nadir_point(self) -> Optional[np.ndarray]:
        return self.problem.nadir_point()
