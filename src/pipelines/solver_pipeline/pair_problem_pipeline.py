import os
import copy
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.blackbox import Blackbox
from src.core.optimizer import Optimizer
from src.core.problem_solving_pair import ProblemSolvingPair

def _worker_solve(blackbox: Blackbox, optimizer: Optimizer, max_evals: int, debug_mode: bool):
    """
    Isolated worker function to run a single ProblemSolvingPair.
    """
    pair = ProblemSolvingPair(blackbox, optimizer, allowed_evals=max_evals, debug_run=debug_mode)
    pair.run()
    return optimizer.optimizer_name

def Run_Solvers_Parallel(
    problem_blackboxes: List[Blackbox], 
    optimizers: List[Optimizer], 
    max_evals: int = 15000, 
    repeats: int = 5, 
    debug: bool = False,
    max_workers: Optional[int] = None
):
    """
    Runs multiple optimizers on multiple problems in parallel.
    Deepcopies the instantiated classes to ensure clean states for every run.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    total_tasks = len(problem_blackboxes) * len(optimizers) * repeats
    print(f"Submitting {total_tasks} tasks to ProcessPoolExecutor (max_workers={max_workers})...")

    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bb in problem_blackboxes:
            for opt in optimizers:
                for r in range(repeats):
                    # Deepcopy to ensure independent states and counters
                    bb_copy = bb.clone()
                    opt_copy = copy.deepcopy(opt)
                    
                    fut = ex.submit(_worker_solve, bb_copy, opt_copy, max_evals, debug)
                    futures[fut] = (bb.problem_name, opt.optimizer_name, r)

        # Collect results as they complete
        completed = 0
        for fut in as_completed(futures):
            p_name, s_name, repeat_idx = futures[fut]
            try:
                fut.result()
                completed += 1
                print(f"[{completed}/{total_tasks}] {s_name} on {p_name} finished (repeat {repeat_idx}).")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{total_tasks}] [ERROR] {s_name} on {p_name} failed (repeat {repeat_idx}): {e}")

    print("\nAll solver tasks finished. Saving problem metadata (True Fronts/Sets)...")
    for bb in problem_blackboxes:
        try:
            bb.save_problem_metadata()
        except Exception as e:
            print(f"Could not save metadata for {bb.problem_name}: {e}")
            
    print("Pipeline complete.")
