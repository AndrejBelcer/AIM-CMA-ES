import os
import time
import copy
import shutil
import uuid
import math
import csv
import tempfile
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from enum import Enum 
from typing import List, Optional
import gc

import cocoex
import numpy as np

from src.core.optimizer import Optimizer
from src.implementations.blackbox_implementations.coco_blackbox import CocoBlackbox
from src.core.paths import COCO_RESULTS_DIR

# ==========================================
# Enum for COCO Observer Logging Level
# ==========================================
class CocoLogLevel(Enum):
    NONE = "none"    
    FINAL = "final"  
    ALL = "all"      

# ==========================================
# Data Structures for Payloads and Tasks
# ==========================================
class Payload:
    """Declarative request for a benchmark run."""
    def __init__(self, optimizer: Optimizer, suite_name: str = "bbob-biobj", dimensions: List[int] = [2, 3, 5, 10, 20, 40], functions: List[int] = list(range(1, 56)), instances: List[int] = list(range(1, 11)), budget_multiplier: float = 1e5, log_level: CocoLogLevel = CocoLogLevel.FINAL, region_bound_fallback: float = 100.0, region_of_interest: float = 5.0
) -> None:
        self.optimizer = optimizer
        self.suite_name: str = suite_name
        self.dimensions: List[int] = dimensions
        self.functions: List[int] = functions
        self.instances: List[int] = instances
        self.budget_multiplier: float = budget_multiplier
        self.log_level: CocoLogLevel = log_level
        self.region_bound_fallback: float = region_bound_fallback
        self.region_of_interest: float = region_of_interest
    
    
def GeneratePayloads(optimizers: List[Optimizer], suite_name: str = "bbob-biobj", dimensions: List[int] = [2, 3, 5, 10, 20, 40], functions: List[int] = list(range(1, 56)), instances: List[int] = list(range(1, 11)), budget_multiplier: float = 1e5, log_level: CocoLogLevel = CocoLogLevel.FINAL, region_bound_fallback: float = 100.0, region_of_interest: float = 5.0) -> List[Payload]:
    payloads: List[Payload] = []
    for optimizer in optimizers:
        payloads.append(Payload(optimizer, suite_name, dimensions, functions, instances, budget_multiplier, log_level, region_bound_fallback, region_of_interest))
    return payloads
        
    

class LiveCocoBlackboxAdapter(CocoBlackbox):
    """
    Adapter that wraps an already instantiated COCO problem into our Blackbox interface.
    """
    def __init__(self, live_problem, region_bound_fallback: float, region_of_interest: float):
        self.problem = live_problem
        
        lb = np.full(live_problem.dimension, -region_bound_fallback)
        ub = np.full(live_problem.dimension, region_bound_fallback)
        roi_lb = np.full(live_problem.dimension, -region_of_interest)
        roi_ub = np.full(live_problem.dimension, region_of_interest)
        
        # Bypass CocoBlackbox's __init__ to avoid re-creating the suite
        super(CocoBlackbox, self).__init__(
            problem_name=self.problem.id,
            x_count=self.problem.dimension,
            f_count=self.problem.number_of_objectives,
            g_count=self.problem.number_of_constraints,
            lower_bounds=lb,
            upper_bounds=ub,
            region_of_interest_lower=roi_lb,
            region_of_interest_upper=roi_ub
        )

# ==========================================
# The Worker Process (Target for multiprocessing.Process)
# ==========================================
def _worker_solve_instance(
    base_opt: Optimizer, 
    suite_name: str, 
    func_idx: int, 
    dim: int, 
    inst_idx: int, 
    budget_multiplier: float, 
    log_level: str,
    region_bound_fallback: float,
    region_of_interest: float,
    dest_path: str,
    heartbeat_path: str
):
    # Ensure relative exdata directory exists
    os.makedirs("exdata", exist_ok=True)
    
    # 1. Create temp directory inside 'exdata'
    # mkdtemp automatically appends a random unique hash to the prefix
    temp_dir_abs = tempfile.mkdtemp(
        prefix=f"coco_f{func_idx:03d}_d{dim:02d}_i{inst_idx:03d}_", 
        dir="exdata"
    )
    
    # 2. Extract relative folder name (e.g. "coco_f001_d02_i001_a8f9d0")
    rel_folder = os.path.basename(temp_dir_abs)
    
    try:
        # Pass relative folder name to COCO
        observer_options = (
            f"result_folder: {rel_folder} "
            f"algorithm_name: {base_opt.optimizer_name} "
            f"log_nondominated: {log_level}"
        )
        suite_options = f"function_indices: {func_idx} dimensions: {dim} instance_indices: {inst_idx}"
        suite = cocoex.Suite(suite_name, "", suite_options)
        observer = cocoex.Observer(suite_name, observer_options)
        
        # Run optimization loop...
        for problem in suite:
            problem.observe_with(observer)
            blackbox = LiveCocoBlackboxAdapter(problem, region_bound_fallback, region_of_interest)
            
            opt = copy.deepcopy(base_opt)
            budget = int(round(budget_multiplier * blackbox.x_count))
            max_evals = max(1, budget)
            
            opt.setup(
                expected_number_of_evaluations=budget,
                n_vars=blackbox.x_count,
                n_obj=blackbox.f_count,
                n_constr=blackbox.g_count,
                lower_bounds=blackbox.lower_bounds,
                upper_bounds=blackbox.upper_bounds,
                region_of_interest=blackbox.region_of_interest
            )
            
            evals = 0
            stall_count = 0
            last_hb_time = time.time()
            
            while evals < max_evals:
                # --- HEARTBEAT ---
                now = time.time()
                if now - last_hb_time >= 10.0:
                    with open(heartbeat_path, 'w') as f:
                        f.write(str(evals))
                    last_hb_time = now
                
                x_batch = opt.ask()
                if len(x_batch) == 0 and not opt.optimizer_running:
                    break
                if len(x_batch) == 0:
                    stall_count += 1
                    if stall_count > 100:
                        break
                else:
                    stall_count = 0
                    
                f_batch, g_batch = blackbox.evaluate(x_batch)
                opt.tell(x_batch, f_batch, g_batch)
                evals += len(x_batch)

            # Final heartbeat write so the watchdog sees completion
            with open(heartbeat_path, 'w') as f:
                f.write(str(evals))
                
            del opt
            del blackbox
            del problem
            gc.collect()
            
        source_path = observer.result_folder 
        del observer
        del suite
        gc.collect()
            
        # Move completed output to dest_path
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path, ignore_errors=True)
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(source_path, dest_path)
        
        with open(os.path.join(dest_path, ".done"), 'w') as f:
            f.write("completed")
            
    except Exception as e:
        pass
    finally:
        # Clean up temp_dir if it survived (e.g. if error occurred before move)
        if os.path.exists(temp_dir_abs):
            shutil.rmtree(temp_dir_abs, ignore_errors=True)

# ==========================================
# The Unified Scheduler
# ==========================================
def Run_Coco_Benchmark(
    payloads: List[Payload],
    max_workers: int = 61,
    frequency: int = 500,
    max_retries: int = 5,
    force_overwrite: bool = False,
    watchdog_interval: int = 10,      # How often the scheduler checks processes
    stall_timeout: int = 120          # Max seconds allowed without heartbeat update
) -> None:
    """
    Runs multiple optimizer payloads on COCO suites in parallel.
    Uses multiprocessing with an aggressive watchdog timer to prevent hung processes.
    """
    # 1. Flatten payloads into atomic tasks
    tasks = []
    for p in payloads:
        for func_idx in p.functions:
            for dim in p.dimensions:
                for inst_idx in p.instances:
                    dest_path = os.path.join(
                        COCO_RESULTS_DIR, 
                        p.suite_name, 
                        f"{p.optimizer.optimizer_name}_b{int(p.budget_multiplier)}", 
                        f"d{dim:02d}", 
                        f"f{func_idx:03d}", 
                        f"i{inst_idx:03d}"
                    )
                    
                    # Check if already done
                    if not force_overwrite and os.path.exists(os.path.join(dest_path, ".done")):
                        continue
                        
                    timeout = math.ceil(dim * p.budget_multiplier / frequency) + 10
                    
                    tasks.append({
                        "optimizer": p.optimizer,
                        "suite_name": p.suite_name,
                        "func_idx": func_idx,
                        "dim": dim,
                        "inst_idx": inst_idx,
                        "budget_multiplier": p.budget_multiplier,
                        "log_level": p.log_level.value,
                        "region_bound_fallback": p.region_bound_fallback,
                        "region_of_interest": p.region_of_interest,
                        "dest_path": dest_path,
                        "timeout": timeout,
                        "retries_left": max_retries
                    })
                    
    if not tasks:
        print("All tasks are already completed. Exiting benchmark.")
        return
        
    total_tasks = len(tasks)
    print(f"\n=== Starting COCO Benchmark with {total_tasks} total tasks ===")
    
    active_procs = {}
    failed_tasks = []
    task_queue = tasks.copy()
    completed = 0
    
    # Create a temp directory for heartbeat files
    heartbeat_dir = tempfile.mkdtemp(prefix="coco_heartbeats_")
    
    try:
        while task_queue or active_procs:
            # 1. Fill up to max_workers
            while len(active_procs) < max_workers and task_queue:
                task = task_queue.pop(0)
                hb_path = os.path.join(heartbeat_dir, f"hb_{task['func_idx']}_{task['dim']}_{task['inst_idx']}_{uuid.uuid4().hex}.txt")
                
                # Initialize heartbeat file
                with open(hb_path, 'w') as f:
                    f.write("0")
                    
                proc = mp.Process(
                    target=_worker_solve_instance,
                    args=(
                        task['optimizer'],
                        task['suite_name'],
                        task['func_idx'],
                        task['dim'],
                        task['inst_idx'],
                        task['budget_multiplier'],
                        task['log_level'],
                        task['region_bound_fallback'],
                        task['region_of_interest'],
                        task['dest_path'],
                        hb_path
                    )
                )
                # Daemon=True means if the main script is killed, children die too.
                proc.daemon = True 
                proc.start()
                
                active_procs[proc] = {
                    **task,
                    "start_time": time.time(),
                    "heartbeat_path": hb_path,
                    "last_heartbeat_time": time.time()
                }
                
            # 2. Watchdog Loop: Check active processes
            time.sleep(watchdog_interval)
            now = time.time()
            
            for proc, task_info in list(active_procs.items()):
                if not proc.is_alive():
                    proc.join()
                    completed += 1
                    
                    if os.path.exists(os.path.join(task_info['dest_path'], ".done")):
                        print(f"  > [{completed}/{total_tasks}] Finished F{task_info['func_idx']:02d} | D{task_info['dim']:02d} | I{task_info['inst_idx']:02d}")
                    else:
                        print(f"  > [CRASH] F{task_info['func_idx']:02d} | D{task_info['dim']:02d} | I{task_info['inst_idx']:02d} failed without .done")
                        if os.path.exists(task_info['dest_path']):
                            shutil.rmtree(task_info['dest_path'], ignore_errors=True)
                            
                        if task_info['retries_left'] > 0:
                            task_info['retries_left'] -= 1
                            task_queue.append(task_info)
                        else:
                            failed_tasks.append(task_info)
                            
                    if os.path.exists(task_info['heartbeat_path']):
                        os.remove(task_info['heartbeat_path'])
                    del active_procs[proc]
                    
                else:
                    # Process is still alive. Check timeouts.
                    wall_clock_exceeded = (now - task_info['start_time']) > task_info['timeout']
                    
                    # Check heartbeat mtime
                    try:
                        hb_mtime = os.path.getmtime(task_info['heartbeat_path'])
                        heartbeat_stalled = (now - hb_mtime) > stall_timeout
                    except FileNotFoundError:
                        heartbeat_stalled = False 
                        
                    if wall_clock_exceeded or heartbeat_stalled:
                        reason = "Wall-clock timeout" if wall_clock_exceeded else "Heartbeat stalled"
                        print(f"  > [KILL] F{task_info['func_idx']:02d} | D{task_info['dim']:02d} | I{task_info['inst_idx']:02d} ({reason}). Terminating...")
                        
                        # The nuclear option
                        proc.kill()
                        proc.join()
                        
                        if os.path.exists(task_info['dest_path']):
                            shutil.rmtree(task_info['dest_path'], ignore_errors=True)
                            
                        if task_info['retries_left'] > 0:
                            task_info['retries_left'] -= 1
                            task_queue.append(task_info)
                        else:
                            failed_tasks.append(task_info)
                            
                        if os.path.exists(task_info['heartbeat_path']):
                            os.remove(task_info['heartbeat_path'])
                        del active_procs[proc]

    finally:
        # 3. Cleanup: Write failures CSV and remove heartbeat dir
        if os.path.exists(heartbeat_dir):
            shutil.rmtree(heartbeat_dir, ignore_errors=True)
            
        if failed_tasks:
            from collections import defaultdict
            grouped_fails = defaultdict(list)
            for t in failed_tasks:
                grouped_fails[t['optimizer'].optimizer_name].append(t)
                
            for opt_name, fails in grouped_fails.items():
                budget = int(fails[0]['budget_multiplier'])
                csv_path = os.path.join(COCO_RESULTS_DIR, fails[0]['suite_name'], f"{opt_name}_b{budget}", "failures.csv")
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Function", "Dimension", "Instance", "Suite", "BudgetMultiplier"])
                    for fail in fails:
                        writer.writerow([fail['func_idx'], fail['dim'], fail['inst_idx'], fail['suite_name'], fail['budget_multiplier']])
                print(f"  > Wrote failure log for {opt_name} to {csv_path}")

        print("\nBenchmark run finished.")
