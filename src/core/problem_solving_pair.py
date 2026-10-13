import numpy as np
import os
import time
import uuid
from datetime import datetime
from typing import Optional


import cProfile
import pstats

from src.core.blackbox import Blackbox
from src.core.optimizer import Optimizer
from src.core.archive import Archive
from src.core.event_log import Event, EventLog

from src.core.paths import SOLVER_RESULTS_DIR, SOLVER_DEBUG_DIR

class ProblemSolvingPair:
    def __init__(self, blackbox: Blackbox, optimizer: Optimizer, allowed_evals: int, debug_run: bool = False):
        self.blackbox = blackbox
        self.optimizer = optimizer
        self.allowed_evals = allowed_evals
        self.debug_run = debug_run
    
    def run(self):
        """
        The main function to call.
        Calls the setup and solve to run the optimization loop.
        """
        
        self.optimizer.setup(expected_number_of_evaluations = self.allowed_evals, n_vars=self.blackbox.x_count, n_obj=self.blackbox.f_count, n_constr=self.blackbox.g_count, lower_bounds=self.blackbox.lower_bounds, upper_bounds=self.blackbox.upper_bounds, region_of_interest=self.blackbox.region_of_interest)
        
        if self.debug_run:
            self.debug_solve(number_of_evaluations=self.allowed_evals)
        else:
            self.solve(number_of_evaluations=self.allowed_evals)

    def solve(self, number_of_evaluations: int, progress_interval: int = 1000) -> str:
        """
        The main optimization loop.
        Returns the path to the saved results file.
        """
        print(f"Starting {self.optimizer.optimizer_name} on {self.blackbox.problem_name}")

        start_time = time.time()

        evaluation_count: int = 0
        next_progress_eval: int = progress_interval
        archive = Archive(x_dim=self.blackbox.x_count, f_dim=self.blackbox.f_count, g_dim=self.blackbox.g_count)

        while(evaluation_count < number_of_evaluations):
            # 1. Ask optimizer for points
            x_batch = self.optimizer.ask()

            if(x_batch.shape[0] == 0):
                if(not self.optimizer.optimizer_running):
                    print("Optimizer returned zero candidates and is not running. Ending optimization.")
                    break
                

            evaluations_left: int = number_of_evaluations - evaluation_count
            evaluations_exceeded: bool = x_batch.shape[0] > evaluations_left
            
            if (evaluations_exceeded):
                x_batch = x_batch[-evaluations_left:]
                print(f"Evaluation budget exceeded. Trimming batch to last {evaluations_left} points.")

            # 2. Evaluate
            f_batch, g_batch = self.blackbox.evaluate(x_batch)
            # 3. Archive
            archive.update(x_batch, f_batch, g_batch)

            if(not evaluations_exceeded):
                # 4. Tell optimizer if we have not exceeded the evaluation budget
                self.optimizer.tell(x_batch, f_batch, g_batch)

            evaluation_count += x_batch.shape[0]

            # Print progress every N evals
            if evaluation_count >= next_progress_eval:
                print(f"  > {self.optimizer.optimizer_name} Evals: {evaluation_count}/{number_of_evaluations}")
                next_progress_eval += progress_interval

        elapsed = time.time() - start_time
        print(f"\nFinished. Total Evaluations: {evaluation_count}. Time: {elapsed:.2f}s")

        # Save results
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uuid_raw = uuid.uuid7()
        uuid_str = uuid_raw.hex[:8]
        eval_str = f"n{evaluation_count}"

        # Save to: Solver_Results/{Problem_Name}/{Optimizer_Name}/{Filename}
        filename = f"{timestamp_str}_{uuid_str}_{eval_str}.npz"
        save_path = os.path.join(SOLVER_RESULTS_DIR, self.blackbox.problem_name, self.optimizer.optimizer_name, filename)
        archive.save(save_path)
        return save_path
            
        
    def debug_solve(self, number_of_evaluations: int, progress_interval: int = 1000):
        print(f"Starting {self.optimizer.optimizer_name} on {self.blackbox.problem_name}")

        # Profiler start
        start_time = time.time()
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Debug start
        evaluation_count = self._debug_solve(number_of_evaluations, progress_interval)
        
        # Profiler stop
        profiler.disable()
        elapsed = time.time() - start_time
        print(f"\nFinished. Total Evaluations: {evaluation_count}. Time: {elapsed:.2f}s")

        #Profiler save
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
        profiler.dump_stats(f"{self.normalize_filename(self.optimizer.optimizer_name)}_Profiler.prof")
        
        

    def _debug_solve(self, number_of_evaluations: int, progress_interval: int = 1000) -> int:
        evaluation_count: int = 0
        next_progress_eval: int = progress_interval
        archive: Archive = Archive(x_dim=self.blackbox.x_count, f_dim=self.blackbox.f_count, g_dim=self.blackbox.g_count)
        debug_log: EventLog = EventLog(self.optimizer.optimizer_name, self.blackbox.problem_name)


        while(evaluation_count < number_of_evaluations):
            # 1. Ask optimizer for points
            x_batch, ask_debug = self.optimizer.debug_ask()

            if(x_batch.shape[0] == 0):
                if(not self.optimizer.optimizer_running):
                    print("Optimizer returned zero candidates and is not running. Ending optimization.")
                    break


            evaluations_left: int = number_of_evaluations - evaluation_count
            evaluations_exceeded: bool = x_batch.shape[0] > evaluations_left
            
            if (evaluations_exceeded):
                x_batch = x_batch[-evaluations_left:]
                print(f"Evaluation budget exceeded. Trimming batch to last {evaluations_left} points.")

            # 2. Evaluate
            f_batch, g_batch = self.blackbox.evaluate(x_batch)
            # 3. Archive
            archive.update(x_batch, f_batch, g_batch)
            # 4. Tell optimizer if we have not exceeded the evaluation budget
            tell_debug = self.optimizer.debug_tell(x_batch, f_batch, g_batch) if(not evaluations_exceeded) else Event({"Default": "Ran out of evaluations."})

            # Update logs
            debug_log.AddEvent(evaluation_count, ask_debug, tell_debug)
            evaluation_count += x_batch.shape[0]
            

            # Print progress every N evals
            if evaluation_count >= next_progress_eval:
                print(f"  > {self.optimizer.optimizer_name} Evals: {evaluation_count}/{number_of_evaluations}")
                next_progress_eval += progress_interval

        # Save results
        debug_log.Save(archive)
        
        return evaluation_count

    def normalize_filename(self, text: str) -> str:
        return text.replace(" ", "_").replace("-", "_")
