from typing import List

from multiprocessing import freeze_support

from src.core.optimizer import Optimizer
from src.implementations.optimizer_implementations.aim_cma_es.aim_cma import AimCma
from src.core.blackbox import Blackbox
from src.implementations.blackbox_implementations.coco_blackbox import CocoBlackbox
from src.core.problem_solving_pair import ProblemSolvingPair

from src.pipelines.solver_pipeline.coco_benchmark_pipeline import Run_Coco_Benchmark, Payload, GeneratePayloads
from src.pipelines.solver_pipeline.pair_problem_pipeline import Run_Solvers_Parallel

if __name__ == '__main__':
    freeze_support()
    
    optimizer: Optimizer = AimCma()
    blackbox: Blackbox = CocoBlackbox(dimension=2, function_id=1, instance_id=1)
    

    optimizers: List[Optimizer] = [
        optimizer
    ]

    blackboxes: List[Blackbox] = [
        blackbox
    ]

    pair = ProblemSolvingPair(blackbox=blackbox, optimizer=optimizer, allowed_evals=50000)
    pair.run()


    Run_Solvers_Parallel(blackboxes, optimizers, max_evals=50000, repeats=5, max_workers=7)

    payloads = GeneratePayloads(optimizers, dimensions=[2], functions=[11], instances=[2], budget_multiplier=1e5)

    Run_Coco_Benchmark(payloads = payloads, max_workers=7, frequency=250)
