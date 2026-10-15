import numpy as np

from src.implementations.optimizer_implementations.aim_cma_es.initializer.initializer import Initializer
from src.implementations.cobyqa_implementation.optimizer import COBYQAOptimizer
from src.implementations.cobyqa_implementation.settings import COBYQAParameters
from src.implementations.optimizer_implementations.aim_cma_es.component_parameters import InitializationParameters

class InitializerCOBYQA(Initializer):
    def __init__(self, x_count: int, f_count: int, g_count: int, initialization_parameters: InitializationParameters) -> None:
        
        super().__init__(x_count, f_count, g_count, initialization_parameters)
        
        # Extract COBYQA specific parameters
        self.initial_trust_region_radius = initialization_parameters.initial_trust_region_radius
        self.final_trust_region_radius = initialization_parameters.final_trust_region_radius

    def _create_optimizer(self, x0: np.ndarray) -> COBYQAOptimizer:
        # Bounds are strictly [0, 1]^D in the normalized space
        cobyqa_parameters = COBYQAParameters(rhobeg=self.initial_trust_region_radius, rhoend=self.final_trust_region_radius)
        return COBYQAOptimizer(x0=x0, lower_bounds=np.zeros(self.x_count), upper_bounds=np.ones(self.x_count), g_count=self.g_count, h_count = 0, params=cobyqa_parameters)

    def _is_optimizer_running(self) -> bool:
        if self.active_optimizer is None:
            return False
        return not self.active_optimizer.stop()

    def _ask_optimizer(self) -> np.ndarray:
        return self.active_optimizer.ask()

    def _tell_optimizer(self, x_norm: np.ndarray, s: np.ndarray, g: np.ndarray):
        self.active_optimizer.tell(x_norm, s, g)
