import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from src.core.event_log import Event


class Optimizer(ABC):
    """
    Base optimizer class. Defines the interface for optimizers.
    """
    # Properties that all optimizers should have
    def __init__(self, optimizer_name: str):
        if (not isinstance(optimizer_name, str)):
            raise ValueError("Optimizer name must be a string.")

        self.optimizer_name = optimizer_name
        self.optimizer_running: bool = True  # Indicates if the optimizer is still running (not converged or terminated)

    @abstractmethod
    def setup(self, expected_number_of_evaluations: int, n_vars: int, n_obj: int, n_constr: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray, region_of_interest: np.ndarray):
        """Initialize internal state based on problem dimensions."""
        pass

    @abstractmethod
    def ask(self) -> np.ndarray:
        """Return candidate points."""
        pass
    
    @abstractmethod
    def tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        """Update algorithm with results."""
        pass
    
    def debug_ask(self) -> tuple[np.ndarray, Event]:
        """Optional method for debugging purposes."""
        return (self.ask(), Event({"Default": "No Ask Debug Override"}))
    
    
    def debug_tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray) -> Event:
        """Optional method for debugging purposes."""
        self.tell(x, f, g)
        return Event({"Default": "No Tell Debug Override"})

