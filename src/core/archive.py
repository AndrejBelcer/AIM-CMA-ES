import numpy as np
import os

class Archive:
    def __init__(self, x_dim: int, f_dim: int, g_dim: int):
        self.x_dim = x_dim
        self.f_dim = f_dim
        self.g_dim = g_dim
        
        # Lists for O(1) appending
        self._x_list = []
        self._f_list = []
        self._g_list = []

    def update(self, x: np.ndarray, f: np.ndarray, g: np.ndarray):
        # Convert to numpy and force 2D
        x = np.atleast_2d(x).astype(np.float64)
        f = np.atleast_2d(f).astype(np.float64)
        g = np.atleast_2d(g).astype(np.float64)
        
        n_samples = x.shape[0]

        # Validation: Ensure width matches pre-defined dims
        # Note: If batch is empty (size 0), we force it to (0, dim) 
        # to ensure vstack works later.
        if x.shape[1] != self.x_dim and x.size > 0:
            raise ValueError(f"Expected X dim {self.x_dim}, got {x.shape[1]}")
        
        # Append to lists if there are points to add
        if (n_samples > 0):
            self._x_list.append(x.reshape(n_samples, self.x_dim))
            self._f_list.append(f.reshape(n_samples, self.f_dim))
            self._g_list.append(g.reshape(n_samples, self.g_dim))


    def save(self, filepath: str):
        """
        Saves the history to a compressed .npz file.
        """
        if not self._x_list:
            print("Archive is empty. Nothing to save.")
            return
            
        # Concatenate lists into single numpy arrays
        full_x = np.vstack(self._x_list)
        full_f = np.vstack(self._f_list)
        full_g = np.vstack(self._g_list)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(filepath, X=full_x, F=full_f, G=full_g)

    @property
    def n_points(self):
        # Using a generator expression is slightly more memory efficient
        return sum(b.shape[0] for b in self._x_list)
