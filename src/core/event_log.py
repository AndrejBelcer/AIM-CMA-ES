import os
import numpy as np
import gzip
import pickle
import uuid
from datetime import datetime
from typing import List, Dict, Any

from src.core.archive import Archive
from src.core.paths import SOLVER_DEBUG_DIR


class Event:
    def __init__(self, log: Dict[str, Any]) -> None:
        self.log = log

    def __repr__(self) -> str:
        return f"Event({self.log})"


class EventLog:
    def __init__(self, solver_name: str, black_box_name: str) -> None:
        self.solver_name = solver_name
        self.black_box_name = black_box_name
        self.evaluation_indices: List[int] = []
        self.ask_events: List[Event] = []
        self.tell_events: List[Event] = []
        
    def AddEvent(self, evaluation_index: int, ask_event: Event, tell_event: Event) -> None:
        self.evaluation_indices.append(evaluation_index)
        self.ask_events.append(ask_event)
        self.tell_events.append(tell_event)

    def Save(self, archive: Archive) -> None:
        """
        Saves the Archive as .npz and the EventLog as a compressed binary pickle (.pkl.gz).
        """
        target_dir = os.path.join(SOLVER_DEBUG_DIR, self.black_box_name, self.solver_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Generate unique identifiers
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uuid_raw = uuid.uuid7()
        uuid_str = uuid_raw.hex[:8]
        eval_str = f"n{archive.n_points}"
        
        base_filename = f"{timestamp_str}_{uuid_str}_{eval_str}"
        
        # 1. Save Archive (.npz)
        archive_filepath = os.path.join(target_dir, f"{base_filename}.npz")
        archive.save(archive_filepath)
        print(f"Debug Archive saved to: {archive_filepath}")
        
        # 2. Package the Event Log structure
        serialized_events = []
        for idx, ask_ev, tell_ev in zip(self.evaluation_indices, self.ask_events, self.tell_events):
            serialized_events.append({
                "evaluation_index": idx,
                "ask_data": ask_ev.log,
                "tell_data": tell_ev.log
            })
            
        log_data = {
            "metadata": {
                "solver_name": self.solver_name,
                "black_box_name": self.black_box_name,
                "timestamp": timestamp_str,
                "uuid": uuid_str,
                "total_evaluations": archive.n_points
            },
            "events": serialized_events
        }
        
        # 3. Save Event Log as compressed binary (.pkl.gz)
        log_filepath = os.path.join(target_dir, f"{base_filename}_log.pkl.gz")
        try:
            # We use gzip.open with 'wb' (write binary) and higher protocol for speed/size
            with gzip.open(log_filepath, "wb") as f:
                pickle.dump(log_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Compressed Binary Event Log saved to: {log_filepath}")
        except Exception as e:
            print(f"Failed to save binary log file: {e}")





def load_run_data(archive_path: str, log_path: str):
    # Load the evaluation archive
    with np.load(archive_path) as data:
        X = data['X']
        F = data['F']
        G = data['G']
        
    # Load the compressed binary event log
    with gzip.open(log_path, "rb") as f:
        log_data = pickle.load(f)
        
    return X, F, G, log_data
