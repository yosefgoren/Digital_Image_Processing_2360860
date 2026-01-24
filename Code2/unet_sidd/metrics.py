from pathlib import Path
from pydantic import BaseModel
from typing import *

class EpochMetrics(BaseModel):
    """Pydantic model for epoch metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float
    train_time: float
    val_time: float
    total_load_time: float
    total_h2d_time: float
    total_step_time: float

def get_results_basedir() -> str:
    return "./results"

def get_max_results_idx() -> int:
    results_base = Path(get_results_basedir())
    results_base.mkdir(exist_ok=True)
    return max([int(entry.name) for entry in results_base.iterdir() if entry.name.isdecimal()])

def get_existing_results_dir(results_idx: Optional[int] = None) -> Path:
    if results_idx is None:
        results_idx = get_max_results_idx()

    res = Path(f"{get_results_basedir()}/{results_idx}")
    assert res.is_dir()
    return res

def get_next_results_dir() -> Path:
    results_dir = Path(f"{get_results_basedir()}/{get_max_results_idx()+1}")
    results_dir.mkdir(exist_ok=True)
    return results_dir

