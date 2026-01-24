from pathlib import Path
from pydantic import BaseModel

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


def get_next_results_dir() -> Path:
    """Get the lowest available results directory number."""
    results_base = Path("./results")
    results_base.mkdir(exist_ok=True)
    
    # Find the lowest available number
    existing_dirs = set()
    for item in results_base.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_dirs.add(int(item.name))
    
    # Find the lowest number that doesn't exist
    next_num = 0
    while next_num in existing_dirs:
        next_num += 1
    
    results_dir = results_base / str(next_num)
    results_dir.mkdir(exist_ok=True)
    return results_dir

