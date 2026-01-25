from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import json
import torch

class TrainingSpecification(BaseModel):
    patch_size: int
    patches_per_image: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str = "Adam"
    loss_function: str = "L1Loss"
    model_type: str = "UNet"
    device: str = "xpu"
    dataset_path: str
    max_image_pairs: Optional[int] = None  # None means use all pairs
    checkpoint_frequency: int = 1  # Save checkpoint every N epochs


class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float
    train_time: float
    val_time: float
    total_load_time: float
    total_h2d_time: float
    total_step_time: float


class TrainingRun(BaseModel):
    specification: TrainingSpecification
    epochs: List[EpochMetrics] = []



class TrainingDB:
    def _load_training_spec(self) -> TrainingSpecification:
        spec_file = Path("training_specification.json")
        
        if not spec_file.exists():
            raise FileNotFoundError(f"Missing specification file '{spec_file}' not found. Please create and fill it.")
        
        with open(spec_file, 'r') as f:
            spec_dict = json.load(f)
        
        return TrainingSpecification(**spec_dict)

    def _load_previous_training_run(self) -> TrainingRun:
        metrics_file = self._results_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found in {self._results_dir}")
        
        with open(metrics_file, 'r') as f:
            return TrainingRun.model_validate_json(f.read())

    def __init__(self, run_idx: Optional[int]):
        if run_idx is not None:
            print(f"Continuing training of run: {run_idx}")

            self._results_dir = get_existing_results_dir(run_idx)            
            self.run = self._load_previous_training_run()
            self.start_epoch = max([0, len(self.run.epochs)])

            if self.start_epoch >= self.run.specification.epochs:
                raise RuntimeError(f"Training already complete! All {self.run.specification.epochs} epochs have been completed.")
        else:
            spec = self._load_training_spec()
            print(f"Starting new training with specification: {spec.model_dump_json(indent=2)}")
            
            self._results_dir = get_next_results_dir()
            self.run = TrainingRun(specification=spec)
            self.start_epoch = 0

    def get_resource_path(self, filename: str) -> Path:
        return self._results_dir / filename      

    def get_weights_path(self) -> Path:
        return self.get_resource_path("unet_sidd.pth")

    def save_checkpoint(self, model: torch.nn.Module):
        model_path = self.get_weights_path()
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to: {model_path}")

        with open(self.get_resource_path("metrics.json"), "w") as f:
            json.dump(self.run.model_dump(), f, indent=2)


def get_results_basedir() -> str:
    return "./results"

def get_max_results_idx() -> int:
    results_base = Path(get_results_basedir())
    results_base.mkdir(exist_ok=True)
    return max([0] + [int(entry.name) for entry in results_base.iterdir() if entry.name.isdecimal()])

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

