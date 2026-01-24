#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import resource
import json
from typing import List, Optional, Tuple
import click
from datasets.sidd import SIDDPatchDataset
from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
from torchvision import transforms
from PIL import Image
import time
from metrics import (
    EpochMetrics, 
    TrainingSpecification, 
    TrainingRun,
    get_next_results_dir,
    get_existing_results_dir
)


def save_image_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor image to PNG file."""
    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = to_pil(tensor.cpu())
    pil_image.save(path)

def get_dataset_basepath() -> str:
    return "/home/yogo/media/Datasets/SIDD_Small_sRGB_Only/Data"
    return f"{os.path.dirname(__file__)}/datasets/sidd_tmpfs/Data"

def hard_limit_memory_usage():
    memory_limit_gb = 32
    memory_limit_bytes = memory_limit_gb * (1024**3)
    
    try:
        # Set virtual address space limit (RLIMIT_AS)
        # When exceeded, the process will be killed by the OOM killer
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        print(f"Memory limit set to {memory_limit_gb}GB - process will be killed if exceeded")
    except (ValueError, OSError) as e:
        print(f"Warning: Could not set memory limit: {e}")
        raise e
    
    # Also adjust OOM score to make this process more likely to be killed
    # Higher score = more likely to be killed by OOM killer
    try:
        oom_score_adj = 1000  # High priority for OOM killer (range: -1000 to 1000)
        with open(f"/proc/self/oom_score_adj", "w") as f:
            f.write(str(oom_score_adj))
        print(f"OOM score adjusted to {oom_score_adj} (higher = more likely to be killed)")
    except (IOError, OSError) as e:
        print(f"Warning: Could not adjust OOM score: {e}")
        raise e

def load_training_specification() -> TrainingSpecification:
    """Load training specification from training_specification.json."""
    spec_file = Path("training_specification.json")
    
    if not spec_file.exists():
        raise FileNotFoundError(
            f"Training specification file '{spec_file}' not found. "
            "Please create this file with the required hyperparameters."
        )
    
    with open(spec_file, 'r') as f:
        spec_dict = json.load(f)
    
    return TrainingSpecification(**spec_dict)


def load_previous_training_run(results_dir: Path) -> Tuple[TrainingRun, int]:
    """Load previous training run from results directory.
    
    Returns:
        Tuple of (TrainingRun, last_completed_epoch)
    """
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found in {results_dir}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (list) and new format (TrainingRun)
    if isinstance(data, dict) and 'epochs' in data and 'specification' in data:
        training_run = TrainingRun(**data)
    elif isinstance(data, list):
        # Old format - need to load spec from file
        spec = load_training_specification()
        epochs = [EpochMetrics(**m) for m in data]
        training_run = TrainingRun(specification=spec, epochs=epochs)
    else:
        raise ValueError("Invalid metrics file format!")
    
    # Find last completed epoch
    if training_run.epochs:
        last_epoch = max(m.epoch for m in training_run.epochs)
    else:
        last_epoch = 0
    
    return training_run, last_epoch


def save_checkpoint(results_dir: Path, model: torch.nn.Module, training_run: TrainingRun):
    """Save checkpoint (weights and metrics)."""
    # Save model weights
    model_path = results_dir / "unet_sidd.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(training_run.model_dump(), f, indent=2)


@click.command()
@click.option('--resume', type=int, default=None, 
              help='Resume training from results directory with this index')
def main(resume: Optional[int]):
    hard_limit_memory_usage()
    
    # Determine if resuming or starting new training
    if resume is not None:
        # Resume from previous training
        results_dir = get_existing_results_dir(resume)
        print(f"Resuming training from: {results_dir}")
        
        # Load previous training run
        previous_run, last_epoch = load_previous_training_run(results_dir)
        spec = previous_run.specification
        all_metrics = previous_run.epochs.copy()
        
        print(f"Previous training: {len(all_metrics)} epochs completed (last: {last_epoch})")
        print(f"Target epochs: {spec.epochs}")
        
        # Check if training is already complete
        if last_epoch >= spec.epochs:
            print(f"Training already complete! All {spec.epochs} epochs have been completed.")
            return
        
        start_epoch = last_epoch
        print(f"Resuming from epoch {start_epoch + 1} to {spec.epochs}")
    else:
        # Start new training
        spec = load_training_specification()
        print(f"Starting new training with specification: {spec.model_dump_json(indent=2)}")
        
        results_dir = get_next_results_dir()
        print(f"Saving results to: {results_dir}")
        
        all_metrics: List[EpochMetrics] = []
        start_epoch = 0
    
    root = Path(spec.dataset_path)

    # Collect image pairs
    pairs = collect_sidd_pairs(root)
    if spec.max_image_pairs is not None:
        pairs = pairs[:spec.max_image_pairs]
    
    train_pairs, val_pairs, _ = split_dataset(pairs)

    # Load sample patch from first image pair for visualization
    # Extract a single patch from the center of the image for consistent visualization
    sample_noisy_path, sample_clean_path = pairs[0]
    to_tensor = transforms.ToTensor()
    
    sample_noisy_img = Image.open(sample_noisy_path).convert("RGB")
    sample_clean_img = Image.open(sample_clean_path).convert("RGB")
    
    # Convert to tensor
    sample_noisy_full = to_tensor(sample_noisy_img)
    sample_clean_full = to_tensor(sample_clean_img)
    
    # Extract a patch from the center of the image
    _, h, w = sample_noisy_full.shape
    ps = spec.patch_size
    
    # Calculate center patch coordinates
    center_x = (w - ps) // 2
    center_y = (h - ps) // 2
    
    # Extract center patches
    sample_noisy_patch = sample_noisy_full[:, center_y:center_y+ps, center_x:center_x+ps]
    sample_clean_patch = sample_clean_full[:, center_y:center_y+ps, center_x:center_x+ps]

    train_ds = SIDDPatchDataset(train_pairs, spec.patch_size, spec.patches_per_image)
    val_ds = SIDDPatchDataset(val_pairs, spec.patch_size, spec.patches_per_image)

    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=spec.batch_size)

    device = torch.device(spec.device)
    if spec.device == "xpu":
        assert torch.xpu.is_available()

    model = UNet().to(device)
    
    # Load weights if resuming
    if resume is not None:
        model_path = results_dir / "unet_sidd.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}, starting with fresh weights")
    
    # Set up loss function
    loss_functions = {
        "L1Loss": nn.L1Loss(),
        "MSELoss": nn.MSELoss(),
    }
    if not spec.loss_function in loss_functions:
        raise ValueError(f"Unsupported loss function: {spec.loss_function}")
    else:
        criterion = loss_functions[spec.loss_function]
    
    # Set up optimizer
    if spec.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=spec.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {spec.optimizer}")

    # Training loop
    for epoch in range(start_epoch, spec.epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0

        train_start_time = time.time()
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_time = time.time() - train_start_time

        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

                output = model(noisy)
                val_loss += criterion(output, clean).item()
        val_time = time.time() - val_start_time

        epoch_time = time.time() - epoch_start_time

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f}"
        )

        # Generate sample denoised patch
        model.eval()
        with torch.no_grad():
            # Add batch dimension and move to device
            sample_noisy_batch = sample_noisy_patch.unsqueeze(0).to(device)
            sample_denoised_batch = model(sample_noisy_batch)
            sample_denoised_patch = sample_denoised_batch.squeeze(0).cpu()

        # Save sample patch images
        save_image_tensor(sample_noisy_patch, results_dir / f"{epoch+1}_noisy.png")
        save_image_tensor(sample_denoised_patch, results_dir / f"{epoch+1}_denoised.png")
        save_image_tensor(sample_clean_patch, results_dir / f"{epoch+1}_clean.png")

        # Create and store metrics
        metrics = EpochMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            epoch_time=epoch_time,
            train_time=train_time,
            val_time=val_time,
            total_load_time=total_load_time,
            total_h2d_time=total_h2d_time,
            total_step_time=total_step_time,
        )
        all_metrics.append(metrics)

        # Create training run object
        training_run = TrainingRun(specification=spec, epochs=all_metrics)
        
        # Save checkpoint if it's time (based on checkpoint_frequency)
        current_epoch_num = epoch + 1
        if current_epoch_num % spec.checkpoint_frequency == 0 or current_epoch_num == spec.epochs:
            save_checkpoint(results_dir, model, training_run)
            print(f"Checkpoint saved at epoch {current_epoch_num}")

    # Final save (in case checkpoint_frequency didn't catch the last epoch)
    training_run = TrainingRun(specification=spec, epochs=all_metrics)
    save_checkpoint(results_dir, model, training_run)
    
    print(f"\nTraining complete! Results saved to: {results_dir}")
    print(f"Model weights saved to: {results_dir / 'unet_sidd.pth'}")


if __name__ == "__main__":
    main()
