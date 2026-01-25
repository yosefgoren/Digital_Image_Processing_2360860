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
    to_pil = transforms.ToPILImage()
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = to_pil(tensor.cpu())
    pil_image.save(path)

def hard_limit_memory_usage():
    memory_limit_gb = 32
    memory_limit_bytes = memory_limit_gb * (1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    print(f"Memory limit set to {memory_limit_gb}GB - process will be killed if exceeded")
    
    oom_score_adj = 1000  # High priority for OOM killer (range: -1000 to 1000)
    with open(f"/proc/self/oom_score_adj", "w") as f:
        f.write(str(oom_score_adj))
    print(f"OOM score adjusted to {oom_score_adj} (higher = more likely to be killed)")
    
def load_training_specification() -> TrainingSpecification:
    spec_file = Path("training_specification.json")
    
    if not spec_file.exists():
        raise FileNotFoundError(f"Missing specification file '{spec_file}' not found. Please create and fill it.")
    
    with open(spec_file, 'r') as f:
        spec_dict = json.load(f)
    
    return TrainingSpecification(**spec_dict)

def load_previous_training_run(results_dir: Path) -> TrainingRun:
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found in {results_dir}")
    
    with open(metrics_file, 'r') as f:
        return TrainingRun.model_validate_json(f.read())


def save_checkpoint(results_dir: Path, model: torch.nn.Module, training_run: TrainingRun):
    """Save checkpoint (weights and metrics)."""
    model_path = results_dir / "unet_sidd.pth"
    torch.save(model.state_dict(), model_path)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(training_run.model_dump(), f, indent=2)


def init_session(resume: Optional[int]) -> tuple[TrainingRun, int, Path]:
    if resume is not None:
        results_dir = get_existing_results_dir(resume)
        print(f"Resuming training from: {results_dir}")
        
        previous_run = load_previous_training_run(results_dir)
        last_epoch = max([0, len(previous_run.epochs)])
        spec = previous_run.specification
        all_metrics = previous_run.epochs.copy()
        
        print(f"Previous training: {len(all_metrics)} epochs completed (last: {last_epoch})")
        print(f"Target epochs: {spec.epochs}")
        
        if last_epoch >= spec.epochs:
            raise RuntimeError(f"Training already complete! All {spec.epochs} epochs have been completed.")
        
        start_epoch = last_epoch
        print(f"Resuming from epoch {start_epoch + 1} to {spec.epochs}")
        return previous_run, start_epoch, results_dir
    
    else:
        spec = load_training_specification()
        print(f"Starting new training with specification: {spec.model_dump_json(indent=2)}")
        
        results_dir = get_next_results_dir()
        print(f"Saving results to: {results_dir}")

        return TrainingRun(specification=spec), 0, results_dir

def get_center_patch_from_image_path(path: Path, patch_size: int) -> torch.Tensor:
    sample_image = Image.open(path).convert("RGB")

    to_tensor = transforms.ToTensor()
    sample_tensor = to_tensor(sample_image)
    
    _, h, w = sample_tensor.shape
    
    center_x = (w - patch_size) // 2
    center_y = (h - patch_size) // 2
    return sample_tensor[:, center_y:center_y+patch_size, center_x:center_x+patch_size]


@click.command()
@click.option('--resume', type=int, default=None, help='Resume training from results directory with this index')
def main(resume: Optional[int]):
    hard_limit_memory_usage()
    run, start_epoch, results_dir = init_session(resume)
    spec = run.specification

    root = Path(spec.dataset_path)

    pairs = collect_sidd_pairs(root)
    if spec.max_image_pairs is not None:
        pairs = pairs[:spec.max_image_pairs]
    
    train_pairs, val_pairs, _ = split_dataset(pairs)

    sample_noisy_path, sample_clean_path = pairs[0]
    sample_noisy_patch = get_center_patch_from_image_path(sample_noisy_path, spec.patch_size)
    sample_clean_patch = get_center_patch_from_image_path(sample_clean_path, spec.patch_size)

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
        "L1Loss": nn.L1Loss,
        "MSELoss": nn.MSELoss,
    }
    if not spec.loss_function in loss_functions:
        raise ValueError(f"Unsupported loss function: {spec.loss_function}")
    else:
        criterion = loss_functions[spec.loss_function]()
    
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
        total_load_time = 0.0
        total_h2d_time = 0.0
        total_step_time = 0.0

        train_start_time = time.time()
        t0 = time.time()
        for noisy, clean in train_loader:
            t1 = time.time()
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            t2 = time.time()

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            t3 = time.time()

            load_time = t1 - t0
            h2d_time = t2 - t1
            step_time = t3 - t2
            
            total_load_time += load_time
            total_h2d_time += h2d_time
            total_step_time += step_time

            print(f"load: {load_time:.3f}s | h2d: {h2d_time:.3f}s | step: {step_time:.3f}s")

            t0 = time.time()
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
            f"Epoch {epoch:03d} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f}"
        )

        model.eval()
        with torch.no_grad():
            sample_noisy_batch = sample_noisy_patch.unsqueeze(0).to(device)
            sample_denoised_batch = model(sample_noisy_batch)
            sample_denoised_patch = sample_denoised_batch.squeeze(0).cpu()

        save_image_tensor(sample_noisy_patch, results_dir / f"{epoch}_noisy.png")
        save_image_tensor(sample_denoised_patch, results_dir / f"{epoch}_denoised.png")
        save_image_tensor(sample_clean_patch, results_dir / f"{epoch}_clean.png")

        run.epochs.append(EpochMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            epoch_time=epoch_time,
            train_time=train_time,
            val_time=val_time,
            total_load_time=total_load_time,
            total_h2d_time=total_h2d_time,
            total_step_time=total_step_time,
        ))

        if epoch % spec.checkpoint_frequency == 0:
            save_checkpoint(results_dir, model, run)
            print(f"Checkpoint saved at epoch {epoch}")

    save_checkpoint(results_dir, model, run)
    
    print(f"\nTraining complete! Results saved to: {results_dir}")
    print(f"Model weights saved to: {results_dir / 'unet_sidd.pth'}")

if __name__ == "__main__":
    main()
