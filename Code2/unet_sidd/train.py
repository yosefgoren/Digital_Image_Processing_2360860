#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import resource
import json
from typing import *
import click
from datasets.sidd import SIDDPatchDataset
from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
from torchvision import transforms
from PIL import Image
import time
from metrics import *

def save_image_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor image to PNG file."""
    to_pil = transforms.ToPILImage()
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = to_pil(tensor.cpu())
    print(f"Saving image tensor to: {path.name}")
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
    db = TrainingDB(resume)
    spec = db.run.specification

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
        model_path = db.get_weights_path()
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
        raise ValueError(f"Unsupported loss function: {spec.loss_function} (add it to dict?)")
    else:
        criterion = loss_functions[spec.loss_function]()
    
    # Set up optimizer
    if spec.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=spec.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {spec.optimizer}")

    # Training loop
    for epoch in range(db.start_epoch, spec.epochs):
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

        save_image_tensor(sample_noisy_patch, db.get_resource_path(f"{epoch}_noisy.png"))
        save_image_tensor(sample_denoised_patch, db.get_resource_path(f"{epoch}_denoised.png"))
        save_image_tensor(sample_clean_patch, db.get_resource_path(f"{epoch}_clean.png"))

        db.run.epochs.append(EpochMetrics(
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
            db.save_checkpoint(model)
            print(f"Checkpoint saved at epoch {epoch}")

    db.save_checkpoint(model)
    print(f"Training complete! Results saved.")

if __name__ == "__main__":
    main()
