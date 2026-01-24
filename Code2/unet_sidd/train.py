#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import resource
import json
from typing import List
from datasets.sidd import SIDDPatchDataset
from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
from torchvision import transforms
from PIL import Image
import time
from metrics import EpochMetrics, get_next_results_dir

PATCH_SIZE = 128
PATCHES_PER_IMAGE = 32
BATCH_SIZE = 16
EPOCHS = 32


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

def main():
    hard_limit_memory_usage()
    
    root = Path(get_dataset_basepath())

    pairs = collect_sidd_pairs(root)[:100]
    train_pairs, val_pairs, _ = split_dataset(pairs)

    # Get results directory
    results_dir = get_next_results_dir()
    print(f"Saving results to: {results_dir}")

    # Load sample image (pairs[0]) for visualization
    # Resize to a reasonable size to avoid GPU OOM (full images are too large)
    VISUALIZATION_SIZE = 512  # Resize to 512x512 for visualization
    sample_noisy_path, sample_clean_path = pairs[0]
    to_tensor = transforms.ToTensor()
    resize_transform = transforms.Resize((VISUALIZATION_SIZE, VISUALIZATION_SIZE), antialias=True)
    
    sample_noisy_img = Image.open(sample_noisy_path).convert("RGB")
    sample_clean_img = Image.open(sample_clean_path).convert("RGB")
    
    # Resize and convert to tensor
    sample_noisy_full = to_tensor(resize_transform(sample_noisy_img))
    sample_clean_full = to_tensor(resize_transform(sample_clean_img))

    train_ds = SIDDPatchDataset(train_pairs, PATCH_SIZE, PATCHES_PER_IMAGE)
    val_ds = SIDDPatchDataset(val_pairs, PATCH_SIZE, PATCHES_PER_IMAGE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("xpu")
    assert torch.xpu.is_available()

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # List to store all epoch metrics
    all_metrics: List[EpochMetrics] = []

    for epoch in range(EPOCHS):
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
            f"Epoch {epoch+1:03d} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f}"
        )

        # Generate sample denoised image
        model.eval()
        # Clear GPU cache before processing sample image to free up memory
        torch.xpu.empty_cache()
        with torch.no_grad():
            # Add batch dimension and move to device
            sample_noisy_batch = sample_noisy_full.unsqueeze(0).to(device)
            sample_denoised_batch = model(sample_noisy_batch)
            sample_denoised_full = sample_denoised_batch.squeeze(0).cpu()
            # Clear cache after processing
            torch.xpu.empty_cache()

        # Save sample images
        save_image_tensor(sample_noisy_full, results_dir / f"{epoch+1}_noisy.png")
        save_image_tensor(sample_denoised_full, results_dir / f"{epoch+1}_denoised.png")
        save_image_tensor(sample_clean_full, results_dir / f"{epoch+1}_clean.png")

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

        # Save JSON file after each epoch (overwrite with updated list)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump([m.model_dump() for m in all_metrics], f, indent=2)

    torch.save(model.state_dict(), "unet_sidd.pth")
    print(f"\nTraining complete! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
