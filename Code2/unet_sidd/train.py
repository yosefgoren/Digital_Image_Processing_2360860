from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import resource
from datasets.sidd import SIDDPatchDataset
from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
import time

PATCH_SIZE = 128
PATCHES_PER_IMAGE = 4
BATCH_SIZE = 64
EPOCHS = 32

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

    train_ds = SIDDPatchDataset(train_pairs, PATCH_SIZE, PATCHES_PER_IMAGE)
    val_ds = SIDDPatchDataset(val_pairs, PATCH_SIZE, PATCHES_PER_IMAGE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("xpu")
    assert torch.xpu.is_available()

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

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

            print(f"load: {t1-t0:.3f}s | h2d: {t2-t1:.3f}s | step: {t3-t2:.3f}s")

            t0 = time.time()
        print(f"load: {t1-t0:.3f}s | h2d: {t2-t1:.3f}s | step: {t3-t2:.3f}s")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

                output = model(noisy)
                val_loss += criterion(output, clean).item()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {train_loss/len(train_loader):.4f} | "
            f"Val {val_loss/len(val_loader):.4f}"
        )

    torch.save(model.state_dict(), "unet_sidd.pth")


if __name__ == "__main__":
    main()
