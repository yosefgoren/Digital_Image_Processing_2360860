from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from datasets.sidd import SIDDPatchDataset
from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
import time

PATCH_SIZE = 128
PATCHES_PER_IMAGE = 2
BATCH_SIZE = 64
EPOCHS = 32


def main():
    root = Path(f"{os.path.dirname(__file__)}/datasets/sidd_tmpfs/Data")

    pairs = collect_sidd_pairs(root)
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
