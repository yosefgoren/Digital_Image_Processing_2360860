#!/usr/bin/env python3
from pathlib import Path
from pydantic import BaseModel
import json
import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
import enum
from torch.utils.data import DataLoader
import resource
from typing import *
import click
from datasets.sidd import SIDDPatchDataset
from torchvision import transforms
from PIL import Image
import time
import platform
from models.unet import UNet
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    max_image_pairs: int | None = None  # None means use all pairs
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

class DatasetPartition(BaseModel):
    train_set: list[tuple[str, str]]
    valid_set: list[tuple[str, str]]
    test_set: list[tuple[str, str]]

class TrainingRun(BaseModel):
    specification: TrainingSpecification
    dataset_partition: DatasetPartition
    epochs: list[EpochMetrics] = []

class OpenResultsMode(enum.Enum):
    NEW = "new"
    LAST = "last"

def save_image_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor image to PNG file."""
    to_pil = transforms.ToPILImage()
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = to_pil(tensor.cpu())
    print(f"Saving image tensor to: {path.name}")
    pil_image.save(path)

def collect_sidd_pairs(root: Path) -> List[Tuple[str, str]]:
    pairs = []

    for scene in root.iterdir():
        noisy = str(next(scene.glob("NOISY_SRGB*.PNG")))
        clean = str(next(scene.glob("GT_SRGB*.PNG")))
        pairs.append((noisy, clean))

    return pairs

def split_dataset(
    pairs: List[Tuple[str, str]],
    test_images: int = 25,
    val_ratio: float = 0.1,
):
    random.shuffle(pairs)

    test = pairs[:test_images]
    remaining = pairs[test_images:]

    val_size = int(len(remaining) * val_ratio)
    val = remaining[:val_size]
    train = remaining[val_size:]

    return train, val, test

class TrainingDB:
    def _load_training_spec(self) -> TrainingSpecification:
        spec_file = Path("training_specification.json")
        
        if not spec_file.exists():
            raise FileNotFoundError(f"Missing specification file '{spec_file}' not found. Please create and fill it.")
        
        with open(spec_file, 'r') as f:
            spec_dict = json.load(f)
        
        return TrainingSpecification(**spec_dict)

    def get_metrics_path(self) -> Path:
        return self.get_resource_path("metrics.json")

    def _load_previous_training_run(self) -> TrainingRun:
        metrics_file = self.get_metrics_path()
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found in {self._results_dir}")
        
        with open(metrics_file, 'r') as f:
            return TrainingRun.model_validate_json(f.read())

    def _init_device(self) -> None:
        spec = self.run.specification
        self.device = torch.device(spec.device)
        if spec.device == "xpu":
            assert torch.xpu.is_available()
    
    def _init_model(self) -> None:
        self.model = UNet().to(self.device)

    def _init_criterion(self) -> None:
        spec = self.run.specification
        loss_functions = {
            "L1Loss": nn.L1Loss,
            "MSELoss": nn.MSELoss,
        }
        if not spec.loss_function in loss_functions:
            raise ValueError(f"Unsupported loss function: {spec.loss_function} (add it to dict?)")
        
        self.criterion = loss_functions[spec.loss_function]()

    def _init_optimizer(self) -> None:
        spec = self.run.specification
        if spec.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=spec.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {spec.optimizer}")

    def _init_from_existing_run_idx(self, run_idx: int) -> None:
        print(f"Continuing training of run: {run_idx}")
        self._results_dir = get_existing_results_dir(run_idx)            
        self.run = self._load_previous_training_run()
        self.start_epoch = max([0, len(self.run.epochs)])

    def _init_new_run(self) -> None:
        spec = self._load_training_spec()
        print(f"Starting new training with specification: {spec.model_dump_json(indent=2)}")

        pairs = collect_sidd_pairs(Path(spec.dataset_path))
        if spec.max_image_pairs is not None:
            pairs = pairs[:spec.max_image_pairs]
        train_pairs, valid_pairs, test_pairs = split_dataset(pairs)
        
        self._results_dir = get_next_results_dir()
        self.run = TrainingRun(
            specification=spec,
            dataset_partition=DatasetPartition(
                train_set=train_pairs,
                valid_set=valid_pairs,
                test_set=test_pairs
            )
        )
        self.start_epoch = 0

    def __init__(self, run_spec: int | OpenResultsMode):
        if isinstance(run_spec, int):
            self._init_from_existing_run_idx(run_spec)
        else:
            if run_spec == OpenResultsMode.NEW:
                self._init_new_run()
            elif run_spec == OpenResultsMode.LAST:
                last_idx = get_max_results_idx(False)
                self._init_from_existing_run_idx(last_idx)
            else:
                raise RuntimeError(f"run specification: {run_spec}")

        self._init_device()
        self._init_model()
        self._init_criterion()
        self._init_optimizer()

    def get_resource_path(self, filename: str) -> Path:
        return self._results_dir / filename      

    def get_weights_path(self) -> Path:
        return self.get_resource_path("unet_sidd.pth")

    def save_checkpoint(self, model: torch.nn.Module):
        model_path = self.get_weights_path()
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to: {model_path}")

        with open(self.get_metrics_path(), "w") as f:
            json.dump(self.run.model_dump(), f, indent=2)


def get_results_basedir() -> str:
    return "./results"

def get_max_results_idx(zero_if_empty: bool = True) -> int:
    results_base = Path(get_results_basedir())
    results_base.mkdir(exist_ok=True)
    options = [int(entry.name) for entry in results_base.iterdir() if entry.name.isdecimal()]
    if not zero_if_empty and len(options) == 0:
        raise RuntimeError(f"Expected to have at-least 1 results dir but found none.")
        
    return max(options)

def get_existing_results_dir(results_idx: int | None = None) -> Path:
    if results_idx is None:
        results_idx = get_max_results_idx()

    res = Path(f"{get_results_basedir()}") / f"{results_idx}"
    assert res.is_dir()
    return res

def get_next_results_dir() -> Path:
    results_dir = Path(f"{get_results_basedir()}") / f"{get_max_results_idx()+1}"
    results_dir.mkdir(exist_ok=True)
    return results_dir

@click.group()
class cli:
    pass


# * * * Training * * *

def hard_limit_memory_usage_linux():
    memory_limit_gb = 32
    memory_limit_bytes = memory_limit_gb * (1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    print(f"Memory limit set to {memory_limit_gb}GB - process will be killed if exceeded")
    
    oom_score_adj = 1000  # High priority for OOM killer (range: -1000 to 1000)
    with open(f"/proc/self/oom_score_adj", "w") as f:
        f.write(str(oom_score_adj))
    print(f"OOM score adjusted to {oom_score_adj} (higher = more likely to be killed)")
    
def hard_limit_memory_usage():
    os_name = platform.system().lower().strip()
    if os_name == "windows":
        print("Windows: Not limiting memory usage.")
        pass
    elif os_name == "linux":
        print("Linux: Limiting memory usage.")
        hard_limit_memory_usage_linux()
    else:
        print(f"Unknown or unsupported operating system: `{os_name}`")
    

def get_center_patch_from_image_path(path: Path, patch_size: int) -> torch.Tensor:
    sample_image = Image.open(path).convert("RGB")

    to_tensor = transforms.ToTensor()
    sample_tensor = to_tensor(sample_image)
    
    _, h, w = sample_tensor.shape
    
    center_x = (w - patch_size) // 2
    center_y = (h - patch_size) // 2
    return sample_tensor[:, center_y:center_y+patch_size, center_x:center_x+patch_size]


@cli.command("train")
@click.option('--resume', type=int, default=None, help='Resume training from results directory with this index')
def train(resume: Optional[int]):
    hard_limit_memory_usage()
    db = TrainingDB(OpenResultsMode.NEW if resume is None else resume)
    spec = db.run.specification
    dataset_partition = db.run.dataset_partition

    sample_noisy_path, sample_clean_path = dataset_partition.test_set[0]
    sample_noisy_patch = get_center_patch_from_image_path(sample_noisy_path, spec.patch_size)
    sample_clean_patch = get_center_patch_from_image_path(sample_clean_path, spec.patch_size)

    train_ds = SIDDPatchDataset(dataset_partition.train_set, spec.patch_size, spec.patches_per_image)
    val_ds = SIDDPatchDataset(dataset_partition.valid_set, spec.patch_size, spec.patches_per_image)

    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=spec.batch_size)

    # Load weights if resuming
    if resume is not None:
        model_path = db.get_weights_path()
        if model_path.exists():
            db.model.load_state_dict(torch.load(model_path, map_location=db.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}, starting with fresh weights")
    
    # Training loop
    for epoch in range(db.start_epoch, spec.epochs):
        epoch_start_time = time.time()
        
        db.model.train()
        train_loss = 0.0
        total_load_time = 0.0
        total_h2d_time = 0.0
        total_step_time = 0.0

        train_start_time = time.time()
        t0 = time.time()
        for noisy, clean in train_loader:
            t1 = time.time()
            noisy, clean = noisy.to(db.device, non_blocking=True), clean.to(db.device, non_blocking=True)
            t2 = time.time()

            db.optimizer.zero_grad()
            output = db.model(noisy)
            loss = db.criterion(output, clean)
            loss.backward()
            db.optimizer.step()
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

        db.model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(db.device, non_blocking=True), clean.to(db.device, non_blocking=True)

                output = db.model(noisy)
                val_loss += db.criterion(output, clean).item()
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

        db.model.eval()
        with torch.no_grad():
            sample_noisy_batch = sample_noisy_patch.unsqueeze(0).to(db.device)
            sample_denoised_batch = db.model(sample_noisy_batch)
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
            db.save_checkpoint(db.model)
            print(f"Checkpoint saved at epoch {epoch}")

    db.save_checkpoint(db.model)
    print(f"Training complete! Results saved.")


# * * * Test * * *

def denoise_full_image(model: nn.Module, noisy_tensor: torch.Tensor, device: torch.device, patch_size: int) -> torch.Tensor:
    """Denoise a full image by processing it in overlapping patches.
    
    Uses a sliding window approach with overlap to avoid boundary artifacts.
    """
    model.eval()
    original_shape = noisy_tensor.shape
    _, h, w = noisy_tensor.shape
    
    # If image is smaller than patch_size, pad it
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        noisy_tensor = torch.nn.functional.pad(noisy_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, h, w = noisy_tensor.shape
    
    # Create output tensor
    output = torch.zeros_like(noisy_tensor)
    count = torch.zeros_like(noisy_tensor)
    
    # Sliding window with 50% overlap
    stride = patch_size // 2
    
    with torch.no_grad():
        # Generate all patch positions
        y_positions = list(range(0, h - patch_size + 1, stride))
        x_positions = list(range(0, w - patch_size + 1, stride))
        
        # Ensure we cover the edges
        if y_positions[-1] < h - patch_size:
            y_positions.append(h - patch_size)
        if x_positions[-1] < w - patch_size:
            x_positions.append(w - patch_size)
        
        for y in y_positions:
            for x in x_positions:
                # Extract patch
                patch = noisy_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
                
                # Denoise patch
                denoised_patch = model(patch)
                
                # Add to output (with accumulation for overlapping regions)
                output[:, y:y+patch_size, x:x+patch_size] += denoised_patch.squeeze(0).cpu()
                count[:, y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping regions
        output = output / torch.clamp(count, min=1)
    
    # Remove padding if it was added
    if original_shape[1] != h or original_shape[2] != w:
        output = output[:, :original_shape[1], :original_shape[2]]
    
    return output


def extract_center_patch(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extract center patch from a tensor image."""
    _, h, w = tensor.shape
    center_x = (w - patch_size) // 2
    center_y = (h - patch_size) // 2
    return tensor[:, center_y:center_y+patch_size, center_x:center_x+patch_size]


def load_tensor_png(image_path: str) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    return img, transforms.ToTensor()(img)

NUM_TEST_IMAGES_SHOWN = 3

def create_test_results(db: TrainingDB, test_loss_values: list[float]) -> None:
    show_image_pairs = db.run.dataset_partition.test_set[:NUM_TEST_IMAGES_SHOWN]
    pdf_path = db.get_resource_path("test_results.pdf")
    patch_size = db.run.specification.patch_size
    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: Test loss summary ----
        fig, ax = plt.subplots(figsize=(10, 5))

        x = range(1, len(test_loss_values) + 1)
        mean_loss = sum(test_loss_values) / len(test_loss_values)

        ax.plot(x, test_loss_values, marker='o', linewidth=2, label='Test Loss')
        ax.axhline(mean_loss, linestyle='--', linewidth=2, label=f'Mean = {mean_loss:.4f}')

        # Annotate mean value
        ax.text(
            0.99, mean_loss,
            f'{mean_loss:.4f}',
            transform=ax.get_yaxis_transform(),
            ha='right', va='bottom',
            fontsize=10, fontweight='bold'
        )

        ax.set_title('Test Loss Values', fontsize=14, fontweight='bold')
        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Process each image
        for img_idx, (noisy_path, clean_path) in enumerate(show_image_pairs):
            print(f"Processing image {img_idx + 1}/{len(show_image_pairs)}: {noisy_path}")
            
            noisy_img, noisy_tensor = load_tensor_png(noisy_path)
            clean_img, clean_tensor = load_tensor_png(clean_path)
            
            denoised_tensor = denoise_full_image(db.model, noisy_tensor, db.device, patch_size)
            
            noisy_patch = extract_center_patch(noisy_tensor, patch_size)
            denoised_patch = extract_center_patch(denoised_tensor, patch_size)
            clean_patch = extract_center_patch(clean_tensor, patch_size)
            
            # Create figure with 2 rows: full images and patches
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Full images
            axes[0, 0].imshow(noisy_img)
            axes[0, 0].set_title(f'Image {img_idx + 1} - Noisy (Full)', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Convert denoised tensor to PIL for display
            denoised_pil = transforms.ToPILImage()(torch.clamp(denoised_tensor, 0, 1))
            axes[0, 1].imshow(denoised_pil)
            axes[0, 1].set_title(f'Image {img_idx + 1} - Denoised (Full)', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(clean_img)
            axes[0, 2].set_title(f'Image {img_idx + 1} - Clean (Full)', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Row 2: Center patches
            noisy_patch_pil = transforms.ToPILImage()(torch.clamp(noisy_patch, 0, 1))
            axes[1, 0].imshow(noisy_patch_pil)
            axes[1, 0].set_title(f'Image {img_idx + 1} - Noisy (Center Patch)', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            denoised_patch_pil = transforms.ToPILImage()(torch.clamp(denoised_patch, 0, 1))
            axes[1, 1].imshow(denoised_patch_pil)
            axes[1, 1].set_title(f'Image {img_idx + 1} - Denoised (Center Patch)', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            clean_patch_pil = transforms.ToPILImage()(torch.clamp(clean_patch, 0, 1))
            axes[1, 2].imshow(clean_patch_pil)
            axes[1, 2].set_title(f'Image {img_idx + 1} - Clean (Center Patch)', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Test Results - Image {img_idx + 1}', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\nTest results saved to: {pdf_path}")

def evaluate_model(db: TrainingDB) -> list[float]:
    loss_values = []

    spec = db.run.specification
    test_ds = SIDDPatchDataset(db.run.dataset_partition.test_set, spec.patch_size, spec.patches_per_image)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(db.device, non_blocking=True), clean.to(db.device, non_blocking=True)
            output = db.model(noisy)
            loss = db.criterion(output, clean)
            loss_values.append(loss.item())
    
    assert len(loss_values) > 0
    return loss_values

@cli.command("test")
@click.option('--results_idx', type=int, default=None, help='Results directory index (default: latest)')
def test(results_idx: Optional[int]):
    db = TrainingDB(OpenResultsMode.LAST if results_idx is None else results_idx)
    spec = db.run.specification
    
    model_path = db.get_weights_path()
    if not model_path.exists():
        raise click.ClickException(f"Model weights not found at {model_path}")
    
    if spec.device == "xpu":
        assert torch.xpu.is_available(), "XPU not available!"
    
    device = torch.device(spec.device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully")

    test_pairs = db.run.dataset_partition.test_set
    print(f"Processing {len(test_pairs)} images...")
    
    test_loss_values = evaluate_model(db)
    create_test_results(db, test_loss_values)

# * * * Common * * *

if __name__ == "__main__":
    cli()

