#!/usr/bin/env python3
"""Test a trained model on full images."""

from pathlib import Path
import torch
import torch.nn as nn
import click
from typing import Optional
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from PIL import Image
import numpy as np

from models.unet import UNet
from utils import collect_sidd_pairs, split_dataset
from metrics import get_existing_results_dir, TrainingRun


def save_image_tensor(tensor: torch.Tensor, path: Path):
    """Save a tensor image to PNG file."""
    to_pil = transforms.ToPILImage()
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = to_pil(tensor.cpu())
    pil_image.save(path)


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


@click.command()
@click.option('--results_idx', type=int, default=None, help='Results directory index (default: latest)')
@click.option('--num_images', type=int, default=5, help='Number of test images to process (default: 5)')
def test_model(results_idx: Optional[int], num_images: int):
    results_dir = get_existing_results_dir(results_idx)
    print(f"Loading model from: {results_dir}")
    
    # Load training run to get specification
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise click.ClickException(f"Metrics file {metrics_file} does not exist!")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'specification' in data:
        spec_dict = data['specification']
        patch_size = spec_dict['patch_size']
        dataset_path = spec_dict['dataset_path']
        device_name = spec_dict.get('device', 'xpu')
    else:
        raise click.ClickException("Could not find training specification in metrics.json")
    
    print(f"Patch size: {patch_size}")
    print(f"Dataset path: {dataset_path}")
    
    # Load model weights
    model_path = results_dir / "unet_sidd.pth"
    if not model_path.exists():
        raise click.ClickException(f"Model weights not found at {model_path}")
    
    device = torch.device(device_name)
    if device_name == "xpu":
        assert torch.xpu.is_available(), "XPU not available!"
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully")
    
    # Load test images
    root = Path(dataset_path)
    pairs = collect_sidd_pairs(root)
    _, _, test_pairs = split_dataset(pairs)
    
    # Use test images, or fall back to all pairs if test set is small
    if len(test_pairs) < num_images:
        print(f"Warning: Only {len(test_pairs)} test images available, using all pairs")
        test_pairs = pairs[:num_images]
    else:
        test_pairs = test_pairs[:num_images]
    
    print(f"Processing {len(test_pairs)} images...")
    
    # Prepare image processing
    to_tensor = transforms.ToTensor()
    
    # Create PDF
    pdf_path = results_dir / "test_results.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Process each image
        for img_idx, (noisy_path, clean_path) in enumerate(test_pairs):
            print(f"Processing image {img_idx + 1}/{len(test_pairs)}: {noisy_path.name}")
            
            # Load full images
            noisy_img = Image.open(noisy_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")
            
            noisy_tensor = to_tensor(noisy_img)
            clean_tensor = to_tensor(clean_img)
            
            # Denoise full image
            denoised_tensor = denoise_full_image(model, noisy_tensor, device, patch_size)
            
            # Extract center patches
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
    
    print(f"\nTest results saved to: {pdf_path.absolute()}")


if __name__ == "__main__":
    test_model()
