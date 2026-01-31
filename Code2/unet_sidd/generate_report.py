#!/usr/bin/env python3
"""Generate a PDF report from training results."""

from pathlib import Path
import json
import click
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from typing import Optional
from metrics import *


@click.command()
@click.argument('results_idx', type=Optional[int], default=None) #'Results directory number (e.g., 0, 1, 2, ...)'
def generate_report(results_idx: Optional[int]):
    """Generate a PDF report from results/X directory.
    
    The report includes:
    - Training and validation loss plots
    - Runtime performance plots
    - Image triplets (noisy, denoised, clean) for each epoch
    """
    results_dir = get_existing_results_dir(results_idx)
    
    if not results_dir.exists():
        raise click.ClickException(f"Results directory {results_dir} does not exist!")
    
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise click.ClickException(f"Metrics file {metrics_file} does not exist!")
    
    # Load metrics - handle both old format (list) and new format (TrainingRun)
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Check if it's the new TrainingRun format or old list format
    if isinstance(data, dict) and 'epochs' in data and 'specification' in data:
        # New TrainingRun format
        metrics = data['epochs']
        specification = data['specification']
    elif isinstance(data, list):
        # Old format (list of metrics)
        metrics = data
        specification = None
    else:
        raise click.ClickException("Invalid metrics file format!")
    
    if not metrics:
        raise click.ClickException("No metrics found in the JSON file!")
    
    # Sort by epoch to ensure correct order
    metrics = sorted(metrics, key=lambda x: x['epoch'])
    
    # Extract data for plotting
    epochs = [m['epoch'] for m in metrics]
    train_losses = [m['train_loss'] for m in metrics]
    val_losses = [m['val_loss'] for m in metrics]
    epoch_times = [m['epoch_time'] for m in metrics]
    train_times = [m['train_time'] for m in metrics]
    val_times = [m['val_time'] for m in metrics]
    
    # Create PDF
    pdf_path = results_dir / "report.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Set up figure style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('default')
        
        # Figure 1: Training and Validation Loss
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([min(epochs) - 0.5, max(epochs) + 0.5])
        plt.tight_layout()
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Figure 2: Runtime Performance
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(epochs, epoch_times, 'g-o', label='Total Epoch Time', linewidth=2, markersize=6)
        ax2.plot(epochs, train_times, 'b-s', label='Training Time', linewidth=2, markersize=6)
        ax2.plot(epochs, val_times, 'r-^', label='Validation Time', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Runtime Performance', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([min(epochs) - 0.5, max(epochs) + 0.5])
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # Image triplets
        # Find all available epochs from image files
        image_epochs = set()
        for img_file in results_dir.glob("*_noisy.png"):
            epoch_num = int(img_file.stem.split('_')[0])
            image_epochs.add(epoch_num)
        
        image_epochs = sorted(image_epochs)
        
        if not image_epochs:
            click.echo("Warning: No image files found!", err=True)
        else:
            # Create image triplet pages
            # Group epochs into pages (e.g., 2-3 triplets per page)
            triplets_per_page = 2
            num_pages = (len(image_epochs) + triplets_per_page - 1) // triplets_per_page
            
            for page_idx in range(num_pages):
                start_idx = page_idx * triplets_per_page
                end_idx = min(start_idx + triplets_per_page, len(image_epochs))
                page_epochs = image_epochs[start_idx:end_idx]
                
                fig3, axes = plt.subplots(len(page_epochs), 3, figsize=(15, 5 * len(page_epochs)))
                
                # Handle single row case
                if len(page_epochs) == 1:
                    axes = axes.reshape(1, -1)
                
                for row_idx, epoch in enumerate(page_epochs):
                    # Load images
                    noisy_path = results_dir / f"{epoch}_noisy.png"
                    denoised_path = results_dir / f"{epoch}_denoised.png"
                    clean_path = results_dir / f"{epoch}_clean.png"
                    
                    if not all(p.exists() for p in [noisy_path, denoised_path, clean_path]):
                        click.echo(f"Warning: Missing images for epoch {epoch}", err=True)
                        continue
                    
                    noisy_img = Image.open(noisy_path)
                    denoised_img = Image.open(denoised_path)
                    clean_img = Image.open(clean_path)
                    
                    # Display images
                    axes[row_idx, 0].imshow(noisy_img)
                    axes[row_idx, 0].set_title(f'Epoch {epoch} - Noisy', fontsize=11, fontweight='bold')
                    axes[row_idx, 0].axis('off')
                    
                    axes[row_idx, 1].imshow(denoised_img)
                    axes[row_idx, 1].set_title(f'Epoch {epoch} - Denoised', fontsize=11, fontweight='bold')
                    axes[row_idx, 1].axis('off')
                    
                    axes[row_idx, 2].imshow(clean_img)
                    axes[row_idx, 2].set_title(f'Epoch {epoch} - Clean', fontsize=11, fontweight='bold')
                    axes[row_idx, 2].axis('off')
                
                plt.suptitle(f'Image Triplets (Page {page_idx + 1}/{num_pages})', 
                           fontsize=14, fontweight='bold', y=0.995)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                pdf.savefig(fig3, bbox_inches='tight')
                plt.close(fig3)
    
    click.echo(f"Report generated successfully: {pdf_path.absolute()}")


if __name__ == "__main__":
    generate_report()

