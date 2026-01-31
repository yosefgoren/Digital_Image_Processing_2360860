from pathlib import Path
from typing import List, Tuple
import random
import torch
from torchvision import transforms

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
