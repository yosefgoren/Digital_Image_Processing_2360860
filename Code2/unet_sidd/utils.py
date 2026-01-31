from pathlib import Path
from typing import List, Tuple
import random


def collect_sidd_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []

    for scene in root.iterdir():
        noisy = next(scene.glob("NOISY_SRGB*.PNG"))
        clean = next(scene.glob("GT_SRGB*.PNG"))
        pairs.append((noisy, clean))

    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
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
