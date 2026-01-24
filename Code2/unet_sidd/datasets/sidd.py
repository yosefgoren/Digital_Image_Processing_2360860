from __future__ import annotations
from typing import List, Tuple
import random
from pathlib import Path
import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SIDDPatchDataset(Dataset):
    def __init__(
        self,
        image_pairs: List[Tuple[Path, Path]],
        patch_size: int,
        patches_per_image: int,
    ) -> None:
        self.image_pairs = image_pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        self.to_tensor = transforms.ToTensor()
        
        # Preload all images into tensors at initialization
        print(f"Preloading {len(image_pairs)} image pairs into memory...")
        self.noisy_images = []
        self.clean_images = []
        
        for noisy_path, clean_path in tqdm.tqdm(image_pairs):
            noisy = self.to_tensor(Image.open(noisy_path).convert("RGB"))
            clean = self.to_tensor(Image.open(clean_path).convert("RGB"))
            self.noisy_images.append(noisy)
            self.clean_images.append(clean)
        
        print(f"Finished preloading {len(image_pairs)} image pairs.")

    def _extract_patches(self, img: Tensor) -> List[Tensor]:
        _, h, w = img.shape
        ps = self.patch_size

        xs = list(range(0, w - ps + 1, ps))
        ys = list(range(0, h - ps + 1, ps))

        coords = [(x, y) for x in xs for y in ys]
        random.shuffle(coords)

        patches = []
        for x, y in coords[: self.patches_per_image]:
            patches.append(img[:, y:y+ps, x:x+ps])

        return patches

    def __len__(self) -> int:
        return len(self.image_pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_idx = idx // self.patches_per_image

        # Use preloaded tensors instead of loading from disk
        noisy = self.noisy_images[img_idx]
        clean = self.clean_images[img_idx]

        noisy_patches = self._extract_patches(noisy)
        clean_patches = self._extract_patches(clean)

        patch_idx = idx % self.patches_per_image
        return noisy_patches[patch_idx], clean_patches[patch_idx]
