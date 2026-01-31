import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from typing import List,Tuple

class SIDDPatchDataset(Dataset):
    def __init__(self, image_pairs: List[Tuple[Path, Path]], patch_size: int, patches_per_image: int):
        self.image_pairs = image_pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_pairs) *self.patches_per_image

    def __getitem__(self, idx: int):
        image_idx = idx//self.patches_per_image
        noisy_path, clean_path = self.image_pairs[image_idx]
        noisy_img= Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        w,h = noisy_img.size
        if w < self.patch_size or h < self.patch_size:
            raise ValueError(f"image {noisy_path.name} ({w}x{h}) is smaller than patch size {self.patch_size}")
        x=random.randint(0,w-self.patch_size)
        y = random.randint(0,h-self.patch_size)
        noisy_patch = noisy_img.crop((x, y, x+ self.patch_size, y + self.patch_size))
        clean_patch = clean_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        noisy_t = self.to_tensor(noisy_patch)
        clean_t = self.to_tensor(clean_patch)
        return noisy_t, clean_t