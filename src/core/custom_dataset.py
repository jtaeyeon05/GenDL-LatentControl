import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(
            self,
            custom_dataset_path: str,
            transform: Optional[Callable] = None,
            num_calc_samples: Optional[int] = None,
            shuffle: bool = False
    ):
        self.custom_dataset_path = custom_dataset_path
        self.transform = transform
        self.num_calc_samples = num_calc_samples

        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        self.image_list = [
            f for f in os.listdir(custom_dataset_path)
            if f.lower().endswith(self.valid_extensions)
        ]

        if shuffle:
            np.random.shuffle(self.image_list)

        if self.num_calc_samples:
            self.image_list = self.image_list[:min(self.num_calc_samples, len(self.image_list))]

        print(f"[CustomDataset] CustomDataset __init__ success ({len(self.image_list)})")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[Any, str]:
        img_name = self.image_list[idx]
        img_path = os.path.join(self.custom_dataset_path, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def get_custom_dataset_loader(
        custom_dataset_path: str,
        batch_size: int = 64,
        image_size: int = 64,
        shuffle: bool = True,
        num_calc_samples: Optional[int] = None
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(
        custom_dataset_path = custom_dataset_path,
        transform = transform,
        num_calc_samples = num_calc_samples,
        shuffle = shuffle
    )

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 8,
        pin_memory = not (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    )

    print(f"[CustomDataset] get_custom_dataset_loader success")
    return dataloader

