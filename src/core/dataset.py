import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from enum import Enum
from typing import Any, Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebAFeature(Enum):
    _5_o_Clock_Shadow = "5_o_Clock_Shadow"
    Arched_Eyebrows = "Arched_Eyebrows"
    Attractive = "Attractive"
    Bags_Under_Eyes = "Bags_Under_Eyes"
    Bald = "Bald"
    Bangs = "Bangs"
    Big_Lips = "Big_Lips"
    Big_Nose = "Big_Nose"
    Black_Hair = "Black_Hair"
    Blond_Hair = "Blond_Hair"
    Blurry = "Blurry"
    Brown_Hair = "Brown_Hair"
    Bushy_Eyebrows = "Bushy_Eyebrows"
    Chubby = "Chubby"
    Double_Chin = "Double_Chin"
    Eyeglasses = "Eyeglasses"
    Goatee = "Goatee"
    Gray_Hair = "Gray_Hair"
    Heavy_Makeup = "Heavy_Makeup"
    High_Cheekbones = "High_Cheekbones"
    Male = "Male"
    Mouth_Slightly_Open = "Mouth_Slightly_Open"
    Mustache = "Mustache"
    Narrow_Eyes = "Narrow_Eyes"
    No_Beard = "No_Beard"
    Oval_Face = "Oval_Face"
    Pale_Skin = "Pale_Skin"
    Pointy_Nose = "Pointy_Nose"
    Receding_Hairline = "Receding_Hairline"
    Rosy_Cheeks = "Rosy_Cheeks"
    Sideburns = "Sideburns"
    Smiling = "Smiling"
    Straight_Hair = "Straight_Hair"
    Wavy_Hair = "Wavy_Hair"
    Wearing_Earrings = "Wearing_Earrings"
    Wearing_Hat = "Wearing_Hat"
    Wearing_Lipstick = "Wearing_Lipstick"
    Wearing_Necklace = "Wearing_Necklace"
    Wearing_Necktie = "Wearing_Necktie"
    Young = "Young"


class CelebADataset(Dataset):
    def __init__(
            self, 
            celeba_image_path: str,
            celeba_attr_path: str,
            transform: Optional[Callable] = None, 
            filter_attr: None | CelebAFeature | list[CelebAFeature] = None,
            filter_value: None | bool | list[bool] = None,
            num_calc_samples: Optional[int] = None,
            shuffle: bool = False
        ):
        self.celeba_image_path = celeba_image_path
        self.celeba_attr_path = celeba_attr_path
        self.transform = transform
        self.filter_attr = filter_attr
        self.filter_value = filter_value
        self.num_calc_samples = num_calc_samples
        
        self.attr_df: Optional[pd.DataFrame] = None
        self.image_list: Optional[list] = None
        
        self.attr_df = pd.read_csv(celeba_attr_path)
        if "image_id" not in self.attr_df.columns and len(self.attr_df.columns) > 0:
            self.attr_df.columns = ["image_id"] + list(self.attr_df.columns[1:])
        
        if isinstance(filter_attr, list) and isinstance(filter_value, list):
            mask = pd.Series([True] * len(self.attr_df), index = self.attr_df.index)
            for attr, value in zip(filter_attr, filter_value):
                mask = mask & (self.attr_df[attr.value] == 1 if value else -1)
            filtered = self.attr_df[mask]
            self.image_list = filtered["image_id"].tolist()
        elif isinstance(filter_attr, CelebAFeature) and isinstance(filter_value, bool):
            filtered = self.attr_df[self.attr_df[filter_attr.value] == (1 if filter_value else -1)]
            self.image_list = filtered["image_id"].tolist()
        else:
            self.image_list = self.attr_df["image_id"].tolist()

        if shuffle:
            np.random.shuffle(self.image_list)
        
        if self.num_calc_samples:
            self.image_list = self.image_list[:self.num_calc_samples]
        
        print(f"[Dataset] CelebADataset __init__ success ({len(self.image_list)})")
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(
            self, 
            idx: int
        ) -> tuple[Any, str]:
        img_name = self.image_list[idx]
        img_path = os.path.join(self.celeba_image_path, img_name)
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
    

def __not(
        value: None | bool | list[bool]
    ) -> None | bool | list[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return not bool
    else:
        return list(map(lambda x: not x, value))


def get_celeba_loader(
        celeba_image_path: str,
        celeba_attr_path: str,
        batch_size: int = 64, 
        image_size: int = 64,
        filter_attr: None | CelebAFeature | list[CelebAFeature] = None,
        filter_value: None | bool | list[bool] = None,
        shuffle: bool = True,
        num_calc_samples: Optional[int] = None
    ) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = CelebADataset(
        celeba_image_path = celeba_image_path,
        celeba_attr_path = celeba_attr_path,
        transform = transform,
        filter_attr = filter_attr,
        filter_value = filter_value,
        num_calc_samples = num_calc_samples,
        shuffle = shuffle
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = 8,
        pin_memory = not (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    )

    print(f"[Dataset] get_celeba_loader success {f"({list(map(lambda x: x.value, filter_attr))}={filter_value})" if isinstance(filter_attr, list) and isinstance(filter_value, list) else (f"({filter_attr.value}={filter_value})" if isinstance(filter_attr, CelebAFeature) and isinstance(filter_value, bool) else "")}")
    return dataloader


def get_celeba_loader_set(
        celeba_image_path: str,
        celeba_attr_path: str,
        batch_size: int = 64, 
        image_size: int = 64,
        filter_attr: CelebAFeature = CelebAFeature.Eyeglasses,
        filter_value: bool = True,
        shuffle: bool = False,
        num_calc_samples: Optional[int] = None,
        num_samples: int = 8
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    true_celeba_loader = get_celeba_loader(
        celeba_image_path = celeba_image_path,
        celeba_attr_path = celeba_attr_path,
        batch_size = batch_size,
        image_size = image_size,
        filter_attr = filter_attr,
        filter_value = filter_value,
        shuffle = shuffle,
        num_calc_samples = num_calc_samples
    )
    print(f"[Dataset] true_celeba_loader loaded ({len(true_celeba_loader.dataset)})")

    false_celeba_loader = get_celeba_loader(
        celeba_image_path = celeba_image_path,
        celeba_attr_path = celeba_attr_path,
        batch_size = batch_size,
        image_size = image_size,
        filter_attr = filter_attr,
        filter_value = __not(filter_value),
        shuffle = shuffle,
        num_calc_samples = num_calc_samples
    )
    print(f"[Dataset] false_celeba_loader loaded ({len(false_celeba_loader.dataset)})")

    test_celeba_loader = get_celeba_loader(
        celeba_image_path = celeba_image_path,
        celeba_attr_path = celeba_attr_path,
        batch_size = batch_size,
        image_size = image_size,
        filter_attr = filter_attr,
        filter_value = __not(filter_value),
        shuffle = shuffle,
        num_calc_samples = num_samples
    )
    print(f"[Dataset] test_celeba_loader loaded ({len(test_celeba_loader.dataset)})")

    return true_celeba_loader, false_celeba_loader, test_celeba_loader

