import torch
from tqdm import tqdm
from core.dataset import CelebAFeature
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from typing import Optional

from core.model import VAE


@dataclass
class DatasetConfig:
    celeba_image_path: str
    celeba_attr_path: str
    custom_dataset_path: Optional[str] = None
    batch_size: int = 64
    image_size: int = 64
    filter_attr: list[CelebAFeature] = field(default_factory = lambda: [CelebAFeature.Eyeglasses])
    filter_value: list[bool] = field(default_factory = lambda: [True])
    shuffle: bool = False
    num_calc_samples: Optional[int] = None
    num_samples: int = 8


def extract_average_latent(
        model: VAE,
        dataloader: DataLoader, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
    all_latents = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Encoding images"):
            images = images.to(device)
            latents, _ = model.encode(images)
            all_latents.append(latents.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    avg_latent = torch.mean(all_latents, dim=0)
    
    return avg_latent


def run_vae_attribute_experiment(
        model: VAE,
        true_celeba_loader: DataLoader,
        false_celeba_loader: DataLoader,
        test_celeba_loader: DataLoader,
        output_path: str,
        scale: float,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    print(f"[Experiment] {"=" * 60}")
    print(f"[Experiment] VAE Attribute Experiment")
    print(f"[Experiment] {"=" * 60}")
    
    model.eval()

    z_1 = extract_average_latent(
        model = model,
        dataloader = true_celeba_loader,
        device = device
    )
    print()
    print(f"[Experiment] extract_average_latent(true_celeba_loader) success")
    z_2 = extract_average_latent(
        model = model,
        dataloader = false_celeba_loader,
        device = device
    )
    print()
    print(f"[Experiment] extract_average_latent(false_celeba_loader) success")
    v_g = z_1 - z_2
    v_g = v_g.to(device)
    
    test_images = []
    reconstructed_images = []
    transformed_images = []

    with torch.no_grad():
        for test_local_images, _ in tqdm(test_celeba_loader, desc="Transforming test images"):
            test_local_images = test_local_images.to(device)
            encoded_local_vectors = model.encode(test_local_images)[0]
            transformed_local_vectors = encoded_local_vectors + scale * v_g.unsqueeze(0)

            test_images.append(test_local_images.cpu())
            reconstructed_images.append(model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu())
            transformed_images.append(model.decode(transformed_local_vectors).clamp(0.0, 1.0).cpu())

    test_images = torch.cat(test_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    transformed_images = torch.cat(transformed_images, dim=0)
    print()
    print(f"[Experiment] apply_attribute_vector success")

    labels = ["Original", "Reconstructed", "Transformed"]
    grid = make_grid(torch.cat([test_images, reconstructed_images, transformed_images]), nrow = len(test_celeba_loader.dataset))
    grid_pil = to_pil_image(grid)

    margin_left = 128
    width, height = grid_pil.size

    result_image = Image.new("RGB", (width + margin_left, height), (255, 255, 255))
    result_image.paste(grid_pil, (margin_left, 0))

    draw = ImageDraw.Draw(result_image)
    font = ImageFont.load_default()
    row_height = height // len(labels)
    for i, label in enumerate(labels):
        y_position = (i * row_height) + (row_height // 2) - 10
        draw.text((10, y_position), label, fill=(0, 0, 0), font=font)
    result_image.save(output_path)
    print(f"[Experiment] save_image success")

