import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from core.model import VAE


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
    transformed_images = []

    with torch.no_grad():
        for test_local_images, _ in tqdm(test_celeba_loader, desc="Transforming test images"):
            test_local_images = test_local_images.to(device)
            transformed_local_images = model.encode(test_local_images)[0] + scale * v_g.unsqueeze(0)
            transformed_local_images = model.decode(transformed_local_images).clamp(0.0, 1.0)

            test_images.append(test_local_images.cpu())
            transformed_images.append(transformed_local_images.cpu())

    test_images = torch.cat(test_images, dim=0)
    transformed_images = torch.cat(transformed_images, dim=0)
    print()
    print(f"[Experiment] apply_attribute_vector success")

    grid = make_grid(torch.cat([test_images, transformed_images]), nrow = len(test_celeba_loader.dataset))
    save_image(grid, output_path)
    print(f"[Experiment] save_image success")

