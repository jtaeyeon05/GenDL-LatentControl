import os
import torch
from torchvision.utils import make_grid, save_image

import config
from core.dataset import get_celeba_loader
from core.model import VAE, get_vae_model


def reconstuct_test(
        model: VAE,
        celeba_loader: torch.utils.data.DataLoader,
        output_path: str,
        num_samples: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    images = next(iter(celeba_loader))[0][:num_samples].to(device)
    with torch.no_grad():
        enc = model.encode(images)
        mu, logvar = enc[0], enc[1]
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z).clamp(0.0, 1.0)
    grid = make_grid(torch.cat([images.cpu(), recon.cpu()]), nrow=num_samples)
    save_image(grid, output_path)
    print(f"[Test] Test Success ({output_path})")


def repeat_reconstuct_test(
        model: VAE,
        celeba_loader: torch.utils.data.DataLoader,
        output_path: str,
        num_samples: int,
        num_repeats: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    images = [next(iter(celeba_loader))[0][:num_samples].to(device)]
    with torch.no_grad():
        for _ in range(num_repeats):
            enc = model.encode(images[-1])
            mu, logvar = enc[0], enc[1]
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z).clamp(0.0, 1.0)
            images += [recon]
    grid = make_grid(torch.cat(images).cpu(), nrow=num_samples)
    save_image(grid, output_path)
    print(f"[Test] Test Success ({output_path})")


def test() -> None:
    if not os.path.exists(config.model_path):
        print("[Main] model_path does not exist")
        return
    model = get_vae_model(
        model_path = config.model_path, 
        model_latent_dim = config.model_latent_dim,
        image_size = config.image_size,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    if not os.path.exists(config.celeba_image_path) or not os.path.exists(config.celeba_attr_path):
        print("[Test] celeba_image_path or celeba_attr_path  does not exist")
        return
    celeba_loader = get_celeba_loader(
        celeba_image_path = config.celeba_image_path,
        celeba_attr_path = config.celeba_attr_path,
        batch_size = config.batch_size,
        image_size = config.image_size,
        shuffle = config.shuffle
    )

    if not os.path.exists(config.output_path):
        print("[Test] output_path does not exist")
        return
    """
    reconstuct_test(
        model = model,
        celeba_loader = celeba_loader,
        output_path = os.path.join(config.output_path, 'test_tmp.png'),
        num_samples = config.num_samples,
        device = config.device
    )
    """
    repeat_reconstuct_test(
        model = model,
        celeba_loader = celeba_loader,
        output_path = os.path.join(config.output_path, 'test_tmp.png'),
        num_samples = config.num_samples,
        num_repeats = 50,
        device = config.device
    )


if __name__ == '__main__':
    test()

