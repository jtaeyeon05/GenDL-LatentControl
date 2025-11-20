import torch
from torch.utils.data import DataLoader

from core.model import VAE


def run_vae_attribute_experiment(
        model: VAE,
        true_celeba_loader: DataLoader,
        false_celeba_loader: DataLoader,
        test_celeba_loader: DataLoader,
        output_path: str,
        scale: float,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    pass

