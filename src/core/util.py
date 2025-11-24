import os
import torch

from core.dataset import CelebAFeature


def __load_latent_vector(
        latent_vector_path: str
    ) -> torch.Tensor:
    return torch.load(latent_vector_path, map_location='cpu')


def load_latent_vector(
        latent_vector_path: str,
        latent_vector_name: str,
        filter_attr: CelebAFeature,
        filter_value: bool
) -> torch.Tensor:
    save_name = f"{filter_attr.value}_{filter_value}"
    save_path = os.path.join(latent_vector_path, f"{latent_vector_name}_{save_name}.pt")
    return __load_latent_vector(save_path)

