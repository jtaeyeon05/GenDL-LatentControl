import torch
from tqdm import tqdm
from core.custom_dataset import get_custom_dataset_loader
from core.dataset import CelebAFeature, get_celeba_loader
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from core.model import VAE


@dataclass
class DatasetConfig:
    celeba_image_path: str
    celeba_attr_path: str
    custom_dataset_path: Optional[str] = None
    batch_size: int = 64
    image_size: int = 64
    shuffle: bool = False
    num_calc_samples: Optional[int] = None
    num_samples: int = 8


@dataclass
class ExperimentConfig:
    filter_attr: list[CelebAFeature] = field(default_factory = lambda: [CelebAFeature.Eyeglasses])
    filter_value: list[bool] = field(default_factory = lambda: [True])
    scale: list[float] = field(default_factory= lambda: [1.0])
    output_path: str = "./test.png"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def save_result_image(
        images: torch.Tensor,
        labels: list[str],
        image_size: int,
        nrow: int,
        output_path: str
    ) -> None:
    white_images = torch.ones(len(labels), 1, 3, image_size, image_size).to(images.device)
    reshaped_images = images.view(len(labels), nrow, 3, image_size, image_size)
    final_images_tensor = torch.cat([white_images, reshaped_images], dim=1)
    final_images_tensor = final_images_tensor.view(-1, 3, image_size, image_size)

    grid = make_grid(final_images_tensor, nrow = nrow + 1)
    grid_pil = to_pil_image(grid)

    width, height = grid_pil.size
    result_image = Image.new("RGB", (width, height), (255, 255, 255))
    result_image.paste(grid_pil, (0, 0))

    draw = ImageDraw.Draw(result_image)
    font = ImageFont.load_default()
    for i, label in enumerate(labels):
        y_position = i * height // len(labels)
        draw.text((10, y_position + 10), label, fill=(0, 0, 0), font=font)

    result_image.save(output_path)
    print(f"[Experiment] save_image success")


def __not(
        value: None | bool | list[bool]
) -> None | bool | list[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return not value
    else:
        return list(map(lambda x: not x, value))


def run_vae_synthesized_attribute_experiment(
        model: VAE,
        dataset_config: DatasetConfig,
        experiment_config: ExperimentConfig
) -> None:
    print(f"[Experiment] {"=" * 60}")
    print(f"[Experiment] VAE Synthesized Attribute Experiment")
    print(f"[Experiment] {"=" * 60}")

    filter_length = len(experiment_config.filter_attr)
    if len(experiment_config.filter_attr) != len(experiment_config.filter_value) \
            or len(experiment_config.filter_attr) != len(experiment_config.scale) \
            or len(experiment_config.filter_attr) < 1:
        print(f"[Experiment] filter_attr length is wrong")
        return

    true_celeba_loader = get_celeba_loader(
        celeba_image_path = dataset_config.celeba_image_path,
        celeba_attr_path = dataset_config.celeba_attr_path,
        batch_size = dataset_config.batch_size,
        image_size = dataset_config.image_size,
        filter_attr = experiment_config.filter_attr,
        filter_value = experiment_config.filter_value,
        shuffle = dataset_config.shuffle,
        num_calc_samples = dataset_config.num_calc_samples
    )
    print(f"[Experiment] true_celeba_loader loaded ({len(true_celeba_loader.dataset)})")

    false_celeba_loader = get_celeba_loader(
        celeba_image_path = dataset_config.celeba_image_path,
        celeba_attr_path = dataset_config.celeba_attr_path,
        batch_size = dataset_config.batch_size,
        image_size = dataset_config.image_size,
        filter_attr = experiment_config.filter_attr,
        filter_value = __not(experiment_config.filter_value),
        shuffle = dataset_config.shuffle,
        num_calc_samples = dataset_config.num_calc_samples
    )
    print(f"[Experiment] false_celeba_loader loaded ({len(false_celeba_loader.dataset)})")

    if dataset_config.custom_dataset_path is None:
        test_dataset_loader = get_celeba_loader(
            celeba_image_path = dataset_config.celeba_image_path,
            celeba_attr_path = dataset_config.celeba_attr_path,
            batch_size = dataset_config.batch_size,
            image_size = dataset_config.image_size,
            filter_attr = experiment_config.filter_attr,
            filter_value = __not(experiment_config.filter_value),
            shuffle = dataset_config.shuffle,
            num_calc_samples = dataset_config.num_samples
        )
        print(f"[Experiment] test_dataset_loader loaded ({len(test_dataset_loader.dataset)})")
    else:
        test_dataset_loader = get_custom_dataset_loader(
            custom_dataset_path = dataset_config.custom_dataset_path,
            batch_size = dataset_config.batch_size,
            image_size = dataset_config.image_size,
            shuffle = dataset_config.shuffle,
            num_calc_samples = dataset_config.num_calc_samples,
        )
        print(f"[Experiment] test_dataset_loader loaded ({len(test_dataset_loader.dataset)})")

    model.eval()

    true_vector = extract_average_latent(
        model = model,
        dataloader = true_celeba_loader,
        device = experiment_config.device
    )
    print(f"\r[Experiment] extract_average_latent(true_celeba_loader) success")
    false_vector = extract_average_latent(
        model = model,
        dataloader = false_celeba_loader,
        device = experiment_config.device
    )
    print(f"\r[Experiment] extract_average_latent(false_celeba_loader) success")

    latent_vector = true_vector - false_vector
    latent_vector = latent_vector.to(experiment_config.device)
    print(f"\r[Experiment] calculate latent_vector success")

    test_images = []
    reconstructed_images = []
    transformed_images = []

    with torch.no_grad():
        for test_local_images, _ in tqdm(test_dataset_loader, desc="Transforming test images"):
            test_local_images = test_local_images.to(experiment_config.device)
            encoded_local_vectors = model.encode(test_local_images)[0]
            transformed_local_vectors = encoded_local_vectors + sum(experiment_config.scale) / filter_length * latent_vector.unsqueeze(0)

            test_images.append(test_local_images.cpu())
            reconstructed_images.append(model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu())
            transformed_images.append(model.decode(transformed_local_vectors).clamp(0.0, 1.0).cpu())

    test_images = torch.cat(test_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    transformed_images = torch.cat(transformed_images, dim=0)
    print(f"\r[Experiment] apply_attribute_vector success")

    labels = [
        "Original",
        "Reconstructed",
        f"Transformed (*{sum(experiment_config.scale) / filter_length})\n" +
        f"{
            "\n".join(
                [
                    f"{experiment_config.filter_attr[i].value}={experiment_config.filter_value[i]}"
                    for i in range(filter_length)
                ]
            )
        }"
    ]

    save_result_image(
        images = torch.cat([test_images, reconstructed_images, transformed_images]),
        labels = labels,
        image_size = dataset_config.image_size,
        nrow = len(test_dataset_loader.dataset),
        output_path = experiment_config.output_path
    )


def run_vae_multi_attribute_experiment(
        model: VAE,
        dataset_config: DatasetConfig,
        experiment_config: ExperimentConfig
) -> None:
    print(f"[Experiment] {"=" * 60}")
    print(f"[Experiment] VAE Multi Attribute Experiment")
    print(f"[Experiment] {"=" * 60}")

    filter_length = len(experiment_config.filter_attr)
    if len(experiment_config.filter_attr) != len(experiment_config.filter_value) \
            or len(experiment_config.filter_attr) != len(experiment_config.scale) \
            or len(experiment_config.filter_attr) < 1:
        print(f"[Experiment] filter_attr length is wrong")
        return

    true_celeba_loader_list = []
    for i in range(filter_length):
        true_celeba_loader = get_celeba_loader(
            celeba_image_path = dataset_config.celeba_image_path,
            celeba_attr_path = dataset_config.celeba_attr_path,
            batch_size = dataset_config.batch_size,
            image_size = dataset_config.image_size,
            filter_attr = experiment_config.filter_attr[i],
            filter_value = experiment_config.filter_value[i],
            shuffle = dataset_config.shuffle,
            num_calc_samples = dataset_config.num_calc_samples
        )
        true_celeba_loader_list.append(true_celeba_loader)
        print(f"[Experiment] true_celeba_loader_{i} loaded ({len(true_celeba_loader.dataset)})")

    false_celeba_loader = get_celeba_loader(
        celeba_image_path = dataset_config.celeba_image_path,
        celeba_attr_path = dataset_config.celeba_attr_path,
        batch_size = dataset_config.batch_size,
        image_size = dataset_config.image_size,
        filter_attr = experiment_config.filter_attr,
        filter_value = __not(experiment_config.filter_value),
        shuffle = dataset_config.shuffle,
        num_calc_samples = dataset_config.num_calc_samples
    )
    print(f"[Experiment] false_celeba_loader loaded ({len(false_celeba_loader.dataset)})")

    if dataset_config.custom_dataset_path is None:
        test_dataset_loader = get_celeba_loader(
            celeba_image_path = dataset_config.celeba_image_path,
            celeba_attr_path = dataset_config.celeba_attr_path,
            batch_size = dataset_config.batch_size,
            image_size = dataset_config.image_size,
            filter_attr = experiment_config.filter_attr,
            filter_value = __not(experiment_config.filter_value),
            shuffle = dataset_config.shuffle,
            num_calc_samples = dataset_config.num_samples
        )
        print(f"[Experiment] test_dataset_loader loaded ({len(test_dataset_loader.dataset)})")
    else:
        test_dataset_loader = get_custom_dataset_loader(
            custom_dataset_path = dataset_config.custom_dataset_path,
            batch_size = dataset_config.batch_size,
            image_size = dataset_config.image_size,
            shuffle = dataset_config.shuffle,
            num_calc_samples = dataset_config.num_calc_samples,
        )
        print(f"[Experiment] test_dataset_loader loaded ({len(test_dataset_loader.dataset)})")

    model.eval()

    true_vector_list = []
    for i in range(filter_length):
        true_vector = extract_average_latent(
            model = model,
            dataloader = true_celeba_loader_list[i],
            device = experiment_config.device
        )
        true_vector_list.append(true_vector)
        print(f"\r[Experiment] extract_average_latent(true_celeba_loader_{i}) success")
    false_vector = extract_average_latent(
        model = model,
        dataloader = false_celeba_loader,
        device = experiment_config.device
    )
    print(f"\r[Experiment] extract_average_latent(false_celeba_loader) success")

    latent_vector_list = []
    for i in range(filter_length):
        latent_vector = true_vector_list[i] - false_vector
        latent_vector = latent_vector.to(experiment_config.device)
        latent_vector_list.append(latent_vector)
    print(f"\r[Experiment] calculate latent_vector success")

    test_images = []
    reconstructed_images = []
    transformed_images_list = [[] for _ in range(filter_length)]
    all_transformed_images = []

    with torch.no_grad():
        for test_local_images, _ in tqdm(test_dataset_loader, desc="Transforming test images"):
            test_local_images = test_local_images.to(experiment_config.device)
            encoded_local_vectors = model.encode(test_local_images)[0]
            transformed_local_vectors_list = [
                encoded_local_vectors + experiment_config.scale[i] * latent_vector_list[i].unsqueeze(0)
                for i in range(filter_length)
            ]
            all_transformed_local_vectors = encoded_local_vectors + sum(
                [
                    experiment_config.scale[i] * latent_vector_list[i].unsqueeze(0)
                    for i in range(filter_length)
                ]
            )

            test_images.append(test_local_images.cpu())
            reconstructed_images.append(model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu())
            for i in range(filter_length):
                transformed_images_list[i].append(model.decode(transformed_local_vectors_list[i]).clamp(0.0, 1.0).cpu())
            if filter_length > 1:
                all_transformed_images.append(model.decode(all_transformed_local_vectors).clamp(0.0, 1.0).cpu())

    test_images = torch.cat(test_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    transformed_images_list = [torch.cat(batches, dim=0) for batches in transformed_images_list]
    transformed_images = torch.cat(transformed_images_list, dim=0)
    if all_transformed_images:
        all_transformed_images = torch.cat(all_transformed_images, dim=0)
    else:
        all_transformed_images = torch.tensor([], device="cpu")
    print(f"\r[Experiment] apply_attribute_vector success")

    labels = ["Original", "Reconstructed"]
    for i in range(filter_length):
        labels.append(
            "Transformed\n" +
            f"{experiment_config.filter_attr[i].value}={experiment_config.filter_value[i]} (*{experiment_config.scale[i]})"
        )
    if filter_length > 1:
        labels.append(
            "Transformed\n" +
            f"{
                "\n".join(
                    [
                        f"{experiment_config.filter_attr[i].value}={experiment_config.filter_value[i]} (*{experiment_config.scale[i]})"
                        for i in range(filter_length)
                    ]
                )
            }"
        )

    save_result_image(
        images = torch.cat([test_images, reconstructed_images, transformed_images, all_transformed_images]),
        labels = labels,
        image_size = dataset_config.image_size,
        nrow = len(test_dataset_loader.dataset),
        output_path = experiment_config.output_path
    )

