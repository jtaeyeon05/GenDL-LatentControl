import os

import PIL.Image
import torch
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms
from tqdm import tqdm

import config
from core.custom_dataset import align_and_crop_face
from core.dataset import get_celeba_loader, CelebAFeature
from core.experiment import save_result_image
from core.model import VAE, get_vae_model
from core.util import load_latent_vector


def saved_latent_vector_test(
        model: VAE,
        latent_vector_path: str,
        latent_vector_name: str,
        filter_attr: CelebAFeature,
        filter_value: bool,
        celeba_loader: torch.utils.data.DataLoader,
        image_size: int,
        num_repeats: int,
        output_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    attr = load_latent_vector(
        latent_vector_path = latent_vector_path,
        latent_vector_name = latent_vector_name,
        filter_attr = filter_attr,
        filter_value = filter_value
    ).to(device)
    print(f"\r[Test] Loaded attribute_vector")

    model.eval()

    test_images = []
    reconstructed_images = []
    transformed_images_list = [[] for _ in range(num_repeats)]

    with torch.no_grad():
        for test_local_images, _ in tqdm(celeba_loader, desc="Transforming test images"):
            test_local_images = test_local_images.to(device)
            encoded_local_vectors = model.encode(test_local_images)[0]
            transformed_local_vectors_list = [
                encoded_local_vectors + 0.2 * (i + 1) * attr
                for i in range(num_repeats)
            ]

            test_images.append(test_local_images.cpu())
            reconstructed_images.append(model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu())
            for i in range(num_repeats):
                transformed_images_list[i].append(model.decode(transformed_local_vectors_list[i]).clamp(0.0, 1.0).cpu())

    test_images = torch.cat(test_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)
    transformed_images_list = [torch.cat(batches, dim=0) for batches in transformed_images_list]
    transformed_images = torch.cat(transformed_images_list, dim=0)
    print(f"\r[Test] Applied attribute_vector")

    labels = ["Original", "Reconstructed"]
    for i in range(num_repeats):
        labels.append(
            "Transformed\n" +
            f"{filter_attr.value}\n" +
            f" ={filter_value} (*{0.2 * (i + 1):.2f})"
        )

    save_result_image(
        images = torch.cat([test_images, reconstructed_images, transformed_images]),
        labels = labels,
        image_size = image_size,
        nrow = len(celeba_loader.dataset),
        output_path = output_path
    )


def single_image_test(
        model: VAE,
        latent_vector_path: str,
        latent_vector_name: str,
        filter_attr_list: list[CelebAFeature],
        filter_value_list: list[bool],
        scale_list: list[float],
        image_path: str,
        image_size: int,
        output_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    attr_len = min(len(filter_attr_list), len(filter_value_list), len(scale_list))
    attr_list = [
        load_latent_vector(
            latent_vector_path = latent_vector_path,
            latent_vector_name = latent_vector_name,
            filter_attr = filter_attr_list[i],
            filter_value = filter_value_list[i]
        ).to(device) * scale_list[i]
        for i in range(attr_len)
    ]
    if attr_len != 0:
        attr = torch.sum(torch.stack(attr_list, dim=0), dim=0)
    print(f"\r[Test] Loaded attribute_vector")

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        image = PIL.Image.open(image_path).convert("RGB")
        image = transform(image).to(device).unsqueeze(0)
        if attr_len == 0:
            transformed_image = model.decode(model.encode(image)[0])[0].cpu()
        else:
            transformed_image = model.decode(model.encode(image)[0] + attr)[0].cpu()
        transformed_image = transforms.ToPILImage()(transformed_image)
        transformed_image.save(output_path)
    print(f"\r[Test] Applied attribute_vector")


def reconstruct_test(
        model: VAE,
        celeba_loader: torch.utils.data.DataLoader,
        output_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    test_images = []
    reconstructed_images = []

    with torch.no_grad():
        for test_local_images, _ in tqdm(celeba_loader):
            test_local_images = test_local_images.to(device)
            encoded_local_vectors = model.encode(test_local_images)[0]

            test_images.append(test_local_images.cpu())
            reconstructed_images.append(model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu())

    test_images = torch.cat(test_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)

    grid = make_grid(torch.cat([test_images, reconstructed_images]), nrow=len(celeba_loader.dataset))
    save_image(grid, output_path)
    print(f"[Test] Test Success ({output_path})")


def repeat_reconstruct_test(
        model: VAE,
        celeba_loader: torch.utils.data.DataLoader,
        num_repeats: int,
        output_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
    images: list[list | torch.Tensor] = [[] for _ in range(num_repeats + 1)]

    with torch.no_grad():
        for test_local_images, _ in tqdm(celeba_loader):
            test_local_images = test_local_images.to(device)
            images[0].append(test_local_images.cpu())
            images[0] = torch.cat(images[0], dim=0)
        for i in range(num_repeats):
            test_local_images = images[i].to(device)
            encoded_local_vectors = model.encode(test_local_images)[0]
            images[i + 1] = model.decode(encoded_local_vectors).clamp(0.0, 1.0).cpu()

    grid = make_grid(torch.cat(images), nrow=len(celeba_loader.dataset))
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
        device = config.device
    )

    if not os.path.exists(config.celeba_image_path) or not os.path.exists(config.celeba_attr_path):
        print("[Test] celeba_image_path or celeba_attr_path does not exist")
        return
    celeba_loader = get_celeba_loader(
        celeba_image_path = config.celeba_image_path,
        celeba_attr_path = config.celeba_attr_path,
        batch_size = config.batch_size,
        image_size = config.image_size,
        shuffle = config.shuffle,
        num_calc_samples = config.num_samples
    )

    if not os.path.exists(config.output_path):
        print("[Test] output_path does not exist")
        return
    """
    reconstruct_test(
        model = model,
        celeba_loader = celeba_loader,
        output_path = os.path.join(config.output_path, 'test_tmp.png'),
        device = config.device
    )
    repeat_reconstruct_test(
        model = model,
        celeba_loader = celeba_loader,
        num_repeats = 50,
        output_path = os.path.join(config.output_path, 'test_tmp.png'),
        device = config.device
    )
    saved_latent_vector_test(
        model = model,
        latent_vector_path = config.latent_vector_path,
        latent_vector_name = config.latent_vector_name,
        filter_attr = CelebAFeature.Eyeglasses,
        filter_value = True,
        celeba_loader = celeba_loader,
        image_size = config.image_size,
        num_repeats = 10,
        output_path = os.path.join(config.output_path, 'test_saved_latent_vector.png'),
        device = config.device
    )
    """
    align_and_crop_face(
        os.path.join(config.custom_dataset_path, "raw", "target_new.jpg"),
        os.path.join(config.custom_dataset_path, "target_new.jpg"),
        os.path.join(config.project_dir, "model", "etc" , "shape_predictor_68_face_landmarks.dat")
    )
    single_image_test(
        model = model,
        latent_vector_path = config.latent_vector_path,
        latent_vector_name = config.latent_vector_name,
        filter_attr_list = [CelebAFeature.Eyeglasses],
        filter_value_list = [True],
        scale_list = [2.25],
        image_path = os.path.join(config.custom_dataset_path, "target_new.jpg"),
        image_size = config.image_size,
        output_path = os.path.join(config.output_path, 'test_single.png'),
        device = config.device
    )


if __name__ == '__main__':
    test()

