import os

import config
from core.experiment import run_vae_attribute_experiment
from core.dataset import CelebAFeature, get_celeba_loader_set
from core.model import get_vae_model


def main() -> None:
    if not os.path.exists(config.model_path):
        print("[Main] model_path does not exist")
        return
    model = get_vae_model(
        model_path = config.model_path, 
        model_latent_dim = config.model_latent_dim,
        image_size = config.image_size,
        device = config.device
    )

    if not os.path.exists(config.celebA_image_path) or not os.path.exists(config.celebA_attr_path):
        print("[Main] celebA_image_path or celebA_attr_path does not exist")
        return
    true_celeba_loader, false_celeba_loader, test_celeba_loader = get_celeba_loader_set(
        celebA_image_path = config.celebA_image_path,
        celebA_attr_path = config.celebA_attr_path,
        batch_size = config.batch_size,
        image_size = config.image_size,
        filter_attr = config.filter_attr,
        filter_value = config.filter_value,
        shuffle = config.shuffle,
        num_calc_samples = config.num_calc_samples,
        num_samples = config.num_samples
    )

    run_vae_attribute_experiment(
        model = model,
        true_celeba_loader = true_celeba_loader,
        false_celeba_loader = false_celeba_loader,
        test_celeba_loader = test_celeba_loader,
        output_path = os.path.join(config.output_path, 'test.png'),
        scale = config.scale,
        device = config.device
    )


if __name__ == '__main__':
    main()

