import os

import config
from core.experiment import (
    DatasetConfig,
    ExperimentConfig,
    run_vae_synthesized_attribute_experiment,
    run_vae_multi_attribute_experiment,
    run_vae_only_attribute_experiment
)
from core.model import get_vae_model


def main() -> None:
    print(f"[Main] device: {config.device}")

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
        print("[Main] celeba_image_path or celeba_attr_path does not exist")
        return
    """
    run_vae_synthesized_attribute_experiment(
        model = model,
        dataset_config = DatasetConfig(
            celeba_image_path = config.celeba_image_path,
            celeba_attr_path = config.celeba_attr_path,
            # custom_dataset_path = config.custom_dataset_path,
            batch_size = config.batch_size,
            image_size = config.image_size,
            shuffle = config.shuffle,
            num_calc_samples = config.num_calc_samples,
            num_samples = config.num_samples
        ),
        experiment_config = ExperimentConfig(
            filter_attr = config.filter_attr,
            filter_value = config.filter_value,
            scale = config.scale,
            output_path = os.path.join(config.output_path, 'test_synthesized.png'),
            device = config.device,
        )
    )
    run_vae_multi_attribute_experiment(
        model = model,
        dataset_config = DatasetConfig(
            celeba_image_path = config.celeba_image_path,
            celeba_attr_path = config.celeba_attr_path,
            # custom_dataset_path = config.custom_dataset_path,
            batch_size = config.batch_size,
            image_size = config.image_size,
            shuffle = config.shuffle,
            num_calc_samples = config.num_calc_samples,
            num_samples = config.num_samples
        ),
        experiment_config = ExperimentConfig(
            filter_attr = config.filter_attr,
            filter_value = config.filter_value,
            scale = config.scale,
            output_path = os.path.join(config.output_path, 'test_multi.png'),
            device = config.device,
        )
    )
    """
    run_vae_only_attribute_experiment(
        model = model,
        dataset_config = DatasetConfig(
            celeba_image_path = config.celeba_image_path,
            celeba_attr_path = config.celeba_attr_path,
            # custom_dataset_path = config.custom_dataset_path,
            batch_size = config.batch_size,
            image_size = config.image_size,
            shuffle = config.shuffle,
            num_calc_samples = config.num_calc_samples,
            num_samples = config.num_samples
        ),
        experiment_config = ExperimentConfig(
            filter_attr = config.filter_attr,
            filter_value = config.filter_value,
            scale = config.scale,
            output_path = os.path.join(config.output_path, "test_only.png"),
            device = config.device,
        ),
        latent_vector_path = config.latent_vector_path,
        latent_vector_name = config.latent_vector_name
    )


if __name__ == '__main__':
    main()

