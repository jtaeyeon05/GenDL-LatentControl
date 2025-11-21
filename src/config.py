import os
import torch

from core.dataset import CelebAFeature


working_dir = os.getcwd()
project_dir = working_dir[:-4] if working_dir.endswith("/src") else working_dir

model_path = f"{project_dir}/model/vae_celeba_v1.2_128_learning_rate_0.0005_epoch_100_latent_dim_128.pth"
celebA_path = f"{project_dir}/dataset/celebA"
celebA_image_path = f"{celebA_path}/img_align_celeba/img_align_celeba"
celebA_attr_path = f"{celebA_path}/list_attr_celeba.csv"
output_path = f"{project_dir}/output"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

model_latent_dim = 128
batch_size = 128
image_size = 128

filter_attr = CelebAFeature.Bald
filter_value = False
shuffle = True
num_calc_samples = 500
num_samples = 8
scale = 1.25

