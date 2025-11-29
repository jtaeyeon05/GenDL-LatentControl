import os
import tkinter

import numpy as np
import torch
from PIL import Image, ImageTk
from torch.utils.data import DataLoader

import config
from core.custom_dataset import get_custom_single_dataset_loader
from core.dataset import CelebAFeature
from core.model import VAE, get_vae_model
from core.util import load_latent_vector


def _set_image(
        label: tkinter.Label,
        image_tensor: torch.Tensor
    ):
    image = image_tensor.detach().squeeze(0)
    image = (image * 255).byte().permute(1, 2, 0).cpu()
    image = image.numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((256, 256))
    image = ImageTk.PhotoImage(image)
    # noinspection PyTypeChecker
    label.config(image = image)
    label.image = image


class DemoApp:
    def __init__(
            self,
            model: VAE,
            custom_single_dataset_loader: DataLoader,
            latent_vector_path: str,
            latent_vector_name: str,
            filter_attr: CelebAFeature,
            filter_value: bool,
            device: str
        ):
        self.model = model
        self.custom_single_dataset_loader = custom_single_dataset_loader
        self.latent_vector_path = latent_vector_path
        self.latent_vector_name = latent_vector_name
        self.filter_attr = filter_attr
        self.filter_value = filter_value
        self.filter_scale = 1.0
        self.device = device

        with torch.no_grad():
            self.image, self.image_name = next(iter(custom_single_dataset_loader))
            self.image = self.image.to(device)
            self.image_encoded = model.encode(self.image)[0]
            print(f"\r[Demo] loaded image")

        if not os.path.exists(config.latent_vector_path):
            raise ValueError("[Demo] latent_vector_path does not exist")
        self.attr = load_latent_vector(
            latent_vector_path = latent_vector_path,
            latent_vector_name = latent_vector_name,
            filter_attr = filter_attr,
            filter_value = filter_value
        ).to(device)
        print(f"\r[Demo] loaded attr")

        self.window = tkinter.Tk()
        self._init_window()
        self._init_ui()

    def _init_window(self):
        self.window.title("Demo - Controlling Latent Vectors in VAE")
        self.window.geometry("900x600")
        self.window.resizable(False, False)

    def _init_ui(self):
        self.top_frame = tkinter.Frame(
            self.window,
            width = 900,
            height = 600 * 0.75,
            padx = 25,
            pady = 25
        )
        self.top_frame.place(relx = 0.0, rely = 0.0)

        self.bottom_frame = tkinter.Frame(
            self.window,
            width = 900,
            height = 600 * 0.25,
            padx = 25,
            pady = 25
        )
        self.bottom_frame.place(relx = 0.0, rely = 0.75)

        self.original_image_label = tkinter.Label(
            self.top_frame,
            bg = "#ffffff",
        )
        self.original_image_label.place(
            anchor = "w",
            relx = 0.1,
            rely = 0.50,
            width = 256,
            height = 256
        )
        _set_image(self.original_image_label, self.image)

        self.transformed_image_label = tkinter.Label(
            self.top_frame,
            bg = "#ffffff",
        )
        self.transformed_image_label.place(
            anchor = "e",
            relx = 0.9,
            rely = 0.50,
            width = 256,
            height = 256
        )
        _set_image(self.transformed_image_label, self._transformed_image())

        self.desc_label = tkinter.Label(
            self.top_frame,
            text = f"{self.filter_attr.value}={self.filter_value}\n{self.filter_scale}"
        )
        self.desc_label.place(
            anchor = "center",
            relx = 0.50,
            rely = 0.50
        )

        self.min_label = tkinter.Label(
            self.bottom_frame,
            text = "0.0"
        )
        self.min_label.place(
            anchor = "w",
            relx = 0.05,
            rely = 0.50
        )

        self.max_label = tkinter.Label(
            self.bottom_frame,
            text = "5.0"
        )
        self.max_label.place(
            anchor = "e",
            relx = 0.95,
            rely = 0.50
        )

        self.scale = tkinter.Scale(
            self.bottom_frame,
            from_ = 0.0,
            to = 5.0,
            resolution = 0.01,
            tickinterval = 0,
            showvalue = False,
            orient = "horizontal",
            command = lambda value: self._on_slider_change(float(value))
        )
        self.scale.set(1.0)
        self.scale.place(
            anchor = "center",
            relx = 0.50,
            rely = 0.50,
            relwidth = 0.75
        )

    def _transformed_image(self) -> torch.Tensor:
        with torch.no_grad():
            transformed_image_encoded = self.image_encoded + self.filter_scale * self.attr
            return self.model.decode(transformed_image_encoded).clamp(0.0, 1.0)

    def _on_slider_change(self, value: float):
        print(f"[Demo]_on_slider_change ({value})")
        self.filter_scale = value
        self.desc_label.config(text = f"{self.filter_attr.value}={self.filter_value}\n{self.filter_scale}")
        _set_image(self.transformed_image_label, self._transformed_image())

    def mainloop(self):
        self.window.mainloop()


def demo():
    print(f"[Demo] {"=" * 60}")
    print(f"[Demo] Demo - Controlling Latent Vectors in VAE")
    print(f"[Demo] {"=" * 60}")

    print(f"[Demo] device: {config.device}")

    if not os.path.exists(config.model_path):
        raise ValueError("[Demo] model_path does not exist")
    model = get_vae_model(
        model_path = config.model_path,
        model_latent_dim = config.model_latent_dim,
        image_size = config.image_size,
        device = config.device
    )
    print("[Demo] loaded model")

    if not os.path.exists(config.custom_dataset_path):
        raise ValueError("[Demo] custom_dataset_path does not exist")
    custom_single_dataset_loader = get_custom_single_dataset_loader(
        image_path = os.path.join(config.custom_dataset_path, "이재용.png"),
        image_size = config.image_size
    )
    print("[Demo] loaded custom_single_dataset_loader")

    demo_app = DemoApp(
        model = model,
        custom_single_dataset_loader = custom_single_dataset_loader,
        latent_vector_path = config.latent_vector_path,
        latent_vector_name = config.latent_vector_name,
        filter_attr = CelebAFeature.Smiling,
        filter_value = True,
        device = config.device
    )
    demo_app.mainloop()


if __name__ == '__main__':
    demo()

