import os

import torch
from models.resnet import ResNet18
from models.resnet import BAddResNet50
from torchvision.models.resnet import resnet50
import gdown
from torch import nn

# https://drive.google.com/file/d/16ZKRuyqZv_Nh97UAfh51JIxtPfsrQy9i/view?usp=sharing
# https://drive.google.com/file/d/1nDvOTHe--VhkmCUM6d8VHb75j0h5_0cR/view?usp=sharing
bcc_urls_dict = {
    "biased_mnist_color": "https://drive.google.com/uc?id=1ZjMO_KsnJZW6y_vn9iWo9F_VgONspr1x",
    "fb_biased_mnist_bgcolor": "https://drive.google.com/uc?id=1nDvOTHe--VhkmCUM6d8VHb75j0h5_0cR",
    "fb_biased_mnist_fgcolor": "https://drive.google.com/uc?id=16ZKRuyqZv_Nh97UAfh51JIxtPfsrQy9i",
    "celeba_gender": "https://drive.google.com/uc?id=1owTN1ouOoZM2Sj507QoSH3VM2wdIItzw",
    "celeba_lipstick": "https://drive.google.com/uc?id=176zg_VYxhJt5TsSjlb87dZcTf9PXfJli",
    "celeba_makeup": "https://drive.google.com/uc?id=1w1h3bKvNV2sCpKyB2wz_aL7i_noqQLX8",
    "utkface_race": "https://drive.google.com/uc?id=1u7KTRXT3uYetIUiCuFgmC-Ifrzw2dpFA",
    "utkface_age": "https://drive.google.com/uc?id=1gnnVKJPY8I0br9MzQCfY1B44VU5bdffn",
    "waterbirds_background": "TODO",
}


def get_model_dict(dataset_name, bias_name):
    model_dir = "./pretrained"
    model_file = f"{dataset_name}_{bias_name}.pth"
    model_path = os.path.join(model_dir, model_file)

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Bias-capturing classifier not found at {model_path}. Downloading...")
        # URL to download the model file if it's missing
        model_url = bcc_urls_dict[f"{dataset_name}_{bias_name}"]

        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    model_dict = torch.load(model_path)
    return model_dict
