import torch
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.transforms import Normalize, Compose, Resize, ToTensor


class ZeroPadResize:
    def __init__(self, target_size):
        """
        Initializes the transform with the target size.

        Args:
            target_size (tuple): Target size (height, width).
        """
        self.target_size = target_size

    def __call__(self, image):
        """
        Pads the image to the target size with zeros and places it at the center.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: The padded image as a tensor (C, H, W).
        """
        from torchvision.transforms.functional import pil_to_tensor  # Import functional

        # Convert PIL.Image to tensor if needed
        if isinstance(image, Image.Image):
            image = pil_to_tensor(image).float() / 255.0  # Normalize to [0, 1] range

        # Ensure the image is 3D (C, H, W)
        if image.dim() == 2:  # Grayscale image
            image = image.unsqueeze(0)

        c, h, w = image.shape
        target_h, target_w = self.target_size

        if h > target_h or w > target_w:
            raise ValueError("Target size must be larger than the original image size.")

        # Create a black canvas of the target size
        padded_image = torch.zeros((c, target_h, target_w), dtype=image.dtype)

        # Calculate offsets for centering
        y_offset = (target_h - h) // 2
        x_offset = (target_w - w) // 2

        # Place the original image in the center of the canvas
        padded_image[:, y_offset : y_offset + h, x_offset : x_offset + w] = image

        return padded_image


import torch
from torchvision.transforms import functional as F
from PIL import Image


class RepeatToFill:
    def __init__(self, target_size):
        """
        Initializes the transform with the target size.

        Args:
            target_size (tuple): Target size (height, width).
        """
        self.target_size = target_size

    def __call__(self, image):
        """
        Repeats the image to fill the target size.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: The repeated image as a tensor.
        """
        # Convert PIL.Image to tensor if needed
        if isinstance(image, Image.Image):
            image = F.pil_to_tensor(image).float() / 255.0  # Normalize to [0, 1]

        # Ensure the image is 3D (C, H, W)
        if image.dim() == 2:  # Grayscale image
            image = image.unsqueeze(0)

        c, h, w = image.shape
        target_h, target_w = self.target_size

        # Repeat the image to fill the target size
        repeat_h = -(-target_h // h)  # Ceiling division
        repeat_w = -(-target_w // w)

        # Tile the image
        tiled_image = image.repeat(1, repeat_h, repeat_w)

        # Crop to the target size
        tiled_image = tiled_image[:, :target_h, :target_w]

        return tiled_image


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose(
        [
            convert_to_rgb,
            ZeroPadResize((image_size, image_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
