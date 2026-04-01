import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from cold_diffusion.utils.cold_diffusion_utils import get_gaussian_blur_image
from cold_diffusion.utils.misc import load_config


def test_get_gaussian_blur_image():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, img_size[0], img_size[1], img_size[2]
    img = torch.randn(B, C, H, W)
    t = 10
    img_blurred = get_gaussian_blur_image(img, t, config)
    assert img_blurred.shape == img.shape
