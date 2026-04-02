import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

from cold_diffusion.utils.cold_diffusion_utils import (
    get_batch_of_gaussian_blur_images,
    get_gaussian_blur_image,
    get_random_t,
    sample_from_gmm,
)
from cold_diffusion.utils.misc import load_config
from cold_diffusion.utils.plotters import plot


def test_get_random_t():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    T = config["NOISE"]["T"]
    B = 2
    t = get_random_t(B, config)
    assert t.shape == (B, 1)
    assert t.all() >= 0 and t.all() <= T


def test_get_batch_of_gaussian_blur_images():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, img_size[0], img_size[1], img_size[2]
    img = torch.randn(B, C, H, W)
    t = get_random_t(B, config)
    img_blur = get_batch_of_gaussian_blur_images(img, t, config)
    assert img_blur.shape == (B, C, H, W)


def test_get_gaussian_blur_image():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 2, img_size[0], img_size[1], img_size[2]
    img = torch.randn(B, C, H, W)
    t = get_random_t(B, config)
    img_blurred = get_gaussian_blur_image(img, t[0], config)
    assert img_blurred.shape == img.shape


@pytest.mark.parametrize(
    "mean, std, config_name",
    [
        (torch.tensor([0.1]), torch.tensor([0.1]), "mnist_config.yaml"),
        (torch.tensor([0.1, 0.1, 0.1]), torch.tensor([0.1, 0.1, 0.1]), "cifar10_config.yaml"),
    ],
)
def test_sample_from_gmm(mean, std, config_name):
    config = load_config(f"cold_diffusion/config/{config_name}")
    img_size = config["DATA"]["img_size"]
    xt = sample_from_gmm(mean, std, config)
    assert list(xt.shape) == img_size


@pytest.mark.skip()
def test_get_gaussian_blur_img_output():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    img_size = config["DATA"]["img_size"]
    B, C, H, W = 1, img_size[0], img_size[1], img_size[2]
    img = torch.randn(B, C, H, W)
    for t in range(0, 300, 30):
        t = torch.tensor(t)
        img_blurred = get_gaussian_blur_image(img, t, config)
        plot(img_blurred, f"tmp/img_{str(t.item()).zfill(3)}.png", save_figure=True)
