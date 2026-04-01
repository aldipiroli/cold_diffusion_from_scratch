import math

from torchvision.transforms import v2


def get_gaussian_blur_image(img, t, config):
    kernel_size = config["NOISE"]["kernel_size"]
    sigma = config["NOISE"]["sigma"]
    sigma_increase = config["NOISE"]["sigma_increase"]
    T = config["NOISE"]["T"]
    assert t >= 0 and t <= T
    sigma = sigma * math.exp(sigma_increase * t)
    blurrer = v2.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    img_blurred = blurrer(img)
    return img_blurred
