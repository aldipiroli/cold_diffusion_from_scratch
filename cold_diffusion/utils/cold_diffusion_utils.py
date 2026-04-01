import torch
from torchvision.transforms import v2


def get_random_t(config):
    T = config["NOISE"]["T"]
    t = torch.randint(0, T, (1,))
    return t


def get_gaussian_blur_image(img, t, config):
    kernel_size = config["NOISE"]["kernel_size"]
    sigma = config["NOISE"]["sigma"]
    sigma_increase = config["NOISE"]["sigma_increase"]
    T = config["NOISE"]["T"]
    assert t >= 0 and t <= T, f"t: {t}"
    sigma = sigma * torch.exp(sigma_increase * t)
    sigma = sigma.item()
    blurrer = v2.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    img_blurred = blurrer(img)
    return img_blurred


def sample_from_gmm(mean, std, config):
    img_size = config["DATA"]["img_size"]
    colors = torch.normal(mean, std)
    xt = colors.reshape(img_size[0], 1, 1).expand(-1, img_size[1], img_size[2])  # C, H, W
    xt = xt + torch.randn_like(xt) * config["NOISE"]["additional_noise_std"]
    return xt
