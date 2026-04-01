import torch
from torchvision.transforms import v2


def get_random_t(batch_size, config):
    T = config["NOISE"]["T"]
    t = torch.randint(0, T, (batch_size, 1))
    return t


def get_batch_of_gaussian_blur_images(img, t, config):
    all_imgs = []
    for i in range(img.shape[0]):
        curr_img = get_gaussian_blur_image(img[i], t[i], config)
        all_imgs.append(curr_img)
    all_imgs = torch.stack(all_imgs, 0)
    return all_imgs


def get_gaussian_blur_image(img, t, config):
    if t <= 0:
        return img

    kernel_size = config["NOISE"]["kernel_size"]
    sigma = config["NOISE"]["sigma"]
    sigma_increase = config["NOISE"]["sigma_increase"]
    sigma = sigma + sigma_increase * t
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
