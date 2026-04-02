import os

import torch
import torchvision.transforms as T
from PIL import Image

from cold_diffusion.dataset.dataset_base import DatasetBaseClass


class CelebADataset(DatasetBaseClass):
    def __init__(self, cfg, mode, logger):
        super().__init__(cfg, mode, logger)
        img_size = cfg["DATA"]["img_size"]
        self.transform = T.Compose(
            [
                T.CenterCrop(178),
                T.Resize((img_size[1], img_size[2])),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        img_dir = os.path.join(self.root_dir, "img_align_celeba")
        self.dataset = sorted(
            [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith((".jpg", ".png"))]
        )
        split_idx = int(0.8 * len(self.dataset))
        if self.train:
            self.dataset = self.dataset[:split_idx]
        else:
            self.dataset = self.dataset[split_idx:]
        self.get_channel_stats()

    def compute_channel_stats(self):
        self.logger.info(f"Computing channel mean and std..")
        all_means = []
        for img in self.dataset:
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            img_mean = img.mean(dim=(1, 2))
            all_means.append(img_mean)
        all_means_tensor = torch.stack(all_means)
        self.channel_mean = all_means_tensor.mean(dim=0)
        self.channel_std = all_means_tensor.std(dim=0)
        self.logger.info(f"Channel Stats mean: {self.channel_mean}, std: {self.channel_std}")
        torch.save({"channel_mean": self.channel_mean, "channel_std": self.channel_std}, self.dataset_stats_path)
        self.logger.info(f"Saved channel stats is: {self.dataset_stats_path}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img
