import gzip
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from cold_diffusion.dataset.dataset_base import DatasetBaseClass


class AnimalMNISTDataset(DatasetBaseClass):
    def __init__(self, cfg, mode, logger):
        super().__init__(cfg, mode, logger)
        img_size = cfg["DATA"]["img_size"]
        self.transform = T.Compose(
            [
                T.Resize((img_size[1], img_size[2])),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        data_path = os.path.join(self.root_dir, "animal_data_version_3.gz")
        images = self.load_gzip(data_path)
        self.dataset = list(images)
        random.seed(0)
        random.shuffle(self.dataset)
        split_idx = int(0.8 * len(self.dataset))
        if self.train:
            self.dataset = self.dataset[:split_idx]
        else:
            self.dataset = self.dataset[split_idx:]
        self.get_channel_stats()

    def load_gzip(self, data_path):
        with gzip.GzipFile(data_path, "rb") as f:
            data = pickle.load(f)
        return data

    def compute_channel_stats(self):
        self.logger.info(f"Computing channel mean and std..")
        all_means = []
        for img in self.dataset:
            img = Image.fromarray(img.astype(np.uint8), mode="L")
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
        img_array = self.dataset[idx]
        img = Image.fromarray(img_array.astype(np.uint8), mode="L")
        img = self.transform(img)
        return img
