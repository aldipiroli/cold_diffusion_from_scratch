import os

import torch
from torch.utils.data import Dataset


class DatasetBaseClass(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger = logger
        self.root_dir = cfg["DATA"]["root_dir"]
        self.train = mode == "train"
        self.dataset_stats_path = os.path.join(self.root_dir, self.cfg["DATA"]["channel_stats"])

    def get_channel_stats(self):
        if os.path.exists(self.dataset_stats_path):
            self.logger.info(f"Loading channel mean and std!")
            stats = torch.load(self.dataset_stats_path)
            self.channel_mean = stats["channel_mean"]
            self.channel_std = stats["channel_std"]
            self.logger.info(f"Channel mean: {self.channel_mean}, std: {self.channel_std}")
        else:
            self.compute_channel_stats()

    def compute_channel_stats(self):
        self.logger.info(f"Computing channel mean and std..")
        all_means = []
        for img, _ in self.dataset:
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
        pass

    def __getitem__(self, idx):
        pass
